"""OCR processing (UC-6) through fakes at every port boundary."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from docgrep.domain.errors import (
    IndexLockHeld,
    OcrFailed,
    OcrToolsMissing,
    ValidationError,
)
from docgrep.domain.models import FileType, ManifestRunRecord
from docgrep.domain.run_state import latest_build_run, run_state
from docgrep.usecases.cancel_index import CancelIndex
from docgrep.usecases.discover_scans import DiscoverScans
from docgrep.usecases.process_scans import (
    OcrFileResult,
    OcrPlan,
    OcrRequest,
    OcrSummary,
    ProcessScans,
)
from fakes import InMemoryManifest, InMemoryOcrRegistry
from ocr_support import BROKEN, NATIVE_PAGE, FakeConverter, FakeCorpus, FakeEngine

SCAN: list[str | None] = [None, None, "7", None]


class Lock:
    def __init__(self, refuse: Exception | None = None, held: bool = False) -> None:
        self.refuse = refuse
        self.held = held
        self.events: list[str] = []

    def is_held(self) -> bool:
        return self.held

    @contextmanager
    def hold(self, operation: str = "build") -> Iterator[None]:
        if self.refuse is not None:
            raise self.refuse
        self.events.append(f"acquired:{operation}")
        try:
            yield
        finally:
            self.events.append("released")


class Setup:
    def __init__(self, **options: Any) -> None:
        self.corpus = FakeCorpus()
        self.engine = FakeEngine(self.corpus)
        self.converter = FakeConverter(self.corpus)
        self.registry = InMemoryOcrRegistry(
            exists=self.corpus.exists, content_hash=self.corpus.content_hash
        )
        self.manifest = InMemoryManifest()
        self.lock = Lock()
        self.options = {"languages": "rus+eng+ukr", "poll_seconds": 0, **options}

    @property
    def use_case(self) -> ProcessScans:
        return ProcessScans(
            lock=self.lock,
            manifest=self.manifest,
            scanner=self.corpus,
            discovery=DiscoverScans(
                scanner=self.corpus,
                probe=self.corpus,
                density_lower=100,
                density_upper=500,
            ),
            registry=self.registry,
            engine=self.engine,
            converter=self.converter,
            files=self.corpus,
            **self.options,
        )


def _setup(**options: Any) -> Setup:
    setup = Setup(**options)
    setup.corpus.add("books/scan.pdf", SCAN)
    setup.corpus.add("books/book.djvu", [None, None])
    setup.corpus.add("books/native.pdf", [NATIVE_PAGE] * 3)
    setup.corpus.add("books/broken.pdf", [BROKEN + "x" * 300] * 3)
    setup.corpus.add("books/notes.md", None)
    return setup


def _statuses(summary: OcrSummary) -> dict[str, str]:
    return {f["path"]: f["status"] for f in summary.as_dict()["files"]}


def _run(setup: Setup, **request: Any) -> OcrSummary:
    return setup.use_case.run(OcrRequest(**request))


def test_plan_classifies_orders_and_counts_without_lock_or_ocr() -> None:
    setup = _setup()
    setup.corpus.add("books/locked.pdf", SCAN, encrypted=True)
    setup.corpus.add("books/corrupt.pdf", None)

    plan = setup.use_case.plan(
        OcrRequest(
            paths=(
                "books/scan.pdf",
                "books/native.pdf",
                "books/corrupt.pdf",
                "books/broken.pdf",
                "books/locked.pdf",
                "books/book.djvu",
            )
        )
    )

    assert [(t.path, t.status, t.mode) for t in plan.targets] == [
        ("books/book.djvu", "to_process", "skip-text"),
        ("books/broken.pdf", "to_process", "force-ocr"),
        ("books/corrupt.pdf", "failed:unreadable", None),
        ("books/locked.pdf", "failed:encrypted", None),
        ("books/native.pdf", "skipped:does_not_need_ocr", None),
        ("books/scan.pdf", "to_process", "skip-text"),
    ]
    report = plan.as_dict()
    assert report["to_process"] == 3
    assert report["to_skip"] == {"does_not_need_ocr": 1}
    assert report["to_fail"] == {"encrypted": 1, "unreadable": 1}
    assert report["by_file_type"]["djvu"] == {
        "to_process": 1,
        "to_skip": {},
        "to_fail": {},
    }
    assert report["by_file_type"]["pdf"]["to_process"] == 2
    work = [t for t in plan.targets if t.status == "to_process"]
    assert report["input_bytes"] == sum(t.size for t in work)
    assert report["pages"] == 2 + 3 + 4
    assert setup.lock.events == []
    assert setup.manifest.latest_run() is None
    assert setup.engine.calls == [] and setup.converter.calls == []


def test_a_run_writes_registers_and_delegates_each_output() -> None:
    setup = _setup(jobs=2)
    original = setup.corpus.files["books/scan.pdf"].data
    seen: list[OcrPlan] = []
    lines: list[OcrFileResult] = []

    summary = setup.use_case.run(
        OcrRequest(paths=("books/scan.pdf", "books/broken.pdf")),
        on_plan=seen.append,
        on_file=lines.append,
    )

    assert summary.status == "completed" and summary.failed == 0
    assert _statuses(summary) == {
        "books/broken.pdf": "processed",
        "books/scan.pdf": "processed",
    }
    assert len(seen) == 1 and [(r.index, r.total) for r in lines] == [(1, 2), (2, 2)]
    assert lines[-1].eta_seconds == 0
    assert setup.corpus.files["books/scan.pdf"].data == original
    assert setup.corpus.files["books/scan.ocr.pdf"].pages == [NATIVE_PAGE] * 4
    assert [
        (c["source"], c["mode"], c["jobs"], c["languages"]) for c in setup.engine.calls
    ] == [
        ("books/broken.pdf", "force-ocr", 2, "rus+eng+ukr"),
        ("books/scan.pdf", "skip-text", 2, "rus+eng+ukr"),
    ]
    record = setup.registry.records["books/scan.pdf"]
    assert (record.output_path, record.input_format, record.mode) == (
        "books/scan.ocr.pdf",
        "pdf",
        "skip-text",
    )
    assert record.input_hash == setup.corpus.content_hash("books/scan.pdf")
    assert record.output_hash == setup.corpus.content_hash("books/scan.ocr.pdf")
    assert (record.engine, record.engine_version) == ("fake-ocr", "1.0")
    assert setup.registry.live_delegation("books/scan.pdf") is not None
    # No temporary is left behind.
    assert not [path for path in setup.corpus.files if path.endswith(".part")]
    run = setup.manifest.latest_run()
    assert run is not None and (run.kind, run.status) == ("ocr", "completed")
    assert dict(run.totals)["processed"] == 2
    assert [r.status for r in setup.manifest.run_sources(run.run_id)] == [
        "processed",
        "processed",
    ]
    assert setup.lock.events == ["acquired:ocr", "released"]


def test_a_djvu_file_is_converted_first_and_its_output_is_a_pdf() -> None:
    setup = _setup()

    summary = _run(setup, paths=("books/book.djvu",))

    assert _statuses(summary) == {"books/book.djvu": "processed"}
    ((source, converted),) = setup.converter.calls
    assert source == "books/book.djvu"
    assert setup.engine.calls[0]["source"] == converted
    assert setup.engine.calls[0]["mode"] == "skip-text"
    assert converted not in setup.corpus.files
    assert setup.corpus.files["books/book.ocr.pdf"].pages == [NATIVE_PAGE] * 2
    assert setup.registry.records["books/book.djvu"].input_format == "djvu"
    assert "books/book.djvu" in setup.corpus.files


@pytest.mark.parametrize(
    ("failure", "status"),
    [
        (
            OcrFailed("bad", details={"reason": "djvu_conversion"}),
            "failed:djvu_conversion",
        ),
        (
            OcrFailed("missing", details={"reason": "djvu_tools_missing"}),
            "failed:djvu_tools_missing",
        ),
    ],
)
def test_a_failed_djvu_conversion_fails_only_that_file(
    failure: Exception, status: str
) -> None:
    setup = _setup()
    setup.converter.failures["books/book.djvu"] = failure

    summary = _run(setup, paths=("books/book.djvu", "books/scan.pdf"))

    assert _statuses(summary) == {
        "books/book.djvu": status,
        "books/scan.pdf": "processed",
    }
    assert summary.status == "completed" and summary.failed == 1
    assert not [path for path in setup.corpus.files if path.endswith(".part")]
    assert "books/book.ocr.pdf" not in setup.corpus.files


@pytest.mark.parametrize(
    ("reason", "status"),
    [
        ("ocr_engine_error", "failed:ocr_engine_error"),
        ("timeout", "failed:timeout"),
        ("encrypted", "failed:encrypted"),
    ],
)
def test_an_engine_failure_cleans_up_and_the_batch_goes_on(
    reason: str, status: str
) -> None:
    setup = _setup()
    setup.engine.failures["books/broken.pdf"] = OcrFailed(
        "engine said no", details={"reason": reason}
    )

    summary = _run(setup, paths=("books/broken.pdf", "books/scan.pdf"))

    assert _statuses(summary) == {
        "books/broken.pdf": status,
        "books/scan.pdf": "processed",
    }
    failed = next(f for f in summary.as_dict()["files"] if f["status"] == status)
    assert failed["message"] == "engine said no" and failed["output_path"] is None
    assert "books/broken.ocr.pdf" not in setup.corpus.files
    assert not [path for path in setup.corpus.files if path.endswith(".part")]
    assert "books/broken.pdf" not in setup.registry.records


def test_a_time_limit_is_passed_to_the_engine_as_a_deadline() -> None:
    clock = iter(float(n) for n in range(1000))
    setup = _setup(timeout_seconds=30, clock=lambda: next(clock))

    _run(setup, paths=("books/scan.pdf",))
    unlimited = _setup()
    _run(unlimited, paths=("books/scan.pdf",))

    deadline = setup.engine.calls[0]["deadline"]
    assert isinstance(deadline, float) and 30 < deadline < 60
    assert unlimited.engine.calls[0]["deadline"] is None


def test_an_output_with_another_page_count_is_rejected() -> None:
    setup = _setup()
    setup.engine.pages_delta = -1

    summary = _run(setup, paths=("books/scan.pdf",))

    assert _statuses(summary) == {"books/scan.pdf": "failed:output_invalid"}
    assert "books/scan.ocr.pdf" not in setup.corpus.files
    assert setup.registry.records == {}


def test_an_existing_unregistered_output_is_never_overwritten() -> None:
    setup = _setup()
    setup.corpus.add("books/scan.ocr.pdf", [NATIVE_PAGE], data=b"mine")

    summary = _run(setup, paths=("books/scan.pdf",))

    assert _statuses(summary) == {"books/scan.pdf": "failed:output_collision"}
    assert setup.corpus.files["books/scan.ocr.pdf"].data == b"mine"
    assert setup.engine.calls == []


def test_an_output_appearing_during_ocr_is_not_overwritten() -> None:
    setup = _setup()
    setup.engine.on_run = lambda source: setup.corpus.add(
        "books/scan.ocr.pdf", [NATIVE_PAGE], data=b"raced"
    )

    summary = _run(setup, paths=("books/scan.pdf",))

    assert _statuses(summary) == {"books/scan.pdf": "failed:output_collision"}
    assert setup.corpus.files["books/scan.ocr.pdf"].data == b"raced"
    assert not [path for path in setup.corpus.files if path.endswith(".part")]


def test_a_failed_registry_write_keeps_the_output_and_reprocesses_later() -> None:
    setup = _setup()
    setup.registry.fail_writes = True

    summary = _run(setup, paths=("books/scan.pdf",))

    assert _statuses(summary) == {"books/scan.pdf": "failed:registry_write"}
    assert "books/scan.ocr.pdf" in setup.corpus.files
    assert summary.failed == 1

    setup.registry.fail_writes = False
    setup.corpus.remove("books/scan.ocr.pdf")
    again = _run(setup, paths=("books/scan.pdf",))
    assert _statuses(again) == {"books/scan.pdf": "processed"}


def test_too_little_free_space_aborts_the_rest_of_the_batch() -> None:
    setup = _setup()
    setup.corpus.free = 1

    summary = _run(setup, paths=("books/scan.pdf", "books/broken.pdf"))

    assert summary.status == "aborted" and summary.abort_reason == "disk_full"
    assert _statuses(summary) == {
        "books/broken.pdf": "failed:disk_full",
        "books/scan.pdf": "not_started",
    }
    assert setup.engine.calls == []
    run = setup.manifest.latest_run()
    assert run is not None and run.status == "aborted"


def test_a_full_disk_reported_by_the_engine_aborts_but_keeps_earlier_files() -> None:
    setup = _setup()
    setup.corpus.add("books/later.pdf", SCAN)
    setup.engine.failures["books/later.pdf"] = OcrFailed(
        "full", details={"reason": "disk_full"}
    )

    summary = _run(
        setup, paths=("books/scan.pdf", "books/later.pdf", "books/broken.pdf")
    )

    assert _statuses(summary) == {
        "books/broken.pdf": "processed",
        "books/later.pdf": "failed:disk_full",
        "books/scan.pdf": "not_started",
    }
    assert summary.status == "aborted"
    assert "books/broken.pdf" in setup.registry.records


def test_a_second_run_skips_what_is_done_and_force_redoes_it() -> None:
    setup = _setup()
    _run(setup, paths=("books/scan.pdf",))
    first = setup.corpus.files["books/scan.ocr.pdf"].data

    again = _run(setup, paths=("books/scan.pdf",))
    assert _statuses(again) == {"books/scan.pdf": "skipped:already_processed"}
    assert len(setup.engine.calls) == 1

    setup.engine.failures.clear()
    forced = _run(setup, paths=("books/scan.pdf",), force=True)
    assert _statuses(forced) == {"books/scan.pdf": "processed"}
    assert len(setup.engine.calls) == 2
    assert setup.corpus.files["books/scan.ocr.pdf"].data == first
    assert list(setup.registry.records) == ["books/scan.pdf"]


def test_a_changed_original_replaces_its_own_earlier_output() -> None:
    setup = _setup()
    _run(setup, paths=("books/scan.pdf",))
    setup.corpus.add("books/scan.pdf", [None] * 5, data=b"rescanned")

    summary = _run(setup, paths=("books/scan.pdf",))

    assert _statuses(summary) == {"books/scan.pdf": "processed"}
    assert setup.corpus.files["books/scan.ocr.pdf"].pages == [NATIVE_PAGE] * 5


def test_an_ocr_output_is_never_a_target_again() -> None:
    setup = _setup()
    _run(setup, paths=("books/scan.pdf",))

    plan = setup.use_case.plan(
        OcrRequest(paths=("books/scan.ocr.pdf",), force=True, override_native=True)
    )

    assert [t.status for t in plan.targets] == ["skipped:already_processed"]


def test_override_native_rebuilds_a_native_text_layer() -> None:
    setup = _setup()

    plain = setup.use_case.plan(OcrRequest(paths=("books/native.pdf",)))
    summary = _run(setup, paths=("books/native.pdf",), override_native=True)

    assert [t.status for t in plain.targets] == ["skipped:does_not_need_ocr"]
    assert _statuses(summary) == {"books/native.pdf": "processed"}
    assert setup.engine.calls[0]["mode"] == "force-ocr"
    assert setup.registry.records["books/native.pdf"].mode == "force-ocr"


def test_all_discovered_selects_scans_and_optionally_uncertain_files() -> None:
    setup = _setup()
    setup.corpus.add("books/mixed.pdf", ["x" * 300] * 4)

    scans = setup.use_case.plan(OcrRequest(all_discovered=True))
    wider = setup.use_case.plan(OcrRequest(all_discovered=True, include_uncertain=True))

    assert [t.path for t in scans.targets] == [
        "books/book.djvu",
        "books/broken.pdf",
        "books/scan.pdf",
    ]
    assert [(t.path, t.mode) for t in wider.targets if t.path == "books/mixed.pdf"] == [
        ("books/mixed.pdf", "skip-text")
    ]


def test_a_glob_filters_other_types_and_reports_how_many() -> None:
    setup = _setup()

    plan = setup.use_case.plan(OcrRequest(path_glob="books/*"))

    assert plan.filtered == 1
    assert [t.path for t in plan.targets] == [
        "books/book.djvu",
        "books/broken.pdf",
        "books/native.pdf",
        "books/scan.pdf",
    ]


def test_targets_are_ordered_by_type_then_path_whatever_the_request_order() -> None:
    setup = _setup()
    paths = ("books/scan.pdf", "books/book.djvu", "books/broken.pdf")

    first = setup.use_case.plan(OcrRequest(paths=paths))
    second = setup.use_case.plan(OcrRequest(paths=tuple(reversed(paths))))

    assert (
        [t.path for t in first.targets]
        == [t.path for t in second.targets]
        == [
            "books/book.djvu",
            "books/broken.pdf",
            "books/scan.pdf",
        ]
    )


@pytest.mark.parametrize(
    ("request_options", "reason"),
    [
        ({}, "targets"),
        ({"paths": ("books/scan.pdf",), "all_discovered": True}, "targets"),
        ({"path_glob": "*.pdf", "all_discovered": True}, "targets"),
        ({"paths": ("books/scan.pdf", "../outside.pdf")}, "outside_corpus"),
        ({"paths": ("/etc/passwd",)}, "outside_corpus"),
        ({"paths": ("books/missing.pdf",)}, "not_found"),
        ({"paths": ("books/scan.pdf", "books/notes.md")}, "extension"),
        ({"all_discovered": True, "override_native": True}, "override_native"),
        (
            {"paths": ("books/scan.pdf",), "include_uncertain": True},
            "include_uncertain",
        ),
        ({"path_glob": " "}, "path_glob"),
        ({"paths": ("books/scan.pdf",), "languages": "rus+"}, "languages"),
        ({"paths": ("books/scan.pdf",), "languages": "Rus"}, "languages"),
    ],
)
def test_an_invalid_request_is_rejected_as_a_whole(
    request_options: dict[str, Any], reason: str
) -> None:
    setup = _setup()

    with pytest.raises(ValidationError) as raised:
        _run(setup, **request_options)
    with pytest.raises(ValidationError):
        setup.use_case.plan(OcrRequest(**request_options))

    assert raised.value.details["reason"] == reason
    if reason == "extension":
        assert raised.value.details["extensions"] == ".md"
    assert setup.lock.events == [] and setup.engine.calls == []


def test_missing_tools_or_languages_are_refused_before_any_work() -> None:
    setup = _setup()
    setup.engine.present = False
    with pytest.raises(OcrToolsMissing):
        _run(setup, paths=("books/scan.pdf",))
    with pytest.raises(OcrToolsMissing):
        setup.use_case.plan(OcrRequest(paths=("books/scan.pdf",)))

    setup.engine.present = True
    setup.engine.installed = ("eng", "osd")
    with pytest.raises(OcrToolsMissing) as raised:
        _run(setup, paths=("books/scan.pdf",))
    assert raised.value.details["missing_languages"] == "rus, ukr"
    assert "rus, ukr" in str(raised.value)

    assert setup.lock.events == [] and setup.engine.calls == []
    assert setup.manifest.latest_run() is None
    assert _statuses(_run(setup, paths=("books/scan.pdf",), languages="eng")) == {
        "books/scan.pdf": "processed"
    }


def test_a_held_lock_refuses_the_run() -> None:
    setup = _setup()
    setup.lock.refuse = IndexLockHeld("busy")

    with pytest.raises(IndexLockHeld):
        _run(setup, paths=("books/scan.pdf",))

    assert setup.engine.calls == [] and setup.manifest.latest_run() is None


def _cancelling(setup: Setup, use_case: ProcessScans) -> None:
    setup.engine.on_run = lambda source: use_case.request_cancel()


def test_terminate_mode_stops_the_running_file_and_ends_the_batch() -> None:
    setup = _setup()
    setup.engine.block = True
    use_case = setup.use_case
    _cancelling(setup, use_case)

    summary = use_case.run(OcrRequest(paths=("books/scan.pdf", "books/broken.pdf")))

    assert summary.status == "cancelled"
    assert _statuses(summary) == {
        "books/broken.pdf": "cancelled",
        "books/scan.pdf": "not_started",
    }
    assert not [path for path in setup.corpus.files if path.endswith(".part")]
    assert "books/broken.ocr.pdf" not in setup.corpus.files
    run = setup.manifest.latest_run()
    assert run is not None and run.status == "cancelled"
    assert run.cancel_requested_at is not None
    assert dict(run.totals)["cancelled"] == 1


def test_a_cancel_from_another_process_stops_the_running_engine() -> None:
    setup = _setup()
    setup.engine.block = True

    def persisted(source: str) -> None:
        latest = setup.manifest.latest_run()
        assert latest is not None
        setup.manifest.request_cancel(latest.run_id)

    setup.engine.on_run = persisted

    summary = _run(setup, paths=("books/scan.pdf",))

    assert _statuses(summary) == {"books/scan.pdf": "cancelled"}
    assert summary.status == "cancelled"


def test_finish_mode_completes_the_current_file_first() -> None:
    setup = _setup(cancel_mode="finish")
    use_case = setup.use_case
    _cancelling(setup, use_case)

    summary = use_case.run(OcrRequest(paths=("books/scan.pdf", "books/broken.pdf")))

    assert summary.status == "cancelled"
    assert _statuses(summary) == {
        "books/broken.pdf": "processed",
        "books/scan.pdf": "not_started",
    }
    assert "books/broken.pdf" in setup.registry.records

    resumed = _run(setup, paths=("books/scan.pdf", "books/broken.pdf"))
    assert _statuses(resumed) == {
        "books/broken.pdf": "skipped:already_processed",
        "books/scan.pdf": "processed",
    }


def test_a_cancel_between_files_keeps_the_completed_ones() -> None:
    setup = _setup()
    use_case = setup.use_case

    def cancel_after_first(result: OcrFileResult) -> None:
        CancelIndex(lock=Lock(held=True), manifest=setup.manifest).run()

    summary = use_case.run(
        OcrRequest(paths=("books/scan.pdf", "books/broken.pdf")),
        on_file=cancel_after_first,
    )

    assert _statuses(summary) == {
        "books/broken.pdf": "processed",
        "books/scan.pdf": "not_started",
    }
    assert summary.status == "cancelled"


def test_a_dead_ocr_run_is_closed_as_aborted() -> None:
    setup = _setup()
    dead = setup.manifest.start_run({}, kind="ocr")

    _run(setup, paths=("books/scan.pdf",))

    runs = setup.manifest._runs
    assert [(r.run_id, r.kind, r.status) for r in runs] == [
        (dead, "ocr", "aborted"),
        ("2", "ocr", "completed"),
    ]


def test_an_ocr_run_neither_hides_nor_fakes_a_resumable_build() -> None:
    manifest = InMemoryManifest()
    build = manifest.start_run({}, kind="build")
    manifest.finish_run(build, "completed", {})
    ocr = manifest.start_run({}, kind="ocr")
    manifest.finish_run(ocr, "cancelled", {})

    latest = latest_build_run(manifest)
    assert latest is not None and latest.run_id == build
    assert run_state(latest, lock_held=False) == "completed"

    unfinished = InMemoryManifest()
    unfinished.start_run({}, kind="build")
    unfinished.start_run({}, kind="ocr")
    found = latest_build_run(unfinished)
    assert isinstance(found, ManifestRunRecord) and found.kind == "build"
    assert run_state(found, lock_held=False) == "interrupted"


def test_an_empty_selection_completes_with_nothing_done() -> None:
    setup = _setup()

    summary = _run(setup, path_glob="nothing/**")

    assert summary.status == "completed"
    assert summary.as_dict()["totals"] == {
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "cancelled": 0,
        "not_started": 0,
    }


def test_the_summary_json_has_stable_keys() -> None:
    setup = _setup()

    report = _run(setup, paths=("books/scan.pdf", "books/native.pdf")).as_dict()

    assert set(report) == {
        "schema_version",
        "run_id",
        "status",
        "abort_reason",
        "plan",
        "totals",
        "skipped_by_reason",
        "failed_by_reason",
        "elapsed_seconds",
        "files",
        "hint",
    }
    assert set(report["files"][0]) == {
        "path",
        "file_type",
        "size_bytes",
        "page_count",
        "classification",
        "reason",
        "mode",
        "status",
        "output_path",
        "message",
        "elapsed_seconds",
    }
    assert report["skipped_by_reason"] == {"does_not_need_ocr": 1}
    assert "docgrep update" in report["hint"]
    assert report["files"][1]["file_type"] == FileType.PDF.value
