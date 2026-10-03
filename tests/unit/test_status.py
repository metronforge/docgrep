"""The status use case on fakes: states, degradation, the active view and the ETA."""

import threading
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from docgrep.adapters.chunking import TokenChunker, WhitespaceTokenizer
from docgrep.adapters.filelock import FileIndexLock
from docgrep.adapters.models.deterministic import DeterministicEmbedder
from docgrep.domain.errors import ManifestUnavailable, StoreUnavailable
from docgrep.domain.models import (
    FileType,
    ManifestRunRecord,
    ManifestSourceRecord,
    ParsedDocument,
    ParsedSection,
    RunStatus,
    Source,
)
from docgrep.domain.protocols import ManifestReader, StatusStore
from docgrep.domain.status import INDEX_STATES
from docgrep.usecases.build_index import BuildIndex
from docgrep.usecases.inspect_status import InspectStatus
from fakes import InMemoryManifest, InMemoryPassageStore

NOW = datetime(2026, 10, 3, 12, 0, 0, tzinfo=UTC)
META: dict[str, str | int | float] = {
    "embedding_model": "m",
    "embedding_dimension": 64,
    "embedding_dtype": "float32",
    "docgrep_version": "1.4.0",
    "completed_at": "2026-10-03T10:00:00+00:00",
    "chunk_size": 10,
    "chunk_overlap": 2,
}


class StubLock:
    def __init__(self, held: bool | Exception = False) -> None:
        self.held = held

    def is_held(self) -> bool:
        if isinstance(self.held, Exception):
            raise self.held
        return self.held


class StubManifest:
    """A manifest reader with crafted runs and journals."""

    def __init__(
        self,
        run: ManifestRunRecord | None,
        journal: Sequence[ManifestSourceRecord] = (),
    ) -> None:
        self.run = run
        self.journal = tuple(journal)

    def latest_run(self) -> ManifestRunRecord | None:
        return self.run

    def run_sources(self, run_id: str) -> tuple[ManifestSourceRecord, ...]:
        return self.journal

    def get_source(self, path: str) -> ManifestSourceRecord | None:
        return None

    def cancel_requested_at(self, run_id: str) -> str | None:
        return None


class StubStore:
    def __init__(
        self,
        metadata: dict[str, str | int | float] | None = None,
        passages: int = 0,
        fail: Exception | None = None,
    ) -> None:
        self.metadata = metadata
        self.passages = passages
        self.fail = fail
        self.closed = False

    def read_completion_metadata(self) -> dict[str, str | int | float] | None:
        if self.fail:
            raise self.fail
        return self.metadata

    def count_passages(self) -> int:
        return self.passages

    def retrieve(self, *args: Any) -> tuple[()]:
        return ()

    def staging_info(self, name: str) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def _run(status: str = "completed", **extra: Any) -> ManifestRunRecord:
    return ManifestRunRecord(
        "3",
        cast(RunStatus, status),
        (),
        started_at=(NOW - timedelta(seconds=80)).isoformat(),
        **extra,
    )


def _status(
    *,
    lock: StubLock | None = None,
    manifest: Callable[[], ManifestReader | None] | None = None,
    store: Callable[[], StatusStore | None] | None = None,
    runtime: dict[str, str | int | float] | None = None,
) -> InspectStatus:
    return InspectStatus(
        lock=lock or StubLock(),
        open_manifest=manifest or (lambda: None),
        open_store=store or (lambda: None),
        store_mode="local",
        runtime_config=runtime or {},
        now=lambda: NOW,
    )


def _ready_store() -> StubStore:
    return StubStore(META, 42)


@pytest.mark.parametrize(
    ("run", "held", "state", "run_state"),
    [
        (None, False, "ready", None),
        ("completed", False, "ready", "completed"),
        ("aborted", False, "aborted_pending_resume", "aborted"),
        ("cancelled", False, "cancelled_pending_resume", "cancelled"),
        ("running", False, "interrupted_pending_resume", "interrupted"),
        ("swapped", False, "swap_pending_commit", "swapped"),
        ("running", True, "ready", "running"),
    ],
)
def test_each_index_state(
    run: str | None, held: bool, state: str, run_state: str | None
) -> None:
    record = None if run is None else _run(run)
    report = _status(
        lock=StubLock(held),
        manifest=lambda: StubManifest(record),
        store=_ready_store,
    ).run()

    assert report.index_state == state and state in INDEX_STATES
    assert (report.latest_run.state if report.latest_run else None) == run_state


def test_nothing_built_yet() -> None:
    assert _status().run().index_state == "not_built"
    empty = _status(
        manifest=lambda: StubManifest(None), store=lambda: StubStore()
    ).run()
    assert empty.index_state == "not_built" and empty.collection is None
    first_build = _status(
        lock=StubLock(True),
        manifest=lambda: StubManifest(_run("running")),
        store=lambda: StubStore(),
    ).run()
    assert first_build.index_state == "not_built"
    assert first_build.active_operation is not None


def test_a_manifest_that_cannot_be_read_degrades_the_report() -> None:
    def broken() -> ManifestReader:
        raise ManifestUnavailable("Manifest database operation failed")

    report = _status(manifest=broken, store=_ready_store).run()

    assert report.index_state == "manifest_unreadable"
    assert report.collection is not None and report.collection.passage_count == 42
    assert [w.code for w in report.warnings] == ["manifest_unreadable"]


def test_an_unreachable_store_is_reported_not_raised() -> None:
    def down() -> StatusStore:
        raise StoreUnavailable(
            "Cannot reach the Qdrant server",
            details={"reason": "connection refused " + "x" * 400},
        )

    report = _status(
        manifest=lambda: StubManifest(_run()), store=down, runtime={}
    ).run()

    assert report.store.status == "unreachable"
    assert report.store.reason is not None and len(report.store.reason) <= 200
    assert report.collection is None and report.index_state == "ready"
    assert "store_unreachable" in [w.code for w in report.warnings]


def test_a_store_that_fails_on_read_is_unreachable_and_still_closed() -> None:
    store = StubStore(fail=OSError("socket timeout"))

    report = _status(store=lambda: store).run()

    assert report.store.status == "unreachable" and store.closed


def test_a_busy_local_store_is_busy_not_a_failure() -> None:
    def busy() -> StatusStore:
        raise StoreUnavailable(
            "Cannot open Qdrant storage",
            details={"reason": "already accessed", "busy": True},
        )

    report = _status(manifest=lambda: StubManifest(_run()), store=busy).run()

    assert report.store.status == "busy"
    assert [w.code for w in report.warnings] == ["store_busy"]


def test_a_missing_local_store_is_absent() -> None:
    report = _status(store=lambda: None).run()

    assert report.store.status == "absent" and report.index_state == "not_built"


def test_a_lock_that_cannot_be_probed_does_not_raise() -> None:
    report = _status(
        lock=StubLock(StoreUnavailable("Cannot access the index lock")),
        manifest=lambda: StubManifest(_run("running")),
    ).run()

    assert report.index_state == "interrupted_pending_resume"
    assert "lock_unreadable" in [w.code for w in report.warnings]


def test_the_collection_facts_and_drift_for_vector_keys_only() -> None:
    report = _status(
        manifest=lambda: StubManifest(_run()),
        store=_ready_store,
        runtime={
            "embedding_model": "other",
            "embedding_dimension": 64,
            "chunk_size": 10,
            "chunk_overlap": 2,
            "device": "cpu",
        },
    ).run()

    collection = report.collection
    assert collection is not None
    assert (collection.passage_count, collection.embedding_model) == (42, "m")
    assert (collection.embedding_dimension, collection.embedding_dtype) == (
        64,
        "float32",
    )
    assert collection.docgrep_version == "1.4.0"
    assert collection.completed_at == META["completed_at"]
    assert [(d.field, d.indexed, d.runtime) for d in report.drift] == [
        ("embedding_model", "m", "other")
    ]
    assert "config_drift" in [w.code for w in report.warnings]


def test_a_completed_run_without_an_index_warns() -> None:
    report = _status(
        manifest=lambda: StubManifest(_run()), store=lambda: StubStore()
    ).run()

    assert "index_incomplete" in [w.code for w in report.warnings]


def _journal(count: int, passages: int = 3) -> list[ManifestSourceRecord]:
    return [
        ManifestSourceRecord(
            Source(f"d{i}", f"d{i}", FileType.TXT), "indexed", passages=passages
        )
        for i in range(count)
    ]


def test_the_active_operation_and_its_eta_arithmetic() -> None:
    run = _run("running", total_sources=10, cancel_requested_at="2026-10-03T11:59:00")
    report = _status(
        lock=StubLock(True),
        manifest=lambda: StubManifest(run, _journal(4)),
        store=_ready_store,
    ).run()

    active = report.active_operation
    assert active is not None
    assert (active.kind, active.run_id) == ("build", "3")
    assert (active.sources_done, active.sources_total) == (4, 10)
    assert active.passages_written == 12
    assert active.elapsed_seconds == 80.0
    assert active.eta_seconds == 120  # 6 left at 20 s per source
    assert active.cancel_requested_at == "2026-10-03T11:59:00"


def test_a_resumed_run_ignores_carried_sources_for_the_pace() -> None:
    run = _run("running", total_sources=10, resumed_from="2", carried_sources=2)

    active = (
        _status(
            lock=StubLock(True),
            manifest=lambda: StubManifest(run, _journal(4)),
            store=_ready_store,
        )
        .run()
        .active_operation
    )

    assert active is not None and active.kind == "resume"
    assert active.eta_seconds == 240  # 2 processed in 80 s, 6 left at 40 s each


@pytest.mark.parametrize(
    ("journal", "total"),
    [(0, 10), (4, None)],
)
def test_the_eta_is_null_until_it_is_computable(
    journal: int, total: int | None
) -> None:
    run = _run("running", total_sources=total)

    active = (
        _status(
            lock=StubLock(True),
            manifest=lambda: StubManifest(run, _journal(journal)),
        )
        .run()
        .active_operation
    )

    assert active is not None and active.eta_seconds is None


def test_a_held_lock_without_a_running_run_is_a_warning() -> None:
    report = _status(
        lock=StubLock(True), manifest=lambda: StubManifest(_run("completed"))
    ).run()

    assert report.active_operation is None
    assert "lock_held_without_run" in [w.code for w in report.warnings]


def test_the_view_of_a_build_running_in_another_thread(tmp_path: Path) -> None:
    release = threading.Event()
    reached = threading.Event()
    manifest = InMemoryManifest()
    store = InMemoryPassageStore()
    lock_path = tmp_path / "index.lock"

    class Parser:
        def parse(self, source: Source) -> ParsedDocument:
            return ParsedDocument(source, (ParsedSection("alpha beta gamma delta"),))

    class Sources:
        def enumerate(self) -> tuple[Source, ...]:
            return tuple(
                Source(f"d{i}.txt", f"d{i}.txt", FileType.TXT, content_hash="h")
                for i in range(5)
            )

    class Sparse:
        def encode(self, text: str) -> dict[int, float]:
            return {1: 1.0}

    class Blocking(DeterministicEmbedder):
        calls = 0

        def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
            Blocking.calls += 1
            if Blocking.calls == 3:
                reached.set()
                assert release.wait(10)
            return super().embed(texts)

    build = BuildIndex(
        lock=FileIndexLock(lock_path),
        enumerator=Sources(),
        parsers={FileType.TXT: Parser()},
        chunker=TokenChunker(WhitespaceTokenizer(), 2, 0),
        embedder=Blocking(),
        sparse_encoder=Sparse(),
        passage_writer=store,
        manifest=manifest,
        config={"chunk_size": 2, "chunk_overlap": 0},
        docgrep_version="0.0.0",
        batch_size=100,
        sleep=lambda seconds: None,
    )
    thread = threading.Thread(target=build.run)
    thread.start()
    try:
        assert reached.wait(10)
        report = InspectStatus(
            lock=FileIndexLock(lock_path),
            open_manifest=lambda: manifest,
            open_store=lambda: None,
            store_mode="local",
            runtime_config={},
        ).run()
    finally:
        release.set()
        thread.join(10)

    active = report.active_operation
    assert active is not None
    assert (active.sources_done, active.sources_total) == (2, 5)
    assert active.passages_written == 4 and active.kind == "build"
    assert report.latest_run is not None and report.latest_run.state == "running"
    assert report.index_state == "not_built"


def test_the_json_form_names_every_section() -> None:
    report = _status(
        lock=StubLock(True),
        manifest=lambda: StubManifest(_run("running", total_sources=10), _journal(2)),
        store=_ready_store,
        runtime={"chunk_size": 11},
    ).run()

    data = report.as_dict()

    assert list(data) == [
        "schema_version",
        "index_state",
        "latest_run",
        "active_operation",
        "vector_store",
        "collection",
        "drift",
        "warnings",
    ]
    assert data["drift"] == [{"field": "chunk_size", "indexed": 10, "runtime": 11}]
