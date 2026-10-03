"""Cooperative cancel: the build, the use case, the signals and the exit code."""

import logging
import signal
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from docgrep.adapters.chunking import TokenChunker, WhitespaceTokenizer
from docgrep.adapters.filelock import FileIndexLock
from docgrep.adapters.models.deterministic import DeterministicEmbedder
from docgrep.api.cli import main
from docgrep.api.container import Services
from docgrep.api.signals import graceful_cancel
from docgrep.domain.models import (
    FileType,
    ManifestRunRecord,
    ParsedDocument,
    ParsedSection,
    Source,
)
from docgrep.usecases.build_index import BuildIndex, BuildSummary
from docgrep.usecases.cancel_index import CancelIndex
from fakes import InMemoryManifest, InMemoryPassageStore


class Sources:
    def __init__(self, *paths: str) -> None:
        self.paths = paths

    def enumerate(self) -> tuple[Source, ...]:
        return tuple(Source(p, p, FileType.TXT, content_hash="h") for p in self.paths)


class Parser:
    """One passage per word pair, so a long source needs several batches."""

    def parse(self, source: Source) -> ParsedDocument:
        words = " ".join(f"w{i}" for i in range(8))
        return ParsedDocument(source, (ParsedSection(words),))


class Sparse:
    def encode(self, text: str) -> dict[int, float]:
        return {len(text): 1.0}


class Hooked(DeterministicEmbedder):
    """Run a callback before the n-th embed call."""

    def __init__(self) -> None:
        self.calls = 0
        self.on_call: dict[int, Callable[[], None]] = {}

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        self.calls += 1
        hook = self.on_call.get(self.calls)
        if hook:
            hook()
        return super().embed(texts)


def _build(
    tmp_path: Path, *paths: str, batch_size: int = 2
) -> tuple[BuildIndex, InMemoryManifest, InMemoryPassageStore, Hooked]:
    manifest = InMemoryManifest()
    store = InMemoryPassageStore()
    embedder = Hooked()
    use_case = BuildIndex(
        lock=FileIndexLock(tmp_path / "index.lock"),
        enumerator=Sources(*paths),
        parsers={FileType.TXT: Parser()},
        chunker=TokenChunker(WhitespaceTokenizer(), 2, 0),
        embedder=embedder,
        sparse_encoder=Sparse(),
        passage_writer=store,
        manifest=manifest,
        config={"chunk_size": 2, "chunk_overlap": 0},
        docgrep_version="0.0.0",
        batch_size=batch_size,
        sleep=lambda seconds: None,
    )
    return use_case, manifest, store, embedder


def _run(manifest: InMemoryManifest) -> ManifestRunRecord:
    run = manifest.latest_run()
    assert run is not None
    return run


def test_cancel_between_sources_stops_before_the_next_source(tmp_path: Path) -> None:
    use_case, manifest, store, embedder = _build(
        tmp_path, "a.txt", "b.txt", "c.txt", batch_size=10
    )
    embedder.on_call[1] = use_case.request_cancel
    manifest.record_source(Source("old.txt", "old.txt", FileType.TXT), "indexed")

    summary = use_case.run()

    assert summary.cancelled and summary.indexed == 1
    run = _run(manifest)
    assert run.status == "cancelled" and run.cancel_requested_at is not None
    assert [r.source.path for r in manifest.run_sources(run.run_id)] == ["a.txt"]
    assert manifest.get_source("old.txt") is not None
    assert manifest.get_source("a.txt") is None
    assert store.staging_name() == "memory-staging"
    assert store.read_completion_metadata() is None
    with use_case.lock.hold():  # released
        pass


def test_cancel_in_the_middle_of_a_source_does_not_journal_it(
    tmp_path: Path,
) -> None:
    use_case, manifest, store, embedder = _build(tmp_path, "a.txt", "b.txt")
    # Each source has four passages, so two embed calls at batch size 2. Cancel
    # while the first batch of b.txt (call 3) is embedded: the batch finishes and
    # the second one never starts.
    embedder.on_call[3] = use_case.request_cancel
    manifest.record_source(Source("old.txt", "old.txt", FileType.TXT), "indexed")

    summary = use_case.run()

    assert summary.cancelled
    run = _run(manifest)
    assert run.status == "cancelled"
    assert [r.source.path for r in manifest.run_sources(run.run_id)] == ["a.txt"]
    assert embedder.calls == 3
    assert manifest.get_source("old.txt") is not None
    assert manifest.get_source("a.txt") is None and manifest.get_source("b.txt") is None
    assert store.count_passages() == 0  # nothing published


def test_a_flag_written_by_another_process_stops_the_build(tmp_path: Path) -> None:
    use_case, manifest, _, embedder = _build(tmp_path, "a.txt", "b.txt")
    embedder.on_call[1] = lambda: manifest.request_cancel(_run(manifest).run_id)

    summary = use_case.run()

    assert summary.cancelled
    assert _run(manifest).status == "cancelled"


def test_cancel_after_the_last_source_still_stops_before_the_swap(
    tmp_path: Path,
) -> None:
    use_case, manifest, store, embedder = _build(tmp_path, "a.txt", batch_size=10)
    embedder.on_call[1] = use_case.request_cancel

    summary = use_case.run()

    assert summary.cancelled and store.read_completion_metadata() is None
    assert _run(manifest).status == "cancelled"


def test_a_later_build_is_not_cancelled_by_an_earlier_request(
    tmp_path: Path,
) -> None:
    use_case, manifest, _, embedder = _build(tmp_path, "a.txt")
    embedder.on_call[1] = use_case.request_cancel
    assert use_case.run().cancelled

    embedder.on_call.clear()
    summary = use_case.run()

    assert not summary.cancelled and _run(manifest).status == "completed"


def test_the_build_logs_a_cancel_without_raising(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    use_case, _, _, embedder = _build(tmp_path, "a.txt", "b.txt")
    embedder.on_call[1] = use_case.request_cancel

    with caplog.at_level(logging.INFO, logger="docgrep"):
        use_case.run()

    assert [r for r in caplog.records if r.getMessage() == "index_build_cancelled"]


# --- the cancel use case ----------------------------------------------------


def _manifest_with_run(status: str = "running") -> InMemoryManifest:
    manifest = InMemoryManifest()
    run_id = manifest.start_run({})
    if status == "completed":
        manifest.finish_run(run_id, "completed", {})
    return manifest


def test_cancel_with_nothing_running_says_so(tmp_path: Path) -> None:
    lock = FileIndexLock(tmp_path / "index.lock")

    assert CancelIndex(lock=lock, manifest=None).run().outcome == "nothing_running"
    empty = InMemoryManifest()
    assert CancelIndex(lock=lock, manifest=empty).run().outcome == "nothing_running"
    done = _manifest_with_run("completed")
    assert CancelIndex(lock=lock, manifest=done).run().outcome == "nothing_running"
    with lock.hold():  # held, but the latest run is not running
        assert CancelIndex(
            lock=FileIndexLock(lock.path), manifest=done
        ).run().outcome == ("nothing_running")


def test_cancel_on_an_interrupted_run_reports_it(tmp_path: Path) -> None:
    manifest = _manifest_with_run()
    result = CancelIndex(
        lock=FileIndexLock(tmp_path / "index.lock"), manifest=manifest
    ).run()

    assert result.outcome == "interrupted" and result.run_id == "1"
    assert manifest.cancel_requested_at("1") is None  # nothing was written


def test_cancel_is_idempotent_and_reports_the_original_time(tmp_path: Path) -> None:
    manifest = _manifest_with_run()
    holder = FileIndexLock(tmp_path / "index.lock")
    with holder.hold():
        cancel = CancelIndex(lock=FileIndexLock(holder.path), manifest=manifest)
        first = cancel.run()
        second = cancel.run()

    assert first.outcome == "requested" and first.requested_at is not None
    assert second.outcome == "already_requested"
    assert second.requested_at == first.requested_at


def test_the_probe_creates_no_lock_file(tmp_path: Path) -> None:
    lock = FileIndexLock(tmp_path / "data" / "index.lock")

    assert lock.is_held() is False
    assert not (tmp_path / "data").exists()


# --- signals ----------------------------------------------------------------


class Registry:
    """Stand-in for signal.signal that records handlers and returns the old one."""

    def __init__(self) -> None:
        self.handlers: dict[int, Any] = {
            signal.SIGINT: "old-int",
            signal.SIGTERM: "old-term",
        }
        self.delivered: list[int] = []

    def __call__(self, number: int, handler: Any) -> Any:
        previous = self.handlers[number]
        self.handlers[number] = handler
        return previous


def test_the_first_signal_requests_a_cancel_once_and_handlers_are_restored() -> None:
    registry = Registry()
    requests: list[str] = []
    notes: list[str] = []

    with graceful_cancel(
        lambda: requests.append("cancel"),
        notes.append,
        register=registry,
        raise_signal=registry.delivered.append,
    ):
        handler = registry.handlers[signal.SIGINT]
        assert callable(handler) and registry.handlers[signal.SIGTERM] is handler
        handler(signal.SIGINT, None)

        assert requests == ["cancel"]
        assert len(notes) == 1
        assert "after the current batch" in notes[0] and "Ctrl-C again" in notes[0]
        assert registry.delivered == []

    assert registry.handlers == {signal.SIGINT: "old-int", signal.SIGTERM: "old-term"}
    assert requests == ["cancel"]


def test_the_second_signal_restores_the_defaults_and_redelivers_it() -> None:
    registry = Registry()
    requests: list[str] = []

    with graceful_cancel(
        lambda: requests.append("cancel"),
        lambda message: None,
        register=registry,
        raise_signal=registry.delivered.append,
    ):
        handler = registry.handlers[signal.SIGTERM]
        handler(signal.SIGTERM, None)
        handler(signal.SIGTERM, None)

        assert requests == ["cancel"]  # not requested twice
        assert registry.delivered == [signal.SIGTERM]
        assert registry.handlers == {
            signal.SIGINT: "old-int",
            signal.SIGTERM: "old-term",
        }


def test_handlers_are_restored_when_the_block_raises() -> None:
    registry = Registry()

    with pytest.raises(RuntimeError):
        with graceful_cancel(lambda: None, lambda m: None, register=registry):
            raise RuntimeError("boom")

    assert registry.handlers == {signal.SIGINT: "old-int", signal.SIGTERM: "old-term"}


# --- the console ------------------------------------------------------------


class _FakeBuild:
    def __init__(self, summary: BuildSummary) -> None:
        self.summary = summary
        self.cancel_requests = 0

    def run(self, *, refuse_resumable: bool = False) -> BuildSummary:
        return self.summary

    def resume(self, *, retry_failed: bool = True) -> BuildSummary:
        return self.summary

    def request_cancel(self) -> None:
        self.cancel_requests += 1


def _services(summary: BuildSummary) -> Services:
    return Services(search=None, build=_FakeBuild(summary))  # type: ignore[arg-type]  # only build is used


def test_a_cancelled_build_exits_with_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("DOCGREP_DATA_DIR", str(tmp_path / "data"))
    registry = Registry()
    summary = BuildSummary("7", 3, 0, 0, 1.5, (), cancelled=True)

    status = main(
        ["index"],
        container_factory=lambda settings: _services(summary),
        register_signal=registry,
    )

    assert status == 2
    assert "Build 7 cancelled" in capsys.readouterr().out
    assert registry.handlers == {signal.SIGINT: "old-int", signal.SIGTERM: "old-term"}


def test_a_completed_build_still_exits_with_0_and_a_failed_one_with_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCGREP_DATA_DIR", str(tmp_path / "data"))
    ok = BuildSummary("1", 1, 0, 0, 0.1, ())
    bad = BuildSummary("2", 0, 0, 1, 0.1, ())

    assert (
        main(
            ["index"],
            container_factory=lambda s: _services(ok),
            register_signal=Registry(),
        )
        == 0
    )
    assert (
        main(
            ["index"],
            container_factory=lambda s: _services(bad),
            register_signal=Registry(),
        )
        == 1
    )


@pytest.mark.parametrize(
    ("setup", "status", "text"),
    [("none", 1, "Nothing is running"), ("interrupted", 0, "interrupted")],
)
def test_the_cancel_command_reports_and_sets_its_exit_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    setup: str,
    status: int,
    text: str,
) -> None:
    monkeypatch.setenv("DOCGREP_DATA_DIR", str(tmp_path / "data"))
    manifest = _manifest_with_run() if setup == "interrupted" else None
    cancel = CancelIndex(lock=FileIndexLock(tmp_path / "index.lock"), manifest=manifest)

    assert main(["cancel"], cancel_factory=lambda settings: cancel) == status
    assert text in capsys.readouterr().out


def test_the_cancel_command_creates_nothing_in_an_unused_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCGREP_DATA_DIR", str(tmp_path / "data"))

    assert main(["cancel"]) == 1
    assert not (tmp_path / "data").exists()
