"""Run state machine: transitions, crash recovery, interrupted runs and aborts."""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest

from docgrep.adapters.chunking import TokenChunker, WhitespaceTokenizer
from docgrep.adapters.filelock import FileIndexLock
from docgrep.adapters.models.deterministic import DeterministicEmbedder
from docgrep.domain.errors import IndexLockHeld, StoreUnavailable
from docgrep.domain.models import (
    FileType,
    ManifestRunRecord,
    ParsedDocument,
    ParsedSection,
    Passage,
    RunStatus,
    Source,
)
from docgrep.domain.run_state import is_interrupted, run_state
from docgrep.usecases.build_index import BuildIndex
from fakes import InMemoryManifest, InMemoryPassageStore


class Crash(BaseException):
    """A process death: not an Exception, so no handler of the build sees it."""


class Sources:
    def __init__(self, *paths: str) -> None:
        self.paths = paths

    def enumerate(self) -> tuple[Source, ...]:
        return tuple(Source(p, p, FileType.TXT, content_hash="h") for p in self.paths)


class Parser:
    def parse(self, source: Source) -> ParsedDocument:
        return ParsedDocument(source, (ParsedSection(f"alpha {source.path} delta"),))


class Sparse:
    def encode(self, text: str) -> dict[int, float]:
        return {len(text): 1.0}


class CrashingManifest(InMemoryManifest):
    crash_on_commit = False
    seen_status_at_commit: str | None = None

    def commit_run(self, run_id: str) -> None:
        run = self.latest_run()
        self.seen_status_at_commit = run.status if run else None
        if self.crash_on_commit:
            raise Crash
        super().commit_run(run_id)


class FailingStore(InMemoryPassageStore):
    fail_for: str | None = None

    def upsert_batch(
        self,
        items: Sequence[tuple[Passage, Source, Sequence[float], Mapping[int, float]]],
    ) -> None:
        if any(source.path == self.fail_for for _, source, _, _ in items):
            raise StoreUnavailable("disk full")
        super().upsert_batch(items)


def _build(
    tmp_path: Path, *paths: str
) -> tuple[BuildIndex, CrashingManifest, FailingStore]:
    manifest = CrashingManifest()
    store = FailingStore()
    use_case = BuildIndex(
        lock=FileIndexLock(tmp_path / "index.lock"),
        enumerator=Sources(*paths),
        parsers={FileType.TXT: Parser()},
        chunker=TokenChunker(WhitespaceTokenizer(), 20, 0),
        embedder=DeterministicEmbedder(),
        sparse_encoder=Sparse(),
        passage_writer=store,
        manifest=manifest,
        config={"chunk_size": 20, "chunk_overlap": 0},
        docgrep_version="0.0.0",
        sleep=lambda seconds: None,
    )
    return use_case, manifest, store


def _run(manifest: InMemoryManifest) -> ManifestRunRecord:
    run = manifest.latest_run()
    assert run is not None
    return run


def test_a_completed_run_passes_through_swapped_and_records_its_plan(
    tmp_path: Path,
) -> None:
    use_case, manifest, _ = _build(tmp_path, "a.txt", "b.txt", "c.bin")

    use_case.run()

    assert manifest.seen_status_at_commit == "swapped"
    run = _run(manifest)
    assert run.status == "completed" and run.total_sources == 3


def test_a_crash_between_swap_and_commit_leaves_the_run_swapped(
    tmp_path: Path,
) -> None:
    use_case, manifest, store = _build(tmp_path, "a.txt")
    manifest.record_source(Source("old.txt", "old.txt", FileType.TXT), "indexed")
    manifest.crash_on_commit = True

    with pytest.raises(Crash):
        use_case.run()

    run = _run(manifest)
    assert run.status == "swapped"
    assert manifest.get_source("old.txt") is not None
    assert manifest.get_source("a.txt") is None
    assert store.count_passages() == 1  # the new index is already live
    with use_case.lock.hold():  # the lock was released on the way out
        pass


def test_the_next_build_start_finishes_a_swapped_run_first(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    use_case, manifest, _ = _build(tmp_path, "a.txt")
    manifest.crash_on_commit = True
    with pytest.raises(Crash):
        use_case.run()
    crashed = _run(manifest).run_id
    manifest.crash_on_commit = False
    seen: dict[str, object] = {}

    class Probe(DeterministicEmbedder):
        def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
            if not seen:
                seen["recovered"] = manifest.get_source("a.txt")
                seen["status"] = manifest.latest_run()
            return super().embed(texts)

    use_case.embedder = Probe()
    with caplog.at_level(logging.WARNING, logger="docgrep"):
        summary = use_case.run()

    recovered = seen["recovered"]
    assert recovered is not None
    assert summary.run_id != crashed
    previous = [r for r in manifest._runs if r.run_id == crashed][0]
    assert previous.status == "completed"
    events = [r for r in caplog.records if r.getMessage() == "run_recovered"]
    assert len(events) == 1 and events[0].levelno == logging.WARNING
    assert events[0].__dict__["run_id"] == crashed


def test_a_run_stored_as_running_is_interrupted_only_without_the_lock(
    tmp_path: Path,
) -> None:
    use_case, manifest, _ = _build(tmp_path, "a.txt")
    run = ManifestRunRecord("1", "running", ())
    probe = FileIndexLock(tmp_path / "index.lock")

    assert probe.is_held() is False
    assert is_interrupted(run, lock_held=probe.is_held()) is True
    assert run_state(run, lock_held=False) == "interrupted"

    with use_case.lock.hold():
        assert probe.is_held() is True
        assert is_interrupted(run, lock_held=probe.is_held()) is False
        assert run_state(run, lock_held=True) == "running"
    assert probe.is_held() is False  # the probe never keeps the lock


def test_only_a_running_run_can_be_interrupted() -> None:
    for status in ("swapped", "completed", "aborted", "cancelled"):
        run = ManifestRunRecord("1", cast(RunStatus, status), ())
        assert not is_interrupted(run, lock_held=False)
        assert run_state(run, lock_held=False) == status


def test_the_probe_does_not_block_a_later_holder(tmp_path: Path) -> None:
    lock = FileIndexLock(tmp_path / "index.lock")
    lock.is_held()
    with lock.hold(), pytest.raises(IndexLockHeld):
        with FileIndexLock(tmp_path / "index.lock").hold():
            pass


def test_an_aborted_build_retains_staging_and_journal_and_keeps_the_live_index(
    tmp_path: Path,
) -> None:
    use_case, manifest, store = _build(tmp_path, "a.txt", "b.txt")
    use_case.run()
    live_before = [c.passage.text for c in store.retrieve((1.0,) * 64, {}, 10)]
    manifest.record_source(Source("old.txt", "old.txt", FileType.TXT), "indexed")
    store.fail_for = "b.txt"

    with pytest.raises(StoreUnavailable):
        use_case.run()

    run = _run(manifest)
    assert run.status == "aborted"
    assert store.staging_name() == "memory-staging"
    assert store.staging_collections() == ("memory-staging",)
    assert run.staging_collection == "memory-staging"
    assert [r.source.path for r in manifest.run_sources(run.run_id)] == ["a.txt"]
    assert manifest.get_source("old.txt") is not None
    assert store.read_completion_metadata() is not None
    assert [c.passage.text for c in store.retrieve((1.0,) * 64, {}, 10)] == live_before


def test_a_fresh_build_removes_the_retained_staging_collection(
    tmp_path: Path,
) -> None:
    use_case, _, store = _build(tmp_path, "a.txt", "b.txt")
    store.fail_for = "b.txt"
    with pytest.raises(StoreUnavailable):
        use_case.run()
    assert store.staging_collections() == ("memory-staging",)

    store.fail_for = None
    use_case.run()

    assert store.staging_collections() == ()
    assert store.count_passages() == 2
