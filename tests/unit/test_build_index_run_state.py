"""Build run state: journal, commit order, staging name and orphan cleanup."""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import pytest

from docgrep.adapters.chunking import TokenChunker, WhitespaceTokenizer
from docgrep.adapters.models.deterministic import DeterministicEmbedder
from docgrep.domain.errors import BuildFailed, StoreUnavailable
from docgrep.domain.models import (
    FileType,
    ParsedDocument,
    ParsedSection,
    Passage,
    Source,
)
from docgrep.usecases.build_index import BuildIndex
from fakes import InMemoryManifest, InMemoryPassageStore


class Sources:
    def __init__(self, *sources: Source) -> None:
        self.sources = sources

    def enumerate(self) -> tuple[Source, ...]:
        return self.sources


class Parser:
    def parse(self, source: Source) -> ParsedDocument:
        if source.path == "bad.txt":
            raise ValueError("unparseable")
        return ParsedDocument(source, (ParsedSection("alpha beta gamma delta"),))


class Sparse:
    def encode(self, text: str) -> dict[int, float]:
        return {len(text): 1.0}


class Events:
    def __init__(self) -> None:
        self.log: list[str] = []


class LoggingStore(InMemoryPassageStore):
    def __init__(self, events: Events) -> None:
        super().__init__()
        self.events = events
        self.fail_for: str | None = None

    def recreate_collection(self, dimension: int) -> None:
        self.events.log.append("recreate_collection")
        super().recreate_collection(dimension)

    def upsert(
        self,
        passage: Passage,
        source: Source,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float],
    ) -> None:
        if source.path == self.fail_for:
            raise StoreUnavailable("disk full")
        super().upsert(passage, source, dense_vector, sparse_vector)

    def write_completion_metadata(
        self, metadata: Mapping[str, str | int | float]
    ) -> None:
        self.events.log.append("write_completion_metadata")
        super().write_completion_metadata(metadata)


class LoggingManifest(InMemoryManifest):
    def __init__(self, events: Events) -> None:
        super().__init__()
        self.events = events
        self.fail_commit = False

    def set_staging_collection(self, run_id: str, name: str) -> None:
        self.events.log.append("set_staging_collection")
        super().set_staging_collection(run_id, name)

    def record_source(
        self,
        source: Source,
        status: str,
        error: str | None = None,
        embedding_model: str | None = None,
        *,
        run_id: str | None = None,
        passages: int = 0,
    ) -> None:
        self.events.log.append("record_source")
        super().record_source(
            source,
            status,
            error,
            embedding_model,
            run_id=run_id,
            passages=passages,
        )

    def commit_run(self, run_id: str) -> None:
        self.events.log.append("commit_run")
        if self.fail_commit:
            raise OSError("manifest disk error")
        super().commit_run(run_id)

    def finish_run(
        self,
        run_id: str,
        status: Literal["completed", "aborted"],
        totals: Mapping[str, int],
    ) -> None:
        self.events.log.append(f"finish_run:{status}")
        super().finish_run(run_id, status, totals)


def _source(path: str) -> Source:
    return Source(path, path, FileType.TXT, content_hash="h", size=10)


def _build(
    tmp_path: Path, *paths: str
) -> tuple[BuildIndex, LoggingManifest, LoggingStore, Events]:
    events = Events()
    manifest = LoggingManifest(events)
    store = LoggingStore(events)
    from docgrep.adapters.filelock import FileIndexLock

    use_case = BuildIndex(
        lock=FileIndexLock(tmp_path / "index.lock"),
        enumerator=Sources(*(_source(p) for p in paths)),
        parsers={FileType.TXT: Parser()},
        chunker=TokenChunker(WhitespaceTokenizer(), 2, 0),
        embedder=DeterministicEmbedder(),
        sparse_encoder=Sparse(),
        passage_writer=store,
        manifest=manifest,
        config={"chunk_size": 2, "chunk_overlap": 0},
        docgrep_version="0.0.0",
        sleep=lambda seconds: None,
    )
    return use_case, manifest, store, events


def test_outcomes_go_to_the_journal_and_reach_sources_only_on_completion(
    tmp_path: Path,
) -> None:
    use_case, manifest, _, _ = _build(tmp_path, "good.txt", "bad.txt")
    manifest.record_source(_source("stale.txt"), "indexed")

    summary = use_case.run()

    journal = {r.source.path: r for r in manifest.run_sources(summary.run_id)}
    assert journal["good.txt"].status == "indexed" and journal["good.txt"].passages == 2
    assert journal["bad.txt"].status == "failed:parse_error"
    assert manifest.get_source("stale.txt") is None
    good = manifest.get_source("good.txt")
    assert good is not None and good.status == "indexed"
    assert manifest.get_source("bad.txt") is not None


def test_the_committed_sources_follow_the_alias_swap(tmp_path: Path) -> None:
    use_case, manifest, _, events = _build(tmp_path, "good.txt")

    use_case.run()

    tail = [e for e in events.log if e not in ("record_source",)]
    assert tail == [
        "recreate_collection",
        "set_staging_collection",
        "write_completion_metadata",
        "commit_run",
        "finish_run:completed",
    ]


def test_sources_are_unchanged_while_the_build_is_still_running(
    tmp_path: Path,
) -> None:
    use_case, manifest, store, events = _build(tmp_path, "good.txt", "second.txt")
    manifest.record_source(_source("old.txt"), "indexed", embedding_model="old")
    seen: dict[str, object] = {}

    class Probe(DeterministicEmbedder):
        def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
            if "second" not in seen:
                run = manifest.latest_run()
                assert run is not None and run.status == "running"
                seen["second"] = True
                seen["old"] = manifest.get_source("old.txt")
                seen["new"] = manifest.get_source("good.txt")
                seen["journal"] = [
                    r.source.path for r in manifest.run_sources(run.run_id)
                ]
            return super().embed(texts)

    use_case.embedder = Probe()
    use_case.run()

    assert seen["old"] is not None and seen["new"] is None
    assert seen["journal"] == []


def test_an_aborted_build_leaves_the_committed_sources_untouched(
    tmp_path: Path,
) -> None:
    use_case, manifest, store, _ = _build(tmp_path, "good.txt", "second.txt")
    manifest.record_source(_source("old.txt"), "indexed", embedding_model="old")
    store.fail_for = "second.txt"

    with pytest.raises(StoreUnavailable):
        use_case.run()

    run = manifest.latest_run()
    assert run is not None and run.status == "aborted"
    assert [r.source.path for r in manifest.run_sources(run.run_id)] == ["good.txt"]
    assert manifest.get_source("old.txt") is not None
    assert manifest.get_source("good.txt") is None
    assert run.staging_collection == "memory-staging"
    assert store.read_completion_metadata() is None


def test_the_staging_name_is_recorded_before_the_first_outcome(
    tmp_path: Path,
) -> None:
    use_case, manifest, _, events = _build(tmp_path, "good.txt")

    summary = use_case.run()

    assert events.log.index("set_staging_collection") < events.log.index(
        "record_source"
    )
    run = manifest.latest_run()
    assert run is not None and run.run_id == summary.run_id
    assert run.staging_collection == "memory-staging"
    assert run.resumed_from is None and run.cancel_requested_at is None


def test_orphaned_staging_collections_are_dropped_at_the_start(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    use_case, _, store, events = _build(tmp_path, "good.txt")
    store.orphans.extend(["docgrep_" + "a" * 32, "docgrep_" + "b" * 32])

    with caplog.at_level(logging.INFO, logger="docgrep"):
        use_case.run()

    assert store.orphans == []
    dropped = [
        r.__dict__["collection"]
        for r in caplog.records
        if r.getMessage() == "orphaned_staging_collection_dropped"
    ]
    assert dropped == ["docgrep_" + "a" * 32, "docgrep_" + "b" * 32]


def test_a_failing_commit_aborts_the_run_and_reports_the_error(
    tmp_path: Path,
) -> None:
    use_case, manifest, store, events = _build(tmp_path, "good.txt")
    manifest.record_source(_source("old.txt"), "indexed")
    manifest.fail_commit = True

    with pytest.raises(BuildFailed):
        use_case.run()

    # The live index is complete and valid, so the run stays swapped and the
    # next start finishes it.
    run = manifest.latest_run()
    assert run is not None and run.status == "swapped"
    assert manifest.get_source("old.txt") is not None
    assert events.log[-1] == "commit_run"


def test_an_enumeration_failure_drops_nothing_it_should_not(tmp_path: Path) -> None:
    use_case, manifest, store, _ = _build(tmp_path, "good.txt")
    manifest.record_source(_source("old.txt"), "indexed")

    class Broken:
        def enumerate(self) -> tuple[Source, ...]:
            raise OSError("corpus unavailable")

    use_case.enumerator = Broken()
    with pytest.raises(BuildFailed):
        use_case.run()

    assert manifest.get_source("old.txt") is not None
    run = manifest.latest_run()
    assert run is not None and run.staging_collection is None
