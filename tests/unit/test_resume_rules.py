"""Resume rules on the in-memory fakes: plan, carry-over, drift and refusal."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from docgrep.adapters.chunking import TokenChunker, WhitespaceTokenizer
from docgrep.adapters.filelock import FileIndexLock
from docgrep.adapters.models.deterministic import DeterministicEmbedder
from docgrep.domain.errors import (
    ConfigDrift,
    CorpusDrift,
    NothingToResume,
    ResumeAvailable,
    StagingInconsistent,
    StoreUnavailable,
)
from docgrep.domain.models import (
    FileType,
    ParsedDocument,
    ParsedSection,
    Source,
)
from docgrep.domain.run_state import VECTOR_CONFIG_KEYS, config_drift
from docgrep.usecases.build_index import BuildIndex
from fakes import InMemoryManifest, InMemoryPassageStore


class Corpus:
    """A mutable corpus: path -> content hash."""

    def __init__(self, count: int) -> None:
        self.hashes = {f"doc{i:03d}.txt": "h1" for i in range(count)}

    def enumerate(self) -> tuple[Source, ...]:
        return tuple(
            Source(path, path, FileType.TXT, content_hash=digest)
            for path, digest in sorted(self.hashes.items())
        )


class Parser:
    def __init__(self) -> None:
        self.failing: set[str] = set()

    def parse(self, source: Source) -> ParsedDocument:
        if source.path in self.failing:
            raise ValueError("unparseable")
        return ParsedDocument(
            source, (ParsedSection(f"alpha {source.path} {source.content_hash}"),)
        )


class Sparse:
    def encode(self, text: str) -> dict[int, float]:
        return {len(text): 1.0}


class Crash(BaseException):
    pass


class Harness:
    def __init__(self, tmp_path: Path, count: int = 5) -> None:
        self.corpus = Corpus(count)
        self.parser = Parser()
        self.manifest = InMemoryManifest()
        self.store = InMemoryPassageStore()
        self.tmp_path = tmp_path
        self.config: dict[str, str | int | float] = {
            "chunk_size": 3,
            "chunk_overlap": 0,
        }

    def build(self, embedder: DeterministicEmbedder | None = None) -> BuildIndex:
        return BuildIndex(
            lock=FileIndexLock(self.tmp_path / "index.lock"),
            enumerator=self.corpus,
            parsers={FileType.TXT: self.parser},
            chunker=TokenChunker(WhitespaceTokenizer(), 3, 0),
            embedder=embedder or DeterministicEmbedder(),
            sparse_encoder=Sparse(),
            passage_writer=self.store,
            manifest=self.manifest,
            config=self.config,
            docgrep_version="0.0.0",
            sleep=lambda seconds: None,
        )

    def interrupted_after(self, sources: int) -> BuildIndex:
        """Run a build that is cancelled after ``sources`` sources."""
        use_case = self.build()
        embedder = use_case.embedder
        calls = 0

        class Hook(DeterministicEmbedder):
            def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
                nonlocal calls
                calls += 1
                if calls == sources:
                    use_case.request_cancel()
                return embedder.embed(texts)

        use_case.embedder = Hook()
        summary = use_case.run()
        assert summary.cancelled
        return self.build()


def test_the_resume_plan_covers_what_the_run_did_not_finish(tmp_path: Path) -> None:
    harness = Harness(tmp_path, 6)
    harness.interrupted_after(2)

    summary = harness.build().resume()

    assert (summary.carried_over, summary.processed) == (2, 4)
    assert (summary.retried, summary.removed) == (0, 0)
    assert (summary.indexed, summary.skipped, summary.failed) == (6, 0, 0)
    assert harness.store.count_passages() == 6
    run = harness.manifest.latest_run()
    assert run is not None and run.resumed_from == "1" and run.status == "completed"
    assert harness.store.staging_collections() == ()


def test_failed_sources_are_retried_unless_told_not_to(tmp_path: Path) -> None:
    for retry, indexed, failed, retried in ((True, 5, 0, 1), (False, 4, 1, 0)):
        harness = Harness(tmp_path / str(retry), 5)
        (tmp_path / str(retry)).mkdir()
        harness.parser.failing.add("doc001.txt")
        harness.interrupted_after(3)
        harness.parser.failing.clear()  # the failure was transient

        summary = harness.build().resume(retry_failed=retry)

        assert (summary.indexed, summary.failed, summary.retried) == (
            indexed,
            failed,
            retried,
        )
        assert [f.path for f in summary.failures] == ([] if retry else ["doc001.txt"])


def test_a_changed_source_is_staged_again_and_a_vanished_one_is_removed(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path, 100)
    harness.interrupted_after(40)
    harness.corpus.hashes["doc003.txt"] = "h2"  # 1 of 40, below the limit
    del harness.corpus.hashes["doc004.txt"]  # 2 of 40 is exactly 5%

    summary = harness.build().resume()

    assert (summary.carried_over, summary.removed) == (38, 1)
    assert summary.processed == 99 - 38
    texts = {
        item[0].text for item in harness.store._items.values()
    }  # the published index
    assert "alpha doc003.txt h2" in texts and "alpha doc003.txt h1" not in texts
    assert not any("doc004.txt" in text for text in texts)
    assert harness.store.count_passages() == 99


def test_drift_above_five_percent_is_rejected_before_anything_starts(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path, 100)
    harness.interrupted_after(40)
    harness.corpus.hashes["doc003.txt"] = "h2"
    harness.corpus.hashes["doc005.txt"] = "h2"
    del harness.corpus.hashes["doc004.txt"]  # 3 of 40 is 7.5%
    before = harness.manifest.latest_run()

    with pytest.raises(CorpusDrift) as raised:
        harness.build().resume()

    assert raised.value.category == "corpus_drift" and "--restart" in str(raised.value)
    assert harness.manifest.latest_run() == before


def test_config_drift_names_the_fields_and_ignores_operational_keys(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path)
    harness.interrupted_after(2)
    harness.config = {"chunk_size": 4, "chunk_overlap": 0, "device": "cpu"}

    with pytest.raises(ConfigDrift) as raised:
        harness.build().resume()

    assert raised.value.details["fields"] == "chunk_size"
    harness.config = {"chunk_size": 3, "chunk_overlap": 0, "device": "cuda"}
    assert not harness.build().resume().cancelled


def test_the_compared_keys_are_the_vector_affecting_ones() -> None:
    assert VECTOR_CONFIG_KEYS == (
        "embedding_model",
        "embedding_dimension",
        "chunk_size",
        "chunk_overlap",
        "tokenizer",
        "normalize_embeddings",
    )
    same = {"chunk_size": 1, "device": "cpu", "batch_size": 8}
    assert config_drift(same, {"chunk_size": 1, "device": "cuda"}) == ()
    assert config_drift({"chunk_size": 1}, {"chunk_size": 1, "tokenizer": "m"}) == (
        "tokenizer",
    )


def test_a_staging_collection_that_vanished_is_rejected(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.interrupted_after(2)
    harness.store.clear_completion_metadata()  # the staging collection is gone

    with pytest.raises(StagingInconsistent) as raised:
        harness.build().resume()

    assert "--restart" in str(raised.value)


def test_nothing_to_resume_without_a_run_or_after_a_completed_one(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path)
    with pytest.raises(NothingToResume):
        harness.build().resume()
    harness.build().run()
    with pytest.raises(NothingToResume) as raised:
        harness.build().resume()
    assert raised.value.details["state"] == "completed"


def test_a_killed_run_is_resumable(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    use_case = harness.build()
    embedder = use_case.embedder
    calls = 0

    class Dies(DeterministicEmbedder):
        def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
            nonlocal calls
            calls += 1
            if calls == 3:
                raise Crash
            return embedder.embed(texts)

    use_case.embedder = Dies()
    with pytest.raises(Crash):
        use_case.run()
    run = harness.manifest.latest_run()
    assert run is not None and run.status == "running"  # interrupted, not stored

    with pytest.raises(ResumeAvailable) as refused:
        harness.build().run(refuse_resumable=True)
    assert refused.value.details["state"] == "interrupted"

    summary = harness.build().resume()
    assert (summary.carried_over, summary.processed) == (2, 3)
    assert harness.store.count_passages() == 5


def test_a_swapped_run_is_only_finished_by_resume(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    original = harness.manifest.commit_run

    def die(run_id: str) -> None:
        raise Crash

    harness.manifest.commit_run = die  # type: ignore[method-assign]  # simulate a crash
    with pytest.raises(Crash):
        harness.build().run()
    harness.manifest.commit_run = original  # type: ignore[method-assign]
    run = harness.manifest.latest_run()
    assert run is not None and run.status == "swapped"

    summary = harness.build().resume()

    assert summary.recovered and summary.run_id == run.run_id and summary.indexed == 5
    done = harness.manifest.latest_run()
    assert done is not None and done.status == "completed"
    assert harness.manifest.get_source("doc000.txt") is not None
    with pytest.raises(NothingToResume):
        harness.build().resume()


def test_plain_index_refuses_every_resumable_state_and_passes_otherwise(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path)
    harness.build().run(refuse_resumable=True)  # no run, then a completed one
    harness.build().run(refuse_resumable=True)
    harness.interrupted_after(2)
    with pytest.raises(ResumeAvailable) as refused:
        harness.build().run(refuse_resumable=True)
    assert refused.value.category == "resume_available"
    assert refused.value.details["state"] == "cancelled"
    assert "--resume" in str(refused.value) and "--restart" in str(refused.value)

    harness.build().run()  # --restart: discards the retained staging
    assert harness.store.staging_collections() == ()
    latest = harness.manifest.latest_run()
    assert latest is not None and latest.status == "completed"


def test_plain_index_refuses_an_aborted_run(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    original = harness.store.upsert_batch

    def boom(items: object) -> None:
        raise StoreUnavailable("disk full")

    harness.store.upsert_batch = boom  # type: ignore[method-assign,assignment]  # inject a failure
    with pytest.raises(StoreUnavailable):
        harness.build().run()
    harness.store.upsert_batch = original  # type: ignore[method-assign]

    with pytest.raises(ResumeAvailable) as refused:
        harness.build().run(refuse_resumable=True)
    assert refused.value.details["state"] == "aborted"
    assert harness.build().resume().processed == 5


def test_plain_index_finishes_a_swapped_run_first_and_then_builds(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path)

    def die(run_id: str) -> None:
        raise Crash

    original = harness.manifest.commit_run
    harness.manifest.commit_run = die  # type: ignore[method-assign]  # simulate a crash
    with pytest.raises(Crash):
        harness.build().run()
    harness.manifest.commit_run = original  # type: ignore[method-assign]

    summary = harness.build().run(refuse_resumable=True)

    assert not summary.cancelled and summary.run_id == "2"
    first = [r for r in harness.manifest._runs if r.run_id == "1"][0]
    assert first.status == "completed"
