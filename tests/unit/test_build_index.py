"""Build orchestration through domain ports and in-memory fakes."""

import fcntl
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest

from docgrep.adapters.chunking import TokenChunker, WhitespaceTokenizer
from docgrep.adapters.filelock import FileIndexLock
from docgrep.adapters.models.deterministic import DeterministicEmbedder
from docgrep.domain.errors import (
    BuildFailed,
    IndexLockHeld,
    ParseError,
    StoreUnavailable,
)
from docgrep.domain.models import (
    FileType,
    ParsedDocument,
    ParsedSection,
    Passage,
    Source,
)
from docgrep.domain.protocols import IndexLock
from docgrep.usecases.build_index import BuildIndex
from fakes import InMemoryManifest, InMemoryPassageStore


class Sources:
    def __init__(self, *sources: Source) -> None:
        self.sources = sources

    def enumerate(self) -> tuple[Source, ...]:
        return self.sources


class BrokenSources:
    def enumerate(self) -> tuple[Source, ...]:
        raise OSError("corpus unavailable")


class Parser:
    def parse(self, source: Source) -> ParsedDocument:
        if source.path == "bad.txt":
            raise ParseError("bad source")
        if source.path == "scan.pdf":
            return ParsedDocument(source, (), True)
        return ParsedDocument(source, (ParsedSection("alpha beta gamma"),))


class Sparse:
    def encode(self, text: str) -> dict[int, float]:
        return {len(text): 1.0}


class FlakyEmbedder(DeterministicEmbedder):
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("temporary model failure")
        return super().embed(texts)


class FailingStore(InMemoryPassageStore):
    def upsert(
        self,
        passage: Passage,
        source: Source,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float],
    ) -> None:
        raise StoreUnavailable("disk full")


def source(path: str, file_type: FileType = FileType.TXT) -> Source:
    return Source(path, path, file_type, content_hash="digest", size=10)


def build(
    tmp_path: Path,
    sources: tuple[Source, ...],
    *,
    store: InMemoryPassageStore | None = None,
    embedder: DeterministicEmbedder | None = None,
    lock: IndexLock | None = None,
    manifest: InMemoryManifest | None = None,
) -> tuple[BuildIndex, InMemoryManifest, InMemoryPassageStore]:
    manifest = manifest or InMemoryManifest()
    passage_store = store or InMemoryPassageStore()
    use_case = BuildIndex(
        lock=lock or FileIndexLock(tmp_path / "index.lock"),
        enumerator=Sources(*sources),
        parsers={FileType.TXT: Parser(), FileType.PDF: Parser()},
        chunker=TokenChunker(WhitespaceTokenizer(), 2, 0),
        embedder=embedder or DeterministicEmbedder(),
        sparse_encoder=Sparse(),
        passage_writer=passage_store,
        manifest=manifest,
        config={"chunk_size": 2, "chunk_overlap": 0},
        docgrep_version="0.1.0",
        sleep=lambda seconds: None,
    )
    return use_case, manifest, passage_store


def status(manifest: InMemoryManifest, path: str) -> str:
    record = manifest.get_source(path)
    assert record is not None
    return record.status


def run_status(manifest: InMemoryManifest) -> str:
    run = manifest.latest_run()
    assert run is not None
    return run.status


def test_build_records_each_outcome_and_completes_last(tmp_path: Path) -> None:
    use_case, manifest, store = build(
        tmp_path,
        (
            source("good.txt"),
            source("bad.txt"),
            source("scan.pdf", FileType.PDF),
            source("other.epub", FileType.EPUB),
        ),
    )

    summary = use_case.run()

    assert (summary.indexed, summary.skipped, summary.failed) == (1, 2, 1)
    assert summary.failures[0].path == "bad.txt"
    assert summary.failures[0].reason == "parse_error"
    assert status(manifest, "good.txt") == "indexed"
    assert status(manifest, "bad.txt") == "failed:parse_error"
    assert status(manifest, "scan.pdf") == "skipped:requires_ocr"
    assert status(manifest, "other.epub") == "skipped:unsupported"
    assert run_status(manifest) == "completed"
    metadata = store.read_completion_metadata()
    assert metadata is not None
    assert metadata["embedding_model"] == "deterministic-hash-v1"
    assert len(store.retrieve((0.0,) * 64, {}, 10)) == 2


def test_embedding_batch_retries_then_indexes(tmp_path: Path) -> None:
    embedder = FlakyEmbedder(2)
    use_case, manifest, _ = build(tmp_path, (source("good.txt"),), embedder=embedder)

    assert use_case.run().indexed == 1
    assert embedder.calls == 3
    assert status(manifest, "good.txt") == "indexed"


def test_exhausted_embedding_retry_records_failure(tmp_path: Path) -> None:
    embedder = FlakyEmbedder(3)
    use_case, manifest, store = build(
        tmp_path, (source("good.txt"),), embedder=embedder
    )

    summary = use_case.run()

    assert summary.failed == 1
    assert embedder.calls == 3
    assert status(manifest, "good.txt") == "failed:embedding_error"
    assert store.read_completion_metadata() is not None


def test_fatal_upsert_aborts_without_completion_marker(tmp_path: Path) -> None:
    use_case, manifest, store = build(
        tmp_path, (source("good.txt"),), store=FailingStore()
    )

    with pytest.raises(StoreUnavailable):
        use_case.run()

    assert run_status(manifest) == "aborted"
    assert store.read_completion_metadata() is None


def test_exclusive_file_lock_rejects_second_build(tmp_path: Path) -> None:
    use_case, manifest, store = build(tmp_path, (source("good.txt"),))
    lock_path = tmp_path / "index.lock"
    lock_path.touch()
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(IndexLockHeld):
            use_case.run()
    assert manifest.latest_run() is None
    assert store.read_completion_metadata() is None


def test_empty_corpus_completes_with_zero_counts(tmp_path: Path) -> None:
    use_case, manifest, store = build(tmp_path, ())

    summary = use_case.run()

    assert (summary.indexed, summary.skipped, summary.failed) == (0, 0, 0)
    assert run_status(manifest) == "completed"
    assert store.read_completion_metadata() is not None


def test_enumeration_failure_preserves_existing_collection(tmp_path: Path) -> None:
    use_case, manifest, store = build(tmp_path, ())
    store.write_completion_metadata({"embedding_model": "previous"})
    use_case.enumerator = BrokenSources()

    with pytest.raises(BuildFailed, match="Index build aborted"):
        use_case.run()

    assert run_status(manifest) == "aborted"
    assert store.read_completion_metadata() == {"embedding_model": "previous"}


class FakeLock:
    """Records acquisition and release; optionally refuses to be acquired."""

    def __init__(self, refuse: Exception | None = None) -> None:
        self.refuse = refuse
        self.events: list[str] = []

    def is_held(self) -> bool:
        return False

    @contextmanager
    def hold(self) -> Iterator[None]:
        if self.refuse is not None:
            self.events.append("refused")
            raise self.refuse
        self.events.append("acquired")
        try:
            yield
        finally:
            self.events.append("released")


def test_build_runs_inside_the_injected_lock(tmp_path: Path) -> None:
    lock = FakeLock()
    use_case, manifest, _ = build(tmp_path, (source("good.txt"),), lock=lock)

    use_case.run()

    assert lock.events == ["acquired", "released"]
    assert run_status(manifest) == "completed"


@pytest.mark.parametrize(
    "refusal",
    [IndexLockHeld("busy"), StoreUnavailable("Cannot access the index lock")],
)
def test_lock_acquisition_failures_propagate_before_any_work(
    tmp_path: Path, refusal: Exception
) -> None:
    lock = FakeLock(refuse=refusal)
    use_case, manifest, store = build(tmp_path, (source("good.txt"),), lock=lock)

    with pytest.raises(type(refusal)) as raised:
        use_case.run()

    assert raised.value is refusal
    assert lock.events == ["refused"]
    assert manifest.latest_run() is None
    assert store.read_completion_metadata() is None


class OSErrorManifest(InMemoryManifest):
    def start_run(self, config: Mapping[str, str | int | float]) -> str:
        raise OSError(5, "input/output error")


@pytest.mark.parametrize("use_real_lock", [False, True])
def test_os_error_in_the_build_is_not_reported_as_a_lock_failure(
    tmp_path: Path, use_real_lock: bool
) -> None:
    lock = FileIndexLock(tmp_path / "index.lock") if use_real_lock else FakeLock()
    use_case, _, _ = build(
        tmp_path, (source("good.txt"),), lock=lock, manifest=OSErrorManifest()
    )

    with pytest.raises(OSError, match="input/output error"):
        use_case.run()

    if isinstance(lock, FakeLock):
        assert lock.events == ["acquired", "released"]
    else:
        # The real lock was released: a second build can acquire it again.
        with lock.hold():
            pass
