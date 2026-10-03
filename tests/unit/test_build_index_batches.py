"""How the build groups, retries and fails batched passage writes."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from docgrep.adapters.chunking import TokenChunker, WhitespaceTokenizer
from docgrep.adapters.filelock import FileIndexLock
from docgrep.adapters.models.deterministic import DeterministicEmbedder
from docgrep.domain.errors import StoreUnavailable
from docgrep.domain.models import (
    FileType,
    ParsedDocument,
    ParsedSection,
    Passage,
    Source,
)
from docgrep.usecases.build_index import BuildIndex
from fakes import InMemoryManifest, InMemoryPassageStore

Item = tuple[Passage, Source, Sequence[float], Mapping[int, float]]


class Sources:
    def enumerate(self) -> tuple[Source, ...]:
        return (
            Source("a.txt", "a.txt", FileType.TXT),
            Source("b.txt", "b.txt", FileType.TXT),
        )


class Parser:
    def parse(self, source: Source) -> ParsedDocument:
        words = " ".join(f"{source.path[0]}{i}" for i in range(5))
        return ParsedDocument(source, (ParsedSection(words),))


class Sparse:
    def encode(self, text: str) -> dict[int, float]:
        return {len(text): 1.0}


class BatchStore(InMemoryPassageStore):
    def __init__(self, failures: int = 0) -> None:
        super().__init__()
        self.failures = failures
        self.sizes: list[int] = []
        self.attempts = 0

    def upsert_batch(self, items: Sequence[Item]) -> None:
        self.attempts += 1
        if self.attempts <= self.failures:
            raise StoreUnavailable("temporary write failure")
        self.sizes.append(len(items))
        super().upsert_batch(items)


def _build(
    tmp_path: Path, store: BatchStore, size: int, sleeps: list[float]
) -> tuple[BuildIndex, InMemoryManifest]:
    manifest = InMemoryManifest()
    return (
        BuildIndex(
            lock=FileIndexLock(tmp_path / "index.lock"),
            enumerator=Sources(),
            parsers={FileType.TXT: Parser()},
            chunker=TokenChunker(WhitespaceTokenizer(), 1, 0),
            embedder=DeterministicEmbedder(),
            sparse_encoder=Sparse(),
            passage_writer=store,
            manifest=manifest,
            config={"chunk_size": 1, "chunk_overlap": 0},
            docgrep_version="0.0.0",
            upsert_batch_size=size,
            sleep=sleeps.append,
        ),
        manifest,
    )


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (1, [1] * 10),
        (2, [2, 2, 1, 2, 2, 1]),
        (4, [4, 1, 4, 1]),
        (5, [5, 5]),
        (256, [5, 5]),
    ],
)
def test_each_sources_passages_are_written_in_batches_of_the_configured_size(
    tmp_path: Path, size: int, expected: list[int]
) -> None:
    store = BatchStore()
    use_case, _ = _build(tmp_path, store, size, [])

    use_case.run()

    assert store.sizes == expected
    assert store.count_passages() == 10


def test_a_failing_batch_is_retried_with_the_existing_backoff(tmp_path: Path) -> None:
    store = BatchStore(failures=2)
    sleeps: list[float] = []
    use_case, manifest = _build(tmp_path, store, 256, sleeps)

    summary = use_case.run()

    assert (summary.indexed, summary.failed) == (2, 0)
    assert sleeps == [1.0, 2.0]
    assert store.count_passages() == 10


def test_an_exhausted_batch_raises_the_usual_error_and_aborts(tmp_path: Path) -> None:
    store = BatchStore(failures=99)
    sleeps: list[float] = []
    use_case, manifest = _build(tmp_path, store, 256, sleeps)

    with pytest.raises(StoreUnavailable, match="Passage upsert failed") as raised:
        use_case.run()

    assert raised.value.details == {
        "path": "a.txt",
        "reason": "temporary write failure",
    }
    assert store.attempts == 3 and sleeps == [1.0, 2.0]
    run = manifest.latest_run()
    assert run is not None and run.status == "aborted"
    assert store.read_completion_metadata() is None


def test_a_non_positive_batch_size_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="upsert_batch_size"):
        _build(tmp_path, BatchStore(), 0, [])
