"""Build, update and search encode with the weighting the index records
(ADR-0025), through in-memory fakes."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from hashlib import sha256
from typing import Any

import pytest

from docgrep.adapters.chunking import TokenChunker, WhitespaceTokenizer
from docgrep.adapters.models.deterministic import DeterministicEmbedder, LexicalReranker
from docgrep.adapters.models.sparse import (
    BM25DocumentEncoder,
    SparseEncoderSet,
    TermFrequencyEncoder,
    UniqueTermEncoder,
)
from docgrep.domain.errors import ConfigDrift, ResumeAvailable
from docgrep.domain.models import (
    FileType,
    ParsedDocument,
    ParsedSection,
    ScannedFile,
    Source,
)
from docgrep.domain.sparse import PROVISIONAL_AVGDL
from docgrep.usecases.build_index import BuildIndex
from docgrep.usecases.find_passages import FindPassages
from docgrep.usecases.update_index import UpdateIndex
from fakes import InMemoryManifest, InMemoryPassageStore

CONFIG = {"chunk_size": 4, "chunk_overlap": 0, "tokenizer": "whitespace"}


class Corpus:
    def __init__(self, files: Mapping[str, str]) -> None:
        self.files = dict(files)

    def _source(self, path: str) -> Source:
        text = self.files[path]
        return Source(
            path,
            path,
            FileType.TXT,
            content_hash=sha256(text.encode()).hexdigest(),
            mtime=1.0,
            size=len(text),
        )

    def enumerate(self) -> tuple[Source, ...]:
        return tuple(self._source(path) for path in sorted(self.files))

    def scan(self) -> tuple[ScannedFile, ...]:
        return tuple(
            ScannedFile(path, len(text), ".txt", 1.0)
            for path, text in sorted(self.files.items())
        )

    def describe(self, path: str) -> Source:
        return self._source(path)


class Parser:
    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus

    def parse(self, source: Source) -> ParsedDocument:
        return ParsedDocument(source, (ParsedSection(self.corpus.files[source.path]),))


class Lock:
    @contextmanager
    def hold(self, operation: str = "build") -> Iterator[None]:
        yield

    def is_held(self) -> bool:
        return False


class Harness:
    def __init__(self, files: Mapping[str, str]) -> None:
        self.corpus = Corpus(files)
        self.store = InMemoryPassageStore()
        self.manifest = InMemoryManifest()

    def common(self) -> dict[str, Any]:
        return {
            "lock": Lock(),
            "parsers": {FileType.TXT: Parser(self.corpus)},
            "chunker": TokenChunker(WhitespaceTokenizer(), 4, 0),
            "embedder": DeterministicEmbedder(),
            "sparse_encoder": TermFrequencyEncoder(),
            "manifest": self.manifest,
            "config": CONFIG,
            "sleep": lambda seconds: None,
        }

    def build(self, *, bm25: bool = True, **extra: Any) -> BuildIndex:
        if bm25:
            extra = {
                "sparse_encoders": SparseEncoderSet(),
                "sparse_store": self.store,
                **extra,
            }
        return BuildIndex(
            enumerator=self.corpus,
            passage_writer=self.store,
            docgrep_version="test",
            **self.common(),
            **extra,
        )

    def update(self) -> UpdateIndex:
        return UpdateIndex(
            corpus=self.corpus,
            index=self.store,
            sparse_encoders=SparseEncoderSet(),
            **self.common(),
        )

    def passage_ids(self) -> list[str]:
        return [
            passage_id for passage_id, _ in self.store.passage_text_page(None, 1000)[0]
        ]

    def texts(self) -> dict[str, str]:
        return dict(self.store.passage_text_page(None, 1000)[0])


FILES = {
    "a.txt": "one two three four five six seven eight nine",
    "b.txt": "alpha alpha alpha beta",
}


def test_a_fresh_build_writes_bm25_with_the_average_length_of_its_passages() -> None:
    harness = Harness(FILES)

    assert harness.build().run().indexed == 2

    metadata = harness.store.read_completion_metadata()
    assert metadata is not None
    # Passages of 4, 4, 1 and 4 tokens.
    assert metadata["sparse_encoder"] == "bm25-v2"
    assert (metadata["bm25_k1"], metadata["bm25_b"]) == (1.2, 0.75)
    assert metadata["bm25_avgdl"] == pytest.approx(13 / 4)
    expected = BM25DocumentEncoder(1.2, 0.75, 13 / 4)
    for passage_id, text in harness.texts().items():
        assert harness.store.sparse_vector(passage_id) == pytest.approx(
            expected.encode(text)
        )


def test_the_bm25_parameters_of_a_build_are_recorded() -> None:
    harness = Harness(FILES)

    harness.build(bm25_k1=0.9, bm25_b=0.4).run()

    metadata = harness.store.read_completion_metadata()
    assert metadata is not None
    assert (metadata["bm25_k1"], metadata["bm25_b"]) == (0.9, 0.4)
    assert metadata["bm25_avgdl"] != PROVISIONAL_AVGDL


def test_a_build_without_the_encoder_set_writes_tf_v1() -> None:
    harness = Harness(FILES)

    harness.build(bm25=False).run()

    metadata = harness.store.read_completion_metadata()
    assert metadata is not None and "sparse_encoder" not in metadata
    for passage_id, text in harness.texts().items():
        assert harness.store.sparse_vector(passage_id) == (
            TermFrequencyEncoder().encode(text)
        )


def test_an_update_of_a_tf_v1_index_writes_term_counts() -> None:
    harness = Harness(FILES)
    harness.build(bm25=False).run()
    harness.corpus.files["c.txt"] = "gamma gamma delta"

    assert harness.update().run().indexed == 1

    metadata = harness.store.read_completion_metadata()
    assert metadata is not None and "sparse_encoder" not in metadata
    for passage_id, text in harness.texts().items():
        assert harness.store.sparse_vector(passage_id) == (
            TermFrequencyEncoder().encode(text)
        )


def test_an_update_of_a_bm25_index_uses_the_recorded_average_length() -> None:
    harness = Harness(FILES)
    harness.build().run()
    before = harness.store.read_completion_metadata()
    assert before is not None
    harness.corpus.files["c.txt"] = "gamma gamma delta"

    assert harness.update().run().indexed == 1

    after = harness.store.read_completion_metadata()
    assert after is not None
    # The recorded weighting is kept, though the new passage changes the
    # true average length.
    assert after["bm25_avgdl"] == before["bm25_avgdl"] == pytest.approx(13 / 4)
    expected = BM25DocumentEncoder(1.2, 0.75, 13 / 4)
    for passage_id, text in harness.texts().items():
        assert harness.store.sparse_vector(passage_id) == pytest.approx(
            expected.encode(text)
        )


def test_an_update_refuses_an_unknown_recorded_weighting() -> None:
    harness = Harness(FILES)
    harness.build().run()
    harness.store.merge_completion_metadata({"sparse_encoder": "bm99"})
    harness.corpus.files["c.txt"] = "gamma"

    with pytest.raises(ConfigDrift):
        harness.update().run()


def _search(harness: Harness) -> FindPassages:
    return FindPassages(
        embedder=DeterministicEmbedder(),
        sparse_encoder=TermFrequencyEncoder(),
        sparse_encoders=SparseEncoderSet(),
        retriever=harness.store,
        reranker=LexicalReranker(),
        top_k=10,
        default_top_n=3,
        top_n_max=10,
        max_query_len=100,
    )


@pytest.mark.parametrize(
    ("bm25", "expected"),
    [
        (False, TermFrequencyEncoder().encode("alpha alpha")),
        (True, UniqueTermEncoder().encode("alpha")),
    ],
)
def test_search_encodes_the_query_for_the_recorded_weighting(
    bm25: bool, expected: dict[int, float]
) -> None:
    harness = Harness(FILES)
    harness.build(bm25=bm25).run()
    search = _search(harness)
    metadata = search.check_index()

    assert search.query_encoder(metadata).encode("alpha alpha") == expected
    assert search.run("alpha alpha").results


def test_search_refuses_an_unknown_recorded_weighting() -> None:
    harness = Harness(FILES)
    harness.build().run()
    harness.store.merge_completion_metadata({"sparse_encoder": "bm99"})

    with pytest.raises(ConfigDrift):
        _search(harness).run("alpha")


def _unfinished_reencode(harness: Harness) -> str:
    run_id = harness.manifest.start_run({}, kind="reencode_sparse")
    harness.manifest.finish_run(run_id, "cancelled", {})
    return run_id


def test_an_update_waits_for_an_unfinished_re_encode() -> None:
    harness = Harness(FILES)
    harness.build().run()
    run_id = _unfinished_reencode(harness)
    harness.corpus.files["c.txt"] = "gamma"

    with pytest.raises(ResumeAvailable) as refused:
        harness.update().run()

    assert refused.value.details["run_id"] == run_id
    assert "reencode-sparse --yes" in str(refused.value)


def test_a_completed_build_replaces_the_index_of_an_unfinished_re_encode() -> None:
    harness = Harness(FILES)
    harness.build().run()
    _unfinished_reencode(harness)
    harness.build().run()
    harness.corpus.files["c.txt"] = "gamma"

    assert harness.update().run().indexed == 1
