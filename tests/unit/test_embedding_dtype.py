"""Embedding precision: resolution, float32 output, metadata and compatibility."""

import logging
import math
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError as SettingsValidationError

from docgrep.adapters.chunking import TokenChunker, WhitespaceTokenizer
from docgrep.adapters.filelock import FileIndexLock
from docgrep.adapters.models.deterministic import DeterministicEmbedder, LexicalReranker
from docgrep.adapters.models.real import BgeEmbedder
from docgrep.adapters.models.sparse import BM25SparseEncoder
from docgrep.domain.models import FileType, ParsedDocument, ParsedSection, Source
from docgrep.settings import Settings
from docgrep.usecases.build_index import BuildIndex
from docgrep.usecases.find_passages import FindPassages
from fakes import InMemoryManifest, InMemoryPassageStore


class _Tokenizer:
    def __call__(self, text: str, **_: object) -> dict[str, list[tuple[int, int]]]:
        return {"offset_mapping": []}


class _Model:
    tokenizer = _Tokenizer()

    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        self.vectors = vectors

    def get_sentence_embedding_dimension(self) -> int:
        return 2

    def encode(self, texts: Sequence[str], **_: object) -> list[list[float]]:
        return self.vectors or [[0.6, 0.8] for _ in texts]


class _Factory:
    def __init__(self, model: _Model | None = None) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.model = model or _Model()

    def __call__(self, name: str, device: str, dtype: str) -> _Model:
        self.calls.append((name, device, dtype))
        return self.model


MATRIX = [
    ("auto", True, "float16", False),
    ("auto", False, "float32", False),
    ("float16", True, "float16", False),
    ("float16", False, "float32", True),
    ("float32", True, "float32", False),
    ("float32", False, "float32", False),
]


@pytest.mark.parametrize(("setting", "cuda", "expected", "warns"), MATRIX)
def test_dtype_resolution_through_the_injected_probe(
    setting: str,
    cuda: bool,
    expected: str,
    warns: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    factory = _Factory()
    embedder = BgeEmbedder(
        "e", "auto", cuda_available=lambda: cuda, dtype=setting, factory=factory
    )

    assert embedder.dtype == expected
    with caplog.at_level(logging.INFO, logger="docgrep"):
        embedder.embed(["a"])
        embedder.embed(["b"])

    assert factory.calls == [("e", "cuda" if cuda else "cpu", expected)]
    fallbacks = [
        r for r in caplog.records if r.getMessage() == "embedding_dtype_fallback"
    ]
    assert bool(fallbacks) is warns
    if warns:
        assert fallbacks[0].levelno == logging.WARNING
        assert fallbacks[0].__dict__["requested"] == "float16"
    [loaded] = [r for r in caplog.records if r.getMessage() == "model_loaded"]
    assert loaded.__dict__["dtype"] == expected


def test_an_explicit_cpu_device_never_gets_float16() -> None:
    embedder = BgeEmbedder(
        "e", "cpu", cuda_available=lambda: True, dtype="float16", factory=_Factory()
    )

    assert embedder.dtype == "float32"


def test_half_precision_output_is_returned_as_unit_length_python_floats() -> None:
    sloppy = [[0.6004, 0.7996], [3.0, 4.0]]
    half = BgeEmbedder(
        "e",
        "cuda",
        dtype="float16",
        factory=_Factory(_Model(sloppy)),
        cuda_available=lambda: True,
    )
    full = BgeEmbedder(
        "e",
        "cpu",
        dtype="float32",
        factory=_Factory(_Model(sloppy)),
        cuda_available=lambda: False,
    )

    vectors = half.embed(["a", "b"])

    assert all(type(value) is float for vector in vectors for value in vector)
    assert all(
        math.isclose(math.hypot(*vector), 1.0, abs_tol=1e-12) for vector in vectors
    )
    assert vectors[1] == (0.6, 0.8)
    # The float32 path hands back exactly what the model produced.
    assert full.embed(["a", "b"]) == ((0.6004, 0.7996), (3.0, 4.0))


def test_injected_loaders_keep_working_and_do_not_claim_a_dtype(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="docgrep"):
        _ = BgeEmbedder("e", "cpu", lambda *_: _Model()).dimension

    [loaded] = [r for r in caplog.records if r.getMessage() == "model_loaded"]
    assert "dtype" not in loaded.__dict__


def test_embedding_dtype_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DOCGREP_EMBEDDING_DTYPE", raising=False)
    assert Settings().embedding_dtype == "auto"
    monkeypatch.setenv("DOCGREP_EMBEDDING_DTYPE", "float16")
    assert Settings().embedding_dtype == "float16"
    monkeypatch.setenv("DOCGREP_EMBEDDING_DTYPE", "bfloat16")
    with pytest.raises(SettingsValidationError):
        Settings()


class _Parser:
    def parse(self, source: Source) -> ParsedDocument:
        return ParsedDocument(source, (ParsedSection("alpha beta gamma delta"),))


class _Sources:
    def enumerate(self) -> tuple[Source, ...]:
        return (Source("a.txt", "a.txt", FileType.TXT),)


class HalfEmbedder(DeterministicEmbedder):
    """The same vectors, but reported as computed in half precision."""

    dtype = "float16"


def _build(
    tmp_path: Path, embedder: DeterministicEmbedder, store: InMemoryPassageStore
) -> None:
    BuildIndex(
        lock=FileIndexLock(tmp_path / "index.lock"),
        enumerator=_Sources(),
        parsers={FileType.TXT: _Parser()},
        chunker=TokenChunker(WhitespaceTokenizer(), 4, 0),
        embedder=embedder,
        sparse_encoder=BM25SparseEncoder(),
        passage_writer=store,
        manifest=InMemoryManifest(),
        config={"chunk_size": 4, "chunk_overlap": 0},
        docgrep_version="0.0.0",
    ).run()


def _finder(
    store: InMemoryPassageStore, embedder: DeterministicEmbedder
) -> FindPassages:
    return FindPassages(
        embedder=embedder,
        sparse_encoder=BM25SparseEncoder(),
        retriever=store,
        reranker=LexicalReranker(),
        top_k=10,
        default_top_n=5,
        top_n_max=10,
        max_query_len=100,
    )


@pytest.mark.parametrize(
    ("embedder", "expected"),
    [(DeterministicEmbedder(), "float32"), (HalfEmbedder(), "float16")],
)
def test_the_effective_dtype_is_recorded_in_the_completion_metadata(
    tmp_path: Path, embedder: DeterministicEmbedder, expected: str
) -> None:
    store = InMemoryPassageStore()

    _build(tmp_path, embedder, store)

    metadata = store.read_completion_metadata()
    assert metadata is not None and metadata["embedding_dtype"] == expected
    assert metadata["embedding_model"] == "deterministic-hash-v1"


@pytest.mark.parametrize(
    ("built_with", "searched_with"),
    [
        (HalfEmbedder(), DeterministicEmbedder()),
        (DeterministicEmbedder(), HalfEmbedder()),
    ],
)
def test_an_index_stays_searchable_across_dtypes(
    tmp_path: Path,
    built_with: DeterministicEmbedder,
    searched_with: DeterministicEmbedder,
) -> None:
    store = InMemoryPassageStore()
    _build(tmp_path, built_with, store)

    result = _finder(store, searched_with).run("alpha beta")

    assert [hit.passage.text for hit in result.results] == ["alpha beta gamma delta"]


def test_metadata_without_a_dtype_key_keeps_working(tmp_path: Path) -> None:
    store = InMemoryPassageStore()
    _build(tmp_path, DeterministicEmbedder(), store)
    metadata = dict(store.read_completion_metadata() or {})
    del metadata["embedding_dtype"]
    store.write_completion_metadata(metadata)

    result = _finder(store, HalfEmbedder()).run("alpha")

    assert len(result.results) == 1
