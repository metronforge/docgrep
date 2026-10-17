"""The dense search effort and the rerank pool default (ADR-0026)."""

from collections.abc import Collection, Mapping, Sequence
from typing import Any

import pytest
from pydantic import ValidationError as SettingsValidationError

from docgrep.adapters.models.deterministic import DeterministicEmbedder, LexicalReranker
from docgrep.adapters.models.sparse import BM25SparseEncoder
from docgrep.domain.models import (
    FileType,
    Passage,
    RetrievalCandidate,
    Source,
    make_passage_id,
)
from docgrep.domain.search_effort import (
    DEFAULT_DENSE_HNSW_EF,
    DEFAULT_RERANK_POOL,
    MAX_DENSE_HNSW_EF,
    default_rerank_pool,
    parse_dense_effort,
)
from docgrep.settings import Settings
from docgrep.usecases.find_passages import FindPassages


class OneRetriever:
    """One fixed candidate; the search effort is not its concern here."""

    def retrieve(
        self,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float],
        top_k: int,
        source_ids: Collection[str] | None = None,
        **options: Any,
    ) -> tuple[RetrievalCandidate, ...]:
        source = Source("a.txt", "a.txt", FileType.TXT)
        passage = Passage(make_passage_id("a.txt", 0), "a.txt", "an answer", 0, 2)
        return (RetrievalCandidate(passage, source, 0.5, stage1_rank=0),)

    def read_completion_metadata(self) -> Mapping[str, str | int | float]:
        return {"embedding_model": "deterministic-hash-v1"}

    def count_passages(self) -> int:
        return 1


def _search(**extra: Any) -> FindPassages:
    return FindPassages(
        embedder=DeterministicEmbedder(),
        sparse_encoder=BM25SparseEncoder(),
        retriever=OneRetriever(),
        reranker=LexicalReranker(),
        top_k=3,
        default_top_n=1,
        top_n_max=5,
        max_query_len=100,
        **extra,
    )


@pytest.mark.parametrize(
    ("top_k", "expected"), [(100, 150), (75, 150), (74, 148), (10, 20), (500, 150)]
)
def test_the_default_pool_is_150_within_the_union(top_k: int, expected: int) -> None:
    assert default_rerank_pool(top_k) == expected
    assert DEFAULT_RERANK_POOL == 150


@pytest.mark.parametrize(
    ("text", "expected"),
    [("exact", "exact"), (" EXACT ", "exact"), ("512", 512), ("1", 1)],
)
def test_the_effort_is_parsed(text: str, expected: object) -> None:
    assert parse_dense_effort(text) == expected


@pytest.mark.parametrize("text", ["0", "16385", "fast", "-1", "1.5", ""])
def test_an_invalid_effort_is_refused(text: str) -> None:
    with pytest.raises(ValueError, match="exact"):
        parse_dense_effort(text)


def test_the_setting_defaults_to_1024() -> None:
    assert Settings(_env_file=None).dense_hnsw_ef == DEFAULT_DENSE_HNSW_EF == 1024


@pytest.mark.parametrize(("value", "expected"), [("exact", "exact"), ("256", 256)])
def test_the_setting_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: object
) -> None:
    monkeypatch.setenv("DOCGREP_DENSE_HNSW_EF", value)

    assert Settings(_env_file=None).dense_hnsw_ef == expected


@pytest.mark.parametrize("value", ["0", str(MAX_DENSE_HNSW_EF + 1), "fast"])
def test_the_setting_is_validated(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("DOCGREP_DENSE_HNSW_EF", value)

    with pytest.raises(SettingsValidationError, match="dense_hnsw_ef"):
        Settings(_env_file=None)


def test_search_reports_the_effort_it_was_given() -> None:
    result = _search(dense_hnsw_ef=1024).run("answer")
    exact = _search(dense_hnsw_ef="exact").run("answer")
    unset = _search().run("answer")

    assert dict(result.applied_config)["dense_hnsw_ef"] == 1024
    assert dict(exact.applied_config)["dense_hnsw_ef"] == "exact"
    assert "dense_hnsw_ef" not in dict(unset.applied_config)
    # The effort is the retriever's business; the use case only reports it.
    assert result.results == unset.results
