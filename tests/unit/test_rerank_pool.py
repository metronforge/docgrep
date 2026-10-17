"""The reranker scores a configurable pool of fused candidates (ADR-0025)."""

from collections.abc import Collection, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError as SettingsValidationError

from docgrep.adapters.models.deterministic import DeterministicEmbedder
from docgrep.adapters.models.sparse import BM25SparseEncoder
from docgrep.api.http import SearchResponse
from docgrep.domain.models import (
    FileType,
    Passage,
    RetrievalCandidate,
    Source,
    make_passage_id,
)
from docgrep.settings import Settings
from docgrep.usecases.find_passages import FindPassages
from search_regression_support import (
    ScriptedReranker,
    StaticRetriever,
    load_fixture,
    mask,
)

TOP_K = 3
# Fused first-stage order; the best answer is fused rank 5 (1-based).
TEXTS = ("decoy one", "decoy two", "decoy three", "decoy four", "the answer")


def _candidates() -> tuple[RetrievalCandidate, ...]:
    found = []
    for rank, text in enumerate(TEXTS):
        source = Source(f"{rank}.txt", f"{rank}.txt", FileType.TXT)
        passage = Passage(
            make_passage_id(source.source_id, 0), source.source_id, text, 0, 2
        )
        found.append(RetrievalCandidate(passage, source, 0.5, stage1_rank=rank))
    return tuple(found)


class PoolRetriever:
    """The fused list is cut at the pool, as Qdrant cuts it; records calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def retrieve(
        self,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float],
        top_k: int,
        source_ids: Collection[str] | None = None,
        **options: Any,
    ) -> tuple[RetrievalCandidate, ...]:
        self.calls.append({"top_k": top_k, "source_ids": source_ids, **options})
        return _candidates()[: options.get("pool") or top_k]

    def read_completion_metadata(self) -> Mapping[str, str | int | float]:
        return {"embedding_model": "deterministic-hash-v1"}

    def count_passages(self) -> int:
        return len(TEXTS)


class AnswerReranker:
    model_name = "answer-first"

    def score(self, query: str, passages: Sequence[str]) -> tuple[float, ...]:
        return tuple(3.0 if text == "the answer" else 0.0 for text in passages)


def _search(retriever: Any, **extra: Any) -> FindPassages:
    return FindPassages(
        embedder=DeterministicEmbedder(),
        sparse_encoder=BM25SparseEncoder(),
        retriever=retriever,
        reranker=AnswerReranker(),
        top_k=TOP_K,
        default_top_n=1,
        top_n_max=5,
        max_query_len=100,
        **extra,
    )


def test_by_default_a_candidate_beyond_top_k_is_not_reranked() -> None:
    retriever = PoolRetriever()

    result = _search(retriever).run("answer")

    assert result.results[0].passage.text != "the answer"
    assert result.total_hits == TOP_K
    # The retriever is called exactly as before the pool existed.
    assert retriever.calls == [{"top_k": TOP_K, "source_ids": None}]
    assert "rerank_pool" not in dict(result.applied_config)


def test_a_candidate_between_top_k_and_the_pool_is_reranked_and_returned() -> None:
    retriever = PoolRetriever()

    result = _search(retriever, rerank_pool=5).run("answer")

    assert result.results[0].passage.text == "the answer"
    assert result.total_hits == 5
    assert retriever.calls == [{"top_k": TOP_K, "source_ids": None, "pool": 5}]
    assert dict(result.applied_config)["rerank_pool"] == 5
    diagnosis = _search(PoolRetriever(), rerank_pool=5).diagnose("answer")
    assert diagnosis.stage1_ranks[result.results[0].passage.passage_id] == 4


def test_a_pool_equal_to_top_k_is_the_default() -> None:
    retriever = PoolRetriever()

    search = _search(retriever, rerank_pool=TOP_K)
    result = search.run("answer")

    assert search.rerank_pool == TOP_K
    assert retriever.calls == [{"top_k": TOP_K, "source_ids": None}]
    assert "rerank_pool" not in dict(result.applied_config)


@pytest.mark.parametrize("pool", [0, -1, 2 * TOP_K + 1])
def test_the_pool_is_positive_and_at_most_twice_top_k(pool: int) -> None:
    with pytest.raises(ValueError, match="rerank_pool"):
        _search(PoolRetriever(), rerank_pool=pool)


def test_the_largest_pool_is_twice_top_k() -> None:
    assert _search(PoolRetriever(), rerank_pool=2 * TOP_K).rerank_pool == 2 * TOP_K


def test_the_setting_defaults_to_150_within_twice_top_k() -> None:
    # ADR-0026: 150 unless the union of the two channels is smaller.
    assert Settings(_env_file=None).effective_rerank_pool == 150
    settings = Settings(_env_file=None, top_k=40, top_n=10)

    assert settings.rerank_pool is None
    assert settings.effective_rerank_pool == 80


def test_the_setting_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DOCGREP_RERANK_POOL", "150")

    settings = Settings(_env_file=None)

    assert settings.rerank_pool == settings.effective_rerank_pool == 150


@pytest.mark.parametrize("pool", [0, 201])
def test_the_setting_is_validated(pool: int) -> None:
    with pytest.raises(SettingsValidationError, match="rerank_pool"):
        Settings(_env_file=None, top_k=100, rerank_pool=pool)


def test_a_default_pool_keeps_the_recorded_body_of_the_use_case() -> None:
    """The use-case bodies of the 1.20.0 fixture, with the pool set to top_k."""
    runs: list[tuple[int, dict[str, Any]]] = [
        (6, {}),
        (6, {"top_n": 10}),
        (6, {"top_n": 10, "collapse_duplicates": False}),
        (3, {"top_n": 10}),
    ]
    bodies = []
    for top_k, options in runs:
        search = FindPassages(
            embedder=DeterministicEmbedder(),
            sparse_encoder=BM25SparseEncoder(),
            retriever=StaticRetriever(),
            reranker=ScriptedReranker(),
            top_k=top_k,
            rerank_pool=top_k,
            default_top_n=3,
            top_n_max=10,
            max_query_len=100,
            now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
        )
        body = SearchResponse.from_domain(search.run("comet orbit", **options))
        bodies.append(mask(body.model_dump_json().encode("utf-8")).decode("utf-8"))

    assert bodies == load_fixture()["use_case"][:4]
