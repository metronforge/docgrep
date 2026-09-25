"""FindPassages behavior through domain ports."""

import logging
from collections.abc import Mapping, Sequence

import pytest

from docgrep.adapters.models.deterministic import DeterministicEmbedder, LexicalReranker
from docgrep.adapters.models.sparse import BM25SparseEncoder
from docgrep.domain.errors import (
    ConfigDrift,
    IndexIncomplete,
    ModelUnavailable,
    StoreUnavailable,
    ValidationError,
)
from docgrep.domain.models import FileType, Passage, RetrievalCandidate, Source
from docgrep.usecases.find_passages import FindPassages


class StaticRetriever:
    def __init__(
        self,
        candidates: tuple[RetrievalCandidate, ...] = (),
        metadata: Mapping[str, str | int | float] | None = None,
        count: int | None = None,
    ) -> None:
        self.candidates = candidates
        self.metadata = metadata
        self.count = len(candidates) if count is None else count

    def retrieve(
        self,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float],
        top_k: int,
    ) -> tuple[RetrievalCandidate, ...]:
        return self.candidates[:top_k]

    def read_completion_metadata(self) -> Mapping[str, str | int | float] | None:
        return self.metadata

    def count_passages(self) -> int:
        return self.count


class EqualReranker:
    model_name = "equal-reranker"

    def score(self, query: str, passages: Sequence[str]) -> tuple[float, ...]:
        return (0.0,) * len(passages)


class FailingReranker:
    model_name = "broken-reranker"

    def score(self, query: str, passages: Sequence[str]) -> tuple[float, ...]:
        raise ModelUnavailable(
            "reranker unavailable", details={"model": self.model_name}
        )


class FailingEmbedder(DeterministicEmbedder):
    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        raise ModelUnavailable(
            "embedder unavailable", details={"model": self.model_name}
        )


class FailingRetriever(StaticRetriever):
    def read_completion_metadata(self) -> Mapping[str, str | int | float] | None:
        raise StoreUnavailable("store unavailable")


def _candidate(
    passage_id: str,
    text: str,
    dense_score: float = 0.5,
    *,
    path: str = "notes.txt",
) -> RetrievalCandidate:
    source = Source("source", path, FileType.TXT)
    passage = Passage(passage_id, "source", text, 0, len(text.split()))
    return RetrievalCandidate(passage, source, dense_score)


def _finder(
    retriever: StaticRetriever,
    *,
    embedder: DeterministicEmbedder | None = None,
    reranker: LexicalReranker | EqualReranker | FailingReranker | None = None,
) -> FindPassages:
    return FindPassages(
        embedder=embedder or DeterministicEmbedder(),
        sparse_encoder=BM25SparseEncoder(),
        retriever=retriever,
        reranker=reranker or LexicalReranker(),
        top_k=10,
        default_top_n=2,
        top_n_max=3,
        max_query_len=12,
    )


def _metadata() -> dict[str, str | int | float]:
    return {"embedding_model": "deterministic-hash-v1", "embedding_dimension": 64}


@pytest.mark.parametrize(
    ("query", "top_n", "detail"),
    [
        ("  ", None, "query"),
        ("more than twelve", None, "12"),
        ("valid", 0, "1"),
        ("valid", -1, "1"),
        ("valid", 4, "3"),
    ],
)
def test_invalid_requests_raise_validation_error(
    query: str, top_n: int | None, detail: str
) -> None:
    finder = _finder(StaticRetriever(metadata=_metadata()))
    with pytest.raises(ValidationError, match=detail):
        finder.run(query, top_n=top_n)


def test_missing_completion_marker_is_structured_error() -> None:
    with pytest.raises(IndexIncomplete) as caught:
        _finder(StaticRetriever()).run("valid")
    assert caught.value.category == "index_incomplete"


def test_model_drift_names_both_models() -> None:
    with pytest.raises(ConfigDrift) as caught:
        _finder(
            StaticRetriever(
                metadata={"embedding_model": "old", "embedding_dimension": 64}
            )
        ).run("valid")
    assert caught.value.details["indexed_model"] == "old"
    assert caught.value.details["configured_model"] == "deterministic-hash-v1"


def test_empty_completed_index_returns_hint_without_embedding() -> None:
    result = _finder(
        StaticRetriever(metadata=_metadata(), count=0), embedder=FailingEmbedder()
    ).run("valid")
    assert result.results == ()
    assert result.total_hits == 0
    assert result.hint is not None and "empty" in result.hint


def test_result_ceiling_may_exceed_recall_pool() -> None:
    retriever = StaticRetriever((_candidate("one", "valid match"),), _metadata())
    finder = FindPassages(
        embedder=DeterministicEmbedder(),
        sparse_encoder=BM25SparseEncoder(),
        retriever=retriever,
        reranker=LexicalReranker(),
        top_k=1,
        default_top_n=1,
        top_n_max=50,
        max_query_len=12,
    )
    assert len(finder.run("valid", top_n=50).results) == 1


def test_embedding_failure_is_structured() -> None:
    retriever = StaticRetriever((_candidate("a", "valid text"),), _metadata())
    with pytest.raises(ModelUnavailable) as caught:
        _finder(retriever, embedder=FailingEmbedder()).run("valid")
    assert caught.value.details["model"] == "deterministic-hash-v1"


def test_store_failure_is_structured() -> None:
    with pytest.raises(StoreUnavailable):
        _finder(FailingRetriever()).run("valid")


def test_no_hybrid_candidates_returns_empty_without_reranking() -> None:
    result = _finder(
        StaticRetriever(metadata=_metadata(), count=1), reranker=FailingReranker()
    ).run("valid")
    assert result.results == ()
    assert result.total_hits == 0
    assert not result.rerank_skipped


def test_rerank_sorts_with_dense_and_id_tiebreak_and_preserves_text() -> None:
    candidates = (
        _candidate("z", "original z", 0.5),
        _candidate("b", "original b", 0.9),
        _candidate("a", "original a", 0.9),
    )
    result = _finder(
        StaticRetriever(candidates, _metadata()), reranker=EqualReranker()
    ).run("valid", top_n=3)
    assert [hit.passage.passage_id for hit in result.results] == ["a", "b", "z"]
    assert [hit.passage.text for hit in result.results] == [
        "original a",
        "original b",
        "original z",
    ]
    assert result.total_hits == 3
    assert all(0 <= hit.score <= 1 for hit in result.results)
    assert dict(result.applied_config)["reranker_model"] == "equal-reranker"
    assert result.took_ms >= 0


def test_rerank_can_move_lexical_match_above_dense_favorite() -> None:
    candidates = (
        _candidate("dense", "unrelated content", 0.9),
        _candidate("lexical", "valid match", 0.1),
    )
    result = _finder(StaticRetriever(candidates, _metadata())).run("valid")
    assert [hit.passage.passage_id for hit in result.results] == ["lexical", "dense"]


def test_reranker_failure_returns_dense_order_with_reason() -> None:
    candidates = (_candidate("low", "low", 0.1), _candidate("high", "high", 0.8))
    result = _finder(
        StaticRetriever(candidates, _metadata()), reranker=FailingReranker()
    ).run("valid", top_n=1)
    assert [hit.passage.passage_id for hit in result.results] == ["high"]
    assert result.total_hits == 2
    assert result.rerank_skipped
    assert result.rerank_skip_reason is not None


def test_corrupt_candidate_is_excluded_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    candidates = (
        _candidate("bad", "bad text", path="bad\x00path"),
        _candidate("good", "verbatim text"),
    )
    with caplog.at_level(logging.WARNING):
        result = _finder(StaticRetriever(candidates, _metadata())).run("valid")
    assert [hit.passage.passage_id for hit in result.results] == ["good"]
    assert result.total_hits == 1
    assert "bad" in caplog.text
