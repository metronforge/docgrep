"""FindPassages.diagnose ranks like run and returns every candidate with its
first-stage position (ADR-0024)."""

from collections.abc import Mapping, Sequence

import pytest

from docgrep.adapters.models.deterministic import DeterministicEmbedder
from docgrep.domain.errors import ModelUnavailable, ValidationError
from docgrep.domain.models import RankedPassage, RetrievalCandidate
from docgrep.domain.sources import SourceFilter
from docgrep.usecases.find_passages import NO_MATCHING_SOURCES_HINT
from fakes import InMemoryManifest
from search_regression_support import (
    CANDIDATES,
    FailingReranker,
    ScriptedReranker,
    use_case,
)

QUERY = "comet orbit"


def _ids(items: Sequence[RankedPassage | RetrievalCandidate]) -> list[str]:
    return [item.passage.passage_id for item in items]


def test_diagnose_returns_the_public_order_without_the_cut() -> None:
    search = use_case(ScriptedReranker())
    public = search.run(QUERY, top_n=10)
    diagnosis = search.diagnose(QUERY)

    assert search.default_top_n < len(diagnosis.passages)
    assert _ids(diagnosis.passages) == _ids(public.results)
    assert [_ids(item.duplicates) for item in diagnosis.passages] == [
        _ids(item.duplicates) for item in public.results
    ]
    assert diagnosis.top_k == 6
    assert not diagnosis.rerank_skipped


def test_diagnose_gives_every_candidate_its_first_stage_position() -> None:
    diagnosis = use_case(ScriptedReranker()).diagnose(QUERY)

    assert diagnosis.stage1_ranks == {
        item.passage.passage_id: item.stage1_rank for item in CANDIDATES
    }


def test_diagnose_follows_the_duplicate_setting() -> None:
    search = use_case(ScriptedReranker())

    collapsed = search.diagnose(QUERY)
    separate = search.diagnose(QUERY, collapse_duplicates=False)

    assert len(collapsed.passages) == len(CANDIDATES) - 1
    assert len(separate.passages) == len(CANDIDATES)
    assert _ids(separate.passages) == _ids(
        search.run(QUERY, top_n=10, collapse_duplicates=False).results
    )


def test_diagnose_without_a_reranker_keeps_the_first_stage_order() -> None:
    search = use_case(FailingReranker())
    diagnosis = search.diagnose(QUERY, collapse_duplicates=False)

    assert diagnosis.rerank_skipped
    assert diagnosis.rerank_skip_reason == "reranker unavailable"
    assert _ids(diagnosis.passages) == _ids(CANDIDATES)
    assert all(item.reranker_score is None for item in diagnosis.passages)


def test_diagnose_uses_a_given_query_vector() -> None:
    search = use_case(ScriptedReranker())
    (vector,) = DeterministicEmbedder().embed((QUERY,))

    assert search.diagnose(QUERY, dense_vector=vector) == search.diagnose(QUERY)
    with pytest.raises(ModelUnavailable):
        search.diagnose(QUERY, dense_vector=vector[:3])


def test_diagnose_validates_the_query_like_run() -> None:
    search = use_case(ScriptedReranker())
    with pytest.raises(ValidationError, match="query is required"):
        search.diagnose("  ")
    with pytest.raises(ValidationError, match="at most 100"):
        search.diagnose("x" * 101)


class _PositionlessRetriever:
    """Candidates in first-stage order without a ``stage1_rank``."""

    def retrieve(
        self,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float],
        top_k: int,
        source_ids: object = None,
    ) -> tuple[RetrievalCandidate, ...]:
        return tuple(
            RetrievalCandidate(item.passage, item.source, item.dense_score)
            for item in CANDIDATES
        )[:top_k]

    def read_completion_metadata(self) -> Mapping[str, str | int | float] | None:
        return {"embedding_model": "deterministic-hash-v1"}

    def count_passages(self) -> int:
        return len(CANDIDATES)


def test_a_retriever_without_positions_is_ranked_by_list_order() -> None:
    search = use_case(ScriptedReranker())
    search.retriever = _PositionlessRetriever()

    assert search.diagnose(QUERY).stage1_ranks == {
        item.passage.passage_id: position for position, item in enumerate(CANDIDATES)
    }


def test_a_filter_matching_nothing_gives_an_empty_diagnosis() -> None:
    search = use_case(ScriptedReranker())
    search.manifest = InMemoryManifest()

    diagnosis = search.diagnose(QUERY, source_filter=SourceFilter(source_ids=("x",)))

    assert diagnosis.passages == ()
    assert diagnosis.hint == NO_MATCHING_SOURCES_HINT
