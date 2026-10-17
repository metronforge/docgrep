"""The evaluation over a Qdrant index whose passages rank differently in
each channel, in local mode and against a real server (set
DOCGREP_TEST_QDRANT_URL) (ADR-0024)."""

import os
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from docgrep.adapters.qdrant import QdrantPassageStore
from docgrep.domain.evaluation import EvalCase, EvalReport, ListRank
from docgrep.usecases.evaluate_retrieval import EvaluateRetrieval
from docgrep.usecases.find_passages import FindPassages
from eval_support import ALIAS, QUERY_DENSE, QUERY_SPARSE, build, wipe

URL = os.environ.get("DOCGREP_TEST_QDRANT_URL")
_SERVER = pytest.param(
    "server",
    marks=[
        pytest.mark.qdrant_server,
        pytest.mark.skipif(not URL, reason="DOCGREP_TEST_QDRANT_URL is not set"),
    ],
)


class FixedEmbedder:
    """Every query embeds to the layout's query vector."""

    model_name = "m"
    dimension = 2
    dtype = "float32"

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(QUERY_DENSE for _ in texts)


class FixedSparse:
    def encode(self, text: str) -> Mapping[int, float]:
        return dict(QUERY_SPARSE)


class ReverseReranker:
    """Prefers passages late in the alphabet, so b ends up low."""

    model_name = "reverse"

    def score(self, query: str, passages: Sequence[str]) -> tuple[float, ...]:
        return tuple(float(ord(text[0]) - ord("a")) for text in passages)


@pytest.fixture(params=["local", _SERVER])
def store(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Iterator[QdrantPassageStore]:
    if request.param == "local":
        local = QdrantPassageStore(tmp_path / "qdrant", alias=ALIAS)
        build(local)
        yield local
        local.close()
        return
    assert URL is not None
    raw = QdrantClient(url=URL)
    wipe(raw)
    remote = QdrantPassageStore(url=URL, alias=ALIAS)
    build(remote)
    yield remote
    remote.close()
    wipe(raw)
    raw.close()


def _run(store: QdrantPassageStore, top_k: int, depth: int) -> EvalReport:
    search = FindPassages(
        embedder=FixedEmbedder(),
        sparse_encoder=FixedSparse(),
        retriever=store,
        reranker=ReverseReranker(),
        top_k=top_k,
        default_top_n=1,
        top_n_max=10,
        max_query_len=100,
        now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )
    case = EvalCase("b", 1, "beta", source="two.txt", fragment="beta")
    return EvaluateRetrieval(search=search, probe=store).run(
        [case], depth=depth, control=False
    )


def test_one_passage_ranks_differently_in_each_list(
    store: QdrantPassageStore,
) -> None:
    (result,) = _run(store, top_k=6, depth=10).cases

    # Dense a, b, ...; sparse f, c, e, b; fused f, c, b, ...; reranked by
    # the reverse alphabet: f, e, d, c, b, a. two.txt holds b and e.
    assert result.lists["dense"] == ListRank(2, 2)
    assert result.lists["sparse"] == ListRank(4, 3)
    assert result.lists["fused_deep"] == ListRank(3, 3)
    assert result.lists["stage1"] == ListRank(3, 3)
    assert result.lists["final"] == ListRank(5, 2)
    assert result.within_top_k
    assert result.rerank_score is not None


def test_a_passage_outside_top_k_is_not_within_it(store: QdrantPassageStore) -> None:
    (result,) = _run(store, top_k=3, depth=10).cases

    # At depth 3 the fusion keeps c, a and f.
    assert result.lists["fused_deep"].rank == 3
    assert result.lists["stage1"] == ListRank(None, None)
    assert result.lists["final"] == ListRank(None, None)
    assert not result.within_top_k


def test_ranks_beyond_the_depth_are_null(store: QdrantPassageStore) -> None:
    (result,) = _run(store, top_k=6, depth=1).cases

    assert result.lists["dense"] == ListRank(None, None)
    assert result.lists["sparse"] == ListRank(None, None)
    assert result.lists["fused_deep"] == ListRank(None, None)
