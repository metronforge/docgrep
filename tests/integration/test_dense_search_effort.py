"""The search effort of the dense channel in the Qdrant store, in local mode
and against a real server (set DOCGREP_TEST_QDRANT_URL) (ADR-0026)."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from docgrep.adapters.qdrant import QdrantPassageStore
from docgrep.domain.errors import ValidationError
from docgrep.domain.protocols import ChannelProbe
from docgrep.domain.search_effort import DenseSearchEffort
from eval_support import (
    ALIAS,
    DENSE_ORDER,
    QUERY_DENSE,
    QUERY_SPARSE,
    SPARSE_ORDER,
    build,
    texts,
    wipe,
)

URL = os.environ.get("DOCGREP_TEST_QDRANT_URL")
_SERVER = pytest.param(
    "server",
    marks=[
        pytest.mark.qdrant_server,
        pytest.mark.skipif(not URL, reason="DOCGREP_TEST_QDRANT_URL is not set"),
    ],
)
EFFORTS: tuple[DenseSearchEffort | None, ...] = (None, 1, 1024, "exact")


def _open(
    mode: str, root: Path, effort: DenseSearchEffort | None
) -> QdrantPassageStore:
    if mode == "local":
        return QdrantPassageStore(root / "qdrant", alias=ALIAS, dense_hnsw_ef=effort)
    assert URL is not None
    return QdrantPassageStore(url=URL, alias=ALIAS, dense_hnsw_ef=effort)


@pytest.fixture(params=["local", _SERVER])
def mode(request: pytest.FixtureRequest) -> Iterator[str]:
    if request.param == "server":
        assert URL is not None
        raw = QdrantClient(url=URL)
        wipe(raw)
        yield "server"
        wipe(raw)
        raw.close()
        return
    yield "local"


@pytest.mark.parametrize(
    ("effort", "limit", "expected"),
    [
        (None, 100, None),
        ("exact", 100, models.SearchParams(exact=True)),
        (1024, 100, models.SearchParams(hnsw_ef=1024)),
        # A request for more hits than the effort never searches less.
        (1024, 2000, models.SearchParams(hnsw_ef=2000)),
        (64, 100, models.SearchParams(hnsw_ef=100)),
    ],
)
def test_the_parameters_follow_the_effort_and_the_limit(
    tmp_path: Path,
    effort: DenseSearchEffort | None,
    limit: int,
    expected: models.SearchParams | None,
) -> None:
    store = QdrantPassageStore(tmp_path / "qdrant", alias=ALIAS, dense_hnsw_ef=effort)
    try:
        assert store.dense_hnsw_ef == effort
        assert store.dense_search_params(limit) == expected
    finally:
        store.close()


@pytest.mark.parametrize("effort", EFFORTS)
def test_every_effort_returns_the_exact_lists_on_a_small_index(
    mode: str, tmp_path: Path, effort: DenseSearchEffort | None
) -> None:
    """Below the full-scan threshold every effort searches exactly: the
    probe lists are the exact orders, and the first stage is the fused list
    at that depth, whatever the effort."""
    store = _open(mode, tmp_path, effort)
    try:
        build(store)
        lists = store.probe_channels(QUERY_DENSE, QUERY_SPARSE, 10)
        first_stage = store.retrieve(QUERY_DENSE, QUERY_SPARSE, 3)
        fused_at_three = store.probe_channels(QUERY_DENSE, QUERY_SPARSE, 3).fused
    finally:
        store.close()

    assert texts(lists.dense) == list(DENSE_ORDER)
    assert texts(lists.sparse) == list(SPARSE_ORDER)
    assert [c.passage.passage_id for c in first_stage] == [
        hit.passage_id for hit in fused_at_three
    ]


@pytest.mark.parametrize("effort", EFFORTS)
def test_dense_recall_against_exact_search(
    mode: str, tmp_path: Path, effort: DenseSearchEffort | None
) -> None:
    store = _open(mode, tmp_path, effort)
    try:
        probe: ChannelProbe = store
        assert probe.dense_recall(QUERY_DENSE, 3) is None  # nothing published
        build(store)

        assert probe.dense_recall(QUERY_DENSE, 3) == 1.0
        assert probe.dense_recall(QUERY_DENSE, 3, source_ids={"two.txt"}) == 1.0
        assert probe.dense_recall(QUERY_DENSE, 3, source_ids=set()) is None
        assert probe.dense_recall(QUERY_DENSE, 3, source_ids={"absent.txt"}) is None
    finally:
        store.close()


@pytest.mark.parametrize("limit", [0, 10001])
def test_dense_recall_validates_the_limit(tmp_path: Path, limit: int) -> None:
    store = QdrantPassageStore(tmp_path / "qdrant", alias=ALIAS)
    try:
        build(store)
        with pytest.raises(ValidationError, match="limit"):
            store.dense_recall(QUERY_DENSE, limit)
    finally:
        store.close()
