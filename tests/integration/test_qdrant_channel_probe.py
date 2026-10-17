"""The per-channel probe of the Qdrant store, in local mode and against a
real server (set DOCGREP_TEST_QDRANT_URL) (ADR-0024)."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from docgrep.adapters.qdrant import QdrantPassageStore
from docgrep.domain.errors import ValidationError
from docgrep.domain.evaluation import MAX_PROBE_DEPTH
from docgrep.domain.protocols import ChannelProbe
from eval_support import (
    DENSE_ORDER,
    QUERY_DENSE,
    QUERY_SPARSE,
    SPARSE_ORDER,
    build,
    dense_only_index,
    texts,
)

URL = os.environ.get("DOCGREP_TEST_QDRANT_URL")
ALIAS = "alpha"
_SERVER = pytest.param(
    "server",
    marks=[
        pytest.mark.qdrant_server,
        pytest.mark.skipif(not URL, reason="DOCGREP_TEST_QDRANT_URL is not set"),
    ],
)


def _wipe(client: QdrantClient) -> None:
    for alias in client.get_aliases().aliases:
        if alias.alias_name == ALIAS:
            client.update_collection_aliases(
                [
                    models.DeleteAliasOperation(
                        delete_alias=models.DeleteAlias(alias_name=ALIAS)
                    )
                ]
            )
    for collection in client.get_collections().collections:
        if collection.name.startswith(f"{ALIAS}_"):
            client.delete_collection(collection.name)


@pytest.fixture(params=["local", _SERVER])
def store(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Iterator[QdrantPassageStore]:
    if request.param == "local":
        local = QdrantPassageStore(tmp_path / "qdrant", alias=ALIAS)
        yield local
        local.close()
        return
    assert URL is not None
    raw = QdrantClient(url=URL)
    _wipe(raw)
    remote = QdrantPassageStore(url=URL, alias=ALIAS)
    yield remote
    remote.close()
    _wipe(raw)
    raw.close()


def test_each_channel_is_listed_apart(store: QdrantPassageStore) -> None:
    build(store)
    probe: ChannelProbe = store

    lists = probe.probe_channels(QUERY_DENSE, QUERY_SPARSE, 10)

    assert texts(lists.dense) == list(DENSE_ORDER)
    assert texts(lists.sparse) == list(SPARSE_ORDER)
    assert sorted(texts(lists.fused)) == sorted(DENSE_ORDER)
    # The marker point is never a hit, and text is read only when asked.
    assert all(hit.text is None for hit in (*lists.dense, *lists.fused))


def test_the_fused_list_is_the_first_stage_at_that_depth(
    store: QdrantPassageStore,
) -> None:
    build(store)

    for depth in (2, 3, 6):
        lists = store.probe_channels(QUERY_DENSE, QUERY_SPARSE, depth)
        retrieved = store.retrieve(QUERY_DENSE, QUERY_SPARSE, depth)
        assert [hit.passage_id for hit in lists.fused] == [
            item.passage.passage_id for item in retrieved
        ], depth
        assert len(lists.dense) == depth
        assert len(lists.sparse or ()) == min(depth, len(SPARSE_ORDER))


def test_the_channels_rank_one_passage_differently(store: QdrantPassageStore) -> None:
    build(store)

    lists = store.probe_channels(QUERY_DENSE, QUERY_SPARSE, 10)

    ranks = [
        texts(found).index("b beta") + 1
        for found in (lists.dense, lists.sparse, lists.fused)
    ]
    # Qdrant's fusion scores 1 / (rank + 1): b has 1/3 + 1/5, above a's
    # 1/2 and below c's 1/4 + 1/3.
    assert ranks == [2, 4, 3]


def test_text_is_read_on_request(store: QdrantPassageStore) -> None:
    build(store)

    lists = store.probe_channels(QUERY_DENSE, QUERY_SPARSE, 3, with_text=True)

    assert [hit.text for hit in lists.dense] == list(DENSE_ORDER[:3])
    assert all(hit.text for hit in (*(lists.sparse or ()), *lists.fused))


def test_the_allow_list_applies_to_every_list(store: QdrantPassageStore) -> None:
    build(store)

    lists = store.probe_channels(QUERY_DENSE, QUERY_SPARSE, 10, source_ids={"two.txt"})
    empty = store.probe_channels(QUERY_DENSE, QUERY_SPARSE, 10, source_ids=())

    assert texts(lists.dense) == ["b beta", "e epsilon"]
    assert texts(lists.sparse) == ["e epsilon", "b beta"]
    assert sorted(texts(lists.fused)) == ["b beta", "e epsilon"]
    assert (empty.dense, empty.sparse, empty.fused) == ((), (), ())


def test_a_query_without_sparse_terms_fuses_the_dense_list_alone(
    store: QdrantPassageStore,
) -> None:
    build(store)

    lists = store.probe_channels(QUERY_DENSE, {}, 4)

    assert lists.sparse == ()
    assert texts(lists.fused) == texts(lists.dense) == list(DENSE_ORDER[:4])


def test_no_published_index_gives_empty_lists(store: QdrantPassageStore) -> None:
    lists = store.probe_channels(QUERY_DENSE, QUERY_SPARSE, 5)

    assert (lists.dense, lists.sparse, lists.fused) == ((), (), ())


@pytest.mark.parametrize("depth", [0, MAX_PROBE_DEPTH + 1])
def test_the_depth_is_bounded(store: QdrantPassageStore, depth: int) -> None:
    build(store)
    with pytest.raises(ValidationError, match="depth"):
        store.probe_channels(QUERY_DENSE, QUERY_SPARSE, depth)


def test_an_index_without_a_sparse_channel_reports_none(tmp_path: Path) -> None:
    dense_only_index(tmp_path / "qdrant")
    store = QdrantPassageStore(tmp_path / "qdrant")
    try:
        lists = store.probe_channels(QUERY_DENSE, QUERY_SPARSE, 10)
    finally:
        store.close()

    assert lists.sparse is None
    assert texts(lists.fused) == texts(lists.dense) == list(DENSE_ORDER)
