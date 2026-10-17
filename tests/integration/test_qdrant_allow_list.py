"""First-stage retrieval restricted to an allow-list of sources, in local
mode and against a real server (set DOCGREP_TEST_QDRANT_URL) (UC-02)."""

import os
import random
import statistics
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from docgrep.adapters.qdrant import QdrantPassageStore
from docgrep.domain.models import FileType, Passage, Source, make_passage_id

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


def _source(name: str) -> Source:
    return Source(name, name, FileType.TXT)


def _upsert(
    store: QdrantPassageStore,
    source: Source,
    text: str,
    offset: int,
    dense: tuple[float, ...],
    sparse: dict[int, float],
) -> None:
    passage = Passage(
        make_passage_id(source.source_id, offset),
        source.source_id,
        text,
        offset,
        1,
    )
    store.upsert(passage, source, dense, sparse)


QUERY_DENSE = (1.0, 0.0)
QUERY_SPARSE = {7: 1.0}


def _build(store: QdrantPassageStore) -> None:
    """Decoys win both prefetches; the target has one passage that only the
    dense query ranks well and one that only the sparse query finds."""
    store.recreate_collection(2)
    for number in range(6):
        _upsert(
            store, _source(f"decoy{number}"), f"decoy {number}", 0, (1.0, 0.0), {7: 5.0}
        )
    target = _source("target")
    _upsert(store, target, "dense", 0, (0.8, 0.6), {99: 1.0})
    _upsert(store, target, "sparse", 1, (0.0, 1.0), {7: 1.0})
    _upsert(store, target, "neither", 2, (0.6, 0.8), {98: 1.0})
    store.write_completion_metadata({"embedding_model": "m"})


def _texts(store: QdrantPassageStore, top_k: int, **kwargs: object) -> list[str]:
    candidates = store.retrieve(QUERY_DENSE, QUERY_SPARSE, top_k, **kwargs)  # type: ignore[arg-type]  # source_ids by name
    return [candidate.passage.text for candidate in candidates]


def test_the_allow_list_applies_before_both_prefetches_rank(
    store: QdrantPassageStore,
) -> None:
    _build(store)

    assert all(text.startswith("decoy") for text in _texts(store, 2))

    allowed = _texts(store, 2, source_ids=["target"])
    # "sparse" is reached only by a filtered sparse prefetch and "dense" only
    # by a filtered dense one; filtering the unfiltered top_k afterwards would
    # find neither.
    assert sorted(allowed) == ["dense", "sparse"]
    assert set(_texts(store, 10, source_ids={"target", "decoy3"})) == {
        "dense",
        "sparse",
        "neither",
        "decoy 3",
    }


def test_an_empty_or_unknown_allow_list_returns_nothing(
    store: QdrantPassageStore,
) -> None:
    _build(store)

    assert _texts(store, 5, source_ids=[]) == []
    assert _texts(store, 5, source_ids=["missing"]) == []
    assert len(_texts(store, 5, source_ids=None)) == 5


@pytest.mark.qdrant_server
@pytest.mark.skipif(not URL, reason="DOCGREP_TEST_QDRANT_URL is not set")
def test_a_two_thousand_id_allow_list_on_the_server() -> None:
    """Measure a 2,000-id allow-list against the unfiltered query; with -s the
    timings are also printed."""
    assert URL is not None
    raw = QdrantClient(url=URL)
    _wipe(raw)
    store = QdrantPassageStore(url=URL, alias=ALIAS)
    try:
        rng = random.Random(7)
        dimension = 64
        store.recreate_collection(dimension)
        sources = [_source(f"source-{number:05}") for number in range(5_000)]
        batch = []
        for source in sources:
            vector = tuple(rng.uniform(-1, 1) for _ in range(dimension))
            sparse = {rng.randrange(1_000): 1.0 for _ in range(8)}
            passage = Passage(
                make_passage_id(source.source_id, 0), source.source_id, "text", 0, 1
            )
            batch.append((passage, source, vector, sparse))
        for start in range(0, len(batch), 500):
            store.upsert_batch(batch[start : start + 500])
        store.write_completion_metadata({"embedding_model": "m"})
        allowed = {source.source_id for source in rng.sample(sources, 2_000)}
        query = tuple(rng.uniform(-1, 1) for _ in range(dimension))
        sparse_query = {rng.randrange(1_000): 1.0 for _ in range(4)}

        def timed(source_ids: set[str] | None) -> float:
            samples = []
            for _ in range(15):
                started = time.perf_counter()
                found = store.retrieve(query, sparse_query, 100, source_ids=source_ids)
                samples.append((time.perf_counter() - started) * 1000)
                assert len(found) == 100
                if source_ids is not None:
                    assert {item.source.source_id for item in found} <= source_ids
            return statistics.median(samples)

        timed(None)  # warm-up
        unfiltered = timed(None)
        filtered = timed(allowed)
        print(
            f"\nretrieve median over 15: unfiltered {unfiltered:.1f} ms, "
            f"2,000-id allow-list {filtered:.1f} ms"
        )
        assert filtered < 2_000
    finally:
        store.close()
        _wipe(raw)
        raw.close()
