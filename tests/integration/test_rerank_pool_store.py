"""The rerank pool against local-mode Qdrant and over HTTP (ADR-0025)."""

from pathlib import Path

from fastapi.testclient import TestClient

from docgrep.adapters.models.deterministic import DeterministicEmbedder
from docgrep.adapters.models.sparse import UniqueTermEncoder
from docgrep.adapters.qdrant import QdrantPassageStore
from docgrep.api.container import create_container
from docgrep.api.http import create_app
from resume_support import close
from search_filter_support import TOP_K, filter_settings
from search_regression_support import http_bodies, load_fixture, write_library
from sparse_support import as_tf_v1_build


def test_the_fused_list_keeps_the_pool_while_each_channel_keeps_top_k(
    tmp_path: Path,
) -> None:
    write_library(tmp_path / "corpus")
    settings = filter_settings(tmp_path)
    services = create_container(settings)
    services.build.run()
    close(services)
    store = QdrantPassageStore(settings.data_dir / "qdrant")
    try:
        query = "comet orbit tables"
        dense = DeterministicEmbedder().embed((query,))[0]
        sparse = UniqueTermEncoder().encode(query)
        plain = store.retrieve(dense, sparse, TOP_K)
        pooled = store.retrieve(dense, sparse, TOP_K, pool=2 * TOP_K)
        lists = store.probe_channels(dense, sparse, TOP_K)
    finally:
        store.close()

    union = {hit.passage_id for hit in (*lists.dense, *(lists.sparse or ()))}
    assert len(plain) == TOP_K
    assert len(pooled) == len(union) > TOP_K
    assert {c.passage.passage_id for c in pooled} == union
    # The first top_k of the pool are the default first stage.
    assert [c.passage.passage_id for c in pooled[:TOP_K]] == [
        c.passage.passage_id for c in plain
    ]
    assert [c.stage1_rank for c in pooled] == list(range(len(pooled)))


def test_a_pool_equal_to_top_k_keeps_the_recorded_http_bodies(
    tmp_path: Path,
) -> None:
    write_library(tmp_path / "corpus")
    settings = filter_settings(tmp_path, rerank_pool=TOP_K)
    services = create_container(settings)
    summary = as_tf_v1_build(services.build).run()
    assert (summary.indexed, summary.failed) == (18, 0), summary
    try:
        api = TestClient(
            create_app(settings, services=services), base_url="http://127.0.0.1"
        )
        bodies = http_bodies(api)
    finally:
        close(services)

    assert bodies == load_fixture()["http"]
