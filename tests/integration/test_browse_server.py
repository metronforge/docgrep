"""Source browsing against a real Qdrant server (set DOCGREP_TEST_QDRANT_URL)
(UC-08)."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

from docgrep.api.container import create_container
from docgrep.api.http import create_app
from docgrep.settings import Settings
from mcp_support import write_library
from resume_support import close

URL = os.environ.get("DOCGREP_TEST_QDRANT_URL")

pytestmark = [
    pytest.mark.qdrant_server,
    pytest.mark.skipif(not URL, reason="DOCGREP_TEST_QDRANT_URL is not set"),
]


def _wipe(client: QdrantClient) -> None:
    for alias in client.get_aliases().aliases:
        client.update_collection_aliases(
            [
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name=alias.alias_name)
                )
            ]
        )
    for collection in client.get_collections().collections:
        if collection.name.startswith("docgrep_"):
            client.delete_collection(collection.name)


@pytest.fixture
def server() -> Iterator[QdrantClient]:
    assert URL is not None
    raw = QdrantClient(url=URL)
    _wipe(raw)
    yield raw
    _wipe(raw)
    raw.close()


def test_counts_come_from_the_server_and_match_the_local_body(
    tmp_path: Path, server: QdrantClient
) -> None:
    assert URL is not None
    write_library(tmp_path / "corpus")
    settings = Settings(
        corpus_root=tmp_path / "corpus",
        data_dir=tmp_path / "data",
        backend="fake",
        chunk_size=64,
        chunk_overlap=8,
        qdrant_url=URL,
    )
    services = create_container(settings)
    services.build.run()
    try:
        client = TestClient(
            create_app(settings, services=services), base_url="http://127.0.0.1"
        )
        body = client.post(
            "/sources/browse", json={"sort_by": "passage_count", "sort_order": "desc"}
        ).json()
    finally:
        close(services)

    assert body["total_hits"] == 4
    counts = {item["path"]: item["passage_count"] for item in body["sources"]}
    stored = server.count(
        "docgrep",
        count_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="record_type", match=models.MatchValue(value="passage")
                )
            ]
        ),
        exact=True,
    ).count
    assert sum(counts.values()) == stored
    assert list(counts) == sorted(counts, key=lambda path: (-counts[path], path))
