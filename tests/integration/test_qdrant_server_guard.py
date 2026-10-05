"""The qdrant_server guard against a real server (set DOCGREP_TEST_QDRANT_URL)."""

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient, models

from qdrant_server_guard import check_server

URL = os.environ.get("DOCGREP_TEST_QDRANT_URL")

pytestmark = [
    pytest.mark.qdrant_server,
    pytest.mark.skipif(not URL, reason="DOCGREP_TEST_QDRANT_URL is not set"),
]

ROOT = Path(__file__).resolve().parents[2]


def _names(client: QdrantClient) -> tuple[set[str], dict[str, str]]:
    return (
        {c.name for c in client.get_collections().collections},
        {a.alias_name: a.collection_name for a in client.get_aliases().aliases},
    )


def _create(client: QdrantClient, collection: str, alias: str) -> None:
    client.create_collection(
        collection,
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    client.update_collection_aliases(
        [
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=collection, alias_name=alias
                )
            )
        ]
    )


def _drop(client: QdrantClient, collection: str, alias: str) -> None:
    if alias in _names(client)[1]:
        client.update_collection_aliases(
            [
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name=alias)
                )
            ]
        )
    if collection in _names(client)[0]:
        client.delete_collection(collection)


@pytest.fixture
def client() -> Iterator[QdrantClient]:
    assert URL is not None
    raw = QdrantClient(url=URL)
    yield raw
    raw.close()


@pytest.fixture
def foreign(client: QdrantClient) -> Iterator[tuple[str, str]]:
    """An unrelated collection and alias, removed again by this fixture only."""
    collection, alias = f"library{uuid4().hex[:8]}", f"books{uuid4().hex[:8]}"
    _create(client, collection, alias)
    yield collection, alias
    _drop(client, collection, alias)


def test_the_guard_passes_with_only_suite_leftovers(client: QdrantClient) -> None:
    assert URL is not None
    collection = f"alpha_{uuid4().hex}"
    _create(client, collection, "alpha")
    try:
        assert check_server(URL) is None
    finally:
        _drop(client, collection, "alpha")
    assert check_server(URL) is None


def test_the_guard_refuses_a_foreign_collection_and_alias_and_deletes_nothing(
    client: QdrantClient, foreign: tuple[str, str]
) -> None:
    assert URL is not None
    collection, alias = foreign
    before = _names(client)

    message = check_server(URL)
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/integration/test_qdrant_server.py"
            "::test_delete_source_points_removes_exactly_that_source",
        ],
        cwd=ROOT,
        env={**os.environ, "DOCGREP_TEST_QDRANT_URL": URL},
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert message is not None
    assert collection in message and alias in message
    assert run.returncode == pytest.ExitCode.USAGE_ERROR
    assert collection in run.stdout + run.stderr
    assert "passed" not in run.stdout
    assert _names(client) == before
