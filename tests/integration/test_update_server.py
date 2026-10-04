"""Incremental update against a real Qdrant server (set DOCGREP_TEST_QDRANT_URL)."""

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient, models

from docgrep.adapters.manifest import SQLiteManifest
from resume_support import close, dump_points, search_snapshot, settings_for
from update_support import (
    CHANGES,
    FINAL,
    INITIAL,
    PLAN,
    QUERIES,
    cancel_after_sources,
    change_corpus,
    fail_live_upsert_at,
    sources_table,
    update_services,
    without_epub,
    write_files,
)

URL = os.environ.get("DOCGREP_TEST_QDRANT_URL")

pytestmark = [
    pytest.mark.qdrant_server,
    pytest.mark.skipif(not URL, reason="DOCGREP_TEST_QDRANT_URL is not set"),
]

CLEAN = "upclean"
UPDATED = "upupdated"


def _wipe(client: QdrantClient) -> None:
    for alias in client.get_aliases().aliases:
        if alias.alias_name in (CLEAN, UPDATED):
            client.update_collection_aliases(
                [
                    models.DeleteAliasOperation(
                        delete_alias=models.DeleteAlias(alias_name=alias.alias_name)
                    )
                ]
            )
    for collection in client.get_collections().collections:
        if collection.name.startswith((f"{CLEAN}_", f"{UPDATED}_")):
            client.delete_collection(collection.name)


@pytest.fixture(autouse=True)
def server() -> Iterator[QdrantClient]:
    assert URL is not None
    client = QdrantClient(url=URL)
    _wipe(client)
    yield client
    _wipe(client)
    client.close()


def _snapshot(settings: Any) -> tuple[Any, ...]:
    services = update_services(settings)
    try:
        searched = search_snapshot(services, QUERIES)
    finally:
        close(services)
    return dump_points(settings), searched, sources_table(settings)


def _reference(tmp_path: Path) -> tuple[Any, ...]:
    corpus = tmp_path / "final"
    write_files(corpus, FINAL)
    settings = settings_for(corpus, tmp_path / "clean", url=URL, alias=CLEAN)
    services = update_services(settings)
    services.build.run()
    close(services)
    return _snapshot(settings)


def _initial_index(tmp_path: Path) -> Any:
    corpus = tmp_path / "corpus"
    write_files(corpus, INITIAL)
    settings = settings_for(corpus, tmp_path / "data", url=URL, alias=UPDATED)
    services = update_services(settings)
    without_epub(services)
    services.build.run()
    close(services)
    change_corpus(corpus)
    return settings


def _update(settings: Any) -> Any:
    services = update_services(settings)
    try:
        assert services.update is not None
        return services.update.run()
    finally:
        close(services)


def test_an_update_equals_a_clean_build_on_a_server(tmp_path: Path) -> None:
    reference = _reference(tmp_path)
    settings = _initial_index(tmp_path)

    summary = _update(settings)

    plan = summary.plan
    assert (plan.added, plan.modified, plan.removed, plan.unchanged) == (
        PLAN["added"],
        PLAN["modified"],
        PLAN["removed"],
        PLAN["unchanged"],
    )
    assert _snapshot(settings) == reference
    assert _update(settings).no_changes
    assert _snapshot(settings) == reference


@pytest.mark.parametrize("after", [0, 3, CHANGES])
def test_a_cancelled_update_on_a_server_continues_to_the_same_index(
    tmp_path: Path, after: int
) -> None:
    reference = _reference(tmp_path)
    settings = _initial_index(tmp_path)
    services = update_services(settings)
    cancel_after_sources(services, after)
    assert services.update is not None
    assert services.update.run().cancelled
    close(services)

    _update(settings)

    assert _snapshot(settings) == reference


def test_a_failure_inside_a_source_on_a_server_is_repaired(tmp_path: Path) -> None:
    reference = _reference(tmp_path)
    settings = _initial_index(tmp_path)
    services = update_services(settings)
    update: Any = services.update
    update.upsert_batch_size = 2
    fail_live_upsert_at(services, 2)
    assert update.run().failed > 0
    close(services)
    run = SQLiteManifest(settings.data_dir).latest_run()
    assert run is not None and run.status == "completed"

    assert _update(settings).failed == 0

    assert _snapshot(settings) == reference
