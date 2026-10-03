"""Resume against a real Qdrant server (set DOCGREP_TEST_QDRANT_URL)."""

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient, models

from docgrep.adapters.manifest import SQLiteManifest
from docgrep.domain.errors import NothingToResume, StoreUnavailable
from resume_support import (
    Crash,
    cancel_at_embed_call,
    close,
    dump_points,
    fail_at_swapped_mark,
    fail_upsert_at,
    new_services,
    search_snapshot,
    settings_for,
    write_corpus,
)

URL = os.environ.get("DOCGREP_TEST_QDRANT_URL")

pytestmark = [
    pytest.mark.qdrant_server,
    pytest.mark.skipif(not URL, reason="DOCGREP_TEST_QDRANT_URL is not set"),
]

CLEAN = "rsclean"
INTERRUPTED = "rsint"
SOURCES = 5


def _wipe(client: QdrantClient) -> None:
    for alias in client.get_aliases().aliases:
        if alias.alias_name in (CLEAN, INTERRUPTED):
            client.update_collection_aliases(
                [
                    models.DeleteAliasOperation(
                        delete_alias=models.DeleteAlias(alias_name=alias.alias_name)
                    )
                ]
            )
    for collection in client.get_collections().collections:
        if collection.name.startswith((f"{CLEAN}_", f"{INTERRUPTED}_")):
            client.delete_collection(collection.name)


@pytest.fixture(autouse=True)
def server() -> Iterator[QdrantClient]:
    assert URL is not None
    client = QdrantClient(url=URL)
    _wipe(client)
    yield client
    _wipe(client)
    client.close()


def _staging_collections(client: QdrantClient) -> list[str]:
    return [
        c.name
        for c in client.get_collections().collections
        if c.name.startswith(f"{INTERRUPTED}_")
    ]


def _clean_reference(corpus: Path, tmp_path: Path) -> tuple[list[Any], list[Any]]:
    settings = settings_for(corpus, tmp_path / "clean", url=URL, alias=CLEAN)
    services = new_services(settings)
    services.build.run()
    searched = search_snapshot(services)
    close(services)
    return dump_points(settings), searched


@pytest.mark.parametrize("after", range(SOURCES + 1))
def test_the_key_invariant_holds_on_a_server(tmp_path: Path, after: int) -> None:
    corpus = tmp_path / "corpus"
    write_corpus(corpus, SOURCES)
    reference_points, reference_search = _clean_reference(corpus, tmp_path)
    settings = settings_for(corpus, tmp_path / "data", url=URL, alias=INTERRUPTED)
    services = new_services(settings)
    cancel_at_embed_call(services, after)
    assert services.build.run().cancelled
    close(services)

    resumed = new_services(settings)
    summary = resumed.build.resume()
    searched = search_snapshot(resumed)
    close(resumed)

    assert (summary.carried_over, summary.processed) == (after, SOURCES - after)
    assert dump_points(settings) == reference_points
    assert searched == reference_search


@pytest.mark.parametrize("failing_call", [1, 2, 5, 9])
def test_a_failure_in_the_middle_of_a_source_resumes_on_a_server(
    tmp_path: Path, failing_call: int
) -> None:
    corpus = tmp_path / "corpus"
    write_corpus(corpus, SOURCES)
    reference_points, reference_search = _clean_reference(corpus, tmp_path)
    settings = settings_for(
        corpus, tmp_path / "data", url=URL, alias=INTERRUPTED, upsert_batch_size=1
    )
    services = new_services(settings)
    fail_upsert_at(services, failing_call)
    with pytest.raises(StoreUnavailable):
        services.build.run()
    close(services)

    resumed = new_services(
        settings_for(corpus, tmp_path / "data", url=URL, alias=INTERRUPTED)
    )
    resumed.build.resume()
    searched = search_snapshot(resumed)
    close(resumed)

    assert dump_points(settings) == reference_points
    assert searched == reference_search


def test_points_of_a_half_written_source_that_changed_or_vanished_are_removed(
    tmp_path: Path, server: QdrantClient
) -> None:
    corpus = tmp_path / "corpus"
    write_corpus(corpus, SOURCES)
    settings = settings_for(
        corpus, tmp_path / "data", url=URL, alias=INTERRUPTED, upsert_batch_size=1
    )
    services = new_services(settings)
    # doc000 has three passages; the failure leaves its first point behind.
    fail_upsert_at(services, 2)
    with pytest.raises(StoreUnavailable):
        services.build.run()
    close(services)
    run = SQLiteManifest(tmp_path / "data").latest_run()
    assert run is not None and run.status == "aborted"
    assert server.count(run.staging_collection or "", exact=True).count == 1
    assert SQLiteManifest(tmp_path / "data").run_sources(run.run_id) == ()

    # The half-written source is gone from the corpus; another one changed.
    (corpus / "doc000.txt").unlink()
    (corpus / "doc001.txt").write_text(
        "a completely different and shorter text", "utf-8"
    )
    resumed = new_services(
        settings_for(corpus, tmp_path / "data", url=URL, alias=INTERRUPTED)
    )
    summary = resumed.build.resume()
    close(resumed)

    assert summary.carried_over == 0 and summary.processed == SOURCES - 1
    reference_points, _ = _clean_reference(corpus, tmp_path)
    assert dump_points(settings) == reference_points
    assert not any(p[1]["path"] == "doc000.txt" for p in dump_points(settings))
    assert _staging_collections(server) == [
        c.collection_name
        for c in server.get_aliases().aliases
        if c.alias_name == INTERRUPTED
    ]


def test_a_run_that_died_after_the_swap_is_finished_on_a_server(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    write_corpus(corpus, SOURCES)
    reference_points, reference_search = _clean_reference(corpus, tmp_path)
    settings = settings_for(corpus, tmp_path / "data", url=URL, alias=INTERRUPTED)
    services = new_services(settings)
    fail_at_swapped_mark(services, crash=True)
    with pytest.raises(Crash):
        services.build.run()
    close(services)
    manifest = SQLiteManifest(tmp_path / "data")
    run = manifest.latest_run()
    assert run is not None and run.status == "running"

    resumed = new_services(settings)
    summary = resumed.build.resume()
    searched = search_snapshot(resumed)
    close(resumed)

    assert summary.recovered and summary.indexed == SOURCES
    done = SQLiteManifest(tmp_path / "data")
    finished = done.latest_run()
    assert finished is not None and finished.status == "completed"
    assert all(done.get_source(f"doc{i:03d}.txt") is not None for i in range(SOURCES))
    assert dump_points(settings) == reference_points
    assert searched == reference_search
    again = new_services(settings)
    with pytest.raises(NothingToResume):
        again.build.resume()
    close(again)
