"""POST /index/build refuses to discard a resumable run, like the console."""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from docgrep.adapters.filelock import FileIndexLock
from docgrep.adapters.manifest import SQLiteManifest
from docgrep.api.http import create_app
from docgrep.settings import Settings
from resume_support import (
    Crash,
    cancel_at_embed_call,
    close,
    fail_at_swapped_mark,
    new_services,
    settings_for,
    write_corpus,
)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    corpus = tmp_path / "corpus"
    write_corpus(corpus, 5)
    return settings_for(corpus, tmp_path / "data")


def _client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings), base_url="http://127.0.0.1")


def _cancelled(settings: Settings) -> None:
    services = new_services(settings)
    cancel_at_embed_call(services, 2)
    assert services.build.run().cancelled
    close(services)


def test_a_resumable_run_is_refused_with_409_and_a_hint(settings: Settings) -> None:
    _cancelled(settings)
    run = SQLiteManifest(settings.data_dir).latest_run()
    assert run is not None

    with _client(settings) as client:
        response = client.post("/index/build")

    assert response.status_code == 409
    body = response.json()
    assert set(body) == {"category", "message", "details"}
    assert body["category"] == "resume_available"
    assert "docgrep index --resume" in body["message"]
    assert '"restart": true' in body["message"]
    assert body["details"] == {"run_id": run.run_id, "state": "cancelled"}
    unchanged = SQLiteManifest(settings.data_dir).latest_run()
    assert unchanged == run  # nothing was started or discarded


@pytest.mark.parametrize("body", [None, {}, {"restart": False}])
def test_the_body_is_optional_and_restart_defaults_to_false(
    settings: Settings, body: dict[str, Any] | None
) -> None:
    _cancelled(settings)

    with _client(settings) as client:
        response = (
            client.post("/index/build")
            if body is None
            else client.post("/index/build", json=body)
        )

    assert response.status_code == 409
    assert response.json()["category"] == "resume_available"


def test_restart_starts_a_fresh_build_and_removes_the_retained_staging(
    settings: Settings,
) -> None:
    _cancelled(settings)
    retained = SQLiteManifest(settings.data_dir).latest_run()
    assert retained is not None and retained.staging_collection is not None

    with _client(settings) as client:
        response = client.post("/index/build", json={"restart": True})
        build = response.json()
    assert response.status_code == 200, response.text
    assert (build["indexed"], build["failed"]) == (5, 0)
    after = SQLiteManifest(settings.data_dir).latest_run()
    assert after is not None and after.status == "completed"
    assert after.run_id != retained.run_id and after.resumed_from is None

    from qdrant_client import QdrantClient

    reader = QdrantClient(path=str(settings.data_dir / "qdrant"))
    names = {c.name for c in reader.get_collections().collections}
    reader.close()
    assert retained.staging_collection not in names and len(names) == 1


def test_an_interrupted_run_is_refused_too(settings: Settings) -> None:
    SQLiteManifest(settings.data_dir).start_run({"chunk_size": 10})  # killed

    with _client(settings) as client:
        response = client.post("/index/build")

    assert response.status_code == 409
    assert response.json()["details"]["state"] == "interrupted"


def test_a_run_that_only_lacks_its_commit_is_finished_and_the_build_proceeds(
    settings: Settings,
) -> None:
    services = new_services(settings)
    fail_at_swapped_mark(services, crash=True)
    with pytest.raises(Crash):
        services.build.run()
    close(services)

    with _client(settings) as client:
        response = client.post("/index/build")

    assert response.status_code == 200
    manifest = SQLiteManifest(settings.data_dir)
    with manifest._connection() as connection:
        rows = connection.execute("SELECT id, status FROM runs ORDER BY id").fetchall()
    assert [tuple(r) for r in rows] == [(1, "completed"), (2, "completed")]


def test_behaviour_is_unchanged_without_a_run_or_after_a_completed_one(
    settings: Settings,
) -> None:
    with _client(settings) as client:
        first = client.post("/index/build")
        second = client.post("/index/build")
        third = client.post("/index/build", json={"restart": True})

    assert [r.status_code for r in (first, second, third)] == [200, 200, 200]
    assert set(first.json()) == {
        "run_id",
        "indexed",
        "skipped",
        "failed",
        "elapsed_seconds",
        "failures",
    }
    assert first.json()["indexed"] == 5


def test_the_lock_held_409_is_still_distinguishable(settings: Settings) -> None:
    _cancelled(settings)
    with _client(settings) as client:
        with FileIndexLock(settings.data_dir / "index.lock").hold():
            held = client.post("/index/build")
        refused = client.post("/index/build")

    assert held.status_code == refused.status_code == 409
    assert held.json()["category"] == "index_lock_held"
    assert refused.json()["category"] == "resume_available"


def test_a_non_boolean_restart_is_a_validation_error(settings: Settings) -> None:
    with _client(settings) as client:
        response = client.post("/index/build", json={"restart": "later"})

    assert response.status_code == 400
    assert response.json()["category"] == "validation_error"
