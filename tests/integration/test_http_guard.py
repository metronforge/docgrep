"""Host and Origin checks on the local HTTP transport."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from docgrep.adapters.manifest import SQLiteManifest
from docgrep.api.http import create_app
from docgrep.settings import Settings

EVIL = "evil.example.com"


def _settings(
    tmp_path: Path,
    host: str = "127.0.0.1",
    allowed_hosts: tuple[str, ...] = (),
) -> Settings:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.txt").write_text(
        "A rare blue comet crosses the sky.", encoding="utf-8"
    )
    return Settings(
        corpus_root=corpus,
        data_dir=tmp_path / "data",
        backend="fake",
        chunk_size=10,
        chunk_overlap=2,
        host=host,
        allowed_hosts=allowed_hosts,
    )


def _client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings), base_url="http://127.0.0.1")


@pytest.mark.parametrize(
    ("method", "path"),
    [("POST", "/search"), ("GET", "/info"), ("GET", "/health")],
)
def test_foreign_host_is_rejected_with_the_structured_envelope(
    tmp_path: Path, method: str, path: str
) -> None:
    with _client(_settings(tmp_path)) as client:
        response = client.request(
            method,
            path,
            headers={"Host": EVIL},
            json={"query": "blue comet"} if method == "POST" else None,
        )

    assert response.status_code == 403
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {
        "category": "host_not_allowed",
        "message": "Host header is not allowed",
        "details": {"host": EVIL},
    }


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "127.0.0.1:8000", "localhost", "LOCALHOST:8000", "[::1]:8000"],
)
def test_local_host_names_are_accepted(tmp_path: Path, host: str) -> None:
    with _client(_settings(tmp_path)) as client:
        assert client.get("/info", headers={"Host": host}).status_code == 200


def test_configured_host_and_allowed_hosts_extend_the_list(tmp_path: Path) -> None:
    settings = _settings(tmp_path, host="docgrep.lan", allowed_hosts=("box.internal",))
    with _client(settings) as client:
        for host in ("docgrep.lan:8000", "box.internal", "box.internal:9000"):
            assert client.get("/info", headers={"Host": host}).status_code == 200
        assert client.get("/info", headers={"Host": EVIL}).status_code == 403


def test_foreign_origin_cannot_start_a_build(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _client(settings) as client:
        response = client.post("/index/build", headers={"Origin": f"http://{EVIL}"})

        assert response.status_code == 403
        assert response.json()["category"] == "origin_not_allowed"
        assert response.json()["details"] == {"origin": f"http://{EVIL}"}
        assert SQLiteManifest(settings.data_dir).latest_run() is None

        assert client.post("/index/build").status_code == 200
        first = SQLiteManifest(settings.data_dir).latest_run()
        assert first is not None

        blocked = client.post(
            "/index/build", headers={"Origin": "https://attacker.test:8443"}
        )
        assert blocked.status_code == 403
        assert SQLiteManifest(settings.data_dir).latest_run() == first


def test_opaque_origin_is_rejected(tmp_path: Path) -> None:
    with _client(_settings(tmp_path)) as client:
        response = client.post(
            "/search", headers={"Origin": "null"}, json={"query": "x"}
        )

    assert response.status_code == 403
    assert response.json()["category"] == "origin_not_allowed"


def test_local_requests_work_without_origin_and_with_a_local_origin(
    tmp_path: Path,
) -> None:
    with _client(_settings(tmp_path)) as client:
        assert client.post("/index/build").status_code == 200
        local = {"Origin": "http://127.0.0.1:8000"}
        assert client.post("/index/build", headers=local).status_code == 200

        for headers in ({}, local, {"Origin": "http://localhost:3000"}):
            response = client.post(
                "/search", headers=headers, json={"query": "blue comet"}
            )
            assert response.status_code == 200
            assert response.json()["results"][0]["source"]["path"] == "note.txt"


def test_origin_is_only_enforced_on_unsafe_methods(tmp_path: Path) -> None:
    with _client(_settings(tmp_path)) as client:
        response = client.get("/info", headers={"Origin": f"http://{EVIL}"})

    assert response.status_code == 200
