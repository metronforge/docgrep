"""HTTP transport over local storage and deterministic models."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from docgrep.api.container import Services
from docgrep.api.http import create_app
from docgrep.domain.errors import (
    ConfigDrift,
    IndexIncomplete,
    IndexLockHeld,
    ManifestUnavailable,
    ModelUnavailable,
    ParseError,
    StoreUnavailable,
    ValidationError,
)
from docgrep.domain.models import SearchResult
from docgrep.settings import Settings
from docgrep.usecases.build_index import BuildSummary


def _settings(tmp_path: Path) -> Settings:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    return Settings(
        corpus_root=corpus,
        data_dir=tmp_path / "data",
        backend="fake",
        chunk_size=10,
        chunk_overlap=2,
    )


def test_build_and_search_through_http(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    (settings.corpus_root / "note.txt").write_text(
        "A rare blue comet crosses the sky.", encoding="utf-8"
    )
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        assert client.get("/health").json() == {"status": "ok"}
        info = client.get("/info").json()
        assert info["backend"] == "fake"
        assert info["version"]
        assert client.get("/search").status_code == 405

        before = client.post("/search", json={"query": "blue comet"})
        assert before.status_code == 409
        assert before.json()["category"] == "index_incomplete"

        invalid = client.post("/search", json={"query": " "})
        assert invalid.status_code == 400
        assert invalid.json()["category"] == "validation_error"

        built = client.post("/index/build")
        assert built.status_code == 200
        assert built.json()["indexed"] == 1
        assert built.json()["failed"] == 0

        response = client.post("/search", json={"query": "blue comet", "top_n": 1})
        assert response.status_code == 200
        result = response.json()
        assert result["total_hits"] == 1
        assert result["results"][0]["passage"]["text"] == (
            "A rare blue comet crosses the sky."
        )
        assert result["results"][0]["source"]["path"] == "note.txt"
        assert result["applied_config"]["embedding_model"] == ("deterministic-hash-v1")


class FailingSearch:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def run(self, query: str, *, top_n: int | None = None) -> SearchResult:
        raise self.error


class FailingBuild:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def run(self) -> BuildSummary:
        raise self.error


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (ValidationError("bad input", details={"field": "query"}), 400),
        (ParseError("bad document"), 400),
        (IndexLockHeld("busy"), 409),
        (ConfigDrift("drift"), 409),
        (IndexIncomplete("incomplete"), 409),
        (ModelUnavailable("model missing"), 503),
        (StoreUnavailable("store missing"), 503),
        (ManifestUnavailable("manifest missing"), 503),
    ],
)
def test_domain_errors_have_structured_http_mapping(
    tmp_path: Path, error: Exception, status: int
) -> None:
    services = Services(search=FailingSearch(error), build=FailingBuild(error))
    with TestClient(
        create_app(_settings(tmp_path), services=services), base_url="http://127.0.0.1"
    ) as client:
        for path, body in (("/search", {"query": "valid"}), ("/index/build", {})):
            response = client.post(path, json=body)
            assert response.status_code == status
            assert response.json() == {
                "category": error.category,
                "message": str(error),
                "details": error.details,
            }


def test_invalid_json_body_is_a_structured_400(tmp_path: Path) -> None:
    with TestClient(
        create_app(_settings(tmp_path)), base_url="http://127.0.0.1"
    ) as client:
        response = client.post("/search", json={"top_n": "wrong"})
    assert response.status_code == 400
    assert response.json()["category"] == "validation_error"
