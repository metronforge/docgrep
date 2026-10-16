"""POST /sources/browse over a generated library with the deterministic
backend: the body, validation errors, the OpenAPI schema and error mapping
(UC-08)."""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from docgrep.api.container import Services, create_container
from docgrep.api.http import create_app
from docgrep.domain.errors import StoreUnavailable
from docgrep.domain.sources import BrowseRequest, BrowseResult
from docgrep.settings import Settings
from mcp_support import built_library, settings_for
from resume_support import close


@pytest.fixture
def library(tmp_path: Path) -> Any:
    settings = built_library(tmp_path)
    services = create_container(settings)
    client = TestClient(
        create_app(settings, services=services), base_url="http://127.0.0.1"
    )
    yield client
    close(services)


def _paths(body: dict[str, Any]) -> list[str]:
    return [item["path"] for item in body["sources"]]


def test_an_empty_body_lists_every_indexed_source(library: TestClient) -> None:
    response = library.post("/sources/browse", json={})

    assert response.status_code == 200
    body = response.json()
    # broken.epub failed to parse and is not browsable.
    assert _paths(body) == ["memo.docx", "notes/a.txt", "notes/b.txt", "paper.pdf"]
    assert body["total_hits"] == 4
    assert (body["page"], body["page_size"]) == (1, 50)
    assert body["aggregates"] == {
        "total_sources": 4,
        "by_file_type": {"docx": 1, "pdf": 1, "txt": 2},
    }
    assert body["applied_filters"] == {}
    memo = body["sources"][0]
    assert set(memo) == {
        "source_id",
        "path",
        "file_type",
        "title",
        "author",
        "year",
        "passage_count",
        "indexed_at",
        "content_hash",
        "ocr_status",
    }
    assert memo["title"] == "Sea Memo" and memo["ocr_status"] == "not_applicable"
    assert all(item["passage_count"] >= 1 for item in body["sources"])
    assert library.post("/sources/browse").json() == body


def test_filters_sorting_and_paging(library: TestClient) -> None:
    body = library.post(
        "/sources/browse",
        json={
            "file_types": ["TXT", "pdf"],
            "path_globs": ["**/*.txt", "*.pdf"],
            "sort_by": "passage_count",
            "sort_order": "desc",
            "page": 1,
            "page_size": 2,
        },
    ).json()

    assert _paths(body) == ["paper.pdf", "notes/a.txt"]
    assert body["total_hits"] == 3
    assert body["aggregates"]["by_file_type"] == {"pdf": 1, "txt": 2}
    assert body["applied_filters"] == {
        "file_types": ["pdf", "txt"],
        "path_globs": ["**/*.txt", "*.pdf"],
    }

    beyond = library.post("/sources/browse", json={"page": 9}).json()
    assert beyond["sources"] == [] and beyond["total_hits"] == 4

    none = library.post(
        "/sources/browse", json={"authors": ["nobody"], "has_metadata": "with_author"}
    ).json()
    assert none["sources"] == [] and none["total_hits"] == 0
    assert none["applied_filters"] == {
        "authors": ["nobody"],
        "has_metadata": "with_author",
    }


@pytest.mark.parametrize(
    ("body", "location", "fragment"),
    [
        ({"query": "x"}, ["body", "query"], "Extra inputs are not permitted"),
        ({"page": 0}, ["body", "page"], "greater than or equal to 1"),
        ({"page_size": 501}, ["body", "page_size"], "less than or equal to 500"),
        ({"sort_by": "relevance"}, ["body", "sort_by"], "'path'"),
        ({"has_metadata": "some"}, ["body", "has_metadata"], "'without_any'"),
        ({"file_types": ["ocr_pdf"]}, ["body", "file_types"], "allowed: djvu,"),
        ({"authors": [""]}, ["body", "authors"], "authors must not contain empty"),
        ({"year_from": 2000, "year_to": 1990}, ["body"], "year_from must not be"),
    ],
)
def test_invalid_requests_are_validation_errors(
    library: TestClient, body: dict[str, Any], location: list[str], fragment: str
) -> None:
    response = library.post("/sources/browse", json=body)

    assert response.status_code == 400
    error = response.json()
    assert error["category"] == "validation_error"
    (problem,) = error["details"]["errors"]
    assert problem["location"] == location
    assert fragment in problem["message"]


def test_the_endpoint_is_in_the_openapi_schema_with_its_models(
    library: TestClient,
) -> None:
    schema = library.get("/openapi.json").json()

    operation = schema["paths"]["/sources/browse"]["post"]
    ok = operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert ok == {"$ref": "#/components/schemas/BrowseSourcesResponse"}
    components = schema["components"]["schemas"]
    assert set(components["BrowseSourcesResponse"]["required"]) == {
        "sources",
        "total_hits",
        "page",
        "page_size",
        "aggregates",
        "applied_filters",
    }
    assert components["BrowseSourcesRequest"]["additionalProperties"] is False
    assert "BrowsedSourceResponse" in components


def test_a_data_directory_without_an_index_gives_an_empty_listing(
    tmp_path: Path,
) -> None:
    (tmp_path / "corpus").mkdir()
    client = TestClient(create_app(settings_for(tmp_path)), base_url="http://127.0.0.1")

    body = client.post("/sources/browse", json={"page": 2}).json()

    assert body["sources"] == [] and body["total_hits"] == 0
    assert body["aggregates"] == {"total_sources": 0, "by_file_type": {}}
    assert not (tmp_path / "data").exists()


class _DownBrowse:
    def run(self, request: BrowseRequest) -> BrowseResult:
        raise StoreUnavailable("Qdrant operation failed", details={"reason": "down"})


def test_a_store_failure_maps_to_503(tmp_path: Path) -> None:
    settings = built_library(tmp_path)
    services = create_container(settings)
    close(services)
    broken = Services(
        search=services.search, build=services.build, browse=_DownBrowse()
    )
    client = TestClient(
        create_app(settings, services=broken), base_url="http://127.0.0.1"
    )

    response = client.post("/sources/browse", json={})

    assert response.status_code == 503
    assert response.json() == {
        "category": "store_unavailable",
        "message": "Qdrant operation failed",
        "details": {"reason": "down"},
    }


def test_a_server_store_that_is_down_is_unavailable(tmp_path: Path) -> None:
    built_library(tmp_path)
    settings = Settings(
        corpus_root=tmp_path / "corpus",
        data_dir=tmp_path / "data",
        backend="fake",
        chunk_size=64,
        chunk_overlap=8,
        # Nothing listens on the discard port.
        qdrant_url="http://127.0.0.1:9",
    )
    client = TestClient(create_app(settings), base_url="http://127.0.0.1")

    response = client.post("/sources/browse", json={})

    assert response.status_code == 503
    assert response.json()["category"] == "store_unavailable"
