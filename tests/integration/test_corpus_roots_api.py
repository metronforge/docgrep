"""Corpus roots over HTTP and MCP: /info, the roots filter and the root of
each source, and the roots of the status (ADR-0023)."""

import os
from pathlib import Path
from typing import Any

import anyio
import jsonschema
import pytest
from fastapi.testclient import TestClient

from docgrep import __version__
from docgrep.api.container import Services, create_container
from docgrep.api.http import create_app
from docgrep.api.mcp import create_mcp_server
from mcp_support import structured
from resume_support import close
from roots_support import save_roots, settings, two_roots

QUERY = "kite comet moss bread"


@pytest.fixture
def indexed(tmp_path: Path) -> tuple[Path, Services]:
    books, notes = two_roots(tmp_path)
    data = tmp_path / "data"
    save_roots(data, ("books", books, True), ("notes", notes, False))
    services = create_container(settings(data))
    services.build.run()
    yield data, services
    close(services)


def _client(data: Path, services: Services | None = None) -> TestClient:
    return TestClient(
        create_app(settings(data), services=services), base_url="http://127.0.0.1"
    )


def _call(data: Path, services: Services, tool: str, arguments: dict[str, Any]) -> Any:
    server = create_mcp_server(settings(data), services=lambda: services)
    return anyio.run(server.call_tool, tool, arguments)


def test_info_keeps_its_keys_and_lists_the_configured_roots(tmp_path: Path) -> None:
    books, notes = two_roots(tmp_path)
    data = tmp_path / "data"
    save_roots(data, ("books", books, True), ("notes", notes, False))
    (notes / "a.txt").unlink()
    (notes / "c.txt").unlink()
    notes.rmdir()

    with _client(data) as client:
        body = client.get("/info").json()

    assert body == {
        "version": __version__,
        "backend": "fake",
        "roots": [
            {
                "name": "books",
                "path": str(books.resolve()),
                "legacy": True,
                "readable": True,
            },
            {
                "name": "notes",
                "path": str(notes.resolve()),
                "legacy": False,
                "readable": False,
            },
        ],
        "rerank_pool": 100,
    }
    assert not (data / "manifest.sqlite3").exists()


def test_info_without_roots_and_with_the_deprecated_variable(tmp_path: Path) -> None:
    books, _ = two_roots(tmp_path)
    with _client(tmp_path / "data") as client:
        assert client.get("/info").json()["roots"] == []
    app = create_app(settings(tmp_path / "data", corpus_root=books))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        roots = client.get("/info").json()["roots"]
    assert roots == [
        {
            "name": "books",
            "path": str(books.resolve()),
            "legacy": True,
            "readable": True,
        }
    ]


def test_the_openapi_schema_describes_info_and_the_filter(tmp_path: Path) -> None:
    with _client(tmp_path / "data") as client:
        schema = client.get("/openapi.json").json()
    components = schema["components"]["schemas"]
    info = schema["paths"]["/info"]["get"]["responses"]["200"]["content"]
    assert info["application/json"]["schema"] == {
        "$ref": "#/components/schemas/InfoResponse"
    }
    assert set(components["InfoResponse"]["properties"]) == {
        "version",
        "backend",
        "roots",
        "rerank_pool",
    }
    assert set(components["InfoRootResponse"]["properties"]) == {
        "name",
        "path",
        "legacy",
        "readable",
    }
    assert "roots" in components["BrowseSourcesRequest"]["properties"]


def test_search_and_browse_carry_and_filter_roots(
    indexed: tuple[Path, Services],
) -> None:
    data, services = indexed
    with _client(data, services) as client:
        everything = client.post("/search", json={"query": QUERY, "top_n": 10}).json()
        notes = client.post(
            "/search", json={"query": QUERY, "top_n": 10, "roots": ["notes"]}
        ).json()
        combined = client.post(
            "/search",
            json={"query": QUERY, "roots": ["books", "notes"], "file_types": ["txt"]},
        ).json()
        browse = client.post("/sources/browse", json={"roots": ["books"]}).json()
        unknown = client.post("/search", json={"query": QUERY, "roots": ["music"]})

    roots = {
        r["source"]["source_id"]: r["source"]["root"] for r in everything["results"]
    }
    assert roots == {
        source_id: "notes" if source_id.startswith("notes/") else "books"
        for source_id in roots
    }
    assert {"a.txt", "notes/a.txt"} <= set(roots)
    assert notes["applied_filters"] == {"roots": ["notes"]}
    assert {r["source"]["root"] for r in notes["results"]} == {"notes"}
    assert combined["applied_filters"] == {
        "file_types": ["txt"],
        "roots": ["books", "notes"],
    }
    assert {r["source"]["source_id"] for r in combined["results"]} <= {
        "a.txt",
        "notes/a.txt",
        "notes/c.txt",
    }
    assert [(s["path"], s["root"]) for s in browse["sources"]] == [
        ("a.txt", "books"),
        ("shelf/b.md", "books"),
    ]
    assert browse["applied_filters"] == {"roots": ["books"]}
    assert unknown.status_code == 400
    assert unknown.json()["category"] == "validation_error"
    assert (
        unknown.json()["message"] == "unknown root 'music'; known roots: books, notes"
    )


def test_mcp_tools_take_the_filter_and_match_http(
    indexed: tuple[Path, Services],
) -> None:
    data, services = indexed
    with _client(data, services) as client:
        rest = client.post(
            "/sources/browse", json={"roots": ["notes"], "page_size": 50}
        ).json()
        search = client.post(
            "/search", json={"query": QUERY, "top_n": 5, "roots": ["notes"]}
        ).json()

    browsed = structured(_call(data, services, "browse_sources", {"roots": ["notes"]}))
    found = structured(
        _call(
            data,
            services,
            "find_passages",
            {"query": QUERY, "top_n": 5, "roots": ["notes"]},
        )
    )
    unknown = _call(data, services, "find_passages", {"query": QUERY, "roots": ["x"]})

    server = create_mcp_server(settings(data), services=lambda: services)
    tools = {tool.name: tool for tool in anyio.run(server.list_tools)}
    jsonschema.validate(found, tools["find_passages"].outputSchema)
    jsonschema.validate(browsed, tools["browse_sources"].outputSchema)
    assert {r["source"]["root"] for r in found["results"]} == {"notes"}
    assert browsed["sources"] == rest["sources"]
    assert [r["source"] for r in found["results"]] == [
        r["source"] for r in search["results"]
    ]
    assert unknown.isError
    assert structured(unknown)["category"] == "validation_error"


def test_status_over_mcp_lists_roots_without_paths(
    indexed: tuple[Path, Services], tmp_path: Path
) -> None:
    data, services = indexed
    server = create_mcp_server(settings(data), services=lambda: services)
    tools = {tool.name: tool for tool in anyio.run(server.list_tools)}
    close(services)

    result = anyio.run(server.call_tool, "get_index_status", {})
    body = structured(result)

    assert body["roots"] == [
        {
            "name": "books",
            "legacy": True,
            "configured": True,
            "readable": True,
            "indexed": 2,
            "skipped": 0,
            "failed": 0,
        },
        {
            "name": "notes",
            "legacy": False,
            "configured": True,
            "readable": True,
            "indexed": 2,
            "skipped": 0,
            "failed": 0,
        },
    ]
    assert str(tmp_path) not in repr(body)
    jsonschema.validate(body, tools["get_index_status"].outputSchema)
    properties = tools["find_passages"].inputSchema["properties"]
    assert (
        "roots" in properties
        and "roots" in (tools["browse_sources"].inputSchema["properties"])
    )


def test_a_single_legacy_root_answers_as_before(tmp_path: Path) -> None:
    books, _ = two_roots(tmp_path)
    data = tmp_path / "data"
    services = create_container(settings(data, corpus_root=books))
    services.build.run()
    try:
        with _client(data, services) as client:
            search = client.post("/search", json={"query": QUERY}).json()
            browse = client.post("/sources/browse", json={}).json()
            filtered = client.post(
                "/search", json={"query": QUERY, "roots": ["books"]}
            ).json()
    finally:
        close(services)

    # Unfiltered bodies have exactly the fields they had before roots.
    assert search["results"] and browse["sources"]
    assert all("root" not in r["source"] for r in search["results"])
    assert all("root" not in s for s in browse["sources"])
    assert [r["passage"]["passage_id"] for r in filtered["results"]] == [
        r["passage"]["passage_id"] for r in search["results"]
    ]
    assert os.environ.get("DOCGREP_CORPUS_ROOT") is None
