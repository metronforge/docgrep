"""Filtered search over a generated library with the deterministic backend:
filtering before the first stage in local mode and against a real server
(set DOCGREP_TEST_QDRANT_URL), and the HTTP, MCP and CLI fronts (UC-02)."""

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import jsonschema
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

from docgrep.api.container import create_container
from docgrep.api.http import create_app
from docgrep.settings import Settings
from docgrep.usecases.find_passages import NO_MATCHING_SOURCES_HINT
from mcp_support import (
    Serve,
    environment,
    http_session,
    run,
    stdio_session,
    structured,
    text,
    without_timing,
)
from resume_support import close
from search_filter_support import (
    ARNOLD,
    QUERY,
    TOP_K,
    filter_settings,
    write_filter_library,
)

URL = os.environ.get("DOCGREP_TEST_QDRANT_URL")
ALIAS = "alpha"
# The environment of a docgrep process over the filter library.
_ENV = {"DOCGREP_TOP_K": str(TOP_K), "DOCGREP_TOP_N": str(TOP_K)}
_SERVER = pytest.param(
    "server",
    marks=[
        pytest.mark.qdrant_server,
        pytest.mark.skipif(not URL, reason="DOCGREP_TEST_QDRANT_URL is not set"),
    ],
)


def _wipe(client: QdrantClient) -> None:
    for alias in client.get_aliases().aliases:
        if alias.alias_name == ALIAS:
            client.update_collection_aliases(
                [
                    models.DeleteAliasOperation(
                        delete_alias=models.DeleteAlias(alias_name=ALIAS)
                    )
                ]
            )
    for collection in client.get_collections().collections:
        if collection.name.startswith(f"{ALIAS}_"):
            client.delete_collection(collection.name)


def _built(root: Path, **extra: object) -> Settings:
    write_filter_library(root / "corpus")
    settings = filter_settings(root, **extra)
    services = create_container(settings)
    summary = services.build.run()
    assert (summary.indexed, summary.failed) == (16, 0), summary
    close(services)
    return settings


@pytest.fixture(params=["local", _SERVER])
def mode_settings(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Settings]:
    if request.param == "local":
        yield _built(tmp_path)
        return
    assert URL is not None
    raw = QdrantClient(url=URL)
    _wipe(raw)
    try:
        yield _built(tmp_path, qdrant_url=URL, qdrant_alias=ALIAS)
    finally:
        _wipe(raw)
        raw.close()


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = _built(tmp_path)
    services = create_container(settings)
    yield TestClient(
        create_app(settings, services=services), base_url="http://127.0.0.1"
    )
    close(services)


def _paths(body: dict[str, Any]) -> list[str]:
    return [item["source"]["path"] for item in body["results"]]


def test_filtering_happens_before_the_first_stage(mode_settings: Settings) -> None:
    # The setup relies on the reranker seeing exactly the top_k candidates;
    # the default pool of 1.22.0 is larger (ADR-0026).
    settings = mode_settings.model_copy(update={"rerank_pool": TOP_K})
    services = create_container(settings)
    try:
        api = TestClient(
            create_app(settings, services=services), base_url="http://127.0.0.1"
        )
        unfiltered = api.post("/search", json={"query": QUERY}).json()
        filtered = api.post(
            "/search", json={"query": QUERY, "authors": ["arnold"]}
        ).json()
    finally:
        close(services)

    # The setup: unfiltered, the top_k candidates are all strong decoys, so
    # filtering them afterwards would leave nothing.
    assert unfiltered["total_hits"] == TOP_K
    assert not set(_paths(unfiltered)) & set(ARNOLD)
    # Filtered, the candidates come from Arnold's sources only.
    assert sorted(_paths(filtered)) == sorted(ARNOLD)
    assert filtered["total_hits"] == 2
    assert filtered["applied_filters"] == {"authors": ["arnold"]}
    assert filtered["sources_excluded_for_missing_metadata"] == 2


def test_each_returned_passage_satisfies_the_filters(client: TestClient) -> None:
    cases: list[tuple[dict[str, Any], set[str]]] = [
        ({"source_ids": ["papers/kepler.txt"]}, {"papers/kepler.txt"}),
        ({"authors": ["ARNOLD"]}, set(ARNOLD)),
        ({"year_from": 1980, "year_to": 2001}, {"books/late.md", "decoys/d00.md"}),
        ({"file_types": ["txt"]}, {"papers/kepler.txt"}),
        ({"path_prefixes": ["/books/"]}, set(ARNOLD)),
        (
            {"path_globs": ["notes/*.md", "**/*.txt"]},
            {"notes/plain.md", "papers/kepler.txt"},
        ),
        (
            {"authors": ["arnold", "decoy"], "year_to": 2000},
            set(ARNOLD),
        ),
    ]
    for fields, expected in cases:
        body = client.post("/search", json={"query": QUERY, **fields}).json()
        assert set(_paths(body)) == expected, fields
        for item in body["results"]:
            assert item["source"]["path"] in expected


def test_filter_errors_are_validation_bodies(client: TestClient) -> None:
    bad: list[dict[str, Any]] = [
        {"year_from": 2000, "year_to": 1990},
        {"file_types": ["odt"]},
        {"authors": ["  "]},
        {"year_from": "late"},
        {"author": ["Arnold"]},
    ]
    for fields in bad:
        response = client.post("/search", json={"query": QUERY, **fields})
        assert response.status_code == 400, fields
        body = response.json()
        assert body["category"] == "validation_error"
        assert body["details"]["errors"], fields
    unknown = client.post("/search", json={"query": QUERY, "author": ["Arnold"]})
    assert unknown.json()["details"]["errors"][0]["location"] == ["body", "author"]


def test_a_filter_matching_nothing_is_an_empty_success(client: TestClient) -> None:
    response = client.post("/search", json={"query": QUERY, "authors": ["Tolstoy"]})

    assert response.status_code == 200
    body = response.json()
    assert body["results"] == [] and body["total_hits"] == 0
    assert body["hint"] == NO_MATCHING_SOURCES_HINT
    assert body["applied_filters"] == {"authors": ["Tolstoy"]}
    assert body["sources_excluded_for_missing_metadata"] == 2


def test_an_unfiltered_body_only_gains_empty_applied_filters(
    client: TestClient,
) -> None:
    plain = without_timing(client.post("/search", json={"query": QUERY}).json())
    empty = without_timing(
        client.post(
            "/search", json={"query": QUERY, "authors": [], "year_from": None}
        ).json()
    )

    assert plain["applied_filters"] == {}
    assert "sources_excluded_for_missing_metadata" not in plain
    assert empty == plain


def test_the_openapi_schema_shows_the_filters(client: TestClient) -> None:
    schemas = client.get("/openapi.json").json()["components"]["schemas"]

    request = schemas["SearchRequest"]
    assert set(request["properties"]) == {
        "query",
        "top_n",
        "collapse_duplicates",
        "source_ids",
        "authors",
        "year_from",
        "year_to",
        "file_types",
        "path_prefixes",
        "path_globs",
        "roots",
    }
    assert request["additionalProperties"] is False
    response = schemas["SearchResponse"]["properties"]
    assert {"applied_filters", "sources_excluded_for_missing_metadata"} <= set(response)


_MCP_REQUESTS: tuple[dict[str, Any], ...] = (
    {"query": QUERY, "top_n": TOP_K},
    {"query": QUERY, "top_n": TOP_K, "authors": ["arnold"]},
    {"query": QUERY, "top_n": TOP_K, "file_types": ["TXT"], "path_globs": ["**/*.txt"]},
    {"query": QUERY, "year_from": 1980, "top_n": 2},
    {"query": QUERY, "top_n": TOP_K, "authors": ["Tolstoy"]},
)


def test_the_tool_and_the_endpoint_return_the_same_body(tmp_path: Path) -> None:
    _built(tmp_path)
    seen: list[Any] = []
    # top_n is explicit: the tool's default differs from the endpoint's.

    with Serve(tmp_path, **_ENV) as server:

        async def work() -> None:
            async with http_session(f"{server.base}/mcp") as session:
                schema = next(
                    tool
                    for tool in (await session.list_tools()).tools
                    if tool.name == "find_passages"
                ).outputSchema
                seen.append(schema)
                for arguments in _MCP_REQUESTS:
                    seen.append(await session.call_tool("find_passages", arguments))

        run(work)
        rest = [
            httpx.post(f"{server.base}/search", json=arguments, timeout=30)
            for arguments in _MCP_REQUESTS
        ]

    schema, *results = seen
    for result, response in zip(results, rest, strict=True):
        assert not result.isError and response.status_code == 200
        assert without_timing(structured(result)) == without_timing(response.json())
        jsonschema.validate(structured(result), schema)
    filtered = structured(results[1])
    assert sorted(_paths(filtered)) == sorted(ARNOLD)
    assert "Filters: authors=['arnold']" in text(results[1])
    assert "Sources left out for a missing author or year: 2" in text(results[1])
    assert NO_MATCHING_SOURCES_HINT in text(results[4])


def test_the_tool_describes_and_checks_its_filters(tmp_path: Path) -> None:
    _built(tmp_path)
    seen: dict[str, Any] = {}

    async def work() -> None:
        async with stdio_session(tmp_path, **_ENV) as session:
            tools = {tool.name: tool for tool in (await session.list_tools()).tools}
            seen["tool"] = tools["find_passages"]
            for name, arguments in {
                "bounds": {"query": QUERY, "year_from": 2000, "year_to": 1990},
                "type": {"query": QUERY, "file_types": ["odt"]},
                "unknown": {"query": QUERY, "author": ["Arnold"]},
            }.items():
                seen[name] = await session.call_tool("find_passages", arguments)

    run(work)

    tool = seen["tool"]
    assert tool.description is not None
    assert "browse_sources" in tool.description
    assert "authors" in tool.inputSchema["properties"]
    for name in ("bounds", "type", "unknown"):
        assert seen[name].isError, name
        body = structured(seen[name])
        assert body["category"] == "validation_error", name
        jsonschema.validate(body, tool.outputSchema)
    assert structured(seen["unknown"])["details"]["errors"] == [
        {"location": ["author"], "message": "Extra inputs are not permitted"}
    ]


def _cli(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "docgrep.api.cli", "search", *args],
        env=environment(root, **_ENV),
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_the_cli_takes_the_same_filters(tmp_path: Path) -> None:
    settings = _built(tmp_path)
    services = create_container(settings)
    try:
        api = TestClient(
            create_app(settings, services=services), base_url="http://127.0.0.1"
        )
        expected = api.post(
            "/search",
            json={"query": QUERY, "authors": ["arnold"], "year_to": 1980},
        ).json()
    finally:
        close(services)

    as_json = _cli(tmp_path, QUERY, "--author", "arnold", "--year-to", "1980", "--json")
    assert as_json.returncode == 0, as_json.stderr
    assert without_timing(json.loads(as_json.stdout)) == without_timing(expected)

    done = _cli(
        tmp_path, QUERY, "--author", "arnold", "--author", "Decoy", "--year-to", "1980"
    )
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    assert lines[0] == "Filters: authors=['Decoy', 'arnold']; year_to=1980"
    assert lines[1] == "Sources left out for a missing author or year: 2"
    assert lines[2].startswith("books/arnold.md")

    nothing = _cli(tmp_path, QUERY, "--type", "pdf")
    assert nothing.returncode == 0, nothing.stderr
    assert nothing.stdout.splitlines() == [
        "Filters: file_types=['pdf']",
        NO_MATCHING_SOURCES_HINT,
    ]

    plain = _cli(tmp_path, QUERY)
    assert plain.returncode == 0 and not plain.stdout.startswith("Filters:")

    bad = _cli(tmp_path, QUERY, "--year-from", "2000", "--year-to", "1990")
    assert bad.returncode == 1
    assert bad.stderr.startswith("validation_error")
    assert bad.stdout == ""
