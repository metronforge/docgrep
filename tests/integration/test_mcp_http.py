"""MCP over streamable HTTP, mounted at /mcp of the FastAPI app: a real
`docgrep serve` process with the SDK's HTTP client, parity with /search,
transport security, the setting that turns the mount off, and the fallback
without the SDK (UC-1, UC-2, UC-7)."""

import logging
import sys
from pathlib import Path
from typing import Any

import anyio
import httpx
import pytest
from fastapi.testclient import TestClient

from docgrep.api.container import Services, create_container
from docgrep.api.http import create_app
from docgrep.api.mcp import RESULT_BUDGET_CHARS, create_mcp_server
from docgrep.settings import Settings
from mcp_support import (
    DUPLICATE,
    QUERY,
    Serve,
    built_library,
    http_session,
    run,
    structured,
    text,
    without_timing,
)
from resume_support import close

_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1"},
    },
}
_MCP_HEADERS = {
    "accept": "application/json, text/event-stream",
    "content-type": "application/json",
}
_SEARCHES: tuple[dict[str, Any], ...] = (
    {"query": QUERY},
    {"query": QUERY, "top_n": 2},
    {"query": DUPLICATE, "top_n": 3},
    {"query": DUPLICATE, "top_n": 3, "collapse_duplicates": False},
)


def test_serve_answers_mcp_and_rest_with_the_same_results(tmp_path: Path) -> None:
    built_library(tmp_path)
    seen: dict[str, Any] = {}

    with Serve(tmp_path) as server:

        async def work() -> None:
            async with http_session(f"{server.base}/mcp") as session:
                seen["tools"] = [t.name for t in (await session.list_tools()).tools]
                seen["finds"] = [
                    await session.call_tool("find_passages", arguments)
                    for arguments in _SEARCHES
                ]
                seen["again"] = await session.call_tool("find_passages", _SEARCHES[0])
                seen["status"] = await session.call_tool("get_index_status", {})
                seen["above_cap"] = await session.call_tool(
                    "find_passages", {"query": QUERY, "top_n": 26}
                )

        run(work)
        rest = [
            httpx.post(f"{server.base}/search", json=arguments, timeout=30)
            for arguments in _SEARCHES
        ]

    assert seen["tools"] == [
        "find_passages",
        "get_index_status",
        "browse_sources",
        "expand_passage_context",
    ]
    for result, response in zip(seen["finds"], rest, strict=True):
        assert not result.isError and response.status_code == 200
        assert without_timing(structured(result)) == without_timing(response.json())
    assert without_timing(structured(seen["again"])) == without_timing(
        structured(seen["finds"][0])
    )
    top = structured(seen["finds"][0])["results"][0]
    assert (top["source"]["path"], top["passage"]["page_number"]) == (
        "memo.docx",
        None,
    )
    sources = structured(seen["status"])["sources"]
    assert sources["by_state"]["failed"] == 1
    assert sources["failed"][0]["path"] == "broken.epub"
    assert str(tmp_path) not in text(seen["status"])
    assert seen["above_cap"].isError
    assert structured(seen["above_cap"])["category"] == "validation_error"


def test_a_foreign_host_is_refused_by_the_app_and_by_the_sdk(tmp_path: Path) -> None:
    settings = Settings(corpus_root=tmp_path, data_dir=tmp_path / "d", backend="fake")
    foreign = {**_MCP_HEADERS, "host": "attacker.example"}

    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        refused = client.post("/mcp", json=_INITIALIZE, headers=foreign)
        accepted = client.post("/mcp", json=_INITIALIZE, headers=_MCP_HEADERS)
    server = create_mcp_server(settings, services=_no_services)
    with TestClient(server.streamable_http_app(), base_url="http://127.0.0.1") as sdk:
        sdk_refused = sdk.post("/mcp", json=_INITIALIZE, headers=foreign)
        sdk_origin = sdk.post(
            "/mcp",
            json=_INITIALIZE,
            headers={**_MCP_HEADERS, "origin": "http://attacker.example"},
        )
        sdk_accepted = sdk.post("/mcp", json=_INITIALIZE, headers=_MCP_HEADERS)

    assert refused.status_code == 403
    assert refused.json()["category"] == "host_not_allowed"
    assert accepted.status_code == 200
    assert sdk_refused.status_code == 421
    assert sdk_origin.status_code == 403
    assert sdk_accepted.status_code == 200
    assert not (tmp_path / "d").exists()


def test_the_configured_bind_host_is_allowed(tmp_path: Path) -> None:
    settings = Settings(
        corpus_root=tmp_path,
        data_dir=tmp_path / "d",
        backend="fake",
        host="0.0.0.0",
        allowed_hosts=("docgrep.internal",),
    )
    server = create_mcp_server(settings, services=_no_services)

    with TestClient(server.streamable_http_app(), base_url="http://127.0.0.1") as sdk:
        statuses = [
            sdk.post(
                "/mcp", json=_INITIALIZE, headers={**_MCP_HEADERS, "host": host}
            ).status_code
            for host in ("0.0.0.0:8000", "docgrep.internal", "localhost:9", "evil")
        ]

    assert statuses == [200, 200, 200, 421]


def test_the_mount_can_be_turned_off(tmp_path: Path) -> None:
    settings = Settings(
        corpus_root=tmp_path,
        data_dir=tmp_path / "d",
        backend="fake",
        mcp_http_enabled=False,
    )

    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        response = client.post("/mcp", json=_INITIALIZE, headers=_MCP_HEADERS)
        health = client.get("/health")

    assert response.status_code == 404
    assert health.status_code == 200


def test_the_mount_survives_a_second_lifespan(tmp_path: Path) -> None:
    settings = Settings(corpus_root=tmp_path, data_dir=tmp_path / "d", backend="fake")
    app = create_app(settings)

    for _ in range(2):
        with TestClient(app, base_url="http://127.0.0.1") as client:
            response = client.post("/mcp", json=_INITIALIZE, headers=_MCP_HEADERS)
            assert response.status_code == 200
    with TestClient(app, base_url="http://127.0.0.1") as client:
        pass
    outside = TestClient(app, base_url="http://127.0.0.1").post(
        "/mcp", json=_INITIALIZE, headers=_MCP_HEADERS
    )
    assert outside.status_code == 503


def test_serve_starts_without_the_sdk_and_warns_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    for name in [n for n in sys.modules if n == "mcp" or n.startswith("mcp.")]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "mcp", None)
    settings = Settings(corpus_root=tmp_path, data_dir=tmp_path / "d", backend="fake")
    app = create_app(settings)

    with caplog.at_level(logging.WARNING, logger="docgrep"):
        with TestClient(app, base_url="http://127.0.0.1") as client:
            mcp = client.post("/mcp", json=_INITIALIZE, headers=_MCP_HEADERS)
            health = client.get("/health")

    assert mcp.status_code == 404 and health.status_code == 200
    warnings = [r for r in caplog.records if r.getMessage() == "mcp_http_unavailable"]
    assert len(warnings) == 1


def _no_services() -> Services:
    raise AssertionError("this tool must not open the composition root")


def test_status_never_opens_the_models_or_the_search_store(tmp_path: Path) -> None:
    settings = built_library(tmp_path)
    server = create_mcp_server(settings, services=_no_services)

    result: Any = anyio.run(server.call_tool, "get_index_status", {})

    assert not result.isError
    assert structured(result)["sources"]["total"] == 5


# Document N holds the first N + 1 terms of the query, so the lexical reranker
# of the fake backend gives every document a different score and the order of
# the results does not rest on ties (#42).
_LONG_TERMS = tuple(["heron", *(f"plume{index:02d}" for index in range(1, 30))])
_LONG_QUERY = " ".join(_LONG_TERMS)


def _long_library(root: Path) -> Settings:
    corpus = root / "corpus"
    corpus.mkdir(parents=True)
    words = " ".join(f"word{index:05d}" for index in range(1200))
    for number in range(30):
        terms = " ".join(_LONG_TERMS[: number + 1])
        (corpus / f"long{number:02d}.txt").write_text(
            f"{terms} marker{number} {words}", encoding="utf-8"
        )
    return Settings(
        corpus_root=corpus,
        data_dir=root / "data",
        backend="fake",
        chunk_size=1500,
        chunk_overlap=8,
        top_n_max=50,
    )


def test_a_large_result_drops_whole_passages_to_stay_within_the_budget(
    tmp_path: Path,
) -> None:
    settings = _long_library(tmp_path)
    services = create_container(settings)
    services.build.run()
    try:
        server = create_mcp_server(settings, services=lambda: services)
        arguments = {"query": _LONG_QUERY, "top_n": 25}
        result: Any = anyio.run(server.call_tool, "find_passages", arguments)
        with TestClient(
            create_app(settings, services=services), base_url="http://127.0.0.1"
        ) as client:
            rest = client.post("/search", json=arguments).json()
    finally:
        close(services)

    body = structured(result)
    kept = len(body["results"])
    assert 1 <= kept < 25 == len(rest["results"])
    by_id = {item["passage"]["passage_id"]: item for item in rest["results"]}
    kept_ids = {item["passage"]["passage_id"] for item in body["results"]}
    assert len(by_id) == 25 and len(kept_ids) == kept
    assert kept_ids <= by_id.keys()
    for item in body["results"]:
        assert item == by_id[item["passage"]["passage_id"]]
    scores = [item["score"] for item in rest["results"]]
    assert scores == sorted(set(scores), reverse=True)
    assert body["results"] == rest["results"][:kept]
    assert "left out" in body["hint"]
    rendered = text(result)
    assert len(rendered) + len(str(body)) < RESULT_BUDGET_CHARS * 1.1
    for item in body["results"]:
        assert item["passage"]["text"] in rendered


def test_tool_errors_carry_no_absolute_paths(tmp_path: Path) -> None:
    settings = Settings(
        corpus_root=tmp_path / "corpus", data_dir=tmp_path / "data", backend="fake"
    )
    (tmp_path / "corpus").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "manifest.sqlite3").write_bytes(b"not a database")
    (tmp_path / "data" / "qdrant").mkdir()

    def broken() -> Services:
        return create_container(settings)

    server = create_mcp_server(settings, services=broken)
    result: Any = anyio.run(server.call_tool, "find_passages", {"query": QUERY})

    assert result.isError
    assert str(tmp_path) not in text(result)
    assert "Traceback" not in text(result)
