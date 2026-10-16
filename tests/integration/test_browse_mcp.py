"""The browse_sources MCP tool: parity with POST /sources/browse, its
schemas, the page size cap, argument errors and the result budget (UC-08)."""

from pathlib import Path
from typing import Any

import anyio
import httpx
import jsonschema
from fastapi.testclient import TestClient

from docgrep.api.container import create_container
from docgrep.api.http import create_app
from docgrep.api.mcp import RESULT_BUDGET_CHARS, create_mcp_server
from docgrep.settings import MCP_MAX_PAGE_SIZE, Settings
from mcp_support import (
    Serve,
    built_library,
    http_session,
    run,
    stdio_session,
    structured,
    text,
)
from resume_support import close

_REQUESTS: tuple[dict[str, Any], ...] = (
    {},
    {"file_types": ["txt"], "sort_by": "path", "sort_order": "desc"},
    {"path_prefixes": ["notes/"], "page": 2, "page_size": 1},
    {"sort_by": "passage_count", "sort_order": "desc", "page_size": 3},
    {"has_metadata": "with_author", "authors": ["PYTHON"]},
    {"page": 7},
)


def test_the_tool_and_the_endpoint_return_the_same_body(tmp_path: Path) -> None:
    built_library(tmp_path)
    seen: list[Any] = []

    with Serve(tmp_path) as server:

        async def work() -> None:
            async with http_session(f"{server.base}/mcp") as session:
                for arguments in _REQUESTS:
                    seen.append(await session.call_tool("browse_sources", arguments))

        run(work)
        rest = [
            httpx.post(f"{server.base}/sources/browse", json=arguments, timeout=30)
            for arguments in _REQUESTS
        ]

    for result, response in zip(seen, rest, strict=True):
        assert not result.isError and response.status_code == 200
        assert structured(result) == response.json()
    assert [item["path"] for item in structured(seen[2])["sources"]] == ["notes/b.txt"]
    assert structured(seen[5])["sources"] == []


def _stdio(root: Path, calls: dict[str, dict[str, Any]], **env: str) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    async def work() -> None:
        async with stdio_session(root, **env) as session:
            tools = (await session.list_tools()).tools
            seen["tool"] = next(t for t in tools if t.name == "browse_sources")
            for name, arguments in calls.items():
                seen[name] = await session.call_tool("browse_sources", arguments)

    run(work)
    return seen


def test_the_tool_is_read_only_and_describes_its_purpose(tmp_path: Path) -> None:
    built_library(tmp_path)

    seen = _stdio(tmp_path, {})

    tool = seen["tool"]
    annotations = tool.annotations
    assert annotations.readOnlyHint is True and annotations.destructiveHint is False
    assert annotations.idempotentHint is True and annotations.openWorldHint is False
    description = tool.description or ""
    assert "list and count indexed sources" in description.lower()
    assert "whether a book is indexed" in description
    assert "which authors and years exist" in description
    assert "Read-only" in description
    properties = tool.inputSchema["properties"]
    assert set(properties) == {
        "source_ids",
        "authors",
        "year_from",
        "year_to",
        "file_types",
        "path_prefixes",
        "path_globs",
        "has_metadata",
        "page",
        "page_size",
        "sort_by",
        "sort_order",
    }
    assert "required" not in tool.inputSchema
    assert properties["page_size"]["maximum"] == MCP_MAX_PAGE_SIZE
    assert properties["page_size"]["default"] == 50
    assert properties["sort_by"]["enum"] == [
        "path",
        "indexed_at",
        "passage_count",
        "year",
        "title",
    ]


def test_success_and_error_results_pass_the_output_schema(tmp_path: Path) -> None:
    built_library(tmp_path)

    seen = _stdio(
        tmp_path,
        {
            "ok": {"file_types": ["pdf"]},
            "above_cap": {"page_size": MCP_MAX_PAGE_SIZE + 1},
            "unknown": {"query": "x", "page": 1},
            "malformed": {"page": "first", "authors": "Arnold"},
            "semantic": {"file_types": ["ocr_pdf"]},
            "years": {"year_from": 2000, "year_to": 1999},
        },
    )

    schema = seen["tool"].outputSchema
    assert schema is not None and schema["type"] == "object"
    jsonschema.Draft202012Validator.check_schema(schema)
    assert {"$ref": "#/$defs/BrowseSourcesResponse"} in schema["anyOf"]
    assert {"$ref": "#/$defs/ToolErrorBody"} in schema["anyOf"]
    assert not seen["ok"].isError
    assert [s["path"] for s in structured(seen["ok"])["sources"]] == ["paper.pdf"]
    assert "paper.pdf | pdf" in text(seen["ok"])
    for name in ("ok", "above_cap", "unknown", "malformed", "semantic", "years"):
        jsonschema.validate(structured(seen[name]), schema)
    assert structured(seen["above_cap"]) == {
        "category": "validation_error",
        "message": f"page_size must be between 1 and {MCP_MAX_PAGE_SIZE}",
        "details": {"minimum": 1, "maximum": MCP_MAX_PAGE_SIZE},
    }
    assert structured(seen["unknown"])["details"] == {
        "errors": [{"location": ["query"], "message": "Extra inputs are not permitted"}]
    }
    locations = [
        problem["location"]
        for problem in structured(seen["malformed"])["details"]["errors"]
    ]
    assert locations == [["authors"], ["page"]]
    (problem,) = structured(seen["semantic"])["details"]["errors"]
    assert problem["location"] == ["file_types"]
    assert "allowed: djvu, docx, epub, markdown, pdf, txt" in problem["message"]
    (problem,) = structured(seen["years"])["details"]["errors"]
    assert "year_from must not be after year_to" in problem["message"]
    for name in ("above_cap", "unknown", "malformed", "semantic", "years"):
        assert seen[name].isError
        assert structured(seen[name])["category"] == "validation_error"


def test_the_page_size_cap_follows_the_setting(tmp_path: Path) -> None:
    built_library(tmp_path)

    seen = _stdio(
        tmp_path,
        {"three": {"page_size": 3}, "four": {"page_size": 4}, "default": {}},
        DOCGREP_MCP_MAX_PAGE_SIZE="3",
    )

    properties = seen["tool"].inputSchema["properties"]
    assert (properties["page_size"]["maximum"], properties["page_size"]["default"]) == (
        3,
        3,
    )
    assert len(structured(seen["three"])["sources"]) == 3
    assert structured(seen["four"])["details"] == {"minimum": 1, "maximum": 3}
    assert structured(seen["default"])["page_size"] == 3


def test_a_missing_index_lists_nothing_and_creates_nothing(tmp_path: Path) -> None:
    (tmp_path / "corpus").mkdir()

    seen = _stdio(tmp_path, {"empty": {}})

    assert not seen["empty"].isError
    body = structured(seen["empty"])
    assert body["sources"] == [] and body["total_hits"] == 0
    assert not (tmp_path / "data").exists()


def _long_names(root: Path) -> Settings:
    corpus = root / "corpus"
    corpus.mkdir(parents=True)
    for number in range(100):
        name = f"{number:03d} " + "a rather long descriptive file name " * 6
        (corpus / f"{name.strip()}.txt").write_text(f"heron {number}", encoding="utf-8")
    return Settings(
        corpus_root=corpus,
        data_dir=root / "data",
        backend="fake",
        chunk_size=64,
        chunk_overlap=8,
    )


def test_a_large_page_drops_its_last_sources_to_stay_within_the_budget(
    tmp_path: Path,
) -> None:
    settings = _long_names(tmp_path)
    services = create_container(settings)
    services.build.run()
    try:
        server = create_mcp_server(settings, services=lambda: services)
        result: Any = anyio.run(server.call_tool, "browse_sources", {"page_size": 100})
        with TestClient(
            create_app(settings, services=services), base_url="http://127.0.0.1"
        ) as client:
            rest = client.post("/sources/browse", json={"page_size": 100}).json()
    finally:
        close(services)

    body = structured(result)
    kept = len(body["sources"])
    assert 1 <= kept < 100 == len(rest["sources"])
    assert body["sources"] == rest["sources"][:kept]
    assert body["total_hits"] == rest["total_hits"] == 100
    assert body["page_size"] == 100
    assert f"the last {100 - kept} sources of this page were left out" in body["hint"]
    assert "hint" not in rest
    rendered = text(result)
    assert len(rendered) + len(str(body)) < RESULT_BUDGET_CHARS * 1.1
