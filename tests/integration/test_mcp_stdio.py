"""`docgrep mcp` as a real subprocess, driven by the MCP SDK's stdio client
with the deterministic backend (UC-1, UC-2, UC-7)."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from docgrep.api.cli import main
from docgrep.settings import MCP_DEFAULT_TOP_N, MCP_MAX_TOP_N
from mcp_support import (
    DUPLICATE,
    QUERY,
    built_library,
    environment,
    run,
    snapshot,
    status_json,
    stdio_session,
    structured,
    text,
    without_timing,
)


def _status_and_search(root: Path) -> dict[str, Any]:
    """One session: list the tools, then call each one; status comes first,
    before a search opens the local store in the server process."""
    seen: dict[str, Any] = {}

    async def work() -> None:
        async with stdio_session(root) as session:
            seen["tools"] = (await session.list_tools()).tools
            seen["status"] = await session.call_tool("get_index_status", {})
            seen["status_again"] = await session.call_tool("get_index_status", {})
            seen["find"] = await session.call_tool("find_passages", {"query": QUERY})
            seen["find_again"] = await session.call_tool(
                "find_passages", {"query": QUERY}
            )
            seen["duplicate"] = await session.call_tool(
                "find_passages", {"query": DUPLICATE, "top_n": 3}
            )
            seen["separate"] = await session.call_tool(
                "find_passages",
                {"query": DUPLICATE, "top_n": 3, "collapse_duplicates": False},
            )

    run(work)
    return seen


def test_tools_are_listed_read_only_with_their_input_schemas(tmp_path: Path) -> None:
    built_library(tmp_path)

    tools = {tool.name: tool for tool in _status_and_search(tmp_path)["tools"]}

    assert sorted(tools) == [
        "browse_sources",
        "expand_passage_context",
        "find_passages",
        "get_index_status",
    ]
    for tool in tools.values():
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True
        assert tool.annotations.destructiveHint is False
        assert tool.annotations.idempotentHint is True
        assert tool.annotations.openWorldHint is False
    find = tools["find_passages"].inputSchema
    assert find["required"] == ["query"]
    assert set(find["properties"]) == {
        "query",
        "top_n",
        "collapse_duplicates",
        # The source filter (UC-02).
        "source_ids",
        "authors",
        "year_from",
        "year_to",
        "file_types",
        "path_prefixes",
        "path_globs",
        "roots",
    }
    assert find["properties"]["top_n"]["default"] == MCP_DEFAULT_TOP_N
    assert find["properties"]["top_n"]["maximum"] == MCP_MAX_TOP_N
    assert find["properties"]["top_n"]["minimum"] == 1
    assert tools["get_index_status"].inputSchema["properties"] == {}
    description = tools["find_passages"].description or ""
    assert "verbatim" in description and "heading path" in description
    assert "never a summary" in description
    assert "DOCX and EPUB passages have no page number" in description


def test_every_tool_answers_and_page_less_passages_stay_page_less(
    tmp_path: Path,
) -> None:
    built_library(tmp_path)

    seen = _status_and_search(tmp_path)

    find = seen["find"]
    assert not find.isError
    body = structured(find)
    top = body["results"][0]
    assert top["source"]["path"] == "memo.docx"
    assert top["source"]["file_type"] == "docx"
    assert top["passage"]["text"] == "The violet kraken lantern rose from the deep sea."
    assert top["passage"]["page_number"] is None
    assert top["passage"]["heading_path"] == ["Expedition", "Storm"]
    pages = {
        r["passage"]["page_number"]
        for r in body["results"]
        if r["source"]["path"] == "paper.pdf"
    }
    assert pages and None not in pages
    rendered = text(find)
    docx_line = next(line for line in rendered.splitlines() if "memo.docx" in line)
    assert "Expedition > Storm" in docx_line and "p." not in docx_line
    assert "The violet kraken lantern rose from the deep sea." in rendered
    assert "paper.pdf | p. " in rendered
    assert without_timing(body) == without_timing(structured(seen["find_again"]))

    collapsed = structured(seen["duplicate"])
    first = collapsed["results"][0]
    assert first["passage"]["text"] == DUPLICATE
    paths = {first["source"]["path"], *(d["path"] for d in first["duplicates"])}
    assert paths == {"notes/a.txt", "notes/b.txt"}
    assert len(first["duplicates"]) == 1
    assert collapsed["collapsed_duplicates"] == 1
    separate = structured(seen["separate"])
    assert "collapsed_duplicates" not in separate
    assert {r["source"]["path"] for r in separate["results"][:2]} == {
        "notes/a.txt",
        "notes/b.txt",
    }


def test_status_reports_the_failed_source_and_matches_the_cli(tmp_path: Path) -> None:
    built_library(tmp_path)

    seen = _status_and_search(tmp_path)
    report = structured(seen["status"])

    assert not seen["status"].isError
    assert report == structured(seen["status_again"])
    assert report == status_json(tmp_path)
    sources = report["sources"]
    assert sources["by_state"] == {"indexed": 4, "skipped": 0, "failed": 1}
    (failed,) = sources["failed"]
    assert (failed["path"], failed["category"]) == ("broken.epub", "parse_error")
    assert failed["reason"] == "EPUB is not a valid ZIP archive"
    assert str(tmp_path) not in json.dumps(report)
    assert "failed: broken.epub: parse_error" in text(seen["status"])


def _errors(root: Path) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    async def work() -> None:
        async with stdio_session(root) as session:
            call = session.call_tool
            seen["above_cap"] = await call(
                "find_passages", {"query": QUERY, "top_n": MCP_MAX_TOP_N + 1}
            )
            seen["zero"] = await call("find_passages", {"query": QUERY, "top_n": 0})
            seen["wrong_type"] = await call(
                "find_passages", {"query": QUERY, "top_n": "many"}
            )
            seen["blank"] = await call("find_passages", {"query": "   "})
            seen["missing"] = await call("find_passages", {})

    run(work)
    return seen


def test_invalid_arguments_are_tool_errors(tmp_path: Path) -> None:
    built_library(tmp_path)

    seen = _errors(tmp_path)

    assert all(result.isError for result in seen.values())
    above = structured(seen["above_cap"])
    assert above == {
        "category": "validation_error",
        "message": f"top_n must be between 1 and {MCP_MAX_TOP_N}",
        "details": {"minimum": 1, "maximum": MCP_MAX_TOP_N},
    }
    assert json.loads(text(seen["above_cap"])) == above
    assert structured(seen["zero"])["category"] == "validation_error"
    assert structured(seen["blank"])["category"] == "validation_error"
    assert "top_n" in text(seen["wrong_type"])
    assert "query" in text(seen["missing"])
    for result in seen.values():
        assert "Traceback" not in text(result)
        assert str(tmp_path) not in text(result)


def _one_search(root: Path) -> Any:
    found: list[Any] = []

    async def work() -> None:
        async with stdio_session(root) as session:
            found.append(await session.call_tool("find_passages", {"query": QUERY}))

    run(work)
    return found[0]


def test_a_missing_index_is_a_tool_error_and_nothing_is_created(
    tmp_path: Path,
) -> None:
    (tmp_path / "corpus").mkdir()

    result = _one_search(tmp_path)

    assert result.isError
    body = structured(result)
    assert body["category"] == "index_incomplete"
    assert "docgrep index" in body["message"]
    assert not (tmp_path / "data").exists()


def test_an_empty_index_is_a_tool_error(tmp_path: Path) -> None:
    from docgrep.api.container import create_container
    from mcp_support import settings_for
    from resume_support import close

    (tmp_path / "corpus").mkdir()
    services = create_container(settings_for(tmp_path))
    services.build.run()
    close(services)

    result = _one_search(tmp_path)

    assert result.isError
    assert structured(result) == {
        "category": "index_empty",
        "message": "The index is empty",
        "details": {"total_hits": 0},
    }


def test_tools_write_nothing(tmp_path: Path) -> None:
    built_library(tmp_path)
    before = snapshot(tmp_path / "corpus", tmp_path / "data")

    _status_and_search(tmp_path)

    assert snapshot(tmp_path / "corpus", tmp_path / "data") == before


def _frames(*messages: dict[str, Any]) -> str:
    return "".join(json.dumps(message) + "\n" for message in messages)


def test_standard_output_carries_only_protocol_frames(tmp_path: Path) -> None:
    built_library(tmp_path)
    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1"},
        },
    }

    requests = [
        initialize,
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "find_passages", "arguments": {"query": QUERY}},
        },
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "get_index_status", "arguments": {}},
        },
    ]
    stderr_path = tmp_path / "stderr.log"
    with stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            [sys.executable, "-m", "docgrep.api.cli", "mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            env=environment(tmp_path),
            cwd=str(tmp_path),
        )
        assert process.stdin is not None and process.stdout is not None
        frames = []
        try:
            for request in requests:
                process.stdin.write(_frames(request).encode())
                process.stdin.flush()
                if "id" in request:
                    line = process.stdout.readline()
                    assert line.endswith(b"\n")
                    frames.append(json.loads(line))
            # Keep the session open until every answer arrived, as a client does.
            process.stdin.close()
            rest = process.stdout.read()
            assert process.wait(timeout=60) == 0
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    assert rest == b""
    assert [frame["id"] for frame in frames] == [1, 2, 3, 4]
    assert all(frame["jsonrpc"] == "2.0" and "result" in frame for frame in frames)
    assert frames[2]["result"]["isError"] is False
    errors = stderr_path.read_text(encoding="utf-8")
    assert "search_completed" in errors
    assert QUERY not in errors


def test_startup_loads_nothing_and_needs_no_index(tmp_path: Path) -> None:
    (tmp_path / "corpus").mkdir()
    listed: list[str] = []

    async def work() -> None:
        async with stdio_session(tmp_path) as session:
            listed.extend(tool.name for tool in (await session.list_tools()).tools)
            status = await session.call_tool("get_index_status", {})
            assert not status.isError
            assert structured(status)["index_state"] == "not_built"

    run(work)

    assert listed == [
        "find_passages",
        "get_index_status",
        "browse_sources",
        "expand_passage_context",
    ]
    assert not (tmp_path / "data").exists()


def test_the_cap_follows_the_setting(tmp_path: Path) -> None:
    built_library(tmp_path)
    seen: dict[str, Any] = {}

    async def work() -> None:
        async with stdio_session(tmp_path, DOCGREP_MCP_MAX_TOP_N="3") as session:
            tools = (await session.list_tools()).tools
            seen["schema"] = next(t for t in tools if t.name == "find_passages")
            seen["four"] = await session.call_tool(
                "find_passages", {"query": QUERY, "top_n": 4}
            )
            seen["default"] = await session.call_tool("find_passages", {"query": QUERY})

    run(work)

    properties = seen["schema"].inputSchema["properties"]
    assert (properties["top_n"]["maximum"], properties["top_n"]["default"]) == (3, 3)
    assert structured(seen["four"])["details"] == {"minimum": 1, "maximum": 3}
    assert len(structured(seen["default"])["results"]) == 3


def test_invalid_settings_exit_2_without_touching_standard_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DOCGREP_CHUNK_SIZE", "0")

    status = main(["mcp"])

    out, err = capsys.readouterr()
    assert status == 2 and out == ""
    assert err.startswith("invalid_settings:")
