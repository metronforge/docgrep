"""The expand_passage_context MCP tool (UC-12): parity with
POST /passages/context, its schemas, the id cap, argument errors and the
result budget."""

from pathlib import Path
from typing import Any

import anyio
import httpx
import jsonschema
from fastapi.testclient import TestClient

from context_support import MISSING, source_passages
from docgrep.api.container import create_container
from docgrep.api.http import create_app
from docgrep.api.mcp import RESULT_BUDGET_CHARS, create_mcp_server
from docgrep.settings import MCP_MAX_CONTEXT_PASSAGES, Settings
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


def _requests(settings: Settings) -> list[dict[str, Any]]:
    paper = source_passages(settings, "paper.pdf")
    memo = source_passages(settings, "memo.docx")
    return [
        {"passage_ids": [paper[1].passage_id]},
        {"passage_ids": [memo[0].passage_id, MISSING], "window_tokens": 0},
        {"passage_ids": [p.passage_id for p in paper[:3]], "window_tokens": 70},
    ]


def test_the_tool_and_the_endpoint_return_the_same_body(tmp_path: Path) -> None:
    settings = built_library(tmp_path)
    requests = _requests(settings)
    seen: list[Any] = []

    with Serve(tmp_path) as server:

        async def work() -> None:
            async with http_session(f"{server.base}/mcp") as session:
                for arguments in requests:
                    seen.append(
                        await session.call_tool("expand_passage_context", arguments)
                    )

        run(work)
        rest = [
            httpx.post(f"{server.base}/passages/context", json=body, timeout=30)
            for body in requests
        ]

    for result, response in zip(seen, rest, strict=True):
        assert not result.isError and response.status_code == 200
        assert structured(result) == response.json()
    assert [item["status"] for item in structured(seen[1])["results"]] == [
        "ok",
        "not_found",
    ]
    rendered = text(seen[0])
    assert "--- before" in rendered and "--- passage ---" in rendered
    assert structured(seen[0])["results"][0]["passage"]["text"] in rendered


def _stdio(root: Path, calls: dict[str, dict[str, Any]], **env: str) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    async def work() -> None:
        async with stdio_session(root, **env) as session:
            tools = (await session.list_tools()).tools
            seen["tool"] = next(t for t in tools if t.name == "expand_passage_context")
            for name, arguments in calls.items():
                seen[name] = await session.call_tool(
                    "expand_passage_context", arguments
                )

    run(work)
    return seen


def test_the_tool_is_read_only_and_says_when_to_use_it(tmp_path: Path) -> None:
    built_library(tmp_path)

    tool = _stdio(tmp_path, {})["tool"]

    annotations = tool.annotations
    assert annotations.readOnlyHint is True and annotations.destructiveHint is False
    assert annotations.idempotentHint is True and annotations.openWorldHint is False
    description = tool.description or ""
    assert "Fetch the surrounding text of a passage you want to verify or quote" in (
        description
    )
    assert "pass passage ids from search results" in description
    assert "Read-only" in description
    schema = tool.inputSchema
    assert set(schema["properties"]) == {"passage_ids", "window_tokens"}
    assert schema["required"] == ["passage_ids"]
    ids = schema["properties"]["passage_ids"]
    assert (ids["minItems"], ids["maxItems"]) == (1, MCP_MAX_CONTEXT_PASSAGES)
    window = schema["properties"]["window_tokens"]
    assert (window["default"], window["minimum"], window["maximum"]) == (500, 0, 5000)


def test_success_and_error_results_pass_the_output_schema(tmp_path: Path) -> None:
    settings = built_library(tmp_path)
    paper = source_passages(settings, "paper.pdf")

    seen = _stdio(
        tmp_path,
        {
            "ok": {"passage_ids": [paper[0].passage_id, MISSING]},
            "above_cap": {"passage_ids": [MISSING] * (MCP_MAX_CONTEXT_PASSAGES + 1)},
            "unknown": {"passage_ids": [MISSING], "window": 3},
            "malformed": {"passage_ids": MISSING, "window_tokens": "wide"},
            "bad_id": {"passage_ids": [MISSING, "abc"]},
            "window": {"passage_ids": [MISSING], "window_tokens": 5001},
        },
    )

    schema = seen["tool"].outputSchema
    assert schema is not None and schema["type"] == "object"
    jsonschema.Draft202012Validator.check_schema(schema)
    assert {"$ref": "#/$defs/PassageContextResponse"} in schema["anyOf"]
    assert {"$ref": "#/$defs/ToolErrorBody"} in schema["anyOf"]
    for name in ("ok", "above_cap", "unknown", "malformed", "bad_id", "window"):
        jsonschema.validate(structured(seen[name]), schema)
    assert not seen["ok"].isError
    assert structured(seen["above_cap"]) == {
        "category": "validation_error",
        "message": f"passage_ids must hold between 1 and {MCP_MAX_CONTEXT_PASSAGES} "
        "ids",
        "details": {"minimum": 1, "maximum": MCP_MAX_CONTEXT_PASSAGES},
    }
    assert structured(seen["unknown"])["details"] == {
        "errors": [
            {"location": ["window"], "message": "Extra inputs are not permitted"}
        ]
    }
    locations = [
        problem["location"]
        for problem in structured(seen["malformed"])["details"]["errors"]
    ]
    assert locations == [["passage_ids"], ["window_tokens"]]
    assert structured(seen["bad_id"])["details"] == {"malformed": ["abc"]}
    assert structured(seen["window"])["details"] == {"minimum": 0, "maximum": 5000}
    for name in ("above_cap", "unknown", "malformed", "bad_id", "window"):
        assert seen[name].isError
        assert structured(seen[name])["category"] == "validation_error"


def test_the_cap_follows_the_setting_within_the_http_maximum(tmp_path: Path) -> None:
    built_library(tmp_path)

    two = _stdio(
        tmp_path,
        {"three": {"passage_ids": [MISSING] * 3}},
        DOCGREP_MCP_MAX_CONTEXT_PASSAGES="2",
    )
    one = _stdio(
        tmp_path,
        {"two": {"passage_ids": [MISSING] * 2}},
        DOCGREP_CONTEXT_MAX_PASSAGES="1",
    )

    assert two["tool"].inputSchema["properties"]["passage_ids"]["maxItems"] == 2
    assert structured(two["three"])["details"] == {"minimum": 1, "maximum": 2}
    assert one["tool"].inputSchema["properties"]["passage_ids"]["maxItems"] == 1
    assert structured(one["two"])["details"] == {"minimum": 1, "maximum": 1}


def test_a_missing_index_finds_nothing_and_creates_nothing(tmp_path: Path) -> None:
    (tmp_path / "corpus").mkdir()

    seen = _stdio(tmp_path, {"empty": {"passage_ids": [MISSING]}})

    assert not seen["empty"].isError
    assert [item["status"] for item in structured(seen["empty"])["results"]] == [
        "not_found"
    ]
    assert not (tmp_path / "data").exists()


def _long_book(root: Path) -> Settings:
    corpus = root / "corpus"
    corpus.mkdir(parents=True)
    words = " ".join(f"word{number}" for number in range(6000))
    (corpus / "book.txt").write_text(words, encoding="utf-8")
    return Settings(
        corpus_root=corpus,
        data_dir=root / "data",
        backend="fake",
        chunk_size=64,
        chunk_overlap=8,
    )


def test_a_large_result_drops_its_last_entries_to_stay_within_the_budget(
    tmp_path: Path,
) -> None:
    settings = _long_book(tmp_path)
    services = create_container(settings)
    services.build.run()
    close(services)
    passages = source_passages(settings, "book.txt")
    ids = [passages[index].passage_id for index in (10, 30, 50, 70, 90)]
    body = {"passage_ids": ids, "window_tokens": 2000}
    services = create_container(settings)
    try:
        server = create_mcp_server(settings, services=lambda: services)
        result: Any = anyio.run(server.call_tool, "expand_passage_context", body)
        with TestClient(
            create_app(settings, services=services), base_url="http://127.0.0.1"
        ) as client:
            rest = client.post("/passages/context", json=body).json()
    finally:
        close(services)

    trimmed = structured(result)
    kept = len(trimmed["results"])
    assert 1 <= kept < 5 == len(rest["results"])
    assert trimmed["results"] == rest["results"][:kept]
    assert f"the last {5 - kept} requested passages were left out" in trimmed["hint"]
    assert "hint" not in rest
    assert len(text(result)) + len(str(trimmed)) < RESULT_BUDGET_CHARS * 1.1
