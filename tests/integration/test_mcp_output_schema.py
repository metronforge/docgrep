"""Declared output schemas of the MCP tools: success and error results both
pass the SDK's validation on the server and in the client (UC-1, UC-7)."""

from pathlib import Path
from typing import Any

import jsonschema

from docgrep.api.container import create_container
from docgrep.settings import MCP_MAX_TOP_N
from mcp_support import QUERY, built_library, run, settings_for, stdio_session
from mcp_support import structured as structured_body
from resume_support import close


def _session(root: Path) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    async def work() -> None:
        async with stdio_session(root) as session:
            seen["tools"] = {t.name: t for t in (await session.list_tools()).tools}
            seen["find"] = await session.call_tool("find_passages", {"query": QUERY})
            seen["status"] = await session.call_tool("get_index_status", {})
            seen["above_cap"] = await session.call_tool(
                "find_passages", {"query": QUERY, "top_n": MCP_MAX_TOP_N + 1}
            )

    run(work)
    return seen


def _one_search(root: Path) -> Any:
    found: list[Any] = []

    async def work() -> None:
        async with stdio_session(root) as session:
            schema = next(
                t
                for t in (await session.list_tools()).tools
                if t.name == "find_passages"
            ).outputSchema
            found.append(
                (schema, await session.call_tool("find_passages", {"query": QUERY}))
            )

    run(work)
    return found[0]


def test_both_tools_declare_an_object_output_schema(tmp_path: Path) -> None:
    built_library(tmp_path)

    tools = _session(tmp_path)["tools"]

    find = tools["find_passages"].outputSchema
    status = tools["get_index_status"].outputSchema
    for schema in (find, status):
        assert schema is not None and schema["type"] == "object"
        jsonschema.Draft202012Validator.check_schema(schema)
        assert {"$ref": "#/$defs/ToolErrorBody"} in schema["anyOf"]
        assert schema["$defs"]["ToolErrorBody"]["required"] == [
            "category",
            "message",
            "details",
        ]
    assert {"$ref": "#/$defs/SearchResponse"} in find["anyOf"]
    assert set(find["$defs"]["SearchResponse"]["required"]) >= {
        "results",
        "total_hits",
        "applied_config",
    }
    assert {"$ref": "#/$defs/IndexStatusBody"} in status["anyOf"]
    assert set(status["$defs"]["IndexStatusBody"]["required"]) >= {
        "schema_version",
        "index_state",
        "vector_store",
    }


def test_success_and_error_results_pass_validation(tmp_path: Path) -> None:
    built_library(tmp_path)

    seen = _session(tmp_path)

    tools = seen["tools"]
    assert not seen["find"].isError and not seen["status"].isError
    assert structured_body(seen["find"])["results"]
    assert structured_body(seen["status"])["index_state"] == "ready"
    assert seen["above_cap"].isError
    assert structured_body(seen["above_cap"])["category"] == "validation_error"
    for name, key in (
        ("find_passages", "find"),
        ("get_index_status", "status"),
        ("find_passages", "above_cap"),
    ):
        jsonschema.validate(structured_body(seen[key]), tools[name].outputSchema)


def test_an_empty_index_error_passes_validation(tmp_path: Path) -> None:
    (tmp_path / "corpus").mkdir()
    services = create_container(settings_for(tmp_path))
    services.build.run()
    close(services)

    schema, result = _one_search(tmp_path)

    assert result.isError
    body = structured_body(result)
    assert body["category"] == "index_empty"
    jsonschema.validate(body, schema)


def test_a_missing_index_error_passes_validation(tmp_path: Path) -> None:
    (tmp_path / "corpus").mkdir()

    schema, result = _one_search(tmp_path)

    assert result.isError
    body = structured_body(result)
    assert body["category"] == "index_incomplete"
    jsonschema.validate(body, schema)
