"""Malformed tool arguments come back as tool errors with the body of an
HTTP validation error, through the SDK's own stdio client (UC-1, UC-7)."""

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from docgrep.api.http import create_app
from docgrep.settings import MCP_MAX_TOP_N
from mcp_support import QUERY, run, settings_for, stdio_session, structured, text

_MALFORMED: dict[str, dict[str, Any]] = {
    "wrong_type": {"query": QUERY, "top_n": "many"},
    "missing": {"top_n": 3},
    "query_type": {"query": 7},
    "flag_type": {"query": QUERY, "collapse_duplicates": "sometimes"},
}


def _calls(root: Path) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    async def work() -> None:
        async with stdio_session(root) as session:
            for name, arguments in _MALFORMED.items():
                seen[name] = await session.call_tool("find_passages", arguments)
            seen["negative"] = await session.call_tool(
                "find_passages", {"query": QUERY, "top_n": -1}
            )

    run(work)
    return seen


def test_malformed_arguments_carry_category_and_details(tmp_path: Path) -> None:
    (tmp_path / "corpus").mkdir()

    seen = _calls(tmp_path)

    assert all(result.isError for result in seen.values())
    assert structured(seen["wrong_type"]) == {
        "category": "validation_error",
        "message": "Invalid tool arguments",
        "details": {
            "errors": [
                {
                    "location": ["top_n"],
                    "message": "Input should be a valid integer, "
                    "unable to parse string as an integer",
                }
            ]
        },
    }
    assert structured(seen["missing"])["details"] == {
        "errors": [{"location": ["query"], "message": "Field required"}]
    }
    for name, argument in (
        ("query_type", "query"),
        ("flag_type", "collapse_duplicates"),
    ):
        (problem,) = structured(seen[name])["details"]["errors"]
        assert problem["location"] == [argument]
    assert structured(seen["negative"]) == {
        "category": "validation_error",
        "message": f"top_n must be between 1 and {MCP_MAX_TOP_N}",
        "details": {"minimum": 1, "maximum": MCP_MAX_TOP_N},
    }
    for result in seen.values():
        assert "Traceback" not in text(result)
        assert str(tmp_path) not in text(result)


def test_malformed_arguments_match_the_http_error_body(tmp_path: Path) -> None:
    (tmp_path / "corpus").mkdir()
    client = TestClient(create_app(settings_for(tmp_path)), base_url="http://127.0.0.1")

    seen = _calls(tmp_path)

    for name, arguments in _MALFORMED.items():
        response = client.post("/search", json=arguments)
        assert response.status_code == 400
        rest = response.json()
        tool = structured(seen[name])
        assert tool["category"] == rest["category"] == "validation_error"
        shared = tool["details"].keys() & rest["details"].keys()
        assert shared == {"errors"}
        # HTTP locates a problem in the request body, MCP in the arguments.
        assert [
            {"location": ["body", *problem["location"]], "message": problem["message"]}
            for problem in tool["details"]["errors"]
        ] == rest["details"]["errors"]
