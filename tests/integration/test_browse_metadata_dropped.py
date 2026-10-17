"""The dropped-metadata audit and filter over HTTP, MCP and the CLI, and the
sanitized metadata of search results (ADR-0021, issue #22)."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import jsonschema
from fastapi.testclient import TestClient

from docgrep.api.container import create_container
from docgrep.api.http import create_app
from docgrep.settings import Settings
from mcp_support import (
    QUERY,
    environment,
    run,
    settings_for,
    stdio_session,
    structured,
    write_library,
)
from resume_support import close

_JUNK_NOTE = """---
title: book title
author: Unknown
year: 101
---
# Chapter

The amber walrus counts the tides.
"""

_OLD_NOTE = """---
title: A Tale of Old
author: Teller
year: 1970
---
The amber walrus is older than it looks.
"""


def _library(root: Path) -> Settings:
    """The shared MCP library (its DOCX author is python-docx) and two
    Markdown notes: one with junk frontmatter, one with year 1970."""
    write_library(root / "corpus")
    (root / "corpus" / "junk.md").write_text(_JUNK_NOTE, encoding="utf-8")
    (root / "corpus" / "old.md").write_text(_OLD_NOTE, encoding="utf-8")
    settings = settings_for(root)
    services = create_container(settings)
    try:
        summary = services.build.run()
    finally:
        close(services)
    assert (summary.indexed, summary.failed) == (6, 1), summary
    return settings


def _post(settings: Settings, path: str, body: dict[str, Any]) -> Any:
    services = create_container(settings)
    try:
        client = TestClient(
            create_app(settings, services=services), base_url="http://127.0.0.1"
        )
        return client.post(path, json=body)
    finally:
        close(services)


def _cli(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "docgrep.api.cli", "sources", *args],
        env=environment(root),
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def _by_path(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["path"]: item for item in body["sources"]}


def test_browse_entries_carry_the_audit_only_when_something_was_dropped(
    tmp_path: Path,
) -> None:
    settings = _library(tmp_path)

    response = _post(settings, "/sources/browse", {})

    assert response.status_code == 200
    entries = _by_path(response.json())
    junk = entries["junk.md"]
    assert (junk["title"], junk["author"], junk["year"]) == (None, None, None)
    assert junk["metadata_dropped"] == {
        "title": {"raw": "book title", "reason": "placeholder"},
        "author": {"raw": "Unknown", "reason": "placeholder"},
        "year": {"raw": 101, "reason": "implausible_year"},
    }
    memo = entries["memo.docx"]
    assert (memo["title"], memo["author"]) == ("Sea Memo", None)
    assert memo["metadata_dropped"] == {
        "author": {"raw": "python-docx", "reason": "placeholder"}
    }
    old = entries["old.md"]
    assert (old["title"], old["author"], old["year"]) == (
        "A Tale of Old",
        "Teller",
        1970,
    )
    assert "metadata_dropped" not in old
    assert "metadata_dropped" not in entries["notes/a.txt"]


def test_the_filter_lists_only_affected_sources(tmp_path: Path) -> None:
    settings = _library(tmp_path)

    dropped = _post(settings, "/sources/browse", {"metadata_dropped": True}).json()
    kept = _post(settings, "/sources/browse", {"metadata_dropped": False}).json()

    assert list(_by_path(dropped)) == ["junk.md", "memo.docx"]
    assert dropped["applied_filters"] == {"metadata_dropped": True}
    assert dropped["aggregates"] == {
        "total_sources": 2,
        "by_file_type": {"docx": 1, "markdown": 1},
    }
    assert "junk.md" not in _by_path(kept) and "old.md" in _by_path(kept)
    assert kept["applied_filters"] == {"metadata_dropped": False}


def test_a_year_filter_and_without_any_see_the_sanitized_values(
    tmp_path: Path,
) -> None:
    settings = _library(tmp_path)

    early = _post(settings, "/sources/browse", {"year_to": 1990}).json()
    bare = _post(settings, "/sources/browse", {"has_metadata": "without_any"}).json()

    assert list(_by_path(early)) == ["old.md"]
    assert "junk.md" in _by_path(bare) and "memo.docx" in _by_path(bare)


def test_the_filter_is_a_strict_boolean(tmp_path: Path) -> None:
    settings = _library(tmp_path)

    response = _post(settings, "/sources/browse", {"metadata_dropped": "yes"})

    assert response.status_code == 400
    assert response.json()["category"] == "validation_error"


def test_cli_json_and_mcp_return_the_http_body(tmp_path: Path) -> None:
    settings = _library(tmp_path)
    rest = _post(settings, "/sources/browse", {"metadata_dropped": True}).json()
    seen: dict[str, Any] = {}

    async def work() -> None:
        async with stdio_session(tmp_path) as session:
            tools = {t.name: t for t in (await session.list_tools()).tools}
            seen["schema"] = tools["browse_sources"].outputSchema
            seen["input"] = tools["browse_sources"].inputSchema
            seen["result"] = await session.call_tool(
                "browse_sources", {"metadata_dropped": True}
            )

    run(work)
    done = _cli(tmp_path, "--dropped-metadata", "--json")

    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == rest
    assert not seen["result"].isError
    body = structured(seen["result"])
    assert body == rest
    assert "metadata_dropped" in seen["input"]["properties"]
    jsonschema.validate(body, seen["schema"])
    entry = seen["schema"]["$defs"]["BrowsedSourceResponse"]["properties"]
    assert "metadata_dropped" in entry
    assert set(seen["schema"]["$defs"]["DroppedFieldResponse"]["properties"]) == {
        "raw",
        "reason",
        "inherited_from",
    }


def test_the_table_output_is_unchanged(tmp_path: Path) -> None:
    _library(tmp_path)

    done = _cli(tmp_path, "--dropped-metadata")

    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    assert lines[0].split() == ["PATH", "TYPE", "YEAR", "AUTHOR", "TITLE", "PASSAGES"]
    assert [line.split()[0] for line in lines[1:3]] == ["junk.md", "memo.docx"]
    assert "python-docx" not in done.stdout and "Unknown" not in done.stdout
    assert lines[3] == "Sources 1-2 of 2 (page 1 of 1); docx 1, markdown 1"


def test_search_results_show_sanitized_values_without_the_audit(
    tmp_path: Path,
) -> None:
    settings = _library(tmp_path)

    body = _post(settings, "/search", {"query": QUERY}).json()
    walrus = _post(settings, "/search", {"query": "amber walrus"}).json()
    early = _post(
        settings, "/search", {"query": "amber walrus", "year_to": 1990}
    ).json()

    memo = next(r for r in body["results"] if r["source"]["path"] == "memo.docx")
    assert (memo["source"]["title"], memo["source"]["author"]) == ("Sea Memo", None)
    assert all("metadata_dropped" not in r["source"] for r in body["results"])
    junk = next(r for r in walrus["results"] if r["source"]["path"] == "junk.md")
    assert (junk["source"]["title"], junk["source"]["year"]) == (None, None)
    assert {r["source"]["path"] for r in early["results"]} == {"old.md"}
    # junk.md's only year was 101, so it is left out for missing metadata.
    assert early["sources_excluded_for_missing_metadata"] >= 1
