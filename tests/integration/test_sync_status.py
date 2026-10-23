"""The `sync` section of `docgrep status`, `GET /info` and the MCP status tool
(ADR-0029): counts and names for every client, a few stored paths in
`docgrep status` only."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anyio
import pytest
from fastapi.testclient import TestClient

from docgrep.api.cli import main
from docgrep.api.container import Services, create_status
from docgrep.api.http import create_app
from docgrep.api.mcp import create_mcp_server
from mcp_support import structured
from sync_support import BOOKS, Rig

SECTION_KEYS = {
    "enabled",
    "running",
    "interval_seconds",
    "settle_seconds",
    "last_scan_at",
    "last_applied",
    "pending",
    "unsettled",
    "unsettled_over_limit",
    "held_removals",
    "paused_roots",
    "awaiting_first_update",
    "idle_reason",
    "last_error",
    "applying",
}


@pytest.fixture
def held(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Rig]:
    """A synced index whose books root had all its files removed: the
    removals are held."""
    rig = Rig(tmp_path)
    for name, value in {
        "DOCGREP_DATA_DIR": str(rig.data),
        "DOCGREP_BACKEND": "fake",
        "DOCGREP_SYNC": "true",
        "DOCGREP_CHUNK_SIZE": "10",
        "DOCGREP_CHUNK_OVERLAP": "2",
        "DOCGREP_LOG_LEVEL": "WARNING",
    }.items():
        monkeypatch.setenv(name, value)
    rig.open()
    rig.cycle()
    for name in BOOKS:
        (rig.books / name).unlink()
    rig.cycle()
    rig.cycle()
    assert rig.sync.status().held
    yield rig
    rig.close()


def _status_cli(capsys: pytest.CaptureFixture[str], *args: str) -> Any:
    code = main(["status", *args])
    out, _ = capsys.readouterr()
    assert code == 0
    return json.loads(out) if "--json" in args else out


def test_status_has_a_disabled_sync_section_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = Rig(tmp_path)
    monkeypatch.setenv("DOCGREP_DATA_DIR", str(rig.data))
    monkeypatch.setenv("DOCGREP_BACKEND", "fake")

    body = _status_cli(capsys, "--json")
    text = _status_cli(capsys)

    assert set(body["sync"]) == SECTION_KEYS
    assert (body["sync"]["enabled"], body["sync"]["running"]) == (False, False)
    assert "Sync" not in text


def test_docgrep_status_shows_held_removals_with_a_few_paths(
    held: Rig, capsys: pytest.CaptureFixture[str]
) -> None:
    body = _status_cli(capsys, "--json")["sync"]

    assert set(body) == SECTION_KEYS
    # The server of this data directory (this process) is alive.
    assert (body["enabled"], body["running"]) == (True, True)
    [entry] = body["held_removals"]
    assert (entry["root"], entry["count"]) == ("books", len(BOOKS))
    assert 0 < len(entry["paths"]) <= 5
    assert set(entry["paths"]) <= set(BOOKS)
    assert body["last_applied"] is not None and body["last_scan_at"] is not None
    assert body["pending"] == {"added": 0, "modified": 0, "removed": 0}

    text = _status_cli(capsys)
    assert "Sync: running, scanning every 120 s, settling 60 s" in text
    assert f"Held removals in root books: {len(BOOKS)}" in text
    assert f"    {entry['paths'][0]}" in text


def test_docgrep_status_says_when_no_server_runs_sync(
    held: Rig, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import docgrep.adapters.syncstate as syncstate

    monkeypatch.setattr(syncstate, "process_alive", lambda pid: False)

    body = _status_cli(capsys, "--json")["sync"]

    assert (body["enabled"], body["running"]) == (True, False)
    assert body["held_removals"][0]["count"] == len(BOOKS)
    assert "enabled, but no server is running it" in _status_cli(capsys)


def test_info_carries_the_live_section_without_paths(held: Rig) -> None:
    assert held.services is not None
    app = create_app(held.settings(sync=True), services=held.services)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        body = client.get("/info").json()["sync"]

    assert set(body) == SECTION_KEYS
    assert body["held_removals"] == [{"root": "books", "count": len(BOOKS)}]
    assert not any(name in json.dumps(body) for name in BOOKS)


def test_info_reports_sync_off_without_creating_a_container(tmp_path: Path) -> None:
    rig = Rig(tmp_path)

    with TestClient(create_app(rig.settings()), base_url="http://127.0.0.1") as client:
        body = client.get("/info").json()["sync"]

    assert set(body) == SECTION_KEYS and body["enabled"] is False
    assert not (rig.data / "sync-status.json").exists()


def _mcp_body(rig: Rig, services: Services, **kwargs: Any) -> dict[str, Any]:
    server = create_mcp_server(
        rig.settings(sync=True), services=lambda: services, **kwargs
    )
    return structured(anyio.run(server.call_tool, "get_index_status", {}))  # type: ignore[no-any-return]  # test helper


def test_the_mcp_status_body_has_the_section_and_never_a_path(held: Rig) -> None:
    assert held.services is not None
    sync = held.sync

    body = _mcp_body(
        held,
        held.services,
        status=lambda: create_status(held.settings(sync=True), sync=sync.status),
    )

    assert set(body["sync"]) == SECTION_KEYS
    assert body["sync"]["held_removals"] == [{"root": "books", "count": len(BOOKS)}]
    dumped = json.dumps(body["sync"])
    assert not any(name in dumped for name in BOOKS) and str(held.books) not in dumped


def test_the_stdio_mcp_status_reads_the_file_of_the_server(held: Rig) -> None:
    assert held.services is not None

    # No live sync is given: as in `docgrep mcp`, the file is read.
    body = _mcp_body(held, held.services)

    assert body["sync"]["running"] is True
    assert body["sync"]["held_removals"] == [{"root": "books", "count": len(BOOKS)}]
    assert not any(name in json.dumps(body["sync"]) for name in BOOKS)


def test_the_live_state_shows_the_applying_progress(held: Rig) -> None:
    seen: list[Any] = []
    sync = held.sync
    inner = sync.sleep
    sync.sleep = lambda seconds: (seen.append(sync.status().applying), inner(seconds))[
        1
    ]
    for index in range(3):
        (held.books / f"new{index}.txt").write_text(f"new file {index}", "utf-8")
    held.cycle()
    held.cycle()

    totals = {a for a in seen if a is not None}
    assert any(total == 3 for _, total in totals)
    assert sync.status().applying is None


def test_idle_and_error_reasons_reach_the_status(held: Rig) -> None:
    update: Any = held.services.update  # type: ignore[union-attr]  # container internals

    from docgrep.domain.errors import ConfigDrift

    def drifted() -> None:
        raise ConfigDrift("The configuration differs from the index")

    update.check_ready = drifted

    held.cycle()

    body = held.sync.status().to_dict()
    assert body["idle_reason"].startswith("config_drift: ")
    assert held.sync.status().idle_reason == body["idle_reason"]
