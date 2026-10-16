"""The one private SDK attribute docgrep uses (``lowlevel_server`` in
docgrep.api.mcp) still behaves as checked against mcp 1.30.0; an SDK upgrade
that changes it fails here, not in a user's client."""

import inspect
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.server.lowlevel import Server
from mcp.types import CallToolRequest, ListToolsRequest

from docgrep import __version__
from docgrep.api.container import Services
from docgrep.api.mcp import create_mcp_server, lowlevel_server
from docgrep.settings import Settings

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "docgrep"
UPGRADE = (
    "the MCP SDK changed FastMCP._mcp_server, which docgrep.api.mcp."
    "lowlevel_server relies on (checked against mcp 1.30.0): look for a public "
    "replacement before upgrading"
)


def _no_services() -> Services:
    raise AssertionError("not called")


def _server(tmp_path: Path) -> FastMCP:
    settings = Settings(
        corpus_root=tmp_path, data_dir=tmp_path / "data", backend="fake"
    )
    return create_mcp_server(settings, services=_no_services)


def test_fastmcp_still_holds_the_low_level_server(tmp_path: Path) -> None:
    server = _server(tmp_path)

    assert hasattr(server, "_mcp_server"), UPGRADE
    lowlevel = lowlevel_server(server)
    assert isinstance(lowlevel, Server), UPGRADE
    # The tools registered through FastMCP are served by this very server.
    assert CallToolRequest in lowlevel.request_handlers, UPGRADE
    assert ListToolsRequest in lowlevel.request_handlers, UPGRADE


def test_the_low_level_server_reports_the_docgrep_version(tmp_path: Path) -> None:
    lowlevel = lowlevel_server(_server(tmp_path))

    options = lowlevel.create_initialization_options()

    assert options.server_name == "docgrep", UPGRADE
    assert options.server_version == __version__, UPGRADE


def test_the_low_level_run_still_takes_streams_and_options() -> None:
    parameters = list(inspect.signature(Server.run).parameters)

    assert parameters[:4] == [
        "self",
        "read_stream",
        "write_stream",
        "initialization_options",
    ], UPGRADE


def test_the_private_attribute_is_read_in_one_place_only() -> None:
    uses = [
        f"{path.relative_to(SOURCE_ROOT).as_posix()}:{number}"
        for path in sorted(SOURCE_ROOT.rglob("*.py"))
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if "._mcp_server" in line
    ]

    assert len(uses) == 1 and uses[0].startswith("api/mcp.py:"), uses
