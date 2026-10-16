"""The initialize result names docgrep and its version as the server, over
standard input and output and over streamable HTTP."""

import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import Implementation

from docgrep import __version__
from mcp_support import Serve, environment, run


def _stdio_server_info(root: Path) -> Implementation:
    found: list[Implementation] = []

    async def work() -> None:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "docgrep.api.cli", "mcp"],
            env=environment(root),
            cwd=str(root),
        )
        with (root / "mcp-stderr.log").open("a", encoding="utf-8") as errlog:
            async with stdio_client(params, errlog=errlog) as (read, write):
                async with ClientSession(read, write) as session:
                    found.append((await session.initialize()).serverInfo)

    run(work)
    return found[0]


def _http_server_info(url: str) -> Implementation:
    found: list[Implementation] = []

    async def work() -> None:
        async with streamable_http_client(url) as (read, write, _session_id):
            async with ClientSession(read, write) as session:
                found.append((await session.initialize()).serverInfo)

    run(work)
    return found[0]


def test_stdio_reports_the_docgrep_version(tmp_path: Path) -> None:
    (tmp_path / "corpus").mkdir()

    info = _stdio_server_info(tmp_path)

    assert (info.name, info.version) == ("docgrep", __version__)


def test_http_reports_the_docgrep_version(tmp_path: Path) -> None:
    (tmp_path / "corpus").mkdir()

    with Serve(tmp_path) as server:
        info = _http_server_info(f"{server.base}/mcp")

    assert (info.name, info.version) == ("docgrep", __version__)
