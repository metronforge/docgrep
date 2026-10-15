"""The MCP front-end stays in the api layer and degrades without its SDK."""

import ast
import sys
from pathlib import Path

import pytest

from docgrep.api.cli import MCP_INSTALL_HINT, main

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "docgrep"


def _sdk_importers(root: Path) -> list[str]:
    """Modules outside docgrep/api that import the MCP SDK."""
    found: list[str] = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if relative.parts[0] == "api":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module or ""]
            if any(name.split(".")[0] == "mcp" for name in names):
                found.append(relative.as_posix())
    return found


def test_the_sdk_is_imported_only_in_the_api_layer() -> None:
    assert _sdk_importers(SOURCE_ROOT) == []


def test_the_guard_finds_an_sdk_import_elsewhere(tmp_path: Path) -> None:
    for layer, source in (
        ("domain", "import mcp\n"),
        ("usecases", "from mcp.server.fastmcp import FastMCP\n"),
        ("api", "from mcp.types import TextContent\n"),
    ):
        (tmp_path / layer).mkdir()
        (tmp_path / layer / "m.py").write_text(source, encoding="utf-8")

    assert _sdk_importers(tmp_path) == ["domain/m.py", "usecases/m.py"]


def test_mcp_without_the_sdk_exits_2_with_the_install_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Block the SDK and any of its modules an earlier test already imported.
    for name in [n for n in sys.modules if n == "mcp" or n.startswith("mcp.")]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "mcp", None)
    monkeypatch.delitem(sys.modules, "docgrep.api.mcp", raising=False)

    status = main(["mcp"])

    out, err = capsys.readouterr()
    assert status == 2 and out == ""
    assert "package `mcp`" in err and MCP_INSTALL_HINT in err


def test_mcp_is_a_listed_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])

    assert stopped.value.code == 0
    overview = " ".join(capsys.readouterr().out.split())
    assert "mcp Serve the read-only MCP tools on standard input and output" in overview
