"""The docgrep commands shown in the README exist with the flags they use."""

import re
import shlex
from pathlib import Path

import pytest

from docgrep.api.cli import main

README = Path(__file__).resolve().parents[2] / "README.md"
_FENCE = re.compile(r"^```sh\n(.*?)^```", re.MULTILINE | re.DOTALL)
_PLACEHOLDER = re.compile(r"<[^<>\s][^<>]*>")
_SEPARATOR = re.compile(r"\||&&|;")
# Commands whose first argument is a subcommand of their own.
_NESTED = {"roots"}


def _invocations() -> list[tuple[str, list[str], str]]:
    """Each `docgrep <subcommand>` in an sh block, with its long flags; for a
    command with subcommands of its own (`docgrep roots add`), both names."""
    found = []
    for block in _FENCE.findall(README.read_text(encoding="utf-8")):
        for line in block.splitlines():
            line = _PLACEHOLDER.sub("", line.split(" #", 1)[0])
            if line.lstrip().startswith("#"):
                continue
            for segment in _SEPARATOR.split(line):
                if "docgrep" not in segment:
                    continue
                try:
                    words = shlex.split(segment)
                except ValueError:
                    continue
                if "docgrep" not in words[:-1]:
                    continue
                rest = words[words.index("docgrep") + 1 :]
                flags = [word.split("=", 1)[0] for word in rest if word[:2] == "--"]
                command = " ".join(rest[:2]) if rest[0] in _NESTED else rest[0]
                found.append((command, flags, line.strip()))
    return found


def _help(capsys: pytest.CaptureFixture[str], *args: str) -> str:
    with pytest.raises(SystemExit) as stopped:
        main([*args, "--help"])
    assert stopped.value.code == 0, args
    return capsys.readouterr().out


def test_readme_commands_use_real_subcommands_and_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    invocations = _invocations()
    used = {subcommand.split()[0] for subcommand, _flags, _line in invocations}
    # Guard against a parser that silently finds nothing.
    assert used >= {"index", "update", "search", "status", "serve", "dry-run", "ocr"}

    overview = _help(capsys)
    for subcommand, flags, line in invocations:
        first = subcommand.split()[0]
        assert re.search(rf"[{{,]{re.escape(first)}[,}}]", overview), line
        options = set(re.findall(r"--[a-z][a-z-]*", _help(capsys, *subcommand.split())))
        for flag in flags:
            assert flag in options, f"{flag} is not an option of: {line}"
