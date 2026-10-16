"""Invalid settings reported by real `docgrep` processes: one invalid_settings
line on standard error, no traceback, and the documented exit status."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Starts the interpreter the way the console script does.
CONSOLE_SCRIPT = "import sys; from docgrep.api.cli import main; sys.exit(main())"


def _environment(root: Path, name: str, value: str) -> dict[str, str]:
    """A minimal environment with one invalid setting and nothing inherited
    from the caller's DOCGREP_ settings."""
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(root),
        "HF_HUB_OFFLINE": "1",
        "DOCGREP_CORPUS_ROOT": str(root / "corpus"),
        "DOCGREP_DATA_DIR": str(root / "data"),
        "DOCGREP_BACKEND": "fake",
        name: value,
    }


def _docgrep(
    root: Path, *args: str, setting: tuple[str, str] = ("DOCGREP_PORT", "70000")
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", CONSOLE_SCRIPT, *args],
        env=_environment(root, *setting),
        cwd=str(root),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _assert_reported(done: subprocess.CompletedProcess[str], status: int) -> None:
    assert done.returncode == status, done.stderr
    assert "Traceback" not in done.stderr
    lines = done.stderr.splitlines()
    assert lines and lines[0].startswith("invalid_settings: ")
    assert sum(line.startswith("invalid_settings:") for line in lines) == 1


@pytest.mark.parametrize(
    ("args", "status"),
    [
        (("discover-scans",), 2),
        (("ocr", "--plan", "--all-discovered"), 2),
        (("mcp",), 2),
        (("status",), 1),
        (("search", "kraken"), 1),
    ],
)
def test_invalid_port_reports_the_documented_status(
    tmp_path: Path, args: tuple[str, ...], status: int
) -> None:
    done = _docgrep(tmp_path, *args)

    _assert_reported(done, status)
    assert "port" in done.stderr


def test_invalid_chunk_overlap_reports_the_documented_status(tmp_path: Path) -> None:
    overlap = ("DOCGREP_CHUNK_OVERLAP", "600")

    _assert_reported(_docgrep(tmp_path, "discover-scans", setting=overlap), 2)
    _assert_reported(_docgrep(tmp_path, "status", setting=overlap), 1)


def test_mcp_leaves_standard_output_empty(tmp_path: Path) -> None:
    done = _docgrep(tmp_path, "mcp")

    _assert_reported(done, 2)
    assert done.stdout == ""


def test_python_m_entry_point_reports_the_same_status(tmp_path: Path) -> None:
    done = subprocess.run(
        [sys.executable, "-m", "docgrep.api.cli", "mcp"],
        env=_environment(tmp_path, "DOCGREP_PORT", "70000"),
        cwd=str(tmp_path),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )

    _assert_reported(done, 2)
    assert done.stdout == ""


@pytest.mark.parametrize("args", [("--help",), ("mcp", "--help"), ("ocr", "--help")])
def test_help_works_with_invalid_settings(
    tmp_path: Path, args: tuple[str, ...]
) -> None:
    done = _docgrep(tmp_path, *args)

    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("usage: docgrep")
    assert done.stderr == ""


def test_api_modules_import_with_invalid_settings(tmp_path: Path) -> None:
    done = subprocess.run(
        [sys.executable, "-c", "import docgrep.api.http, docgrep.api.cli"],
        env=_environment(tmp_path, "DOCGREP_PORT", "70000"),
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert done.returncode == 0, done.stderr
    assert done.stderr == ""
