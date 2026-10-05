"""A reader that closes the output pipe early neither breaks nor alters a command.

Each command runs as a real subprocess with the deterministic backend, once
with its output read in full and once with the pipe closed by the reader.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from resume_support import dump_points, settings_for
from update_support import INITIAL, change_corpus, sources_table, write_files


def _environment(root: Path) -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("DOCGREP_")
    }
    environment.update(
        DOCGREP_CORPUS_ROOT=str(root / "corpus"),
        DOCGREP_DATA_DIR=str(root / "data"),
        DOCGREP_BACKEND="fake",
        DOCGREP_CHUNK_SIZE="10",
        DOCGREP_CHUNK_OVERLAP="2",
    )
    return environment


def _command(*args: str) -> list[str]:
    return [sys.executable, "-m", "docgrep.api.cli", *args]


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _command(*args),
        capture_output=True,
        text=True,
        env=_environment(root),
        check=False,
        timeout=120,
    )


def _run_closed(root: Path, *args: str, read_lines: int = 0) -> tuple[int, str]:
    """Run with a pipe whose reader closes after ``read_lines`` lines."""
    process = subprocess.Popen(
        _command(*args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_environment(root),
    )
    assert process.stdout is not None and process.stderr is not None
    for _ in range(read_lines):
        process.stdout.readline()
    process.stdout.close()
    stderr = process.stderr.read().decode("utf-8")
    process.stderr.close()
    return process.wait(timeout=120), stderr


def _assert_quiet(stderr: str) -> None:
    assert "Traceback" not in stderr
    assert "BrokenPipeError" not in stderr
    assert "Broken pipe" not in stderr
    assert "io_error" not in stderr


def _built(root: Path) -> Path:
    write_files(root / "corpus", INITIAL)
    built = _run(root, "index")
    assert built.returncode in (0, 1), built.stderr
    return root


def _state(root: Path) -> tuple[Any, ...]:
    settings = settings_for(root / "corpus", root / "data")
    return sources_table(settings), dump_points(settings)


@pytest.fixture
def indexed(tmp_path: Path) -> Path:
    return _built(tmp_path / "indexed")


@pytest.mark.parametrize("read_lines", [0, 1])
def test_status_with_a_closed_pipe(indexed: Path, read_lines: int) -> None:
    full = _run(indexed, "status")

    code, stderr = _run_closed(indexed, "status", read_lines=read_lines)

    assert full.stdout.count("\n") > 1
    _assert_quiet(stderr)
    assert code == full.returncode == 0


@pytest.mark.parametrize("read_lines", [0, 1])
def test_search_with_a_closed_pipe(indexed: Path, read_lines: int) -> None:
    args = ("search", "alpha beta", "--top-n", "20")
    full = _run(indexed, *args)

    code, stderr = _run_closed(indexed, *args, read_lines=read_lines)

    assert full.stdout.count("\n") > 10
    _assert_quiet(stderr)
    assert code == full.returncode == 0


def test_search_exit_code_of_a_failure_is_kept_with_a_closed_pipe(
    tmp_path: Path,
) -> None:
    (tmp_path / "corpus").mkdir()
    full = _run(tmp_path, "search", "anything")

    code, stderr = _run_closed(tmp_path, "search", "anything")

    assert full.returncode != 0
    assert code == full.returncode
    assert "Traceback" not in stderr and "BrokenPipeError" not in stderr


@pytest.mark.parametrize("read_lines", [0, 1])
def test_update_with_a_closed_pipe_completes_and_records_the_same_result(
    tmp_path: Path, read_lines: int
) -> None:
    reference = _built(tmp_path / "reference")
    closed = _built(tmp_path / "closed")
    change_corpus(reference / "corpus")
    change_corpus(closed / "corpus")

    full = _run(reference, "update")
    code, stderr = _run_closed(closed, "update", read_lines=read_lines)

    assert full.stdout.startswith("Update plan:")
    assert full.stdout.count("\n") >= 3
    _assert_quiet(stderr)
    assert code == full.returncode
    assert _state(closed) == _state(reference)
    assert _run(closed, "update").stdout.startswith("No changes detected")


def test_normal_output_is_unchanged_by_the_pipe_handling(indexed: Path) -> None:
    status = _run(indexed, "status")
    searched = _run(indexed, "search", "alpha beta", "--top-n", "3")

    assert status.returncode == 0 and searched.returncode == 0
    assert status.stdout.startswith("Index: ready\n")
    assert "(score " in searched.stdout
    _assert_quiet(status.stderr + searched.stderr)
