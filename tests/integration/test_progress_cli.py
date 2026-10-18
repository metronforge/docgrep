"""The progress line of the long commands, on a pseudo-terminal and without."""

import fcntl
import os
import pty
import re
import struct
import subprocess
import sys
import termios
from pathlib import Path

import pytest

COLUMNS = 100


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


def _corpus(root: Path, names: tuple[str, ...] = ("alpha", "bravo", "charlie")) -> None:
    corpus = root / "corpus"
    corpus.mkdir(exist_ok=True)
    for name in names:
        (corpus / f"{name}.txt").write_text(
            f"The {name} passage talks about comets and stars. " * 6, encoding="utf-8"
        )


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a command with standard error and output on pipes."""
    return subprocess.run(
        [sys.executable, "-m", "docgrep.api.cli", *args],
        capture_output=True,
        text=True,
        env=_environment(root),
        check=False,
        timeout=60,
    )


def _run_on_terminal(root: Path, *args: str) -> tuple[int, str, str]:
    """Run a command with standard error on a pseudo-terminal of 100 columns;
    returns the status, what the terminal received and standard output."""
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, COLUMNS, 0, 0))
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "docgrep.api.cli", *args],
            stdout=subprocess.PIPE,
            stderr=slave,
            env=_environment(root),
            text=True,
        )
        os.close(slave)
        slave = -1
        chunks: list[bytes] = []
        while True:
            try:
                data = os.read(master, 65536)
            except OSError:  # the terminal closes with the last writer
                break
            if not data:
                break
            chunks.append(data)
        stdout, _ = process.communicate(timeout=60)
    finally:
        os.close(master)
        if slave != -1:
            os.close(slave)
    return process.returncode, b"".join(chunks).decode(), stdout


def _normalized(text: str, root: Path) -> str:
    """Output without what differs between two runs: times, ids, durations."""
    text = text.replace(str(root), "<root>")
    text = re.sub(r"\d{4}-\d\d-\d\dT[\d:]+Z", "<time>", text)
    text = re.sub(r"run_id=\S+", "run_id=<id>", text)
    text = re.sub(
        r"(elapsed_seconds|duration_ms|passages_per_second)=\S+", r"\1=<n>", text
    )
    text = re.sub(r"Build \S+:", "Build <id>:", text)
    return re.sub(r"in \d+\.\d+s", "in <n>s", text)


def _lines(terminal: str) -> list[str]:
    """The distinct states of the status line, in order."""
    return [part for part in re.split(r"[\r\n]", terminal) if part.strip()]


def test_index_draws_a_status_line_and_leaves_a_final_line(tmp_path: Path) -> None:
    _corpus(tmp_path)

    status, terminal, stdout = _run_on_terminal(tmp_path, "index")

    assert status == 0, terminal
    assert "3 indexed" in stdout
    assert "\r" in terminal
    drawn = [line for line in _lines(terminal) if line.startswith("Indexing")]
    assert len(drawn) >= 2  # the first state and the final line
    assert all(len(line) == COLUMNS - 1 for line in drawn)
    final = drawn[-1]
    assert final.startswith("Indexing  100%  3/3 sources")
    assert "ETA" not in final
    # The final line is followed by a line break and stays on screen.
    assert re.search(r"Indexing  100%[^\n]*\n", terminal)


def test_no_status_line_without_a_terminal(tmp_path: Path) -> None:
    _corpus(tmp_path)

    result = _run(tmp_path, "index")

    assert result.returncode == 0, result.stderr
    assert "\r" not in result.stderr
    assert "Indexing" not in result.stderr
    assert "index_progress" in result.stderr


def test_without_a_terminal_log_and_default_output_are_the_same(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    for root in (first, second):
        root.mkdir()
        _corpus(root)

    default = _run(first, "index")
    logged = _run(second, "index", "--progress=log")

    assert default.returncode == logged.returncode == 0
    assert _normalized(default.stdout, first) == _normalized(logged.stdout, second)
    assert _normalized(default.stderr, first) == _normalized(logged.stderr, second)


def test_progress_log_draws_nothing_even_on_a_terminal(tmp_path: Path) -> None:
    _corpus(tmp_path)

    status, terminal, _ = _run_on_terminal(tmp_path, "index", "--progress=log")

    assert status == 0
    assert "Indexing" not in terminal
    assert "index_progress" in terminal


def test_progress_bar_draws_without_a_terminal_too(tmp_path: Path) -> None:
    _corpus(tmp_path)

    result = _run(tmp_path, "index", "--progress=bar")

    assert result.returncode == 0, result.stderr
    assert "Indexing  100%  3/3 sources" in result.stderr


def test_progress_off_lowers_the_progress_records_to_debug(tmp_path: Path) -> None:
    _corpus(tmp_path)

    quiet = _run(tmp_path, "index", "--progress=off")

    assert quiet.returncode == 0, quiet.stderr
    assert "index_progress" not in quiet.stderr
    assert "index_build_finished" in quiet.stderr


def test_progress_off_is_all_the_log_shows_at_debug(tmp_path: Path) -> None:
    _corpus(tmp_path)
    environment = _environment(tmp_path)
    environment["DOCGREP_LOG_LEVEL"] = "DEBUG"

    result = subprocess.run(
        [sys.executable, "-m", "docgrep.api.cli", "index", "--progress=off"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
        timeout=60,
    )

    assert "DEBUG docgrep.usecases.build_index index_progress" in result.stderr


def test_update_draws_its_line_on_a_terminal(tmp_path: Path) -> None:
    _corpus(tmp_path)
    assert _run(tmp_path, "index").returncode == 0
    _corpus(tmp_path, ("delta", "echo"))

    status, terminal, stdout = _run_on_terminal(tmp_path, "update")

    assert status == 0, terminal
    assert "Update" in stdout
    drawn = [line for line in _lines(terminal) if line.startswith("Updating")]
    assert drawn
    assert drawn[-1].startswith("Updating  100%  2/2 sources")


def test_an_update_without_changes_leaves_no_line_behind(tmp_path: Path) -> None:
    _corpus(tmp_path)
    assert _run(tmp_path, "index").returncode == 0

    status, terminal, _ = _run_on_terminal(tmp_path, "update")

    assert status == 0
    assert "Updating" not in terminal


def test_reencode_shows_the_scan_and_the_writing_apart(tmp_path: Path) -> None:
    _corpus(tmp_path)
    assert _run(tmp_path, "index").returncode == 0

    status, terminal, stdout = _run_on_terminal(
        tmp_path, "reencode-sparse", "--yes", "--k1", "1.5"
    )

    assert status == 0, terminal
    scanning = [line for line in _lines(terminal) if line.startswith("Scanning")]
    writing = [line for line in _lines(terminal) if line.startswith("Re-encoding")]
    assert scanning and writing
    assert scanning[-1].startswith("Scanning  100%")
    assert writing[-1].startswith("Re-encoding  100%")
    assert terminal.index(scanning[-1]) < terminal.index(writing[0])


@pytest.mark.parametrize("mode", ["auto", "log", "off"])
def test_the_option_is_accepted_by_every_long_command(
    mode: str, tmp_path: Path
) -> None:
    _corpus(tmp_path)

    assert _run(tmp_path, "index", f"--progress={mode}").returncode == 0
    assert _run(tmp_path, "update", f"--progress={mode}").returncode == 0
    reencode = _run(tmp_path, "reencode-sparse", f"--progress={mode}")
    assert reencode.returncode == 0, reencode.stderr


def test_an_unknown_mode_is_a_usage_error(tmp_path: Path) -> None:
    result = _run(tmp_path, "index", "--progress=loud")

    assert result.returncode == 2
    assert "invalid choice" in result.stderr
