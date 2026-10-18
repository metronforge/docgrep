"""The OCR progress line on a pseudo-terminal, by pages."""

import fcntl
import io
import json
import os
import pty
import re
import struct
import sys
import termios
from collections.abc import Callable
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from docgrep.api.cli import main
from docgrep.api.container import create_process_scans
from docgrep.settings import Settings
from docgrep.usecases.process_scans import ProcessScans
from ocr_support import CopyConverter, TextLayerEngine
from pdf_support import pdf_bytes, write_pdf

COLUMNS = 100


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "corpus"
    write_pdf(root / "books" / "scan.pdf", pdf_bytes([None, None, None]))
    write_pdf(root / "books" / "other scan.pdf", pdf_bytes([None]))
    monkeypatch.setenv("DOCGREP_CORPUS_ROOT", str(root))
    monkeypatch.setenv("DOCGREP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DOCGREP_BACKEND", "fake")
    return root


def _factory() -> Callable[[Settings], ProcessScans]:
    def create(settings: Settings) -> ProcessScans:
        use_case = create_process_scans(settings)
        use_case.engine = TextLayerEngine(settings.corpus_root)
        use_case.converter = CopyConverter(settings.corpus_root)
        use_case.poll_seconds = 0
        return use_case

    return create


def _on_terminal(monkeypatch: pytest.MonkeyPatch, *args: str) -> tuple[int, str, str]:
    """Run ``docgrep ocr`` with standard error on a pseudo-terminal."""
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, COLUMNS, 0, 0))
    terminal = os.fdopen(slave, "w")
    out = io.StringIO()
    monkeypatch.setattr(sys, "stderr", terminal)
    try:
        with redirect_stdout(out):
            status = main(["ocr", *args], ocr_factory=_factory())
        terminal.flush()
    finally:
        monkeypatch.undo()
        terminal.close()
    chunks: list[bytes] = []
    while True:
        try:
            data = os.read(master, 65536)
        except OSError:
            break
        if not data:
            break
        chunks.append(data)
    os.close(master)
    return status, b"".join(chunks).decode(), out.getvalue()


def test_ocr_draws_pages_and_leaves_a_final_line(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status, terminal, out = _on_terminal(monkeypatch, "--all-discovered")

    assert status == 0, terminal
    lines = [p for p in terminal.replace("\r\n", "\n").split("\r") if p.strip()]
    ocr_lines = [line.strip() for line in lines if line.startswith("OCR")]
    assert ocr_lines
    assert ocr_lines[-1].startswith("OCR  100%  4/4 pages")
    assert "ETA" not in ocr_lines[-1]
    assert "[1/2] processed" in out
    assert re.search(r"OCR  100%  4/4 pages *\r?\n", terminal)


def test_json_never_draws(corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    status, terminal, out = _on_terminal(monkeypatch, "--all-discovered", "--json")

    assert status == 0, terminal
    assert "OCR" not in terminal
    assert json.loads(out)["totals"]["processed"] == 2


def test_progress_log_and_off_never_draw(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status, terminal, _ = _on_terminal(
        monkeypatch, "--all-discovered", "--progress=off"
    )

    assert status == 0
    assert "OCR  " not in terminal
