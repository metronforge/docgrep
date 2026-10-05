"""A failed DjVu source is stored and shown without its absolute path (UC-7)."""

import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import pytest

from djvu_support import HAVE_DJVU_TOOLS, FakeRunner
from docgrep.adapters.djvulibre import CommandResult
from docgrep.adapters.manifest import SQLiteManifest
from docgrep.adapters.parsers import DjvuParser
from docgrep.api.cli import main
from docgrep.domain.models import FileType
from resume_support import close, settings_for
from update_support import update_services

RELATIVE = "shelf/bad book.djvu"


def _corpus(tmp_path: Path) -> Path:
    root = tmp_path / "my corpus"
    (root / "shelf").mkdir(parents=True)
    (root / "note.txt").write_text("marker1 plain words here", encoding="utf-8")
    (root / RELATIVE).write_bytes(b"not a djvu file at all\n")
    return root


def _build(root: Path, data: Path, runner: Any | None) -> Any:
    settings = settings_for(root, data)
    services = update_services(settings)
    if runner is not None:
        build: Any = services.build
        build.parsers = {
            **build.parsers,
            FileType.DJVU: DjvuParser(settings.corpus_root, runner),
        }
    summary = services.build.run()
    close(services)
    assert (summary.indexed, summary.failed) == (1, 1)
    return settings


def _status(settings: Any, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("DOCGREP_CORPUS_ROOT", str(settings.corpus_root))
    monkeypatch.setenv("DOCGREP_DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("DOCGREP_BACKEND", "fake")
    monkeypatch.setenv("DOCGREP_CHUNK_SIZE", "10")
    monkeypatch.setenv("DOCGREP_CHUNK_OVERLAP", "2")
    out = io.StringIO()
    with redirect_stdout(out), redirect_stderr(io.StringIO()):
        assert main(["status"]) == 0
    return out.getvalue()


def _check(settings: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = SQLiteManifest(settings.data_dir).get_source(RELATIVE)
    assert record is not None and record.status == "failed:parse_error"
    assert record.error is not None and "djvu_invalid" in record.error
    assert str(tmp_path) not in record.error and "my%20corpus" not in record.error

    text = _status(settings, monkeypatch)

    lines = [line for line in text.splitlines() if RELATIVE in line]
    assert len(lines) == 1
    assert lines[0].startswith(f"  {RELATIVE}: parse_error: DjVu cannot be read")
    assert lines[0].count("bad book") == 1
    assert str(tmp_path) not in text and "my%20corpus" not in text


def test_tool_output_with_the_absolute_path_is_scrubbed_in_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _corpus(tmp_path)
    absolute = (root / RELATIVE).resolve()
    stderr = (
        f"*** Failed to open '{absolute}': bad.\n{absolute}: not ready\n"
        f"*** file://localhost{str(absolute).replace(' ', '%20')}\n"
    ).encode()
    runner = FakeRunner({"bad book.djvu": CommandResult(10, b"", stderr)})

    settings = _build(root, tmp_path / "data", runner)

    _check(settings, tmp_path, monkeypatch)


@pytest.mark.skipif(not HAVE_DJVU_TOOLS, reason="the djvulibre tools are not installed")
def test_real_tool_failure_is_scrubbed_in_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _build(_corpus(tmp_path), tmp_path / "data", None)

    _check(settings, tmp_path, monkeypatch)
