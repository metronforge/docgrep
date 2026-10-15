"""The status sources block over a real SQLite manifest (UC-7)."""

import io
import json
import sqlite3
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import pytest

from docgrep.api.cli import main
from docgrep.api.container import create_status
from epub_support import AES, encryption_xml, epub_bytes, write_epub, xhtml
from pdf_support import broken, pdf_bytes, prose, write_pdf
from resume_support import close, settings_for
from update_support import update_services


def _corpus(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "a.txt").write_text("marker1 plain words", encoding="utf-8")
    (root / "b.md").write_text("# Title\n\nmarker2 words\n", encoding="utf-8")
    write_pdf(root / "clean.pdf", pdf_bytes([prose(1)]))
    write_pdf(root / "layer.pdf", pdf_bytes([broken(1), broken(2), prose(2)]))
    write_pdf(root / "scan.pdf", pdf_bytes([None]))
    (root / "letter.doc").write_bytes(b"not parsed yet")
    write_epub(
        root / "locked.epub",
        epub_bytes([("a.xhtml", xhtml("<p>x</p>"))], encryption=encryption_xml(AES)),
    )


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
    }


@pytest.fixture
def built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    _corpus(tmp_path / "corpus")
    settings = settings_for(tmp_path / "corpus", tmp_path / "data")
    services = update_services(settings)
    services.build.run()
    close(services)
    monkeypatch.setenv("DOCGREP_CORPUS_ROOT", str(settings.corpus_root))
    monkeypatch.setenv("DOCGREP_DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("DOCGREP_BACKEND", "fake")
    monkeypatch.setenv("DOCGREP_CHUNK_SIZE", "10")
    monkeypatch.setenv("DOCGREP_CHUNK_OVERLAP", "2")
    return settings


def _cli(*args: str) -> str:
    out = io.StringIO()
    with redirect_stdout(out), redirect_stderr(io.StringIO()):
        assert main(["status", *args]) == 0
    return out.getvalue()


def test_the_block_describes_the_sources_table(built: Any) -> None:
    sources = create_status(built).run().as_dict()["sources"]

    assert sources["total"] == 7
    assert sources["by_state"] == {"indexed": 3, "skipped": 3, "failed": 1}
    assert sources["skipped_by_reason"] == {
        "requires_ocr": 1,
        "garbled_text_layer": 1,
        "unsupported": 1,
        "other": 0,
    }
    assert sources["skipped_by_extension"] == [
        {"extension": ".pdf", "sources": 2},
        {"extension": ".doc", "sources": 1},
    ]
    (failed,) = sources["failed"]
    assert (failed["path"], failed["category"]) == ("locked.epub", "parse_error")
    assert failed["reason"]
    assert sources["parser_changed"] == 0


def test_status_writes_nothing(built: Any) -> None:
    before = _snapshot(built.data_dir)

    create_status(built).run()
    _cli()
    _cli("--json")

    assert _snapshot(built.data_dir) == before


def test_legacy_and_changed_versions_are_counted_like_update(built: Any) -> None:
    with sqlite3.connect(built.data_dir / "manifest.sqlite3") as connection:
        connection.execute(
            "UPDATE sources SET parser_version=NULL WHERE path IN ('a.txt', 'b.md')"
        )
        connection.execute(
            "UPDATE sources SET parser_version='2' "
            "WHERE path IN ('clean.pdf', 'scan.pdf')"
        )

    sources = create_status(built).run().as_dict()["sources"]
    text = _cli()

    assert sources["parser_changed"] == 2
    assert "Sources: 7 total, 3 indexed, 3 skipped, 1 failed; 2 to re-extract" in text


def test_text_and_json_output(built: Any) -> None:
    text = _cli()
    data = json.loads(_cli("--json"))

    assert "Sources: 7 total, 3 indexed, 3 skipped, 1 failed\n" in text
    assert (
        "Skipped by reason: requires_ocr 1, garbled_text_layer 1, unsupported 1, "
        "other 0"
    ) in text
    assert "Failed sources:\n  locked.epub: parse_error: " in text
    assert data["sources"]["by_state"] == {"indexed": 3, "skipped": 3, "failed": 1}


def test_an_unreadable_manifest_gives_no_block(built: Any) -> None:
    (built.data_dir / "manifest.sqlite3").write_bytes(b"garbage" * 100)

    report = create_status(built).run()
    text = _cli()

    assert report.index_state == "manifest_unreadable"
    assert report.as_dict()["sources"] is None
    assert "Sources:" not in text and "Index: manifest_unreadable" in text
