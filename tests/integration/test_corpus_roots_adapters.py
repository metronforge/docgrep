"""The corpus, parser and OCR registry adapters over several roots
(ADR-0023)."""

from pathlib import Path

import pytest
from blake3 import blake3

from docgrep.adapters.corpus import (
    describe_source,
    enumerate_corpus,
    require_available,
    scan_corpus,
)
from docgrep.adapters.manifest import SQLiteOcrRegistry
from docgrep.adapters.parsers.text import TextParser
from docgrep.domain.errors import ValidationError
from docgrep.domain.models import OcrRecord
from docgrep.domain.roots import CorpusRoot, CorpusRoots


def _roots(tmp_path: Path) -> CorpusRoots:
    books, notes = tmp_path / "books", tmp_path / "notes"
    for folder, files in (
        (books, {"a.txt": "rare blue comet", "shelf/b.md": "# moss"}),
        (notes, {"a.txt": "yellow kite meadow", "c.txt": "rye bread"}),
    ):
        for name, text in files.items():
            (folder / name).parent.mkdir(parents=True, exist_ok=True)
            (folder / name).write_text(text, encoding="utf-8")
    return CorpusRoots(
        (CorpusRoot("books", books, legacy=True), CorpusRoot("notes", notes))
    )


def test_enumeration_and_scan_store_prefixed_paths(tmp_path: Path) -> None:
    roots = _roots(tmp_path)

    sources = enumerate_corpus(roots, exclude_globs=("c.txt",))
    scanned = scan_corpus(roots, exclude_globs=("c.txt",))

    assert [s.source_id for s in sources] == ["a.txt", "notes/a.txt", "shelf/b.md"]
    assert [(s.path, s.excluded_by) for s in scanned] == [
        ("a.txt", None),
        ("notes/a.txt", None),
        ("notes/c.txt", "c.txt"),
        ("shelf/b.md", None),
    ]
    assert describe_source(roots, "notes/a.txt") == sources[1]


def test_a_single_legacy_root_enumerates_as_a_bare_folder(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    legacy = roots.only(("books",))

    assert enumerate_corpus(legacy) == enumerate_corpus(tmp_path / "books")


def test_an_unreadable_root_is_named_before_any_file_is_listed(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    for file in (tmp_path / "notes").iterdir():
        file.unlink()
    (tmp_path / "notes").rmdir()

    with pytest.raises(ValidationError) as raised:
        require_available(roots)
    with pytest.raises(ValidationError) as walked:
        enumerate_corpus(roots)

    assert str(raised.value) == (
        f"corpus root 'notes' is not accessible: {tmp_path / 'notes'}"
    )
    assert raised.value.details["root"] == "notes"
    assert walked.value.details["root"] == "notes"
    with pytest.raises(ValidationError, match="the root of the source"):
        describe_source(roots.only(("books",)), "notes/a.txt")


def test_no_roots_is_a_configuration_error() -> None:
    with pytest.raises(ValidationError, match="no corpus roots configured"):
        require_available(CorpusRoots(()))


def test_parsers_read_a_source_through_its_root(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    sources = {s.path: s for s in enumerate_corpus(roots)}

    parsed = TextParser(roots).parse(sources["notes/a.txt"])

    assert [section.text for section in parsed.sections] == ["yellow kite meadow"]


def _record(folder: Path, stored: str, original: str, output: str) -> OcrRecord:
    for name in (original, output):
        (folder / name).write_bytes(f"%PDF {name}".encode())
    return OcrRecord(
        input_path=f"{stored}{original}",
        input_format="pdf",
        input_hash=blake3((folder / original).read_bytes()).hexdigest(),
        output_path=f"{stored}{output}",
        output_hash=blake3((folder / output).read_bytes()).hexdigest(),
        engine="ocrmypdf",
        engine_version="16.0.0",
        languages="eng",
        mode="skip-text",
        created_at="2026-01-01T00:00:00+00:00",
    )


def test_registry_outputs_are_live_only_under_a_configured_root(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    data = tmp_path / "data"
    registry = SQLiteOcrRegistry(data, roots)
    registry.register(_record(tmp_path / "notes", "notes/", "scan.pdf", "scan.ocr.pdf"))
    registry.register(_record(tmp_path / "books", "", "old.pdf", "old.ocr.pdf"))

    assert sorted(registry.live_outputs()) == ["notes/scan.ocr.pdf", "old.ocr.pdf"]
    books_only = SQLiteOcrRegistry(data, roots.only(("books",)), read_only=True)
    assert sorted(books_only.live_outputs()) == ["old.ocr.pdf"]
