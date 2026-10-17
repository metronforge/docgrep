"""OCR across corpus roots: outputs stay next to their originals (ADR-0023)."""

from pathlib import Path

import pytest

from docgrep.adapters.manifest import SQLiteOcrRegistry
from docgrep.adapters.ocrmypdf import LocalOcrFiles
from docgrep.api.container import create_process_scans
from docgrep.domain.errors import ValidationError
from docgrep.domain.ocr import output_path
from docgrep.domain.roots import CorpusRoot, CorpusRoots
from docgrep.usecases.process_scans import OcrRequest
from ocr_support import CopyConverter, TextLayerEngine
from pdf_support import pdf_bytes, write_pdf
from roots_support import settings


class _RootsFolder:
    """A stand-in for a corpus folder that resolves stored paths through the
    roots, for the test engines that join paths onto a folder."""

    def __init__(self, roots: CorpusRoots) -> None:
        self.roots = roots

    def __truediv__(self, stored: str) -> Path:
        located = self.roots.locate(stored)
        assert located is not None
        return located


@pytest.fixture
def roots(tmp_path: Path) -> CorpusRoots:
    books, notes = tmp_path / "books", tmp_path / "notes"
    write_pdf(books / "scan.pdf", pdf_bytes([None, None]))
    write_pdf(notes / "scans" / "scan.pdf", pdf_bytes([None]))
    return CorpusRoots(
        (
            CorpusRoot("books", books.resolve(), legacy=True),
            CorpusRoot("notes", notes.resolve()),
        )
    )


def test_outputs_land_next_to_their_originals_in_each_root(
    tmp_path: Path, roots: CorpusRoots
) -> None:
    data = tmp_path / "data"
    use_case = create_process_scans(settings(data).with_corpus_roots(roots))
    folder = _RootsFolder(roots)
    use_case.engine = TextLayerEngine(folder)
    use_case.converter = CopyConverter(folder)
    use_case.poll_seconds = 0

    summary = use_case.run(OcrRequest(all_discovered=True))

    assert summary.as_dict()["totals"]["processed"] == 2
    assert (tmp_path / "books" / "scan.ocr.pdf").is_file()
    assert (tmp_path / "notes" / "scans" / "scan.ocr.pdf").is_file()
    assert not (tmp_path / "notes" / "notes").exists()
    registry = SQLiteOcrRegistry(data, roots, read_only=True)
    assert set(registry.live_outputs()) == {"scan.ocr.pdf", "notes/scans/scan.ocr.pdf"}
    assert registry.output_owner("notes/scans/scan.ocr.pdf") == "notes/scans/scan.pdf"
    assert registry.live_delegation("notes/scans/scan.pdf") is not None
    # Without the root configured its output is not live: root not available.
    books_only = roots.only({"books"})
    alone = SQLiteOcrRegistry(data, books_only, read_only=True)
    assert set(alone.live_outputs()) == {"scan.ocr.pdf"}
    assert alone.live_delegation("notes/scans/scan.pdf") is None


@pytest.mark.parametrize("stored", ["scan.pdf", "notes/scans/scan.pdf", "notes/x.djvu"])
def test_an_output_always_has_the_root_of_its_original(
    roots: CorpusRoots, stored: str
) -> None:
    assert roots.root_of(output_path(stored)) == roots.root_of(stored)


def test_ocr_paths_are_located_in_their_root(
    tmp_path: Path, roots: CorpusRoots
) -> None:
    files = LocalOcrFiles(roots)
    assert files.locate("notes/scans/scan.pdf") == "notes/scans/scan.pdf"
    assert files.locate(str(tmp_path / "notes" / "scans" / "scan.pdf")) == (
        "notes/scans/scan.pdf"
    )
    assert files.locate(str(tmp_path / "books" / "scan.pdf")) == "scan.pdf"
    for outside in ("notes/../../books/scan.pdf", str(tmp_path / "elsewhere.pdf")):
        with pytest.raises(ValidationError, match="outside the corpus root"):
            files.locate(outside)
