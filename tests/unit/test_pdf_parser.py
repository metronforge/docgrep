from pathlib import Path

import pymupdf
import pytest

from docgrep.adapters.parsers import PdfParser
from docgrep.domain.errors import ParseError
from docgrep.domain.models import FileType, Source
from docgrep.domain.protocols import DocumentParser


def source(path: str, file_type: FileType = FileType.PDF) -> Source:
    return Source(source_id=path, path=path, file_type=file_type)


def test_pdf_parser_preserves_page_numbers_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "book.pdf"
    with pymupdf.open() as pdf:
        pdf.set_metadata(
            {
                "title": "Research Notes",
                "author": "Ada Lovelace",
                "creationDate": "D:18430101000000",
            }
        )
        pdf.new_page().insert_text((72, 72), "First page text.")
        pdf.new_page()
        pdf.new_page().insert_text((72, 72), "Third page text.")
        pdf.save(path)

    parser: DocumentParser = PdfParser(tmp_path)
    document = parser.parse(source("book.pdf"))

    assert document.source.title == "Research Notes"
    assert document.source.author == "Ada Lovelace"
    assert document.source.year == 1843
    assert [(section.page_number, section.text) for section in document.sections] == [
        (1, "First page text.\n"),
        (3, "Third page text.\n"),
    ]
    assert document.requires_ocr is False


def test_image_only_pdf_requires_ocr(tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 2, 2), 0)
    pixmap.clear_with(255)
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_image(page.rect, pixmap=pixmap)
        pdf.save(path)

    document = PdfParser(tmp_path).parse(source("scan.pdf"))

    assert document.requires_ocr is True
    assert document.sections == ()
    assert document.source.title is None
    assert document.source.author is None
    assert document.source.year is None


def test_corrupt_pdf_raises_parse_error(tmp_path: Path) -> None:
    (tmp_path / "bad.pdf").write_bytes(b"not a PDF")

    with pytest.raises(ParseError) as raised:
        PdfParser(tmp_path).parse(source("bad.pdf"))

    assert raised.value.category == "parse_error"
    assert raised.value.details["path"] == "bad.pdf"


def test_pdf_parser_rejects_other_file_type(tmp_path: Path) -> None:
    with pytest.raises(ParseError):
        PdfParser(tmp_path).parse(source("note.txt", FileType.TXT))
