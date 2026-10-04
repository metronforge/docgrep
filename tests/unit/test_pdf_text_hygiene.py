"""PDF soft hyphens and garbled text layers on generated PDFs (UC-3, UC-5)."""

import logging
from pathlib import Path

import pytest

from docgrep.adapters.chunking import TokenChunker, WhitespaceTokenizer
from docgrep.adapters.parsers import PdfParser
from docgrep.domain.models import FileType, ParsedDocument, Source
from pdf_support import broken, pdf_bytes, prose, write_pdf

SHY = "\u00ad"


def parse(
    root: Path, pages: list[list[str] | None], name: str = "doc.pdf"
) -> ParsedDocument:
    write_pdf(root / name, pdf_bytes(pages))
    return PdfParser(root).parse(Source(name, name, FileType.PDF))


def test_the_parser_declares_version_3() -> None:
    assert PdfParser.version == "3"


def test_soft_hyphens_at_line_ends_join_words(tmp_path: Path) -> None:
    document = parse(
        tmp_path,
        [
            [
                f"The hyphen{SHY}",
                "ation of a well-known co-",
                f"operation and a mid{SHY}word {SHY}  ",
                "end.",
            ]
        ],
    )

    assert [s.text for s in document.sections] == [
        "The hyphenation of a well-known co-\noperation and a midword end.\n"
    ]
    assert document.quality is not None
    assert document.quality.soft_hyphens_removed == 3


def test_the_russian_soft_hyphen_at_a_line_end(tmp_path: Path) -> None:
    document = parse(
        tmp_path, [[f"Это пример пере{SHY}", "носа слов и обычный-дефис."]]
    )

    assert document.sections[0].text == "Это пример переноса слов и обычный-дефис.\n"


def test_text_without_soft_hyphens_is_unchanged(tmp_path: Path) -> None:
    lines = prose(1)

    document = parse(tmp_path, [lines])

    assert document.sections[0].text == "\n".join(lines) + "\n"
    assert document.quality is not None
    assert (document.quality.soft_hyphens_removed, document.quality.pages_garbled) == (
        0,
        0,
    )


def test_garbled_pages_are_dropped_and_logged_without_text(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pages = [prose(1), broken(2), prose(3), None, prose(4)]

    with caplog.at_level(logging.WARNING):
        document = parse(tmp_path, pages)

    assert [s.page_number for s in document.sections] == [1, 3, 5]
    assert not document.requires_ocr
    assert document.quality is not None
    assert (
        document.quality.pages_total,
        document.quality.pages_with_text,
        document.quality.pages_garbled,
        document.quality.skip_reason,
    ) == (5, 4, 1, None)
    (record,) = [r for r in caplog.records if r.getMessage() == "pdf_pages_dropped"]
    assert record.levelno == logging.WARNING
    assert (record.__dict__["source_id"], record.__dict__["dropped"]) == ("doc.pdf", 1)
    assert record.__dict__["total"] == 5
    assert "QWERTY" not in caplog.text and "marker" not in caplog.text


def test_a_mostly_garbled_document_requires_ocr(tmp_path: Path) -> None:
    document = parse(tmp_path, [prose(1), broken(2), None])

    assert document.requires_ocr and document.sections == ()
    assert document.quality is not None
    assert document.quality.skip_reason == "garbled_text_layer"
    assert (document.quality.pages_with_text, document.quality.pages_garbled) == (2, 1)


def test_a_minority_of_garbled_pages_keeps_the_document(tmp_path: Path) -> None:
    document = parse(tmp_path, [prose(1), broken(2), prose(3)])

    assert not document.requires_ocr and len(document.sections) == 2


def test_a_text_less_pdf_still_requires_ocr_without_a_reason(tmp_path: Path) -> None:
    document = parse(tmp_path, [None, None])

    assert document.requires_ocr and document.sections == ()
    assert document.quality is not None and document.quality.skip_reason is None


def test_the_same_file_gives_the_same_passages(tmp_path: Path) -> None:
    pages = [prose(1, lines=12), broken(2), [f"split hyphen{SHY}", "ation"] + prose(3)]
    chunker = TokenChunker(WhitespaceTokenizer(), chunk_size=20, chunk_overlap=4)

    first = chunker.chunk(parse(tmp_path, pages))
    second = chunker.chunk(parse(tmp_path, pages))

    assert first == second and len(first) > 3
    assert {p.page_number for p in first} == {1, 3}
    assert any("hyphenation" in p.text for p in first)
    assert all(SHY not in p.text for p in first)
