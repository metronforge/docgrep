"""PDF parser preserving page locations and available document metadata."""

import re
from dataclasses import replace
from pathlib import Path

import pymupdf

from docgrep.domain.errors import ParseError
from docgrep.domain.models import FileType, ParsedDocument, ParsedSection, Source

_PDF_DATE_YEAR = re.compile(r"^D:(\d{4})")


def _metadata_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _creation_year(value: str | None) -> int | None:
    if value is None:
        return None
    match = _PDF_DATE_YEAR.match(value)
    return int(match.group(1)) if match else None


class PdfParser:
    """Extract each PDF page's text and mark documents without text for OCR."""

    def __init__(self, corpus_root: Path) -> None:
        self.corpus_root = corpus_root

    def parse(self, source: Source) -> ParsedDocument:
        """Return text-bearing pages with 1-based locations and PDF metadata."""
        if source.file_type is not FileType.PDF:
            raise ParseError("source is not PDF", details={"path": source.path})
        try:
            # PyMuPDF's Document constructor lacks complete type annotations.
            with pymupdf.open(self.corpus_root / source.path) as pdf:  # type: ignore[no-untyped-call]
                metadata = pdf.metadata or {}
                sections = tuple(
                    ParsedSection(text=text, page_number=page_number)
                    for page_number in range(1, pdf.page_count + 1)
                    if (text := pdf.load_page(page_number - 1).get_text("text")).strip()
                )
                enriched = replace(
                    source,
                    title=_metadata_text(metadata.get("title")),
                    author=_metadata_text(metadata.get("author")),
                    year=_creation_year(metadata.get("creationDate")),
                )
        except (OSError, RuntimeError, ValueError) as error:
            raise ParseError(
                "PDF cannot be parsed",
                details={"path": source.path, "reason": str(error)},
            ) from error
        return ParsedDocument(
            source=enriched, sections=sections, requires_ocr=not sections
        )
