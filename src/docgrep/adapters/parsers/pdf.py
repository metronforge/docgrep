"""PDF parser preserving page locations and available document metadata."""

import logging
import re
from dataclasses import replace
from pathlib import Path

import pymupdf

from docgrep.adapters.parsers.page_policy import (
    ANY_RULE_DOCUMENT_SHARE,
    GARBLED_DOCUMENT_SHARE,
    apply_page_policy,
)
from docgrep.adapters.tool_output import scrub_tool_output
from docgrep.domain.errors import ParseError
from docgrep.domain.models import (
    GARBLED_TEXT_LAYER,
    FileType,
    ParsedDocument,
    Source,
)

_LOG = logging.getLogger(__name__)
_PDF_DATE_YEAR = re.compile(r"^D:(\d{4})")
# The document shares live with the shared page policy; named here too.
__all__ = ["ANY_RULE_DOCUMENT_SHARE", "GARBLED_DOCUMENT_SHARE", "PdfParser"]


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
    """Extract each PDF page's text and mark documents without text for OCR.

    Page text is verbatim except for soft hyphens (see ``remove_soft_hyphens``).
    Pages whose text layer is garbled (R1 of ``classify_page``) are left out.
    The whole document requires OCR, with ``garbled_text_layer`` as the
    reason, when garbled pages are at least ``GARBLED_DOCUMENT_SHARE`` of the
    pages with text, or when pages firing any rule, weak signals included,
    are at least ``ANY_RULE_DOCUMENT_SHARE`` of them.
    """

    # Recorded with every source it parses; a change makes `docgrep update`
    # extract the sources of this type again. 2: soft hyphens and garbled
    # pages are handled. 3: only broken characters drop a page.
    version = "3"

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
                pages_total = pdf.page_count
                sections, quality = apply_page_policy(
                    (
                        (number, pdf.load_page(number - 1).get_text("text"))
                        for number in range(1, pages_total + 1)
                    ),
                    pages_total,
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
                details={
                    "path": source.path,
                    "reason": scrub_tool_output(
                        str(error), (self.corpus_root / source.path, self.corpus_root)
                    ),
                },
            ) from error
        garbled = quality.pages_garbled
        if quality.skip_reason == GARBLED_TEXT_LAYER:
            return ParsedDocument(
                source=enriched, sections=(), requires_ocr=True, quality=quality
            )
        if garbled:
            _LOG.warning(
                "pdf_pages_dropped",
                extra={
                    "source_id": source.source_id,
                    "dropped": garbled,
                    "total": pages_total,
                },
            )
        return ParsedDocument(
            source=enriched,
            sections=sections,
            requires_ocr=not sections,
            quality=quality,
        )
