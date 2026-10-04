"""PDF parser preserving page locations and available document metadata."""

import logging
import re
from dataclasses import replace
from pathlib import Path

import pymupdf

from docgrep.adapters.parsers.text_quality import classify_page, remove_soft_hyphens
from docgrep.domain.errors import ParseError
from docgrep.domain.models import (
    GARBLED_TEXT_LAYER,
    FileType,
    ParsedDocument,
    ParsedSection,
    ParseQuality,
    Source,
)

_LOG = logging.getLogger(__name__)
_PDF_DATE_YEAR = re.compile(r"^D:(\d{4})")
# When at least this share of the pages that have text are garbled (R1), the
# text layer as a whole is unusable and the document is left to OCR.
GARBLED_DOCUMENT_SHARE = 0.5
# When at least this share of the pages that have text fire any page rule
# (R1, or the weak signals R2 and R3), the document is left to OCR as well:
# weak signals on nearly every page mean a broken layer, not a few tables of
# contents or pages of code.
ANY_RULE_DOCUMENT_SHARE = 0.8


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
                kept: list[ParsedSection] = []
                with_text = 0
                dropped: list[int] = []
                any_rule = long_tokens = low_alnum = 0
                soft_hyphens = 0
                for page_number in range(1, pages_total + 1):
                    raw = pdf.load_page(page_number - 1).get_text("text")
                    text, removed = remove_soft_hyphens(raw)
                    soft_hyphens += removed
                    if not text.strip():
                        continue
                    with_text += 1
                    signals = classify_page(text)
                    any_rule += signals.any_rule
                    long_tokens += signals.long_tokens
                    low_alnum += signals.low_alnum
                    if signals.garbled:
                        dropped.append(page_number)
                        continue
                    kept.append(ParsedSection(text=text, page_number=page_number))
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
        garbled = len(dropped)
        unusable = (garbled > 0 and garbled >= GARBLED_DOCUMENT_SHARE * with_text) or (
            any_rule > 0 and any_rule >= ANY_RULE_DOCUMENT_SHARE * with_text
        )
        quality = ParseQuality(
            pages_total=pages_total,
            pages_with_text=with_text,
            pages_garbled=garbled,
            soft_hyphens_removed=soft_hyphens,
            skip_reason=GARBLED_TEXT_LAYER if unusable else None,
            pages_rule_broken=garbled,
            pages_rule_long_tokens=long_tokens,
            pages_rule_low_alnum=low_alnum,
            dropped_pages=tuple(dropped),
        )
        if unusable:
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
        sections = tuple(kept)
        return ParsedDocument(
            source=enriched,
            sections=sections,
            requires_ocr=not sections,
            quality=quality,
        )
