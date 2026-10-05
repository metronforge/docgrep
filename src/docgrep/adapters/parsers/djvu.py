"""DjVu parser reading an existing text layer through the djvulibre tools."""

import logging
from dataclasses import replace
from pathlib import Path

from docgrep.adapters.djvulibre import (
    CommandRunner,
    DjvuError,
    SubprocessRunner,
    page_texts,
)
from docgrep.adapters.parsers.page_policy import apply_page_policy
from docgrep.adapters.tool_output import scrub_tool_output
from docgrep.domain.errors import ParseError
from docgrep.domain.models import (
    GARBLED_TEXT_LAYER,
    NO_TEXT_LAYER,
    FileType,
    ParsedDocument,
    Source,
)

_LOG = logging.getLogger(__name__)


class DjvuParser:
    """Extract each DjVu page's hidden text with 1-based page numbers.

    The same text hygiene and page policy as for PDF apply (see
    ``page_policy``). A file without any text is left to OCR with the reason
    ``no_text_layer``; an unusable text layer with ``garbled_text_layer``.
    Failures, including missing tools, raise ``ParseError`` with a reason
    code, so the source is recorded as failed and retried by the next update.
    """

    # Recorded with every source it parses; a change makes `docgrep update`
    # extract the sources of this type again.
    version = "1"

    def __init__(self, corpus_root: Path, runner: CommandRunner | None = None) -> None:
        self.corpus_root = corpus_root
        self.runner: CommandRunner = runner or SubprocessRunner()

    def parse(self, source: Source) -> ParsedDocument:
        """Return text-bearing pages; title, author and year are not known."""
        if source.file_type is not FileType.DJVU:
            raise ParseError("source is not DjVu", details={"path": source.path})
        path = (self.corpus_root / source.path).resolve()
        try:
            texts = page_texts(self.runner, path)
        except DjvuError as error:
            raise ParseError(
                scrub_tool_output(
                    f"DjVu cannot be read ({error.reason}): {error}",
                    (path, self.corpus_root),
                ),
                details={"path": source.path, "reason": error.reason},
            ) from error
        sections, quality = apply_page_policy(
            ((number, text) for number, text in enumerate(texts, start=1)),
            len(texts),
        )
        if quality.skip_reason == GARBLED_TEXT_LAYER:
            return ParsedDocument(
                source=source, sections=(), requires_ocr=True, quality=quality
            )
        if not sections:
            return ParsedDocument(
                source=source,
                sections=(),
                requires_ocr=True,
                quality=replace(quality, skip_reason=NO_TEXT_LAYER),
            )
        if quality.pages_garbled:
            _LOG.warning(
                "djvu_pages_dropped",
                extra={
                    "source_id": source.source_id,
                    "dropped": quality.pages_garbled,
                    "total": quality.pages_total,
                },
            )
        return ParsedDocument(source=source, sections=sections, quality=quality)
