"""Scan probes for PDF (PyMuPDF) and DjVu (djvulibre), read-only (UC-5)."""

import logging
from collections.abc import Sequence
from pathlib import Path

import pymupdf

from docgrep.adapters.djvulibre import (
    DJVUTXT,
    CommandRunner,
    DjvuError,
    SubprocessRunner,
    parse_page_texts,
    run_tool,
)
from docgrep.adapters.djvulibre import (
    INSTALL_HINT as DJVU_INSTALL_HINT,
)
from docgrep.adapters.djvulibre import (
    page_count as djvu_page_count,
)
from docgrep.adapters.parsers.text_quality import classify_page, remove_soft_hyphens
from docgrep.domain.errors import InspectionFailed
from docgrep.domain.models import FileType
from docgrep.domain.scans import PageSample

_LOG = logging.getLogger(__name__)

ENCRYPTED = "encrypted"
CORRUPT_HEADER = "corrupt_header"


def _page(page_number: int, raw: str) -> PageSample:
    text, _ = remove_soft_hyphens(raw)
    return PageSample(
        page_number=page_number,
        non_whitespace=sum(1 for character in text if not character.isspace()),
        dropped=classify_page(text).garbled,
    )


def _failed(path: str, reason: str, message: str) -> InspectionFailed:
    details: dict[str, object] = {"path": path, "reason": reason}
    if reason == "djvu_tools_missing":
        details["hint"] = DJVU_INSTALL_HINT
    return InspectionFailed(message, details=details)


class FileScanProbe:
    """Count characters on sampled pages of corpus files; never writes."""

    def __init__(self, corpus_root: Path, runner: CommandRunner | None = None) -> None:
        self.corpus_root = corpus_root
        self.runner: CommandRunner = runner or SubprocessRunner()

    def _absolute(self, path: str) -> Path:
        return (self.corpus_root / path).resolve()

    def _open_pdf(self, path: str) -> pymupdf.Document:
        try:
            # PyMuPDF's Document constructor lacks complete type annotations.
            document = pymupdf.open(self._absolute(path))  # type: ignore[no-untyped-call]
        except Exception as error:
            raise _failed(
                path, CORRUPT_HEADER, f"PDF cannot be opened: {error}"
            ) from error
        if document.needs_pass:
            document.close()  # type: ignore[no-untyped-call]  # PyMuPDF lacks annotations here
            raise _failed(path, ENCRYPTED, "PDF is encrypted")
        return document

    def page_count(self, path: str, file_type: FileType) -> int:
        if file_type is FileType.PDF:
            with self._open_pdf(path) as document:
                return int(document.page_count)
        if file_type is FileType.DJVU:
            try:
                return djvu_page_count(self.runner, self._absolute(path))
            except DjvuError as error:
                raise _failed(path, error.reason, str(error)) from error
        raise _failed(path, CORRUPT_HEADER, f"not a PDF or DjVu file: {file_type}")

    def sample(
        self, path: str, file_type: FileType, pages: Sequence[int]
    ) -> tuple[PageSample | None, ...]:
        if file_type is FileType.PDF:
            return self._sample_pdf(path, pages)
        if file_type is FileType.DJVU:
            return self._sample_djvu(path, pages)
        raise _failed(path, CORRUPT_HEADER, f"not a PDF or DjVu file: {file_type}")

    def _sample_pdf(
        self, path: str, pages: Sequence[int]
    ) -> tuple[PageSample | None, ...]:
        samples: list[PageSample | None] = []
        with self._open_pdf(path) as document:
            for number in pages:
                try:
                    raw = document.load_page(number - 1).get_text("text")
                except Exception:
                    _LOG.debug("scan_page_failed", extra={"page": number})
                    samples.append(None)
                    continue
                samples.append(_page(number, raw))
        return tuple(samples)

    def _sample_djvu(
        self, path: str, pages: Sequence[int]
    ) -> tuple[PageSample | None, ...]:
        if not pages:
            return ()
        args = [
            DJVUTXT,
            "--detail=page",
            "--page=" + ",".join(str(number) for number in pages),
            str(self._absolute(path)),
        ]
        try:
            texts = parse_page_texts(run_tool(self.runner, args))
        except DjvuError as error:
            raise _failed(path, error.reason, str(error)) from error
        if len(texts) != len(pages):
            raise _failed(
                path, "djvu_invalid", "djvutxt returned a different page count"
            )
        return tuple(
            _page(number, text) for number, text in zip(pages, texts, strict=True)
        )
