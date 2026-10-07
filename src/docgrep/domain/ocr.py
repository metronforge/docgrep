"""Pure rules of OCR processing (UC-6): eligible files, output names,
engine modes and the outcome vocabulary."""

from collections.abc import Sequence
from pathlib import PurePosixPath

from docgrep.domain.models import FileType, OcrMode
from docgrep.domain.scans import GARBLED_TEXT_LAYER

# Extensions an OCR request may name, and the file type each one has.
ELIGIBLE_EXTENSIONS = {
    ".pdf": FileType.PDF,
    ".djvu": FileType.DJVU,
    ".djv": FileType.DJVU,
}
# The searchable PDF is written next to the original as <stem> + this suffix.
OUTPUT_SUFFIX = ".ocr.pdf"
# Free space required in the output directory, as a multiple of the input size.
DISK_SPACE_FACTOR = 3
# Seconds a terminated engine gets to exit before it is killed.
TERMINATE_GRACE_SECONDS = 10.0

SKIP_TEXT: OcrMode = "skip-text"
FORCE_OCR: OcrMode = "force-ocr"

PROCESSED = "processed"
CANCELLED = "cancelled"
ALREADY_PROCESSED = "already_processed"
DOES_NOT_NEED_OCR = "does_not_need_ocr"

# Per-file failure reasons.
ENCRYPTED = "encrypted"
UNREADABLE = "unreadable"
DJVU_CONVERSION = "djvu_conversion"
DJVU_TOOLS_MISSING = "djvu_tools_missing"
ENGINE_ERROR = "ocr_engine_error"
TIMEOUT = "timeout"
OUTPUT_INVALID = "output_invalid"
OUTPUT_COLLISION = "output_collision"
REGISTRY_WRITE = "registry_write"
DISK_FULL = "disk_full"

INSTALL_COMMAND = (
    "apt install ocrmypdf tesseract-ocr tesseract-ocr-eng "
    "tesseract-ocr-rus tesseract-ocr-ukr"
)


def eligible_type(path: str) -> FileType | None:
    """The file type of a path an OCR request may name, or None."""
    return ELIGIBLE_EXTENSIONS.get(PurePosixPath(path).suffix.lower())


def output_path(path: str) -> str:
    """Where the searchable PDF of ``path`` goes: ``<stem>.ocr.pdf`` beside it."""
    source = PurePosixPath(path)
    return str(source.with_name(source.stem + OUTPUT_SUFFIX))


def ocr_mode(reason: str | None, *, override_native: bool) -> OcrMode:
    """The engine mode for a file classified with ``reason``.

    A garbled text layer must be discarded and rebuilt from the page images,
    and so must the layer of a native file processed on request; every other
    file (no or little text, a DjVu conversion) keeps pages that have text.
    """
    if override_native or reason == GARBLED_TEXT_LAYER:
        return FORCE_OCR
    return SKIP_TEXT


def parse_languages(value: str) -> tuple[str, ...]:
    """The language codes of a ``+``-separated list such as ``rus+eng+ukr``.

    Raises ValueError for an empty list or a code that is not lower-case
    letters, digits and underscores.
    """
    codes = tuple(value.strip().split("+"))
    for code in codes:
        if not code or not all(
            c.isascii() and (c.islower() or c.isdigit() or c == "_") for c in code
        ):
            raise ValueError(f"invalid language code: {code!r}")
    return codes


def eta_seconds(
    *, done_bytes: int, total_bytes: int, elapsed_seconds: float
) -> float | None:
    """Seconds left when the remaining input is processed at the pace so far,
    in input bytes; None until a byte is done."""
    if done_bytes <= 0 or elapsed_seconds <= 0:
        return None
    return max(total_bytes - done_bytes, 0) * elapsed_seconds / done_bytes


def missing_languages(
    requested: Sequence[str], available: Sequence[str]
) -> tuple[str, ...]:
    """The requested languages the engine does not have, in request order."""
    have = set(available)
    return tuple(code for code in requested if code not in have)
