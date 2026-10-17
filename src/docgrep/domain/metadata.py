"""Sanitize the bibliographic metadata of a source on read (ADR-0021).

Parsers store titles, authors and years exactly as the files declare them,
and some of those values are junk: epoch dates, template placeholders, file
names. ``sanitize_source`` drops such values, so every front shows and
matches the same cleaned metadata, and records what it dropped and why. The
stored values are never changed. A dropped value becomes None; nothing is
guessed. An OCR output takes a missing or dropped value from its original,
once the original's own value has passed the same rules.

Everything here is pure: the caller reads the manifest and the OCR registry.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Literal, get_args

from docgrep.domain.models import FileType, OcrOutput, Source

DropReason = Literal[
    "implausible_year",
    "filetime_epoch",
    "processing_year",
    "placeholder",
    "numeric_only",
    "file_name",
    "empty",
    "inherited",
]
DROP_REASONS: tuple[str, ...] = get_args(DropReason)

# The earliest plausible publication year: printing with movable type.
EARLIEST_YEAR = 1450
# The FILETIME epoch, read as a year when a zero Windows timestamp is decoded.
FILETIME_EPOCH_YEAR = 1601

AUTHOR_PLACEHOLDERS: frozenset[str] = frozenset(
    {
        # Unknown or withheld authorship.
        "unknown",
        "anonymous",
        "n/a",
        "none",
        "-",
        # Field labels left in templates.
        "author",
        # Operating system and office account names.
        "admin",
        "administrator",
        "user",
        "microsoft office user",
        "windows user",
        # Libraries that write their own name as the author.
        "python-docx",
    }
)

TITLE_PLACEHOLDERS: frozenset[str] = frozenset(
    {
        # Field labels and defaults left in templates.
        "untitled",
        "book title",
        "title",
        "unknown",
        # Default names of new office documents.
        "document",
        "document1",
        "new document",
        # Scanner defaults.
        "scan",
    }
)

# Prefixes that office print drivers put before the document's file name.
TITLE_FILE_PREFIXES: tuple[str, ...] = ("microsoft word - ", "microsoft excel - ")

# Extensions of authoring files, so a title ending in one is a file name.
TITLE_FILE_EXTENSIONS: tuple[str, ...] = (
    ".doc",
    ".docx",
    ".indd",
    ".qxd",
    ".pmd",
    ".tex",
    ".rtf",
    ".cdr",
    ".pdf",
)

# Digits with optional spaces, dots and dashes, and at least one digit.
_NUMERIC_ONLY = re.compile(r"[\d\s.\-]*\d[\d\s.\-]*")


def _normal(text: str) -> str:
    """Stripped, inner whitespace collapsed, case-folded."""
    return " ".join(text.split()).casefold()


@dataclass(frozen=True, slots=True)
class FieldAudit:
    """What happened to one metadata field.

    ``raw`` is the stored value that was dropped, None when nothing was
    stored; ``reason`` says why it was dropped, or is ``inherited`` when a
    missing value was taken from the original; ``inherited_from`` is the
    original's path when the shown value comes from it.
    """

    reason: DropReason
    raw: str | int | None = None
    inherited_from: str | None = None


@dataclass(frozen=True, slots=True)
class MetadataAudit:
    """The fields of a source that were dropped or inherited."""

    title: FieldAudit | None = None
    author: FieldAudit | None = None
    year: FieldAudit | None = None

    @property
    def empty(self) -> bool:
        return self.title is None and self.author is None and self.year is None

    def fields(self) -> tuple[tuple[str, FieldAudit], ...]:
        """The audited fields by name, in a fixed order."""
        found = (("title", self.title), ("author", self.author), ("year", self.year))
        return tuple((name, item) for name, item in found if item is not None)


@dataclass(frozen=True, slots=True)
class SanitizedSource:
    """A source with cleaned metadata and the record of what changed."""

    source: Source
    audit: MetadataAudit = MetadataAudit()


@dataclass(frozen=True, slots=True)
class OcrOrigin:
    """Where an OCR output came from: its registry entry and, when the
    manifest has a row for it, the original source."""

    entry: OcrOutput
    original: Source | None = None


def _file_names(path: str) -> tuple[str, str]:
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0] if "." in name.lstrip(".") else name
    return _normal(name), _normal(stem)


def title_reason(title: str, path: str) -> DropReason | None:
    """Why a title is junk, or None when it is kept."""
    value = _normal(title)
    if not value:
        return "empty"
    if _NUMERIC_ONLY.fullmatch(value):
        return "numeric_only"
    if value in TITLE_PLACEHOLDERS:
        return "placeholder"
    if value.startswith(TITLE_FILE_PREFIXES) or value.endswith(TITLE_FILE_EXTENSIONS):
        return "file_name"
    if value in _file_names(path):
        return "file_name"
    return None


def author_reason(author: str) -> DropReason | None:
    """Why an author is junk, or None when it is kept."""
    value = _normal(author)
    if not value:
        return "empty"
    if _NUMERIC_ONLY.fullmatch(value):
        return "numeric_only"
    if value in AUTHOR_PLACEHOLDERS:
        return "placeholder"
    return None


def _calendar_year(stamp: str | None) -> int | None:
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(stamp).year
    except ValueError:
        return None


def year_reason(
    year: int, *, current_year: int, processed_at: str | None = None
) -> DropReason | None:
    """Why a year is junk, or None when it is kept.

    ``processed_at`` is the OCR registry stamp of an OCR output; a year equal
    to its calendar year is the processing date, not the book's.
    """
    if year == FILETIME_EPOCH_YEAR:
        return "filetime_epoch"
    if year < EARLIEST_YEAR or year > current_year:
        return "implausible_year"
    if year == _calendar_year(processed_at):
        return "processing_year"
    return None


def _clean(
    source: Source, *, current_year: int, processed_at: str | None
) -> tuple[Source, MetadataAudit]:
    title = author = year = None
    if source.title is not None:
        reason = title_reason(source.title, source.path)
        title = None if reason is None else FieldAudit(reason, source.title)
    if source.author is not None:
        reason = author_reason(source.author)
        author = None if reason is None else FieldAudit(reason, source.author)
    if source.year is not None:
        reason = year_reason(
            source.year, current_year=current_year, processed_at=processed_at
        )
        year = None if reason is None else FieldAudit(reason, source.year)
    audit = MetadataAudit(title, author, year)
    cleaned = replace(
        source,
        title=None if title else source.title,
        author=None if author else source.author,
        year=None if year else source.year,
    )
    return cleaned, audit


def _inherit[T](
    value: T | None, audit: FieldAudit | None, inherited: T | None, original: str
) -> tuple[T | None, FieldAudit | None]:
    """The shown value and the audit of one field of an OCR output."""
    if value is not None or inherited is None:
        return value, audit
    if audit is None:
        return inherited, FieldAudit("inherited", inherited_from=original)
    return inherited, replace(audit, inherited_from=original)


def sanitize_source(
    source: Source, *, current_year: int, origin: OcrOrigin | None = None
) -> SanitizedSource:
    """Drop the implausible title, author and year of a source.

    ``current_year`` comes from the caller's clock. ``origin`` is given for
    a source that is the live output of an OCR registry entry: a missing or
    dropped field is then taken from the original's cleaned value.
    """
    processed_at = None if origin is None else origin.entry.created_at
    cleaned, audit = _clean(
        source, current_year=current_year, processed_at=processed_at
    )
    if origin is not None and origin.original is not None:
        original, _ = _clean(
            origin.original, current_year=current_year, processed_at=None
        )
        path = origin.entry.input_path
        title, title_audit = _inherit(cleaned.title, audit.title, original.title, path)
        author, author_audit = _inherit(
            cleaned.author, audit.author, original.author, path
        )
        year, year_audit = _inherit(cleaned.year, audit.year, original.year, path)
        cleaned = replace(cleaned, title=title, author=author, year=year)
        audit = MetadataAudit(title_audit, author_audit, year_audit)
    return SanitizedSource(cleaned, audit)


def ocr_origin(
    source: Source,
    outputs: Mapping[str, OcrOutput],
    originals: Mapping[str, Source],
) -> OcrOrigin | None:
    """The origin of a source that is a live OCR output, else None.

    ``outputs`` maps live output paths to their registry entries and
    ``originals`` maps manifest paths to their sources.
    """
    # OCR always writes a PDF, so no other type can be an OCR output.
    if source.file_type is not FileType.PDF:
        return None
    entry = outputs.get(source.path)
    if entry is None:
        return None
    return OcrOrigin(entry, originals.get(entry.input_path))
