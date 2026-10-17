"""Sanitizing source metadata on read (ADR-0021, issue #22)."""

import pytest

from docgrep.domain.metadata import (
    AUTHOR_PLACEHOLDERS,
    TITLE_PLACEHOLDERS,
    DropReason,
    FieldAudit,
    MetadataAudit,
    OcrOrigin,
    ocr_origin,
    sanitize_source,
)
from docgrep.domain.models import FileType, OcrOutput, Source

CURRENT_YEAR = 2026


def _source(path: str = "books/a-book.epub", **metadata: object) -> Source:
    file_type = FileType.PDF if path.endswith(".pdf") else FileType.EPUB
    return Source(
        source_id=path,
        path=path,
        file_type=file_type,
        **metadata,  # type: ignore[arg-type]  # title, author, year by name
    )


def _clean(source: Source, origin: OcrOrigin | None = None) -> Source:
    return sanitize_source(source, current_year=CURRENT_YEAR, origin=origin).source


def _audit(source: Source, origin: OcrOrigin | None = None) -> MetadataAudit:
    return sanitize_source(source, current_year=CURRENT_YEAR, origin=origin).audit


@pytest.mark.parametrize(
    ("year", "reason"),
    [
        (101, "implausible_year"),
        (0, "implausible_year"),
        (1449, "implausible_year"),
        (CURRENT_YEAR + 1, "implausible_year"),
        (9999, "implausible_year"),
        (1601, "filetime_epoch"),
    ],
)
def test_an_implausible_year_is_dropped(year: int, reason: DropReason) -> None:
    audit = _audit(_source(year=year))

    assert _clean(_source(year=year)).year is None
    assert audit.year == FieldAudit(reason, year)


@pytest.mark.parametrize("year", [1450, 1900, 1970, 2001, CURRENT_YEAR])
def test_a_plausible_year_is_kept(year: int) -> None:
    assert _clean(_source(year=year)).year == year
    assert _audit(_source(year=year)).empty


def test_the_current_year_comes_from_the_caller() -> None:
    source = _source(year=2027)

    assert sanitize_source(source, current_year=2026).source.year is None
    assert sanitize_source(source, current_year=2027).source.year == 2027


def test_a_non_ocr_source_keeps_a_year_equal_to_its_indexing_year() -> None:
    source = _source(year=2026, indexed_at="2026-10-01T00:00:00+00:00")

    assert _clean(source).year == 2026


@pytest.mark.parametrize("author", sorted(AUTHOR_PLACEHOLDERS))
def test_every_author_placeholder_is_dropped(author: str) -> None:
    audit = _audit(_source(author=author))

    assert _clean(_source(author=author)).author is None
    assert audit.author == FieldAudit("placeholder", author)


@pytest.mark.parametrize(
    "author", ["UNKNOWN", "  Unknown  ", "Microsoft   Office User", "Python-Docx"]
)
def test_author_placeholders_match_after_normalising(author: str) -> None:
    assert _clean(_source(author=author)).author is None


@pytest.mark.parametrize(
    ("author", "reason"),
    [
        ("", "empty"),
        ("   ", "empty"),
        ("712313159", "numeric_only"),
        ("12-34", "numeric_only"),
    ],
)
def test_an_empty_or_numeric_author_is_dropped(author: str, reason: DropReason) -> None:
    assert _audit(_source(author=author)).author == FieldAudit(reason, author)


@pytest.mark.parametrize(
    "author",
    [
        "Unknown Pleasures",
        "User Experience Group",
        "Ivanov, I.; Petrov, P.",
        "ДОСТОЕВСКИЙ Ф. М.",
        "Anonymous Coward Collective",
        "R2D2",
    ],
)
def test_an_author_that_only_looks_odd_is_kept(author: str) -> None:
    assert _clean(_source(author=author)).author == author


@pytest.mark.parametrize("title", sorted(TITLE_PLACEHOLDERS))
def test_every_title_placeholder_is_dropped(title: str) -> None:
    audit = _audit(_source(title=title))

    assert _clean(_source(title=title)).title is None
    assert audit.title == FieldAudit("placeholder", title)


@pytest.mark.parametrize(
    ("title", "reason"),
    [
        ("", "empty"),
        (" \t ", "empty"),
        ("712313159", "numeric_only"),
        ("978-5-17 123.4", "numeric_only"),
        ("Book Title", "placeholder"),
        ("  UNTITLED ", "placeholder"),
        ("Microsoft Word - thesis.doc", "file_name"),
        ("Microsoft Excel - budget", "file_name"),
        ("chapter-3.docx", "file_name"),
        ("layout.INDD", "file_name"),
        ("draft.qxd", "file_name"),
        ("flyer.pmd", "file_name"),
        ("paper.tex", "file_name"),
        ("letter.rtf", "file_name"),
        ("cover.cdr", "file_name"),
        ("scan0001.pdf", "file_name"),
        ("A-Book", "file_name"),
        ("a-book.epub", "file_name"),
    ],
)
def test_a_junk_title_is_dropped(title: str, reason: DropReason) -> None:
    source = _source("books/a-book.epub", title=title)

    assert _clean(source).title is None
    assert _audit(source).title == FieldAudit(reason, title)


@pytest.mark.parametrize(
    "title",
    [
        "The Untitled Symphony",
        "Untitled: Notes on Art",
        "1984 and After",
        "Document Analysis Methods",
        "THE COMPLETE WORKS OF SHAKESPEARE",
        "A" * 400,
        "Война и мир",
        "Teaching .pdf workflows",
        "Microsoft Word for Beginners",
        "a-book, second edition",
    ],
)
def test_a_title_that_only_looks_odd_is_kept(title: str) -> None:
    assert _clean(_source("books/a-book.epub", title=title)).title == title


def test_the_real_library_examples_are_dropped() -> None:
    source = _source("books/novel.epub", title="book title", author="Unknown", year=101)
    docx = _source("notes/report.epub", title="712313159", author="python-docx")
    scan = _source("scans/old.ocr.pdf", year=1601)

    assert _audit(source) == MetadataAudit(
        FieldAudit("placeholder", "book title"),
        FieldAudit("placeholder", "Unknown"),
        FieldAudit("implausible_year", 101),
    )
    assert (_clean(docx).title, _clean(docx).author) == (None, None)
    assert _audit(scan).year == FieldAudit("filetime_epoch", 1601)


def test_other_fields_and_clean_values_are_untouched() -> None:
    source = Source(
        "s",
        "books/a.epub",
        FileType.EPUB,
        title="A Book",
        author="An Author",
        year=1999,
        content_hash="h",
        mtime=1.0,
        size=2,
        indexed_at="2026-01-01",
        parser_version="1",
    )

    sanitized = sanitize_source(source, current_year=CURRENT_YEAR)

    assert sanitized.source == source
    assert sanitized.audit.empty and sanitized.audit.fields() == ()


def _origin(original: Source | None, created_at: str | None = None) -> OcrOrigin:
    return OcrOrigin(
        OcrOutput("scans/book.ocr.pdf", "scans/book.djvu", created_at), original
    )


def _original(**metadata: object) -> Source:
    return Source(
        "scans/book.djvu",
        "scans/book.djvu",
        FileType.DJVU,
        **metadata,  # type: ignore[arg-type]  # title, author, year by name
    )


def test_an_ocr_output_drops_its_processing_year() -> None:
    output = _source("scans/book.ocr.pdf", year=2025)

    audit = _audit(output, _origin(None, "2025-03-04T10:00:00+00:00"))

    assert audit.year == FieldAudit("processing_year", 2025)
    assert _clean(output, _origin(None, "2025-03-04T10:00:00+00:00")).year is None


def test_an_ocr_output_keeps_a_year_other_than_its_processing_year() -> None:
    output = _source("scans/book.ocr.pdf", year=1989)

    assert _clean(output, _origin(None, "2025-03-04T10:00:00+00:00")).year == 1989


def test_an_unreadable_processing_stamp_drops_nothing() -> None:
    output = _source("scans/book.ocr.pdf", year=2025)

    assert _clean(output, _origin(None, "not a date")).year == 2025


def test_an_ocr_output_without_metadata_inherits_from_a_clean_original() -> None:
    output = _source("scans/book.ocr.pdf")
    original = _original(title="Old Book", author="A. Writer", year=1958)

    sanitized = sanitize_source(
        output, current_year=CURRENT_YEAR, origin=_origin(original)
    )

    assert (sanitized.source.title, sanitized.source.author, sanitized.source.year) == (
        "Old Book",
        "A. Writer",
        1958,
    )
    inherited = FieldAudit("inherited", inherited_from="scans/book.djvu")
    assert sanitized.audit == MetadataAudit(inherited, inherited, inherited)


def test_an_ocr_output_replaces_its_junk_with_the_originals_values() -> None:
    output = _source("scans/book.ocr.pdf", title="scan", year=2025)
    original = _original(title="Old Book", year=1958)

    sanitized = sanitize_source(
        output,
        current_year=CURRENT_YEAR,
        origin=_origin(original, "2025-01-01T00:00:00+00:00"),
    )

    assert (sanitized.source.title, sanitized.source.year) == ("Old Book", 1958)
    assert sanitized.audit.title == FieldAudit("placeholder", "scan", "scans/book.djvu")
    assert sanitized.audit.year == FieldAudit(
        "processing_year", 2025, "scans/book.djvu"
    )


def test_an_ocr_output_keeps_its_own_plausible_values() -> None:
    output = _source("scans/book.ocr.pdf", title="Own Title", year=1990)
    original = _original(title="Old Book", year=1958)

    sanitized = sanitize_source(
        output, current_year=CURRENT_YEAR, origin=_origin(original)
    )

    assert (sanitized.source.title, sanitized.source.year) == ("Own Title", 1990)
    assert sanitized.audit.empty


def test_an_original_with_junk_gives_nothing_to_inherit() -> None:
    output = _source("scans/book.ocr.pdf", year=1601)
    original = _original(title="Untitled", author="Administrator", year=101)

    sanitized = sanitize_source(
        output, current_year=CURRENT_YEAR, origin=_origin(original)
    )

    assert (sanitized.source.title, sanitized.source.author, sanitized.source.year) == (
        None,
        None,
        None,
    )
    assert sanitized.audit == MetadataAudit(year=FieldAudit("filetime_epoch", 1601))


def test_an_original_title_equal_to_its_own_file_name_is_not_inherited() -> None:
    output = _source("scans/book.ocr.pdf")

    assert _clean(output, _origin(_original(title="book"))).title is None


def test_an_origin_without_a_manifest_row_inherits_nothing() -> None:
    output = _source("scans/book.ocr.pdf", author="Someone")

    sanitized = sanitize_source(output, current_year=CURRENT_YEAR, origin=_origin(None))

    assert sanitized.source == output and sanitized.audit.empty


def test_ocr_origin_finds_only_live_pdf_outputs() -> None:
    entry = OcrOutput("scans/book.ocr.pdf", "scans/book.djvu", None)
    outputs = {entry.output_path: entry}
    original = _original(title="Old Book")
    originals = {original.path: original}

    found = ocr_origin(_source("scans/book.ocr.pdf"), outputs, originals)

    assert found == OcrOrigin(entry, original)
    assert ocr_origin(_source("scans/other.pdf"), outputs, originals) is None
    not_pdf = Source("x", "scans/book.ocr.pdf", FileType.TXT)
    assert ocr_origin(not_pdf, outputs, originals) is None
    assert ocr_origin(_source("scans/book.ocr.pdf"), outputs, {}) == OcrOrigin(entry)
