"""Browsing shows and matches sanitized metadata (ADR-0021, issue #22)."""

from datetime import UTC, datetime

from docgrep.domain.metadata import FieldAudit, MetadataAudit
from docgrep.domain.models import FileType, OcrRecord, Passage, Source
from docgrep.domain.sources import BrowseRequest, BrowseResult, SourceFilter
from docgrep.usecases.browse_sources import BrowseSources
from fakes import InMemoryManifest, InMemoryOcrRegistry, InMemoryPassageStore

_TYPES = {"pdf": FileType.PDF, "epub": FileType.EPUB, "djvu": FileType.DJVU}


def _source(path: str, **metadata: object) -> Source:
    return Source(
        source_id=path,
        path=path,
        file_type=_TYPES[path.rsplit(".", 1)[-1]],
        **metadata,  # type: ignore[arg-type]  # title, author, year by name
    )


def _record(input_path: str, output_path: str, created_at: str) -> OcrRecord:
    return OcrRecord(
        input_path=input_path,
        input_format="djvu",
        input_hash=f"hash-{input_path}",
        output_path=output_path,
        output_hash=f"hash-{output_path}",
        engine="ocrmypdf",
        engine_version="16.0.0",
        languages="rus+eng",
        mode="skip-text",
        created_at=created_at,
    )


class Library:
    def __init__(self) -> None:
        self.manifest = InMemoryManifest()
        self.store = InMemoryPassageStore()
        self.ocr = InMemoryOcrRegistry()
        self.store.write_completion_metadata({"embedding_model": "fake"})

    def add(self, source: Source, status: str = "indexed") -> None:
        self.manifest.record_source(source, status, passages=1)
        if status == "indexed":
            self.store.upsert(
                Passage(f"{source.source_id}#0", source.source_id, "t", 0, 1),
                source,
                (1.0,),
                {},
            )

    def browse(self, request: BrowseRequest | None = None) -> BrowseResult:
        return BrowseSources(
            manifest=self.manifest,
            store=self.store,
            ocr=self.ocr,
            now=lambda: datetime(2026, 10, 17, tzinfo=UTC),
        ).run(request or BrowseRequest())


def _library() -> Library:
    library = Library()
    library.add(
        _source("a/clean.epub", title="A Clean Book", author="A. Writer", year=1989)
    )
    library.add(_source("b/epoch.epub", title="Modern", author="Unknown", year=101))
    library.add(_source("c/filetime.pdf", title="Report", year=1601))
    library.add(_source("d/unix.epub", title="Old Tale", author="Teller", year=1970))
    # An OCR output stamped with its processing year, and its delegated
    # original whose row carries the book's metadata.
    library.add(
        _source("e/book.djvu", title="Old Book", author="B. Author", year=1958),
        status="skipped:delegated_to_ocr",
    )
    library.add(_source("e/book.ocr.pdf", title="scan", year=2025))
    library.ocr.register(
        _record("e/book.djvu", "e/book.ocr.pdf", "2025-05-06T07:08:09+00:00")
    )
    return library


def _by_path(result: BrowseResult) -> dict[str, tuple[object, ...]]:
    return {
        item.source.path: (item.source.title, item.source.author, item.source.year)
        for item in result.sources
    }


def test_the_listing_shows_sanitized_values_and_inherited_ones() -> None:
    result = _library().browse()

    assert _by_path(result) == {
        "a/clean.epub": ("A Clean Book", "A. Writer", 1989),
        "b/epoch.epub": ("Modern", None, None),
        "c/filetime.pdf": ("Report", None, None),
        "d/unix.epub": ("Old Tale", "Teller", 1970),
        "e/book.ocr.pdf": ("Old Book", "B. Author", 1958),
    }


def test_each_entry_records_what_was_dropped_or_inherited() -> None:
    result = _library().browse()
    audits = {item.source.path: item.metadata_dropped for item in result.sources}

    assert audits["a/clean.epub"] is None
    assert audits["d/unix.epub"] is None
    assert audits["b/epoch.epub"] == MetadataAudit(
        author=FieldAudit("placeholder", "Unknown"),
        year=FieldAudit("implausible_year", 101),
    )
    assert audits["c/filetime.pdf"] == MetadataAudit(
        year=FieldAudit("filetime_epoch", 1601)
    )
    assert audits["e/book.ocr.pdf"] == MetadataAudit(
        title=FieldAudit("placeholder", "scan", "e/book.djvu"),
        author=FieldAudit("inherited", inherited_from="e/book.djvu"),
        year=FieldAudit("processing_year", 2025, "e/book.djvu"),
    )
    by_path = {item.source.path: item for item in result.sources}
    assert by_path["e/book.ocr.pdf"].ocr_status == "processed_via_ocr"


def test_the_registry_is_read_once_per_request() -> None:
    library = _library()

    library.browse()
    assert library.ocr.live_output_reads == 1
    library.browse(BrowseRequest(SourceFilter(year_to=1990), sort_by="year"))
    assert library.ocr.live_output_reads == 2


def test_a_year_filter_no_longer_matches_junk_years() -> None:
    result = _library().browse(BrowseRequest(SourceFilter(year_to=1990)))

    assert sorted(_by_path(result)) == [
        "a/clean.epub",
        "d/unix.epub",
        "e/book.ocr.pdf",
    ]


def test_an_author_filter_sees_the_inherited_author() -> None:
    result = _library().browse(BrowseRequest(SourceFilter(authors=("b. author",))))

    assert list(_by_path(result)) == ["e/book.ocr.pdf"]


def test_without_any_counts_a_source_whose_author_and_year_were_junk() -> None:
    result = _library().browse(BrowseRequest(has_metadata="without_any"))

    assert sorted(_by_path(result)) == ["b/epoch.epub", "c/filetime.pdf"]


def test_sorting_by_year_uses_the_sanitized_years() -> None:
    result = _library().browse(BrowseRequest(sort_by="year"))

    assert [item.source.year for item in result.sources] == [
        1958,
        1970,
        1989,
        None,
        None,
    ]


def test_the_metadata_dropped_filter_selects_affected_sources() -> None:
    library = _library()

    dropped = library.browse(BrowseRequest(metadata_dropped=True))
    kept = library.browse(BrowseRequest(metadata_dropped=False))

    assert sorted(_by_path(dropped)) == [
        "b/epoch.epub",
        "c/filetime.pdf",
        "e/book.ocr.pdf",
    ]
    assert dropped.applied_filters == {"metadata_dropped": True}
    assert dropped.by_file_type == (("epub", 1), ("pdf", 2))
    assert sorted(_by_path(kept)) == ["a/clean.epub", "d/unix.epub"]
    assert kept.applied_filters == {"metadata_dropped": False}


def test_the_metadata_dropped_filter_combines_with_the_others() -> None:
    result = _library().browse(
        BrowseRequest(SourceFilter(file_types=("epub",)), metadata_dropped=True)
    )

    assert list(_by_path(result)) == ["b/epoch.epub"]


def test_an_ocr_output_without_a_registry_entry_is_not_changed() -> None:
    library = Library()
    library.add(_source("e/book.djvu", title="Old Book"), status="skipped:x")
    library.add(_source("e/book.ocr.pdf", year=2025))

    result = library.browse()

    assert _by_path(result) == {"e/book.ocr.pdf": (None, None, 2025)}
    assert result.sources[0].metadata_dropped is None


def test_an_ocr_output_whose_original_is_also_junk_stays_null() -> None:
    library = Library()
    library.add(
        _source("e/book.djvu", title="Untitled", author="admin", year=1601),
        status="skipped:delegated_to_ocr",
    )
    library.add(_source("e/book.ocr.pdf"))
    library.ocr.register(_record("e/book.djvu", "e/book.ocr.pdf", "2025-05-06"))

    result = library.browse()

    assert _by_path(result) == {"e/book.ocr.pdf": (None, None, None)}
    assert result.sources[0].metadata_dropped is None


def test_the_stored_values_are_not_changed() -> None:
    library = _library()

    library.browse()

    stored = {r.source.path: r.source for r in library.manifest.committed_sources()}
    assert (stored["b/epoch.epub"].author, stored["b/epoch.epub"].year) == (
        "Unknown",
        101,
    )
    assert stored["e/book.ocr.pdf"].title == "scan"


def test_the_current_year_comes_from_the_clock() -> None:
    library = Library()
    library.add(_source("a/new.epub", year=2026))

    def browse(year: int) -> BrowseResult:
        return BrowseSources(
            manifest=library.manifest,
            store=library.store,
            ocr=library.ocr,
            now=lambda: datetime(year, 1, 1, tzinfo=UTC),
        ).run(BrowseRequest())

    assert browse(2026).sources[0].source.year == 2026
    assert browse(2025).sources[0].source.year is None
