"""Browsing indexed sources through in-memory ports (UC-08)."""

import logging

import pytest

from docgrep.domain.errors import IndexIncomplete, StoreUnavailable
from docgrep.domain.models import FileType, OcrRecord, Passage, Source
from docgrep.domain.sources import BrowseRequest, BrowseResult, SourceFilter
from docgrep.usecases.browse_sources import BrowseSources
from fakes import InMemoryManifest, InMemoryOcrRegistry, InMemoryPassageStore

_TYPES = {
    "pdf": FileType.PDF,
    "txt": FileType.TXT,
    "md": FileType.MARKDOWN,
    "djvu": FileType.DJVU,
}


def _source(path: str, **metadata: object) -> Source:
    return Source(
        source_id=path,
        path=path,
        file_type=_TYPES[path.rsplit(".", 1)[-1]],
        content_hash=f"hash-{path}",
        indexed_at="2026-10-01T00:00:00+00:00",
        **metadata,  # type: ignore[arg-type]  # title, author, year by name
    )


class CountingStore:
    """A passage counter that records every call and can fail for some ids."""

    def __init__(
        self,
        store: InMemoryPassageStore,
        *,
        failing: frozenset[str] = frozenset(),
        down: bool = False,
    ) -> None:
        self.store = store
        self.failing = failing
        self.down = down
        self.calls: list[str] = []

    def count_source_passages(self, source_id: str) -> int:
        self.calls.append(source_id)
        if self.down or source_id in self.failing:
            raise StoreUnavailable("Qdrant operation failed")
        return self.store.count_source_passages(source_id)


class Library:
    def __init__(self) -> None:
        self.manifest = InMemoryManifest()
        self.store = InMemoryPassageStore()
        self.ocr = InMemoryOcrRegistry()

    def add(self, source: Source, passages: int, status: str = "indexed") -> None:
        self.manifest.record_source(source, status, passages=passages)
        for offset in range(passages):
            self.store.upsert(
                Passage(
                    f"{source.source_id}#{offset}", source.source_id, "t", offset, 1
                ),
                source,
                (1.0,),
                {},
            )

    def publish(self) -> None:
        self.store.write_completion_metadata({"embedding_model": "fake"})

    def browse(self, counter: CountingStore | None = None) -> BrowseSources:
        return BrowseSources(
            manifest=self.manifest,
            store=counter or CountingStore(self.store),
            ocr=self.ocr,
        )


def _library() -> Library:
    library = Library()
    library.add(_source("b/arnold.pdf", author="V. I. Arnold", year=1989), 4)
    library.add(_source("a/notes.md", title="Notes"), 1)
    library.add(_source("c/landau.djvu", author="Лев Ландау", year=1958), 2)
    library.add(_source("d/plain.txt", year=2001), 3)
    library.add(_source("e/skipped.pdf"), 0, status="skipped:requires_ocr")
    library.add(_source("f/failed.txt"), 0, status="failed:parse_error")
    library.add(_source("g/delegated.djvu"), 0, status="skipped:delegated_to_ocr")
    library.publish()
    return library


def _paths(result: BrowseResult) -> list[str]:
    return [item.source.path for item in result.sources]


def test_lists_only_indexed_sources_with_counts_and_metadata() -> None:
    result = _library().browse().run(BrowseRequest())

    assert _paths(result) == [
        "a/notes.md",
        "b/arnold.pdf",
        "c/landau.djvu",
        "d/plain.txt",
    ]
    assert [item.passage_count for item in result.sources] == [1, 4, 2, 3]
    assert result.total_hits == 4
    assert result.by_file_type == (("djvu", 1), ("markdown", 1), ("pdf", 1), ("txt", 1))
    assert result.applied_filters == {}
    first = result.sources[1]
    assert (first.source.author, first.source.year) == ("V. I. Arnold", 1989)
    assert first.source.content_hash == "hash-b/arnold.pdf"
    assert first.ocr_status == "not_applicable" and not first.degraded


def test_counts_come_from_the_store_not_the_manifest_and_drift_shows_as_zero() -> None:
    library = _library()
    # The manifest says 4 passages, but the store lost them.
    library.store.delete_live_source_points("b/arnold.pdf")

    result = library.browse().run(BrowseRequest(SourceFilter(authors=("arnold",))))

    assert _paths(result) == ["b/arnold.pdf"]
    assert result.sources[0].passage_count == 0
    assert not result.sources[0].degraded


def test_only_the_page_is_counted() -> None:
    library = _library()
    counter = CountingStore(library.store)

    result = library.browse(counter).run(BrowseRequest(page=2, page_size=2))

    assert _paths(result) == ["c/landau.djvu", "d/plain.txt"]
    assert counter.calls == ["c/landau.djvu", "d/plain.txt"]


def test_sorting_by_passage_count_counts_every_match_once() -> None:
    library = _library()
    counter = CountingStore(library.store)

    result = library.browse(counter).run(
        BrowseRequest(page_size=2, sort_by="passage_count", sort_order="desc")
    )

    assert _paths(result) == ["b/arnold.pdf", "d/plain.txt"]
    assert sorted(counter.calls) == [
        "a/notes.md",
        "b/arnold.pdf",
        "c/landau.djvu",
        "d/plain.txt",
    ]


def test_aggregates_cover_every_match_not_only_the_page() -> None:
    request = BrowseRequest(SourceFilter(year_from=1900), page_size=1, sort_by="year")

    result = _library().browse().run(request)

    assert _paths(result) == ["c/landau.djvu"]
    assert result.total_hits == 3
    assert result.by_file_type == (("djvu", 1), ("pdf", 1), ("txt", 1))
    assert result.applied_filters == {"year_from": 1900}


def test_has_metadata_and_filters_apply_together() -> None:
    library = _library()

    without = library.browse().run(BrowseRequest(has_metadata="without_any"))
    both = library.browse().run(
        BrowseRequest(
            SourceFilter(file_types=("pdf", "djvu")), has_metadata="with_both"
        )
    )

    assert _paths(without) == ["a/notes.md"]
    assert without.applied_filters == {"has_metadata": "without_any"}
    assert _paths(both) == ["b/arnold.pdf", "c/landau.djvu"]


def test_one_failed_count_degrades_that_entry_and_logs_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    library = _library()
    counter = CountingStore(library.store, failing=frozenset({"b/arnold.pdf"}))

    with caplog.at_level(logging.WARNING, logger="docgrep.usecases.browse_sources"):
        result = library.browse(counter).run(BrowseRequest())

    degraded = {item.source.path: item for item in result.sources if item.degraded}
    assert list(degraded) == ["b/arnold.pdf"]
    assert degraded["b/arnold.pdf"].passage_count is None
    assert [item.passage_count for item in result.sources if not item.degraded] == [
        1,
        2,
        3,
    ]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "b/arnold.pdf" in warnings[0].getMessage()


def test_a_store_that_fails_for_every_source_on_the_page_is_unavailable() -> None:
    library = _library()
    counter = CountingStore(library.store, down=True)

    with pytest.raises(StoreUnavailable):
        library.browse(counter).run(BrowseRequest())
    with pytest.raises(StoreUnavailable):
        library.browse(counter).run(BrowseRequest(sort_by="passage_count"))


def test_an_unpublished_index_or_an_empty_manifest_gives_an_empty_success() -> None:
    unpublished = Library()
    unpublished.add(_source("a.txt"), 2)
    empty = Library()
    empty.publish()

    for library in (unpublished, empty):
        for request in (BrowseRequest(), BrowseRequest(page=5)):
            result = library.browse().run(request)
            assert result.sources == () and result.total_hits == 0
            assert result.by_file_type == ()


def test_a_page_beyond_the_end_is_empty_with_the_right_total() -> None:
    library = _library()
    counter = CountingStore(library.store)

    result = library.browse(counter).run(BrowseRequest(page=3, page_size=2))

    assert result.sources == ()
    assert (result.total_hits, result.page, result.page_size) == (4, 3, 2)
    # One probe tells a published index from none.
    assert len(counter.calls) <= 1


def test_ocr_outputs_are_reported_as_processed_via_ocr() -> None:
    library = _library()
    library.add(_source("h/scan.ocr.pdf"), 1)
    library.ocr.register(
        OcrRecord(
            input_path="h/scan.djvu",
            input_format="djvu",
            input_hash="in",
            output_path="h/scan.ocr.pdf",
            output_hash="out",
            engine="ocrmypdf",
            engine_version="16",
            languages="rus+eng",
            mode="skip-text",
        )
    )

    result = library.browse().run(BrowseRequest(SourceFilter(path_prefixes=("h/",))))

    assert [(i.source.path, i.ocr_status) for i in result.sources] == [
        ("h/scan.ocr.pdf", "processed_via_ocr")
    ]


def test_an_ocr_output_whose_file_is_gone_is_not_applicable() -> None:
    library = _library()
    library.ocr = InMemoryOcrRegistry(exists=lambda path: False)

    result = library.browse().run(BrowseRequest())

    assert {item.ocr_status for item in result.sources} == {"not_applicable"}


def test_a_default_page_of_a_large_manifest_makes_one_store_call_per_entry() -> None:
    library = Library()
    for number in range(2000):
        library.add(_source(f"lib/{number:04d}.txt", year=1900 + number % 100), 1)
    library.publish()
    counter = CountingStore(library.store)

    result = library.browse(counter).run(BrowseRequest())

    assert len(result.sources) == 50 and result.total_hits == 2000
    assert len(counter.calls) == 50

    counter.calls.clear()
    library.browse(counter).run(BrowseRequest(sort_by="passage_count"))
    # The expensive case: every matching source is counted once.
    assert len(counter.calls) == 2000


def test_index_incomplete_from_the_counter_never_escapes() -> None:
    library = Library()
    library.add(_source("a.txt"), 1)
    counter = CountingStore(library.store)

    result = library.browse(counter).run(BrowseRequest(sort_by="passage_count"))

    assert result.total_hits == 0
    with pytest.raises(IndexIncomplete):
        library.store.count_source_passages("a.txt")
