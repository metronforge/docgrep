"""Search filters and results see sanitized metadata (ADR-0021, issue #22)."""

from collections.abc import Mapping
from datetime import UTC, datetime

from docgrep.adapters.models.deterministic import DeterministicEmbedder, LexicalReranker
from docgrep.domain.models import (
    FileType,
    ManifestSourceRecord,
    OcrRecord,
    Passage,
    SearchResult,
    Source,
    make_passage_id,
)
from docgrep.domain.protocols import CommittedSourceReader
from docgrep.domain.sources import SourceFilter
from docgrep.usecases.find_passages import FindPassages
from fakes import InMemoryManifest, InMemoryOcrRegistry, InMemoryPassageStore

QUERY = "comet orbit"
_TYPES = {"pdf": FileType.PDF, "epub": FileType.EPUB, "djvu": FileType.DJVU}


def _source(path: str, **metadata: object) -> Source:
    return Source(
        path,
        path,
        _TYPES[path.rsplit(".", 1)[-1]],
        **metadata,  # type: ignore[arg-type]  # title, author, year by name
    )


class NoSparse:
    def encode(self, text: str) -> Mapping[int, float]:
        return {}


class CountingManifest:
    """The in-memory manifest, counting reads of the committed sources."""

    def __init__(self, inner: InMemoryManifest) -> None:
        self.inner = inner
        self.reads = 0

    def committed_sources(self) -> tuple[ManifestSourceRecord, ...]:
        self.reads += 1
        return self.inner.committed_sources()


class Library:
    def __init__(self) -> None:
        self.manifest = InMemoryManifest()
        self.store = InMemoryPassageStore()
        self.ocr = InMemoryOcrRegistry()
        self.store.write_completion_metadata(
            {"embedding_model": "deterministic-hash-v1", "embedding_dimension": 64}
        )

    def add(self, source: Source, *texts: str, status: str = "indexed") -> None:
        self.manifest.record_source(source, status, passages=len(texts))
        vectors = DeterministicEmbedder().embed(texts) if texts else ()
        for offset, (text, vector) in enumerate(zip(texts, vectors, strict=True)):
            self.store.upsert(
                Passage(
                    make_passage_id(source.source_id, offset),
                    source.source_id,
                    text,
                    offset,
                    len(text.split()),
                ),
                source,
                vector,
                {},
            )

    def search(
        self, manifest: CommittedSourceReader | None = None, *, ocr: bool = True
    ) -> FindPassages:
        return FindPassages(
            embedder=DeterministicEmbedder(),
            sparse_encoder=NoSparse(),
            retriever=self.store,
            reranker=LexicalReranker(),
            top_k=10,
            default_top_n=10,
            top_n_max=10,
            max_query_len=200,
            manifest=manifest or self.manifest,
            ocr=self.ocr if ocr else None,
            now=lambda: datetime(2026, 10, 17, tzinfo=UTC),
        )


def _library() -> Library:
    library = Library()
    library.add(
        _source("a/old.epub", title="An Old Book", author="Writer", year=1974),
        "A comet orbit in old tables.",
    )
    library.add(
        _source("b/epoch.epub", title="A Modern Book", author="Unknown", year=101),
        "Comet orbit computed again.",
    )
    library.add(
        _source("c/filetime.pdf", title="712313159", year=1601),
        "The orbit of a comet, measured.",
    )
    library.add(
        _source("d/book.djvu", title="Scanned Classic", author="C. Old", year=1958),
        status="skipped:delegated_to_ocr",
    )
    library.add(
        _source("d/book.ocr.pdf", title="book title", year=2025),
        "Comet orbit as printed long ago.",
    )
    library.ocr.register(
        OcrRecord(
            input_path="d/book.djvu",
            input_format="djvu",
            input_hash="h1",
            output_path="d/book.ocr.pdf",
            output_hash="h2",
            engine="ocrmypdf",
            engine_version="16.0.0",
            languages="eng",
            mode="skip-text",
            created_at="2025-02-03T04:05:06+00:00",
        )
    )
    return library


def _metadata(result: SearchResult) -> dict[str, tuple[object, ...]]:
    return {
        item.source.path: (item.source.title, item.source.author, item.source.year)
        for item in result.results
    }


def test_results_show_sanitized_and_inherited_values() -> None:
    result = _library().search().run(QUERY)

    assert _metadata(result) == {
        "a/old.epub": ("An Old Book", "Writer", 1974),
        "b/epoch.epub": ("A Modern Book", None, None),
        "c/filetime.pdf": (None, None, None),
        "d/book.ocr.pdf": ("Scanned Classic", "C. Old", 1958),
    }


def test_an_unfiltered_body_is_otherwise_unchanged() -> None:
    library = _library()

    cleaned = library.search().run(QUERY)
    raw = library.search(ocr=False).run(QUERY)

    assert cleaned.applied_filters == {} and raw.applied_filters == {}
    assert cleaned.sources_excluded_for_missing_metadata is None
    assert cleaned.total_hits == raw.total_hits
    assert [item.passage for item in cleaned.results] == [
        item.passage for item in raw.results
    ]
    assert [item.score for item in cleaned.results] == [
        item.score for item in raw.results
    ]
    assert [item.source.content_hash for item in cleaned.results] == [
        item.source.content_hash for item in raw.results
    ]


def test_a_year_filter_no_longer_matches_junk_years() -> None:
    result = _library().search().run(QUERY, source_filter=SourceFilter(year_to=1990))

    assert sorted(_metadata(result)) == ["a/old.epub", "d/book.ocr.pdf"]
    # The raw years 101 and 1601 were dropped, so those sources are left out
    # for missing metadata.
    assert result.sources_excluded_for_missing_metadata == 2


def test_an_author_filter_counts_a_junk_author_as_missing() -> None:
    result = (
        _library().search().run(QUERY, source_filter=SourceFilter(authors=("unknown",)))
    )

    assert result.results == ()
    assert result.sources_excluded_for_missing_metadata == 2


def test_the_registry_and_the_manifest_are_read_once_per_request() -> None:
    library = _library()
    manifest = CountingManifest(library.manifest)
    search = library.search(manifest)

    search.run(QUERY, source_filter=SourceFilter(year_to=1990))
    assert (library.ocr.live_output_reads, manifest.reads) == (1, 1)

    search.run(QUERY)
    assert (library.ocr.live_output_reads, manifest.reads) == (2, 2)


def test_an_unfiltered_search_without_ocr_outputs_reads_no_manifest() -> None:
    library = Library()
    library.add(_source("a/old.epub", author="Unknown"), "comet orbit")
    manifest = CountingManifest(library.manifest)

    result = library.search(manifest).run(QUERY)

    assert manifest.reads == 0 and library.ocr.live_output_reads == 1
    assert result.results[0].source.author is None


def test_the_stored_payload_keeps_the_raw_values() -> None:
    library = _library()

    library.search().run(QUERY)

    payload = {
        candidate.source.path: candidate.source
        for candidate in library.store.retrieve(
            DeterministicEmbedder().embed((QUERY,))[0], {}, 10
        )
    }
    assert payload["b/epoch.epub"].year == 101
    assert payload["d/book.ocr.pdf"].title == "book title"
