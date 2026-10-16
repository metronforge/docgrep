"""Search restricted by the shared source filter, through in-memory ports
(UC-02)."""

from collections.abc import Collection, Mapping, Sequence

import pytest

from docgrep.adapters.models.deterministic import DeterministicEmbedder, LexicalReranker
from docgrep.domain.errors import IndexIncomplete, ValidationError
from docgrep.domain.models import (
    FileType,
    ManifestSourceRecord,
    Passage,
    RetrievalCandidate,
    Source,
    make_passage_id,
)
from docgrep.domain.sources import SourceFilter
from docgrep.usecases.find_passages import NO_MATCHING_SOURCES_HINT, FindPassages
from fakes import InMemoryManifest, InMemoryPassageStore

QUERY = "comet orbit"
_TYPES = {"md": FileType.MARKDOWN, "txt": FileType.TXT, "pdf": FileType.PDF}


def _source(path: str, author: str | None = None, year: int | None = None) -> Source:
    return Source(path, path, _TYPES[path.rsplit(".", 1)[-1]], author=author, year=year)


class RecordingEmbedder:
    def __init__(self) -> None:
        self.inner = DeterministicEmbedder()
        self.calls = 0

    @property
    def model_name(self) -> str:
        return self.inner.model_name

    @property
    def dimension(self) -> int:
        return self.inner.dimension

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        self.calls += 1
        return self.inner.embed(texts)


class NoSparse:
    def encode(self, text: str) -> Mapping[int, float]:
        return {}


class RecordingRetriever:
    """The in-memory store, recording the allow-list of every retrieval."""

    def __init__(self, store: InMemoryPassageStore) -> None:
        self.store = store
        self.calls: list[frozenset[str] | None] = []

    def retrieve(
        self,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float],
        top_k: int,
        source_ids: Collection[str] | None = None,
    ) -> tuple[RetrievalCandidate, ...]:
        self.calls.append(None if source_ids is None else frozenset(source_ids))
        return self.store.retrieve(dense_vector, sparse_vector, top_k, source_ids)

    def read_completion_metadata(self) -> Mapping[str, str | int | float] | None:
        return self.store.read_completion_metadata()

    def count_passages(self) -> int:
        return self.store.count_passages()


class FailingManifest:
    def committed_sources(self) -> tuple[ManifestSourceRecord, ...]:
        raise OSError("disk I/O error")


class Library:
    def __init__(self) -> None:
        self.manifest = InMemoryManifest()
        self.store = InMemoryPassageStore()
        self.embedder = RecordingEmbedder()
        self.retriever = RecordingRetriever(self.store)

    def add(self, source: Source, *texts: str, status: str = "indexed") -> None:
        self.manifest.record_source(source, status, passages=len(texts))
        vectors = DeterministicEmbedder().embed(texts)
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

    def publish(self) -> None:
        self.store.write_completion_metadata(
            {"embedding_model": "deterministic-hash-v1", "embedding_dimension": 64}
        )

    def search(self, top_k: int = 3, **kwargs: object) -> FindPassages:
        return FindPassages(
            embedder=self.embedder,
            sparse_encoder=NoSparse(),
            retriever=self.retriever,
            reranker=LexicalReranker(),
            top_k=top_k,
            default_top_n=min(3, top_k),
            top_n_max=10,
            max_query_len=200,
            manifest=self.manifest,
            **kwargs,  # type: ignore[arg-type]  # collapse_duplicates only
        )


@pytest.fixture
def library() -> Library:
    """Strong decoys, a weak target by Arnold, and sources without metadata."""
    lib = Library()
    for number in range(6):
        lib.add(
            _source(f"decoys/d{number}.md", "Decoy Writer", 2001 + number),
            f"Comet orbit, comet orbit {number}.",
        )
    lib.add(
        _source("books/arnold.md", "V. I. Arnold", 1974),
        "Mathematical methods of classical mechanics mention a comet once.",
        "Another chapter on rigid bodies and the comet of the old tables.",
    )
    lib.add(
        _source("books/late.pdf", "Vladimir ARNOLD", 1990),
        "Late lectures on ordinary differential equations and a comet.",
    )
    lib.add(_source("notes/plain.md"), "A plain comet orbit note.")
    lib.add(_source("papers/kepler.txt", year=1609), "Kepler on the orbit of Mars.")
    lib.add(_source("papers/anon.txt", author="Anon"), "Comet orbit by nobody known.")
    lib.add(_source("broken.md", "V. I. Arnold", 1974), status="failed")
    lib.publish()
    return lib


def _paths(result: object) -> list[str]:
    return [item.source.path for item in result.results]  # type: ignore[attr-defined]  # SearchResult


def test_the_filtered_candidates_come_from_the_allowed_sources(
    library: Library,
) -> None:
    unfiltered = library.search().run(QUERY)
    assert "books/arnold.md" not in _paths(unfiltered)

    result = library.search().run(
        QUERY, source_filter=SourceFilter(authors=("arnold",))
    )

    assert set(_paths(result)) == {"books/arnold.md", "books/late.pdf"}
    assert library.retriever.calls[-1] == {"books/arnold.md", "books/late.pdf"}
    assert result.applied_filters == {"authors": ["arnold"]}
    # Without an author: notes/plain.md and papers/kepler.txt.
    assert result.sources_excluded_for_missing_metadata == 2


def test_each_field_alone_and_combined(library: Library) -> None:
    search = library.search(top_k=10)

    def paths(**fields: object) -> set[str]:
        result = search.run(QUERY, top_n=10, source_filter=SourceFilter(**fields))  # type: ignore[arg-type]  # filter fields by name
        return set(_paths(result))

    assert paths(source_ids=("papers/kepler.txt",)) == {"papers/kepler.txt"}
    assert paths(year_from=1980, year_to=1990) == {"books/late.pdf"}
    assert paths(year_to=1700) == {"papers/kepler.txt"}
    assert paths(file_types=("TXT",)) == {"papers/kepler.txt", "papers/anon.txt"}
    assert paths(path_prefixes=("/books/",)) == {"books/arnold.md", "books/late.pdf"}
    assert paths(path_globs=("**/*.txt", "notes/*")) == {
        "papers/kepler.txt",
        "papers/anon.txt",
        "notes/plain.md",
    }
    # OR within a list, AND across fields.
    assert paths(authors=("arnold", "anon"), file_types=("pdf", "txt")) == {
        "books/late.pdf",
        "papers/anon.txt",
    }
    assert paths(authors=("arnold",), year_from=1980) == {"books/late.pdf"}


def test_every_returned_passage_satisfies_the_filter(library: Library) -> None:
    source_filter = SourceFilter(year_from=1990, file_types=("markdown", "pdf"))

    result = library.search(top_k=10).run(QUERY, top_n=10, source_filter=source_filter)

    assert result.results
    for item in result.results:
        assert item.source.year is not None and item.source.year >= 1990
        assert item.source.file_type in (FileType.MARKDOWN, FileType.PDF)


def test_null_metadata_is_counted_only_when_it_alone_excludes(
    library: Library,
) -> None:
    search = library.search(top_k=10)

    years = search.run(QUERY, source_filter=SourceFilter(year_from=1980))
    # No year: notes/plain.md and papers/anon.txt; kepler has an old year.
    assert years.sources_excluded_for_missing_metadata == 2

    both = search.run(
        QUERY, source_filter=SourceFilter(authors=("arnold",), year_to=1980)
    )
    # notes/plain.md lacks both; anon.txt has the wrong author and no year;
    # kepler.txt has no author and a year in range.
    assert both.sources_excluded_for_missing_metadata == 2

    narrowed = search.run(
        QUERY, source_filter=SourceFilter(authors=("arnold",), file_types=("txt",))
    )
    assert narrowed.results == ()
    # Only kepler.txt: the other txt has an author that does not match.
    assert narrowed.sources_excluded_for_missing_metadata == 1

    other = search.run(QUERY, source_filter=SourceFilter(file_types=("txt",)))
    assert other.sources_excluded_for_missing_metadata is None


def test_an_empty_match_set_touches_no_model_and_no_retrieval(
    library: Library,
) -> None:
    result = library.search().run(
        QUERY, source_filter=SourceFilter(authors=("Tolstoy",))
    )

    assert result.results == () and result.total_hits == 0
    assert result.hint == NO_MATCHING_SOURCES_HINT
    assert result.applied_filters == {"authors": ["Tolstoy"]}
    assert library.embedder.calls == 0
    assert library.retriever.calls == []


def test_a_filter_matching_every_indexed_source_sends_no_allow_list(
    library: Library,
) -> None:
    # The failed source is not indexed, so it does not count.
    every = SourceFilter(path_globs=("**",))

    result = library.search().run(QUERY, source_filter=every)

    assert library.retriever.calls == [None]
    assert result.applied_filters == {"path_globs": ["**"]}


def test_no_filter_and_an_empty_filter_are_the_old_search(library: Library) -> None:
    class NoManifest:
        def committed_sources(self) -> tuple[ManifestSourceRecord, ...]:
            raise AssertionError("an unfiltered search reads no manifest")

    search = library.search()
    search.manifest = NoManifest()

    plain = search.run(QUERY)
    empty = search.run(QUERY, source_filter=SourceFilter())

    assert library.retriever.calls == [None, None]
    for result in (plain, empty):
        assert result.applied_filters == {}
        assert result.sources_excluded_for_missing_metadata is None
    assert [item.passage.passage_id for item in plain.results] == [
        item.passage.passage_id for item in empty.results
    ]


def test_fewer_matching_passages_than_top_k_are_all_returned_ranked(
    library: Library,
) -> None:
    result = library.search(top_k=10).run(
        QUERY, top_n=10, source_filter=SourceFilter(path_prefixes=("books/",))
    )

    assert sorted(_paths(result)) == [
        "books/arnold.md",
        "books/arnold.md",
        "books/late.pdf",
    ]
    assert result.total_hits == 3
    scores = [item.score for item in result.results]
    assert scores == sorted(scores, reverse=True)


def test_invalid_filters_fail_before_any_model_call(library: Library) -> None:
    with pytest.raises(ValidationError) as bounds:
        library.search().run(
            QUERY, source_filter=SourceFilter(year_from=2000, year_to=1990)
        )
    assert bounds.value.category == "validation_error"
    with pytest.raises(ValidationError):
        SourceFilter(file_types=("odt",))
    with pytest.raises(ValidationError):
        SourceFilter(authors=(" ",))
    assert library.embedder.calls == 0 and library.retriever.calls == []


def test_a_filtered_search_needs_a_readable_manifest(library: Library) -> None:
    source_filter = SourceFilter(authors=("arnold",))
    search = library.search()

    search.manifest = FailingManifest()
    with pytest.raises(IndexIncomplete):
        search.run(QUERY, source_filter=source_filter)
    search.manifest = None
    with pytest.raises(IndexIncomplete):
        search.run(QUERY, source_filter=source_filter)
    # An unfiltered search does not need it.
    assert search.run(QUERY).results
    assert library.embedder.calls == 1


def test_an_unbuilt_index_fails_the_same_way_with_or_without_filters() -> None:
    lib = Library()

    with pytest.raises(IndexIncomplete):
        lib.search().run(QUERY)
    with pytest.raises(IndexIncomplete):
        lib.search().run(QUERY, source_filter=SourceFilter(year_from=1900))
