"""The context use case (UC-12): per-id statuses, request validation, the
reads it makes and the sanitized source it shows."""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import pytest
from test_find_passages_sanitized import _library as junk_metadata_library

from docgrep.adapters.chunking import TokenChunker, WhitespaceTokenizer
from docgrep.domain.context import OutlineEntry, select_window, stitch
from docgrep.domain.errors import StoreUnavailable, ValidationError
from docgrep.domain.models import (
    FileType,
    ManifestSourceRecord,
    ParsedDocument,
    ParsedSection,
    Passage,
    Source,
    make_passage_id,
)
from docgrep.usecases.expand_context import (
    NOT_FOUND_HINT,
    SOURCE_MODIFIED_HINT,
    SOURCE_REMOVED_HINT,
    ExpandContext,
)
from fakes import InMemoryManifest, InMemoryOcrRegistry, InMemoryPassageStore

_PAPER = Source(
    "paper.pdf",
    "paper.pdf",
    FileType.PDF,
    title="A Paper",
    author="A. Writer",
    year=1999,
    content_hash="hash-paper",
)
_MEMO = Source("memo.docx", "memo.docx", FileType.DOCX, content_hash="hash-memo")


def _words(count: int, start: int = 0) -> str:
    return " ".join(f"w{number}" for number in range(start, start + count))


def _chunks(source: Source, sections: Sequence[ParsedSection]) -> tuple[Passage, ...]:
    chunker = TokenChunker(WhitespaceTokenizer(), 8, 2)
    return chunker.chunk(ParsedDocument(source, tuple(sections)))


class Library:
    """A published in-memory index of a paged paper and a page-less memo."""

    def __init__(self) -> None:
        self.store = InMemoryPassageStore()
        self.manifest = InMemoryManifest()
        self.ocr = InMemoryOcrRegistry()
        self.paper = self.add(
            _PAPER,
            [
                ParsedSection(_words(20), 1, ("Intro",)),
                ParsedSection(_words(14, 20), 2, ("Intro",)),
            ],
        )
        self.memo = self.add(
            _MEMO,
            [
                ParsedSection(_words(10, 100), None, ("Memo",)),
                ParsedSection(_words(10, 110), None, ("Memo", "Storm")),
            ],
        )
        self.store.write_completion_metadata({"embedding_model": "fake"})

    def add(self, source: Source, sections: Sequence[ParsedSection]) -> list[Passage]:
        passages = _chunks(source, sections)
        for passage in passages:
            self.store.upsert(passage, source, (1.0,), {})
        self.manifest.record_source(source, "indexed")
        return list(passages)

    def expand(self, **overrides: Any) -> ExpandContext:
        options: dict[str, Any] = {
            "store": self.store,
            "manifest": self.manifest,
            "ocr": self.ocr,
            "max_passages": 20,
            "default_window_tokens": 8,
            "max_window_tokens": 50,
            "now": lambda: datetime(2026, 10, 17, tzinfo=UTC),
        }
        options.update(overrides)
        return ExpandContext(**options)


def _expected_context(passages: Sequence[Passage], index: int, window: int) -> Any:
    outline = [
        OutlineEntry(item.passage_id, item.chunk_offset, item.token_count)
        for item in passages
    ]
    chosen = select_window(outline, outline[index], window)
    by_id = {item.passage_id: item for item in passages}
    return stitch(
        passages[index],
        [by_id[entry.passage_id] for entry in chosen.before],
        [by_id[entry.passage_id] for entry in chosen.after],
        at_source_start=chosen.at_source_start,
        at_source_end=chosen.at_source_end,
    )


def test_an_entry_carries_the_passage_its_source_and_the_context() -> None:
    library = Library()
    target = library.paper[2]

    result = library.expand().run([target.passage_id])

    (entry,) = result.entries
    assert result.window_tokens == 8
    assert (entry.passage_id, entry.status, entry.hint) == (
        target.passage_id,
        "ok",
        None,
    )
    assert entry.passage == target
    assert entry.source == _PAPER
    assert entry.context == _expected_context(library.paper, 2, 8)
    assert entry.context is not None and entry.context.before and entry.context.after


def test_a_page_less_source_gives_a_context_without_pages() -> None:
    library = Library()

    (entry,) = (
        library.expand().run([library.memo[1].passage_id], window_tokens=50).entries
    )

    assert entry.context is not None and entry.passage is not None
    locations = (*entry.context.before_locations, *entry.context.after_locations)
    assert locations and all(item.page_number is None for item in locations)
    assert entry.context.at_source_start and entry.context.at_source_end
    # The memo's two sections join with a blank line, without inventing text.
    whole = entry.context.before + entry.passage.text + entry.context.after
    assert whole == _words(10, 100) + "\n\n" + _words(10, 110)


def test_the_window_defaults_to_the_configured_size() -> None:
    library = Library()
    target = library.paper[1]

    default = library.expand().run([target.passage_id])
    zero = library.expand().run([target.passage_id], window_tokens=0)

    assert default.window_tokens == 8
    assert default.entries[0].context == _expected_context(library.paper, 1, 8)
    (entry,) = zero.entries
    assert entry.context is not None
    assert (entry.context.before, entry.context.after) == ("", "")
    assert not entry.context.at_source_start and not entry.context.at_source_end


def test_a_missing_id_is_not_found() -> None:
    library = Library()
    missing = make_passage_id("paper.pdf", 9999)

    (entry,) = library.expand().run([missing]).entries

    assert (entry.status, entry.hint) == ("not_found", NOT_FOUND_HINT)
    assert entry.passage is None and entry.context is None


def test_a_passage_whose_source_left_the_manifest_is_source_removed() -> None:
    library = Library()
    library.manifest.remove_source("paper.pdf")
    library.manifest.record_source(_MEMO, "skipped:delegated_to_ocr")

    entries = (
        library.expand()
        .run([library.paper[0].passage_id, library.memo[0].passage_id])
        .entries
    )

    for entry, path in zip(entries, ("paper.pdf", "memo.docx"), strict=True):
        assert (entry.status, entry.hint, entry.path) == (
            "source_removed",
            SOURCE_REMOVED_HINT,
            path,
        )
        assert entry.context is None


def test_a_passage_of_a_changed_source_is_source_modified() -> None:
    library = Library()
    library.manifest.record_source(replace(_PAPER, content_hash="hash-new"), "indexed")

    (entry,) = library.expand().run([library.paper[0].passage_id]).entries

    assert (entry.status, entry.hint, entry.path) == (
        "source_modified",
        SOURCE_MODIFIED_HINT,
        "paper.pdf",
    )
    assert (entry.indexed_content_hash, entry.current_content_hash) == (
        "hash-paper",
        "hash-new",
    )


def test_without_a_published_index_every_id_is_not_found() -> None:
    library = Library()
    empty = InMemoryPassageStore()

    result = library.expand(store=empty).run(
        [library.paper[0].passage_id, library.memo[0].passage_id]
    )

    assert [entry.status for entry in result.entries] == ["not_found", "not_found"]
    assert empty.outline_reads == [] and empty.text_fetches == []


def test_mixed_statuses_come_back_in_request_order_and_once_per_id() -> None:
    library = Library()
    library.manifest.remove_source("memo.docx")
    missing = make_passage_id("gone.txt", 0)
    ids = [
        library.paper[1].passage_id,
        missing,
        library.memo[0].passage_id,
        library.paper[1].passage_id,
        library.paper[0].passage_id,
    ]

    result = library.expand().run(ids)

    assert [(entry.passage_id, entry.status) for entry in result.entries] == [
        (library.paper[1].passage_id, "ok"),
        (missing, "not_found"),
        (library.memo[0].passage_id, "source_removed"),
        (library.paper[0].passage_id, "ok"),
    ]


@pytest.mark.parametrize(
    ("ids", "window", "message", "details"),
    [
        ([], None, "between 1 and 3 ids", {"minimum": 1, "maximum": 3}),
        (
            [make_passage_id("s", n) for n in range(4)],
            None,
            "between 1 and 3 ids",
            {"minimum": 1, "maximum": 3},
        ),
        (
            [make_passage_id("s", 0), "not-an-id"],
            None,
            "passage ids from search or browse results",
            {"malformed": ["not-an-id"]},
        ),
        (
            [make_passage_id("s", 0)],
            -1,
            "between 0 and 50",
            {"minimum": 0, "maximum": 50},
        ),
        (
            [make_passage_id("s", 0)],
            51,
            "between 0 and 50",
            {"minimum": 0, "maximum": 50},
        ),
    ],
)
def test_an_invalid_request_fails_before_any_read(
    ids: list[str], window: int | None, message: str, details: dict[str, object]
) -> None:
    library = Library()
    ocr_reads = library.ocr.live_output_reads

    with pytest.raises(ValidationError, match=message) as raised:
        library.expand(max_passages=3).run(ids, window_tokens=window)

    assert raised.value.details == details
    assert library.store.passage_fetches == []
    assert library.ocr.live_output_reads == ocr_reads


def test_one_outline_per_source_one_text_fetch_and_one_registry_read() -> None:
    library = Library()
    ids = [
        library.paper[0].passage_id,
        library.paper[3].passage_id,
        library.memo[1].passage_id,
        library.paper[2].passage_id,
    ]

    library.expand().run(ids, window_tokens=20)

    assert len(library.store.passage_fetches) == 1
    assert sorted(library.store.outline_reads) == ["memo.docx", "paper.pdf"]
    assert len(library.store.text_fetches) == 1
    assert library.ocr.live_output_reads == 1
    # Passages already in hand are not fetched again.
    (fetched,) = library.store.text_fetches
    assert not set(fetched) & set(ids)


class CountingManifest:
    """The in-memory manifest, counting reads by path."""

    def __init__(self, inner: InMemoryManifest) -> None:
        self.inner = inner
        self.paths: list[str] = []

    def get_source(self, path: str) -> ManifestSourceRecord | None:
        self.paths.append(path)
        return self.inner.get_source(path)

    def committed_sources(self) -> tuple[ManifestSourceRecord, ...]:
        return self.inner.committed_sources()


def test_the_manifest_is_read_once_per_source() -> None:
    library = Library()
    manifest = CountingManifest(library.manifest)

    library.expand(manifest=manifest).run(
        [item.passage_id for item in (*library.paper, *library.memo)]
    )

    assert sorted(manifest.paths) == ["memo.docx", "paper.pdf"]


class Recorder:
    """Forward to an object and record the name of every method called."""

    def __init__(self, inner: object, calls: list[str]) -> None:
        self._inner = inner
        self._calls = calls

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._inner, name)
        if callable(attribute):

            def call(*args: Any, **kwargs: Any) -> Any:
                self._calls.append(name)
                return attribute(*args, **kwargs)

            return call
        return attribute


def test_only_reads_are_made() -> None:
    library = Library()
    ocr_output = library.paper[0]
    calls: list[str] = []
    before = (library.store.count_passages(), library.manifest.committed_sources())

    library.expand(
        store=Recorder(library.store, calls),
        manifest=Recorder(library.manifest, calls),
        ocr=Recorder(library.ocr, calls),
    ).run([ocr_output.passage_id, library.memo[1].passage_id], window_tokens=20)

    assert set(calls) <= {
        "fetch_passages",
        "source_outline",
        "passage_texts",
        "get_source",
        "committed_sources",
        "live_outputs",
    }
    assert (library.store.count_passages(), library.manifest.committed_sources()) == (
        before
    )


class FailingStore(InMemoryPassageStore):
    def fetch_passages(
        self, passage_ids: Sequence[str]
    ) -> tuple[tuple[Passage, Source] | None, ...]:
        raise StoreUnavailable("Qdrant operation failed")


def test_an_unavailable_store_fails_the_request() -> None:
    library = Library()

    with pytest.raises(StoreUnavailable):
        library.expand(store=FailingStore()).run([library.paper[0].passage_id])


class LosingStore:
    """The library's store, losing one passage between the outline and the
    text fetch, as a concurrent update would."""

    def __init__(self, inner: InMemoryPassageStore, lost: str) -> None:
        self.inner = inner
        self.lost = lost

    def fetch_passages(
        self, passage_ids: Sequence[str]
    ) -> tuple[tuple[Passage, Source] | None, ...]:
        return self.inner.fetch_passages(passage_ids)

    def source_outline(self, source_id: str) -> tuple[OutlineEntry, ...]:
        return self.inner.source_outline(source_id)

    def passage_texts(self, passage_ids: Collection[str]) -> Mapping[str, Passage]:
        found = self.inner.passage_texts(passage_ids)
        return {key: value for key, value in found.items() if key != self.lost}


def test_a_neighbour_gone_during_the_request_shortens_the_window() -> None:
    library = Library()
    target = library.paper[2]
    store = LosingStore(library.store, library.paper[0].passage_id)

    (entry,) = (
        library.expand(store=store).run([target.passage_id], window_tokens=50).entries
    )

    assert entry.status == "ok" and entry.context is not None
    # Only the nearest earlier neighbour is left, and the window no longer
    # claims to reach the start of the source.
    assert (
        entry.context.before_locations
        == (_expected_context(library.paper, 2, 50).before_locations[1:])
    )
    assert (
        entry.context.before
        == library.paper[1].text[: target.chunk_offset - library.paper[1].chunk_offset]
    )
    assert not entry.context.at_source_start and entry.context.at_source_end


def test_the_source_is_sanitized_like_search_results() -> None:
    library = junk_metadata_library()
    ids = [
        make_passage_id(path, 0)
        for path in ("a/old.epub", "b/epoch.epub", "c/filetime.pdf", "d/book.ocr.pdf")
    ]
    expand = ExpandContext(
        store=library.store,
        manifest=library.manifest,
        ocr=library.ocr,
        max_passages=20,
        default_window_tokens=500,
        max_window_tokens=5000,
        now=lambda: datetime(2026, 10, 17, tzinfo=UTC),
    )

    result = expand.run(ids)

    assert {
        entry.source.path: (entry.source.title, entry.source.author, entry.source.year)
        for entry in result.entries
        if entry.source is not None
    } == {
        "a/old.epub": ("An Old Book", "Writer", 1974),
        "b/epoch.epub": ("A Modern Book", None, None),
        "c/filetime.pdf": (None, None, None),
        "d/book.ocr.pdf": ("Scanned Classic", "C. Old", 1958),
    }
    assert library.ocr.live_output_reads == 1
