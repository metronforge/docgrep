"""The context reads of the Qdrant store (UC-12): passages by id, the
outline of a source and passage texts, in local mode and, with
DOCGREP_TEST_QDRANT_URL set, against a server."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from docgrep.adapters.qdrant import QdrantPassageStore
from docgrep.domain.context import OutlineEntry
from docgrep.domain.models import FileType, Passage, Source, make_passage_id
from docgrep.domain.protocols import PassageContextReader

URL = os.environ.get("DOCGREP_TEST_QDRANT_URL")

_BOOK = Source(
    "books/Ландау.pdf",
    "books/Ландау.pdf",
    FileType.PDF,
    title="Mechanics",
    author="L. Landau",
    year=1957,
    content_hash="hash-book",
)
_NOTE = Source("note.txt", "note.txt", FileType.TXT, content_hash="hash-note")


def _passage(source: Source, offset: int, text: str, page: int | None) -> Passage:
    return Passage(
        make_passage_id(source.source_id, offset),
        source.source_id,
        text,
        offset,
        len(text.split()),
        page_number=page,
        heading_path=("Chapter",) if page else (),
    )


# Written out of order, so the outline has to sort them.
_BOOK_PASSAGES = (
    _passage(_BOOK, 40, "third passage of the book", 2),
    _passage(_BOOK, 0, "first passage", 1),
    _passage(_BOOK, 12, "second passage here", 1),
)
_NOTE_PASSAGES = (_passage(_NOTE, 0, "a note", None),)


def _fill(store: QdrantPassageStore, *, publish: bool = True) -> None:
    store.recreate_collection(2)
    store.upsert_batch(
        [
            (passage, source, (1.0, 0.0), {1: 1.0})
            for source, passages in ((_BOOK, _BOOK_PASSAGES), (_NOTE, _NOTE_PASSAGES))
            for passage in passages
        ]
    )
    if publish:
        store.write_completion_metadata({"embedding_model": "fake"})


def _wipe_server_alias(url: str, alias: str) -> None:
    """Remove the alias and every collection made for it, on a server the
    suite owns (see qdrant_server_guard)."""
    client = QdrantClient(url=url)
    try:
        if any(item.alias_name == alias for item in client.get_aliases().aliases):
            client.update_collection_aliases(
                [
                    models.DeleteAliasOperation(
                        delete_alias=models.DeleteAlias(alias_name=alias)
                    )
                ]
            )
        for collection in client.get_collections().collections:
            if collection.name.startswith(f"{alias}_"):
                client.delete_collection(collection.name)
    finally:
        client.close()


@pytest.fixture(
    params=[
        "local",
        pytest.param(
            "server",
            marks=[
                pytest.mark.qdrant_server,
                pytest.mark.skipif(
                    not URL, reason="DOCGREP_TEST_QDRANT_URL is not set"
                ),
            ],
        ),
    ]
)
def store(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Iterator[QdrantPassageStore]:
    if request.param == "local":
        opened = QdrantPassageStore(tmp_path / "qdrant")
    else:
        assert URL is not None
        _wipe_server_alias(URL, "alpha")
        opened = QdrantPassageStore(url=URL, alias="alpha")
    try:
        yield opened
    finally:
        opened.close()
        if URL is not None and request.param == "server":
            _wipe_server_alias(URL, "alpha")


def test_passages_come_back_in_request_order_with_their_source(
    store: QdrantPassageStore,
) -> None:
    _fill(store)
    reader: PassageContextReader = store
    missing = make_passage_id(_BOOK.source_id, 999)
    wanted = [
        _BOOK_PASSAGES[0].passage_id,
        missing,
        _NOTE_PASSAGES[0].passage_id,
        _BOOK_PASSAGES[0].passage_id,
    ]

    found = reader.fetch_passages(wanted)

    assert found[1] is None
    assert found[0] == found[3] == (_BOOK_PASSAGES[0], _BOOK)
    assert found[2] == (_NOTE_PASSAGES[0], _NOTE)


def test_the_outline_orders_one_source_by_offset_without_text(
    store: QdrantPassageStore,
) -> None:
    _fill(store)

    outline = store.source_outline(_BOOK.source_id)

    assert outline == tuple(
        OutlineEntry(item.passage_id, item.chunk_offset, item.token_count)
        for item in sorted(_BOOK_PASSAGES, key=lambda item: item.chunk_offset)
    )
    assert store.source_outline("unknown.txt") == ()


def test_texts_come_back_by_id_and_skip_missing_ids(
    store: QdrantPassageStore,
) -> None:
    _fill(store)
    wanted = {item.passage_id for item in _BOOK_PASSAGES[1:]}

    texts = store.passage_texts({*wanted, make_passage_id("x", 1)})

    assert texts == {item.passage_id: item for item in _BOOK_PASSAGES[1:]}
    assert store.passage_texts(set()) == {}


def test_nothing_is_read_from_an_unpublished_collection(
    store: QdrantPassageStore,
) -> None:
    passage_id = _BOOK_PASSAGES[0].passage_id
    assert store.fetch_passages([passage_id]) == (None,)
    _fill(store, publish=False)

    assert store.fetch_passages([passage_id]) == (None,)
    assert store.source_outline(_BOOK.source_id) == ()
    assert store.passage_texts({passage_id}) == {}


def test_the_completion_marker_is_never_a_passage(
    store: QdrantPassageStore,
) -> None:
    _fill(store)
    # The id of the marker point is derived from this name, not from a
    # passage id; asking for it must not return the marker.
    assert store.fetch_passages(["docgrep:completion"]) == (None,)
