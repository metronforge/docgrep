"""Reading the passages of one source with their vectors, in pages, from the
local-mode store (ADR-0028)."""

from collections.abc import Callable
from pathlib import Path

import pytest

from docgrep.adapters.qdrant import QdrantPassageStore
from docgrep.domain.errors import StoreUnavailable
from docgrep.domain.models import FileType, Passage, Source, make_passage_id
from docgrep.domain.protocols import (
    LiveIndexWriter,
    PassageWriter,
    StoredPassage,
    StoredPassagePage,
)


def _source(name: str) -> Source:
    return Source(
        source_id=name,
        path=name,
        file_type=FileType.TXT,
        title="Notes",
        author="A. Reader",
        year=2024,
        content_hash="abc",
        mtime=12.5,
        size=42,
        parser_version="1",
    )


def _items(source: Source, count: int) -> list[StoredPassage]:
    return [
        (
            Passage(
                make_passage_id(source.source_id, offset),
                source.source_id,
                f"text {offset}",
                offset,
                2,
                page_number=offset + 1,
                heading_path=("Part", f"{offset}"),
            ),
            source,
            # Unit length and exact in float32, so they read back as given.
            (0.0, 1.0) if offset % 2 else (1.0, 0.0),
            {offset + 1: 0.5, 99: 1.25},
        )
        for offset in range(count)
    ]


def _read_all(
    read: Callable[[str, str | None, int], StoredPassagePage],
    source_id: str,
    limit: int,
) -> tuple[list[StoredPassage], int]:
    """Every passage of a source, by chunk offset, and the pages read."""
    pages = 0
    found: list[StoredPassage] = []
    offset: str | None = None
    while True:
        page, offset = read(source_id, offset, limit)
        pages += 1
        found.extend(page)
        if offset is None:
            return sorted(found, key=lambda item: item[0].chunk_offset), pages


def test_the_pages_hold_every_passage_of_the_source_with_its_vectors(
    tmp_path: Path,
) -> None:
    store = QdrantPassageStore(tmp_path / "qdrant")
    writer: PassageWriter = store
    live: LiveIndexWriter = store
    wanted, other = _items(_source("a.txt"), 5), _items(_source("b.txt"), 3)
    writer.recreate_collection(2)
    writer.upsert_batch([*wanted, *other])

    staged, pages = _read_all(writer.staged_source_page, "a.txt", 2)

    assert pages == 3
    assert staged == wanted
    with pytest.raises(StoreUnavailable):
        live.live_source_page("a.txt", None, 2)

    writer.write_completion_metadata({"embedding_model": "fake"})
    published, _ = _read_all(live.live_source_page, "b.txt", 10)

    assert published == other
    with pytest.raises(StoreUnavailable):
        writer.staged_source_page("a.txt", None, 2)
    empty: StoredPassagePage = live.live_source_page("missing.txt", None, 2)
    assert empty == ((), None)
    store.close()
