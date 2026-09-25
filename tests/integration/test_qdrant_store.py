"""Integration checks for the embedded Qdrant passage store."""

from pathlib import Path

import pytest

from docgrep.adapters.qdrant import QdrantPassageStore
from docgrep.domain.errors import StoreUnavailable
from docgrep.domain.models import FileType, Passage, Source, make_passage_id
from docgrep.domain.protocols import PassageRetriever, PassageWriter


def _source(name: str = "notes.txt") -> Source:
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
        indexed_at="2026-09-25T00:00:00Z",
        parser_version="1",
    )


def _passage(source: Source, text: str, offset: int = 0) -> Passage:
    return Passage(
        make_passage_id(source.source_id, offset),
        source.source_id,
        text,
        offset,
        len(text.split()),
        page_number=2,
        heading_path=("Section",),
    )


def _store(path: Path) -> QdrantPassageStore:
    store = QdrantPassageStore(path)
    writer: PassageWriter = store
    reader: PassageRetriever = store
    assert writer is reader
    return store


def test_payload_and_completion_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path / "qdrant")
    source = _source()
    passage = _passage(source, "Exact original text")
    store.recreate_collection(2)
    store.upsert(passage, source, (1.0, 0.0), {7: 1.0})
    assert store.read_completion_metadata() is None
    store.write_completion_metadata(
        {"embedding_model": "fake", "embedding_dimension": 2}
    )
    assert store.read_completion_metadata() == {
        "embedding_model": "fake",
        "embedding_dimension": 2,
    }
    assert store.count_passages() == 1
    assert store.retrieve((1.0, 0.0), {7: 1.0}, 10)[0].passage == passage
    assert store.retrieve((1.0, 0.0), {7: 1.0}, 10)[0].source == source


def test_lexical_match_recalled_when_dense_score_is_poor(tmp_path: Path) -> None:
    store = _store(tmp_path / "qdrant")
    store.recreate_collection(2)
    source = _source()
    lexical = _passage(source, "rare exact term", 0)
    dense = _passage(source, "unrelated", 20)
    store.upsert(lexical, source, (-1.0, 0.0), {101: 1.0})
    store.upsert(dense, source, (1.0, 0.0), {202: 1.0})
    store.write_completion_metadata({"embedding_model": "fake"})
    hits = store.retrieve((1.0, 0.0), {101: 1.0}, 2)
    assert {hit.passage.passage_id for hit in hits} == {
        lexical.passage_id,
        dense.passage_id,
    }
    assert hits[0].passage == lexical


def test_rebuild_keeps_old_alias_until_completion(tmp_path: Path) -> None:
    store = _store(tmp_path / "qdrant")
    source = _source()
    old = _passage(source, "old text")
    new = _passage(source, "new text", 10)
    store.recreate_collection(2)
    store.upsert(old, source, (1.0, 0.0), {1: 1.0})
    store.write_completion_metadata({"embedding_model": "old"})
    old_collection = store._active_collection()
    store.recreate_collection(2)
    store.upsert(new, source, (1.0, 0.0), {1: 1.0})
    assert store.read_completion_metadata() == {"embedding_model": "old"}
    assert store.retrieve((1.0, 0.0), {1: 1.0}, 10)[0].passage == old
    store.write_completion_metadata({"embedding_model": "new"})
    assert store.read_completion_metadata() == {"embedding_model": "new"}
    assert store.retrieve((1.0, 0.0), {1: 1.0}, 10)[0].passage == new
    assert old_collection not in {
        collection.name for collection in store._client.get_collections().collections
    }


def test_empty_completed_collection_is_distinct_from_missing_marker(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "qdrant")
    assert store.read_completion_metadata() is None
    store.recreate_collection(2)
    store.write_completion_metadata({"embedding_model": "fake"})
    assert store.read_completion_metadata() == {"embedding_model": "fake"}
    assert store.count_passages() == 0
    assert store.retrieve((1.0, 0.0), {1: 1.0}, 10) == ()


def test_aborted_rebuild_preserves_previous_alias(tmp_path: Path) -> None:
    store = _store(tmp_path / "qdrant")
    source = _source()
    passage = _passage(source, "kept")
    store.recreate_collection(2)
    store.upsert(passage, source, (1.0, 0.0), {1: 1.0})
    store.write_completion_metadata({"embedding_model": "old"})
    store.recreate_collection(2)
    store.clear_completion_metadata()
    assert store.read_completion_metadata() == {"embedding_model": "old"}
    assert store.retrieve((1.0, 0.0), {1: 1.0}, 10)[0].passage == passage


def test_unusable_storage_path_raises_domain_error(tmp_path: Path) -> None:
    path = tmp_path / "file"
    path.write_text("not a directory")
    with pytest.raises(StoreUnavailable):
        QdrantPassageStore(path)


def test_qdrant_query_failure_raises_domain_error(tmp_path: Path) -> None:
    store = _store(tmp_path / "qdrant")
    store.recreate_collection(2)
    store.write_completion_metadata({"embedding_model": "fake"})
    with pytest.raises(StoreUnavailable):
        store.retrieve((1.0,), {1: 1.0}, 10)
