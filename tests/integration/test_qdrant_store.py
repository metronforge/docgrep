"""Integration checks for the embedded Qdrant passage store."""

import warnings
from pathlib import Path

import pytest

from docgrep.adapters.qdrant import QdrantPassageStore, _without_local_index_warning
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


def test_building_a_collection_does_not_emit_the_local_index_notice(
    tmp_path: Path,
) -> None:
    store = QdrantPassageStore(tmp_path / "qdrant")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        store.recreate_collection(2)
        store.upsert(_passage(_source(), "alpha beta"), _source(), (1.0, 0.0), {1: 1.0})
        store.write_completion_metadata({"embedding_model": "m"})

    assert [str(item.message) for item in caught] == []


def test_only_the_local_index_notice_is_suppressed() -> None:
    notice = "Payload indexes have no effect in the local Qdrant. Use a server."
    large = (
        "Local mode is not recommended for collections with more than 20,000 points."
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _without_local_index_warning():
            warnings.warn(notice, UserWarning, stacklevel=1)
            warnings.warn(large, UserWarning, stacklevel=1)
            warnings.warn("something else", RuntimeWarning, stacklevel=1)
        warnings.warn(notice, UserWarning, stacklevel=1)

    assert [str(item.message) for item in caught] == [large, "something else", notice]


def _hybrid_store(tmp_path: Path) -> QdrantPassageStore:
    """Three passages: two close to the query vector, one lexical match only."""
    source = _source()
    store = QdrantPassageStore(tmp_path / "qdrant")
    store.recreate_collection(2)
    vectors = {
        "near one": (1.0, 0.0),
        "near two": (0.9, 0.435889894),
        "zebra lexical": (0.6, 0.8),
    }
    for offset, (text, vector) in enumerate(vectors.items()):
        sparse = {777: 1.0} if "zebra" in text else {offset + 1: 1.0}
        store.upsert(_passage(source, text, offset), source, vector, sparse)
    store.write_completion_metadata({"embedding_model": "m"})
    return store


def test_sparse_only_hit_carries_its_real_dense_score(tmp_path: Path) -> None:
    store = _hybrid_store(tmp_path)

    candidates = store.retrieve((1.0, 0.0), {777: 1.0}, 2)

    by_text = {item.passage.text: item for item in candidates}
    assert "zebra lexical" in by_text, "the lexical match must be recalled"
    assert "near two" not in by_text, "it must be outside the dense top 2"
    assert by_text["zebra lexical"].dense_score == pytest.approx(0.6, abs=1e-6)
    assert by_text["near one"].dense_score == pytest.approx(1.0, abs=1e-6)


def test_candidates_come_back_in_first_stage_order_with_ranks(tmp_path: Path) -> None:
    store = _hybrid_store(tmp_path)

    candidates = store.retrieve((1.0, 0.0), {777: 1.0}, 3)

    assert [item.stage1_rank for item in candidates] == [0, 1, 2]
    assert {item.passage.text for item in candidates} == {
        "near one",
        "near two",
        "zebra lexical",
    }
    # The fused list is not re-sorted by dense similarity.
    dense_only = sorted(candidates, key=lambda item: -item.dense_score)
    assert [item.passage.text for item in dense_only][0] == "near one"
    assert candidates[0].passage.text in {"near one", "zebra lexical"}
