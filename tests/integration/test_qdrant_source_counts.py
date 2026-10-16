"""Per-source passage counts of the embedded Qdrant store (UC-08)."""

from pathlib import Path

import pytest

from docgrep.adapters.qdrant import QdrantPassageStore
from docgrep.domain.errors import IndexIncomplete
from docgrep.domain.models import FileType, Passage, Source, make_passage_id
from docgrep.domain.protocols import SourcePassageCounter


def _source(name: str) -> Source:
    return Source(source_id=name, path=name, file_type=FileType.TXT)


def _passage(source: Source, offset: int) -> Passage:
    return Passage(
        make_passage_id(source.source_id, offset),
        source.source_id,
        f"text {offset}",
        offset,
        2,
    )


def test_counts_only_the_published_passages_of_one_source(tmp_path: Path) -> None:
    store = QdrantPassageStore(tmp_path / "qdrant")
    counter: SourcePassageCounter = store
    first, second = _source("a b.txt"), _source("Ландау [1].txt")
    try:
        with pytest.raises(IndexIncomplete):
            counter.count_source_passages(first.source_id)
        store.recreate_collection(2)
        for source, total in ((first, 3), (second, 1)):
            for offset in range(total):
                store.upsert(_passage(source, offset), source, (1.0, 0.0), {1: 1.0})
        # Not published yet: the staging collection is not counted.
        with pytest.raises(IndexIncomplete):
            counter.count_source_passages(first.source_id)
        store.write_completion_metadata({"embedding_model": "fake"})

        assert counter.count_source_passages(first.source_id) == 3
        assert counter.count_source_passages(second.source_id) == 1
        assert counter.count_source_passages("missing.txt") == 0
        # The completion marker is not a passage of any source.
        assert store.count_passages() == 4
    finally:
        store.close()
