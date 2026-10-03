"""Batched passage writes: same points and results as one write per passage."""

import random
from pathlib import Path

import pytest
from pydantic import ValidationError as SettingsValidationError
from qdrant_client import QdrantClient

from docgrep.adapters.qdrant import QdrantPassageStore
from docgrep.api.container import create_container
from docgrep.domain.errors import StoreUnavailable
from docgrep.domain.models import FileType, Passage, Source, make_passage_id
from docgrep.settings import Settings

QUERIES = ("w7 w12", "w301", "w55 w2 w9", "unmatched")


def _corpus(root: Path) -> Path:
    root.mkdir()
    rng = random.Random(3)
    (root / "big.txt").write_text(
        " ".join(f"w{rng.randrange(400)}" for _ in range(3000)), encoding="utf-8"
    )
    (root / "small.md").write_text(
        "# Title\n\nw7 w12 and some more words here", "utf-8"
    )
    (root / "tiny.txt").write_text("w301 alone", encoding="utf-8")
    return root


def _build_and_dump(
    tmp_path: Path, corpus: Path, batch_size: int
) -> tuple[dict[str, object], list[list[tuple[object, ...]]], int]:
    settings = Settings(
        corpus_root=corpus,
        data_dir=tmp_path / f"data-{batch_size}",
        backend="fake",
        chunk_size=10,
        chunk_overlap=2,
        upsert_batch_size=batch_size,
    )
    services = create_container(settings)
    summary = services.build.run()
    assert (summary.indexed, summary.failed) == (3, 0)
    results = [
        [
            (hit.passage.passage_id, hit.score, hit.dense_score, hit.passage.text)
            for hit in services.search.run(query, top_n=10).results
        ]
        for query in QUERIES
    ]
    store = services.build.passage_writer
    assert isinstance(store, QdrantPassageStore)
    count = store.count_passages()
    store.close()
    client = QdrantClient(path=str(settings.data_dir / "qdrant"))
    try:
        points: dict[str, object] = {}
        offset = None
        while True:
            batch, offset = client.scroll(
                "docgrep",
                limit=200,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for point in batch:
                payload = dict(point.payload or {})
                if payload.get("record_type") == "completion":
                    # Everything but the build timestamp must match as well.
                    payload["metadata"] = {
                        k: v
                        for k, v in payload["metadata"].items()
                        if k != "completed_at"
                    }
                points[str(point.id)] = (payload, point.vector)
            if offset is None:
                break
    finally:
        client.close()
    return points, results, count


def test_batch_sizes_store_identical_points_and_give_identical_results(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path / "corpus")

    dumps = {n: _build_and_dump(tmp_path, corpus, n) for n in (1, 7, 256)}

    reference = dumps[1]
    assert reference[2] > 300, "the corpus must span several batches of 256"
    assert any(reference[1]), "the queries must find something"
    for size in (7, 256):
        points, results, count = dumps[size]
        assert count == reference[2]
        assert points.keys() == reference[0].keys()
        assert points == reference[0]
        assert results == reference[1]


def test_upsert_batch_size_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DOCGREP_UPSERT_BATCH_SIZE", raising=False)
    assert Settings().upsert_batch_size == 256

    monkeypatch.setenv("DOCGREP_UPSERT_BATCH_SIZE", "64")
    assert Settings().upsert_batch_size == 64
    for bad in ("0", "-5", "many"):
        monkeypatch.setenv("DOCGREP_UPSERT_BATCH_SIZE", bad)
        with pytest.raises(SettingsValidationError):
            Settings()


def _item(
    name: str, offset: int, dimension: int = 2
) -> tuple[Passage, Source, tuple[float, ...], dict[int, float]]:
    source = Source(name, name, FileType.TXT)
    passage = Passage(
        make_passage_id(name, offset), name, f"{name} {offset}", offset, 2
    )
    return passage, source, (1.0,) + (0.0,) * (dimension - 1), {offset + 1: 1.0}


def test_an_empty_batch_is_a_no_op_and_a_bad_item_writes_nothing(
    tmp_path: Path,
) -> None:
    store = QdrantPassageStore(tmp_path / "q")
    with pytest.raises(StoreUnavailable, match="No collection"):
        store.upsert_batch([_item("a.txt", 0)])
    store.recreate_collection(2)
    store.upsert_batch([])
    store.write_completion_metadata({"embedding_model": "m"})
    assert store.count_passages() == 0

    store.recreate_collection(2)
    with pytest.raises(StoreUnavailable) as raised:
        store.upsert_batch([_item("a.txt", 0), _item("a.txt", 1, dimension=3)])
    assert "dimension" in str(raised.value.details["reason"])
    store.upsert_batch([_item("a.txt", i) for i in range(5)])
    store.write_completion_metadata({"embedding_model": "m"})

    assert store.count_passages() == 5


def test_single_and_batched_upserts_store_the_same_point(tmp_path: Path) -> None:
    def dump(path: Path, batched: bool) -> object:
        store = QdrantPassageStore(path)
        store.recreate_collection(2)
        items = [_item("a.txt", i) for i in range(3)]
        if batched:
            store.upsert_batch(items)
        else:
            for item in items:
                store.upsert(*item)
        store.write_completion_metadata({"embedding_model": "m"})
        store.close()
        client = QdrantClient(path=str(path))
        try:
            points, _ = client.scroll(
                "docgrep", limit=10, with_payload=True, with_vectors=True
            )
            return sorted(((str(p.id), p.payload, p.vector) for p in points), key=str)
        finally:
            client.close()

    assert dump(tmp_path / "one", False) == dump(tmp_path / "many", True)
