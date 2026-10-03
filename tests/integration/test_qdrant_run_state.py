"""Staging lifecycle, orphan cleanup and delete-by-source on the Qdrant adapter."""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from docgrep.adapters.manifest import SQLiteManifest
from docgrep.adapters.models.deterministic import DeterministicEmbedder
from docgrep.adapters.qdrant import QdrantPassageStore
from docgrep.api.container import Services, create_container
from docgrep.domain.errors import StoreUnavailable
from docgrep.domain.models import FileType, Passage, Source, make_passage_id
from docgrep.settings import Settings


def _source(name: str) -> Source:
    return Source(name, name, FileType.TXT, title=name)


def _passage(source: Source, text: str, offset: int) -> Passage:
    return Passage(
        make_passage_id(source.source_id, offset),
        source.source_id,
        text,
        offset,
        len(text.split()),
    )


def _publish(store: QdrantPassageStore, texts_by_source: dict[str, list[str]]) -> None:
    store.recreate_collection(2)
    offset = 0
    for name, texts in texts_by_source.items():
        for text in texts:
            store.upsert(
                _passage(_source(name), text, offset),
                _source(name),
                (1.0, 0.0),
                {1: 1.0},
            )
            offset += 1
    store.write_completion_metadata({"embedding_model": "m"})


def _orphan(store: QdrantPassageStore) -> str:
    """Leave a staging collection behind, as a crashed build would."""
    store.recreate_collection(2)
    source = _source("half.txt")
    store.upsert(_passage(source, "half built", 0), source, (1.0, 0.0), {1: 1.0})
    name = store.staging_name()
    assert name is not None
    return name


def _names(path: Path) -> tuple[dict[str, str], set[str]]:
    client = QdrantClient(path=str(path))
    try:
        aliases = {
            a.alias_name: a.collection_name for a in client.get_aliases().aliases
        }
        return aliases, {c.name for c in client.get_collections().collections}
    finally:
        client.close()


def test_staging_name_follows_the_build_lifecycle(tmp_path: Path) -> None:
    store = QdrantPassageStore(tmp_path / "q")
    assert store.staging_name() is None

    store.recreate_collection(2)
    name = store.staging_name()
    assert name is not None and name.startswith("docgrep_") and len(name) == 8 + 32

    store.write_completion_metadata({"embedding_model": "m"})
    assert store.staging_name() is None

    store.recreate_collection(2)
    store.clear_completion_metadata()
    assert store.staging_name() is None


def test_a_crashed_build_leaves_a_listed_leftover_beside_the_published_index(
    tmp_path: Path,
) -> None:
    path = tmp_path / "q"
    store = QdrantPassageStore(path)
    _publish(store, {"a.txt": ["alpha"]})
    assert store.staging_collections() == ()
    orphan = _orphan(store)
    store.close()

    fresh = QdrantPassageStore(path)
    assert fresh.staging_collections() == (orphan,)
    published = fresh.count_passages()
    fresh.close()

    aliases, collections = _names(path)
    assert collections == {aliases["docgrep"], orphan}
    assert published == 1


def test_drop_collection_removes_only_unpublished_staging_collections(
    tmp_path: Path,
) -> None:
    path = tmp_path / "q"
    first = QdrantPassageStore(path)
    _publish(first, {"a.txt": ["alpha"]})
    orphan = _orphan(first)
    first.close()
    store = QdrantPassageStore(path)
    published = store._active_collection()
    assert published is not None

    with pytest.raises(StoreUnavailable):
        store.drop_collection(published)
    with pytest.raises(StoreUnavailable):
        store.drop_collection("not-a-staging-collection")
    store.drop_collection(orphan)

    assert store.staging_collections() == ()
    assert store.count_passages() == 1
    store.close()
    assert published in _names(path)[1]


def test_a_longer_alias_with_the_same_prefix_is_not_listed(tmp_path: Path) -> None:
    path = tmp_path / "q"
    other = QdrantPassageStore(path, alias="docgrep_x")
    _publish(other, {"a.txt": ["alpha"]})
    foreign = _orphan(other)
    other.close()

    store = QdrantPassageStore(path)  # default alias "docgrep"
    _publish(store, {"b.txt": ["beta"]})

    assert foreign.startswith("docgrep_x_")
    assert store.staging_collections() == ()
    with pytest.raises(StoreUnavailable):
        store.drop_collection(foreign)


def test_delete_source_points_removes_exactly_that_source(tmp_path: Path) -> None:
    store = QdrantPassageStore(tmp_path / "q")
    _publish(
        store, {"a.txt": ["a one", "a two"], "b.txt": ["b one"], "c.txt": ["c one"]}
    )

    removed = store.delete_source_points("a.txt")

    assert removed == 2
    assert store.count_passages() == 2
    texts = {c.passage.text for c in store.retrieve((1.0, 0.0), {1: 1.0}, 10)}
    assert texts == {"b one", "c one"}
    assert store.read_completion_metadata() == {"embedding_model": "m"}
    assert store.delete_source_points("a.txt") == 0
    assert store.delete_source_points("missing.txt") == 0
    assert store.count_passages() == 2


def test_delete_source_points_targets_the_open_build_not_the_live_index(
    tmp_path: Path,
) -> None:
    store = QdrantPassageStore(tmp_path / "q")
    _publish(store, {"a.txt": ["live a"]})
    store.recreate_collection(2)
    source = _source("a.txt")
    store.upsert(_passage(source, "staged a", 0), source, (1.0, 0.0), {1: 1.0})

    assert store.delete_source_points("a.txt") == 1

    assert store.count_passages() == 1
    assert [c.passage.text for c in store.retrieve((1.0, 0.0), {1: 1.0}, 5)] == [
        "live a"
    ]


def test_delete_source_points_on_an_empty_store_removes_nothing(tmp_path: Path) -> None:
    assert QdrantPassageStore(tmp_path / "q").delete_source_points("a.txt") == 0


# --- the whole pipeline with real SQLite and real local Qdrant ---------------


class FailForSource:
    """Delegate to a real writer but fail the upsert of one source."""

    def __init__(self, inner: QdrantPassageStore, path: str) -> None:
        self.inner = inner
        self.path = path

    def upsert(
        self,
        passage: Passage,
        source: Source,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float],
    ) -> None:
        if source.path == self.path:
            raise StoreUnavailable("disk full")
        self.inner.upsert(passage, source, dense_vector, sparse_vector)

    def upsert_batch(
        self,
        items: Sequence[tuple[Passage, Source, Sequence[float], Mapping[int, float]]],
    ) -> None:
        if any(source.path == self.path for _, source, _, _ in items):
            raise StoreUnavailable("disk full")
        self.inner.upsert_batch(items)

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)


def _services(tmp_path: Path) -> tuple[Services, Path]:
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    settings = Settings(
        corpus_root=corpus,
        data_dir=tmp_path / "data",
        backend="fake",
        chunk_size=10,
        chunk_overlap=2,
    )
    return create_container(settings), corpus


def _hits(services: Services, query: str) -> list[str]:
    result = services.search.run(query, top_n=5)
    return [hit.source.path for hit in result.results if query in hit.passage.text]


def test_aborted_and_running_builds_leave_search_and_sources_untouched(
    tmp_path: Path,
) -> None:
    services, corpus = _services(tmp_path)
    (corpus / "a.txt").write_text("alpha apple", encoding="utf-8")
    (corpus / "b.txt").write_text("beta banana", encoding="utf-8")
    services.build.run()
    manifest = SQLiteManifest(tmp_path / "data")
    assert _hits(services, "apple") == ["a.txt"]
    first_sources = {p: manifest.get_source(p) for p in ("a.txt", "b.txt")}
    assert all(r is not None and r.status == "indexed" for r in first_sources.values())

    (corpus / "a.txt").unlink()
    (corpus / "c.txt").write_text("gamma cherry", encoding="utf-8")
    inner = services.build.passage_writer
    assert isinstance(inner, QdrantPassageStore)
    during: dict[str, object] = {}

    class Probe(DeterministicEmbedder):
        def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
            if any("cherry" in t for t in texts) and not during:
                during["apple"] = _hits(services, "apple")
                during["cherry"] = _hits(services, "cherry")
                during["a"] = manifest.get_source("a.txt")
                during["c"] = manifest.get_source("c.txt")
                run = manifest.latest_run()
                assert run is not None
                during["status"] = run.status
                during["journal"] = [
                    r.source.path for r in manifest.run_sources(run.run_id)
                ]
            return super().embed(texts)

    services.build.embedder = Probe()
    services.build.passage_writer = FailForSource(inner, "c.txt")

    with pytest.raises(StoreUnavailable):
        services.build.run()

    assert during["status"] == "running"
    assert during["apple"] == ["a.txt"] and during["cherry"] == []
    assert during["a"] is not None and during["c"] is None
    assert during["journal"] == ["b.txt"]
    # After the abort the live index and the committed sources are still the old ones.
    assert _hits(services, "apple") == ["a.txt"]
    assert _hits(services, "cherry") == []
    assert (
        manifest.get_source("a.txt") is not None
        and manifest.get_source("c.txt") is None
    )
    run = manifest.latest_run()
    assert run is not None and run.status == "aborted"
    assert run.staging_collection is not None
    assert [
        (r.source.path, r.status, r.passages) for r in manifest.run_sources(run.run_id)
    ] == [("b.txt", "indexed", 1)]
    # The staging collection is retained so the run can be resumed.
    assert inner.staging_collections() == (run.staging_collection,)
    assert run.staging_collection != inner._active_collection()


def test_a_completed_build_swaps_the_index_and_then_commits_the_sources(
    tmp_path: Path,
) -> None:
    services, corpus = _services(tmp_path)
    (corpus / "a.txt").write_text("alpha apple", encoding="utf-8")
    services.build.run()
    manifest = SQLiteManifest(tmp_path / "data")
    (corpus / "a.txt").unlink()
    (corpus / "c.txt").write_text("gamma cherry", encoding="utf-8")
    (corpus / "d.bin").write_bytes(b"x")

    summary = services.build.run()

    assert (summary.indexed, summary.skipped, summary.failed) == (1, 1, 0)
    assert _hits(services, "cherry") == ["c.txt"] and _hits(services, "apple") == []
    assert manifest.get_source("a.txt") is None
    c_source = manifest.get_source("c.txt")
    assert c_source is not None and c_source.status == "indexed"
    d_source = manifest.get_source("d.bin")
    assert d_source is not None and d_source.status == "skipped:unsupported"
    run = manifest.latest_run()
    assert run is not None and run.status == "completed"
    assert run.staging_collection is not None
    assert len(manifest.run_sources(run.run_id)) == 2


def test_an_orphan_from_an_earlier_crash_is_removed_by_the_next_build(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    crashed = QdrantPassageStore(tmp_path / "data" / "qdrant")
    _publish(crashed, {"old.txt": ["old"]})
    orphan = _orphan(crashed)
    crashed.close()
    services, corpus = _services(tmp_path)
    (corpus / "a.txt").write_text("alpha apple", encoding="utf-8")
    store = services.build.passage_writer
    assert isinstance(store, QdrantPassageStore)
    assert store.staging_collections() == (orphan,)

    with caplog.at_level(logging.INFO, logger="docgrep"):
        services.build.run()

    dropped = [
        r.__dict__["collection"]
        for r in caplog.records
        if r.getMessage() == "orphaned_staging_collection_dropped"
    ]
    assert dropped == [orphan]
    assert store.staging_collections() == ()
    assert _hits(services, "apple") == ["a.txt"]
