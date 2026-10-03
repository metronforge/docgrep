"""Crash recovery and aborts on the real SQLite manifest and local Qdrant."""

import logging
from pathlib import Path

import pytest

from docgrep.adapters.manifest import SQLiteManifest
from docgrep.adapters.qdrant import QdrantPassageStore
from docgrep.api.container import Services, create_container
from docgrep.domain.errors import StoreUnavailable
from docgrep.settings import Settings


class Crash(BaseException):
    """A process death: nothing in the build handles it."""


class CrashOnCommit:
    def __init__(self, inner: SQLiteManifest) -> None:
        self.inner = inner

    def commit_run(self, run_id: str) -> None:
        raise Crash

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


def test_a_crash_after_the_swap_is_recovered_by_the_next_build_start(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    services, corpus = _services(tmp_path)
    (corpus / "a.txt").write_text("alpha apple", encoding="utf-8")
    services.build.run()
    (corpus / "a.txt").unlink()
    (corpus / "c.txt").write_text("gamma cherry", encoding="utf-8")
    real = services.build.manifest
    services.build.manifest = CrashOnCommit(real)  # type: ignore[assignment]  # test double
    with pytest.raises(Crash):
        services.build.run()
    manifest = SQLiteManifest(tmp_path / "data")
    crashed = manifest.latest_run()
    assert crashed is not None and crashed.status == "swapped"
    assert crashed.total_sources == 1
    # The index is already the new one; the sources still describe the old one.
    assert _hits(services, "cherry") == ["c.txt"]
    assert manifest.get_source("a.txt") is not None
    assert manifest.get_source("c.txt") is None

    services.build.manifest = real
    with caplog.at_level(logging.WARNING, logger="docgrep"):
        services.build.run()

    assert [r for r in caplog.records if r.getMessage() == "run_recovered"]
    after = SQLiteManifest(tmp_path / "data")
    first = [r for r in [after.latest_run()] if r is not None][0]
    assert first.status == "completed" and first.run_id != crashed.run_id
    c_source = after.get_source("c.txt")
    assert c_source is not None and c_source.status == "indexed"
    assert after.get_source("a.txt") is None


class FailUpserts:
    """Delegate to the real writer but fail every upsert."""

    def __init__(self, inner: QdrantPassageStore) -> None:
        self.inner = inner

    def upsert_batch(self, items: object) -> None:
        raise StoreUnavailable("disk full")

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)


def test_an_abort_retains_staging_and_the_live_index_keeps_serving(
    tmp_path: Path,
) -> None:
    services, corpus = _services(tmp_path)
    (corpus / "a.txt").write_text("alpha apple", encoding="utf-8")
    services.build.run()
    (corpus / "b.txt").write_text("beta banana", encoding="utf-8")
    store = services.build.passage_writer
    assert isinstance(store, QdrantPassageStore)
    services.build.passage_writer = FailUpserts(store)  # type: ignore[assignment]  # test double

    with pytest.raises(StoreUnavailable):
        services.build.run()

    manifest = SQLiteManifest(tmp_path / "data")
    run = manifest.latest_run()
    assert run is not None and run.status == "aborted"
    assert store.staging_collections() == (run.staging_collection,)
    assert _hits(services, "apple") == ["a.txt"] and _hits(services, "banana") == []
    a_source = manifest.get_source("a.txt")
    assert a_source is not None and manifest.get_source("b.txt") is None
    assert store.read_completion_metadata() is not None
