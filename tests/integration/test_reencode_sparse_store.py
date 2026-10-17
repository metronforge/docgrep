"""Re-encoding sparse vectors against real SQLite and local-mode Qdrant
(ADR-0025)."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from docgrep.adapters.filelock import FileIndexLock
from docgrep.adapters.manifest import SQLiteManifest
from docgrep.adapters.models.sparse import (
    BM25DocumentEncoder,
    SparseEncoderSet,
    TermFrequencyEncoder,
)
from docgrep.adapters.qdrant import QdrantPassageStore
from docgrep.domain.errors import IndexLockHeld
from docgrep.settings import Settings
from docgrep.usecases.reencode_sparse import ReencodeSparse
from resume_support import close, dump_points, new_services, settings_for, write_corpus
from sparse_support import as_tf_v1_build

SOURCES = 6


class StoppingStore(QdrantPassageStore):
    """Asks for a cancel of the latest run once ``pages`` pages are written."""

    manifest: SQLiteManifest
    pages = 0
    written = 0

    def write_sparse_vectors(
        self, vectors: Sequence[tuple[str, Mapping[int, float]]]
    ) -> None:
        super().write_sparse_vectors(vectors)
        self.written += 1
        if self.written == self.pages:
            run = self.manifest.latest_run()
            assert run is not None
            self.manifest.request_cancel(run.run_id)


def _tf_v1_index(tmp_path: Path) -> Settings:
    write_corpus(tmp_path / "corpus", SOURCES, words=37)
    settings = settings_for(tmp_path / "corpus", tmp_path / "data")
    services = new_services(settings)
    assert as_tf_v1_build(services.build).run().indexed == SOURCES
    close(services)
    return settings


def _store(
    settings: Settings, cls: type[QdrantPassageStore] = QdrantPassageStore
) -> Any:
    return cls(settings.data_dir / "qdrant", alias=settings.qdrant_alias)


def _use_case(settings: Settings, store: QdrantPassageStore) -> ReencodeSparse:
    return ReencodeSparse(
        lock=FileIndexLock(settings.data_dir / "index.lock"),
        index=store,
        manifest=SQLiteManifest(settings.data_dir),
        encoders=SparseEncoderSet(),
        page_size=4,
    )


def _without_sparse(points: list[Any]) -> list[Any]:
    return [
        (point_id, payload, vector["dense"]) for point_id, payload, vector in points
    ]


def _average(points: list[Any]) -> float:
    lengths = [SparseEncoderSet().document_length(p[1]["text"]) for p in points]
    return sum(lengths) / len(lengths)


def _assert_sparse(points: list[Any], encoder: Any) -> None:
    for _, payload, vector in points:
        expected = encoder.encode(payload["text"])
        stored = dict(
            zip(vector["sparse"].indices, vector["sparse"].values, strict=True)
        )
        assert stored == pytest.approx(expected)


def test_only_the_sparse_vectors_change(tmp_path: Path) -> None:
    settings = _tf_v1_index(tmp_path)
    before = dump_points(settings)
    _assert_sparse(before, TermFrequencyEncoder())

    store = _store(settings)
    try:
        summary = _use_case(settings, store).run()
        metadata = store.read_completion_metadata()
    finally:
        store.close()

    after = dump_points(settings)
    assert summary.written == len(before) > 8
    # Dense vectors and payloads, bit for bit.
    assert _without_sparse(after) == _without_sparse(before)
    avgdl = _average(before)
    _assert_sparse(after, BM25DocumentEncoder(1.2, 0.75, avgdl))
    assert metadata is not None
    assert metadata["sparse_encoder"] == "bm25-v2"
    assert metadata["bm25_avgdl"] == pytest.approx(avgdl)
    assert (metadata["bm25_k1"], metadata["bm25_b"]) == (1.2, 0.75)


def test_a_cancelled_run_resumes_in_a_new_process_from_its_checkpoint(
    tmp_path: Path,
) -> None:
    settings = _tf_v1_index(tmp_path)
    before = dump_points(settings)
    store = _store(settings, StoppingStore)
    store.manifest = SQLiteManifest(settings.data_dir)
    store.pages = 2
    try:
        first = _use_case(settings, store).run()
    finally:
        store.close()

    assert first.cancelled and first.written == 8
    run = SQLiteManifest(settings.data_dir).latest_run()
    assert run is not None and run.status == "cancelled"
    assert run.kind == "reencode_sparse"
    assert json.loads(run.checkpoint or "")["done"] == 8

    store = _store(settings)
    try:
        second = _use_case(settings, store).run()
    finally:
        store.close()

    assert second.plan.resumes == run.run_id
    assert second.written == len(before) - 8
    after = dump_points(settings)
    assert _without_sparse(after) == _without_sparse(before)
    _assert_sparse(after, BM25DocumentEncoder(1.2, 0.75, _average(before)))


def test_another_index_operation_holds_the_lock(tmp_path: Path) -> None:
    settings = _tf_v1_index(tmp_path)
    before = dump_points(settings)
    store = _store(settings)
    try:
        with FileIndexLock(settings.data_dir / "index.lock").hold("update"):
            with pytest.raises(IndexLockHeld):
                _use_case(settings, store).run()
    finally:
        store.close()

    assert dump_points(settings) == before
    run = SQLiteManifest(settings.data_dir).latest_run()
    assert run is not None and run.kind == "build"


def test_search_works_after_the_re_encode(tmp_path: Path) -> None:
    settings = _tf_v1_index(tmp_path)
    store = _store(settings)
    try:
        _use_case(settings, store).run()
    finally:
        store.close()

    services = new_services(settings)
    try:
        result = services.search.run("marker3", top_n=3)
    finally:
        close(services)

    assert result.results[0].source.path == "doc003.txt"
