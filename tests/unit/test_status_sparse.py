"""What status reports about the sparse weighting and a re-encode (ADR-0025)."""

import json
from datetime import UTC, datetime
from typing import Any

from docgrep.api.cli import _format_status
from docgrep.domain.models import StagingInfo
from docgrep.usecases.inspect_status import InspectStatus
from fakes import InMemoryManifest

META: dict[str, str | int | float] = {
    "embedding_model": "m",
    "embedding_dimension": 4,
    "completed_at": "2026-10-01T00:00:00+00:00",
}
BM25 = {
    "sparse_encoder": "bm25-v2",
    "bm25_k1": 1.2,
    "bm25_b": 0.75,
    "bm25_avgdl": 231.5,
}


class Lock:
    def __init__(self, held: bool) -> None:
        self.held = held

    def is_held(self) -> bool:
        return self.held


class Store:
    def __init__(self, metadata: dict[str, str | int | float] | None) -> None:
        self.metadata = metadata

    def read_completion_metadata(self) -> dict[str, str | int | float] | None:
        return self.metadata

    def count_passages(self) -> int:
        return 1000

    def staging_info(self, name: str) -> StagingInfo | None:
        return None

    def close(self) -> None:
        pass


def _report(
    metadata: dict[str, str | int | float] | None,
    manifest: InMemoryManifest,
    *,
    held: bool = False,
) -> dict[str, Any]:
    status = InspectStatus(
        lock=Lock(held),
        open_manifest=lambda: manifest,
        open_store=lambda: Store(metadata),  # type: ignore[arg-type, return-value]
        store_mode="server",
        runtime_config={},
        now=lambda: datetime(2026, 10, 17, 12, 0, 10, tzinfo=UTC),
    )
    return status.run().as_dict()


def _reencode(manifest: InMemoryManifest, done: int) -> str:
    run_id = manifest.start_run(BM25, kind="reencode_sparse")
    manifest._update_run(run_id, started_at="2026-10-17T12:00:00+00:00")
    manifest.set_total_sources(run_id, 1000)
    manifest.set_checkpoint(
        run_id, json.dumps({"offset": "x", "done": done, "scanned": False})
    )
    return run_id


def test_a_tf_v1_index_gets_a_hint_and_no_warning() -> None:
    report = _report(META, InMemoryManifest())

    sparse = report["sparse"]
    assert sparse["encoder"] == "tf-v1"
    assert (sparse["bm25_k1"], sparse["bm25_b"], sparse["bm25_avgdl"]) == (
        None,
        None,
        None,
    )
    assert sparse["reencode"] is None
    assert "docgrep reencode-sparse" in sparse["hint"]
    assert report["warnings"] == []
    text = _format_status(report)
    assert "Sparse encoder: tf-v1\n  hint: " in text


def test_a_bm25_index_shows_its_parameters() -> None:
    report = _report({**META, **BM25}, InMemoryManifest())

    assert report["sparse"] == {
        "encoder": "bm25-v2",
        "bm25_k1": 1.2,
        "bm25_b": 0.75,
        "bm25_avgdl": 231.5,
        "reencode": None,
        "hint": None,
    }
    assert "Sparse encoder: bm25-v2 (k1 1.2, b 0.75, avgdl 231.50)" in (
        _format_status(report)
    )


def test_a_running_re_encode_shows_its_progress() -> None:
    manifest = InMemoryManifest()
    run_id = _reencode(manifest, 250)

    report = _report(META, manifest, held=True)

    assert report["sparse"]["reencode"] == {
        "run_id": run_id,
        "state": "running",
        "passages_done": 250,
        "passages_total": 1000,
    }
    assert report["sparse"]["hint"] is None
    active = report["active_operation"]
    assert (active["kind"], active["sources_done"], active["sources_total"]) == (
        "reencode_sparse",
        250,
        1000,
    )
    assert active["passages_written"] == 250
    # 250 passages in 10 seconds: 750 more take 30.
    assert active["eta_seconds"] == 30
    text = _format_status(report)
    assert "Sparse encoder: reencoding (250 of 1000)" in text
    assert "Active operation: reencode_sparse (run 1), 250/1000 passages" in text
    assert "Latest run: 1 (reencode_sparse) running" in text
    assert report["warnings"] == []


def test_an_interrupted_re_encode_is_a_warning_with_a_hint() -> None:
    manifest = InMemoryManifest()
    run_id = _reencode(manifest, 400)

    report = _report(META, manifest, held=False)

    reencode = report["sparse"]["reencode"]
    assert (reencode["run_id"], reencode["state"]) == (run_id, "interrupted")
    assert "reencode-sparse --yes" in report["sparse"]["hint"]
    assert [w["code"] for w in report["warnings"]] == ["reencode_unfinished"]
    assert report["index_state"] == "ready"


def test_a_completed_re_encode_is_not_reported_as_running() -> None:
    manifest = InMemoryManifest()
    run_id = _reencode(manifest, 1000)
    manifest.finish_run(run_id, "completed", {"passages": 1000})

    report = _report({**META, **BM25}, manifest)

    assert report["sparse"]["reencode"] is None
    assert report["latest_run"]["kind"] == "reencode_sparse"


def test_without_an_index_there_is_no_sparse_section() -> None:
    assert _report(None, InMemoryManifest())["sparse"] is None
