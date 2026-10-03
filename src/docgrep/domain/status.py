"""The report of what the index and the last operations look like right now."""

from dataclasses import dataclass
from typing import Any

IndexState = str  # one of INDEX_STATES


INDEX_STATES = (
    "not_built",
    "ready",
    "aborted_pending_resume",
    "cancelled_pending_resume",
    "interrupted_pending_resume",
    "swap_pending_commit",
    "manifest_unreadable",
)

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class RunView:
    run_id: str
    status: str
    state: str
    started_at: str | None
    finished_at: str | None
    totals: tuple[tuple[str, int], ...]
    total_sources: int | None
    resumed_from: str | None
    carried_sources: int


@dataclass(frozen=True, slots=True)
class ActiveOperation:
    kind: str
    run_id: str
    started_at: str | None
    sources_done: int
    sources_total: int | None
    passages_written: int
    elapsed_seconds: float | None
    eta_seconds: float | None
    cancel_requested_at: str | None


@dataclass(frozen=True, slots=True)
class StoreView:
    """``status`` is reachable, unreachable, busy or absent (no local store yet)."""

    mode: str
    status: str
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class CollectionView:
    passage_count: int | None
    embedding_model: str | None
    embedding_dimension: int | None
    embedding_dtype: str | None
    docgrep_version: str | None
    completed_at: str | None


@dataclass(frozen=True, slots=True)
class DriftView:
    field: str
    indexed: str | int | float
    runtime: str | int | float


@dataclass(frozen=True, slots=True)
class StatusWarning:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class StatusReport:
    index_state: IndexState
    latest_run: RunView | None
    active_operation: ActiveOperation | None
    store: StoreView
    collection: CollectionView | None
    drift: tuple[DriftView, ...]
    warnings: tuple[StatusWarning, ...]

    def as_dict(self) -> dict[str, Any]:
        """The JSON form; its key names are stable and documented in the README."""
        run = self.latest_run
        active = self.active_operation
        collection = self.collection
        return {
            "schema_version": SCHEMA_VERSION,
            "index_state": self.index_state,
            "latest_run": None
            if run is None
            else {
                "run_id": run.run_id,
                "status": run.status,
                "state": run.state,
                "started_at": run.started_at,
                "finished_at": run.finished_at,
                "totals": dict(run.totals),
                "total_sources": run.total_sources,
                "resumed_from": run.resumed_from,
                "carried_sources": run.carried_sources,
            },
            "active_operation": None
            if active is None
            else {
                "kind": active.kind,
                "run_id": active.run_id,
                "started_at": active.started_at,
                "sources_done": active.sources_done,
                "sources_total": active.sources_total,
                "passages_written": active.passages_written,
                "elapsed_seconds": active.elapsed_seconds,
                "eta_seconds": active.eta_seconds,
                "cancel_requested_at": active.cancel_requested_at,
            },
            "vector_store": {
                "mode": self.store.mode,
                "status": self.store.status,
                "reason": self.store.reason,
            },
            "collection": None
            if collection is None
            else {
                "passage_count": collection.passage_count,
                "embedding_model": collection.embedding_model,
                "embedding_dimension": collection.embedding_dimension,
                "embedding_dtype": collection.embedding_dtype,
                "docgrep_version": collection.docgrep_version,
                "completed_at": collection.completed_at,
            },
            "drift": [
                {"field": d.field, "indexed": d.indexed, "runtime": d.runtime}
                for d in self.drift
            ],
            "warnings": [{"code": w.code, "message": w.message} for w in self.warnings],
        }
