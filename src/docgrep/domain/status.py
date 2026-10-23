"""The report of what the index and the last operations look like right now."""

from dataclasses import dataclass
from typing import Any

from docgrep.domain.sync import SyncStatus

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
    kind: str = "build"


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
    last_updated_at: str | None = None


@dataclass(frozen=True, slots=True)
class ReencodeView:
    """The latest sparse re-encode while it runs or after it stopped early;
    ``passages_done`` of ``passages_total`` are written."""

    run_id: str
    state: str
    passages_done: int
    passages_total: int | None


@dataclass(frozen=True, slots=True)
class SparseView:
    """The sparse weighting the index records (ADR-0025); ``encoder`` is the
    recorded name, ``tf-v1`` when none is recorded. ``hint`` says what to run
    to change it, when there is something to do."""

    encoder: str
    k1: float | None = None
    b: float | None = None
    avgdl: float | None = None
    reencode: ReencodeView | None = None
    hint: str | None = None


@dataclass(frozen=True, slots=True)
class DriftView:
    field: str
    indexed: str | int | float
    runtime: str | int | float


@dataclass(frozen=True, slots=True)
class StatusWarning:
    code: str
    message: str


# The skip reasons reported by name; any other reason counts as "other".
SKIP_REASONS = ("requires_ocr", "garbled_text_layer", "unsupported")
# Reported by name too, but only when a source has it (keeps older reports).
OPTIONAL_SKIP_REASONS = ("delegated_to_ocr",)
# At most this many extensions and failed sources are listed.
SKIPPED_EXTENSIONS_LIMIT = 10
FAILED_SOURCES_LIMIT = 20


@dataclass(frozen=True, slots=True)
class FailedSourceView:
    path: str
    category: str
    reason: str | None


@dataclass(frozen=True, slots=True)
class SourcesView:
    """What the committed sources table says; counts, paths and reasons only.

    ``parser_changed`` is the number of sources the next update would extract
    again because their parser's version changed, or None when the registered
    parsers are not known.
    """

    total: int
    by_state: tuple[tuple[str, int], ...]
    skipped_by_reason: tuple[tuple[str, int], ...]
    skipped_by_extension: tuple[tuple[str, int], ...]
    failed: tuple[FailedSourceView, ...]
    parser_changed: int | None


@dataclass(frozen=True, slots=True)
class RootStatusView:
    """A corpus root and its committed sources by state (ADR-0023).

    ``configured`` is false for a root found only in the index; such a root
    has no folder here, so ``readable`` is None.
    """

    name: str
    legacy: bool
    configured: bool
    readable: bool | None
    indexed: int = 0
    skipped: int = 0
    failed: int = 0


@dataclass(frozen=True, slots=True)
class StatusReport:
    index_state: IndexState
    latest_run: RunView | None
    active_operation: ActiveOperation | None
    store: StoreView
    collection: CollectionView | None
    drift: tuple[DriftView, ...]
    warnings: tuple[StatusWarning, ...]
    sources: SourcesView | None = None
    # None when the roots are not known to the report (ADR-0023).
    roots: tuple[RootStatusView, ...] | None = None
    # None without a completed index (ADR-0025).
    sparse: SparseView | None = None
    # Automatic sync (ADR-0029); None when the report was made without it.
    sync: SyncStatus | None = None

    def as_dict(self, *, sync_paths: bool = False) -> dict[str, Any]:
        """The JSON form; its key names are stable and documented in the README.

        ``sync_paths`` adds a few stored paths to each root with held
        removals (``docgrep status`` only; never to a tool's client).
        """
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
                "kind": run.kind,
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
                "last_updated_at": collection.last_updated_at,
            },
            "drift": [
                {"field": d.field, "indexed": d.indexed, "runtime": d.runtime}
                for d in self.drift
            ],
            "warnings": [{"code": w.code, "message": w.message} for w in self.warnings],
            "sources": None if self.sources is None else _sources(self.sources),
            "sparse": None if self.sparse is None else _sparse(self.sparse),
            **({} if self.roots is None else {"roots": _roots(self.roots)}),
            **(
                {}
                if self.sync is None
                else {"sync": self.sync.to_dict(paths=sync_paths)}
            ),
        }


def _sparse(view: SparseView) -> dict[str, Any]:
    reencode = view.reencode
    return {
        "encoder": view.encoder,
        "bm25_k1": view.k1,
        "bm25_b": view.b,
        "bm25_avgdl": view.avgdl,
        "reencode": None
        if reencode is None
        else {
            "run_id": reencode.run_id,
            "state": reencode.state,
            "passages_done": reencode.passages_done,
            "passages_total": reencode.passages_total,
        },
        "hint": view.hint,
    }


def _roots(roots: tuple[RootStatusView, ...]) -> list[dict[str, Any]]:
    return [
        {
            "name": root.name,
            "legacy": root.legacy,
            "configured": root.configured,
            "readable": root.readable,
            "indexed": root.indexed,
            "skipped": root.skipped,
            "failed": root.failed,
        }
        for root in roots
    ]


def _sources(view: SourcesView) -> dict[str, Any]:
    return {
        "total": view.total,
        "by_state": dict(view.by_state),
        "skipped_by_reason": dict(view.skipped_by_reason),
        "skipped_by_extension": [
            {"extension": extension, "sources": count}
            for extension, count in view.skipped_by_extension
        ],
        "failed": [
            {"path": f.path, "category": f.category, "reason": f.reason}
            for f in view.failed
        ],
        "parser_changed": view.parser_changed,
    }
