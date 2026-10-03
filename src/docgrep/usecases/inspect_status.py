"""Report the state of the index and of long operations without changing anything."""

import logging
from collections.abc import Callable, Mapping
from datetime import UTC, datetime

from docgrep.domain.errors import DocgrepError
from docgrep.domain.models import ManifestRunRecord
from docgrep.domain.progress import estimate_eta_seconds
from docgrep.domain.protocols import IndexLock, ManifestReader, StatusStore
from docgrep.domain.run_state import VECTOR_CONFIG_KEYS, run_state
from docgrep.domain.status import (
    ActiveOperation,
    CollectionView,
    DriftView,
    RunView,
    StatusReport,
    StatusWarning,
    StoreView,
)

_LOG = logging.getLogger(__name__)
_REASON_LIMIT = 200

_PENDING = {
    "aborted": "aborted_pending_resume",
    "cancelled": "cancelled_pending_resume",
    "interrupted": "interrupted_pending_resume",
    "swapped": "swap_pending_commit",
}


def _short(error: Exception) -> str:
    reason = error.details.get("reason") if isinstance(error, DocgrepError) else None
    text = " ".join(str(reason or error).split())
    return text if len(text) <= _REASON_LIMIT else text[: _REASON_LIMIT - 3] + "..."


def _parse(stamp: str | None) -> datetime | None:
    if stamp is None:
        return None
    try:
        return datetime.fromisoformat(stamp)
    except ValueError:
        return None


class InspectStatus:
    """Read the lock, the manifest and the store; degrade instead of raising.

    The openers return None when nothing exists yet, so that a status of an
    unused data directory creates nothing. Neither the manifest nor the store
    is written, and the lock is only probed.
    """

    def __init__(
        self,
        *,
        lock: IndexLock,
        open_manifest: Callable[[], ManifestReader | None],
        open_store: Callable[[], StatusStore | None],
        store_mode: str,
        runtime_config: Mapping[str, str | int | float],
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.lock = lock
        self.open_manifest = open_manifest
        self.open_store = open_store
        self.store_mode = store_mode
        self.runtime_config = dict(runtime_config)
        self.now = now

    def run(self) -> StatusReport:
        warnings: list[StatusWarning] = []

        def warn(code: str, message: str) -> None:
            warnings.append(StatusWarning(code, message))

        try:
            lock_held = self.lock.is_held()
        except Exception as error:
            lock_held = False
            warn(
                "lock_unreadable",
                f"The index lock could not be probed: {_short(error)}",
            )

        manifest_ok = True
        manifest: ManifestReader | None = None
        latest: ManifestRunRecord | None = None
        try:
            manifest = self.open_manifest()
            latest = None if manifest is None else manifest.latest_run()
        except Exception as error:
            manifest_ok = False
            warn("manifest_unreadable", f"The manifest cannot be read: {_short(error)}")

        store_view, metadata, passages = self._read_store()
        if store_view.status == "unreachable":
            warn(
                "store_unreachable",
                store_view.reason or "The vector store is unreachable",
            )
        elif store_view.status == "busy":
            warn(
                "store_busy",
                "Another process holds the local vector store; "
                "its contents cannot be read right now",
            )

        collection = None
        if metadata is not None or passages:
            collection = CollectionView(
                passage_count=passages,
                embedding_model=_text(metadata, "embedding_model"),
                embedding_dimension=_number(metadata, "embedding_dimension"),
                embedding_dtype=_text(metadata, "embedding_dtype"),
                docgrep_version=_text(metadata, "docgrep_version"),
                completed_at=_text(metadata, "completed_at"),
            )
        drift = self._drift(metadata)
        if drift:
            warn(
                "config_drift",
                "The runtime configuration differs from the index in: "
                + ", ".join(d.field for d in drift),
            )

        run_view: RunView | None = None
        state = None
        if latest is not None:
            state = run_state(latest, lock_held=lock_held)
            run_view = RunView(
                latest.run_id,
                latest.status,
                state,
                latest.started_at,
                latest.finished_at,
                latest.totals,
                latest.total_sources,
                latest.resumed_from,
                latest.carried_sources,
            )

        active = None
        if lock_held:
            if latest is not None and latest.status in ("running", "swapped"):
                active = self._active(manifest, latest)
            else:
                warn(
                    "lock_held_without_run",
                    "The index lock is held, but no run is recorded as running",
                )

        index_state = self._index_state(manifest_ok, latest, state, metadata)
        if state in _PENDING:
            if state == "swapped":
                warn(
                    "swap_pending_commit",
                    "A run published its index but did not commit its sources; "
                    "the next build or `docgrep index --resume` finishes it",
                )
            else:
                warn(
                    "resume_available",
                    "The latest run is unfinished; use `docgrep index --resume` "
                    "to continue it or `--restart` to discard it",
                )
        if (
            state == "completed"
            and store_view.status == "reachable"
            and metadata is None
        ):
            warn(
                "index_incomplete",
                "The latest run completed, but the store holds no completed index",
            )
        return StatusReport(
            index_state,
            run_view,
            active,
            store_view,
            collection,
            drift,
            tuple(warnings),
        )

    def _read_store(
        self,
    ) -> tuple[StoreView, Mapping[str, str | int | float] | None, int | None]:
        mode = self.store_mode
        try:
            store = self.open_store()
        except Exception as error:
            busy = isinstance(error, DocgrepError) and error.details.get("busy") is True
            return (
                StoreView(mode, "busy" if busy else "unreachable", _short(error)),
                None,
                None,
            )
        if store is None:
            return StoreView(mode, "absent", "No local store exists yet"), None, None
        try:
            return (
                StoreView(mode, "reachable"),
                store.read_completion_metadata(),
                store.count_passages(),
            )
        except Exception as error:
            return StoreView(mode, "unreachable", _short(error)), None, None
        finally:
            try:
                store.close()
            except Exception:
                _LOG.debug("status_store_close_failed", exc_info=True)

    def _drift(
        self, metadata: Mapping[str, str | int | float] | None
    ) -> tuple[DriftView, ...]:
        if metadata is None:
            return ()
        return tuple(
            DriftView(key, metadata[key], self.runtime_config[key])
            for key in VECTOR_CONFIG_KEYS
            if key in metadata
            and key in self.runtime_config
            and metadata[key] != self.runtime_config[key]
        )

    def _index_state(
        self,
        manifest_ok: bool,
        latest: ManifestRunRecord | None,
        state: str | None,
        metadata: Mapping[str, str | int | float] | None,
    ) -> str:
        if not manifest_ok:
            return "manifest_unreadable"
        if state in _PENDING:
            return _PENDING[state]
        if latest is None or state == "running":
            return "ready" if metadata is not None else "not_built"
        return "ready"

    def _active(
        self, manifest: ManifestReader | None, run: ManifestRunRecord
    ) -> ActiveOperation | None:
        try:
            journal = [] if manifest is None else manifest.run_sources(run.run_id)
        except Exception:
            journal = []
        done = len(journal)
        started = _parse(run.started_at)
        elapsed = (
            None
            if started is None
            else max((self.now() - started).total_seconds(), 0.0)
        )
        eta = (
            None
            if elapsed is None or run.total_sources is None
            else estimate_eta_seconds(
                done=done,
                total=run.total_sources,
                elapsed_seconds=elapsed,
                carried=run.carried_sources,
            )
        )
        return ActiveOperation(
            kind="resume" if run.resumed_from else "build",
            run_id=run.run_id,
            started_at=run.started_at,
            sources_done=done,
            sources_total=run.total_sources,
            passages_written=sum(r.passages or 0 for r in journal),
            elapsed_seconds=None if elapsed is None else round(elapsed, 1),
            eta_seconds=None if eta is None else round(eta),
            cancel_requested_at=run.cancel_requested_at,
        )


def _text(metadata: Mapping[str, str | int | float] | None, key: str) -> str | None:
    value = None if metadata is None else metadata.get(key)
    return None if value is None else str(value)


def _number(metadata: Mapping[str, str | int | float] | None, key: str) -> int | None:
    value = None if metadata is None else metadata.get(key)
    return value if isinstance(value, int) else None
