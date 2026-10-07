"""Report the state of the index and of long operations without changing anything."""

import logging
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import PurePosixPath

from docgrep.domain.errors import DocgrepError
from docgrep.domain.models import GARBLED_TEXT_LAYER, FileType, ManifestRunRecord
from docgrep.domain.progress import estimate_eta_seconds
from docgrep.domain.protocols import (
    IndexLock,
    ManifestReader,
    StatusStore,
    parser_reprocess_reason,
)
from docgrep.domain.run_state import (
    VECTOR_CONFIG_KEYS,
    latest_build_run,
    run_state,
    swap_already_happened,
)
from docgrep.domain.status import (
    FAILED_SOURCES_LIMIT,
    SKIP_REASONS,
    SKIPPED_EXTENSIONS_LIMIT,
    ActiveOperation,
    CollectionView,
    DriftView,
    FailedSourceView,
    RunView,
    SourcesView,
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
    return _shorten(str(reason or error))


def _shorten(value: str) -> str:
    text = " ".join(value.split())
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
        parser_versions: Mapping[FileType, str | None] | None = None,
    ) -> None:
        """``parser_versions`` maps each file type with a registered parser to
        the version it declares; without it the parser-version count is None."""
        self.lock = lock
        self.open_manifest = open_manifest
        self.open_store = open_store
        self.store_mode = store_mode
        self.runtime_config = dict(runtime_config)
        self.now = now
        self.parser_versions = (
            None if parser_versions is None else dict(parser_versions)
        )

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
        # Only builds publish, swap or resume; an update never changes the
        # index state (ADR-0013).
        build: ManifestRunRecord | None = None
        try:
            manifest = self.open_manifest()
            latest = None if manifest is None else manifest.latest_run()
            if manifest is not None and latest is not None:
                build = latest if latest.kind == "build" else latest_build_run(manifest)
        except Exception as error:
            manifest_ok = False
            warn("manifest_unreadable", f"The manifest cannot be read: {_short(error)}")

        staging = None if build is None else build.staging_collection
        store_view, metadata, passages, published = self._read_store(staging)
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
                last_updated_at=_text(metadata, "last_updated_at"),
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
        if build is not None:
            state = run_state(build, lock_held=lock_held)
            if (
                state in ("interrupted", "aborted")
                and manifest is not None
                and published
                and swap_already_happened(
                    build,
                    journal_rows=self._journal_rows(manifest, build),
                    staging_published=True,
                )
            ):
                # It died between the alias swap and the swapped status; the
                # next build or resume finishes it. Only reported here.
                state = "swapped"
        if latest is not None:
            run_view = RunView(
                latest.run_id,
                latest.status,
                state
                if latest is build and state is not None
                else run_state(latest, lock_held=lock_held),
                latest.started_at,
                latest.finished_at,
                latest.totals,
                latest.total_sources,
                latest.resumed_from,
                latest.carried_sources,
                latest.kind,
            )
            if latest.kind == "update" and run_view.state in (
                "aborted",
                "cancelled",
                "interrupted",
            ):
                warn(
                    "update_unfinished",
                    "The latest update did not finish; what it applied stays "
                    "live, and `docgrep update` continues the work",
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

        index_state = self._index_state(manifest_ok, build, state, metadata)
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
        sources = None
        if manifest_ok and manifest is not None:
            sources = self._sources(manifest)
        return StatusReport(
            index_state,
            run_view,
            active,
            store_view,
            collection,
            drift,
            tuple(warnings),
            sources,
        )

    def _sources(self, manifest: ManifestReader) -> SourcesView | None:
        """Summarise the committed sources; None when they cannot be read."""
        try:
            records = manifest.committed_sources()
        except Exception:
            _LOG.debug("status_sources_unreadable", exc_info=True)
            return None
        states = {"indexed": 0, "skipped": 0, "failed": 0}
        reasons = dict.fromkeys((*SKIP_REASONS, "other"), 0)
        extensions: Counter[str] = Counter()
        failed: list[FailedSourceView] = []
        versions = self.parser_versions
        changed = 0
        for record in records:
            state, _, detail = record.status.partition(":")
            if state in states:
                states[state] += 1
            if state == "skipped":
                # A garbled text layer is stored as requires_ocr with its
                # reason code in the error column.
                if detail == "requires_ocr" and record.error == GARBLED_TEXT_LAYER:
                    detail = GARBLED_TEXT_LAYER
                reasons[detail if detail in SKIP_REASONS else "other"] += 1
                suffix = PurePosixPath(record.source.path).suffix.lower()
                extensions[suffix or "(none)"] += 1
            elif state == "failed" and len(failed) < FAILED_SOURCES_LIMIT:
                failed.append(
                    FailedSourceView(
                        record.source.path,
                        detail,
                        None if record.error is None else _shorten(record.error),
                    )
                )
            if versions is not None:
                file_type = record.source.file_type
                reason = parser_reprocess_reason(
                    record.status,
                    record.source.parser_version,
                    registered=file_type in versions,
                    declared_version=versions.get(file_type),
                )
                changed += reason == "parser_changed"
        top = sorted(extensions.items(), key=lambda item: (-item[1], item[0]))
        return SourcesView(
            total=len(records),
            by_state=tuple(states.items()),
            skipped_by_reason=tuple(reasons.items()),
            skipped_by_extension=tuple(top[:SKIPPED_EXTENSIONS_LIMIT]),
            failed=tuple(failed),
            parser_changed=None if versions is None else changed,
        )

    def _read_store(
        self, staging: str | None
    ) -> tuple[StoreView, Mapping[str, str | int | float] | None, int | None, bool]:
        """Read the store; also whether ``staging`` is already the published index."""
        mode = self.store_mode
        try:
            store = self.open_store()
        except Exception as error:
            busy = isinstance(error, DocgrepError) and error.details.get("busy") is True
            return (
                StoreView(mode, "busy" if busy else "unreachable", _short(error)),
                None,
                None,
                False,
            )
        if store is None:
            view = StoreView(mode, "absent", "No local store exists yet")
            return view, None, None, False
        try:
            metadata = store.read_completion_metadata()
            passages = store.count_passages()
            info = None if staging is None else store.staging_info(staging)
            published = info is not None and info.published
            return StoreView(mode, "reachable"), metadata, passages, published
        except Exception as error:
            view = StoreView(mode, "unreachable", _short(error))
            return view, None, None, False
        finally:
            try:
                store.close()
            except Exception:
                _LOG.debug("status_store_close_failed", exc_info=True)

    @staticmethod
    def _journal_rows(manifest: ManifestReader, run: ManifestRunRecord) -> int:
        try:
            return len(manifest.run_sources(run.run_id))
        except Exception:
            return -1

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
            kind=run.kind
            if run.kind in ("update", "ocr")
            else ("resume" if run.resumed_from else "build"),
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
