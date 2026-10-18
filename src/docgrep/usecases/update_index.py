"""Bring the published index up to date with the corpus, in place (ADR-0013)."""

import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from docgrep.domain.errors import (
    BuildFailed,
    ConfigDrift,
    DocgrepError,
    IndexIncomplete,
    ResumeAvailable,
    ValidationError,
)
from docgrep.domain.models import FileType, ManifestSourceRecord, Source
from docgrep.domain.protocols import (
    Chunker,
    CorpusAvailability,
    CorpusReader,
    DocumentParser,
    EmbeddingProvider,
    IndexLock,
    LiveIndexWriter,
    Manifest,
    OcrDelegations,
    RootCatalog,
    SparseEncoder,
    SparseEncoders,
)
from docgrep.domain.roots import CorpusRoots, known_root_names
from docgrep.domain.run_state import (
    RESUMABLE_STATES,
    VECTOR_CONFIG_KEYS,
    config_drift,
    latest_build_run,
    run_state,
    unfinished_reencode,
)
from docgrep.domain.sparse import SparseWeighting
from docgrep.usecases.build_index import (
    BuildFailure,
    Cancelled,
    Prepared,
    Progress,
    SourceIndexing,
)
from docgrep.usecases.plan_update import (
    CHANGE_ORDER,
    ChangeKind,
    DetectedChanges,
    ModifiedReason,
    PendingRefresh,
    PlannedChange,
    PlanUpdate,
    UpdatePlan,
)

__all__ = [
    "ChangeKind",
    "ModifiedReason",
    "UpdateIndex",
    "UpdatePlan",
    "UpdateSummary",
]

_LOG = logging.getLogger(__name__)

_ORDER = CHANGE_ORDER


@dataclass(frozen=True, slots=True)
class UpdateSummary:
    """Final counts of one update.

    ``added``, ``modified``, ``removed`` and ``retried`` count the changes that
    were processed (fewer than planned when cancelled); ``indexed``, ``skipped``
    and ``failed`` are their outcomes, a failed removal included.
    """

    run_id: str
    plan: UpdatePlan
    added: int
    modified: int
    removed: int
    unchanged: int
    retried: int
    indexed: int
    skipped: int
    failed: int
    passages_written: int
    passages_removed: int
    elapsed_seconds: float
    failures: tuple[BuildFailure, ...]
    cancelled: bool = False

    @property
    def no_changes(self) -> bool:
        return self.plan.changes == 0


class UpdateIndex(SourceIndexing):
    """Apply the corpus changes since the last run to the published collection.

    One source at a time, insert first: a changed source has its new passages
    written before its stale ones are pruned, and its committed record follows
    only once its passages are stored. Not resumable: running it again
    continues the work.
    """

    def __init__(
        self,
        *,
        lock: IndexLock,
        corpus: CorpusReader,
        parsers: Mapping[FileType, DocumentParser],
        chunker: Chunker,
        embedder: EmbeddingProvider,
        sparse_encoder: SparseEncoder,
        index: LiveIndexWriter,
        manifest: Manifest,
        config: Mapping[str, str | int | float],
        batch_size: int = 32,
        upsert_batch_size: int = 256,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        progress_every_sources: int = 25,
        progress_every_seconds: float = 60.0,
        log_context: Mapping[str, str | int | float] | None = None,
        delegations: OcrDelegations | None = None,
        roots: RootCatalog | None = None,
        availability: CorpusAvailability | None = None,
        sparse_encoders: SparseEncoders | None = None,
    ) -> None:
        """With ``sparse_encoders`` new passages get the sparse weighting the
        index records (ADR-0025); without them, ``sparse_encoder``."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if upsert_batch_size <= 0:
            raise ValueError("upsert_batch_size must be positive")
        if progress_every_sources <= 0:
            raise ValueError("progress_every_sources must be positive")
        if progress_every_seconds <= 0:
            raise ValueError("progress_every_seconds must be positive")
        self.delegations = delegations
        self.roots = roots
        self.availability = availability
        self.lock = lock
        self.corpus = corpus
        self.parsers = parsers
        self.chunker = chunker
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        self.sparse_encoders = sparse_encoders
        self.index = index
        self.manifest = manifest
        self.config = dict(config)
        self.batch_size = batch_size
        self.upsert_batch_size = upsert_batch_size
        self.sleep = sleep
        self.clock = clock
        self.now = now
        self.progress_every_sources = progress_every_sources
        self.progress_every_seconds = progress_every_seconds
        self.log_context = dict(log_context or {})
        self._cancel_event = threading.Event()
        self._cancel_persisted = False

    @property
    def run_config(self) -> dict[str, str | int | float]:
        """The configuration recorded on a run, as for a build."""
        return {
            **self.config,
            "embedding_model": self.embedder.model_name,
            "embedding_dimension": self.embedder.dimension,
        }

    def run(
        self,
        *,
        retry_failed: bool = True,
        on_plan: Callable[[UpdatePlan], None] | None = None,
    ) -> UpdateSummary:
        """Update the index; raises before writing anything when it is refused.

        The changes are found first, without writing anything; ``on_plan``
        is then called with the work plan, before any change is applied.

        Refusals: ``IndexLockHeld``, ``ResumeAvailable`` when the latest build
        is unfinished, ``IndexIncomplete`` without a completed index and
        ``ConfigDrift`` when a setting that decides vectors or chunking differs
        from the index.
        """
        started = self.clock()
        with self.lock.hold("update"):
            metadata = self._check_preconditions()
            self._choose_sparse_encoder(metadata)
            roots = self.run_roots()
            try:
                detected = self.detect(retry_failed, roots)
            except DocgrepError:
                raise
            except Exception as error:
                raise BuildFailed(
                    "Index update aborted", details={"reason": str(error)}
                ) from error
            plan = detected.plan()
            if on_plan is not None:
                on_plan(plan)
            self._close_dead_update()
            run_id = self.manifest.start_run(
                {**self.run_config, **self.roots_config(roots)}, kind="update"
            )
            _LOG.info(
                "index_update_started", extra={"run_id": run_id, **self.log_context}
            )
            return self._execute(
                run_id, started, detected.changes, plan, detected.refreshes
            )

    def forget_plan(self, name: str) -> tuple[ManifestSourceRecord, ...]:
        """The committed sources of a root, which ``forget_root`` removes.

        Refuses the legacy root, whose sources have no prefix, and a name
        that is neither configured nor found in the index.
        """
        roots = self._catalog()
        records = self.manifest.committed_sources()
        if name == roots.legacy_name:
            raise ValidationError(
                f"root {name!r} is the legacy root: its sources have no prefix, so "
                "they cannot be told apart from the sources of another unprefixed "
                "root and are not removed this way. Remove the files and run "
                "`docgrep update`, or rebuild the index",
                details={"field": "name", "name": name, "reason": "legacy_root"},
            )
        known = known_root_names(roots, (r.source.source_id for r in records))
        if name not in known:
            raise ValidationError(
                f"unknown root {name!r}; known roots: {', '.join(known) or 'none'}",
                details={"field": "name", "name": name, "known": list(known)},
            )
        return tuple(r for r in records if roots.root_of(r.source.source_id) == name)

    def forget_root(
        self, name: str, *, on_plan: Callable[[UpdatePlan], None] | None = None
    ) -> UpdateSummary:
        """Remove every source of a prefixed root from the index, with the
        removal path of an update; the files are not touched."""
        started = self.clock()
        with self.lock.hold("update"):
            # Removing writes no vectors, so the vector settings may differ.
            self._check_preconditions(drift=False)
            removed = self.forget_plan(name)
            roots = self._catalog()
            self._close_dead_update()
            run_id = self.manifest.start_run(
                {
                    **self.run_config,
                    **self.roots_config(roots, roots.prefixed_names - {name}),
                },
                kind="update",
            )
            _LOG.info(
                "index_forget_root_started",
                extra={"run_id": run_id, "sources": len(removed)},
            )

            changes = tuple(
                PlannedChange("removed", r.source, r, name) for r in removed
            )
            plan = UpdatePlan(removed=len(changes))
            return self._execute(run_id, started, changes, plan, on_plan=on_plan)

    def _catalog(self) -> CorpusRoots:
        """The roots known to this process and to the index; none is read."""
        roots = self.known_roots()
        return CorpusRoots() if roots is None else roots

    def passages_of(self, records: Sequence[ManifestSourceRecord]) -> int:
        """How many passages the published index holds for these sources."""
        return sum(
            self.index.count_source_passages(record.source.source_id)
            for record in records
            if record.status == "indexed"
        )

    def _choose_sparse_encoder(self, metadata: Mapping[str, str | int | float]) -> None:
        """New passages are encoded with the weighting the index records, so
        one collection never mixes weightings."""
        self._run_sparse_encoder = None
        if self.sparse_encoders is None:
            return
        try:
            weighting = SparseWeighting.from_metadata(metadata)
        except ValueError as error:
            raise ConfigDrift(
                "The index records a sparse weighting this version cannot "
                "write; rebuild the index with `docgrep index`",
                details={"reason": str(error)},
            ) from error
        self._run_sparse_encoder = self.sparse_encoders.document_encoder(weighting)

    def _check_preconditions(
        self, *, drift: bool = True
    ) -> Mapping[str, str | int | float]:
        """Refuse an update that cannot run; returns the index metadata.

        Without ``drift`` (a removal, which writes no vectors) neither the
        vector settings nor an unfinished re-encode is checked.
        """
        build = latest_build_run(self.manifest)
        if build is not None:
            # The lock is held here, so a build stored as running is dead.
            state = run_state(build, lock_held=False)
            if state in RESUMABLE_STATES or state == "swapped":
                raise ResumeAvailable(
                    "The latest build is unfinished, so the index cannot be "
                    "updated: continue it with `docgrep index --resume` or "
                    "discard it with `docgrep index --restart`",
                    details={"run_id": build.run_id, "state": state},
                )
        reencode = unfinished_reencode(self.manifest, lock_held=False)
        if drift and reencode is not None:
            raise ResumeAvailable(
                "A re-encode of the sparse vectors is unfinished, so new "
                "passages could get either weighting: finish it with "
                "`docgrep reencode-sparse --yes` first",
                details={"run_id": reencode.run_id},
            )
        metadata = self.index.read_completion_metadata()
        if metadata is None:
            raise IndexIncomplete(
                "No completed index exists to update; build one with `docgrep index`"
            )
        drifted = config_drift(metadata, self.run_config) if drift else ()
        if drifted:
            raise ConfigDrift(
                "The configuration differs from the index in settings that decide "
                "vectors or chunking; an update cannot change them. Run a full "
                "rebuild with `docgrep index` (or restore the settings)",
                details={
                    "fields": ", ".join(drifted),
                    "checked": ", ".join(VECTOR_CONFIG_KEYS),
                },
            )
        return metadata

    def _close_dead_update(self) -> None:
        """Mark an update that lost its process as aborted; its work stays."""
        latest = self.manifest.latest_run()
        if latest is not None and latest.kind == "update":
            if latest.status == "running":
                self.manifest.finish_run(latest.run_id, "aborted", dict(latest.totals))
                _LOG.info("update_run_closed", extra={"run_id": latest.run_id})

    def detect(
        self, retry_failed: bool = True, roots: CorpusRoots | None = None
    ) -> DetectedChanges:
        """What an update with these roots would change; nothing is written."""
        return PlanUpdate(
            corpus=self.corpus,
            manifest=self.manifest,
            parsers=self.parsers,
            delegations=self.delegations,
        ).detect(retry_failed, roots)

    def _refresh(self, pending: PendingRefresh) -> None:
        """Make the recorded size and time follow an unchanged file, in the
        passages first and then in the record."""
        record, current = pending.record, pending.current
        if current.size is not None and current.mtime is not None:
            self.index.refresh_live_source(
                record.source.source_id, size=current.size, mtime=current.mtime
            )
        self.manifest.record_source(
            pending.refreshed, record.status, record.error, record.embedding_model
        )

    def _execute(
        self,
        run_id: str,
        started: float,
        changes: Sequence[PlannedChange],
        plan: UpdatePlan,
        refreshes: Sequence[PendingRefresh] = (),
        *,
        on_plan: Callable[[UpdatePlan], None] | None = None,
    ) -> UpdateSummary:
        """Apply a plan found beforehand: the refreshes first, then the
        changes in order."""
        counts: dict[str, int] = dict.fromkeys(_ORDER, 0)
        totals = {"indexed": 0, "skipped": 0, "failed": 0}
        written = 0
        removed = 0
        failures: list[BuildFailure] = []
        cancelled = False
        try:
            self.manifest.set_total_sources(run_id, len(changes))
            for pending in refreshes:
                self._refresh(pending)
            _LOG.info(
                "index_update_plan",
                extra={
                    "run_id": run_id,
                    "added": plan.added,
                    "modified": plan.modified,
                    "removed": plan.removed,
                    "unchanged": plan.unchanged,
                    "retried": plan.retried,
                    "refreshed": plan.refreshed,
                    "content_changed": plan.content_changed,
                    "parser_changed": plan.parser_changed,
                    "newly_supported": plan.newly_supported,
                    "delegated_to_ocr": plan.delegated_to_ocr,
                    "delegation_ended": plan.delegation_ended,
                },
            )
            if on_plan is not None:
                on_plan(plan)
            progress = Progress(
                run_id=run_id,
                total=len(changes),
                clock=self.clock,
                started=started,
                every_sources=self.progress_every_sources,
                every_seconds=self.progress_every_seconds,
            )
            for change in changes:
                self._check_cancel(run_id)
                outcome, failure, added_points, pruned = self._apply(change, run_id)
                counts[change.kind] += 1
                if outcome is not None:
                    totals[outcome] += 1
                written += added_points
                removed += pruned
                if failure is not None:
                    failures.append(failure)
                progress.advance(added_points, totals["failed"])
            if changes:
                progress.finish(totals["failed"])
            self._check_cancel(run_id)
            self.index.merge_completion_metadata(
                {"last_updated_at": self.now().isoformat()}
            )
            self.manifest.finish_run(
                run_id,
                "completed",
                self._run_totals(counts, totals, plan, written, removed),
            )
        except Cancelled:
            self.manifest.finish_run(
                run_id,
                "cancelled",
                self._run_totals(counts, totals, plan, written, removed),
            )
            cancelled = True
        except Exception as error:
            # Everything applied so far is consistent and stays; the next
            # update detects what is left.
            self.manifest.finish_run(
                run_id,
                "aborted",
                self._run_totals(counts, totals, plan, written, removed),
            )
            if isinstance(error, DocgrepError):
                raise
            raise BuildFailed(
                "Index update aborted", details={"reason": str(error)}
            ) from error
        finally:
            self._cancel_event.clear()
            self._cancel_persisted = False
        elapsed = self.clock() - started
        _LOG.info(
            "index_update_cancelled" if cancelled else "index_update_finished",
            extra={
                "run_id": run_id,
                **counts,
                **totals,
                "passages_written": written,
                "passages_removed": removed,
                "duration_ms": round(elapsed * 1000, 1),
            },
        )
        return UpdateSummary(
            run_id=run_id,
            plan=plan,
            added=counts["added"],
            modified=counts["modified"],
            removed=counts["removed"],
            unchanged=plan.unchanged,
            retried=counts["retried"],
            indexed=totals["indexed"],
            skipped=totals["skipped"],
            failed=totals["failed"],
            passages_written=written,
            passages_removed=removed,
            elapsed_seconds=elapsed,
            failures=tuple(failures),
            cancelled=cancelled,
        )

    @staticmethod
    def _run_totals(
        counts: Mapping[str, int],
        totals: Mapping[str, int],
        plan: UpdatePlan,
        written: int,
        removed: int,
    ) -> dict[str, int]:
        return {
            **counts,
            **totals,
            "unchanged": plan.unchanged,
            "passages_written": written,
            "passages_removed": removed,
        }

    def _apply(
        self, change: PlannedChange, run_id: str
    ) -> tuple[str | None, BuildFailure | None, int, int]:
        """Process one change.

        Returns the outcome (indexed, skipped or failed; None for a removal),
        the failure, and how many passages were written and removed.
        """
        if change.kind == "removed":
            return self._remove(change, run_id)
        prepared = self._prepare(change.source, run_id)
        if prepared.error is not None:
            reason = prepared.status.split(":", 1)[1]
            failure = self._record_failure(
                prepared.source, reason, prepared.error, run_id
            )
            # Whatever the record said before (added, indexed, skipped or
            # failed), it now says failed, as after a build: status counts the
            # source as failed and the next update retries it. Passages of a
            # previously indexed source are kept, so it stays searchable until
            # an update succeeds or the file is removed.
            self.manifest.record_source(
                prepared.source, prepared.status, str(prepared.error)
            )
            return "failed", failure, 0, 0
        try:
            pruned = self._write(prepared)
        except Exception as error:
            failure = self._record_failure(
                prepared.source, "store_error", error, run_id
            )
            if change.record is None:
                self._discard_partial(prepared.source)
                self.manifest.record_source(
                    prepared.source, "failed:store_error", str(error)
                )
            return "failed", failure, 0, 0
        indexed = prepared.status == "indexed"
        model = self.embedder.model_name if indexed else None
        # The passages are stored: only now does the committed record follow.
        self.manifest.record_source(
            prepared.source, prepared.status, prepared.detail, model
        )
        self.manifest.record_source(
            prepared.source,
            prepared.status,
            prepared.detail,
            model,
            run_id=run_id,
            passages=len(prepared.items),
        )
        return ("indexed" if indexed else "skipped"), None, len(prepared.items), pruned

    def _write(self, prepared: Prepared) -> int:
        """Insert the new passages, then prune the source's others."""
        items = prepared.items
        for start in range(0, len(items), self.upsert_batch_size):
            group = items[start : start + self.upsert_batch_size]
            self._retry(lambda: self.index.upsert_live_batch(group))
        keep = frozenset(item[0].passage_id for item in items)
        source_id = prepared.source.source_id
        return self._retry(
            lambda: self.index.delete_live_source_points_except(source_id, keep)
        )

    def _discard_partial(self, source: Source) -> None:
        try:
            self.index.delete_live_source_points(source.source_id)
        except Exception:
            _LOG.warning(
                "update_partial_source_kept", extra={"source_id": source.source_id}
            )

    def _remove(
        self, change: PlannedChange, run_id: str
    ) -> tuple[str | None, BuildFailure | None, int, int]:
        source = change.source
        try:
            removed = self._retry(
                lambda: self.index.delete_live_source_points(source.source_id)
            )
        except Exception as error:
            # The record stays, so the next update tries the removal again.
            failure = self._record_failure(source, "removal_failed", error, run_id)
            return "failed", failure, 0, 0
        self.manifest.remove_source(source.path)
        self.manifest.record_source(source, "removed", run_id=run_id)
        return None, None, 0, removed
