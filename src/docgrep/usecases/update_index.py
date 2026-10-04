"""Bring the published index up to date with the corpus, in place (ADR-0013)."""

import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Literal

from docgrep.domain.errors import (
    BuildFailed,
    ConfigDrift,
    DocgrepError,
    IndexIncomplete,
    ResumeAvailable,
)
from docgrep.domain.models import FileType, ManifestSourceRecord, Source
from docgrep.domain.protocols import (
    Chunker,
    CorpusReader,
    DocumentParser,
    EmbeddingProvider,
    IndexLock,
    LiveIndexWriter,
    Manifest,
    SparseEncoder,
    parser_reprocess_reason,
    parser_version,
)
from docgrep.domain.run_state import (
    RESUMABLE_STATES,
    VECTOR_CONFIG_KEYS,
    config_drift,
    latest_build_run,
    run_state,
)
from docgrep.usecases.build_index import (
    BuildFailure,
    Cancelled,
    Prepared,
    Progress,
    SourceIndexing,
)

_LOG = logging.getLogger(__name__)

ChangeKind = Literal["removed", "modified", "retried", "added"]
# Why a recorded source is processed again (the retried ones aside).
ModifiedReason = Literal["content_changed", "parser_changed", "newly_supported"]
_ORDER: tuple[ChangeKind, ...] = ("removed", "modified", "retried", "added")


@dataclass(frozen=True, slots=True)
class UpdatePlan:
    """What an update found to do, before doing any of it.

    ``retried`` counts sources recorded as failed whose content did not change;
    a failed source whose content changed counts as modified. ``refreshed``
    counts unchanged sources whose size or modification time changed while
    their content hash did not; only their recorded size and time are updated.

    ``modified`` is the sum of the reasons: ``content_changed`` (the content
    hash differs), ``parser_changed`` (the registered parser's version differs
    from the recorded one, or the parser is gone) and ``newly_supported``
    (skipped as unsupported before, a parser exists now).
    """

    added: int = 0
    modified: int = 0
    removed: int = 0
    unchanged: int = 0
    retried: int = 0
    refreshed: int = 0
    content_changed: int = 0
    parser_changed: int = 0
    newly_supported: int = 0

    @property
    def changes(self) -> int:
        return self.added + self.modified + self.removed + self.retried


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


@dataclass(frozen=True, slots=True)
class _Change:
    kind: ChangeKind
    # The current file for added, modified and retried; the record for removed.
    source: Source
    record: ManifestSourceRecord | None


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
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if upsert_batch_size <= 0:
            raise ValueError("upsert_batch_size must be positive")
        if progress_every_sources <= 0:
            raise ValueError("progress_every_sources must be positive")
        if progress_every_seconds <= 0:
            raise ValueError("progress_every_seconds must be positive")
        self.lock = lock
        self.corpus = corpus
        self.parsers = parsers
        self.chunker = chunker
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
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

        ``on_plan`` is called with the work plan before any change is applied.

        Refusals: ``IndexLockHeld``, ``ResumeAvailable`` when the latest build
        is unfinished, ``IndexIncomplete`` without a completed index and
        ``ConfigDrift`` when a setting that decides vectors or chunking differs
        from the index.
        """
        started = self.clock()
        with self.lock.hold("update"):
            self._check_preconditions()
            self._close_dead_update()
            run_id = self.manifest.start_run(self.run_config, kind="update")
            _LOG.info(
                "index_update_started", extra={"run_id": run_id, **self.log_context}
            )
            return self._execute(run_id, started, retry_failed, on_plan)

    def _check_preconditions(self) -> None:
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
        metadata = self.index.read_completion_metadata()
        if metadata is None:
            raise IndexIncomplete(
                "No completed index exists to update; build one with `docgrep index`"
            )
        drifted = config_drift(metadata, self.run_config)
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

    def _close_dead_update(self) -> None:
        """Mark an update that lost its process as aborted; its work stays."""
        latest = self.manifest.latest_run()
        if latest is not None and latest.kind == "update":
            if latest.status == "running":
                self.manifest.finish_run(latest.run_id, "aborted", dict(latest.totals))
                _LOG.info("update_run_closed", extra={"run_id": latest.run_id})

    def _plan(self, retry_failed: bool) -> tuple[list[_Change], UpdatePlan]:
        recorded = {r.source.path: r for r in self.manifest.committed_sources()}
        found: dict[ChangeKind, list[_Change]] = {kind: [] for kind in _ORDER}
        reasons: dict[ModifiedReason, int] = {
            "content_changed": 0,
            "parser_changed": 0,
            "newly_supported": 0,
        }
        seen: set[str] = set()
        unchanged = 0
        refreshed = 0
        for item in self.corpus.scan():
            if item.excluded_by is not None:
                continue
            seen.add(item.path)
            record = recorded.get(item.path)
            if record is None:
                found["added"].append(
                    _Change("added", self.corpus.describe(item.path), None)
                )
                continue
            current: Source | None = None
            old = record.source
            if item.size != old.size or item.mtime != old.mtime:
                current = self.corpus.describe(item.path)
                if current.content_hash != old.content_hash:
                    found["modified"].append(_Change("modified", current, record))
                    reasons["content_changed"] += 1
                    continue
                # Same content: only the recorded size and time follow the
                # file, in the passages first and then in the record.
                fresh = replace(old, size=current.size, mtime=current.mtime)
                if current.size is not None and current.mtime is not None:
                    self.index.refresh_live_source(
                        old.source_id, size=current.size, mtime=current.mtime
                    )
                self.manifest.record_source(
                    fresh, record.status, record.error, record.embedding_model
                )
                refreshed += 1
            reason = self._parser_reason(record)
            if reason is not None:
                current = current or self.corpus.describe(item.path)
                found["modified"].append(_Change("modified", current, record))
                reasons[reason] += 1
            elif record.status.startswith("failed:") and retry_failed:
                current = current or self.corpus.describe(item.path)
                found["retried"].append(_Change("retried", current, record))
            else:
                unchanged += 1
        for path in sorted(set(recorded) - seen):
            found["removed"].append(
                _Change("removed", recorded[path].source, recorded[path])
            )
        changes = [
            change
            for kind in _ORDER
            for change in sorted(found[kind], key=lambda c: c.source.path)
        ]
        plan = UpdatePlan(
            added=len(found["added"]),
            modified=len(found["modified"]),
            removed=len(found["removed"]),
            unchanged=unchanged,
            retried=len(found["retried"]),
            refreshed=refreshed,
            **reasons,
        )
        return changes, plan

    def _parser_reason(self, record: ManifestSourceRecord) -> ModifiedReason | None:
        """Why the registered parser would treat the source differently, if so.

        A source skipped as unsupported is newly supported once a parser is
        registered; an indexed or OCR-pending source has a changed parser when
        the parser's version differs from the recorded one or the parser is
        gone. A missing or empty version, recorded or declared, counts as
        version "1" (``effective_parser_version``), so sources recorded before
        versions existed are not reprocessed for that alone. Failed sources are
        left to the retry rule.
        """
        parser = self.parsers.get(record.source.file_type)
        return parser_reprocess_reason(
            record.status,
            record.source.parser_version,
            registered=parser is not None,
            declared_version=parser_version(parser),
        )

    def _execute(
        self,
        run_id: str,
        started: float,
        retry_failed: bool,
        on_plan: Callable[[UpdatePlan], None] | None,
    ) -> UpdateSummary:
        counts: dict[str, int] = dict.fromkeys(_ORDER, 0)
        totals = {"indexed": 0, "skipped": 0, "failed": 0}
        written = 0
        removed = 0
        failures: list[BuildFailure] = []
        plan = UpdatePlan()
        cancelled = False
        try:
            changes, plan = self._plan(retry_failed)
            self.manifest.set_total_sources(run_id, len(changes))
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
        self, change: _Change, run_id: str
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
            if change.record is None:
                # An added source is recorded as failed, as a build would.
                self.manifest.record_source(
                    prepared.source, prepared.status, str(prepared.error)
                )
            # A recorded source keeps its passages and its record, so the next
            # update finds it again.
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
        self, change: _Change, run_id: str
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
