"""Follow the configured roots by periodic scan and apply what has settled,
safely enough to run unattended (ADR-0029).

One cycle scans every readable root from directory entries alone, keeps the
changes it saw in memory until they have settled, and then applies the settled
ones with an update scoped to their paths. A root that cannot be read is
paused, removals are applied only when the removal guard allows them, and
nothing is written while another operation holds the index lock.
"""

import json
import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from docgrep.domain.errors import DocgrepError, IndexLockHeld, ValidationError
from docgrep.domain.models import ManifestSourceRecord, ScannedFile, Source
from docgrep.domain.progress import ProgressEvent, ProgressOptions
from docgrep.domain.protocols import (
    CorpusReader,
    ManifestReader,
    RootCatalog,
    SyncCorpus,
)
from docgrep.domain.roots import RUN_ROOTS_KEY, CorpusRoots
from docgrep.domain.sync import (
    DEFAULT_SYNC_INTERVAL_SECONDS,
    DEFAULT_SYNC_RECONCILE_HOURS,
    DEFAULT_SYNC_SETTLE_SECONDS,
    SETTLE_SCANS,
    UNSETTLED_REPORT_SECONDS,
    HeldRoot,
    LastApplied,
    Pending,
    SyncStatus,
    is_ignored,
)
from docgrep.domain.update_plan import UpdateScope, pair_moves
from docgrep.usecases.plan_update import DetectedChanges, PlanUpdate
from docgrep.usecases.update_index import UpdateIndex, UpdateSummary

_LOG = logging.getLogger(__name__)

# How many stored paths of a held root the status names.
SAMPLE_PATHS = 5
# The pause after each source of an applied run, which lets a search that is
# waiting for the models go first.
YIELD_SECONDS = 0.05


class ScopedUpdates(Protocol):
    """The update sync applies: scoped to paths, recorded as a ``sync`` run."""

    def check_ready(self) -> None:
        """Raise a ``DocgrepError`` when no update can run now."""
        ...

    def apply(self, scope: UpdateScope, progress: ProgressOptions) -> UpdateSummary:
        """Apply the scope; raises ``IndexLockHeld`` without waiting when
        another operation owns the index lock."""
        ...

    def request_cancel(self) -> None:
        """Stop after the source being processed."""
        ...


class UpdateApplier:
    """``ScopedUpdates`` over the update use case of the application."""

    def __init__(self, update: UpdateIndex) -> None:
        self._update = update

    def check_ready(self) -> None:
        self._update.check_ready()

    def apply(self, scope: UpdateScope, progress: ProgressOptions) -> UpdateSummary:
        return self._update.run(scope=scope, run_kind="sync", progress=progress)

    def request_cancel(self) -> None:
        self._update.request_cancel()


class _ReconcileView:
    """The readable roots as a corpus an update plan can read: the files sync
    does not leave alone, root by root."""

    def __init__(self, corpus: SyncCorpus, names: Iterable[str]) -> None:
        self._corpus = corpus
        self._names = tuple(names)

    def scan(self) -> tuple[ScannedFile, ...]:
        return tuple(
            item
            for name in self._names
            for item in self._corpus.scan_root(name)
            if not is_ignored(item.path)
        )

    def describe(self, path: str) -> Source:
        return self._corpus.describe(path)


@dataclass(frozen=True, slots=True)
class CycleResult:
    """What one cycle did, in counts."""

    scanned_roots: int = 0
    applied: UpdateSummary | None = None
    skipped: str | None = None
    held: int = 0
    pending: int = 0


@dataclass(slots=True)
class _State:
    pending: dict[str, Pending] = field(default_factory=dict)
    # Consecutive scans that did not see a recorded path.
    absent: dict[str, int] = field(default_factory=dict)
    # Removals the guard held back, by root.
    held: dict[str, set[str]] = field(default_factory=dict)
    # The size and time a path had when it was last applied, so that a file
    # that failed is retried only when it changes.
    attempted: dict[str, tuple[int, float]] = field(default_factory=dict)
    paused: set[str] = field(default_factory=set)
    awaiting: set[str] = field(default_factory=set)
    started: bool = False
    last_reconcile: float | None = None
    last_scan_at: str | None = None
    last_applied: LastApplied | None = None
    idle_reason: str | None = None
    last_error: str | None = None
    applying: tuple[int, int] | None = None


class SyncIndex:
    """Keep the index following the configured roots (ADR-0029).

    ``run_cycle`` does one scan and, when something has settled, one scoped
    update; the caller decides when. All state is in memory and is rebuilt by
    the reconciliation of the first cycle, so a restart loses nothing.
    """

    def __init__(
        self,
        *,
        corpus: SyncCorpus,
        roots: RootCatalog,
        manifest: ManifestReader,
        updates: ScopedUpdates,
        planner: Callable[[CorpusReader], PlanUpdate],
        interval_seconds: float = DEFAULT_SYNC_INTERVAL_SECONDS,
        settle_seconds: float = DEFAULT_SYNC_SETTLE_SECONDS,
        reconcile_hours: float = DEFAULT_SYNC_RECONCILE_HOURS,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep: Callable[[float], None] = time.sleep,
        yield_seconds: float = YIELD_SECONDS,
        on_change: Callable[[SyncStatus], None] | None = None,
    ) -> None:
        """``planner`` builds the update plan over a corpus view, for the
        reconciliation. ``on_change`` is called with the status after every
        cycle and when an applied run starts or ends."""
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if settle_seconds < 0:
            raise ValueError("settle_seconds must not be negative")
        if reconcile_hours <= 0:
            raise ValueError("reconcile_hours must be positive")
        self.corpus = corpus
        self.roots = roots
        self.manifest = manifest
        self.updates = updates
        self.planner = planner
        self.interval_seconds = interval_seconds
        self.settle_seconds = settle_seconds
        self.reconcile_seconds = reconcile_hours * 3600
        self.clock = clock
        self.wall_clock = wall_clock
        self.now = now
        self.sleep = sleep
        self.yield_seconds = yield_seconds
        self.on_change = on_change
        self._state = _State()
        self._guard = threading.Lock()
        self._stopping = threading.Event()
        # How many sources of the run in progress were already followed by
        # a pause.
        self._yielded = 0

    # -- control ---------------------------------------------------------

    def request_stop(self) -> None:
        """Start no new cycle, and stop an applied run after its current
        source."""
        self._stopping.set()
        self.updates.request_cancel()

    @property
    def stopping(self) -> bool:
        return self._stopping.is_set()

    # -- status ----------------------------------------------------------

    def status(self, *, running: bool = True) -> SyncStatus:
        """The state sync reports; a copy, safe to call during a cycle."""
        with self._guard:
            state = self._state
            kinds = [entry.kind for entry in state.pending.values()]
            loose = [p for p in state.absent if not self._is_held(p)]
            wall = self.wall_clock()
            mono = self.clock()
            waiting = [
                entry
                for entry in state.pending.values()
                if not entry.settled(wall, self.settle_seconds)
            ]
            return SyncStatus(
                enabled=True,
                running=running,
                interval_seconds=self.interval_seconds,
                settle_seconds=self.settle_seconds,
                last_scan_at=state.last_scan_at,
                last_applied=state.last_applied,
                pending_added=kinds.count("added"),
                pending_modified=kinds.count("modified"),
                pending_removed=len(loose),
                unsettled=len(waiting)
                + sum(1 for p in loose if state.absent[p] < SETTLE_SCANS),
                unsettled_over_limit=sum(
                    1
                    for entry in waiting
                    if mono - entry.first_seen >= UNSETTLED_REPORT_SECONDS
                ),
                held=tuple(
                    HeldRoot(root, len(paths), tuple(sorted(paths))[:SAMPLE_PATHS])
                    for root, paths in sorted(state.held.items())
                    if paths
                ),
                paused_roots=tuple(sorted(state.paused)),
                awaiting_first_update=tuple(sorted(state.awaiting)),
                idle_reason=state.idle_reason,
                last_error=state.last_error,
                applying=state.applying,
            )

    def _is_held(self, path: str) -> bool:
        return any(path in paths for paths in self._state.held.values())

    def publish(self, *, running: bool = True) -> None:
        """Hand the current status to ``on_change``."""
        if self.on_change is not None:
            try:
                self.on_change(self.status(running=running))
            except Exception:
                _LOG.warning("sync_status_not_published", exc_info=True)

    # -- one cycle -------------------------------------------------------

    def run_cycle(self) -> CycleResult:
        """Scan, settle and apply once; never raises for a corpus or an
        index that is not ready, only reports why in the status."""
        if self._stopping.is_set():
            return CycleResult(skipped="stopping")
        started = self.clock()
        with self._guard:
            self._state.last_scan_at = self.now().isoformat()
        result = self._cycle()
        _LOG.info(
            "sync_cycle",
            extra={
                "scanned_roots": result.scanned_roots,
                "skipped": result.skipped or "",
                "pending": result.pending,
                "held_removals": result.held,
                "applied": result.applied is not None,
                **_counts(result.applied),
                "duration_ms": round((self.clock() - started) * 1000, 1),
            },
        )
        self.publish()
        return result

    def _cycle(self) -> CycleResult:
        try:
            self.updates.check_ready()
        except DocgrepError as error:
            with self._guard:
                self._state.idle_reason = f"{error.category}: {error}"
            return CycleResult(skipped="not_ready")
        with self._guard:
            self._state.idle_reason = None

        roots = self.roots.roots()
        committed = {
            record.source.path: record for record in self.manifest.committed_sources()
        }
        awaiting = self._awaiting(roots, committed)
        scanned, paused = self._scan(roots, awaiting)
        with self._guard:
            resumed = self._state.paused - paused
            self._state.paused = paused
            self._state.awaiting = awaiting
            self._update_pending(roots, committed, scanned, paused | awaiting)
        self._reconcile_if_due(roots, scanned, resumed)
        return self._apply(roots, scanned)

    def _awaiting(
        self, roots: CorpusRoots, committed: dict[str, ManifestSourceRecord]
    ) -> set[str]:
        """The roots whose first manual update has not completed: they have
        no recorded source and no completed build or update visited them."""
        with_sources = {roots.root_of(path) for path in committed}
        visited: set[str] = set()
        for kind in ("build", "update"):
            run = self.manifest.latest_run_of_kind(kind)
            if run is not None and run.status == "completed":
                visited |= _visited_roots(dict(run.config).get(RUN_ROOTS_KEY))
        return {name for name in roots.names if name not in with_sources | visited}

    def _scan(
        self, roots: CorpusRoots, skip: set[str]
    ) -> tuple[dict[str, dict[str, ScannedFile]], set[str]]:
        """The files sync looks at in each readable root, and the roots it
        pauses because they cannot be read."""
        scanned: dict[str, dict[str, ScannedFile]] = {}
        paused: set[str] = set()
        for name in roots.names:
            if name in skip:
                continue
            try:
                if not self.corpus.root_readable(name):
                    paused.add(name)
                    continue
                files = self.corpus.scan_root(name)
            except (ValidationError, OSError):
                paused.add(name)
                continue
            scanned[name] = {
                item.path: item
                for item in files
                if item.excluded_by is None and not is_ignored(item.path)
            }
        return scanned, paused

    def _update_pending(
        self,
        roots: CorpusRoots,
        committed: dict[str, ManifestSourceRecord],
        scanned: dict[str, dict[str, ScannedFile]],
        inactive: set[str],
    ) -> None:
        """Fold one scan into the pending changes and absent counts."""
        state = self._state
        mono = self.clock()
        for table in (state.pending, state.absent, state.attempted):
            for path in [p for p in table if roots.root_of(p) in inactive]:
                del table[path]
        state.held = {
            root: paths for root, paths in state.held.items() if root not in inactive
        }
        for name, files in scanned.items():
            for path, item in files.items():
                state.absent.pop(path, None)
                record = committed.get(path)
                stat = (item.size, item.mtime)
                entry = state.pending.get(path)
                if record is not None and stat == (
                    record.source.size,
                    record.source.mtime,
                ):
                    # Unchanged on disk; only a reconciliation keeps it.
                    if entry is None or not entry.forced:
                        state.pending.pop(path, None)
                    state.attempted.pop(path, None)
                    continue
                if state.attempted.get(path) == stat:
                    state.pending.pop(path, None)
                    continue
                kind = "added" if record is None else "modified"
                mtime = item.mtime if item.mtime is not None else 0.0
                if entry is None:
                    state.pending[path] = Pending(
                        kind, item.size, mtime, first_seen=mono
                    )
                else:
                    entry.kind = kind
                    entry.observe(item.size, mtime)
            for path in [
                p for p in state.pending if roots.root_of(p) == name and p not in files
            ]:
                del state.pending[path]
        for path, record in committed.items():
            root = roots.root_of(path)
            if root not in scanned or is_ignored(path):
                continue
            if path not in scanned[root]:
                state.absent[path] = state.absent.get(path, 0) + 1
        for path in [p for p in state.absent if p not in committed]:
            del state.absent[path]
        for root, paths in state.held.items():
            paths &= {p for p in paths if p in state.absent}

    # -- reconciliation --------------------------------------------------

    def _reconcile_if_due(
        self,
        roots: CorpusRoots,
        scanned: dict[str, dict[str, ScannedFile]],
        resumed: set[str],
    ) -> None:
        """Compare whole roots with the manifest with the update plan: at
        start-up, daily, and for a root that is readable again."""
        mono = self.clock()
        state = self._state
        everything = (
            not state.started
            or state.last_reconcile is None
            or mono - state.last_reconcile >= self.reconcile_seconds
        )
        names = set(scanned) if everything else resumed & set(scanned)
        if not names:
            return
        try:
            view = _ReconcileView(self.corpus, sorted(names))
            detected = self.planner(view).detect(True, roots.only(names))
        except (DocgrepError, OSError) as error:
            _LOG.warning(
                "sync_reconcile_failed",
                extra={"category": getattr(error, "category", "os_error")},
            )
            return
        with self._guard:
            if everything:
                state.started = True
                state.last_reconcile = mono
                state.attempted.clear()
            self._merge_plan(detected, mono)

    def _merge_plan(self, detected: DetectedChanges, mono: float) -> None:
        """Mark what a reconciliation found as settled, except the targets of
        moves, which wait for the removal of their old path so that both are
        applied together."""
        state = self._state
        removed = [c.source for c in detected.changes if c.kind == "removed"]
        added = [c.source for c in detected.changes if c.kind == "added"]
        delayed = {m.added for m in pair_moves(removed, added)}
        delayed |= {c.source.path for c in detected.changes if c.kind == "moved"}
        for change in detected.changes:
            path = change.source.path
            if change.kind in ("removed", "moved") or path in delayed:
                continue
            kind = "added" if change.kind in ("added", "copied") else "modified"
            entry = state.pending.get(path)
            if entry is None:
                entry = Pending(
                    kind,
                    change.source.size or 0,
                    change.source.mtime or 0.0,
                    first_seen=mono,
                )
                state.pending[path] = entry
            entry.forced = True
            entry.scans = max(entry.scans, SETTLE_SCANS)

    # -- applying --------------------------------------------------------

    def _apply(
        self, roots: CorpusRoots, scanned: dict[str, dict[str, ScannedFile]]
    ) -> CycleResult:
        mono, wall = self.clock(), self.wall_clock()
        state = self._state
        with self._guard:
            files = sorted(
                path
                for path, entry in state.pending.items()
                if entry.settled(wall, self.settle_seconds)
            )
            removals = sorted(
                path
                for path, count in state.absent.items()
                if count >= SETTLE_SCANS and not self._is_held(path)
            )
            resubmit = {
                path
                for root in {roots.root_of(path) for path in removals}
                for path in state.held.get(root, ())
            }
            self._report_unsettled(mono, wall)
            held_total = sum(len(paths) for paths in state.held.values())
            pending_total = len(state.pending) + len(state.absent)
        base = CycleResult(
            scanned_roots=len(scanned), held=held_total, pending=pending_total
        )
        if not files and not removals:
            return base
        scope = UpdateScope(frozenset(files) | frozenset(removals) | resubmit)
        try:
            summary = self.updates.apply(scope, ProgressOptions(self._on_progress))
        except IndexLockHeld:
            return CycleResult(
                scanned_roots=len(scanned),
                skipped="index_lock_held",
                held=held_total,
                pending=pending_total,
            )
        except Exception as error:
            category = (
                error.category if isinstance(error, DocgrepError) else "unexpected"
            )
            _LOG.warning("sync_apply_failed", extra={"category": category})
            with self._guard:
                state.applying = None
                state.last_error = f"{category}: {error}"
            return CycleResult(
                scanned_roots=len(scanned),
                skipped="apply_failed",
                held=held_total,
                pending=pending_total,
            )
        with self._guard:
            self._record_applied(roots, scope, summary)
            held_now = sum(len(paths) for paths in state.held.values())
            return CycleResult(
                scanned_roots=len(scanned),
                applied=summary,
                held=held_now,
                pending=len(state.pending) + len(state.absent),
            )

    def _record_applied(
        self, roots: CorpusRoots, scope: UpdateScope, summary: UpdateSummary
    ) -> None:
        state = self._state
        held = set(summary.held_removals)
        for path in scope.paths:
            entry = state.pending.pop(path, None)
            if entry is not None:
                state.attempted[path] = (entry.size, entry.mtime)
            if path not in held:
                state.absent.pop(path, None)
        touched = {roots.root_of(path) for path in scope.paths}
        for root in touched:
            state.held.pop(root, None)
        for path in held:
            state.held.setdefault(roots.root_of(path), set()).add(path)
        state.applying = None
        state.last_error = None
        state.last_applied = LastApplied(
            run_id=summary.run_id,
            finished_at=self.now().isoformat(),
            added=summary.added,
            modified=summary.modified,
            removed=summary.removed,
            moved=summary.moved,
            copied=summary.copied,
            failed=summary.failed,
            cancelled=summary.cancelled,
        )

    def _report_unsettled(self, mono: float, wall: float) -> None:
        """Log once, with counts, the changes that never settle."""
        late = [
            entry
            for entry in self._state.pending.values()
            if not entry.settled(wall, self.settle_seconds)
            and not entry.reported
            and mono - entry.first_seen >= UNSETTLED_REPORT_SECONDS
        ]
        for entry in late:
            entry.reported = True
        if late:
            _LOG.warning("sync_unsettled", extra={"files": len(late)})

    def _on_progress(self, event: ProgressEvent) -> None:
        """Note how far an applied run has got, and give a waiting search the
        models after each source."""
        with self._guard:
            state = self._state
            state.applying = None if event.finished else (event.done, event.total or 0)
            finished_source = event.done > self._yielded
            self._yielded = 0 if event.finished else event.done
        if finished_source and not event.finished and self.yield_seconds > 0:
            self.sleep(self.yield_seconds)
        if event.done == 0 and not event.finished and event.label is None:
            self.publish()


def _counts(summary: UpdateSummary | None) -> dict[str, int]:
    if summary is None:
        return {}
    return {
        "added": summary.added,
        "modified": summary.modified,
        "removed": summary.removed,
        "moved": summary.moved,
        "copied": summary.copied,
        "failed": summary.failed,
        "held": len(summary.held_removals),
    }


def _visited_roots(value: object) -> set[str]:
    """The root names a run recorded as visited."""
    if not isinstance(value, str):
        return set()
    try:
        entries = json.loads(value)
    except ValueError:
        return set()
    if not isinstance(entries, list):
        return set()
    return {
        entry["name"]
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    }


__all__ = [
    "CycleResult",
    "ScopedUpdates",
    "SyncIndex",
    "UpdateApplier",
]
