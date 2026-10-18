"""Re-encode the stored sparse vectors from the passage text (ADR-0025).

Only the sparse vector of each passage is rewritten, from the text in its
payload; dense vectors and payloads stay as they are, and no model is
loaded. The run is journaled with a checkpoint after every page, so a run
that stopped continues from there with the parameters it started with.
"""

import json
import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from docgrep.domain.errors import (
    BuildFailed,
    DocgrepError,
    IndexIncomplete,
    ResumeAvailable,
    ValidationError,
)
from docgrep.domain.models import ManifestRunRecord
from docgrep.domain.progress import (
    ProgressEvent,
    ProgressOptions,
    estimate_eta_seconds,
)
from docgrep.domain.protocols import (
    IndexLock,
    Manifest,
    SparseEncoders,
    SparseVectorStore,
)
from docgrep.domain.run_state import (
    REENCODE_SPARSE,
    RESUMABLE_STATES,
    latest_build_run,
    run_state,
    unfinished_reencode,
)
from docgrep.domain.sparse import (
    BM25_AVGDL_KEY,
    BM25_B_KEY,
    BM25_K1_KEY,
    DEFAULT_B,
    DEFAULT_K1,
    SparseWeighting,
    check_bm25_parameters,
)
from docgrep.usecases.build_index import Cancelled, Progress
from docgrep.usecases.sparse_vectors import (
    DEFAULT_PAGE_SIZE,
    average_length,
    encode_pages,
)

_LOG = logging.getLogger(__name__)

# The run configuration key that ties a run to the index it re-encodes.
_INDEX_KEY = "index_completed_at"


class SparseIndex(SparseVectorStore, Protocol):
    """The published index as a re-encode reads and changes it."""

    def read_completion_metadata(self) -> Mapping[str, str | int | float] | None: ...

    def merge_completion_metadata(
        self, updates: Mapping[str, str | int | float]
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class ReencodePlan:
    """What a re-encode would do, before anything is written.

    ``noop`` is true when the index is ``bm25-v2`` with the requested
    parameters and no run is unfinished. ``resumes`` names the unfinished
    run that would be continued, with the passages it had done; its
    average length is the one it started with, not a new one.
    """

    passages: int
    current: SparseWeighting
    target: SparseWeighting
    noop: bool = False
    resumes: str | None = None
    done: int = 0


@dataclass(frozen=True, slots=True)
class ReencodeSummary:
    """The outcome of a re-encode run; ``run_id`` is None for a no-op.

    ``skipped`` counts the passages of this run without any sparse token,
    whose stored vector is already empty and is left as it is.
    """

    plan: ReencodePlan
    run_id: str | None = None
    written: int = 0
    elapsed_seconds: float = 0.0
    cancelled: bool = False
    skipped: int = 0


@dataclass(frozen=True, slots=True)
class _Checkpoint:
    """Where a run has got to: the offset of the next page (None at the
    start or after the last page), the passages done (written or skipped),
    whether the last page is written, and the passages skipped so far.

    ``skipped`` is absent from checkpoints written before it existed, which
    load with 0.
    """

    offset: str | None = None
    done: int = 0
    scanned: bool = False
    skipped: int = 0

    def dumps(self) -> str:
        return json.dumps(
            {
                "offset": self.offset,
                "done": self.done,
                "scanned": self.scanned,
                "skipped": self.skipped,
            },
            sort_keys=True,
        )

    @classmethod
    def loads(cls, value: str | None) -> "_Checkpoint":
        if not value:
            return cls()
        data = json.loads(value)
        offset = data.get("offset")
        return cls(
            None if offset is None else str(offset),
            int(data.get("done", 0)),
            bool(data.get("scanned", False)),
            int(data.get("skipped", 0)),
        )


@dataclass(frozen=True, slots=True)
class _Resume:
    run: ManifestRunRecord
    target: SparseWeighting
    checkpoint: _Checkpoint = field(default_factory=_Checkpoint)


def _index_identity(metadata: Mapping[str, str | int | float]) -> str:
    """What tells one published index from the next: its build time."""
    return str(metadata.get("completed_at", ""))


def _totals(state: _Checkpoint) -> dict[str, int]:
    """The run journal totals: passages done and those skipped as empty."""
    return {"passages": state.done, "skipped_empty": state.skipped}


def reencode_progress(run: ManifestRunRecord) -> tuple[int, int | None]:
    """The passages a re-encode run has written and the passages it plans
    to cover (None when not recorded)."""
    try:
        done = _Checkpoint.loads(run.checkpoint).done
    except (ValueError, TypeError):
        done = 0
    return done, run.total_sources


class ReencodeSparse:
    """Rewrite every sparse vector of the published index as ``bm25-v2``."""

    def __init__(
        self,
        *,
        lock: IndexLock,
        index: SparseIndex,
        manifest: Manifest,
        encoders: SparseEncoders,
        page_size: int = DEFAULT_PAGE_SIZE,
        clock: Callable[[], float] = time.monotonic,
        progress_every_passages: int = 10_000,
        progress_every_seconds: float = 60.0,
    ) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        self.lock = lock
        self.index = index
        self.manifest = manifest
        self.encoders = encoders
        self.page_size = page_size
        self.clock = clock
        self.progress_every_passages = progress_every_passages
        self.progress_every_seconds = progress_every_seconds
        self._cancel_event = threading.Event()
        self._cancel_persisted = False

    def request_cancel(self) -> None:
        """Ask a running re-encode to stop after its current page; safe from a
        signal handler."""
        self._cancel_event.set()

    def plan(
        self,
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        progress: ProgressOptions | None = None,
    ) -> ReencodePlan:
        """What ``run`` would do with these parameters; reads, never writes.

        Computing a new average length reads the text of every passage; an
        observer in ``progress`` sees that scan.
        """
        return self._plan(k1, b, lock_held=self.lock.is_held(), progress=progress)[0]

    def run(
        self,
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        on_plan: Callable[[ReencodePlan], None] | None = None,
        progress: ProgressOptions | None = None,
    ) -> ReencodeSummary:
        """Re-encode under the index lock; a no-op when the index already
        has ``bm25-v2`` with these parameters.

        Refusals, before anything is written: ``IndexLockHeld``,
        ``ValidationError`` for bad parameters, ``ResumeAvailable`` while a
        build is unfinished and ``IndexIncomplete`` without a published
        index.
        """
        started = self.clock()
        with self.lock.hold("sparse re-encode"):
            plan, metadata, resume = self._plan(
                k1, b, lock_held=False, progress=progress
            )
            if on_plan is not None:
                on_plan(plan)
            if plan.noop:
                _LOG.info("sparse_reencode_not_needed", extra={"k1": k1, "b": b})
                return ReencodeSummary(plan)
            return self._execute(plan, metadata, resume, started, progress)

    def _plan(
        self,
        k1: float,
        b: float,
        *,
        lock_held: bool,
        progress: ProgressOptions | None = None,
    ) -> tuple[ReencodePlan, Mapping[str, str | int | float], _Resume | None]:
        try:
            check_bm25_parameters(k1, b)
        except ValueError as error:
            raise ValidationError(str(error), details={"k1": k1, "b": b}) from error
        build = latest_build_run(self.manifest)
        if build is not None:
            state = run_state(build, lock_held=lock_held)
            if state in RESUMABLE_STATES or state == "swapped":
                raise ResumeAvailable(
                    "The latest build is unfinished: continue it with "
                    "`docgrep index --resume` or discard it with "
                    "`docgrep index --restart` before re-encoding",
                    details={"run_id": build.run_id, "state": state},
                )
        metadata = self.index.read_completion_metadata()
        if metadata is None:
            raise IndexIncomplete(
                "No completed index exists to re-encode; build one with `docgrep index`"
            )
        try:
            current = SparseWeighting.from_metadata(metadata)
        except ValueError:
            # An unknown weighting is replaced like any other.
            current = SparseWeighting()
        passages = self.index.target_passage_count()
        unfinished = unfinished_reencode(self.manifest, lock_held=lock_held)
        resume = self._resumable(unfinished, metadata, k1, b)
        if resume is not None:
            plan = ReencodePlan(
                passages,
                current,
                resume.target,
                resumes=resume.run.run_id,
                done=resume.checkpoint.done,
            )
            return plan, metadata, resume
        if unfinished is None and current.same_parameters(k1, b):
            return ReencodePlan(passages, current, current, noop=True), metadata, None
        avgdl = self._scan(passages, progress)
        target = SparseWeighting.bm25(k1, b, avgdl)
        return ReencodePlan(passages, current, target), metadata, None

    def _scan(self, passages: int, progress: ProgressOptions | None) -> float:
        """The average passage length, reading every passage once; an
        observer sees the scan as its own phase."""
        observer = None if progress is None else progress.observer
        if observer is None:
            return average_length(self.index, self.encoders, page_size=self.page_size)[
                0
            ]
        started = self.clock()

        def event(done: int, *, finished: bool = False) -> ProgressEvent:
            elapsed = self.clock() - started
            return ProgressEvent(
                operation="reencode",
                unit="passages",
                done=done,
                total=passages,
                elapsed_seconds=elapsed,
                eta_seconds=estimate_eta_seconds(
                    done=done, total=passages, elapsed_seconds=elapsed
                ),
                phase="scanning",
                finished=finished,
            )

        observer(event(0))
        avgdl, scanned = average_length(
            self.index,
            self.encoders,
            page_size=self.page_size,
            after_page=lambda done: observer(event(done)),
        )
        observer(event(scanned, finished=True))
        return avgdl

    @staticmethod
    def _resumable(
        run: ManifestRunRecord | None,
        metadata: Mapping[str, str | int | float],
        k1: float,
        b: float,
    ) -> _Resume | None:
        """The unfinished run to continue: same parameters, same index.

        An unfinished run of other parameters is not continued; a new run
        with the requested ones re-encodes every passage instead.
        """
        if run is None:
            return None
        config = dict(run.config)
        if config.get(_INDEX_KEY) != _index_identity(metadata):
            return None
        try:
            target = SparseWeighting.from_metadata(config)
            checkpoint = _Checkpoint.loads(run.checkpoint)
        except (ValueError, TypeError):
            return None
        if not target.same_parameters(k1, b):
            return None
        return _Resume(run, target, checkpoint)

    def _check_cancel(self, run_id: str) -> None:
        if self._cancel_event.is_set():
            if not self._cancel_persisted:
                self.manifest.request_cancel(run_id)
                self._cancel_persisted = True
            raise Cancelled
        if self.manifest.cancel_requested_at(run_id) is not None:
            raise Cancelled

    def _close_dead_run(self) -> None:
        """An interrupted re-encode still says running; mark it aborted."""
        latest = self.manifest.latest_run()
        if (
            latest is not None
            and latest.kind == REENCODE_SPARSE
            and latest.status == "running"
        ):
            self.manifest.finish_run(latest.run_id, "aborted", dict(latest.totals))

    def _execute(
        self,
        plan: ReencodePlan,
        metadata: Mapping[str, str | int | float],
        resume: _Resume | None,
        started: float,
        options: ProgressOptions | None = None,
    ) -> ReencodeSummary:
        self._close_dead_run()
        target = plan.target
        config: dict[str, str | int | float] = {
            **target.metadata(),
            _INDEX_KEY: _index_identity(metadata),
        }
        checkpoint = _Checkpoint() if resume is None else resume.checkpoint
        run_id = self.manifest.start_run(
            config,
            resumed_from=None if resume is None else resume.run.run_id,
            kind=REENCODE_SPARSE,
        )
        self.manifest.set_total_sources(run_id, plan.passages, carried=checkpoint.done)
        self.manifest.set_checkpoint(run_id, checkpoint.dumps())
        _LOG.info(
            "sparse_reencode_started",
            extra={
                "run_id": run_id,
                "passages": plan.passages,
                "resumed_from": None if resume is None else resume.run.run_id,
                "done": checkpoint.done,
                BM25_K1_KEY: target.k1,
                BM25_B_KEY: target.b,
                BM25_AVGDL_KEY: target.avgdl,
            },
        )
        # The rate and ETA cover the writing only, not the planning scan.
        counter = Progress(
            run_id=run_id,
            total=plan.passages,
            clock=self.clock,
            started=self.clock(),
            every_sources=self.progress_every_passages,
            every_seconds=self.progress_every_seconds,
            carried=checkpoint.done,
            unit="passages",
            operation="reencode",
            options=options,
        )
        counter.start()
        written = 0
        skipped = 0
        state = checkpoint
        cancelled = False

        def after_page(following: str | None, count: int, empty: int) -> None:
            nonlocal state, written, skipped
            written += count
            skipped += empty
            state = _Checkpoint(
                following,
                state.done + count + empty,
                following is None,
                state.skipped + empty,
            )
            self.manifest.set_checkpoint(run_id, state.dumps())
            counter.advance(count, 0, done=count + empty)

        try:
            if not state.scanned:
                encode_pages(
                    self.index,
                    self.encoders.document_encoder(target),
                    start=state.offset,
                    page_size=self.page_size,
                    before_page=lambda: self._check_cancel(run_id),
                    after_page=after_page,
                )
            counter.finish(0)
            self._check_cancel(run_id)
            self.index.merge_completion_metadata(target.metadata())
            self.manifest.finish_run(run_id, "completed", _totals(state))
        except Cancelled:
            self.manifest.finish_run(run_id, "cancelled", _totals(state))
            cancelled = True
        except Exception as error:
            self.manifest.finish_run(run_id, "aborted", _totals(state))
            if isinstance(error, DocgrepError):
                raise
            raise BuildFailed(
                "Sparse re-encode aborted", details={"reason": str(error)}
            ) from error
        finally:
            self._cancel_event.clear()
            self._cancel_persisted = False
        elapsed = self.clock() - started
        _LOG.info(
            "sparse_reencode_cancelled" if cancelled else "sparse_reencode_finished",
            extra={
                "run_id": run_id,
                "passages_written": written,
                "passages_skipped_empty": skipped,
                "duration_ms": round(elapsed * 1000, 1),
            },
        )
        return ReencodeSummary(plan, run_id, written, elapsed, cancelled, skipped)
