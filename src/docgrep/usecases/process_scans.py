"""Make selected scanned PDF and DjVu files searchable through OCR (UC-6).

The files are classified with the scan discovery rules (UC-5), checked
against the OCR registry, and each one that needs it goes through the OCR
engine (a DjVu file is converted to a PDF of page images first). The
searchable PDF is written next to the original as ``<stem>.ocr.pdf``,
registered, and the original is delegated to it, so that the next update
indexes the output instead. The original is never changed. The index is not
touched: ``docgrep update`` picks the outputs up.
"""

import errno
import logging
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any, Literal

from docgrep.domain.errors import (
    BuildFailed,
    DocgrepError,
    InspectionFailed,
    OcrFailed,
    OcrToolsMissing,
    ValidationError,
)
from docgrep.domain.globs import matching_pattern
from docgrep.domain.models import FileType, OcrMode, OcrRecord, Source
from docgrep.domain.ocr import (
    ALREADY_PROCESSED,
    CANCELLED,
    DISK_FULL,
    DISK_SPACE_FACTOR,
    DJVU_TOOLS_MISSING,
    DOES_NOT_NEED_OCR,
    ENCRYPTED,
    INSTALL_COMMAND,
    OUTPUT_COLLISION,
    OUTPUT_INVALID,
    PROCESSED,
    REGISTRY_WRITE,
    UNREADABLE,
    eligible_type,
    missing_languages,
    ocr_mode,
    output_path,
    parse_languages,
)
from docgrep.domain.progress import (
    ProgressEvent,
    ProgressObserver,
    estimate_eta_seconds,
    item_label,
)
from docgrep.domain.protocols import (
    CorpusAvailability,
    CorpusScanner,
    DjvuPdfConverter,
    IndexLock,
    Manifest,
    OcrEngine,
    OcrFiles,
    OcrRegistry,
)
from docgrep.domain.scans import INSPECTION_FAILED, MIXED_OR_UNCERTAIN, NATIVE, SCANNED
from docgrep.usecases.discover_scans import DiscoverScans

_LOG = logging.getLogger(__name__)

SCHEMA_VERSION = 1
TO_PROCESS = "to_process"
NOT_STARTED = "not_started"
CANCEL_MODES = ("terminate", "finish")
UPDATE_HINT = "Run `docgrep update` so that the OCR outputs are indexed."
# Seconds between two reads of a cancel request persisted by another process
# while an engine runs.
CANCEL_POLL_SECONDS = 1.0
_TYPE_NAMES = {FileType.PDF: "pdf", FileType.DJVU: "djvu"}


@dataclass(frozen=True, slots=True)
class OcrRequest:
    """What to process: exactly one of ``paths``, ``path_glob`` and
    ``all_discovered``, with the options of one batch."""

    paths: tuple[str, ...] = ()
    path_glob: str | None = None
    all_discovered: bool = False
    include_uncertain: bool = False
    languages: str | None = None
    force: bool = False
    override_native: bool = False


@dataclass(frozen=True, slots=True)
class OcrTarget:
    """One file of a work plan and what will happen to it.

    ``status`` is ``to_process``, ``skipped:<reason>`` or ``failed:<reason>``.
    """

    path: str
    file_type: FileType
    size: int
    status: str
    page_count: int | None = None
    classification: str | None = None
    reason: str | None = None
    mode: OcrMode | None = None
    input_hash: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "file_type": _TYPE_NAMES[self.file_type],
            "size_bytes": self.size,
            "page_count": self.page_count,
            "classification": self.classification,
            "reason": self.reason,
            "mode": self.mode,
            "status": self.status,
        }


def _by_reason(statuses: Sequence[str], prefix: str) -> dict[str, int]:
    found = Counter(s.split(":", 1)[1] for s in statuses if s.startswith(prefix))
    return dict(sorted(found.items()))


def _plan_counts(targets: Sequence[OcrTarget]) -> dict[str, Any]:
    statuses = [t.status for t in targets]
    return {
        "to_process": statuses.count(TO_PROCESS),
        "to_skip": _by_reason(statuses, "skipped:"),
        "to_fail": _by_reason(statuses, "failed:"),
    }


@dataclass(frozen=True, slots=True)
class OcrPlan:
    """The work plan of a batch (UC-6 step 5), before any OCR."""

    languages: str
    force: bool
    targets: tuple[OcrTarget, ...]
    # Files a path glob matched that are neither PDF nor DjVu.
    filtered: int = 0

    @property
    def to_process(self) -> tuple[OcrTarget, ...]:
        return tuple(t for t in self.targets if t.status == TO_PROCESS)

    def as_dict(self) -> dict[str, Any]:
        """The JSON form; its key names are stable and documented."""
        work = self.to_process
        return {
            "schema_version": SCHEMA_VERSION,
            "languages": self.languages,
            "force": self.force,
            "filtered": self.filtered,
            **_plan_counts(self.targets),
            "by_file_type": {
                name: _plan_counts([t for t in self.targets if t.file_type is kind])
                for kind, name in _TYPE_NAMES.items()
            },
            "input_bytes": sum(t.size for t in work),
            "pages": sum(t.page_count or 0 for t in work),
            "pages_unknown": sum(t.page_count is None for t in work),
            "files": [t.as_dict() for t in self.targets],
        }


@dataclass(frozen=True, slots=True)
class OcrFileResult:
    """The outcome of one file the batch set out to process.

    ``status`` is ``processed``, ``cancelled``, ``failed:<reason>`` or
    ``not_started``.
    """

    target: OcrTarget
    status: str
    index: int
    total: int
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    output_path: str | None = None
    message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.target.as_dict(),
            "status": self.status,
            "output_path": self.output_path,
            "message": self.message,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
        }


@dataclass(frozen=True, slots=True)
class OcrSummary:
    """The final report of a batch.

    ``status`` is ``completed``, ``cancelled`` or ``aborted`` (with
    ``abort_reason``).
    """

    run_id: str
    plan: OcrPlan
    results: tuple[OcrFileResult, ...]
    status: str
    elapsed_seconds: float
    abort_reason: str | None = None

    @property
    def statuses(self) -> list[str]:
        """The final status of every planned file, in plan order."""
        done = {r.target.path: r.status for r in self.results}
        return [done.get(t.path, t.status) for t in self.plan.targets]

    @property
    def failed(self) -> int:
        return sum(s.startswith("failed:") for s in self.statuses)

    def as_dict(self) -> dict[str, Any]:
        """The JSON form; its key names are stable and documented."""
        statuses = self.statuses
        done = {r.target.path: r for r in self.results}
        files = [
            done[t.path].as_dict()
            if t.path in done
            else {
                **t.as_dict(),
                "output_path": None,
                "message": None,
                "elapsed_seconds": 0.0,
            }
            for t in self.plan.targets
        ]
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": self.status,
            "abort_reason": self.abort_reason,
            "plan": {k: v for k, v in self.plan.as_dict().items() if k != "files"},
            "totals": {
                "processed": statuses.count(PROCESSED),
                "skipped": sum(s.startswith("skipped:") for s in statuses),
                "failed": self.failed,
                "cancelled": statuses.count(CANCELLED),
                "not_started": statuses.count(NOT_STARTED),
            },
            "skipped_by_reason": _by_reason(statuses, "skipped:"),
            "failed_by_reason": _by_reason(statuses, "failed:"),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "files": files,
            "hint": UPDATE_HINT,
        }


class _Abort(Exception):
    """Control flow: the batch stops after this file (disk full)."""


class _Cancelled(Exception):
    """Control flow: a cancel was requested."""


@dataclass
class _Run:
    run_id: str
    started: float
    results: list[OcrFileResult] = field(default_factory=list)
    done_units: int = 0
    last_poll: float = float("-inf")
    cancel_seen: bool = False


class ProcessScans:
    """Run OCR over a batch under the exclusive index lock, file by file.

    A failure on one file never stops the others, except a full disk, which
    aborts the rest of the batch. A cancel (``request_cancel`` or a request
    persisted by ``docgrep cancel``) stops the batch: in ``terminate`` mode
    the running engine is stopped and its file reported ``cancelled``, in
    ``finish`` mode the current file completes first. Completed files stay
    registered, so running the batch again continues it.
    """

    def __init__(
        self,
        *,
        lock: IndexLock,
        manifest: Manifest,
        scanner: CorpusScanner,
        discovery: DiscoverScans,
        registry: OcrRegistry,
        engine: OcrEngine,
        converter: DjvuPdfConverter,
        files: OcrFiles,
        languages: str,
        jobs: int | None = None,
        timeout_seconds: float = 0,
        cancel_mode: str = "terminate",
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        poll_seconds: float = CANCEL_POLL_SECONDS,
        availability: CorpusAvailability | None = None,
    ) -> None:
        if cancel_mode not in CANCEL_MODES:
            raise ValueError("cancel_mode must be terminate or finish")
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must not be negative")
        if jobs is not None and jobs <= 0:
            raise ValueError("jobs must be positive")
        self.lock = lock
        self.manifest = manifest
        self.scanner = scanner
        self.discovery = discovery
        self.registry = registry
        self.engine = engine
        self.converter = converter
        self.files = files
        self.languages = languages
        self.jobs = jobs
        self.timeout_seconds = timeout_seconds
        self.cancel_mode = cancel_mode
        self.clock = clock
        self.now = now
        self.poll_seconds = poll_seconds
        self.availability = availability
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        """Ask the batch to stop; safe to call from a signal handler."""
        self._cancel_event.set()

    # Planning -------------------------------------------------------------

    def plan(self, request: OcrRequest) -> OcrPlan:
        """The work plan, without OCR and without taking the lock.

        Raises ``ValidationError`` for an invalid request and
        ``OcrToolsMissing`` when the engine or a language is not installed.
        """
        self._require_roots()
        languages = self._validate(request)
        self._check_tools(languages)
        return self._plan(request, languages)

    def _require_roots(self) -> None:
        """Refuse before any work when a root of the run cannot be read."""
        if self.availability is not None:
            self.availability.require_available()

    def _validate(self, request: OcrRequest) -> str:
        chosen = sum(
            (bool(request.paths), request.path_glob is not None, request.all_discovered)
        )
        if chosen != 1:
            raise ValidationError(
                "name the files to process with exactly one of: paths, a path "
                "glob, or all discovered scans",
                details={"reason": "targets"},
            )
        if request.override_native and not request.paths:
            raise ValidationError(
                "override_native requires an explicit list of paths",
                details={"reason": "override_native"},
            )
        if request.include_uncertain and not request.all_discovered:
            raise ValidationError(
                "include_uncertain only applies to all discovered scans",
                details={"reason": "include_uncertain"},
            )
        if request.path_glob is not None and not request.path_glob.strip():
            raise ValidationError(
                "path_glob must not be empty", details={"reason": "path_glob"}
            )
        languages = request.languages or self.languages
        try:
            parse_languages(languages)
        except ValueError as error:
            raise ValidationError(
                str(error), details={"reason": "languages", "languages": languages}
            ) from error
        # Raises for a path outside the corpus roots or a missing file.
        located = [self.files.locate(path) for path in request.paths]
        wrong = sorted(
            {
                PurePosixPath(path).suffix.lower() or "(none)"
                for path in located
                if eligible_type(path) is None
            }
        )
        if wrong:
            raise ValidationError(
                "only .pdf, .djvu and .djv files can be processed, not "
                + ", ".join(wrong),
                details={"reason": "extension", "extensions": ", ".join(wrong)},
            )
        return languages

    def _check_tools(self, languages: str) -> str:
        """The engine version, after checking the engine and its languages."""
        version = self.engine.version()
        missing = missing_languages(parse_languages(languages), self.engine.languages())
        if missing:
            raise OcrToolsMissing(
                "OCR language data is not installed for: "
                + ", ".join(missing)
                + f"; install it (Debian: {INSTALL_COMMAND})",
                details={"missing_languages": ", ".join(missing)},
            )
        return version

    def _candidates(self, request: OcrRequest) -> tuple[list[tuple[str, int]], int]:
        """The eligible target paths with their sizes, and the filtered count."""
        if request.paths:
            located = sorted({self.files.locate(path) for path in request.paths})
            return [(path, self.files.size(path)) for path in located], 0
        if request.path_glob is not None:
            eligible: list[tuple[str, int]] = []
            filtered = 0
            for item in self.scanner.scan():
                if item.excluded_by is not None:
                    continue
                if matching_pattern(item.path, [request.path_glob]) is None:
                    continue
                if eligible_type(item.path) is None:
                    filtered += 1
                else:
                    eligible.append((item.path, item.size))
            return eligible, filtered
        wanted = (
            {SCANNED, MIXED_OR_UNCERTAIN} if request.include_uncertain else {SCANNED}
        )
        report = self.discovery.run(registry=self.registry)
        return [
            (entry.path, entry.size_bytes)
            for entry in report.entries
            if entry.classification in wanted
        ], 0

    def _plan(
        self,
        request: OcrRequest,
        languages: str,
        check_cancel: Callable[[], None] | None = None,
    ) -> OcrPlan:
        candidates, filtered = self._candidates(request)
        targets = []
        for path, size in candidates:
            if check_cancel is not None:
                check_cancel()
            file_type = eligible_type(path)
            assert file_type is not None
            targets.append(self._preflight(path, file_type, size, request))
        targets.sort(key=lambda t: (_TYPE_NAMES[t.file_type], t.path))
        return OcrPlan(languages, request.force, tuple(targets), filtered)

    def _preflight(
        self, path: str, file_type: FileType, size: int, request: OcrRequest
    ) -> OcrTarget:
        """UC-6 step 4: registry, classification, encryption, readability."""
        target = OcrTarget(path, file_type, size, TO_PROCESS)
        try:
            content_hash = self.files.content_hash(path)
        except OSError:
            return replace(target, status=f"failed:{UNREADABLE}")
        target = replace(target, input_hash=content_hash)
        # An OCR output is never processed again, not even with force: that
        # would stack outputs (x.ocr.ocr.pdf).
        if self.registry.output_owner(path) is not None or (
            not request.force and self.registry.find_by_input_hash(content_hash)
        ):
            return replace(target, status=f"skipped:{ALREADY_PROCESSED}")
        entry = self.discovery.inspect(path, file_type, size)
        target = replace(
            target,
            page_count=entry.page_count,
            classification=entry.classification,
            reason=entry.reason,
        )
        if entry.classification == INSPECTION_FAILED:
            reason = (
                entry.reason
                if entry.reason in (ENCRYPTED, DJVU_TOOLS_MISSING)
                else UNREADABLE
            )
            return replace(target, status=f"failed:{reason}")
        if entry.classification == NATIVE and not request.override_native:
            return replace(target, status=f"skipped:{DOES_NOT_NEED_OCR}")
        mode = ocr_mode(entry.reason, override_native=request.override_native)
        return replace(target, mode=mode)

    # Running ---------------------------------------------------------------

    def run(
        self,
        request: OcrRequest,
        *,
        on_plan: Callable[[OcrPlan], None] | None = None,
        on_file: Callable[[OcrFileResult], None] | None = None,
        on_progress: ProgressObserver | None = None,
    ) -> OcrSummary:
        """Process the batch; raises before any OCR when it is refused.

        Refusals: ``ValidationError``, ``OcrToolsMissing`` and
        ``IndexLockHeld``.
        """
        started = self.clock()
        self._require_roots()
        languages = self._validate(request)
        version = self._check_tools(languages)
        with self.lock.hold("ocr"):
            self._close_dead_run()
            run_id = self.manifest.start_run(
                {
                    "languages": languages,
                    "force": int(request.force),
                    "override_native": int(request.override_native),
                    "cancel_mode": self.cancel_mode,
                    "timeout_seconds": self.timeout_seconds,
                    "jobs": self.jobs or 0,
                },
                kind="ocr",
            )
            _LOG.info("ocr_run_started", extra={"run_id": run_id})
            state = _Run(run_id, started)
            try:
                return self._execute(
                    state, request, languages, version, on_plan, on_file, on_progress
                )
            finally:
                self._cancel_event.clear()

    def _close_dead_run(self) -> None:
        """Mark an OCR run that lost its process as aborted; its outputs stay."""
        latest = self.manifest.latest_run()
        if latest is not None and latest.kind == "ocr" and latest.status == "running":
            self.manifest.finish_run(latest.run_id, "aborted", dict(latest.totals))
            _LOG.info("ocr_run_closed", extra={"run_id": latest.run_id})

    def _execute(
        self,
        state: _Run,
        request: OcrRequest,
        languages: str,
        version: str,
        on_plan: Callable[[OcrPlan], None] | None,
        on_file: Callable[[OcrFileResult], None] | None,
        on_progress: ProgressObserver | None = None,
    ) -> OcrSummary:
        plan = OcrPlan(languages, request.force, ())
        status = "completed"
        abort_reason: str | None = None
        try:
            plan = self._plan(request, languages, lambda: self._check_cancel(state))
            work = plan.to_process
            self.manifest.set_total_sources(state.run_id, len(work))
            self._log_plan(state.run_id, plan)
            if on_plan is not None:
                on_plan(plan)
            # Pages when every file of the batch has a page count, so that a
            # long file weighs more than a short one; files otherwise.
            by_pages = all(t.page_count for t in work)
            unit = "pages" if by_pages else "files"

            def weight(target: OcrTarget) -> int:
                return (target.page_count or 1) if by_pages else 1

            total_units = sum(weight(t) for t in work)
            loop_started = self.clock()

            def report(label: str | None, *, finished: bool = False) -> None:
                if on_progress is None or not work:
                    return
                elapsed = self.clock() - loop_started
                on_progress(
                    ProgressEvent(
                        operation="ocr",
                        unit=unit,
                        done=state.done_units,
                        total=total_units,
                        failed=sum(
                            r.status.startswith("failed:") for r in state.results
                        ),
                        elapsed_seconds=elapsed,
                        eta_seconds=estimate_eta_seconds(
                            done=state.done_units,
                            total=total_units,
                            elapsed_seconds=elapsed,
                        ),
                        label=label,
                        finished=finished,
                    )
                )

            report(None)
            for index, target in enumerate(work, start=1):
                self._check_cancel(state)
                report(item_label(target.path))
                result = self._process(
                    state, target, index, len(work), languages, version
                )
                state.done_units += weight(target)
                eta = estimate_eta_seconds(
                    done=state.done_units,
                    total=total_units,
                    elapsed_seconds=self.clock() - loop_started,
                )
                result = replace(result, eta_seconds=eta)
                state.results.append(result)
                self._journal(state.run_id, result)
                if on_file is not None:
                    on_file(result)
                if result.status == CANCELLED:
                    self._persist_cancel(state)
                    raise _Cancelled
                if result.status == f"failed:{DISK_FULL}":
                    raise _Abort
                report(item_label(target.path))
            self._check_cancel(state)
            report(None, finished=True)
        except _Cancelled:
            status = "cancelled"
        except _Abort:
            status = "aborted"
            abort_reason = DISK_FULL
        except Exception as error:
            self._finish(state, plan, "aborted")
            if isinstance(error, DocgrepError):
                raise
            raise BuildFailed(
                "OCR run aborted", details={"reason": str(error)}
            ) from error
        return self._finish(state, plan, status, abort_reason)

    def _finish(
        self,
        state: _Run,
        plan: OcrPlan,
        status: str,
        abort_reason: str | None = None,
    ) -> OcrSummary:
        summary = OcrSummary(
            run_id=state.run_id,
            plan=plan,
            results=tuple(state.results)
            + tuple(
                OcrFileResult(t, NOT_STARTED, i, len(plan.to_process))
                for i, t in enumerate(plan.to_process, start=1)
                if i > len(state.results)
            ),
            status=status,
            elapsed_seconds=self.clock() - state.started,
            abort_reason=abort_reason,
        )
        totals = summary.as_dict()["totals"]
        final: Literal["completed", "aborted", "cancelled"] = (
            "cancelled"
            if status == "cancelled"
            else ("aborted" if status == "aborted" else "completed")
        )
        try:
            self.manifest.finish_run(state.run_id, final, totals)
        except Exception:
            _LOG.error(
                "ocr_run_finish_failed", extra={"run_id": state.run_id}, exc_info=True
            )
        _LOG.info(
            "ocr_run_finished",
            extra={
                "run_id": state.run_id,
                "status": status,
                "abort_reason": abort_reason,
                **totals,
                "duration_ms": round(summary.elapsed_seconds * 1000, 1),
            },
        )
        return summary

    def _log_plan(self, run_id: str, plan: OcrPlan) -> None:
        counts = plan.as_dict()
        _LOG.info(
            "ocr_plan",
            extra={
                "run_id": run_id,
                "to_process": counts["to_process"],
                **{f"to_skip_{k}": v for k, v in counts["to_skip"].items()},
                **{f"to_fail_{k}": v for k, v in counts["to_fail"].items()},
                "filtered": counts["filtered"],
                "input_bytes": counts["input_bytes"],
                "pages": counts["pages"],
            },
        )

    def _journal(self, run_id: str, result: OcrFileResult) -> None:
        target = result.target
        try:
            self.manifest.record_source(
                Source(
                    source_id=target.path,
                    path=target.path,
                    file_type=target.file_type,
                    content_hash=target.input_hash,
                    size=target.size,
                ),
                result.status,
                result.message,
                run_id=run_id,
            )
        except Exception:
            # Only the progress report reads it; the registry is authoritative.
            _LOG.warning("ocr_journal_write_failed", extra={"run_id": run_id})

    # Cancel ----------------------------------------------------------------

    def _cancel_requested(self, state: _Run) -> bool:
        if state.cancel_seen or self._cancel_event.is_set():
            return True
        now = self.clock()
        if now - state.last_poll < self.poll_seconds:
            return False
        state.last_poll = now
        try:
            state.cancel_seen = (
                self.manifest.cancel_requested_at(state.run_id) is not None
            )
        except Exception:
            _LOG.debug("ocr_cancel_poll_failed", exc_info=True)
        return state.cancel_seen

    def _check_cancel(self, state: _Run) -> None:
        """Between files: stop when a cancel was requested here or elsewhere."""
        state.last_poll = float("-inf")
        if self._cancel_requested(state):
            self._persist_cancel(state)
            raise _Cancelled

    def _persist_cancel(self, state: _Run) -> None:
        """Record a cancel requested in this process, so the run shows it."""
        if not self._cancel_event.is_set():
            return
        try:
            self.manifest.request_cancel(state.run_id)
        except Exception:
            _LOG.debug("ocr_cancel_persist_failed", exc_info=True)

    def _stop_check(self, state: _Run) -> Callable[[], bool]:
        if self.cancel_mode == "finish":
            return lambda: False
        return lambda: self._cancel_requested(state)

    # One file --------------------------------------------------------------

    def _process(
        self,
        state: _Run,
        target: OcrTarget,
        index: int,
        total: int,
        languages: str,
        version: str,
    ) -> OcrFileResult:
        started = self.clock()
        final = output_path(target.path)
        assert target.mode is not None and target.input_hash is not None
        _LOG.info(
            "ocr_file_started",
            extra={
                "run_id": state.run_id,
                "path": target.path,
                "size_bytes": target.size,
                "pages": target.page_count,
                "mode": target.mode,
            },
        )

        def result(status: str, message: str | None = None) -> OcrFileResult:
            elapsed = self.clock() - started
            _LOG.info(
                "ocr_file_finished",
                extra={
                    "run_id": state.run_id,
                    "path": target.path,
                    "status": status,
                    "elapsed_seconds": round(elapsed, 1),
                },
            )
            return OcrFileResult(
                target,
                status,
                index,
                total,
                elapsed,
                output_path=final if status == PROCESSED else None,
                message=message,
            )

        temporaries: list[str] = []
        try:
            if not self._enough_space(final, target.size):
                return result(
                    f"failed:{DISK_FULL}",
                    f"less than {DISK_SPACE_FACTOR} times the input size is free",
                )
            owner = self.registry.output_owner(final)
            if self.files.exists(final) and owner != target.path:
                return result(
                    f"failed:{OUTPUT_COLLISION}", f"{final} exists and is kept"
                )
            deadline = (
                self.clock() + self.timeout_seconds if self.timeout_seconds else None
            )
            should_stop = self._stop_check(state)
            source = target.path
            if target.file_type is FileType.DJVU:
                converted = self.files.temporary(final, "djvu")
                temporaries.append(converted)
                self.converter.convert(
                    source, converted, deadline=deadline, should_stop=should_stop
                )
                source = converted
            produced = self.files.temporary(final, "ocr")
            temporaries.append(produced)
            self.engine.run(
                source,
                produced,
                languages=languages,
                mode=target.mode,
                jobs=self.jobs,
                deadline=deadline,
                should_stop=should_stop,
            )
            problem = self._invalid_output(produced, target)
            if problem is not None:
                return result(f"failed:{OUTPUT_INVALID}", problem)
            try:
                self.files.publish(produced, final, replace=owner == target.path)
            except FileExistsError:
                return result(
                    f"failed:{OUTPUT_COLLISION}", f"{final} exists and is kept"
                )
            temporaries.remove(produced)
            return self._register(target, final, languages, version, result)
        except OcrFailed as error:
            if error.reason == CANCELLED:
                return result(CANCELLED)
            return result(f"failed:{error.reason}", str(error))
        except OSError as error:
            if error.errno == errno.ENOSPC:
                return result(f"failed:{DISK_FULL}", "no space left on device")
            return result(f"failed:{UNREADABLE}", type(error).__name__)
        finally:
            for path in temporaries:
                try:
                    self.files.remove(path)
                except OSError:
                    _LOG.warning("ocr_temporary_kept", extra={"path": path})

    def _enough_space(self, final: str, size: int) -> bool:
        try:
            return self.files.free_bytes(final) >= DISK_SPACE_FACTOR * size
        except OSError:
            _LOG.debug("ocr_free_space_unknown", exc_info=True)
            return True

    def _invalid_output(self, produced: str, target: OcrTarget) -> str | None:
        try:
            pages = self.discovery.probe.page_count(produced, FileType.PDF)
        except InspectionFailed as error:
            return f"the output cannot be opened ({error.details.get('reason')})"
        if target.page_count is not None and pages != target.page_count:
            return f"the output has {pages} pages, the input {target.page_count}"
        return None

    def _register(
        self,
        target: OcrTarget,
        final: str,
        languages: str,
        version: str,
        result: Callable[[str, str | None], OcrFileResult],
    ) -> OcrFileResult:
        assert target.input_hash is not None and target.mode is not None
        try:
            record = OcrRecord(
                input_path=target.path,
                input_format="djvu" if target.file_type is FileType.DJVU else "pdf",
                input_hash=target.input_hash,
                output_path=final,
                output_hash=self.files.content_hash(final),
                engine=self.engine.name,
                engine_version=version,
                languages=languages,
                mode=target.mode,
                created_at=self.now().isoformat(),
            )
            self.registry.register(record)
        except Exception:
            # The output is correct and stays; the next run processes the
            # file again because nothing records it (UC-6 ext. 6.6a).
            _LOG.error(
                "ocr_registry_write_failed",
                extra={"path": target.path, "output_path": final},
                exc_info=True,
            )
            return result(
                f"failed:{REGISTRY_WRITE}",
                f"{final} was written but could not be registered",
            )
        return result(PROCESSED, None)
