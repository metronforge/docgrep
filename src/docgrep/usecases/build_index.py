"""Build an index through domain ports while preserving per-source outcomes."""

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypeVar

from docgrep.domain.errors import (
    BuildFailed,
    DocgrepError,
    StoreUnavailable,
)
from docgrep.domain.models import FileType, ManifestRunRecord, Source
from docgrep.domain.progress import estimate_eta_seconds
from docgrep.domain.protocols import (
    Chunker,
    CorpusEnumerator,
    DocumentParser,
    EmbeddingProvider,
    IndexLock,
    Manifest,
    PassageWriter,
    SparseEncoder,
)

_LOG = logging.getLogger(__name__)
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class BuildFailure:
    """A source that could not be indexed and its failure category."""

    path: str
    reason: str
    message: str


@dataclass(frozen=True, slots=True)
class BuildSummary:
    """Final counts, timing, and source failures for one build."""

    run_id: str
    indexed: int
    skipped: int
    failed: int
    elapsed_seconds: float
    failures: tuple[BuildFailure, ...]


class _Progress:
    """Emit ``index_progress`` records on a count and a time cadence.

    Only counters and timings are logged, never paths, titles or text.
    """

    def __init__(
        self,
        *,
        run_id: str,
        total: int,
        clock: Callable[[], float],
        started: float,
        every_sources: int,
        every_seconds: float,
    ) -> None:
        self._run_id = run_id
        self._total = total
        self._clock = clock
        self._started = started
        self._every_sources = every_sources
        self._every_seconds = every_seconds
        self._last_logged_at = started
        self._logged_done: int | None = None
        self.done = 0
        self.passages = 0

    def advance(self, passages: int, failed: int) -> None:
        self.done += 1
        self.passages += passages
        since = self.done - (self._logged_done or 0)
        waited = self._clock() - self._last_logged_at
        if since >= self._every_sources or waited >= self._every_seconds:
            self._emit(failed)

    def finish(self, failed: int) -> None:
        """Log once at the end of the loop, unless the last record says the same."""
        if self._logged_done != self.done:
            self._emit(failed)

    def _emit(self, failed: int) -> None:
        now = self._clock()
        elapsed = now - self._started
        eta = estimate_eta_seconds(
            done=self.done, total=self._total, elapsed_seconds=elapsed
        )
        _LOG.info(
            "index_progress",
            extra={
                "run_id": self._run_id,
                "sources_done": self.done,
                "sources_total": self._total,
                "passages_written": self.passages,
                "elapsed_seconds": round(elapsed, 1),
                "passages_per_second": (
                    round(self.passages / elapsed, 2) if elapsed > 0 else 0.0
                ),
                "eta_seconds": None if eta is None else round(eta),
                "failed": failed,
            },
        )
        self._last_logged_at = now
        self._logged_done = self.done


class BuildIndex:
    """Rebuild a collection with an exclusive lock and durable source outcomes."""

    def __init__(
        self,
        *,
        lock: IndexLock,
        enumerator: CorpusEnumerator,
        parsers: Mapping[FileType, DocumentParser],
        chunker: Chunker,
        embedder: EmbeddingProvider,
        sparse_encoder: SparseEncoder,
        passage_writer: PassageWriter,
        manifest: Manifest,
        config: Mapping[str, str | int | float],
        docgrep_version: str,
        batch_size: int = 32,
        upsert_batch_size: int = 256,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
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
        self.enumerator = enumerator
        self.parsers = parsers
        self.chunker = chunker
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        self.passage_writer = passage_writer
        self.manifest = manifest
        self.config = dict(config)
        self.docgrep_version = docgrep_version
        self.batch_size = batch_size
        self.upsert_batch_size = upsert_batch_size
        self.sleep = sleep
        self.clock = clock
        self.progress_every_sources = progress_every_sources
        self.progress_every_seconds = progress_every_seconds
        self.log_context = dict(log_context or {})

    def _retry(self, operation: Callable[[], _T]) -> _T:
        for attempt in range(3):
            try:
                return operation()
            except Exception:
                if attempt == 2:
                    raise
                self.sleep(float(2**attempt))
        raise AssertionError("retry loop exhausted")

    def _record_failure(
        self, source: Source, reason: str, error: Exception, run_id: str
    ) -> BuildFailure:
        self.manifest.record_source(
            source, f"failed:{reason}", str(error), run_id=run_id
        )
        _LOG.warning(
            "index_source_failed",
            extra={"source_id": source.source_id, "category": reason},
        )
        return BuildFailure(source.path, reason, str(error))

    def _finish_swapped_run(self) -> ManifestRunRecord | None:
        """Finish a run whose process died after publishing but before committing.

        Called with the lock held, so a run that is still swapped is dead.
        """
        latest = self.manifest.latest_run()
        if latest is None or latest.status != "swapped":
            return None
        totals = {"indexed": 0, "skipped": 0, "failed": 0}
        for record in self.manifest.run_sources(latest.run_id):
            totals[record.status.split(":")[0]] += 1
        self.manifest.commit_run(latest.run_id)
        self.manifest.finish_run(latest.run_id, "completed", totals)
        _LOG.warning(
            "run_recovered",
            extra={"run_id": latest.run_id, "previous_status": "swapped", **totals},
        )
        return latest

    def _drop_orphaned_staging_collections(self) -> None:
        for name in self.passage_writer.staging_collections():
            self.passage_writer.drop_collection(name)
            _LOG.info("orphaned_staging_collection_dropped", extra={"collection": name})

    def _index_source(
        self, source: Source, run_id: str
    ) -> tuple[str, BuildFailure | None, int]:
        parser = self.parsers.get(source.file_type)
        if parser is None:
            self.manifest.record_source(source, "skipped:unsupported", run_id=run_id)
            return "skipped", None, 0
        try:
            document = parser.parse(source)
            if document.requires_ocr:
                self.manifest.record_source(
                    document.source, "skipped:requires_ocr", run_id=run_id
                )
                return "skipped", None, 0
            passages = self.chunker.chunk(document)
        except Exception as error:
            return (
                "failed",
                self._record_failure(source, "parse_error", error, run_id),
                0,
            )

        all_vectors: list[tuple[float, ...]] = []
        for start in range(0, len(passages), self.batch_size):
            batch = passages[start : start + self.batch_size]
            try:
                vectors = self._retry(
                    lambda: self.embedder.embed([passage.text for passage in batch])
                )
                if len(vectors) != len(batch):
                    raise ValueError("embedder returned wrong vector count")
            except Exception as error:
                return (
                    "failed",
                    self._record_failure(
                        document.source, "embedding_error", error, run_id
                    ),
                    0,
                )
            all_vectors.extend(vectors)
        items = [
            (passage, document.source, vector, self.sparse_encoder.encode(passage.text))
            for passage, vector in zip(passages, all_vectors, strict=True)
        ]
        for start in range(0, len(items), self.upsert_batch_size):
            group = items[start : start + self.upsert_batch_size]
            try:
                self._retry(lambda: self.passage_writer.upsert_batch(group))
            except Exception as error:
                raise StoreUnavailable(
                    "Passage upsert failed",
                    details={"path": source.path, "reason": str(error)},
                ) from error
        self.manifest.record_source(
            document.source,
            "indexed",
            embedding_model=self.embedder.model_name,
            run_id=run_id,
            passages=len(passages),
        )
        return "indexed", None, len(passages)

    def run(self) -> BuildSummary:
        """Run a full rebuild, marking completion only after all sources finish."""
        started = self.clock()
        totals = {"indexed": 0, "skipped": 0, "failed": 0}
        failures: list[BuildFailure] = []
        with self.lock.hold():
            self._finish_swapped_run()
            run_id = self.manifest.start_run(self.config)
            _LOG.info(
                "index_build_started", extra={"run_id": run_id, **self.log_context}
            )
            swapped = False
            try:
                self._drop_orphaned_staging_collections()
                sources = self.enumerator.enumerate()
                self.manifest.set_total_sources(run_id, len(sources))
                self.passage_writer.recreate_collection(self.embedder.dimension)
                staging = self.passage_writer.staging_name()
                if staging is not None:
                    self.manifest.set_staging_collection(run_id, staging)
                progress = _Progress(
                    run_id=run_id,
                    total=len(sources),
                    clock=self.clock,
                    started=started,
                    every_sources=self.progress_every_sources,
                    every_seconds=self.progress_every_seconds,
                )
                for source in sources:
                    outcome, failure, written = self._index_source(source, run_id)
                    totals[outcome] += 1
                    if failure is not None:
                        failures.append(failure)
                    progress.advance(written, totals["failed"])
                progress.finish(totals["failed"])
                metadata = dict(self.config)
                metadata.update(
                    embedding_model=self.embedder.model_name,
                    embedding_dimension=self.embedder.dimension,
                    # Information only: vectors of either precision are compatible.
                    embedding_dtype=self.embedder.dtype,
                    docgrep_version=self.docgrep_version,
                    completed_at=datetime.now(UTC).isoformat(),
                )
                self.passage_writer.write_completion_metadata(metadata)
                # The alias now points at the new index; only then do the
                # committed sources follow, so queries never see a description
                # of an index that is not published yet. A process that dies
                # in between leaves the run swapped, and the next start
                # finishes it.
                self.manifest.mark_swapped(run_id)
                swapped = True
                self.manifest.commit_run(run_id)
                self.manifest.finish_run(run_id, "completed", totals)
            except Exception as error:
                # Before the swap the staging collection and the journal stay,
                # so the run can be resumed; the live index and the committed
                # sources are untouched. After the swap the live index is
                # complete and valid, and the run stays swapped for recovery.
                if not swapped:
                    self.manifest.finish_run(run_id, "aborted", totals)
                if isinstance(error, DocgrepError):
                    raise
                raise BuildFailed(
                    "Index build aborted", details={"reason": str(error)}
                ) from error
        elapsed = self.clock() - started
        _LOG.info(
            "index_build_finished",
            extra={
                "run_id": run_id,
                "indexed": totals["indexed"],
                "skipped": totals["skipped"],
                "failed": totals["failed"],
                "duration_ms": round(elapsed * 1000, 1),
            },
        )
        return BuildSummary(
            run_id,
            totals["indexed"],
            totals["skipped"],
            totals["failed"],
            elapsed,
            tuple(failures),
        )
