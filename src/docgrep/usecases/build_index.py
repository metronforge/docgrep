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
from docgrep.domain.models import FileType, Source
from docgrep.domain.protocols import (
    Chunker,
    CorpusEnumerator,
    DocumentParser,
    EmbeddingProvider,
    IndexLock,
    ManifestWriter,
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
        manifest: ManifestWriter,
        config: Mapping[str, str | int | float],
        docgrep_version: str,
        batch_size: int = 32,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
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
        self.sleep = sleep

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
        self, source: Source, reason: str, error: Exception
    ) -> BuildFailure:
        self.manifest.record_source(source, f"failed:{reason}", str(error))
        _LOG.warning(
            "index_source_failed", extra={"path": source.path, "reason": reason}
        )
        return BuildFailure(source.path, reason, str(error))

    def _index_source(self, source: Source) -> tuple[str, BuildFailure | None]:
        parser = self.parsers.get(source.file_type)
        if parser is None:
            self.manifest.record_source(source, "skipped:unsupported")
            return "skipped", None
        try:
            document = parser.parse(source)
            if document.requires_ocr:
                self.manifest.record_source(document.source, "skipped:requires_ocr")
                return "skipped", None
            passages = self.chunker.chunk(document)
        except Exception as error:
            return "failed", self._record_failure(source, "parse_error", error)

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
                return "failed", self._record_failure(
                    document.source, "embedding_error", error
                )
            all_vectors.extend(vectors)
        for passage, vector in zip(passages, all_vectors, strict=True):
            sparse = self.sparse_encoder.encode(passage.text)
            try:
                self._retry(
                    lambda: self.passage_writer.upsert(
                        passage, document.source, vector, sparse
                    )
                )
            except Exception as error:
                raise StoreUnavailable(
                    "Passage upsert failed",
                    details={"path": source.path, "reason": str(error)},
                ) from error
        self.manifest.record_source(
            document.source, "indexed", embedding_model=self.embedder.model_name
        )
        return "indexed", None

    def run(self) -> BuildSummary:
        """Run a full rebuild, marking completion only after all sources finish."""
        started = time.monotonic()
        totals = {"indexed": 0, "skipped": 0, "failed": 0}
        failures: list[BuildFailure] = []
        with self.lock.hold():
            run_id = self.manifest.start_run(self.config)
            replacement_started = False
            try:
                sources = self.enumerator.enumerate()
                replacement_started = True
                self.passage_writer.recreate_collection(self.embedder.dimension)
                for source in sources:
                    outcome, failure = self._index_source(source)
                    totals[outcome] += 1
                    if failure is not None:
                        failures.append(failure)
                metadata = dict(self.config)
                metadata.update(
                    embedding_model=self.embedder.model_name,
                    embedding_dimension=self.embedder.dimension,
                    docgrep_version=self.docgrep_version,
                    completed_at=datetime.now(UTC).isoformat(),
                )
                self.passage_writer.write_completion_metadata(metadata)
                self.manifest.finish_run(run_id, "completed", totals)
            except Exception as error:
                if replacement_started:
                    self.passage_writer.clear_completion_metadata()
                self.manifest.finish_run(run_id, "aborted", totals)
                if isinstance(error, DocgrepError):
                    raise
                raise BuildFailed(
                    "Index build aborted", details={"reason": str(error)}
                ) from error
        return BuildSummary(
            run_id,
            totals["indexed"],
            totals["skipped"],
            totals["failed"],
            time.monotonic() - started,
            tuple(failures),
        )
