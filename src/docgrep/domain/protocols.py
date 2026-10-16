"""Ports implemented by adapters at the domain boundary."""

from collections.abc import Callable, Collection, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Literal, Protocol

from docgrep.domain.models import (
    FileType,
    ManifestRunRecord,
    ManifestSourceRecord,
    OcrDelegation,
    OcrMode,
    OcrRecord,
    ParsedDocument,
    Passage,
    RankedPassage,
    RetrievalCandidate,
    ScannedFile,
    Source,
    StagingInfo,
)
from docgrep.domain.scans import PageSample


class DocumentParser(Protocol):
    """Parse a source into verbatim, location-aware sections or raise an error.

    A parser may also carry a ``version`` string attribute; it is optional and
    read through ``parser_version``.
    """

    def parse(self, source: Source) -> ParsedDocument: ...


def parser_version(parser: DocumentParser | None) -> str | None:
    """The version a parser declares in an optional ``version`` attribute.

    A parser without one, or no parser, gives None, which is also what a
    source records when nothing was declared.
    """
    version = getattr(parser, "version", None)
    return version if isinstance(version, str) else None


# Sources indexed before parsers declared versions recorded none. They count
# as version "1", the first declared version of every built-in parser, so an
# update does not reprocess them merely because nothing was recorded.
LEGACY_PARSER_VERSION = "1"


def effective_parser_version(version: str | None) -> str:
    """A recorded or declared parser version, with missing or empty read as
    ``LEGACY_PARSER_VERSION``."""
    return version or LEGACY_PARSER_VERSION


def parser_reprocess_reason(
    status: str,
    recorded_version: str | None,
    *,
    registered: bool,
    declared_version: str | None,
) -> Literal["newly_supported", "parser_changed"] | None:
    """Why the registered parser would treat a recorded source differently.

    A source skipped as unsupported is newly supported once a parser for its
    type is registered; an indexed or OCR-pending source has a changed parser
    when the declared version differs from the recorded one or no parser is
    registered any more. A missing or empty version, recorded or declared,
    counts as version "1" (``effective_parser_version``), so sources recorded
    before versions existed are not reprocessed for that alone. Failed sources
    are left to the retry rule, sources delegated to an OCR'd sibling to the
    delegation rule.
    """
    if status == "skipped:unsupported":
        return "newly_supported" if registered else None
    if status.startswith("failed:") or status == "skipped:delegated_to_ocr":
        # Delegated sources follow their delegation, not their parser.
        return None
    if not registered:
        return "parser_changed"
    declared = effective_parser_version(declared_version)
    recorded = effective_parser_version(recorded_version)
    return None if declared == recorded else "parser_changed"


class TokenSpanTokenizer(Protocol):
    """Return ordered character spans for countable tokens in exact input text."""

    def token_spans(self, text: str) -> tuple[tuple[int, int], ...]: ...


class Chunker(Protocol):
    """Split parsed sections into deterministic, location-aware verbatim passages."""

    def chunk(self, document: ParsedDocument) -> tuple[Passage, ...]: ...


class CorpusEnumerator(Protocol):
    """Return source records in deterministic order, including unsupported files."""

    def enumerate(self) -> tuple[Source, ...]: ...


class CorpusScanner(Protocol):
    """List every corpus file from directory entries alone, in path order.

    Nothing is read or hashed. Files that the include and exclude patterns keep
    out of a build are included and carry the pattern responsible.
    """

    def scan(self) -> tuple[ScannedFile, ...]: ...


class CorpusReader(Protocol):
    """Scan the corpus cheaply, then describe single files on demand."""

    def scan(self) -> tuple[ScannedFile, ...]:
        """Every corpus file from directory entries alone, as ``CorpusScanner``."""
        ...

    def describe(self, path: str) -> Source:
        """The source record of one corpus file, with its content hash, exactly
        as ``CorpusEnumerator.enumerate`` would give it."""
        ...


class ScanProbe(Protocol):
    """Look at the text layer of a PDF or DjVu file without reading its text
    into any result: page count and per-page character counts only.

    Paths are relative to the corpus root. Both calls raise
    ``InspectionFailed`` (with ``details["reason"]``) when the file as a whole
    cannot be read; nothing is written and no other error is expected.
    """

    def page_count(self, path: str, file_type: FileType) -> int: ...

    def sample(
        self, path: str, file_type: FileType, pages: Sequence[int]
    ) -> tuple[PageSample | None, ...]:
        """One entry per requested page, None for a page that failed."""
        ...


class EmbeddingProvider(Protocol):
    """Embed a batch in input order with a fixed vector dimension."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def dtype(self) -> str:
        """Precision the model computes in; information only, never a contract."""
        ...

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...


class SparseEncoder(Protocol):
    """Encode document or query text into stable sparse term weights."""

    def encode(self, text: str) -> Mapping[int, float]: ...


class PassageWriter(Protocol):
    """Replace an index, persist passages, and mark a build complete last."""

    def recreate_collection(self, dimension: int) -> None: ...

    def upsert(
        self,
        passage: Passage,
        source: Source,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float],
    ) -> None: ...

    def upsert_batch(
        self,
        items: Sequence[tuple[Passage, Source, Sequence[float], Mapping[int, float]]],
    ) -> None:
        """Write several passages at once with the same result as one upsert each."""
        ...

    def write_completion_metadata(
        self, metadata: Mapping[str, str | int | float]
    ) -> None: ...

    def clear_completion_metadata(self) -> None: ...

    def staging_name(self) -> str | None:
        """Name of the collection being built, or None when no build is open."""
        ...

    def staging_collections(self) -> tuple[str, ...]:
        """Unpublished collections that carry this store's staging name pattern.

        The published collection and collections of other stores are never listed.
        """
        ...

    def drop_collection(self, name: str) -> None:
        """Remove one listed staging collection; refuse any other name."""
        ...

    def staging_info(self, name: str) -> StagingInfo | None:
        """Facts about an existing collection with this store's staging name
        pattern (flagged when it is the published one), or None when absent."""
        ...

    def adopt_staging(self, name: str, dimension: int) -> None:
        """Continue writing into an existing unpublished staging collection.

        Raises ``StagingInconsistent`` when it is missing, published, or its
        vector dimension differs.
        """
        ...

    def staged_source_ids(self) -> frozenset[str]:
        """Source ids that have passages in the collection being built."""
        ...

    def delete_source_points(self, source_id: str) -> int:
        """Delete every passage of a source from the collection being built, or
        from the published collection when none is open; return how many."""
        ...


class LiveIndexWriter(Protocol):
    """Change the published collection in place, one source at a time.

    Every call works on the collection the read alias points at, never on a
    staging collection, and raises ``StoreUnavailable`` when none is published.
    """

    def read_completion_metadata(self) -> Mapping[str, str | int | float] | None: ...

    def upsert_live_batch(
        self,
        items: Sequence[tuple[Passage, Source, Sequence[float], Mapping[int, float]]],
    ) -> None:
        """Write passages into the published collection; durable on return."""
        ...

    def delete_live_source_points(self, source_id: str) -> int:
        """Delete every passage of a source; return how many were removed."""
        ...

    def delete_live_source_points_except(
        self, source_id: str, keep: Collection[str]
    ) -> int:
        """Delete the passages of a source whose passage ids are not in ``keep``;
        return how many were removed."""
        ...

    def refresh_live_source(self, source_id: str, *, size: int, mtime: float) -> None:
        """Set the recorded size and modification time on every passage of a
        source whose content did not change."""
        ...

    def merge_completion_metadata(
        self, updates: Mapping[str, str | int | float]
    ) -> None:
        """Add or replace keys of the completion marker, keeping the others.

        Refuses keys that describe vectors or chunking, which only a full build
        may set.
        """
        ...


class IndexLock(Protocol):
    """Guard the exclusive index write lock while a build runs.

    Entering the context manager returned by ``hold`` acquires the lock without
    waiting. It raises ``IndexLockHeld`` when another operation owns the lock
    and ``StoreUnavailable`` when the lock cannot be accessed. Exceptions raised
    inside the managed block propagate unchanged, and the lock is released.
    ``operation`` names what wants the lock (``build`` or ``update``) for the
    refusal message.
    """

    def hold(self, operation: str = "build") -> AbstractContextManager[None]: ...

    def is_held(self) -> bool:
        """Whether the lock is owned right now, by anyone including this process.

        A non-blocking try that never keeps the lock beyond the probe. Callers
        must not ask while they hold the lock themselves.
        """
        ...


class Reranker(Protocol):
    """Score each query-passage pair in input order; larger is more relevant."""

    @property
    def model_name(self) -> str: ...

    def score(self, query: str, passages: Sequence[str]) -> tuple[float, ...]: ...


class PassageRetriever(Protocol):
    """Read hybrid recall hits and the index completion marker without mutation.

    ``retrieve`` returns candidates in first-stage order, best first. Each
    candidate carries its true dense similarity, including hits that only the
    sparse query found, and may carry its position as ``stage1_rank``.
    """

    def retrieve(
        self,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float],
        top_k: int,
    ) -> tuple[RetrievalCandidate, ...]: ...

    def read_completion_metadata(self) -> Mapping[str, str | int | float] | None: ...

    def count_passages(self) -> int: ...


class StatusStore(PassageRetriever, Protocol):
    """A short-timeout read view of the vector store for status reports."""

    def staging_info(self, name: str) -> StagingInfo | None: ...

    def close(self) -> None: ...


class SourcePassageCounter(Protocol):
    """Count the passages of one source in the published index, read-only."""

    def count_source_passages(self, source_id: str) -> int:
        """The exact number of passages of the source; 0 when it has none.

        Raises ``IndexIncomplete`` when no index is published and
        ``StoreUnavailable`` when the store cannot be read.
        """
        ...


class CommittedSourceReader(Protocol):
    """Read the committed source outcomes of the manifest."""

    def committed_sources(self) -> tuple[ManifestSourceRecord, ...]:
        """Every committed source outcome, in path order."""
        ...


class ManifestReader(CommittedSourceReader, Protocol):
    """Read committed source outcomes and the most recent indexing run."""

    def get_source(self, path: str) -> ManifestSourceRecord | None: ...

    def latest_run(self) -> ManifestRunRecord | None: ...

    def latest_run_of_kind(self, kind: str) -> ManifestRunRecord | None:
        """The most recent run of one kind (``build`` or ``update``), or None."""
        ...

    def run_sources(self, run_id: str) -> tuple[ManifestSourceRecord, ...]:
        """The journal of one run: every source outcome it recorded, by path."""
        ...

    def cancel_requested_at(self, run_id: str) -> str | None:
        """When a cancel was requested for the run, or None."""
        ...


class ManifestWriter(Protocol):
    """Record an indexing run and per-source outcomes durably at each call."""

    def start_run(
        self,
        config: Mapping[str, str | int | float],
        *,
        resumed_from: str | None = None,
        kind: str = "build",
    ) -> str:
        """Open a run of a kind; ``resumed_from`` names the run it continues."""
        ...

    def remove_source(self, path: str) -> None:
        """Delete the committed outcome of a source; nothing happens when absent."""
        ...

    def record_source(
        self,
        source: Source,
        status: str,
        error: str | None = None,
        embedding_model: str | None = None,
        *,
        run_id: str | None = None,
        passages: int = 0,
    ) -> None:
        """Record a source outcome.

        With a run id the outcome goes to that run's journal only; without one it
        updates the committed sources directly. ``status`` is ``indexed``,
        ``skipped:<reason>``, ``failed:<reason>`` or, in an update's journal,
        ``removed``.
        """
        ...

    def set_staging_collection(self, run_id: str, name: str) -> None:
        """Remember which collection a run is writing into."""
        ...

    def set_total_sources(self, run_id: str, total: int, *, carried: int = 0) -> None:
        """Remember how many sources the run plans to cover and how many of
        them it took over, already done, from the run it resumes."""
        ...

    def carry_over(
        self, from_run_id: str, to_run_id: str, paths: Collection[str]
    ) -> None:
        """Copy the journal rows of the given paths from one run into another."""
        ...

    def request_cancel(self, run_id: str) -> str:
        """Persist a cancel request and return when it was made.

        Idempotent: a second request returns the time of the first.
        """
        ...

    def mark_swapped(self, run_id: str) -> None:
        """Record that the run's index is published and its journal is pending.

        Accepts a running run and, for recovery, an aborted one.
        """
        ...

    def commit_run(self, run_id: str) -> None:
        """Replace the committed sources with the run's journal, atomically."""
        ...

    def finish_run(
        self,
        run_id: str,
        status: Literal["completed", "aborted", "cancelled"],
        totals: Mapping[str, int],
    ) -> None:
        """Close a running or swapped run with a final status."""
        ...


class Manifest(ManifestReader, ManifestWriter, Protocol):
    """Both sides of the manifest, for operations that read what they wrote."""


class OcrDelegations(Protocol):
    """Which originals are indexed through their OCR'd sibling.

    A delegation is live while its output file exists and the original's
    content hash equals the recorded one. Reads take no lock and never raise:
    a missing or unreadable database reads as no delegation.
    """

    def live_delegation(
        self, path: str, content_hash: str | None = None
    ) -> OcrDelegation | None:
        """The live delegation of the original at ``path``, or None.

        ``content_hash`` is the original's current content hash when the
        caller knows it; otherwise it is computed, and only when a delegation
        is recorded for the path.
        """
        ...


class OcrOutputOwners(Protocol):
    """Which file an OCR output was made from; reads never raise.

    A registry entry is live while its output file exists.
    """

    def output_owner(self, output_path: str) -> str | None:
        """The input path of the live entry whose output is ``output_path``."""
        ...


class OcrRegistryReader(OcrDelegations, OcrOutputOwners, Protocol):
    """Read the OCR registry; the same read rules as ``OcrDelegations``.

    A registry entry is live while its output file exists.
    """

    def find_by_input_hash(self, content_hash: str) -> OcrRecord | None:
        """A live entry for an input with this content, or None; the first by
        input path when several inputs have the same content."""
        ...


class OcrRegistry(OcrRegistryReader, Protocol):
    """The OCR registry with its single write."""

    def register(self, record: OcrRecord) -> None:
        """Record a processed file and delegate its original to the output, in
        one transaction.

        Replaces the entry and the delegation recorded for the same input
        path, so registering again (``--force``) never duplicates a row.
        Raises ``ManifestUnavailable`` when the database cannot be written.
        """
        ...


# Whether a running OCR step should stop now (a cancel in terminate mode).
StopCheck = Callable[[], bool]


class OcrEngine(Protocol):
    """Make a searchable PDF from a PDF whose pages are images.

    Paths are relative to the corpus root. ``deadline`` is a monotonic time
    after which the run is stopped, None for no limit; ``should_stop`` is
    polled while the engine runs. Failures raise ``OcrFailed`` with the
    reason ``ocr_engine_error``, ``timeout``, ``disk_full``, ``encrypted`` or
    ``cancelled``; the target may then be partly written.
    """

    @property
    def name(self) -> str: ...

    def version(self) -> str:
        """The engine version; raises ``OcrToolsMissing`` when not installed."""
        ...

    def languages(self) -> tuple[str, ...]:
        """The installed language codes; raises ``OcrToolsMissing`` when the
        engine or its language listing is not installed."""
        ...

    def run(
        self,
        source: str,
        target: str,
        *,
        languages: str,
        mode: OcrMode,
        jobs: int | None,
        deadline: float | None,
        should_stop: StopCheck,
    ) -> None: ...


class DjvuPdfConverter(Protocol):
    """Convert a DjVu file into a PDF of page images, without text.

    The same path, deadline and stop rules as ``OcrEngine.run``. Failures
    raise ``OcrFailed`` with the reason ``djvu_conversion``,
    ``djvu_tools_missing``, ``timeout``, ``disk_full`` or ``cancelled``.
    """

    def convert(
        self,
        source: str,
        target: str,
        *,
        deadline: float | None,
        should_stop: StopCheck,
    ) -> None: ...


class OcrFiles(Protocol):
    """The files OCR reads and writes, by path relative to the corpus root."""

    def locate(self, path: str) -> str:
        """The normalized relative path of a request path (relative to the
        corpus root, or absolute). Raises ``ValidationError`` with
        ``details["reason"]`` ``outside_corpus`` when it resolves outside the
        corpus root, or ``not_found`` when it is not an existing file."""
        ...

    def exists(self, path: str) -> bool: ...

    def size(self, path: str) -> int: ...

    def content_hash(self, path: str) -> str: ...

    def free_bytes(self, path: str) -> int:
        """Free space in the directory that holds ``path``."""
        ...

    def temporary(self, near: str, label: str) -> str:
        """A new, unused path in the directory of ``near`` whose name cannot be
        taken for a document (hidden, without a document extension)."""
        ...

    def publish(self, temporary: str, final: str, *, replace: bool) -> None:
        """Rename atomically; without ``replace`` an existing ``final`` is
        never overwritten and ``FileExistsError`` is raised."""
        ...

    def remove(self, path: str) -> None:
        """Delete a file if it exists."""
        ...


class CitationFormatter(Protocol):
    """Format a source citation in the requested style using known metadata only."""

    def format(self, source: Source, style: str) -> str: ...


def ranked_passage_key(item: RankedPassage) -> tuple[float, float, str]:
    """Return the deterministic rerank, dense, passage-ID ordering key."""
    return (
        -(item.reranker_score if item.reranker_score is not None else item.dense_score),
        -item.dense_score,
        item.passage.passage_id,
    )
