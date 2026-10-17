"""Build an index through domain ports while preserving per-source outcomes."""

import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TypeVar

from docgrep.domain.errors import (
    BuildFailed,
    ConfigDrift,
    CorpusDrift,
    DocgrepError,
    NothingToResume,
    ResumeAvailable,
    StagingInconsistent,
    StoreUnavailable,
    ValidationError,
)
from docgrep.domain.models import (
    DELEGATED_TO_OCR,
    FileType,
    ManifestRunRecord,
    ManifestSourceRecord,
    Passage,
    Source,
)
from docgrep.domain.progress import estimate_eta_seconds
from docgrep.domain.protocols import (
    Chunker,
    CorpusAvailability,
    CorpusEnumerator,
    DocumentParser,
    EmbeddingProvider,
    IndexLock,
    Manifest,
    OcrDelegations,
    PassageWriter,
    RootCatalog,
    SparseEncoder,
    SparseEncoders,
    SparseVectorStore,
    parser_version,
)
from docgrep.domain.roots import (
    RUN_PREFIXES_KEY,
    RUN_ROOTS_KEY,
    CorpusRoots,
    prefixes_value,
)
from docgrep.domain.run_state import (
    RESUMABLE_STATES,
    VECTOR_CONFIG_KEYS,
    config_drift,
    latest_build_run,
    latest_root_prefixes,
    roots_drift,
    run_state,
    swap_already_happened,
)
from docgrep.domain.sparse import (
    DEFAULT_B,
    DEFAULT_K1,
    PROVISIONAL_AVGDL,
    SparseWeighting,
    check_bm25_parameters,
)
from docgrep.usecases.sparse_vectors import average_length, encode_pages

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
    cancelled: bool = False
    # Set when the build continued an earlier run: how many of the sources it
    # covers were taken over, processed now (of which retried after an earlier
    # failure) and dropped because they vanished from the corpus.
    resumed_from: str | None = None
    carried_over: int = 0
    processed: int = 0
    retried: int = 0
    removed: int = 0
    # Set when the run only finished the commit of an interrupted swap.
    recovered: bool = False


# Share of the sources indexed so far that may change or vanish before a
# resume is refused as a different corpus.
CORPUS_DRIFT_PERCENT = 5


@dataclass(frozen=True, slots=True)
class _Plan:
    sources: tuple[Source, ...]
    total: int
    carried: int = 0
    totals: Mapping[str, int] | None = None
    failures: tuple[BuildFailure, ...] = ()
    retried: int = 0
    removed: int = 0


class Cancelled(Exception):
    """Control flow: a cancel was requested and the current batch is done."""


PassageItem = tuple[Passage, Source, tuple[float, ...], Mapping[int, float]]


@dataclass(frozen=True, slots=True)
class Prepared:
    """A source run through parsing, chunking and embedding, before any write.

    ``status`` is ``indexed`` (``items`` are ready to write), ``skipped:<reason>``
    or ``failed:<reason>`` (``error`` says why).
    """

    source: Source
    status: str
    items: tuple[PassageItem, ...] = ()
    error: Exception | None = None
    # Why a source was skipped, when the parser said more than the status.
    detail: str | None = None


class Progress:
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
        carried: int = 0,
    ) -> None:
        self._carried = carried
        self._run_id = run_id
        self._total = total
        self._clock = clock
        self._started = started
        self._every_sources = every_sources
        self._every_seconds = every_seconds
        self._last_logged_at = started
        self._logged_done: int | None = None
        self.done = carried
        self.passages = 0

    def advance(self, passages: int, failed: int, *, done: int = 1) -> None:
        """Count ``done`` more units (sources; passages for a re-encode)."""
        self.done += done
        self.passages += passages
        since = self.done - (
            self._carried if self._logged_done is None else self._logged_done
        )
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
            done=self.done,
            total=self._total,
            elapsed_seconds=elapsed,
            carried=self._carried,
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


class SourceIndexing:
    """The per-source pipeline shared by full builds and incremental updates.

    Subclasses provide the parsers, chunker, embedder, sparse encoder, manifest,
    batch size and sleep as attributes, read at call time.
    """

    parsers: Mapping[FileType, DocumentParser]
    chunker: Chunker
    embedder: EmbeddingProvider
    sparse_encoder: SparseEncoder
    manifest: Manifest
    batch_size: int
    sleep: Callable[[float], None]
    _cancel_event: threading.Event
    _cancel_persisted: bool
    # Originals indexed through their OCR'd sibling; None: no OCR registry.
    delegations: OcrDelegations | None = None
    # The roots of the run (ADR-0023); None: one unnamed root, as before.
    roots: RootCatalog | None = None
    availability: CorpusAvailability | None = None
    # The document encoder of the weighting this run writes (ADR-0025),
    # chosen when the run starts; None: ``sparse_encoder``.
    _run_sparse_encoder: SparseEncoder | None = None

    def document_sparse_encoder(self) -> SparseEncoder:
        """The encoder that new passages of this run are written with."""
        return self._run_sparse_encoder or self.sparse_encoder

    def run_roots(self) -> CorpusRoots | None:
        """The roots of the run, knowing every prefix the index records.

        Checks first that every root can be read, so that a missing root
        fails the run before any work.
        """
        if self.availability is not None:
            self.availability.require_available()
        return self.known_roots()

    def known_roots(self) -> CorpusRoots | None:
        """The roots of the run with the prefixes the index records; no root
        is read."""
        if self.roots is None:
            return None
        return self.roots.roots().with_recorded(latest_root_prefixes(self.manifest))

    @staticmethod
    def roots_config(
        roots: CorpusRoots | None, prefixes: frozenset[str] | None = None
    ) -> dict[str, str]:
        """The run configuration keys of the roots: the roots visited and the
        prefixes the index may hold afterwards."""
        if roots is None:
            return {}
        known = roots.prefixed_names if prefixes is None else prefixes
        return {
            RUN_ROOTS_KEY: roots.snapshot(),
            RUN_PREFIXES_KEY: prefixes_value(known),
        }

    def is_delegated(self, source: Source) -> bool:
        """Whether a PDF or DjVu source has a live delegation to its OCR'd
        sibling, for its current content."""
        if self.delegations is None or source.file_type not in (
            FileType.PDF,
            FileType.DJVU,
        ):
            return False
        found = self.delegations.live_delegation(source.path, source.content_hash)
        return found is not None

    def request_cancel(self) -> None:
        """Ask the running build to stop after its current batch.

        Safe to call from a signal handler: it only sets a flag. The build
        persists the request itself, so other processes see it too.
        """
        self._cancel_event.set()

    def _check_cancel(self, run_id: str) -> None:
        if self._cancel_event.is_set():
            if not self._cancel_persisted:
                self.manifest.request_cancel(run_id)
                self._cancel_persisted = True
            raise Cancelled
        if self.manifest.cancel_requested_at(run_id) is not None:
            raise Cancelled

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

    def _prepare(self, source: Source, run_id: str) -> Prepared:
        """Parse, chunk and embed one source; checks for a cancel between batches.

        A parser that declares a version stamps it on the source it returns.
        A source delegated to its OCR'd sibling is skipped without parsing.
        """
        if self.is_delegated(source):
            return Prepared(source, f"skipped:{DELEGATED_TO_OCR}")
        parser = self.parsers.get(source.file_type)
        if parser is None:
            return Prepared(source, "skipped:unsupported")
        try:
            document = parser.parse(source)
            parsed = document.source
            version = parser_version(parser)
            if version is not None:
                parsed = replace(parsed, parser_version=version)
            if document.requires_ocr:
                reason = document.quality.skip_reason if document.quality else None
                return Prepared(parsed, "skipped:requires_ocr", detail=reason)
            passages = self.chunker.chunk(document)
        except Exception as error:
            return Prepared(source, "failed:parse_error", error=error)

        all_vectors: list[tuple[float, ...]] = []
        for start in range(0, len(passages), self.batch_size):
            if start:
                # Between batches of one source; nothing of it is journaled yet.
                self._check_cancel(run_id)
            batch = passages[start : start + self.batch_size]
            try:
                vectors = self._retry(
                    lambda: self.embedder.embed([passage.text for passage in batch])
                )
                if len(vectors) != len(batch):
                    raise ValueError("embedder returned wrong vector count")
            except Exception as error:
                return Prepared(parsed, "failed:embedding_error", error=error)
            all_vectors.extend(vectors)
        sparse = self.document_sparse_encoder()
        items = tuple(
            (passage, parsed, vector, sparse.encode(passage.text))
            for passage, vector in zip(passages, all_vectors, strict=True)
        )
        return Prepared(parsed, "indexed", items)


def finish_swapped_run(
    manifest: Manifest, passage_writer: PassageWriter
) -> ManifestRunRecord | None:
    """Finish a build that published its index but did not commit its sources.

    Called with the lock held, so such a run is dead. It is either stored as
    swapped or, when the process died between the alias swap and the write
    of that status, still running or aborted with its staging collection
    already published. Only builds swap, so only the latest build is checked.
    """
    latest = latest_build_run(manifest)
    if latest is None:
        return None
    reason = "swapped"
    if latest.status != "swapped":
        if latest.status not in ("running", "aborted"):
            return None
        if latest.staging_collection is None:
            return None
        info = passage_writer.staging_info(latest.staging_collection)
        if info is None or not swap_already_happened(
            latest,
            journal_rows=len(manifest.run_sources(latest.run_id)),
            staging_published=info.published,
        ):
            return None
        manifest.mark_swapped(latest.run_id)
        reason = "alias_target_reached"
    totals = {"indexed": 0, "skipped": 0, "failed": 0}
    for record in manifest.run_sources(latest.run_id):
        totals[record.status.split(":")[0]] += 1
    manifest.commit_run(latest.run_id)
    manifest.finish_run(latest.run_id, "completed", totals)
    _LOG.warning(
        "run_recovered",
        extra={
            "run_id": latest.run_id,
            "previous_status": latest.status,
            "reason": reason,
            **totals,
        },
    )
    return latest


class BuildIndex(SourceIndexing):
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
        refuse_resumable_by_default: bool = False,
        progress_every_sources: int = 25,
        progress_every_seconds: float = 60.0,
        log_context: Mapping[str, str | int | float] | None = None,
        delegations: OcrDelegations | None = None,
        roots: RootCatalog | None = None,
        availability: CorpusAvailability | None = None,
        sparse_encoders: SparseEncoders | None = None,
        sparse_store: SparseVectorStore | None = None,
        bm25_k1: float = DEFAULT_K1,
        bm25_b: float = DEFAULT_B,
    ) -> None:
        """With ``sparse_encoders`` and ``sparse_store`` the index gets
        ``bm25-v2`` sparse vectors (ADR-0025); without them every passage
        is written with ``sparse_encoder`` and no weighting is recorded."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if upsert_batch_size <= 0:
            raise ValueError("upsert_batch_size must be positive")
        if progress_every_sources <= 0:
            raise ValueError("progress_every_sources must be positive")
        if progress_every_seconds <= 0:
            raise ValueError("progress_every_seconds must be positive")
        check_bm25_parameters(bm25_k1, bm25_b)
        self.sparse_encoders = sparse_encoders
        self.sparse_store = sparse_store
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.delegations = delegations
        self.roots = roots
        self.availability = availability
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
        self.refuse_resumable_by_default = refuse_resumable_by_default
        self.progress_every_sources = progress_every_sources
        self.progress_every_seconds = progress_every_seconds
        self.log_context = dict(log_context or {})
        self._cancel_event = threading.Event()
        self._cancel_persisted = False

    def _finish_swapped_run(self) -> ManifestRunRecord | None:
        """Finish the latest build if it only lacks its commit; see the function."""
        return finish_swapped_run(self.manifest, self.passage_writer)

    def _writes_bm25(self) -> bool:
        return self.sparse_encoders is not None and self.sparse_store is not None

    def _choose_sparse_encoder(self) -> None:
        """Passages are written with BM25 weights under a provisional average
        length while the sources stream in; ``_final_sparse_weighting``
        computes the true one from the staging collection and re-encodes
        every passage before the index is published."""
        self._run_sparse_encoder = None
        if self.sparse_encoders is not None and self._writes_bm25():
            self._run_sparse_encoder = self.sparse_encoders.document_encoder(
                SparseWeighting.bm25(self.bm25_k1, self.bm25_b, PROVISIONAL_AVGDL)
            )

    def _final_sparse_weighting(self, run_id: str) -> SparseWeighting | None:
        """Re-encode the whole staging collection with its own average
        length; None when this build records no weighting.

        Every passage is encoded again, those carried over from an earlier
        run included, so a resumed build never mixes average lengths. A
        cancel is honoured between pages; the staging collection then stays
        resumable.
        """
        encoders, store = self.sparse_encoders, self.sparse_store
        if encoders is None or store is None:
            return None

        def check() -> None:
            self._check_cancel(run_id)

        started = self.clock()
        avgdl, passages = average_length(store, encoders, between_pages=check)
        weighting = SparseWeighting.bm25(self.bm25_k1, self.bm25_b, avgdl)
        encode_pages(store, encoders.document_encoder(weighting), before_page=check)
        _LOG.info(
            "sparse_vectors_encoded",
            extra={
                "run_id": run_id,
                "passages": passages,
                "bm25_avgdl": round(avgdl, 3),
                "elapsed_seconds": round(self.clock() - started, 1),
            },
        )
        return weighting

    def _drop_orphaned_staging_collections(self) -> None:
        for name in self.passage_writer.staging_collections():
            self.passage_writer.drop_collection(name)
            _LOG.info("orphaned_staging_collection_dropped", extra={"collection": name})

    def _index_source(
        self, source: Source, run_id: str
    ) -> tuple[str, BuildFailure | None, int]:
        prepared = self._prepare(source, run_id)
        if prepared.status.startswith("skipped:"):
            self.manifest.record_source(
                prepared.source, prepared.status, prepared.detail, run_id=run_id
            )
            return "skipped", None, 0
        if prepared.error is not None:
            reason = prepared.status.split(":", 1)[1]
            return (
                "failed",
                self._record_failure(prepared.source, reason, prepared.error, run_id),
                0,
            )
        items = prepared.items
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
            prepared.source,
            "indexed",
            embedding_model=self.embedder.model_name,
            run_id=run_id,
            passages=len(items),
        )
        return "indexed", None, len(items)

    @property
    def run_config(self) -> dict[str, str | int | float]:
        """The configuration recorded on a run: the given keys plus the model."""
        return {
            **self.config,
            "embedding_model": self.embedder.model_name,
            "embedding_dimension": self.embedder.dimension,
        }

    def _check_resumable_latest(self) -> ManifestRunRecord:
        """The latest run if it can be resumed, called with the lock held."""
        latest = latest_build_run(self.manifest)
        if latest is None:
            raise NothingToResume("There is no run to resume")
        # Nobody else holds the lock, so a run still stored as running is dead.
        state = run_state(latest, lock_held=False)
        if state not in RESUMABLE_STATES:
            raise NothingToResume(
                "The latest run is not resumable",
                details={"run_id": latest.run_id, "state": state},
            )
        return latest

    def _refuse_if_resumable(self) -> None:
        latest = latest_build_run(self.manifest)
        if latest is None:
            return
        state = run_state(latest, lock_held=False)
        if state in RESUMABLE_STATES:
            raise ResumeAvailable(
                "The latest run can be resumed; use --resume to continue it "
                "or --restart to discard it and build from scratch",
                details={"run_id": latest.run_id, "state": state},
            )

    def run(self, *, refuse_resumable: bool | None = None) -> BuildSummary:
        """Run a full rebuild, marking completion only after all sources finish.

        With ``refuse_resumable`` a latest run that could be resumed is not
        discarded: the call raises ``ResumeAvailable`` instead. Left unset, the
        instance default applies.
        """
        if refuse_resumable is None:
            refuse_resumable = self.refuse_resumable_by_default
        started = self.clock()
        with self.lock.hold():
            self._finish_swapped_run()
            if refuse_resumable:
                self._refuse_if_resumable()
            roots = self.run_roots()
            self._refuse_dropping_other_roots(roots)
            self._choose_sparse_encoder()
            run_id = self.manifest.start_run(
                {**self.run_config, **self.roots_config(roots)}
            )
            _LOG.info(
                "index_build_started", extra={"run_id": run_id, **self.log_context}
            )

            def prepare(run_id: str) -> _Plan:
                self._drop_orphaned_staging_collections()
                sources = self.enumerator.enumerate()
                self.manifest.set_total_sources(run_id, len(sources))
                self.passage_writer.recreate_collection(self.embedder.dimension)
                staging = self.passage_writer.staging_name()
                if staging is not None:
                    self.manifest.set_staging_collection(run_id, staging)
                return _Plan(sources, len(sources))

            return self._execute(run_id, started, prepare)

    def _refuse_dropping_other_roots(self, roots: CorpusRoots | None) -> None:
        """A full build replaces the index with the sources of its roots, so
        it refuses while the index holds sources of any other root."""
        if roots is None:
            return
        others: dict[str, int] = {}
        for record in self.manifest.committed_sources():
            name = roots.root_of(record.source.source_id)
            if roots.get(name) is None:
                others[name] = others.get(name, 0) + 1
        if not others:
            return
        listed = ", ".join(
            f"{name} ({count})" for name, count in sorted(others.items())
        )
        raise ValidationError(
            "The index holds sources of roots that this build does not visit: "
            f"{listed}. A full build would drop them. Include those roots, add "
            "the new roots with `docgrep update` instead, or remove their sources "
            "first with `docgrep forget-root NAME`",
            details={"roots": dict(sorted(others.items()))},
        )

    @staticmethod
    def _totals(records: Sequence[ManifestSourceRecord]) -> dict[str, int]:
        totals = {"indexed": 0, "skipped": 0, "failed": 0}
        for record in records:
            totals[record.status.split(":")[0]] += 1
        return totals

    def resume(self, *, retry_failed: bool = True) -> BuildSummary:
        """Continue the latest aborted, cancelled or interrupted run.

        The run is validated before anything is written: the vector-affecting
        configuration must be unchanged, the staging collection must be intact,
        and the corpus must not have drifted much. A swapped run is only
        finished. Raises ``NothingToResume``, ``ConfigDrift``,
        ``StagingInconsistent`` or ``CorpusDrift`` otherwise.
        """
        started = self.clock()
        with self.lock.hold():
            recovered = self._finish_swapped_run()
            if recovered is not None:
                return self._recovered_summary(recovered, started)
            latest = self._check_resumable_latest()
            roots = self.run_roots()
            if roots is not None and roots_drift(
                dict(latest.config),
                roots.snapshot(),
                single_legacy=len(roots) == 1 and roots.legacy is not None,
            ):
                raise ConfigDrift(
                    "The corpus roots differ from the roots the run started "
                    "with; resume with the same roots (the same --root "
                    "parameters or saved roots), or use --restart",
                    details={
                        "run_id": latest.run_id,
                        "fields": RUN_ROOTS_KEY,
                        "recorded": str(dict(latest.config).get(RUN_ROOTS_KEY)),
                        "runtime": roots.snapshot(),
                    },
                )
            runtime = {**self.run_config, **self.roots_config(roots)}
            drifted = config_drift(dict(latest.config), runtime)
            if drifted:
                raise ConfigDrift(
                    "The configuration changed since the run; resuming would mix "
                    "passages made with different settings. Use --restart",
                    details={
                        "run_id": latest.run_id,
                        "fields": ", ".join(drifted),
                        "checked": ", ".join(VECTOR_CONFIG_KEYS),
                    },
                )
            journal = self.manifest.run_sources(latest.run_id)
            self._check_staging(latest, journal)
            self._choose_sparse_encoder()
            sources = self.enumerator.enumerate()
            carried, retried, removed = self._split_journal(
                journal, sources, retry_failed
            )
            kept_paths = {record.source.path for record in carried}
            plan_sources = tuple(s for s in sources if s.path not in kept_paths)
            run_id = self.manifest.start_run(runtime, resumed_from=latest.run_id)
            _LOG.info(
                "index_build_resumed",
                extra={
                    "run_id": run_id,
                    "resumed_from": latest.run_id,
                    **self.log_context,
                },
            )

            def prepare(run_id: str) -> _Plan:
                assert latest.staging_collection is not None
                name = latest.staging_collection
                self.passage_writer.adopt_staging(name, self.embedder.dimension)
                self.manifest.set_staging_collection(run_id, name)
                # Anything staged for a source that is not carried over (changed,
                # vanished, or half written when the run stopped) is removed;
                # processed sources are written again from scratch.
                keep = {r.source.source_id for r in carried if r.status == "indexed"}
                stale = self.passage_writer.staged_source_ids() - keep
                for source_id in sorted(stale):
                    self.passage_writer.delete_source_points(source_id)
                for other in self.passage_writer.staging_collections():
                    if other != name:
                        self.passage_writer.drop_collection(other)
                        _LOG.info(
                            "orphaned_staging_collection_dropped",
                            extra={"collection": other},
                        )
                self.manifest.carry_over(latest.run_id, run_id, kept_paths)
                self.manifest.set_total_sources(
                    run_id, len(sources), carried=len(carried)
                )
                return _Plan(
                    plan_sources,
                    len(sources),
                    carried=len(carried),
                    totals=self._totals(carried),
                    failures=tuple(
                        BuildFailure(
                            r.source.path, r.status.split(":", 1)[1], r.error or ""
                        )
                        for r in carried
                        if r.status.startswith("failed:")
                    ),
                    retried=retried,
                    removed=removed,
                )

            return self._execute(run_id, started, prepare, resumed_from=latest.run_id)

    def _check_staging(
        self, latest: ManifestRunRecord, journal: Sequence[ManifestSourceRecord]
    ) -> None:
        name = latest.staging_collection
        details: dict[str, object] = {"run_id": latest.run_id}
        if name is None:
            raise StagingInconsistent(
                "The run recorded no staging collection. Use --restart",
                details=details,
            )
        details["collection"] = name
        info = self.passage_writer.staging_info(name)
        if info is None:
            raise StagingInconsistent(
                "The staging collection of the run no longer exists. Use --restart",
                details=details,
            )
        if info.published:
            raise StagingInconsistent(
                "The recorded staging collection is the published index, so the "
                "run is not resumable. Check the state with `docgrep status` and "
                "start a fresh build with --restart",
                details=details,
            )
        if info.dimension != self.embedder.dimension:
            raise StagingInconsistent(
                "The staging collection has a different vector dimension. "
                "Use --restart",
                details={
                    **details,
                    "expected": self.embedder.dimension,
                    "found": info.dimension if info.dimension is not None else "none",
                },
            )
        journaled = sum(r.passages or 0 for r in journal if r.status == "indexed")
        if info.passages < journaled:
            raise StagingInconsistent(
                "The staging collection holds fewer passages than the run "
                "recorded. Use --restart",
                details={**details, "journaled": journaled, "found": info.passages},
            )

    def _split_journal(
        self,
        journal: Sequence[ManifestSourceRecord],
        sources: Sequence[Source],
        retry_failed: bool,
    ) -> tuple[list[ManifestSourceRecord], int, int]:
        """Decide which journal rows stay; returns them with retried and removed."""
        current = {source.path: source for source in sources}
        indexed = 0
        changed = 0
        carried: list[ManifestSourceRecord] = []
        retried = 0
        removed = 0
        for record in journal:
            now = current.get(record.source.path)
            gone = now is None
            differs = now is not None and now.content_hash != record.source.content_hash
            if record.status == "indexed":
                indexed += 1
                changed += gone or differs
            if gone:
                removed += 1
            elif differs:
                continue
            elif record.status.startswith("failed:") and retry_failed:
                retried += 1
            else:
                carried.append(record)
        if indexed and changed * 100 > CORPUS_DRIFT_PERCENT * indexed:
            raise CorpusDrift(
                "More than "
                f"{CORPUS_DRIFT_PERCENT}% of the sources indexed so far changed "
                "or vanished. Use --restart",
                details={
                    "indexed": indexed,
                    "changed_or_missing": changed,
                    "threshold_percent": CORPUS_DRIFT_PERCENT,
                },
            )
        return carried, retried, removed

    def _recovered_summary(
        self, run: ManifestRunRecord, started: float
    ) -> BuildSummary:
        records = self.manifest.run_sources(run.run_id)
        totals = self._totals(records)
        failures = tuple(
            BuildFailure(r.source.path, r.status.split(":", 1)[1], r.error or "")
            for r in records
            if r.status.startswith("failed:")
        )
        return BuildSummary(
            run.run_id,
            totals["indexed"],
            totals["skipped"],
            totals["failed"],
            self.clock() - started,
            failures,
            recovered=True,
        )

    def _execute(
        self,
        run_id: str,
        started: float,
        prepare: Callable[[str], _Plan],
        *,
        resumed_from: str | None = None,
    ) -> BuildSummary:
        """Index the planned sources, then publish; the lock is already held."""
        totals = {"indexed": 0, "skipped": 0, "failed": 0}
        failures: list[BuildFailure] = []
        plan = _Plan((), 0)
        swapped = False
        cancelled = False
        try:
            plan = prepare(run_id)
            totals.update(plan.totals or {})
            failures.extend(plan.failures)
            progress = Progress(
                run_id=run_id,
                total=plan.total,
                clock=self.clock,
                started=started,
                every_sources=self.progress_every_sources,
                every_seconds=self.progress_every_seconds,
                carried=plan.carried,
            )
            for source in plan.sources:
                self._check_cancel(run_id)
                outcome, failure, written = self._index_source(source, run_id)
                totals[outcome] += 1
                if failure is not None:
                    failures.append(failure)
                progress.advance(written, totals["failed"])
            progress.finish(totals["failed"])
            self._check_cancel(run_id)
            weighting = self._final_sparse_weighting(run_id)
            metadata = dict(self.config)
            if weighting is not None:
                metadata.update(weighting.metadata())
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
        except Cancelled:
            self.manifest.finish_run(run_id, "cancelled", totals)
            cancelled = True
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
        finally:
            self._cancel_event.clear()
            self._cancel_persisted = False
        elapsed = self.clock() - started
        _LOG.info(
            "index_build_cancelled" if cancelled else "index_build_finished",
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
            cancelled=cancelled,
            resumed_from=resumed_from,
            carried_over=plan.carried,
            processed=len(plan.sources),
            retried=plan.retried,
            removed=plan.removed,
        )
