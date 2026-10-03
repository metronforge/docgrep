"""Ports implemented by adapters at the domain boundary."""

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Literal, Protocol

from docgrep.domain.models import (
    ManifestRunRecord,
    ManifestSourceRecord,
    ParsedDocument,
    Passage,
    RankedPassage,
    RetrievalCandidate,
    Source,
)


class DocumentParser(Protocol):
    """Parse a source into verbatim, location-aware sections or raise an error."""

    def parse(self, source: Source) -> ParsedDocument: ...


class TokenSpanTokenizer(Protocol):
    """Return ordered character spans for countable tokens in exact input text."""

    def token_spans(self, text: str) -> tuple[tuple[int, int], ...]: ...


class Chunker(Protocol):
    """Split parsed sections into deterministic, location-aware verbatim passages."""

    def chunk(self, document: ParsedDocument) -> tuple[Passage, ...]: ...


class CorpusEnumerator(Protocol):
    """Return source records in deterministic order, including unsupported files."""

    def enumerate(self) -> tuple[Source, ...]: ...


class EmbeddingProvider(Protocol):
    """Embed a batch in input order with a fixed vector dimension."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

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

    def delete_source_points(self, source_id: str) -> int:
        """Delete every passage of a source from the collection being built, or
        from the published collection when none is open; return how many."""
        ...


class IndexLock(Protocol):
    """Guard the exclusive index write lock while a build runs.

    Entering the context manager returned by ``hold`` acquires the lock without
    waiting. It raises ``IndexLockHeld`` when another operation owns the lock
    and ``StoreUnavailable`` when the lock cannot be accessed. Exceptions raised
    inside the managed block propagate unchanged, and the lock is released.
    """

    def hold(self) -> AbstractContextManager[None]: ...


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


class ManifestReader(Protocol):
    """Read committed source outcomes and the most recent indexing run."""

    def get_source(self, path: str) -> ManifestSourceRecord | None: ...

    def latest_run(self) -> ManifestRunRecord | None: ...

    def run_sources(self, run_id: str) -> tuple[ManifestSourceRecord, ...]:
        """The journal of one run: every source outcome it recorded, by path."""
        ...


class ManifestWriter(Protocol):
    """Record an indexing run and per-source outcomes durably at each call."""

    def start_run(self, config: Mapping[str, str | int | float]) -> str: ...

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
        updates the committed sources directly.
        """
        ...

    def set_staging_collection(self, run_id: str, name: str) -> None:
        """Remember which collection a run is writing into."""
        ...

    def commit_run(self, run_id: str) -> None:
        """Replace the committed sources with the run's journal, atomically."""
        ...

    def finish_run(
        self,
        run_id: str,
        status: Literal["completed", "aborted"],
        totals: Mapping[str, int],
    ) -> None: ...


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
