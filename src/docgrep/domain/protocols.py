"""Ports implemented by adapters at the domain boundary."""

from collections.abc import Mapping, Sequence
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


class Reranker(Protocol):
    """Score each query-passage pair in input order; larger is more relevant."""

    @property
    def model_name(self) -> str: ...

    def score(self, query: str, passages: Sequence[str]) -> tuple[float, ...]: ...


class PassageRetriever(Protocol):
    """Read hybrid recall hits and the index completion marker without mutation."""

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


class ManifestWriter(Protocol):
    """Record an indexing run and per-source outcomes durably at each call."""

    def start_run(self, config: Mapping[str, str | int | float]) -> str: ...

    def record_source(
        self,
        source: Source,
        status: str,
        error: str | None = None,
        embedding_model: str | None = None,
    ) -> None: ...

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
