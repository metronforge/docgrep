"""Immutable values shared by indexing and retrieval."""

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from typing import Literal


class FileType(StrEnum):
    """File classifications used by the corpus pipeline."""

    TXT = "txt"
    MARKDOWN = "markdown"
    PDF = "pdf"
    EPUB = "epub"
    DOCX = "docx"
    DJVU = "djvu"
    OCR_PDF = "ocr_pdf"
    UNSUPPORTED = "unsupported"


def make_passage_id(source_id: str, chunk_offset: int) -> str:
    """Derive a stable passage identifier from its source and character offset."""
    if not source_id:
        raise ValueError("source_id must not be empty")
    if chunk_offset < 0:
        raise ValueError("chunk offset must be non-negative")
    payload = source_id.encode("utf-8") + b"\x00" + str(chunk_offset).encode("ascii")
    return sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class Source:
    """A document and its known source metadata."""

    source_id: str
    path: str
    file_type: FileType
    title: str | None = None
    author: str | None = None
    year: int | None = None
    content_hash: str | None = None
    mtime: float | None = None
    size: int | None = None
    indexed_at: str | None = None
    parser_version: str | None = None

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must not be empty")
        if not self.path:
            raise ValueError("path must not be empty")
        if self.size is not None and self.size < 0:
            raise ValueError("size must be non-negative")
        if self.mtime is not None and not isfinite(self.mtime):
            raise ValueError("mtime must be finite")


@dataclass(frozen=True, slots=True)
class Passage:
    """An exact text span from a source document."""

    passage_id: str
    source_id: str
    text: str
    chunk_offset: int
    token_count: int
    page_number: int | None = None
    heading_path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.passage_id:
            raise ValueError("passage_id must not be empty")
        if not self.source_id:
            raise ValueError("source_id must not be empty")
        if not self.text:
            raise ValueError("passage text must not be empty")
        if self.chunk_offset < 0:
            raise ValueError("chunk offset must be non-negative")
        if self.token_count <= 0:
            raise ValueError("token_count must be positive")
        if self.page_number is not None and self.page_number <= 0:
            raise ValueError("page_number must be positive")


@dataclass(frozen=True, slots=True)
class RankedPassage:
    """A passage with source metadata and retrieval scores."""

    passage: Passage
    source: Source
    score: float
    dense_score: float
    reranker_score: float | None = None

    def __post_init__(self) -> None:
        if self.passage.source_id != self.source.source_id:
            raise ValueError("passage source_id must match source")
        if not isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("score must be finite and within [0, 1]")
        if not isfinite(self.dense_score):
            raise ValueError("dense_score must be finite")
        if self.reranker_score is not None and not isfinite(self.reranker_score):
            raise ValueError("reranker_score must be finite")


@dataclass(frozen=True, slots=True)
class Query:
    """A user's free-text search request."""

    text: str
    top_n: int | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("query text must not be empty")
        if self.top_n is not None and self.top_n <= 0:
            raise ValueError("top_n must be positive")


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Ranked findings and metadata describing the retrieval."""

    results: tuple[RankedPassage, ...]
    total_hits: int
    took_ms: float
    applied_config: tuple[tuple[str, str | int | float | bool], ...] = ()
    rerank_skipped: bool = False
    rerank_skip_reason: str | None = None
    hint: str | None = None

    def __post_init__(self) -> None:
        if self.total_hits < len(self.results):
            raise ValueError("total_hits must include all returned results")
        if not isfinite(self.took_ms) or self.took_ms < 0:
            raise ValueError("took_ms must be finite and non-negative")
        if self.rerank_skip_reason is not None and not self.rerank_skipped:
            raise ValueError("rerank_skip_reason requires rerank_skipped")


@dataclass(frozen=True, slots=True)
class ParsedSection:
    """Verbatim text and location produced by a document parser."""

    text: str
    page_number: int | None = None
    heading_path: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Parser output before token-aware chunking."""

    source: Source
    sections: tuple[ParsedSection, ...]
    requires_ocr: bool = False
    metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    """An unnormalized first-stage hit with its dense similarity."""

    passage: Passage
    source: Source
    dense_score: float


@dataclass(frozen=True, slots=True)
class ManifestSourceRecord:
    """A source and its most recent indexing outcome."""

    source: Source
    status: str
    error: str | None = None
    embedding_model: str | None = None


@dataclass(frozen=True, slots=True)
class ManifestRunRecord:
    """A single indexing run and its committed counters."""

    run_id: str
    status: Literal["running", "completed", "aborted"]
    config: tuple[tuple[str, str | int | float], ...]
    totals: tuple[tuple[str, int], ...] = ()
    kind: str = "build"
    started_at: str | None = None
    finished_at: str | None = None
