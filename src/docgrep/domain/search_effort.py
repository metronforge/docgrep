"""Defaults for the size and the search effort of the first stage (ADR-0026)."""

from typing import Literal

# Fused first-stage candidates the reranker scores when DOCGREP_RERANK_POOL
# is unset; never more than the union of the two channels (2 * top_k).
DEFAULT_RERANK_POOL = 150

# Search effort of the dense channel: an hnsw_ef value, or "exact".
DenseSearchEffort = int | Literal["exact"]
DEFAULT_DENSE_HNSW_EF = 1024
MAX_DENSE_HNSW_EF = 16384


def default_rerank_pool(top_k: int) -> int:
    """The rerank pool used when none is configured."""
    return min(DEFAULT_RERANK_POOL, 2 * top_k)


def parse_dense_effort(value: str) -> DenseSearchEffort:
    """An hnsw_ef value or "exact" from text; raises ValueError otherwise."""
    text = value.strip().lower()
    if text == "exact":
        return "exact"
    if not text.isdigit():
        raise ValueError(f"must be an integer or exact, not {value!r}")
    number = int(text)
    if not 1 <= number <= MAX_DENSE_HNSW_EF:
        raise ValueError(f"must be between 1 and {MAX_DENSE_HNSW_EF} or exact")
    return number
