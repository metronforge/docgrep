"""Read and rank verbatim passages through domain ports."""

import logging
import math
import time

from docgrep.domain.duplicates import collapse_duplicates as collapse
from docgrep.domain.errors import (
    ConfigDrift,
    IndexIncomplete,
    ModelUnavailable,
    ValidationError,
)
from docgrep.domain.models import RankedPassage, RetrievalCandidate, SearchResult
from docgrep.domain.protocols import (
    EmbeddingProvider,
    PassageRetriever,
    Reranker,
    SparseEncoder,
    ranked_passage_key,
)

_LOG = logging.getLogger(__name__)


def _normalized_score(value: float) -> float:
    """Map an arbitrary finite model score monotonically into [0, 1]."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _valid_candidate(candidate: RetrievalCandidate) -> bool:
    source = candidate.source
    passage = candidate.passage
    return (
        bool(source.source_id)
        and bool(source.path)
        and "\x00" not in source.path
        and passage.source_id == source.source_id
        and bool(passage.passage_id)
        and bool(passage.text)
        and math.isfinite(candidate.dense_score)
    )


class FindPassages:
    """Validate, recall, rerank, and return exact indexed passage text."""

    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        sparse_encoder: SparseEncoder,
        retriever: PassageRetriever,
        reranker: Reranker,
        top_k: int,
        default_top_n: int,
        top_n_max: int,
        max_query_len: int,
        collapse_duplicates: bool = True,
    ) -> None:
        if top_k <= 0 or not 0 < default_top_n <= top_n_max:
            raise ValueError("result limits must be positive")
        if max_query_len <= 0:
            raise ValueError("max_query_len must be positive")
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        self.retriever = retriever
        self.reranker = reranker
        self.top_k = top_k
        self.default_top_n = default_top_n
        self.top_n_max = top_n_max
        self.max_query_len = max_query_len
        self.collapse_duplicates = collapse_duplicates

    def run(
        self,
        query: str,
        *,
        top_n: int | None = None,
        collapse_duplicates: bool | None = None,
    ) -> SearchResult:
        """Find passages, keeping the indexed text unchanged in every result.

        ``collapse_duplicates`` overrides the configured default when given.
        """
        started = time.monotonic()
        collapsing = (
            self.collapse_duplicates
            if collapse_duplicates is None
            else collapse_duplicates
        )
        if not query.strip():
            raise ValidationError("query is required")
        if len(query) > self.max_query_len:
            raise ValidationError(
                f"query must be at most {self.max_query_len} characters",
                details={"max_query_len": self.max_query_len},
            )
        limit = self.default_top_n if top_n is None else top_n
        if not 1 <= limit <= self.top_n_max:
            raise ValidationError(
                f"top_n must be between 1 and {self.top_n_max}",
                details={"minimum": 1, "maximum": self.top_n_max},
            )

        metadata = self.retriever.read_completion_metadata()
        if metadata is None or not isinstance(metadata.get("embedding_model"), str):
            raise IndexIncomplete("Index has no valid completion metadata")
        indexed_model = metadata["embedding_model"]
        if indexed_model != self.embedder.model_name:
            raise ConfigDrift(
                "Index embedding model differs from runtime model; rebuild the index",
                details={
                    "indexed_model": indexed_model,
                    "configured_model": self.embedder.model_name,
                },
            )
        indexed_dimension = metadata.get("embedding_dimension")
        if (
            indexed_dimension is not None
            and indexed_dimension != self.embedder.dimension
        ):
            raise ConfigDrift(
                "Index embedding dimension differs from runtime; rebuild the index",
                details={
                    "indexed_dimension": indexed_dimension,
                    "configured_dimension": self.embedder.dimension,
                },
            )
        applied_config: tuple[tuple[str, str | int | float | bool], ...] = (
            ("embedding_model", self.embedder.model_name),
            ("max_query_len", self.max_query_len),
            ("reranker_model", self.reranker.model_name),
            ("top_k", self.top_k),
            ("top_n", limit),
        )

        def result(
            passages: tuple[RankedPassage, ...],
            total_hits: int,
            *,
            rerank_skipped: bool = False,
            rerank_skip_reason: str | None = None,
            hint: str | None = None,
            collapsed_duplicates: int = 0,
        ) -> SearchResult:
            outcome = SearchResult(
                passages,
                total_hits,
                (time.monotonic() - started) * 1000,
                applied_config,
                rerank_skipped,
                rerank_skip_reason,
                hint,
                collapsed_duplicates if collapsing else None,
            )
            # Counts and timings only: neither the query nor passage text is logged.
            _LOG.info(
                "search_completed",
                extra={
                    "top_n": limit,
                    "total_hits": outcome.total_hits,
                    "took_ms": round(outcome.took_ms, 1),
                    "rerank_skipped": outcome.rerank_skipped,
                },
            )
            return outcome

        if self.retriever.count_passages() == 0:
            return result((), 0, hint="The index is empty")

        try:
            vectors = self.embedder.embed((query,))
            if len(vectors) != 1 or len(vectors[0]) != self.embedder.dimension:
                raise ValueError("embedder returned an invalid query vector")
        except Exception as error:
            _LOG.exception(
                "query_embedding_failed", extra={"model": self.embedder.model_name}
            )
            if isinstance(error, ModelUnavailable):
                raise
            raise ModelUnavailable(
                "Query embedding failed",
                details={"model": self.embedder.model_name, "reason": str(error)},
            ) from error
        sparse = self.sparse_encoder.encode(query)
        candidates = self.retriever.retrieve(vectors[0], sparse, self.top_k)
        if not candidates:
            return result((), 0)

        valid: list[RetrievalCandidate] = []
        for candidate in candidates:
            if _valid_candidate(candidate):
                valid.append(candidate)
            else:
                _LOG.warning(
                    "invalid_retrieval_candidate passage_id=%s",
                    candidate.passage.passage_id,
                    extra={"passage_id": candidate.passage.passage_id},
                )
        if not valid:
            return result((), 0)

        rerank_skipped = False
        rerank_skip_reason: str | None = None
        try:
            scores = self.reranker.score(query, [item.passage.text for item in valid])
            if len(scores) != len(valid) or not all(
                math.isfinite(score) for score in scores
            ):
                raise ValueError("reranker returned invalid scores")
        except Exception as error:
            _LOG.exception(
                "query_reranking_failed", extra={"model": self.reranker.model_name}
            )
            rerank_skipped = True
            rerank_skip_reason = str(error)
            scores = tuple(item.dense_score for item in valid)

        ranked = tuple(
            RankedPassage(
                item.passage,
                item.source,
                _normalized_score(score),
                item.dense_score,
                None if rerank_skipped else score,
            )
            for item, score in zip(valid, scores, strict=True)
        )
        if rerank_skipped:
            # Without a reranker the first-stage order is the best available.
            stage1 = {
                item.passage.passage_id: item.stage1_rank
                for item in valid
                if item.stage1_rank is not None
            }
            ordered = tuple(
                sorted(
                    ranked,
                    key=lambda item: (
                        item.passage.passage_id not in stage1,
                        stage1.get(item.passage.passage_id, 0),
                        *ranked_passage_key(item),
                    ),
                )
            )
        else:
            ordered = tuple(sorted(ranked, key=ranked_passage_key))
        removed = 0
        if collapsing:
            # Grouping follows the final order, so the cut below keeps the
            # best member of each group and refills the slots of the others.
            ordered, removed = collapse(ordered)
        return result(
            ordered[:limit],
            len(valid),
            rerank_skipped=rerank_skipped,
            rerank_skip_reason=rerank_skip_reason,
            collapsed_duplicates=removed,
        )
