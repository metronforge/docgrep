"""Read and rank verbatim passages through domain ports."""

import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

from docgrep.domain.duplicates import collapse_duplicates as collapse
from docgrep.domain.errors import (
    ConfigDrift,
    IndexIncomplete,
    ModelUnavailable,
    ValidationError,
)
from docgrep.domain.metadata import ocr_origin, sanitize_source
from docgrep.domain.models import (
    ManifestSourceRecord,
    OcrOutput,
    RankedPassage,
    RetrievalCandidate,
    SearchResult,
    Source,
)
from docgrep.domain.protocols import (
    CommittedSourceReader,
    EmbeddingProvider,
    OcrOutputOwners,
    PassageRetriever,
    Reranker,
    RootCatalog,
    SparseEncoder,
    SparseEncoders,
    ranked_passage_key,
)
from docgrep.domain.roots import CorpusRoots, known_root_names
from docgrep.domain.search_effort import DenseSearchEffort
from docgrep.domain.sources import SourceFilter, check_roots, matches
from docgrep.domain.sparse import SparseWeighting

_LOG = logging.getLogger(__name__)

NO_MATCHING_SOURCES_HINT = "The filters matched no indexed source"


@dataclass(frozen=True, slots=True)
class _Scope:
    """The sources a filtered search may return passages from.

    ``source_ids`` is None when every indexed source matches, so no
    allow-list is needed.
    """

    source_ids: frozenset[str] | None
    matched: int
    excluded_for_missing_metadata: int | None


@dataclass(slots=True)
class RequestSanitizer:
    """Sanitize source metadata for one request (ADR-0021).

    The OCR registry is read once, on first use; the manifest is read at
    most once, and only when an OCR output may inherit from its original.
    """

    current_year: int
    ocr: OcrOutputOwners | None
    manifest: CommittedSourceReader | None
    records: Sequence[ManifestSourceRecord] | None = None
    _outputs: Mapping[str, OcrOutput] | None = None
    _originals: dict[str, Source] = field(default_factory=dict)
    _originals_read: bool = False

    def outputs(self) -> Mapping[str, OcrOutput]:
        if self._outputs is None:
            self._outputs = {} if self.ocr is None else self.ocr.live_outputs()
        return self._outputs

    def originals(self) -> Mapping[str, Source]:
        if not self._originals_read:
            self._originals_read = True
            records = self.records
            if records is None and self.manifest is not None:
                records = _committed(self.manifest)
            for record in records or ():
                self._originals[record.source.path] = record.source
        return self._originals

    def clean(self, source: Source) -> Source:
        outputs = self.outputs()
        origin = ocr_origin(source, outputs, {})
        if origin is not None:
            origin = ocr_origin(source, outputs, self.originals())
        return sanitize_source(
            source, current_year=self.current_year, origin=origin
        ).source


@dataclass(frozen=True, slots=True)
class Diagnosis:
    """The whole ranking of one query, for evaluation (ADR-0024).

    ``passages`` is the public order after rerank and duplicate collapse,
    over every first-stage candidate, not cut to ``top_n``; the sources are
    not sanitized. ``stage1_ranks`` maps the passage id of every valid
    candidate, collapsed duplicates included, to its zero-based first-stage
    position. ``source_ids`` is the allow-list the filter resolved to, None
    when every source is eligible. ``hint`` says why nothing was retrieved,
    when it is known.
    """

    passages: tuple[RankedPassage, ...]
    stage1_ranks: Mapping[str, int]
    top_k: int
    rerank_skipped: bool = False
    rerank_skip_reason: str | None = None
    hint: str | None = None
    source_ids: frozenset[str] | None = None


@dataclass(frozen=True, slots=True)
class _Ranking:
    """The ordered candidates of one query, before the result is cut."""

    ordered: tuple[RankedPassage, ...]
    stage1_ranks: Mapping[str, int]
    valid: int
    removed: int
    rerank_skipped: bool
    rerank_skip_reason: str | None


def _committed(manifest: CommittedSourceReader) -> tuple[ManifestSourceRecord, ...]:
    """The committed sources; raises ``IndexIncomplete`` when the manifest
    cannot be read, the same category as a search without a published
    index."""
    try:
        return manifest.committed_sources()
    except Exception as error:
        _LOG.exception("search_manifest_read_failed")
        raise IndexIncomplete(
            "The index manifest cannot be read", details={"reason": str(error)}
        ) from error


def _excluded_for_missing_metadata(
    source_filter: SourceFilter, source: Source, root: str | None = None
) -> bool:
    """Whether a source fails the filter only because a filtered author or
    year is null: it passes once the null fields are no longer filtered."""
    relaxed = source_filter
    if source_filter.authors and source.author is None:
        relaxed = replace(relaxed, authors=())
    if (
        source_filter.year_from is not None or source_filter.year_to is not None
    ) and source.year is None:
        relaxed = replace(relaxed, year_from=None, year_to=None)
    return relaxed is not source_filter and matches(relaxed, source, root)


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
        manifest: CommittedSourceReader | None = None,
        ocr: OcrOutputOwners | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        roots: RootCatalog | None = None,
        sparse_encoders: SparseEncoders | None = None,
        rerank_pool: int | None = None,
        dense_hnsw_ef: DenseSearchEffort | None = None,
    ) -> None:
        """With ``sparse_encoders`` the query is encoded for the sparse
        weighting the index records (ADR-0025); without them, always with
        ``sparse_encoder``. ``rerank_pool`` is how many fused candidates are
        reranked, at most ``2 * top_k``; unset, ``top_k``. ``dense_hnsw_ef``
        is the search effort the retriever applies to the dense channel, for
        the applied configuration only (ADR-0026)."""
        if top_k <= 0 or not 0 < default_top_n <= top_n_max:
            raise ValueError("result limits must be positive")
        if rerank_pool is not None and not 0 < rerank_pool <= 2 * top_k:
            raise ValueError("rerank_pool must be between 1 and 2 * top_k")
        if max_query_len <= 0:
            raise ValueError("max_query_len must be positive")
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        self.sparse_encoders = sparse_encoders
        self.retriever = retriever
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_pool = top_k if rerank_pool is None else rerank_pool
        self.dense_hnsw_ef = dense_hnsw_ef
        self.default_top_n = default_top_n
        self.top_n_max = top_n_max
        self.max_query_len = max_query_len
        self.collapse_duplicates = collapse_duplicates
        self.roots = roots
        self.manifest = manifest
        self.ocr = ocr
        self.now = now

    def run(
        self,
        query: str,
        *,
        top_n: int | None = None,
        collapse_duplicates: bool | None = None,
        source_filter: SourceFilter | None = None,
    ) -> SearchResult:
        """Find passages, keeping the indexed text unchanged in every result.

        ``collapse_duplicates`` overrides the configured default when given.
        ``source_filter`` restricts the first stage to the indexed sources
        that match it (UC-02); a filter without any field set is no filter.
        Filters and the returned sources see the sanitized metadata
        (ADR-0021).
        """
        started = time.monotonic()
        collapsing = (
            self.collapse_duplicates
            if collapse_duplicates is None
            else collapse_duplicates
        )
        self._check_query(query)
        limit = self.default_top_n if top_n is None else top_n
        if not 1 <= limit <= self.top_n_max:
            raise ValidationError(
                f"top_n must be between 1 and {self.top_n_max}",
                details={"minimum": 1, "maximum": self.top_n_max},
            )

        sparse_query = self.query_encoder(self.check_index())
        applied_config: tuple[tuple[str, str | int | float | bool], ...] = (
            *(
                (("dense_hnsw_ef", self.dense_hnsw_ef),)
                if self.dense_hnsw_ef is not None
                else ()
            ),
            ("embedding_model", self.embedder.model_name),
            ("max_query_len", self.max_query_len),
            # Only a pool that differs from top_k changes the first stage, so
            # the body of a default search stays as it was.
            *((("rerank_pool", self.rerank_pool),) if self._pooled() else ()),
            ("reranker_model", self.reranker.model_name),
            ("top_k", self.top_k),
            ("top_n", limit),
        )
        applied_filters: Mapping[str, list[str] | int] = (
            source_filter.applied() if source_filter is not None else {}
        )
        sanitizer = RequestSanitizer(self.now().year, self.ocr, self.manifest)
        roots = CorpusRoots() if self.roots is None else self.roots.roots()
        scope = (
            self._scope(source_filter, sanitizer, roots)
            if source_filter is not None and applied_filters
            else None
        )
        excluded = None if scope is None else scope.excluded_for_missing_metadata

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
                applied_filters,
                excluded,
                {
                    item.source.source_id: roots.root_of(item.source.source_id)
                    for passage in passages
                    for item in (passage, *passage.duplicates)
                }
                if self.roots is not None and roots.named
                else {},
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
        if scope is not None and scope.matched == 0:
            # Nothing can match, so no model and no retrieval is needed.
            return result((), 0, hint=NO_MATCHING_SOURCES_HINT)

        ranking = self._rank(
            query, self._embed(query), scope, sparse_query, collapsing=collapsing
        )
        if ranking is None:
            return result((), 0)
        # Only the returned passages are sanitized, not every candidate.
        returned = tuple(
            replace(item, source=sanitizer.clean(item.source))
            for item in ranking.ordered[:limit]
        )
        return result(
            returned,
            ranking.valid,
            rerank_skipped=ranking.rerank_skipped,
            rerank_skip_reason=ranking.rerank_skip_reason,
            collapsed_duplicates=ranking.removed,
        )

    def diagnose(
        self,
        query: str,
        *,
        dense_vector: Sequence[float] | None = None,
        collapse_duplicates: bool | None = None,
        source_filter: SourceFilter | None = None,
    ) -> Diagnosis:
        """Rank one query as ``run`` does and return every candidate.

        The same validation, scope and ranking as ``run``, without the cut
        to ``top_n`` and without sanitizing sources; nothing is logged per
        query. ``dense_vector`` is the query embedding when the caller has
        embedded several queries at once; otherwise the query is embedded
        here.
        """
        collapsing = (
            self.collapse_duplicates
            if collapse_duplicates is None
            else collapse_duplicates
        )
        self._check_query(query)
        sparse_query = self.query_encoder(self.check_index())
        scope = None
        if source_filter is not None and source_filter.applied():
            sanitizer = RequestSanitizer(self.now().year, self.ocr, self.manifest)
            roots = CorpusRoots() if self.roots is None else self.roots.roots()
            scope = self._scope(source_filter, sanitizer, roots)
        allowed = None if scope is None else scope.source_ids
        if self.retriever.count_passages() == 0:
            return Diagnosis(
                (), {}, self.top_k, hint="The index is empty", source_ids=allowed
            )
        if scope is not None and scope.matched == 0:
            return Diagnosis(
                (),
                {},
                self.top_k,
                hint=NO_MATCHING_SOURCES_HINT,
                source_ids=frozenset(),
            )
        if dense_vector is None:
            dense_vector = self._embed(query)
        elif len(dense_vector) != self.embedder.dimension:
            raise ModelUnavailable(
                "Query embedding failed",
                details={
                    "model": self.embedder.model_name,
                    "reason": "embedder returned an invalid query vector",
                },
            )
        ranking = self._rank(
            query, dense_vector, scope, sparse_query, collapsing=collapsing
        )
        if ranking is None:
            return Diagnosis((), {}, self.top_k, source_ids=allowed)
        return Diagnosis(
            ranking.ordered,
            ranking.stage1_ranks,
            self.top_k,
            ranking.rerank_skipped,
            ranking.rerank_skip_reason,
            source_ids=allowed,
        )

    def _pooled(self) -> bool:
        """Whether the reranker sees a fused list cut elsewhere than top_k."""
        return self.rerank_pool != self.top_k

    def _check_query(self, query: str) -> None:
        if not query.strip():
            raise ValidationError("query is required")
        if len(query) > self.max_query_len:
            raise ValidationError(
                f"query must be at most {self.max_query_len} characters",
                details={"max_query_len": self.max_query_len},
            )

    def check_index(self) -> Mapping[str, str | int | float]:
        """Refuse an index without completion metadata or built with another
        embedding model or dimension: ``IndexIncomplete`` or ``ConfigDrift``.
        Returns the completion metadata."""
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
        return metadata

    def sparse_weighting(
        self, metadata: Mapping[str, str | int | float]
    ) -> SparseWeighting:
        """The sparse weighting an index records; ``ConfigDrift`` for one
        this version does not know."""
        try:
            return SparseWeighting.from_metadata(metadata)
        except ValueError as error:
            raise ConfigDrift(
                "The index records a sparse weighting this version cannot "
                "query; rebuild the index with `docgrep index`",
                details={"reason": str(error)},
            ) from error

    def query_encoder(self, metadata: Mapping[str, str | int | float]) -> SparseEncoder:
        """The encoder of the query side of an index's sparse channel."""
        if self.sparse_encoders is None:
            return self.sparse_encoder
        return self.sparse_encoders.query_encoder(self.sparse_weighting(metadata))

    def _embed(self, query: str) -> tuple[float, ...]:
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
        return vectors[0]

    def _rank(
        self,
        query: str,
        dense_vector: Sequence[float],
        scope: _Scope | None,
        sparse_encoder: SparseEncoder,
        *,
        collapsing: bool,
    ) -> _Ranking | None:
        """Recall, rerank and order the candidates of one query; None when
        no valid candidate was found."""
        sparse = sparse_encoder.encode(query)
        source_ids = None if scope is None else scope.source_ids
        if self._pooled():
            candidates = self.retriever.retrieve(
                dense_vector, sparse, self.top_k, source_ids, pool=self.rerank_pool
            )
        elif source_ids is None:
            candidates = self.retriever.retrieve(dense_vector, sparse, self.top_k)
        else:
            candidates = self.retriever.retrieve(
                dense_vector, sparse, self.top_k, source_ids=source_ids
            )
        if not candidates:
            return None

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
            return None

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
        # A retriever that gives no position lists its candidates in
        # first-stage order, so the list position stands in.
        stage1_ranks: dict[str, int] = {}
        for position, item in enumerate(valid):
            rank = position if item.stage1_rank is None else item.stage1_rank
            stage1_ranks.setdefault(item.passage.passage_id, rank)
        return _Ranking(
            ordered,
            stage1_ranks,
            len(valid),
            removed,
            rerank_skipped,
            rerank_skip_reason,
        )

    def _scope(
        self,
        source_filter: SourceFilter,
        sanitizer: RequestSanitizer,
        roots: CorpusRoots,
    ) -> _Scope:
        """Resolve the filter against the sanitized committed indexed sources.

        Raises ``IndexIncomplete`` when the manifest cannot be read, the same
        category as a search without a published index.
        """
        if self.manifest is None:
            raise IndexIncomplete("Filtered search needs the index manifest")
        records = _committed(self.manifest)
        sanitizer.records = records
        if source_filter.roots:
            check_roots(
                source_filter,
                known_root_names(roots, (r.source.source_id for r in records)),
            )
        indexed = [
            sanitizer.clean(record.source)
            for record in records
            if record.status == "indexed"
        ]
        allowed = frozenset(
            source.source_id
            for source in indexed
            if matches(source_filter, source, roots.root_of(source.source_id))
        )
        excluded: int | None = None
        if source_filter.authors or (
            source_filter.year_from is not None or source_filter.year_to is not None
        ):
            excluded = sum(
                _excluded_for_missing_metadata(
                    source_filter, source, roots.root_of(source.source_id)
                )
                for source in indexed
            )
        everything = len(allowed) == len({source.source_id for source in indexed})
        return _Scope(None if everything else allowed, len(allowed), excluded)
