"""Evaluate known-item retrieval through the search pipeline (ADR-0024).

Read-only: the index is read through the diagnostic seam of
``FindPassages`` and the per-channel probe; nothing is written and no lock
is taken.
"""

import logging
from collections.abc import Sequence
from dataclasses import replace

from docgrep.domain.errors import ModelUnavailable, ValidationError
from docgrep.domain.evaluation import (
    MAX_PROBE_DEPTH,
    NO_SPARSE_TERMS,
    SPARSE_UNAVAILABLE,
    CaseResult,
    ChannelLists,
    EvalCase,
    EvalReport,
    EvalSettings,
    ListRank,
    ProbeHit,
    aggregate,
    list_rank,
)
from docgrep.domain.models import RankedPassage
from docgrep.domain.protocols import ChannelProbe, SparseEncoder
from docgrep.domain.roots import CorpusRoots
from docgrep.domain.sources import SourceFilter
from docgrep.usecases.find_passages import Diagnosis, FindPassages

_LOG = logging.getLogger(__name__)


def _probe_entries(
    hits: Sequence[ProbeHit],
) -> list[tuple[tuple[str, str | None]]]:
    return [((hit.source_id, hit.text),) for hit in hits]


def _members(item: RankedPassage) -> tuple[tuple[str, str | None], ...]:
    """A position of the final order: the passage and the duplicates
    collapsed into it, which search shows at the same place."""
    return tuple(
        (member.source.source_id, member.passage.text)
        for member in (item, *item.duplicates)
    )


def _final(case: EvalCase, diagnosis: Diagnosis) -> tuple[ListRank, float | None]:
    """The ranks of a case in the final order, and the score search returns
    for its best-placed hit (None without a hit or when rerank was skipped).
    """
    final = list_rank(case, (_members(item) for item in diagnosis.passages))
    if final.rank is None or diagnosis.rerank_skipped:
        return final, None
    return final, diagnosis.passages[final.rank - 1].score


def _stage1(case: EvalCase, diagnosis: Diagnosis) -> ListRank:
    """The 1-based ranks of a case in the first-stage order of the pipeline."""
    hits: list[int] = []
    sources: list[int] = []
    for item in diagnosis.passages:
        for member in (item, *item.duplicates):
            rank = diagnosis.stage1_ranks.get(member.passage.passage_id)
            if rank is None:
                continue
            source_id = member.source.source_id
            if case.is_hit(source_id, member.passage.text):
                hits.append(rank + 1)
            if case.names_source and case.source_matches(source_id):
                sources.append(rank + 1)
    return ListRank(min(hits, default=None), min(sources, default=None))


def _control_filter(case: EvalCase) -> SourceFilter:
    """The case's filter restricted to its expected sources; ``source`` and
    ``source_glob`` replace the filter's source ids and path globs."""
    base = case.filters or SourceFilter()
    return replace(
        base,
        source_ids=(case.source,) if case.source else (),
        path_globs=(case.source_glob,) if case.source_glob else (),
    )


class EvaluateRetrieval:
    """Run known-item cases through search and report where each is found."""

    def __init__(
        self,
        *,
        search: FindPassages,
        probe: ChannelProbe,
    ) -> None:
        self.search = search
        self.probe = probe

    def run(
        self,
        cases: Sequence[EvalCase],
        *,
        depth: int,
        control: bool = True,
    ) -> EvalReport:
        """Evaluate every case, in order; models are used for all at once.

        Raises ``ValidationError`` for a depth outside 1 to
        ``MAX_PROBE_DEPTH`` or a query the search would refuse (naming its
        line) before anything is read, and the errors of search otherwise.
        """
        search = self.search
        if not 0 < depth <= MAX_PROBE_DEPTH:
            raise ValidationError(
                f"depth must be between 1 and {MAX_PROBE_DEPTH}",
                details={"minimum": 1, "maximum": MAX_PROBE_DEPTH},
            )
        for case in cases:
            if len(case.query) > search.max_query_len:
                raise ValidationError(
                    f"line {case.line}: query must be at most "
                    f"{search.max_query_len} characters",
                    details={"line": case.line, "max_query_len": search.max_query_len},
                )
        metadata = search.check_index()
        sparse_query = search.query_encoder(metadata)
        passages = search.retriever.count_passages()
        vectors = self._embed([case.query for case in cases])
        roots = CorpusRoots() if search.roots is None else search.roots.roots()
        indexed = self._indexed_sources()
        sparse_channel = True
        results: list[CaseResult] = []
        for case, vector in zip(cases, vectors, strict=True):
            result, lists = self._case(
                case, vector, depth, control, roots, indexed, sparse_query
            )
            sparse_channel = sparse_channel and lists.sparse is not None
            results.append(result)
        report = EvalReport(
            EvalSettings(
                passages=passages,
                embedding_model=search.embedder.model_name,
                reranker_model=search.reranker.model_name,
                top_k=search.top_k,
                depth=depth,
                collapse_duplicates=search.collapse_duplicates,
                sparse_channel=sparse_channel,
                control=control,
            ),
            tuple(results),
            aggregate(results),
        )
        # Counts only: neither queries nor source names are logged.
        _LOG.info(
            "evaluation_completed",
            extra={"cases": len(results), "depth": depth, "control": control},
        )
        return report

    def _embed(self, queries: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        embedder = self.search.embedder
        try:
            vectors = embedder.embed(queries)
            if len(vectors) != len(queries) or any(
                len(vector) != embedder.dimension for vector in vectors
            ):
                raise ValueError("embedder returned invalid query vectors")
        except Exception as error:
            _LOG.exception(
                "query_embedding_failed", extra={"model": embedder.model_name}
            )
            if isinstance(error, ModelUnavailable):
                raise
            raise ModelUnavailable(
                "Query embedding failed",
                details={"model": embedder.model_name, "reason": str(error)},
            ) from error
        return vectors

    def _indexed_sources(self) -> tuple[str, ...] | None:
        """The indexed source ids of the manifest, or None without one."""
        manifest = self.search.manifest
        if manifest is None:
            return None
        return tuple(
            record.source.source_id
            for record in manifest.committed_sources()
            if record.status == "indexed"
        )

    def _case(
        self,
        case: EvalCase,
        vector: Sequence[float],
        depth: int,
        control: bool,
        roots: CorpusRoots,
        indexed: Sequence[str] | None,
        sparse_query: SparseEncoder,
    ) -> tuple[CaseResult, ChannelLists]:
        search = self.search
        diagnosis = search.diagnose(
            case.query, dense_vector=vector, source_filter=case.filters
        )
        # The query side of the sparse channel, as search encodes it.
        sparse_terms = sparse_query.encode(case.query)
        lists = self.probe.probe_channels(
            vector,
            sparse_terms,
            depth,
            source_ids=diagnosis.source_ids,
            with_text=case.fragment is not None,
        )
        reason = None
        if lists.sparse is None:
            reason = SPARSE_UNAVAILABLE
        elif not sparse_terms:
            reason = NO_SPARSE_TERMS
        sparse = (
            ListRank(reason=reason)
            if reason is not None
            else list_rank(case, _probe_entries(lists.sparse or ()))
        )
        fused = list_rank(case, _probe_entries(lists.fused))
        final, score = _final(case, diagnosis)
        stage1 = _stage1(case, diagnosis)
        control_rank = control_score = None
        if control and case.names_source:
            restricted = search.diagnose(
                case.query, dense_vector=vector, source_filter=_control_filter(case)
            )
            controlled, control_score = _final(case, restricted)
            control_rank = controlled.rank
        expected: tuple[str, ...] | None = None
        if indexed is not None and case.names_source:
            expected = tuple(
                source for source in indexed if case.source_matches(source)
            )
        # The root of a named source is known even when it is not indexed.
        rooted = expected or ((case.source,) if case.source is not None else ())
        return (
            CaseResult(
                case,
                {
                    "dense": list_rank(case, _probe_entries(lists.dense)),
                    "sparse": sparse,
                    "fused_deep": replace(fused, reason=reason),
                    "stage1": stage1,
                    "final": final,
                },
                within_top_k=stage1.rank is not None,
                rerank_score=score,
                expected_sources=None if expected is None else len(expected),
                expected_roots=tuple(sorted({roots.root_of(s) for s in rooted})),
                control_rank=control_rank,
                control_score=control_score,
            ),
            lists,
        )
