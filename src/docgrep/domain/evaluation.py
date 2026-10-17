"""Known-item retrieval evaluation (ADR-0024): the cases, what the
per-channel probe reads from the index, and the report."""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from docgrep.domain.errors import ValidationError
from docgrep.domain.globs import matching_pattern
from docgrep.domain.sources import (
    FILTER_FIELDS,
    SourceFilter,
    normalize_path,
    parse_file_types,
)

# The most candidates one channel list of the probe may hold.
MAX_PROBE_DEPTH = 10_000


@dataclass(frozen=True, slots=True)
class ProbeHit:
    """One passage in a channel list; ``text`` only when it was asked for."""

    passage_id: str
    source_id: str
    text: str | None = None


@dataclass(frozen=True, slots=True)
class ChannelLists:
    """The dense, sparse and fused first-stage lists of one query, best first.

    ``sparse`` is None when the index has no sparse channel; it is empty
    when the query has no sparse terms. ``fused`` is the reciprocal rank
    fusion of the channels that took part.
    """

    dense: tuple[ProbeHit, ...]
    sparse: tuple[ProbeHit, ...] | None
    fused: tuple[ProbeHit, ...]


# The default depth of the channel lists.
DEFAULT_PROBE_DEPTH = 2_000
# The k of the recall@k figures on the final ranking.
RECALL_AT = (1, 3, 5, 10, 25)
# A case within top_k whose final rank is worse than this is lost in rerank.
LOST_IN_RERANK_AFTER = 10
# Why a channel list has no rank for every case.
NO_SPARSE_TERMS = "no_sparse_terms"
SPARSE_UNAVAILABLE = "sparse_unavailable"

CASE_FIELDS = ("id", "query", "source", "source_glob", "fragment", "filters", "note")
_EXPECTATIONS = ("source", "source_glob", "fragment")
_YEAR_FIELDS = ("year_from", "year_to")


def normalize_whitespace(text: str) -> str:
    """Collapse every run of whitespace into one space and trim the ends."""
    return " ".join(text.split())


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One known-item query and what its expected passage satisfies.

    A passage is a hit when it satisfies every expectation given: its
    source is ``source`` (a stored path) or matches ``source_glob``, and
    its text contains ``fragment`` once whitespace is normalized in both.
    """

    case_id: str
    line: int
    query: str
    source: str | None = None
    source_glob: str | None = None
    fragment: str | None = None
    filters: SourceFilter | None = None
    note: str | None = None

    @property
    def names_source(self) -> bool:
        return self.source is not None or self.source_glob is not None

    def source_matches(self, source_id: str) -> bool:
        """Whether a stored path satisfies ``source`` and ``source_glob``."""
        if self.source is not None and source_id != self.source:
            return False
        return self.source_glob is None or (
            matching_pattern(normalize_path(source_id), (self.source_glob,)) is not None
        )

    def is_hit(self, source_id: str, text: str | None) -> bool:
        if not self.source_matches(source_id):
            return False
        if self.fragment is None:
            return True
        return text is not None and normalize_whitespace(
            self.fragment
        ) in normalize_whitespace(text)


def _problem(line: int, message: str) -> ValidationError:
    return ValidationError(f"line {line}: {message}", details={"line": line})


def _text_field(line: int, record: Mapping[str, object], name: str) -> str | None:
    value = record.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _problem(line, f"{name} must be a non-empty string")
    return value


def _case_filter(line: int, value: object) -> SourceFilter | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise _problem(line, "filters must be an object")
    unknown = sorted(set(value) - set(FILTER_FIELDS))
    if unknown:
        raise _problem(line, f"unknown filter fields: {', '.join(unknown)}")
    lists: dict[str, tuple[str, ...]] = {}
    years: dict[str, int | None] = {}
    for name, item in value.items():
        if name in _YEAR_FIELDS:
            if item is not None and (
                isinstance(item, bool) or not isinstance(item, int)
            ):
                raise _problem(line, f"filters.{name} must be an integer")
            years[name] = item
        elif not isinstance(item, list) or not all(
            isinstance(entry, str) for entry in item
        ):
            raise _problem(line, f"filters.{name} must be a list of strings")
        else:
            lists[name] = tuple(item)
    try:
        return SourceFilter(
            source_ids=lists.get("source_ids", ()),
            authors=lists.get("authors", ()),
            year_from=years.get("year_from"),
            year_to=years.get("year_to"),
            file_types=parse_file_types(lists.get("file_types", ())),
            path_prefixes=lists.get("path_prefixes", ()),
            path_globs=lists.get("path_globs", ()),
            roots=lists.get("roots", ()),
        )
    except ValidationError as error:
        raise _problem(line, f"filters: {error}") from error


def parse_cases(text: str) -> tuple[EvalCase, ...]:
    """The cases of a JSON Lines file, one object per line, in file order.

    Blank lines and lines starting with ``#`` are skipped. Raises
    ``ValidationError`` naming the line (``details["line"]``) for invalid
    JSON, unknown fields, a missing query or expectation, a malformed
    value or a repeated id, and when the file holds no case.
    """
    cases: list[EvalCase] = []
    seen: dict[str, int] = {}
    for line, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise _problem(line, f"invalid JSON: {error.msg}") from error
        if not isinstance(record, dict):
            raise _problem(line, "a case must be a JSON object")
        unknown = sorted(set(record) - set(CASE_FIELDS))
        if unknown:
            raise _problem(line, f"unknown fields: {', '.join(unknown)}")
        raw_id = record.get("id")
        if raw_id is None:
            case_id = str(line)
        elif isinstance(raw_id, int) and not isinstance(raw_id, bool):
            case_id = str(raw_id)
        elif isinstance(raw_id, str) and raw_id.strip():
            case_id = raw_id
        else:
            raise _problem(line, "id must be a non-empty string or an integer")
        if case_id in seen:
            raise _problem(line, f"id {case_id!r} repeats line {seen[case_id]}")
        seen[case_id] = line
        query = _text_field(line, record, "query")
        if query is None:
            raise _problem(line, "query is required")
        expected = {name: _text_field(line, record, name) for name in _EXPECTATIONS}
        if all(value is None for value in expected.values()):
            raise _problem(line, "one of source, source_glob or fragment is required")
        cases.append(
            EvalCase(
                case_id,
                line,
                query,
                expected["source"],
                expected["source_glob"],
                expected["fragment"],
                _case_filter(line, record.get("filters")),
                _text_field(line, record, "note"),
            )
        )
    if not cases:
        raise ValidationError("the cases file holds no case", details={"line": 0})
    return tuple(cases)


@dataclass(frozen=True, slots=True)
class ListRank:
    """Where a case is found in one list: the 1-based position of the best
    placed hit, and of the first passage of an expected source when the
    case names one. ``reason`` says why the list could not be read."""

    rank: int | None = None
    source_rank: int | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "source_rank": self.source_rank,
            "reason": self.reason,
        }


def list_rank(
    case: EvalCase, entries: Iterable[Iterable[tuple[str, str | None]]]
) -> ListRank:
    """The rank of a case in an ordered list; each entry is one position,
    holding the (source id, text) of every passage shown there."""
    rank: int | None = None
    source_rank: int | None = None
    for position, members in enumerate(entries, start=1):
        for source_id, text in members:
            if rank is None and case.is_hit(source_id, text):
                rank = position
            if source_rank is None and case.source_matches(source_id):
                source_rank = position
        if rank is not None and (source_rank is not None or not case.names_source):
            break
    return ListRank(rank, source_rank if case.names_source else None)


LIST_NAMES = ("dense", "sparse", "fused_deep", "stage1", "final")


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Where one case is found at every stage of the first stage and after.

    ``expected_sources`` counts the indexed sources the case's ``source``
    or ``source_glob`` matches (None for a fragment alone), and
    ``expected_roots`` names their roots. ``rerank_score`` is the score of
    the best-placed hit in the final order, as search returns it, and None
    when the hit is not within ``top_k`` or rerank was skipped. The
    control run is the same search restricted to the expected sources.
    """

    case: EvalCase
    lists: Mapping[str, ListRank]
    within_top_k: bool
    rerank_score: float | None
    expected_sources: int | None = None
    expected_roots: tuple[str, ...] = ()
    control_rank: int | None = None
    control_score: float | None = None

    def rank(self, name: str) -> int | None:
        return self.lists[name].rank

    def as_dict(self) -> dict[str, object]:
        case = self.case
        return {
            "id": case.case_id,
            "line": case.line,
            "query": case.query,
            "note": case.note,
            "expected": {
                "source": case.source,
                "source_glob": case.source_glob,
                "fragment": case.fragment,
                "filters": case.filters.applied() if case.filters else {},
                "sources": self.expected_sources,
                "roots": list(self.expected_roots),
            },
            "ranks": {name: self.lists[name].as_dict() for name in LIST_NAMES},
            "within_top_k": self.within_top_k,
            "rerank_score": self.rerank_score,
            "control_rank": self.control_rank,
            "control_score": self.control_score,
        }


@dataclass(frozen=True, slots=True)
class Aggregates:
    """Figures over every case, each case counting once.

    Recall is the share of all cases found: within k on the final order,
    within ``top_k`` on the first stage and within the depth on the channel
    lists. MRR is over the final order, a case not found counting 0. The
    ``lost_*`` lists name cases by id: found by a channel or the deep
    fusion but not within ``top_k``; within ``top_k`` but ranked after
    ``LOST_IN_RERANK_AFTER``; and found in no list at all.
    """

    cases: int
    found: Mapping[str, int]
    recall_at: Mapping[int, float]
    recall_stage1: float
    recall_dense: float
    recall_sparse: float
    recall_fused_deep: float
    mrr: float
    lost_in_first_stage: tuple[str, ...]
    lost_in_rerank: tuple[str, ...]
    not_found: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "cases": self.cases,
            "found": dict(self.found),
            "recall_at": {str(k): value for k, value in self.recall_at.items()},
            "recall_stage1_top_k": self.recall_stage1,
            "recall_dense_depth": self.recall_dense,
            "recall_sparse_depth": self.recall_sparse,
            "recall_fused_deep_depth": self.recall_fused_deep,
            "mrr": self.mrr,
            "lost_in_first_stage": list(self.lost_in_first_stage),
            "lost_in_rerank": list(self.lost_in_rerank),
            "not_found": list(self.not_found),
        }


def aggregate(results: Sequence[CaseResult]) -> Aggregates:
    """The aggregates of the case results, in case order for the lists."""
    total = len(results)

    def share(count: int) -> float:
        return count / total if total else 0.0

    def found(name: str) -> int:
        return sum(item.rank(name) is not None for item in results)

    finals = [item.rank("final") for item in results]
    channels = ("dense", "sparse", "fused_deep")
    return Aggregates(
        cases=total,
        found={
            **{name: found(name) for name in LIST_NAMES},
            "control": sum(item.control_rank is not None for item in results),
        },
        recall_at={
            k: share(sum(rank is not None and rank <= k for rank in finals))
            for k in RECALL_AT
        },
        recall_stage1=share(sum(item.within_top_k for item in results)),
        recall_dense=share(found("dense")),
        recall_sparse=share(found("sparse")),
        recall_fused_deep=share(found("fused_deep")),
        mrr=sum(1 / rank for rank in finals if rank) / total if total else 0.0,
        lost_in_first_stage=tuple(
            item.case.case_id
            for item in results
            if not item.within_top_k
            and any(item.rank(name) is not None for name in channels)
        ),
        lost_in_rerank=tuple(
            item.case.case_id
            for item in results
            if item.within_top_k and (item.rank("final") or 0) > LOST_IN_RERANK_AFTER
        ),
        not_found=tuple(
            item.case.case_id
            for item in results
            if all(item.rank(name) is None for name in LIST_NAMES)
        ),
    )


@dataclass(frozen=True, slots=True)
class EvalSettings:
    """What an evaluation ran with; nothing that changes between runs."""

    passages: int
    embedding_model: str
    reranker_model: str
    top_k: int
    depth: int
    collapse_duplicates: bool
    sparse_channel: bool
    control: bool
    # The fused candidates reranked, and the sparse weighting of the index
    # with its BM25 parameters (None for tf-v1; ADR-0025).
    rerank_pool: int | None = None
    # The search effort of the dense channel: hnsw_ef, "exact", or None for
    # the store's own default (ADR-0026).
    dense_hnsw_ef: int | str | None = None
    sparse_encoder: str = "tf-v1"
    bm25_k1: float | None = None
    bm25_b: float | None = None
    bm25_avgdl: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "passages": self.passages,
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model,
            "top_k": self.top_k,
            "rerank_pool": self.top_k if self.rerank_pool is None else self.rerank_pool,
            "dense_hnsw_ef": self.dense_hnsw_ef,
            "depth": self.depth,
            "collapse_duplicates": self.collapse_duplicates,
            "sparse_channel": self.sparse_channel,
            "control": self.control,
            "sparse_encoder": self.sparse_encoder,
            "bm25_k1": self.bm25_k1,
            "bm25_b": self.bm25_b,
            "bm25_avgdl": self.bm25_avgdl,
        }


@dataclass(frozen=True, slots=True)
class HnswRecall:
    """Recall at ``limit`` of the dense channel as search runs it, against
    exact search, per case (None when nothing was eligible) and summed up."""

    limit: int
    cases: tuple[tuple[str, float | None], ...]

    def as_dict(self) -> dict[str, object]:
        measured = [value for _, value in self.cases if value is not None]
        return {
            "limit": self.limit,
            "mean": round(sum(measured) / len(measured), 4) if measured else None,
            "min": round(min(measured), 4) if measured else None,
            "cases": [
                {"id": case_id, "recall": None if value is None else round(value, 4)}
                for case_id, value in self.cases
            ],
        }


@dataclass(frozen=True, slots=True)
class EvalReport:
    settings: EvalSettings
    cases: tuple[CaseResult, ...]
    aggregates: Aggregates
    hnsw_recall: HnswRecall | None = None

    def as_dict(self) -> dict[str, object]:
        """The report with a fixed key order and the cases in file order;
        ``hnsw_recall`` only when it was measured."""
        body: dict[str, object] = {
            "settings": self.settings.as_dict(),
            "cases": [item.as_dict() for item in self.cases],
            "aggregates": self.aggregates.as_dict(),
        }
        if self.hnsw_recall is not None:
            body["hnsw_recall"] = self.hnsw_recall.as_dict()
        return body
