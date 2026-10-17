"""EvaluateRetrieval over the in-memory store with the deterministic models:
ranks in every list, the control run, sparse degradation, read-only use and
determinism (ADR-0024)."""

from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from docgrep.adapters.models.deterministic import DeterministicEmbedder, LexicalReranker
from docgrep.adapters.models.sparse import BM25SparseEncoder
from docgrep.domain.errors import IndexIncomplete, ValidationError
from docgrep.domain.evaluation import (
    NO_SPARSE_TERMS,
    SPARSE_UNAVAILABLE,
    EvalCase,
    ListRank,
)
from docgrep.domain.models import FileType, Passage, Source, make_passage_id
from docgrep.domain.protocols import Reranker
from docgrep.domain.sources import SourceFilter
from docgrep.usecases.evaluate_retrieval import EvaluateRetrieval
from docgrep.usecases.find_passages import FindPassages
from fakes import InMemoryManifest, InMemoryPassageStore

DECOYS = 12
TARGET = "books/arnold.md"
TARGET_TEXT = "Rigid bodies, tides and a single comet."


class CountingEmbedder(DeterministicEmbedder):
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        self.calls.append(tuple(texts))
        return super().embed(texts)


class FailingReranker:
    model_name = "broken-reranker"

    def score(self, query: str, passages: Sequence[str]) -> tuple[float, ...]:
        raise RuntimeError("no reranker")


def _library() -> tuple[InMemoryPassageStore, InMemoryManifest]:
    store = InMemoryPassageStore()
    manifest = InMemoryManifest()
    embedder = DeterministicEmbedder()
    sparse = BM25SparseEncoder()
    store.recreate_collection(embedder.dimension)
    texts = [
        (f"decoys/d{number:02}.md", f"Comet orbit, comet orbit {number}.")
        for number in range(DECOYS)
    ]
    texts.append((TARGET, TARGET_TEXT))
    for path, text in texts:
        source = Source(path, path, FileType.MARKDOWN)
        passage = Passage(make_passage_id(path, 0), path, text, 0, len(text.split()))
        (vector,) = embedder.embed((text,))
        store.upsert(passage, source, vector, sparse.encode(text))
        manifest.record_source(source, "indexed")
    store.write_completion_metadata(
        {"embedding_model": embedder.model_name, "embedding_dimension": 64}
    )
    return store, manifest


def _evaluation(
    store: InMemoryPassageStore,
    manifest: InMemoryManifest,
    *,
    top_k: int,
    embedder: DeterministicEmbedder | None = None,
    reranker: Reranker | None = None,
) -> EvaluateRetrieval:
    search = FindPassages(
        embedder=embedder or DeterministicEmbedder(),
        sparse_encoder=BM25SparseEncoder(),
        retriever=store,
        reranker=reranker or LexicalReranker(),
        top_k=top_k,
        default_top_n=1,
        top_n_max=10,
        max_query_len=50,
        manifest=manifest,
        now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )
    return EvaluateRetrieval(search=search, probe=store)


CASE = EvalCase("target", 1, "comet orbit", source=TARGET, note="the Arnold book")


def test_a_passage_lost_in_the_first_stage_is_found_in_the_control_run() -> None:
    store, manifest = _library()

    report = _evaluation(store, manifest, top_k=3).run([CASE], depth=50)

    (result,) = report.cases
    assert result.lists["dense"].rank is not None
    assert result.lists["sparse"].rank == DECOYS + 1
    assert result.lists["fused_deep"].rank is not None
    assert result.lists["stage1"] == ListRank(None, None)
    assert result.lists["final"] == ListRank(None, None)
    assert not result.within_top_k
    assert result.rerank_score is None
    # Restricted to the expected source, it is found and scored.
    assert result.control_rank == 1
    assert result.control_score == pytest.approx(0.6224593)
    assert result.expected_sources == 1
    assert report.aggregates.lost_in_first_stage == ("target",)


def test_without_control_no_control_search_runs() -> None:
    store, manifest = _library()

    report = _evaluation(store, manifest, top_k=3).run([CASE], depth=50, control=False)

    (result,) = report.cases
    assert (result.control_rank, result.control_score) == (None, None)
    assert report.settings.control is False


def test_a_final_rank_beyond_top_n_is_reported() -> None:
    store, manifest = _library()

    report = _evaluation(store, manifest, top_k=20).run([CASE], depth=50)

    (result,) = report.cases
    assert result.within_top_k
    # Every decoy has both query words; the target only one.
    assert result.lists["final"] == ListRank(DECOYS + 1, DECOYS + 1)
    assert result.lists["stage1"].rank == DECOYS + 1
    assert result.rerank_score == pytest.approx(0.6224593)
    assert report.aggregates.lost_in_rerank == ("target",)


def test_a_rank_beyond_the_depth_is_null() -> None:
    store, manifest = _library()

    report = _evaluation(store, manifest, top_k=3).run([CASE], depth=2)

    (result,) = report.cases
    assert result.lists["sparse"].rank is None
    assert result.lists["fused_deep"].rank is None
    assert report.settings.depth == 2


def test_a_query_without_sparse_terms_has_no_sparse_rank() -> None:
    store, manifest = _library()
    case = EvalCase("signs", 1, "?? !!", fragment="single comet")

    report = _evaluation(store, manifest, top_k=20).run([case], depth=50)

    (result,) = report.cases
    assert result.lists["sparse"] == ListRank(reason=NO_SPARSE_TERMS)
    assert result.lists["fused_deep"].reason == NO_SPARSE_TERMS
    assert result.lists["fused_deep"].rank == result.lists["dense"].rank
    assert result.lists["dense"].source_rank is None
    assert report.settings.sparse_channel


def test_an_index_without_a_sparse_channel_degrades() -> None:
    store, manifest = _library()
    store.sparse_channel = False

    report = _evaluation(store, manifest, top_k=20).run([CASE], depth=50)

    (result,) = report.cases
    assert result.lists["sparse"] == ListRank(reason=SPARSE_UNAVAILABLE)
    assert result.lists["fused_deep"].reason == SPARSE_UNAVAILABLE
    assert result.lists["fused_deep"].rank == result.lists["dense"].rank
    assert not report.settings.sparse_channel


def test_every_query_is_embedded_in_one_call() -> None:
    store, manifest = _library()
    embedder = CountingEmbedder()
    cases = [CASE, EvalCase("second", 2, "tides", fragment="tides")]

    _evaluation(store, manifest, top_k=3, embedder=embedder).run(cases, depth=10)

    assert embedder.calls == [("comet orbit", "tides")]


def test_nothing_is_written() -> None:
    store, manifest = _library()
    items = dict(store._items)
    completion = store.read_completion_metadata()
    sources = manifest.committed_sources()

    _evaluation(store, manifest, top_k=3).run([CASE], depth=10)

    assert store._items == items
    assert store.read_completion_metadata() == completion
    assert store.staging_name() is None
    assert manifest.committed_sources() == sources
    assert manifest.latest_run() is None
    assert store.probes == [10]


def test_two_runs_give_the_same_report() -> None:
    store, manifest = _library()
    evaluation = _evaluation(store, manifest, top_k=5)
    cases = [CASE, EvalCase("glob", 2, "tides", source_glob="books/*.md")]

    assert (
        evaluation.run(cases, depth=20).as_dict()
        == evaluation.run(cases, depth=20).as_dict()
    )


def test_a_skipped_rerank_gives_no_score() -> None:
    store, manifest = _library()

    report = _evaluation(store, manifest, top_k=20, reranker=FailingReranker()).run(
        [CASE], depth=20
    )

    (result,) = report.cases
    assert result.lists["final"].rank is not None
    assert (result.rerank_score, result.control_score) == (None, None)


@pytest.mark.parametrize("depth", [0, 10_001])
def test_the_depth_is_checked_before_anything_runs(depth: int) -> None:
    store, manifest = _library()
    embedder = CountingEmbedder()

    with pytest.raises(ValidationError, match="depth"):
        _evaluation(store, manifest, top_k=3, embedder=embedder).run(
            [CASE], depth=depth
        )
    assert embedder.calls == []


def test_a_query_longer_than_search_allows_names_its_line() -> None:
    store, manifest = _library()
    embedder = CountingEmbedder()
    long = EvalCase("long", 4, "x" * 51, fragment="x")

    with pytest.raises(ValidationError, match="line 4: query must be at most 50"):
        _evaluation(store, manifest, top_k=3, embedder=embedder).run(
            [CASE, long], depth=10
        )
    assert embedder.calls == []


def test_no_published_index_fails_before_any_model_runs() -> None:
    embedder = CountingEmbedder()

    with pytest.raises(IndexIncomplete):
        _evaluation(
            InMemoryPassageStore(), InMemoryManifest(), top_k=3, embedder=embedder
        ).run([CASE], depth=10)
    assert embedder.calls == []


def test_case_filters_apply_to_every_list() -> None:
    store, manifest = _library()
    case = EvalCase(
        "filtered",
        1,
        "comet orbit",
        source_glob="decoys/*.md",
        filters=SourceFilter(path_prefixes=("books/",)),
    )

    report = _evaluation(store, manifest, top_k=20).run([case], depth=50)

    (result,) = report.cases
    assert all(rank == ListRank() for name, rank in result.lists.items())
    assert result.expected_sources == DECOYS


def test_the_dense_recall_is_measured_per_case_on_request() -> None:
    store, manifest = _library()
    store.recall_override = 0.5
    other = EvalCase("other", 2, "rigid bodies", source=TARGET)
    filtered = EvalCase(
        "filtered", 3, "comet", filters=SourceFilter(source_ids=(TARGET,))
    )

    plain = _evaluation(store, manifest, top_k=3).run([CASE], depth=50)
    report = _evaluation(store, manifest, top_k=3).run(
        [CASE, other, filtered], depth=50, hnsw_recall=True
    )

    assert plain.hnsw_recall is None
    assert "hnsw_recall" not in plain.as_dict()
    assert report.hnsw_recall is not None
    assert report.hnsw_recall.as_dict() == {
        "limit": 3,
        "mean": 0.5,
        "min": 0.5,
        "cases": [
            {"id": "target", "recall": 0.5},
            {"id": "other", "recall": 0.5},
            {"id": "filtered", "recall": 0.5},
        ],
    }
    # The recall is measured at top_k, once per case.
    assert store.recall_probes == [3, 3, 3]
    # Measuring changes nothing else.
    assert report.cases[0] == plain.cases[0]
