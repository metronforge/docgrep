"""FindPassages collapses passages with the same text from several sources."""

from collections.abc import Mapping, Sequence

from docgrep.adapters.models.deterministic import DeterministicEmbedder, LexicalReranker
from docgrep.adapters.models.sparse import BM25SparseEncoder
from docgrep.domain.errors import ModelUnavailable
from docgrep.domain.models import (
    FileType,
    Passage,
    RetrievalCandidate,
    SearchResult,
    Source,
    make_passage_id,
)
from docgrep.domain.protocols import PassageRetriever, Reranker
from docgrep.usecases.find_passages import FindPassages
from fakes import InMemoryPassageStore

_METADATA = {"embedding_model": "deterministic-hash-v1", "embedding_dimension": 64}
_COMET = "A rare blue comet crosses the sky."


class StaticRetriever:
    def __init__(self, candidates: Sequence[RetrievalCandidate]) -> None:
        self.candidates = tuple(candidates)

    def retrieve(
        self,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float],
        top_k: int,
    ) -> tuple[RetrievalCandidate, ...]:
        return self.candidates[:top_k]

    def read_completion_metadata(self) -> Mapping[str, str | int | float] | None:
        return _METADATA

    def count_passages(self) -> int:
        return len(self.candidates)


class ScriptedReranker:
    """Score each passage by a fixed table keyed by its text."""

    model_name = "scripted-reranker"

    def __init__(self, scores: Mapping[str, float]) -> None:
        self.scores = scores

    def score(self, query: str, passages: Sequence[str]) -> tuple[float, ...]:
        return tuple(self.scores.get(text, 0.0) for text in passages)


class FailingReranker:
    model_name = "broken-reranker"

    def score(self, query: str, passages: Sequence[str]) -> tuple[float, ...]:
        raise ModelUnavailable("reranker unavailable")


def _candidate(
    passage_id: str,
    text: str,
    path: str,
    dense_score: float = 0.5,
    *,
    page: int | None = None,
    heading_path: tuple[str, ...] = (),
    rank: int | None = None,
) -> RetrievalCandidate:
    file_type = FileType.PDF if path.endswith(".pdf") else FileType.TXT
    source = Source(path, path, file_type)
    passage = Passage(passage_id, path, text, 0, len(text.split()), page, heading_path)
    return RetrievalCandidate(passage, source, dense_score, stage1_rank=rank)


def _finder(
    retriever: PassageRetriever,
    *,
    reranker: Reranker | None = None,
    collapse: bool = True,
    top_n: int = 3,
) -> FindPassages:
    return FindPassages(
        embedder=DeterministicEmbedder(),
        sparse_encoder=BM25SparseEncoder(),
        retriever=retriever,
        reranker=reranker or LexicalReranker(),
        top_k=100,
        default_top_n=top_n,
        top_n_max=10,
        max_query_len=100,
        collapse_duplicates=collapse,
    )


def _ids(result: SearchResult) -> list[str]:
    return [item.passage.passage_id for item in result.results]


def test_identical_text_in_two_files_collapses_into_one_result() -> None:
    candidates = (
        _candidate("p-a", _COMET, "copy-a.pdf", page=7),
        _candidate("p-b", _COMET, "copy-b.pdf", page=12),
        _candidate("p-c", "A rare blue comet crosses the sea.", "other.txt"),
    )
    result = _finder(StaticRetriever(candidates)).run("blue comet")

    assert _ids(result) == ["p-a", "p-c"]
    representative = result.results[0]
    assert representative.source.path == "copy-a.pdf"
    assert representative.passage.page_number == 7
    assert len(representative.duplicates) == 1
    duplicate = representative.duplicates[0]
    assert duplicate.passage.passage_id == "p-b"
    assert duplicate.source.path == "copy-b.pdf"
    assert duplicate.source.file_type == FileType.PDF
    assert duplicate.passage.page_number == 12
    assert result.results[1].duplicates == ()
    assert result.total_hits == 3
    assert result.collapsed_duplicates == 1


def test_three_copies_give_two_duplicates_in_ranking_order() -> None:
    candidates = (
        _candidate("p3", _COMET, "c.txt", heading_path=("Part", "Two")),
        _candidate("p1", _COMET, "a.txt"),
        _candidate("p2", _COMET, "b.txt"),
    )
    result = _finder(StaticRetriever(candidates)).run("blue comet")

    assert _ids(result) == ["p1"]
    assert [d.passage.passage_id for d in result.results[0].duplicates] == [
        "p2",
        "p3",
    ]
    assert result.results[0].duplicates[1].passage.heading_path == ("Part", "Two")
    assert result.total_hits == 3
    assert result.collapsed_duplicates == 2


def test_whitespace_variants_collapse_and_the_representative_text_is_its_own() -> None:
    wrapped = "A rare blue\ncomet crosses  the sky.  "
    candidates = (
        _candidate("p1", wrapped, "wrapped.txt"),
        _candidate("p2", _COMET, "plain.txt"),
        _candidate("p3", "A rare  blue comet crosses the sky.", "spaced.txt"),
    )
    result = _finder(StaticRetriever(candidates)).run("blue comet")

    assert _ids(result) == ["p1"]
    # The stored text of the representative, byte for byte, not the grouping key.
    assert result.results[0].passage.text == wrapped
    assert result.results[0].passage.text.encode() == wrapped.encode()
    assert [d.source.path for d in result.results[0].duplicates] == [
        "plain.txt",
        "spaced.txt",
    ]


def test_case_punctuation_and_word_differences_are_not_collapsed() -> None:
    candidates = (
        _candidate("p1", _COMET, "a.txt"),
        _candidate("p2", "a rare blue comet crosses the sky.", "lower.txt"),
        _candidate("p3", "A rare blue comet crosses the sky!", "bang.txt"),
        _candidate("p4", "A rare blue comet crosses the sea.", "word.txt"),
    )
    result = _finder(StaticRetriever(candidates), top_n=5).run("blue comet")

    assert _ids(result) == ["p1", "p2", "p3", "p4"]
    assert all(item.duplicates == () for item in result.results)
    assert result.collapsed_duplicates == 0


def test_freed_slots_are_filled_by_the_next_distinct_passages() -> None:
    texts = {
        "first match text": 0.9,
        "second match text": 0.8,
        "third match text": 0.7,
        "fourth match text": 0.6,
    }
    candidates = (
        _candidate("p1", "first match text", "a.txt"),
        _candidate("p2", "first match text", "b.txt"),
        _candidate("p3", "first  match text", "c.txt"),
        _candidate("p4", "second match text", "a.txt"),
        _candidate("p5", "second match text", "b.txt"),
        _candidate("p6", "third match text", "a.txt"),
        _candidate("p7", "fourth match text", "a.txt"),
    )
    reranker = ScriptedReranker(
        {**texts, "first  match text": texts["first match text"]}
    )

    on = _finder(StaticRetriever(candidates), reranker=reranker).run("match")
    off = _finder(StaticRetriever(candidates), reranker=reranker, collapse=False).run(
        "match"
    )

    assert _ids(on) == ["p1", "p4", "p6"]
    assert [len(item.duplicates) for item in on.results] == [2, 1, 0]
    # The fourth distinct passage was cut, so its group counts nothing; every
    # removed member of the pool is counted, also beyond the cut.
    assert on.collapsed_duplicates == 3
    assert on.total_hits == off.total_hits == 7
    assert _ids(off) == ["p1", "p2", "p3"]


def test_removed_members_beyond_the_cut_are_counted() -> None:
    candidates = (
        _candidate("p1", "alpha", "a.txt", 0.9),
        _candidate("p2", "beta", "a.txt", 0.8),
        _candidate("p3", "beta", "b.txt", 0.7),
    )
    result = _finder(
        StaticRetriever(candidates), reranker=ScriptedReranker({}), top_n=1
    ).run("alpha")

    assert _ids(result) == ["p1"]
    assert result.results[0].duplicates == ()
    assert result.collapsed_duplicates == 1


def test_the_tie_break_rule_picks_the_representative() -> None:
    # Equal reranker scores: the higher dense score wins, then the lower id.
    candidates = (
        _candidate("b-id", _COMET, "b.txt", 0.4),
        _candidate("z-id", _COMET, "z.txt", 0.6),
        _candidate("a-id", _COMET, "a.txt", 0.4),
    )
    for order in (candidates, candidates[::-1], candidates[1:] + candidates[:1]):
        result = _finder(StaticRetriever(order), reranker=ScriptedReranker({})).run(
            "blue comet"
        )
        assert _ids(result) == ["z-id"]
        assert [d.passage.passage_id for d in result.results[0].duplicates] == [
            "a-id",
            "b-id",
        ]


def _store(paths: Sequence[str]) -> InMemoryPassageStore:
    embedder = DeterministicEmbedder()
    sparse = BM25SparseEncoder()
    store = InMemoryPassageStore()
    store.recreate_collection(embedder.dimension)
    for path in paths:
        text = _COMET if path != "other.txt" else "Nothing to see here."
        source = Source(path, path, FileType.TXT)
        passage = Passage(make_passage_id(path, 0), path, text, 0, 7)
        store.upsert(passage, source, embedder.embed((text,))[0], sparse.encode(text))
    store.write_completion_metadata(_METADATA)
    return store


def test_the_representative_does_not_depend_on_the_indexing_order() -> None:
    paths = ["first.txt", "second.txt", "third.txt", "other.txt"]
    outcomes = []
    for order in (paths, paths[::-1], [paths[2], paths[0], paths[3], paths[1]]):
        result = _finder(_store(order)).run("blue comet")
        outcomes.append(
            [
                (
                    item.passage.passage_id,
                    item.source.path,
                    item.score,
                    [d.source.path for d in item.duplicates],
                )
                for item in result.results
            ]
        )
    assert outcomes[0] == outcomes[1] == outcomes[2]
    expected = min(paths[:3], key=lambda path: make_passage_id(path, 0))
    assert outcomes[0][0][1] == expected
    assert len(outcomes[0][0][3]) == 2


def test_a_search_does_not_write_to_the_store() -> None:
    store = _store(["first.txt", "second.txt"])
    vector = DeterministicEmbedder().embed((_COMET,))[0]

    def snapshot() -> object:
        return (
            store.count_passages(),
            store.read_completion_metadata(),
            store.retrieve(vector, {}, 100),
        )

    before = snapshot()
    _finder(store).run("blue comet")
    _finder(store, collapse=False).run("blue comet")

    assert snapshot() == before


def test_rerank_skipped_collapses_in_the_first_stage_order() -> None:
    candidates = (
        _candidate("a-id", _COMET, "late.txt", 0.9, rank=2),
        _candidate("z-id", _COMET, "early.txt", 0.1, rank=0),
        _candidate("m-id", "A rare blue comet crosses the sea.", "sea.txt", rank=1),
    )
    result = _finder(StaticRetriever(candidates), reranker=FailingReranker()).run(
        "blue comet"
    )

    assert result.rerank_skipped
    assert _ids(result) == ["z-id", "m-id"]
    assert [d.passage.passage_id for d in result.results[0].duplicates] == ["a-id"]
    assert result.results[0].reranker_score is None
    assert result.collapsed_duplicates == 1
    assert result.total_hits == 3


def test_scores_of_the_representative_are_unchanged() -> None:
    candidates = (
        _candidate("p1", _COMET, "a.txt", 0.7),
        _candidate("p2", _COMET, "b.txt", 0.7),
    )
    on = _finder(StaticRetriever(candidates)).run("blue comet")
    off = _finder(StaticRetriever(candidates), collapse=False).run("blue comet")

    first_on, first_off = on.results[0], off.results[0]
    assert (first_on.score, first_on.dense_score, first_on.reranker_score) == (
        first_off.score,
        first_off.dense_score,
        first_off.reranker_score,
    )


def test_the_request_overrides_the_configured_default_both_ways() -> None:
    candidates = (
        _candidate("p1", _COMET, "a.txt"),
        _candidate("p2", _COMET, "b.txt"),
    )
    enabled = _finder(StaticRetriever(candidates))
    disabled = _finder(StaticRetriever(candidates), collapse=False)

    assert _ids(enabled.run("comet")) == ["p1"]
    assert _ids(enabled.run("comet", collapse_duplicates=False)) == ["p1", "p2"]
    assert _ids(disabled.run("comet")) == ["p1", "p2"]
    assert _ids(disabled.run("comet", collapse_duplicates=True)) == ["p1"]
    assert enabled.run("comet", collapse_duplicates=None).collapsed_duplicates == 1


def test_collapsing_off_leaves_the_new_fields_unset() -> None:
    candidates = (
        _candidate("p1", _COMET, "a.txt"),
        _candidate("p2", _COMET, "b.txt"),
    )
    result = _finder(StaticRetriever(candidates), collapse=False).run("comet")

    assert result.collapsed_duplicates is None
    assert all(item.duplicates == () for item in result.results)


def test_collapsing_on_reports_zero_for_empty_results() -> None:
    empty = _finder(StaticRetriever(())).run("comet")
    assert empty.results == ()
    assert empty.collapsed_duplicates == 0
