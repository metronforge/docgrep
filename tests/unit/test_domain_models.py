from dataclasses import FrozenInstanceError, fields

import pytest

from docgrep.domain.models import (
    FileType,
    Passage,
    Query,
    RankedPassage,
    SearchResult,
    Source,
    make_passage_id,
)


def test_source_and_passage_construction() -> None:
    source = Source(
        source_id="source-1",
        path="papers/example.pdf",
        file_type=FileType.PDF,
        title="Example",
        author="A. Researcher",
        year=2024,
    )
    passage = Passage(
        passage_id=make_passage_id(source.source_id, 42),
        source_id=source.source_id,
        text="A verbatim passage.",
        chunk_offset=42,
        token_count=4,
        page_number=2,
        heading_path=("Introduction",),
    )
    ranked = RankedPassage(
        passage=passage,
        source=source,
        score=0.75,
        dense_score=-0.2,
        reranker_score=2.1,
    )
    result = SearchResult(
        results=(ranked,),
        total_hits=1,
        took_ms=3.5,
        applied_config=(("embedding_model", "test-model"),),
    )

    assert result.results[0].passage.text == "A verbatim passage."
    assert result.results[0].source.title == "Example"


@pytest.mark.parametrize("text", ["", " \t\n"])
def test_empty_query_is_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="query text"):
        Query(text=text)


@pytest.mark.parametrize("score", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_normalized_score_is_rejected(score: float) -> None:
    source = Source("source-1", "note.md", FileType.MARKDOWN)
    passage = Passage("passage-1", source.source_id, "Exact text", 0, 2)

    with pytest.raises(ValueError, match="score"):
        RankedPassage(passage, source, score, 0.5)


def test_inconsistent_source_is_rejected() -> None:
    source = Source("source-1", "note.md", FileType.MARKDOWN)
    passage = Passage("passage-1", "source-2", "Exact text", 0, 2)

    with pytest.raises(ValueError, match="source"):
        RankedPassage(passage, source, 0.5, 0.5)


def test_models_are_immutable() -> None:
    source = Source("source-1", "note.md", FileType.MARKDOWN)
    passage = Passage("passage-1", source.source_id, "Exact text", 0, 2)
    query = Query("term")
    ranked = RankedPassage(passage, source, 0.5, 0.5)
    result = SearchResult((ranked,), 1, 2.0)

    for model in (source, passage, query, ranked, result):
        field_name = fields(model)[0].name
        with pytest.raises(FrozenInstanceError):
            model.__setattr__(field_name, "value")
        assert not hasattr(model, "__dict__")


def test_passage_id_is_stable_and_unambiguous() -> None:
    assert make_passage_id("source-1", 10) == make_passage_id("source-1", 10)
    assert make_passage_id("source-1", 10) != make_passage_id("source-1", 11)
    assert make_passage_id("source-1", 10) != make_passage_id("source-2", 10)
    assert make_passage_id("a1", 23) != make_passage_id("a12", 3)


def test_invalid_offsets_and_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="offset"):
        make_passage_id("source-1", -1)
    with pytest.raises(ValueError, match="offset"):
        Passage("passage-1", "source-1", "Text", -1, 1)
    with pytest.raises(ValueError, match="token_count"):
        Passage("passage-1", "source-1", "Text", 0, 0)
    with pytest.raises(ValueError, match="page_number"):
        Passage("passage-1", "source-1", "Text", 0, 1, page_number=0)
