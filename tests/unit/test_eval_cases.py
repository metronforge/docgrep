"""Cases files, hits, ranks and aggregates of the retrieval evaluation
(ADR-0024)."""

import json

import pytest

from docgrep.domain.errors import ValidationError
from docgrep.domain.evaluation import (
    CaseResult,
    EvalCase,
    ListRank,
    aggregate,
    list_rank,
    normalize_whitespace,
    parse_cases,
)
from docgrep.domain.models import FileType
from docgrep.domain.sources import SourceFilter


def _line(**fields: object) -> str:
    return json.dumps(fields)


def test_a_valid_file_gives_its_cases_in_order() -> None:
    text = "\n".join(
        [
            _line(id="first", query="comet orbit", source="books/a.md", note="why"),
            _line(query="tides", source_glob="books/*.md", fragment="rigid  bodies"),
            _line(
                id=7,
                query="kepler",
                fragment="Mars",
                filters={"file_types": ["TXT"], "year_from": 1990},
            ),
        ]
    )

    cases = parse_cases(text)

    assert cases == (
        EvalCase("first", 1, "comet orbit", source="books/a.md", note="why"),
        EvalCase("2", 2, "tides", source_glob="books/*.md", fragment="rigid  bodies"),
        EvalCase(
            "7",
            3,
            "kepler",
            fragment="Mars",
            filters=SourceFilter(file_types=(FileType.TXT,), year_from=1990),
        ),
    )


def test_comments_and_blank_lines_are_skipped_and_counted() -> None:
    text = "# seed cases\n\n   \n  # indented comment\n" + _line(
        query="q", fragment="f"
    )

    (case,) = parse_cases(text)

    assert (case.case_id, case.line) == ("5", 5)


@pytest.mark.parametrize(
    ("line", "message"),
    [
        (_line(query="q", source="a", colour="red"), "line 2: unknown fields: colour"),
        (_line(query="q", note="n"), "line 2: one of source, source_glob or fragment"),
        (_line(source="a"), "line 2: query is required"),
        (_line(query="  ", source="a"), "line 2: query must be a non-empty string"),
        ('{"query": "q", "source": ', "line 2: invalid JSON"),
        ("[1, 2]", "line 2: a case must be a JSON object"),
        (_line(query="q", fragment=""), "line 2: fragment must be a non-empty"),
        (_line(id=True, query="q", fragment="f"), "line 2: id must be"),
        (
            _line(query="q", fragment="f", filters={"colour": []}),
            "line 2: unknown filter fields: colour",
        ),
        (
            _line(query="q", fragment="f", filters={"authors": "x"}),
            "line 2: filters.authors must be a list of strings",
        ),
        (
            _line(query="q", fragment="f", filters={"year_to": "1990"}),
            "line 2: filters.year_to must be an integer",
        ),
        (
            _line(query="q", fragment="f", filters={"file_types": ["exe"]}),
            "line 2: filters:",
        ),
        (_line(id="1", query="q", fragment="f"), "line 2: id '1' repeats line 1"),
    ],
)
def test_a_malformed_line_is_named(line: str, message: str) -> None:
    text = _line(id="1", query="first", fragment="x") + "\n" + line

    with pytest.raises(ValidationError) as raised:
        parse_cases(text)

    assert str(raised.value).startswith(message), str(raised.value)
    assert raised.value.details["line"] == 2


def test_a_file_without_cases_is_malformed() -> None:
    with pytest.raises(ValidationError, match="no case"):
        parse_cases("# nothing yet\n\n")


def test_whitespace_normalization_collapses_runs() -> None:
    assert normalize_whitespace("  a\n\tb   c ") == "a b c"


def test_a_source_hit() -> None:
    case = EvalCase("c", 1, "q", source="root/books/a.md")

    assert case.is_hit("root/books/a.md", None)
    assert not case.is_hit("root/books/b.md", "anything")


def test_a_glob_hit_uses_the_stored_path() -> None:
    case = EvalCase("c", 1, "q", source_glob="books/**/*.md")

    assert case.is_hit("books/x/a.md", None)
    assert not case.is_hit("papers/a.md", None)
    assert EvalCase("c", 1, "q", source_glob="a.md").is_hit("deep/dir/a.md", None)


def test_a_fragment_hit_ignores_whitespace_but_not_case() -> None:
    case = EvalCase("c", 1, "q", fragment="context\n  engineering")

    assert case.is_hit("any.md", "about context engineering, in short")
    assert case.is_hit("any.md", "about context\tengineering")
    assert not case.is_hit("any.md", "about Context Engineering")
    assert not case.is_hit("any.md", None)


def test_every_expectation_must_hold() -> None:
    case = EvalCase("c", 1, "q", source="a.md", source_glob="*.md", fragment="x y")

    assert case.is_hit("a.md", "w x  y z")
    assert not case.is_hit("a.md", "w x z")
    assert not case.is_hit("b.md", "x y")


def test_no_hit_in_a_list() -> None:
    case = EvalCase("c", 1, "q", source="a.md", fragment="needle")

    assert list_rank(case, [[("b.md", "needle")], [("a.md", "hay")]]) == ListRank(
        None, 2
    )


def test_ranks_are_the_best_placed_hit_and_source() -> None:
    case = EvalCase("c", 1, "q", source="a.md", fragment="needle")
    entries = [
        [("b.md", "needle")],
        [("a.md", "hay")],
        [("c.md", "x"), ("a.md", "the needle")],
        [("a.md", "needle")],
    ]

    assert list_rank(case, entries) == ListRank(3, 2)
    # Without a named source no source rank is reported.
    fragment_only = EvalCase("c", 1, "q", fragment="needle")
    assert list_rank(fragment_only, entries) == ListRank(1, None)


def _result(case_id: str, **ranks: int | None) -> CaseResult:
    names = ("dense", "sparse", "fused_deep", "stage1", "final")
    stage1 = ranks.get("stage1")
    return CaseResult(
        EvalCase(case_id, 1, "q", source="a.md"),
        {name: ListRank(ranks.get(name)) for name in names},
        within_top_k=stage1 is not None,
        rerank_score=None,
        control_rank=ranks.get("control"),
    )


def test_aggregates_on_a_hand_computed_example() -> None:
    results = [
        _result("top", dense=1, sparse=2, fused_deep=1, stage1=1, final=1),
        _result("third", dense=5, fused_deep=4, stage1=4, final=3),
        _result("demoted", dense=9, sparse=1, fused_deep=2, stage1=2, final=12),
        _result("first_stage", dense=40, sparse=30, fused_deep=35, control=1),
        _result("missing"),
    ]

    figures = aggregate(results)

    assert figures.cases == 5
    assert figures.found == {
        "dense": 4,
        "sparse": 3,
        "fused_deep": 4,
        "stage1": 3,
        "final": 3,
        "control": 1,
    }
    assert figures.recall_at == {1: 0.2, 3: 0.4, 5: 0.4, 10: 0.4, 25: 0.6}
    assert figures.recall_stage1 == pytest.approx(0.6)
    assert figures.recall_dense == pytest.approx(0.8)
    assert figures.recall_sparse == pytest.approx(0.6)
    assert figures.recall_fused_deep == pytest.approx(0.8)
    assert figures.mrr == pytest.approx((1 + 1 / 3 + 1 / 12) / 5)
    assert figures.lost_in_first_stage == ("first_stage",)
    assert figures.lost_in_rerank == ("demoted",)
    assert figures.not_found == ("missing",)


def test_the_report_keys_have_a_fixed_order() -> None:
    result = _result("x", dense=1)

    assert list(result.as_dict()) == [
        "id",
        "line",
        "query",
        "note",
        "expected",
        "ranks",
        "within_top_k",
        "rerank_score",
        "control_rank",
        "control_score",
    ]
    assert list(aggregate([result]).as_dict()) == [
        "cases",
        "found",
        "recall_at",
        "recall_stage1_top_k",
        "recall_dense_depth",
        "recall_sparse_depth",
        "recall_fused_deep_depth",
        "mrr",
        "lost_in_first_stage",
        "lost_in_rerank",
        "not_found",
    ]
