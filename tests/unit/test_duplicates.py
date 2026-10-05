"""Grouping of ranked passages with the same normalized text."""

import pytest

from docgrep.domain.duplicates import collapse_duplicates, duplicate_key
from docgrep.domain.models import FileType, Passage, RankedPassage, Source


def _ranked(passage_id: str, text: str, path: str | None = None) -> RankedPassage:
    source_id = path or f"{passage_id}.txt"
    source = Source(source_id, source_id, FileType.TXT)
    passage = Passage(passage_id, source_id, text, 0, len(text.split()))
    return RankedPassage(passage, source, 0.5, 0.5, 0.0)


@pytest.mark.parametrize(
    "variant",
    [
        "one two\nthree",
        "one  two three",
        "one two three  ",
        "\tone two three",
        "one\r\ntwo\u00a0three",
        "one two\n\n   three\n",
    ],
)
def test_whitespace_variants_share_a_key(variant: str) -> None:
    assert duplicate_key(variant) == duplicate_key("one two three")


@pytest.mark.parametrize(
    "variant",
    ["One two three", "one two three.", "one, two three", "one two four"],
)
def test_case_punctuation_and_words_are_not_folded(variant: str) -> None:
    assert duplicate_key(variant) != duplicate_key("one two three")


def test_key_is_nfc_normalized() -> None:
    assert duplicate_key("cafe\u0301 noir") == duplicate_key("caf\u00e9 noir")


def test_first_member_represents_the_group_and_keeps_its_own_text() -> None:
    ordered = (
        _ranked("p1", "same  text\nhere"),
        _ranked("p2", "other text"),
        _ranked("p3", "same text here"),
        _ranked("p4", "same text here "),
    )

    collapsed, removed = collapse_duplicates(ordered)

    assert [item.passage.passage_id for item in collapsed] == ["p1", "p2"]
    assert collapsed[0].passage.text == "same  text\nhere"
    assert [item.passage.passage_id for item in collapsed[0].duplicates] == [
        "p3",
        "p4",
    ]
    assert collapsed[1].duplicates == ()
    assert removed == 2
    # Everything except the attached duplicates is the representative itself.
    assert collapsed[0].passage == ordered[0].passage
    assert collapsed[0].score == ordered[0].score


def test_nothing_to_collapse_returns_the_same_items() -> None:
    ordered = (_ranked("a", "first"), _ranked("b", "second"))
    assert collapse_duplicates(ordered) == (ordered, 0)
    assert collapse_duplicates(()) == ((), 0)
