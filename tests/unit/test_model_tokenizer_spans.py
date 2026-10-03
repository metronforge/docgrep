"""Span normalisation for offsets returned by real subword tokenizers."""

import pytest

from docgrep.adapters.chunking import TokenChunker
from docgrep.adapters.models.real import ModelTokenizer
from docgrep.domain.errors import ModelUnavailable
from docgrep.domain.models import FileType, ParsedDocument, ParsedSection, Source

Offsets = list[tuple[int, int]]


class OffsetsTokenizer:
    def __init__(self, offsets: object) -> None:
        self.offsets = offsets

    def __call__(
        self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool
    ) -> dict[str, object]:
        return {"offset_mapping": self.offsets}


def _spans(offsets: object) -> tuple[tuple[int, int], ...]:
    return ModelTokenizer(OffsetsTokenizer(offsets), "fake").token_spans("x")


def _assert_contract(text: str, spans: tuple[tuple[int, int], ...]) -> None:
    assert all(0 <= start < end <= len(text) for start, end in spans)
    assert all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))


# Offsets captured from the real multilingual sentencepiece tokenizer with
# special tokens disabled. A lone word-boundary piece shares its start with the
# word piece that follows it, and emoji bytes repeat one span.
CAPTURED = {
    "latin": (
        "The quick brown fox jumps over the lazy dog.",
        [
            (0, 3),
            (4, 9),
            (10, 15),
            (16, 17),
            (16, 19),
            (20, 24),
            (24, 25),
            (26, 30),
            (31, 34),
            (35, 37),
            (37, 39),
            (40, 43),
            (43, 44),
        ],
        [
            (0, 3),
            (4, 9),
            (10, 15),
            (16, 19),
            (20, 24),
            (24, 25),
            (26, 30),
            (31, 34),
            (35, 37),
            (37, 39),
            (40, 43),
            (43, 44),
        ],
    ),
    "background": (
        "The archive catalog describes a quiet reading room.",
        [
            (0, 3),
            (4, 5),
            (4, 11),
            (12, 19),
            (20, 28),
            (28, 29),
            (30, 31),
            (32, 37),
            (38, 45),
            (46, 50),
            (50, 51),
        ],
        [
            (0, 3),
            (4, 11),
            (12, 19),
            (20, 28),
            (28, 29),
            (30, 31),
            (32, 37),
            (38, 45),
            (46, 50),
            (50, 51),
        ],
    ),
    "emoji": (
        "a \U0001f600 b",
        [(0, 1), (2, 3), (2, 3), (4, 5)],
        [(0, 1), (2, 3), (4, 5)],
    ),
}


@pytest.mark.parametrize("name", sorted(CAPTURED))
def test_captured_real_offsets_become_disjoint_spans(name: str) -> None:
    text, raw, expected = CAPTURED[name]
    spans = ModelTokenizer(OffsetsTokenizer(raw), "fake").token_spans(text)

    assert list(spans) == expected
    _assert_contract(text, spans)


@pytest.mark.parametrize("name", sorted(CAPTURED))
def test_chunker_accepts_captured_real_offsets(name: str) -> None:
    text, raw, _ = CAPTURED[name]
    tokenizer = ModelTokenizer(OffsetsTokenizer(raw), "fake")
    document = ParsedDocument(
        source=Source("a.txt", "a.txt", FileType.TXT),
        sections=(ParsedSection(text),),
    )

    passages = TokenChunker(tokenizer, chunk_size=4, chunk_overlap=1).chunk(document)

    assert passages
    assert all(passage.text in text for passage in passages)
    assert passages[0].text == text[: len(passages[0].text)]


@pytest.mark.parametrize(
    ("offsets", "expected"),
    [
        ([], []),
        ([(0, 0), (0, 0)], []),
        ([(0, 0), (0, 3), (3, 3), (4, 6)], [(0, 3), (4, 6)]),
        ([(2, 4), (2, 4), (2, 4)], [(2, 4)]),
        ([(0, 2), (2, 4)], [(0, 2), (2, 4)]),
        ([(0, 1), (0, 5)], [(0, 5)]),
        ([(0, 5), (0, 1)], [(0, 5)]),
        ([(0, 3), (2, 6), (5, 8)], [(0, 8)]),
        ([(0, 1), (4, 6), (2, 7)], [(0, 1), (2, 7)]),
        ([(5, 6), (1, 2)], [(1, 6)]),
        ([[0, 2], [2, 3]], [(0, 2), (2, 3)]),
    ],
)
def test_normalisation_rules(offsets: object, expected: list[tuple[int, int]]) -> None:
    assert list(_spans(offsets)) == expected


def test_malformed_offsets_still_raise_model_unavailable() -> None:
    for offsets in ("abc", [(0,)], [(0, "x")], None):
        with pytest.raises(ModelUnavailable):
            _spans(offsets)
