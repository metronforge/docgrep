"""Token-aware passage chunking contracts."""

import random

import pytest

from docgrep.adapters.chunking import TokenChunker, WhitespaceTokenizer
from docgrep.domain.models import (
    FileType,
    ParsedDocument,
    ParsedSection,
    Source,
    make_passage_id,
)
from docgrep.domain.protocols import Chunker, TokenSpanTokenizer


def document(*sections: ParsedSection) -> ParsedDocument:
    source = Source(source_id="note", path="note.txt", file_type=FileType.TXT)
    return ParsedDocument(source=source, sections=sections)


def test_chunker_keeps_verbatim_text_and_stable_offsets() -> None:
    tokenizer: TokenSpanTokenizer = WhitespaceTokenizer()
    chunker: Chunker = TokenChunker(tokenizer, chunk_size=4, chunk_overlap=1)
    parsed = document(ParsedSection("one two three four five six seven"))

    passages = chunker.chunk(parsed)

    assert [passage.text for passage in passages] == [
        "one two three four",
        "four five six seven",
    ]
    assert [passage.chunk_offset for passage in passages] == [0, 14]
    assert [passage.token_count for passage in passages] == [4, 4]
    assert all(
        passage.passage_id == make_passage_id("note", passage.chunk_offset)
        for passage in passages
    )
    assert chunker.chunk(parsed) == passages


def test_chunker_prefers_paragraph_boundary_and_keeps_overlap() -> None:
    parsed = document(ParsedSection("one two three\n\nfour five six seven eight"))

    passages = TokenChunker(WhitespaceTokenizer(), 5, 1).chunk(parsed)

    assert passages[0].text == "one two three"
    assert passages[1].text.startswith("three\n\nfour")
    assert {"three"} <= set(passages[1].text.split())
    assert all(passage.token_count <= 5 for passage in passages)


def test_chunks_do_not_cross_pages_or_headings() -> None:
    parsed = document(
        ParsedSection("one two three four", page_number=1),
        ParsedSection("five six seven eight", page_number=2),
        ParsedSection("nine ten", heading_path=("Chapter", "Section")),
    )

    passages = TokenChunker(WhitespaceTokenizer(), 3, 1).chunk(parsed)

    assert [passage.page_number for passage in passages] == [1, 1, 2, 2, None]
    assert passages[-1].heading_path == ("Chapter", "Section")
    assert len({passage.chunk_offset for passage in passages}) == len(passages)
    assert [passage.chunk_offset for passage in passages] == sorted(
        passage.chunk_offset for passage in passages
    )


def test_random_paragraphs_preserve_limits_overlap_and_identity() -> None:
    randomizer = random.Random(1789)
    for _ in range(100):
        paragraphs = [
            " ".join(f"word{randomizer.randrange(100)}" for _ in range(length))
            for length in (
                randomizer.randrange(1, 14) for _ in range(randomizer.randrange(1, 9))
            )
        ]
        text = "\n\n".join(paragraphs)
        chunk_size = randomizer.randrange(2, 12)
        overlap = randomizer.randrange(chunk_size)
        parsed = document(ParsedSection(text))

        passages = TokenChunker(WhitespaceTokenizer(), chunk_size, overlap).chunk(
            parsed
        )

        assert passages
        assert all(1 <= passage.token_count <= chunk_size for passage in passages)
        assert all(
            text[passage.chunk_offset : passage.chunk_offset + len(passage.text)]
            == passage.text
            for passage in passages
        )
        assert all(
            passage.passage_id == make_passage_id("note", passage.chunk_offset)
            for passage in passages
        )
        assert len({passage.chunk_offset for passage in passages}) == len(passages)
        tokens = text.split()
        covered = set()
        for passage in passages:
            covered.update(passage.text.split())
        assert covered == set(tokens)
        for previous, current in zip(passages, passages[1:], strict=False):
            if overlap:
                assert (
                    previous.text.split()[-overlap:] == current.text.split()[:overlap]
                )


@pytest.mark.parametrize("chunk_size,overlap", [(0, 0), (2, -1), (2, 2)])
def test_invalid_chunk_limits_raise(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        TokenChunker(WhitespaceTokenizer(), chunk_size, overlap)
