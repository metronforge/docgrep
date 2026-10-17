"""Window selection and exact stitching of the context around a passage
(UC-12, ADR-0022), against the real chunker's offsets."""

import random
from collections.abc import Sequence

import pytest

from docgrep.adapters.chunking import TokenChunker, WhitespaceTokenizer
from docgrep.domain.context import (
    SECTION_BREAK,
    SECTION_GAP,
    AssembledContext,
    ContextWindow,
    OutlineEntry,
    PassageLocation,
    select_window,
    stitch,
)
from docgrep.domain.models import (
    FileType,
    ParsedDocument,
    ParsedSection,
    Passage,
    Source,
    is_passage_id,
    make_passage_id,
)

# The chunker leaves one character between sections that belongs to no
# passage; any character works, and it must never reach a window.
_SEPARATOR = "\x00"
_WINDOWS = (0, 1, 3, 5, 8, 13, 21, 1000)
_CHUNKING = ((8, 2), (8, 0), (5, 4), (16, 3), (3, 1))


def _words(count: int, start: int = 0) -> str:
    return " ".join(f"w{number}" for number in range(start, start + count))


def _pdf_like() -> tuple[ParsedSection, ...]:
    """Pages with heading paths, paragraph gaps, uneven whitespace and a
    page shorter than one chunk."""
    return (
        ParsedSection(
            "  " + _words(20) + "\n\n" + _words(9, 20) + "  \n", 1, ("Part I",)
        ),
        ParsedSection(_words(3, 40), 2, ("Part I",)),
        ParsedSection(
            _words(11, 50) + "\n \n\t" + _words(17, 61), 3, ("Part I", "Chapter 1")
        ),
        ParsedSection("\n" + _words(26, 90) + "\n", 4, ("Part II",)),
    )


def _docx_like() -> tuple[ParsedSection, ...]:
    """No pages; two consecutive sections share a heading path, and one
    section is a single word."""
    return (
        ParsedSection(_words(14) + "\n\n" + _words(14, 14), None, ("Intro",)),
        ParsedSection(_words(19, 30), None, ("Intro",)),
        ParsedSection("lonely", None, ("Intro", "Aside")),
        ParsedSection(
            _words(6, 60) + "\n\n" + _words(6, 66) + "\n\n" + _words(30, 72),
            None,
            ("Body",),
        ),
    )


def _random_like(seed: int) -> tuple[ParsedSection, ...]:
    """Words joined by random runs of whitespace, in sections of random
    length, pages and headings."""
    generator = random.Random(seed)
    sections = []
    for number in range(generator.randint(1, 6)):
        parts = []
        for word in range(generator.randint(1, 45)):
            parts.append(f"t{number}_{word}")
            parts.append(generator.choice((" ", "  ", "\n", "\n\n", " \t ", "\n \n")))
        text = generator.choice(("", " ", "\n")) + "".join(parts)
        page = generator.choice((None, number + 1))
        heading = generator.choice(((), ("H",), (f"H{number}",)))
        sections.append(ParsedSection(text, page, heading))
    return tuple(sections)


_DOCUMENTS = {
    "pdf": (FileType.PDF, _pdf_like()),
    "docx": (FileType.DOCX, _docx_like()),
    **{f"random-{seed}": (FileType.TXT, _random_like(seed)) for seed in range(12)},
}


def _chunk(
    sections: tuple[ParsedSection, ...], file_type: FileType, size: int, overlap: int
) -> tuple[Passage, ...]:
    source = Source("src", "doc", file_type)
    chunker = TokenChunker(WhitespaceTokenizer(), size, overlap)
    return chunker.chunk(ParsedDocument(source, sections))


def _outline(passages: Sequence[Passage]) -> tuple[OutlineEntry, ...]:
    return tuple(
        OutlineEntry(item.passage_id, item.chunk_offset, item.token_count)
        for item in passages
    )


def _expand(
    passages: Sequence[Passage], index: int, window_tokens: int
) -> tuple[ContextWindow, AssembledContext]:
    by_id = {item.passage_id: item for item in passages}
    target = passages[index]
    outline = _outline(passages)
    window = select_window(outline, outline[index], window_tokens)
    context = stitch(
        target,
        [by_id[entry.passage_id] for entry in window.before],
        [by_id[entry.passage_id] for entry in window.after],
        at_source_start=window.at_source_start,
        at_source_end=window.at_source_end,
    )
    return window, context


def _same_section(left: Passage, right: Passage) -> bool:
    return (left.page_number, left.heading_path) == (
        right.page_number,
        right.heading_path,
    )


def _expected(stream: str, chosen: Sequence[Passage]) -> str:
    """The window read from the section stream, not from passage text: the
    stream between the first and the last passage, each uncovered gap
    replaced by its join."""
    parts: list[str] = []
    position = chosen[0].chunk_offset
    previous = chosen[0]
    for passage in chosen:
        end = passage.chunk_offset + len(passage.text)
        if end <= position:
            continue
        if passage.chunk_offset > position:
            parts.append(
                SECTION_GAP if _same_section(previous, passage) else SECTION_BREAK
            )
            position = passage.chunk_offset
        parts.append(stream[position:end])
        position = end
        previous = passage
    return "".join(parts)


@pytest.mark.parametrize(("size", "overlap"), _CHUNKING)
@pytest.mark.parametrize("name", sorted(_DOCUMENTS))
def test_every_window_is_a_contiguous_slice_of_the_section_stream(
    name: str, size: int, overlap: int
) -> None:
    file_type, sections = _DOCUMENTS[name]
    stream = _SEPARATOR.join(section.text for section in sections)
    passages = _chunk(sections, file_type, size, overlap)
    assert passages
    by_id = {item.passage_id: item for item in passages}

    for index, target in enumerate(passages):
        for window_tokens in _WINDOWS:
            window, context = _expand(passages, index, window_tokens)
            chosen = [
                *(by_id[entry.passage_id] for entry in window.before),
                target,
                *(by_id[entry.passage_id] for entry in window.after),
            ]
            # The neighbours are consecutive passages of the source.
            first = passages.index(chosen[0])
            assert chosen == list(passages[first : first + len(chosen)])

            whole = context.before + target.text + context.after
            assert whole == _expected(stream, chosen), (index, window_tokens)
            assert _SEPARATOR not in whole
            assert whole[len(context.before) :].startswith(target.text)
            assert len(context.before_locations) == len(window.before)
            assert len(context.after_locations) == len(window.after)


def test_the_fixtures_exercise_overlaps_gaps_and_short_sections() -> None:
    """Guards the property test against fixtures that never overlap."""
    _, pdf = _DOCUMENTS["pdf"]
    overlapping = _chunk(pdf, FileType.PDF, 8, 2)
    assert any(
        left.chunk_offset + len(left.text) > right.chunk_offset
        for left, right in zip(overlapping, overlapping[1:])
    )
    gapped = _chunk(pdf, FileType.PDF, 8, 0)
    assert all(
        left.chunk_offset + len(left.text) < right.chunk_offset
        for left, right in zip(gapped, gapped[1:])
    )
    # Page 2 holds three words: a section shorter than one chunk.
    assert [item.text for item in overlapping if item.page_number == 2] == [
        _words(3, 40)
    ]
    _, docx = _DOCUMENTS["docx"]
    assert all(item.page_number is None for item in _chunk(docx, FileType.DOCX, 8, 2))


def test_overlap_is_cut_with_the_offsets() -> None:
    sections = (ParsedSection("a b c d e f g h i j"),)
    passages = _chunk(sections, FileType.TXT, 4, 1)
    # Offsets 0, 6 and 12: each passage repeats the last word of the one before.
    assert [(item.chunk_offset, item.text) for item in passages] == [
        (0, "a b c d"),
        (6, "d e f g"),
        (12, "g h i j"),
    ]

    _, context = _expand(passages, 1, 1000)

    # A keeps text[: 6 - 0]; C keeps text[(6 + 7) - 12 :].
    assert context.before == "a b c "
    assert context.after == " h i j"
    assert context.before + passages[1].text + context.after == sections[0].text


def test_gaps_join_with_a_space_inside_a_section_and_a_blank_line_across() -> None:
    sections = (
        ParsedSection("one two three  four", 1, ("H",)),
        ParsedSection("five six", 2, ("H",)),
    )
    passages = _chunk(sections, FileType.PDF, 2, 0)
    assert [item.text for item in passages] == ["one two", "three  four", "five six"]

    _, context = _expand(passages, 1, 1000)

    assert context.before == "one two" + SECTION_GAP
    assert context.after == SECTION_BREAK + "five six"
    assert context.before_locations == (PassageLocation(1, ("H",)),)
    assert context.after_locations == (PassageLocation(2, ("H",)),)


def test_touching_passages_join_with_nothing() -> None:
    first = Passage(make_passage_id("s", 0), "s", "ab", 0, 1)
    second = Passage(make_passage_id("s", 2), "s", "cd", 2, 1, page_number=3)

    context = stitch(first, (), (second,), at_source_start=True, at_source_end=True)

    assert context.after == "cd"


def test_the_window_takes_neighbours_while_their_tokens_fit() -> None:
    outline = tuple(
        OutlineEntry(f"p{index}", index * 10, count)
        for index, count in enumerate((4, 3, 5, 2, 6, 4, 1))
    )

    window = select_window(outline, outline[3], 8)

    # Before: 5 fits, 5 + 3 = 8 fits, + 4 does not; after: 6 fits, + 4 not.
    assert [entry.passage_id for entry in window.before] == ["p1", "p2"]
    assert [entry.passage_id for entry in window.after] == ["p4"]
    assert (window.at_source_start, window.at_source_end) == (False, False)


def test_the_nearest_neighbour_is_taken_even_above_the_window() -> None:
    outline = (
        OutlineEntry("a", 0, 500),
        OutlineEntry("b", 10, 500),
        OutlineEntry("c", 20, 500),
    )

    window = select_window(outline, outline[1], 1)

    assert [entry.passage_id for entry in window.before] == ["a"]
    assert [entry.passage_id for entry in window.after] == ["c"]
    assert (window.at_source_start, window.at_source_end) == (True, True)


@pytest.mark.parametrize(("size", "overlap"), _CHUNKING)
@pytest.mark.parametrize("name", sorted(_DOCUMENTS))
def test_the_budget_is_respected_and_used_up(
    name: str, size: int, overlap: int
) -> None:
    file_type, sections = _DOCUMENTS[name]
    outline = _outline(_chunk(sections, file_type, size, overlap))
    for index, target in enumerate(outline):
        for window_tokens in _WINDOWS:
            window = select_window(outline, target, window_tokens)
            sides = (
                (window.before[::-1], outline[:index][::-1], window.at_source_start),
                (window.after, outline[index + 1 :], window.at_source_end),
            )
            for taken, candidates, at_boundary in sides:
                used = sum(entry.token_count for entry in taken)
                assert tuple(taken) == candidates[: len(taken)]
                if window_tokens > 0 and candidates:
                    assert taken, "the nearest neighbour is always taken"
                assert len(taken) <= 1 or used <= window_tokens
                if len(taken) < len(candidates) and window_tokens > 0:
                    assert used + candidates[len(taken)].token_count > window_tokens
                assert at_boundary == (len(taken) == len(candidates))


def test_a_zero_window_returns_the_passage_alone() -> None:
    passages = _chunk(_pdf_like(), FileType.PDF, 8, 2)

    for index in range(len(passages)):
        window, context = _expand(passages, index, 0)
        assert (window.before, window.after) == ((), ())
        assert (context.before, context.after) == ("", "")
        assert (context.tokens_before, context.tokens_after) == (0, 0)
        assert context.at_source_start == (index == 0)
        assert context.at_source_end == (index == len(passages) - 1)


def test_the_first_and_last_passages_set_the_boundary_flags() -> None:
    passages = _chunk(_docx_like(), FileType.DOCX, 8, 2)

    _, first = _expand(passages, 0, 8)
    _, last = _expand(passages, len(passages) - 1, 8)

    assert first.at_source_start and first.before == "" and not first.at_source_end
    assert last.at_source_end and last.after == "" and not last.at_source_start


def test_a_source_with_a_single_passage() -> None:
    (passage,) = _chunk((ParsedSection("only words"),), FileType.TXT, 8, 2)

    window, context = _expand((passage,), 0, 500)

    assert (window.before, window.after) == ((), ())
    assert (context.before, context.after) == ("", "")
    assert context.at_source_start and context.at_source_end


def test_the_used_tokens_are_the_stored_counts_of_the_neighbours() -> None:
    passages = _chunk(_pdf_like(), FileType.PDF, 8, 2)

    window, context = _expand(passages, 4, 13)

    assert context.tokens_before == sum(entry.token_count for entry in window.before)
    assert context.tokens_after == sum(entry.token_count for entry in window.after)


def test_a_target_missing_from_the_outline_is_placed_by_its_offset() -> None:
    outline = (OutlineEntry("a", 0, 2), OutlineEntry("c", 20, 2))

    window = select_window(outline, OutlineEntry("b", 10, 2), 2)

    assert [entry.passage_id for entry in window.before] == ["a"]
    assert [entry.passage_id for entry in window.after] == ["c"]


def test_stitch_refuses_neighbours_out_of_order_or_from_another_source() -> None:
    target = Passage(make_passage_id("s", 10), "s", "target", 10, 1)
    early = Passage(make_passage_id("s", 0), "s", "early", 0, 1)
    foreign = Passage(make_passage_id("t", 0), "t", "foreign", 0, 1)

    with pytest.raises(ValueError, match="reading order"):
        stitch(target, (), (early,), at_source_start=True, at_source_end=True)
    with pytest.raises(ValueError, match="source"):
        stitch(target, (foreign,), (), at_source_start=True, at_source_end=True)
    with pytest.raises(ValueError, match="non-negative"):
        select_window((), OutlineEntry("x", 0, 1), -1)


def test_a_passage_id_has_the_shape_of_a_derived_id() -> None:
    assert is_passage_id(make_passage_id("src", 12))
    for malformed in ("", "abc", "G" * 64, make_passage_id("s", 0).upper(), " " * 64):
        assert not is_passage_id(malformed)
