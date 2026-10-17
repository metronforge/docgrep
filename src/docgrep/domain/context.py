"""The text around a passage, assembled from its neighbouring passages (UC-12).

Pure functions without I/O (ADR-0022). Passages of one source carry
``chunk_offset``, a character offset in a stream built from the parsed
sections, and their text is an exact slice of a section. Consecutive
passages of a section overlap; the overlap is removed with the offsets
alone, so every character of the window is taken verbatim from a stored
passage. Only the gaps between passages, whose text is not stored, are
joined: with nothing when the next passage starts where the previous one
ended, with one space inside a section and with a blank line between
sections.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from docgrep.domain.models import Passage, Source

# Joins the pieces of a window across a gap whose text is not stored.
SECTION_GAP = " "
SECTION_BREAK = "\n\n"


@dataclass(frozen=True, slots=True)
class OutlineEntry:
    """Where one passage sits in its source, without its text: enough to
    order the passages of a source and to measure a window."""

    passage_id: str
    chunk_offset: int
    token_count: int


@dataclass(frozen=True, slots=True)
class ContextWindow:
    """The neighbours chosen around a passage, each side in reading order.

    ``at_source_start`` is true when no earlier passage of the source was
    left out, ``at_source_end`` when no later one was.
    """

    before: tuple[OutlineEntry, ...]
    after: tuple[OutlineEntry, ...]
    at_source_start: bool
    at_source_end: bool


@dataclass(frozen=True, slots=True)
class PassageLocation:
    """The page and heading path of one passage of a window."""

    page_number: int | None
    heading_path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AssembledContext:
    """The text before and after a passage, verbatim apart from the joins.

    ``before + passage.text + after`` is the whole window: the join between
    the last piece before the passage and the passage ends ``before``, the
    join between the passage and the first piece after it starts ``after``.
    The locations list the neighbours used, in reading order; the token
    counts add up their stored ``token_count``.
    """

    before: str
    after: str
    before_locations: tuple[PassageLocation, ...]
    after_locations: tuple[PassageLocation, ...]
    at_source_start: bool
    at_source_end: bool
    tokens_before: int
    tokens_after: int


ContextStatus = Literal["ok", "not_found", "source_removed", "source_modified"]


@dataclass(frozen=True, slots=True)
class PassageContext:
    """The outcome for one requested passage id.

    ``ok`` carries the passage, its sanitized source and the context. A
    failure carries a hint and, for ``source_removed``, the last known
    ``path``; for ``source_modified`` also the content hash the passage was
    indexed with and the source's current one in the manifest.
    """

    passage_id: str
    status: ContextStatus
    hint: str | None = None
    passage: Passage | None = None
    source: Source | None = None
    context: AssembledContext | None = None
    path: str | None = None
    indexed_content_hash: str | None = None
    current_content_hash: str | None = None


@dataclass(frozen=True, slots=True)
class ContextResult:
    """One entry per distinct requested id, in request order."""

    entries: tuple[PassageContext, ...]
    window_tokens: int


def _take(candidates: Sequence[OutlineEntry], window_tokens: int) -> list[OutlineEntry]:
    """Neighbours outward while their token counts fit the window; the
    nearest one is always taken when the window is not empty."""
    taken: list[OutlineEntry] = []
    used = 0
    for entry in candidates:
        if window_tokens <= 0 or (taken and used + entry.token_count > window_tokens):
            break
        taken.append(entry)
        used += entry.token_count
    return taken


def select_window(
    outline: Sequence[OutlineEntry], target: OutlineEntry, window_tokens: int
) -> ContextWindow:
    """Choose whole neighbours of ``target`` on each side of it.

    ``outline`` holds the passages of the target's source in any order; the
    target is placed by its own offset when the outline lacks it. Each side
    takes neighbours outward while the running total of their stored token
    counts stays within ``window_tokens``, and always takes the nearest one
    when ``window_tokens`` is positive.
    """
    if window_tokens < 0:
        raise ValueError("window_tokens must be non-negative")
    entries = {entry.passage_id: entry for entry in outline}
    entries[target.passage_id] = target
    ordered = sorted(
        entries.values(), key=lambda entry: (entry.chunk_offset, entry.passage_id)
    )
    index = ordered.index(target)
    earlier = ordered[:index][::-1]
    later = ordered[index + 1 :]
    before = _take(earlier, window_tokens)
    after = _take(later, window_tokens)
    return ContextWindow(
        tuple(reversed(before)),
        tuple(after),
        at_source_start=len(before) == len(earlier),
        at_source_end=len(after) == len(later),
    )


def _join(previous: Passage, previous_end: int, following: Passage, start: int) -> str:
    """What joins two pieces: nothing when they touch, else a space inside
    one section and a blank line across sections."""
    if start <= previous_end:
        return ""
    same_section = (
        previous.page_number == following.page_number
        and previous.heading_path == following.heading_path
    )
    return SECTION_GAP if same_section else SECTION_BREAK


def _location(passage: Passage) -> PassageLocation:
    return PassageLocation(passage.page_number, passage.heading_path)


def stitch(
    target: Passage,
    before: Sequence[Passage],
    after: Sequence[Passage],
    *,
    at_source_start: bool,
    at_source_end: bool,
) -> AssembledContext:
    """Join the neighbours of ``target`` into the text before and after it.

    ``before`` and ``after`` are consecutive passages of the target's source
    in reading order. An earlier passage A followed by B keeps
    ``A.text[: B.chunk_offset - A.chunk_offset]``; a later passage C keeps
    what lies past the end of every passage before it,
    ``C.text[end - C.chunk_offset :]``. Nothing else is cut.
    """
    sequence = (*before, target, *after)
    if any(passage.source_id != target.source_id for passage in sequence):
        raise ValueError("neighbours must belong to the target's source")
    if any(
        left.chunk_offset >= right.chunk_offset
        for left, right in zip(sequence, sequence[1:])
    ):
        raise ValueError("neighbours must be in reading order around the target")

    pieces: list[str] = []
    for passage, following in zip(before, (*before[1:], target)):
        kept = passage.text[: following.chunk_offset - passage.chunk_offset]
        pieces.append(kept)
        pieces.append(
            _join(
                passage,
                passage.chunk_offset + len(kept),
                following,
                following.chunk_offset,
            )
        )

    tail: list[str] = []
    previous = target
    reach = target.chunk_offset + len(target.text)
    for passage in after:
        cut = max(reach - passage.chunk_offset, 0)
        kept = passage.text[cut:]
        end = passage.chunk_offset + len(passage.text)
        if kept:
            tail.append(_join(previous, reach, passage, passage.chunk_offset + cut))
            tail.append(kept)
            previous = passage
        reach = max(reach, end)

    return AssembledContext(
        before="".join(pieces),
        after="".join(tail),
        before_locations=tuple(_location(passage) for passage in before),
        after_locations=tuple(_location(passage) for passage in after),
        at_source_start=at_source_start,
        at_source_end=at_source_end,
        tokens_before=sum(passage.token_count for passage in before),
        tokens_after=sum(passage.token_count for passage in after),
    )
