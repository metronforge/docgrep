"""Group ranked passages whose text is the same after whitespace normalization."""

import unicodedata
from collections.abc import Sequence
from dataclasses import replace

from docgrep.domain.models import RankedPassage


def duplicate_key(text: str) -> str:
    """Return the grouping key of a passage text; never shown to the user.

    The text is NFC-normalized, every run of whitespace becomes one space and
    the ends are stripped. Case and punctuation are kept.
    """
    return " ".join(unicodedata.normalize("NFC", text).split())


def collapse_duplicates(
    ordered: Sequence[RankedPassage],
) -> tuple[tuple[RankedPassage, ...], int]:
    """Keep the first passage of each text group and attach the others to it.

    ``ordered`` must already be in the final ranking order: the first member of
    a group is its representative, and the other members follow it as
    ``duplicates`` in their ranking order. Returns the representatives in
    ranking order and the number of passages removed from the ranking.
    """
    groups: dict[str, list[RankedPassage]] = {}
    for item in ordered:
        groups.setdefault(duplicate_key(item.passage.text), []).append(item)
    collapsed = tuple(
        replace(first, duplicates=tuple(rest)) if rest else first
        for first, *rest in groups.values()
    )
    return collapsed, len(ordered) - len(collapsed)
