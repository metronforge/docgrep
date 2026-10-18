"""Pure rules of an update plan: move pairs and the removal guard (ADR-0027)."""

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from docgrep.domain.models import Source

# Defaults of DOCGREP_UPDATE_GUARD_RATIO and DOCGREP_UPDATE_GUARD_COUNT.
DEFAULT_GUARD_RATIO = 0.05
DEFAULT_GUARD_COUNT = 50
# Below this many recorded sources a root is guarded by the count alone: a
# ratio of a handful of sources would refuse almost every removal.
GUARD_SMALL_ROOT = 20


@dataclass(frozen=True, slots=True)
class Move:
    """A removed source and an added file with the same content and size.

    Information only: both are still processed as a removal and an addition.
    """

    removed: str
    added: str


def pair_moves(removed: Iterable[Source], added: Iterable[Source]) -> tuple[Move, ...]:
    """Pair each removed source with at most one added file of the same
    content hash and size.

    The removed sources are taken in stored path order, and each gets the
    first unpaired added file in stored path order, so the pairs are one to
    one and deterministic. A source without a content hash or size never
    pairs.
    """
    candidates: dict[tuple[str, int], list[str]] = {}
    for source in sorted(added, key=lambda source: source.path):
        if source.content_hash is not None and source.size is not None:
            key = (source.content_hash, source.size)
            candidates.setdefault(key, []).append(source.path)
    moves = []
    for source in sorted(removed, key=lambda source: source.path):
        if source.content_hash is None or source.size is None:
            continue
        waiting = candidates.get((source.content_hash, source.size))
        if waiting:
            moves.append(Move(source.path, waiting.pop(0)))
    return tuple(moves)


@dataclass(frozen=True, slots=True)
class RemovalGuard:
    """How many removals one root may take in an update without ``--yes``."""

    ratio: float = DEFAULT_GUARD_RATIO
    count: int = DEFAULT_GUARD_COUNT

    def __post_init__(self) -> None:
        if not 0 <= self.ratio <= 1:
            raise ValueError("ratio must be between 0 and 1")
        if self.count < 0:
            raise ValueError("count must be non-negative")

    def threshold(self, recorded: int) -> int:
        """The most removals allowed for a root with ``recorded`` sources:
        ``min(ratio * recorded, count)`` rounded down, or ``count`` for a
        small root. Removals are whole, so rounding down changes nothing."""
        if recorded < GUARD_SMALL_ROOT:
            return self.count
        # Rounded first so that a product such as 0.05 * 60 counts as 3.
        return min(math.floor(round(self.ratio * recorded, 9)), self.count)


# Why a root is guarded: its removals exceed the threshold, or they would
# remove every source it records, whatever its size and the thresholds.
GuardReason = Literal["threshold", "empties_root"]


@dataclass(frozen=True, slots=True)
class GuardedRoot:
    """A root whose planned removals the guard refuses, and why."""

    root: str
    removals: int
    recorded: int
    threshold: int
    reason: GuardReason = "threshold"


def guarded_roots(
    removals: Mapping[str, int],
    recorded: Mapping[str, int],
    guard: RemovalGuard,
) -> tuple[GuardedRoot, ...]:
    """The roots, by name, whose planned removals the guard refuses.

    A root is refused when its removals would remove every source it
    records (``empties_root``: an existing but empty folder, such as a mount
    point without its disk, passes the check that a root can be read), or
    else when they exceed the threshold (``threshold``).

    ``removals`` counts the planned removals of each root of the run (moved
    sources and sources no longer matched by the patterns included) and
    ``recorded`` the sources the index records for it. Empty: the update may
    proceed.
    """
    found = []
    for root in sorted(removals):
        count = removals[root]
        total = recorded.get(root, 0)
        limit = guard.threshold(total)
        if count > 0 and count == total:
            found.append(GuardedRoot(root, count, total, limit, "empties_root"))
        elif count > limit:
            found.append(GuardedRoot(root, count, total, limit))
    return tuple(found)
