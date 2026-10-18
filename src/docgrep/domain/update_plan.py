"""The removal guard of an update (ADR-0027): a pure rule."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

# Defaults of DOCGREP_UPDATE_GUARD_RATIO and DOCGREP_UPDATE_GUARD_COUNT.
DEFAULT_GUARD_RATIO = 0.05
DEFAULT_GUARD_COUNT = 50
# Below this many recorded sources a root is guarded by the count alone: a
# ratio of a handful of sources would refuse almost every removal.
GUARD_SMALL_ROOT = 20


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


@dataclass(frozen=True, slots=True)
class GuardedRoot:
    """A root whose planned removals exceed the guard's threshold."""

    root: str
    removals: int
    recorded: int
    threshold: int


def guarded_roots(
    removals: Mapping[str, int],
    recorded: Mapping[str, int],
    guard: RemovalGuard,
) -> tuple[GuardedRoot, ...]:
    """The roots, by name, whose planned removals exceed the threshold.

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
        if count > limit:
            found.append(GuardedRoot(root, count, total, limit))
    return tuple(found)
