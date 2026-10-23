"""Pure rules of an update plan: move pairs, reuse by content hash and the
removal guard (ADR-0027, ADR-0028)."""

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Literal

from docgrep.domain.models import (
    ManifestSourceRecord,
    Passage,
    Source,
    make_passage_id,
)
from docgrep.domain.protocols import effective_parser_version

# Defaults of DOCGREP_UPDATE_GUARD_RATIO and DOCGREP_UPDATE_GUARD_COUNT.
DEFAULT_GUARD_RATIO = 0.05
DEFAULT_GUARD_COUNT = 50
# Below this many recorded sources a root is guarded by the count alone: a
# ratio of a handful of sources would refuse almost every removal.
GUARD_SMALL_ROOT = 20


@dataclass(frozen=True, slots=True)
class Move:
    """A removed source and an added file with the same content and size.

    An update whose removed source can be reused (``can_reuse``) copies its
    passages to the added file, then removes it (ADR-0028).
    """

    removed: str
    added: str


@dataclass(frozen=True, slots=True)
class Copy:
    """A file indexed by copying the passages of a source that stays."""

    original: str
    copy: str


def can_reuse(
    record: ManifestSourceRecord,
    file: Source,
    *,
    parser_version: str | None,
    embedding_model: str,
) -> bool:
    """Whether the stored passages of a recorded source can be copied for a
    file instead of parsing and embedding it (ADR-0028).

    The source is indexed, with the content hash, size and file type of the
    file, its recorded parser version is the version the file would be
    parsed with now (``parser_version``, the registered parser's) and its
    embedding model is the index's. OCR delegation is checked by the caller.
    """
    old = record.source
    return (
        record.status == "indexed"
        and old.content_hash is not None
        and old.size is not None
        and old.content_hash == file.content_hash
        and old.size == file.size
        and old.file_type == file.file_type
        and effective_parser_version(old.parser_version)
        == effective_parser_version(parser_version)
        and record.embedding_model == embedding_model
    )


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


def reused_source(original: Source, file: Source) -> Source:
    """The source recorded and stored for a file whose passages are copied
    from ``original``: the identity, size, time and indexing time of the
    file, and what the parser extracted from the content they share."""
    return replace(
        original,
        source_id=file.source_id,
        path=file.path,
        mtime=file.mtime,
        size=file.size,
        indexed_at=file.indexed_at,
    )


def reused_passage(passage: Passage, source_id: str) -> Passage:
    """A stored passage under a new source identity: its id follows from the
    new source id and the same chunk offset; the text and location stay."""
    return replace(
        passage,
        passage_id=make_passage_id(source_id, passage.chunk_offset),
        source_id=source_id,
    )


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
    # Further removals of the root that the guard does not count because each
    # pairs with an added file of the same content (``pair_moves``).
    paired: int = 0


def guarded_roots(
    removals: Mapping[str, int],
    recorded: Mapping[str, int],
    guard: RemovalGuard,
    paired: Mapping[str, int] | None = None,
) -> tuple[GuardedRoot, ...]:
    """The roots, by name, whose planned removals the guard refuses.

    A root is refused when its removals would remove every source it
    records (``empties_root``: an existing but empty folder, such as a mount
    point without its disk, passes the check that a root can be read), or
    else when they exceed the threshold (``threshold``).

    ``removals`` counts the planned removals of each root of the run that
    lose content: sources no longer matched by the patterns included, moved
    sources not, because a removed source that pairs with an added file of
    the same content and size loses nothing (``pair_moves``); a root whose
    sources all moved elsewhere is not emptied. ``recorded`` is the sources
    the index records for the root and ``paired`` the paired removals, which
    are only reported. Empty: the update may proceed.
    """
    found = []
    for root in sorted(removals):
        count = removals[root]
        total = recorded.get(root, 0)
        limit = guard.threshold(total)
        if count > 0 and count == total:
            reason: GuardReason = "empties_root"
        elif count > limit:
            reason = "threshold"
        else:
            continue
        found.append(
            GuardedRoot(root, count, total, limit, reason, (paired or {}).get(root, 0))
        )
    return tuple(found)
