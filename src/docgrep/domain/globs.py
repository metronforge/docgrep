"""Glob patterns over corpus-relative POSIX paths, shared by the corpus scan
and the filters of read-only commands."""

from collections.abc import Sequence
from fnmatch import fnmatchcase


def _match_segments(pattern: Sequence[str], parts: Sequence[str]) -> bool:
    if not pattern:
        return not parts
    head = pattern[0]
    if head == "**":
        rest = pattern[1:]
        if not rest:
            return len(parts) >= 1
        return any(
            _match_segments(rest, parts[start:]) for start in range(len(parts) + 1)
        )
    if not parts:
        return False
    return fnmatchcase(parts[0], head) and _match_segments(pattern[1:], parts[1:])


def matching_pattern(path: str, patterns: Sequence[str]) -> str | None:
    """Return the first glob pattern a POSIX-relative path matches, or None.

    Matching works on whole path segments. ``*``, ``?`` and ``[...]`` never
    cross ``/``; a ``**`` segment matches zero or more segments, except as the
    last segment, where it matches everything beneath a directory (at least one
    segment). A pattern without ``/`` is compared with the basename at any
    depth. A pattern containing ``/`` is anchored at the corpus root. A single
    leading ``/`` is dropped and anchors the pattern at the corpus root even
    when no other ``/`` remains: ``/draft.md`` matches only a root-level
    ``draft.md``, and ``/notes/private/**`` equals ``notes/private/**``. A
    trailing ``/`` is shorthand for a trailing ``/**``: ``dir/`` matches
    everything beneath ``dir``.
    """
    parts = tuple(path.split("/"))
    for original in patterns:
        pattern = original
        anchored = pattern.startswith("/")
        if anchored:
            pattern = pattern[1:]
        if pattern.endswith("/"):
            pattern += "**"
        if not anchored and "/" not in pattern:
            if fnmatchcase(parts[-1], pattern):
                return original
        elif _match_segments(tuple(pattern.split("/")), parts):
            return original
    return None
