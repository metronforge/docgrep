"""Corpus roots and the stored identity of their sources (ADR-0023).

A corpus is one or more named folders. The stored path and the ``source_id``
of a source are its path relative to its root behind the root's prefix: the
legacy root, which owns an index built before roots had names, has an empty
prefix, and every other root the prefix ``<name>/``. Passage ids and vector
point ids derive from the ``source_id``, so this identity never changes for
sources that are already indexed.
"""

import json
import re
from collections.abc import Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from docgrep.domain.errors import ValidationError

ROOT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# The root of unprefixed sources when no legacy root is configured. It is not
# a valid configured name, so it never clashes with one.
UNCONFIGURED_LEGACY_ROOT = "_legacy"
# The name of a single root taken from a folder whose name is not valid.
DEFAULT_ROOT_NAME = "default"
# Why a stored path cannot be read: its root is not configured here.
ROOT_NOT_AVAILABLE = "root_not_available"
NESTED_ROOT_HINT = (
    "a sub-folder of an existing root is a path prefix filter, not a new root"
)

# Run configuration keys: the roots a run visited, and the prefixed root
# names whose sources the index may hold once the run is done.
RUN_ROOTS_KEY = "corpus_roots"
RUN_PREFIXES_KEY = "root_prefixes"


def check_root_name(name: str) -> str:
    """The name, or ``ValidationError`` when it is not a valid root name."""
    if not isinstance(name, str) or ROOT_NAME_PATTERN.fullmatch(name) is None:
        raise ValidationError(
            f"invalid root name {name!r}: use letters, digits, '.', '_' and '-', "
            "starting with a letter or a digit",
            details={"field": "name", "name": str(name)},
        )
    return name


def folder_root_name(path: Path) -> str:
    """The root name a folder gets when none is given."""
    name = path.name
    return name if ROOT_NAME_PATTERN.fullmatch(name) else DEFAULT_ROOT_NAME


@dataclass(frozen=True, slots=True)
class CorpusRoot:
    """A named folder of the corpus; ``legacy`` keeps its sources unprefixed."""

    name: str
    path: Path
    legacy: bool = False

    def __post_init__(self) -> None:
        check_root_name(self.name)

    @property
    def prefix(self) -> str:
        return "" if self.legacy else f"{self.name}/"

    def stored_path(self, relative: str) -> str:
        """The stored path (and ``source_id``) of a root-relative POSIX path."""
        return self.prefix + relative

    def relative_path(self, stored: str) -> str:
        """The root-relative path of a stored path of this root."""
        return stored.removeprefix(self.prefix)


def _contains(outer: Path, inner: Path) -> bool:
    return outer == inner or outer in inner.parents


def _check_roots(roots: tuple[CorpusRoot, ...]) -> None:
    seen: dict[str, CorpusRoot] = {}
    legacy: CorpusRoot | None = None
    for root in roots:
        other = seen.get(root.name.casefold())
        if other is not None:
            raise ValidationError(
                f"root name {root.name!r} is already used by root {other.name!r} "
                "(names are compared case-insensitively)",
                details={"field": "name", "name": root.name},
            )
        if root.legacy:
            if legacy is not None:
                raise ValidationError(
                    f"only one root can be the legacy root; {legacy.name!r} already is",
                    details={"field": "legacy", "name": root.name},
                )
            legacy = root
        for known in seen.values():
            if _contains(known.path, root.path):
                inside = (root, known)
            elif _contains(root.path, known.path):
                inside = (known, root)
            else:
                continue
            raise ValidationError(
                f"root {inside[0].name!r} ({inside[0].path}) is inside root "
                f"{inside[1].name!r} ({inside[1].path}); {NESTED_ROOT_HINT}",
                details={
                    "field": "path",
                    "name": root.name,
                    "inner": inside[0].name,
                    "outer": inside[1].name,
                },
            )
        seen[root.name.casefold()] = root


def root_name_of(source_id: str, prefixed: Collection[str], legacy: str) -> str:
    """The root of a stored source: the prefixed root its first segment
    names, or else the legacy root."""
    head, separator, _ = source_id.partition("/")
    return head if separator and head in prefixed else legacy


@dataclass(frozen=True, slots=True)
class CorpusRoots:
    """The roots of a run or a configuration, in configured order.

    Construction validates them: valid names, unique case-insensitively, at
    most one legacy root and no root inside another. ``recorded`` holds the
    prefixed root names the index knows from earlier runs, configured or not,
    so that their sources are never taken for legacy ones.
    """

    roots: tuple[CorpusRoot, ...] = ()
    recorded: frozenset[str] = field(default_factory=frozenset)
    # The name of the configured legacy root when it is not one of these
    # roots, as in a run of other roots: its sources are named after it.
    other_legacy: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", tuple(self.roots))
        object.__setattr__(self, "recorded", frozenset(self.recorded))
        _check_roots(self.roots)

    @classmethod
    def single(cls, path: Path, name: str | None = None) -> "CorpusRoots":
        """One legacy root, named after its folder unless a name is given."""
        return cls((CorpusRoot(name or folder_root_name(path), path, legacy=True),))

    def __iter__(self) -> Iterator[CorpusRoot]:
        return iter(self.roots)

    def __len__(self) -> int:
        return len(self.roots)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(root.name for root in self.roots)

    @property
    def legacy(self) -> CorpusRoot | None:
        return next((root for root in self.roots if root.legacy), None)

    @property
    def legacy_name(self) -> str:
        """The root unprefixed sources belong to, configured or not."""
        legacy = self.legacy
        if legacy is not None:
            return legacy.name
        return self.other_legacy or UNCONFIGURED_LEGACY_ROOT

    @property
    def prefixed_names(self) -> frozenset[str]:
        """Every root name that is a prefix of stored paths."""
        configured = {root.name for root in self.roots if not root.legacy}
        return frozenset((configured | self.recorded) - {self.legacy_name})

    @property
    def named(self) -> bool:
        """Whether results name the root of each source: not for one legacy
        root that the index knows no other root next to, whose results stay
        exactly as they were before roots had names."""
        return not (
            len(self.roots) == 1 and self.roots[0].legacy and not self.prefixed_names
        )

    def get(self, name: str) -> CorpusRoot | None:
        return next((root for root in self.roots if root.name == name), None)

    def with_recorded(self, names: Iterable[str]) -> "CorpusRoots":
        return CorpusRoots(
            self.roots, self.recorded | frozenset(names), self.other_legacy
        )

    def only(self, names: Collection[str]) -> "CorpusRoots":
        """These roots alone, still knowing every other prefix and the name
        of the legacy root."""
        return CorpusRoots(
            tuple(root for root in self.roots if root.name in names),
            self.recorded | self.prefixed_names,
            self.legacy_name if self.legacy is not None else self.other_legacy,
        )

    def root_of(self, source_id: str) -> str:
        """The name of the root a stored source belongs to (configured or not)."""
        return root_name_of(source_id, self.prefixed_names, self.legacy_name)

    def owner(self, stored: str) -> CorpusRoot | None:
        """The configured root of a stored path; None: root not available."""
        return self.get(self.root_of(stored))

    def locate(self, stored: str) -> Path | None:
        """The file on disk of a stored path; None when its root is not
        configured. Nothing is read."""
        root = self.owner(stored)
        if root is None:
            return None
        return root.path / root.relative_path(stored)

    def stored_path(self, file: Path) -> str | None:
        """The stored path of a file given with an absolute path, or None
        when no root holds it. Both sides are compared as given."""
        for root in self.roots:
            if _contains(root.path, file) and file != root.path:
                relative = PurePosixPath(*file.relative_to(root.path).parts)
                return root.stored_path(relative.as_posix())
        return None

    def check_legacy_entries(self, entries: Collection[str]) -> None:
        """Refuse a prefixed root named like a top-level entry of the legacy
        root, whose sources would share its stored paths."""
        legacy = self.legacy
        if legacy is None:
            return
        for root in self.roots:
            if not root.legacy and root.name in entries:
                raise ValidationError(
                    f"root name {root.name!r} equals a top-level entry of the "
                    f"legacy root {legacy.name!r}; its source ids would collide "
                    "with that entry's. Choose another name",
                    details={"field": "name", "name": root.name},
                )

    def snapshot(self) -> str:
        """The roots as recorded on a run, for a resume to compare."""
        return json.dumps(
            [
                {"name": root.name, "path": str(root.path), "legacy": root.legacy}
                for root in sorted(self.roots, key=lambda root: root.name)
            ],
            sort_keys=True,
        )


def known_root_names(roots: CorpusRoots, source_ids: Iterable[str]) -> tuple[str, ...]:
    """The configured roots and the roots of the given stored sources, by name."""
    return tuple(sorted({*roots.names, *(roots.root_of(s) for s in source_ids)}))


def recorded_prefixes(config: Mapping[str, str | int | float]) -> frozenset[str]:
    """The prefixed root names recorded on a run; none for older runs."""
    value = config.get(RUN_PREFIXES_KEY)
    if not isinstance(value, str):
        return frozenset()
    try:
        names = json.loads(value)
    except ValueError:
        return frozenset()
    if not isinstance(names, list):
        return frozenset()
    return frozenset(name for name in names if isinstance(name, str))


def prefixes_value(names: Iterable[str]) -> str:
    """The run configuration value of the prefixed root names."""
    return json.dumps(sorted(set(names)))
