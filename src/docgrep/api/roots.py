"""Which corpus roots a process uses (ADR-0023).

The saved list in ``<data_dir>/roots.json`` is the configuration. Without
that file the deprecated ``DOCGREP_CORPUS_ROOT`` is one legacy root, with a
warning. ``--root`` parameters replace the list for one run and are never
saved implicitly.
"""

import logging
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from docgrep.adapters.corpus import require_roots, root_readable
from docgrep.adapters.manifest import SQLiteManifest
from docgrep.adapters.roots import (
    checked_folder,
    legacy_entries,
    read_roots_file,
    roots_file,
    with_root,
    without_root,
    write_roots_file,
)
from docgrep.domain.errors import ValidationError
from docgrep.domain.roots import (
    ROOT_NAME_PATTERN,
    CorpusRoot,
    CorpusRoots,
    folder_root_name,
)
from docgrep.domain.run_state import latest_root_prefixes
from docgrep.settings import Settings

_LOG = logging.getLogger(__name__)

RootsOrigin = Literal["file", "bootstrap", "none", "parameters"]
# The configuration warnings this process has logged.
_LOGGED: set[str] = set()
BOOTSTRAP_VARIABLE = "DOCGREP_CORPUS_ROOT"


@dataclass(frozen=True, slots=True)
class RootSelection:
    """The roots of a process, where they come from and what to tell the user.

    ``unsaved`` lists the parameter roots that the roots file does not hold.
    """

    roots: CorpusRoots
    origin: RootsOrigin
    warnings: tuple[str, ...] = ()
    unsaved: tuple[CorpusRoot, ...] = ()


def add_command(root: CorpusRoot) -> str:
    """The command that saves a root."""
    command = (
        f"docgrep roots add {shlex.quote(root.name)} {shlex.quote(str(root.path))}"
    )
    return command + (" --legacy" if root.legacy else "")


def bootstrap_root(settings: Settings) -> CorpusRoot:
    path = settings.corpus_root.resolve()
    return CorpusRoot(folder_root_name(path), path, legacy=True)


def configured_roots(settings: Settings) -> RootSelection:
    """The saved roots, or the deprecated variable without a roots file."""
    variable_set = "corpus_root" in settings.model_fields_set
    saved = read_roots_file(settings.data_dir)
    if saved is not None:
        warnings: tuple[str, ...] = ()
        if variable_set:
            warnings = (
                f"warning: {BOOTSTRAP_VARIABLE} is ignored because "
                f"{roots_file(settings.data_dir)} exists",
            )
        return RootSelection(saved, "file", warnings)
    if variable_set:
        root = bootstrap_root(settings)
        return RootSelection(
            CorpusRoots((root,)),
            "bootstrap",
            (
                f"warning: {BOOTSTRAP_VARIABLE} is deprecated; save it as a "
                f"root with: {add_command(root)}",
            ),
        )
    return RootSelection(CorpusRoots(), "none")


def parse_root_parameter(value: str) -> tuple[str | None, Path]:
    """``NAME=PATH``, or a path alone, whose root is named after its folder."""
    name, separator, path = value.partition("=")
    if separator and ROOT_NAME_PATTERN.fullmatch(name):
        given = path
    else:
        name, given = "", value
    if not given.strip():
        raise ValidationError(
            f"--root {value!r} names no folder; use --root NAME=PATH",
            details={"field": "root", "value": value},
        )
    return (name or None), Path(given).expanduser().resolve()


def parameter_roots(
    settings: Settings, values: Sequence[str], *, saved: CorpusRoots | None = None
) -> RootSelection:
    """The roots given as ``--root`` parameters, for this run only.

    A name the roots file holds must point to the same folder, and takes the
    saved legacy flag; any other parameter root is prefixed. The roots are
    validated together with the saved ones, so a parameter root can neither
    nest with nor duplicate a saved root. The saved prefixes stay known, so
    the sources of saved roots that are not passed are never taken for
    legacy ones.
    """
    stored = configured_roots(settings).roots if saved is None else saved
    given: list[CorpusRoot] = []
    for value in values:
        name, path = parse_root_parameter(value)
        name = name or folder_root_name(path)
        known = stored.get(name)
        if known is not None and known.path != path:
            raise ValidationError(
                f"root {name!r} is saved with the path {known.path}, but --root "
                f"gives {path}; pass the saved path or another name",
                details={"field": "root", "name": name, "path": str(path)},
            )
        same_folder = next((root for root in stored if root.path == path), None)
        if same_folder is not None and same_folder.name != name:
            raise ValidationError(
                f"the folder {path} is saved as root {same_folder.name!r}; pass "
                f"--root {same_folder.name}={path}",
                details={"field": "root", "name": name, "path": str(path)},
            )
        given.append(CorpusRoot(name, path, legacy=known is not None and known.legacy))
    run = CorpusRoots(tuple(given))
    names = set(run.names)
    # One validation of everything: nesting, names, the legacy root and its
    # top-level entries.
    combined = CorpusRoots(
        (*given, *(root for root in stored if root.name not in names))
    )
    combined.check_legacy_entries(legacy_entries(combined))
    unsaved = tuple(root for root in run if stored.get(root.name) is None)
    return RootSelection(
        combined.only(names).with_recorded(stored.prefixed_names),
        "parameters",
        unsaved=unsaved,
    )


def select_roots(settings: Settings, values: Sequence[str] = ()) -> RootSelection:
    """The roots of a command: the ``--root`` parameters when given, or else
    the configured roots."""
    configured = configured_roots(settings)
    if not values:
        return configured
    chosen = parameter_roots(settings, values, saved=configured.roots)
    return RootSelection(
        chosen.roots, chosen.origin, configured.warnings, chosen.unsaved
    )


def corpus_roots(settings: Settings) -> CorpusRoots:
    """The roots of this process: chosen by the command line, or else read
    from the configuration, logging what the user should know."""
    chosen = settings.chosen_corpus_roots
    if chosen is not None:
        return chosen
    selection = configured_roots(settings)
    for warning in selection.warnings:
        # Once per process: a server asks on every request.
        if warning not in _LOGGED:
            _LOGGED.add(warning)
            _LOG.warning("corpus_roots_configuration", extra={"message_text": warning})
    return selection.roots


def require_configured(roots: CorpusRoots) -> None:
    """Fail with "no corpus roots configured" when there is none."""
    require_roots(roots)


def _indexed_counts(settings: Settings, roots: CorpusRoots) -> dict[str, int]:
    """Indexed sources per root name; nothing when there is no manifest."""
    if not (settings.data_dir / "manifest.sqlite3").is_file():
        return {}
    manifest = SQLiteManifest(settings.data_dir, read_only=True)
    known = roots.with_recorded(latest_root_prefixes(manifest))
    counts: dict[str, int] = {}
    for record in manifest.committed_sources():
        if record.status == "indexed":
            name = known.root_of(record.source.source_id)
            counts[name] = counts.get(name, 0) + 1
    return counts


def list_roots(settings: Settings) -> dict[str, Any]:
    """The configured roots for ``docgrep roots list``: where they come from,
    and each root's path, legacy flag, readability and indexed sources."""
    selection = configured_roots(settings)
    counts = _indexed_counts(settings, selection.roots)
    return {
        "origin": selection.origin,
        "roots": [
            {
                "name": root.name,
                "path": str(root.path),
                "legacy": root.legacy,
                "readable": root_readable(root.path),
                "indexed_sources": counts.get(root.name, 0),
            }
            for root in selection.roots
        ],
        "warnings": list(selection.warnings),
    }


def add_root(settings: Settings, name: str, path: Path, *, legacy: bool) -> Path:
    """Save a root and return the roots file; the index is not touched."""
    saved = read_roots_file(settings.data_dir) or CorpusRoots()
    root = CorpusRoot(name, checked_folder(path), legacy=legacy)
    return write_roots_file(settings.data_dir, with_root(saved, root))


def remove_root(settings: Settings, name: str) -> Path:
    """Remove a root from the roots file; its indexed sources stay."""
    saved = read_roots_file(settings.data_dir)
    if saved is None:
        raise ValidationError(
            f"no roots file exists ({roots_file(settings.data_dir)}); nothing to "
            "remove",
            details={"field": "name", "name": name},
        )
    return write_roots_file(settings.data_dir, without_root(saved, name))


def save_parameter_root(settings: Settings, root: CorpusRoot) -> tuple[Path, bool]:
    """Save a root given as a parameter, never as the legacy root.

    Without a roots file the saved list starts from the configured roots, so
    the root of the deprecated variable is saved with it; the second value
    says whether that happened.
    """
    configured = configured_roots(settings)
    base = configured.roots if configured.origin != "none" else CorpusRoots()
    added = with_root(base, CorpusRoot(root.name, root.path))
    path = write_roots_file(settings.data_dir, added)
    return path, configured.origin == "bootstrap"
