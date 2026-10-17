"""The roots file: the saved list of corpus roots in the data directory."""

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from docgrep.adapters.corpus import root_readable
from docgrep.domain.errors import ValidationError
from docgrep.domain.roots import CorpusRoot, CorpusRoots, check_root_name

ROOTS_FILE = "roots.json"
ROOTS_FILE_VERSION = 1


def roots_file(data_dir: Path) -> Path:
    return data_dir / ROOTS_FILE


def legacy_entries(roots: CorpusRoots) -> frozenset[str]:
    """The top-level entry names of the legacy root; none when it cannot be
    listed, so that an unreadable root never blocks reading the file."""
    legacy = roots.legacy
    if legacy is None or not root_readable(legacy.path):
        return frozenset()
    try:
        return frozenset(entry.name for entry in os.scandir(legacy.path))
    except OSError:
        return frozenset()


def _invalid(path: Path, reason: str) -> ValidationError:
    return ValidationError(
        f"the roots file {path} is not valid: {reason}",
        details={"path": str(path), "reason": reason},
    )


def _root(path: Path, item: object) -> CorpusRoot:
    if not isinstance(item, dict):
        raise _invalid(path, "every root must be an object")
    name, folder, legacy = item.get("name"), item.get("path"), item.get("legacy")
    if not isinstance(name, str) or not isinstance(folder, str):
        raise _invalid(path, "every root needs a name and a path")
    if legacy is not None and not isinstance(legacy, bool):
        raise _invalid(path, "legacy must be true or false")
    if not Path(folder).is_absolute():
        raise _invalid(path, f"the path of root {name!r} must be absolute")
    return CorpusRoot(check_root_name(name), Path(folder), legacy=bool(legacy))


def read_roots_file(data_dir: Path) -> CorpusRoots | None:
    """The saved roots, validated again; None when there is no roots file.

    Paths are not required to exist: reading the index needs no root on disk.
    """
    path = roots_file(data_dir)
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError):
        # No roots file, or no data directory to hold one.
        return None
    except OSError as error:
        raise _invalid(path, str(error)) from error
    try:
        document: Any = json.loads(text)
    except ValueError as error:
        raise _invalid(path, f"not JSON ({error})") from error
    if not isinstance(document, dict) or document.get("version") != (
        ROOTS_FILE_VERSION
    ):
        raise _invalid(path, f"expected version {ROOTS_FILE_VERSION}")
    items = document.get("roots")
    if not isinstance(items, list):
        raise _invalid(path, "roots must be a list")
    roots = CorpusRoots(tuple(_root(path, item) for item in items))
    roots.check_legacy_entries(legacy_entries(roots))
    return roots


def _document(roots: CorpusRoots) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for root in roots:
        entry: dict[str, Any] = {"name": root.name, "path": str(root.path)}
        if root.legacy:
            entry["legacy"] = True
        entries.append(entry)
    return {"version": ROOTS_FILE_VERSION, "roots": entries}


def write_roots_file(
    data_dir: Path,
    roots: CorpusRoots,
    *,
    replace: Callable[[str, Path], None] = os.replace,
) -> Path:
    """Save the roots atomically: a temporary file in the data directory,
    then a rename over the old file, which stays intact on any failure."""
    path = roots_file(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_document(roots), indent=2, ensure_ascii=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        dir=data_dir, prefix=f".{ROOTS_FILE}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


def checked_folder(path: Path) -> Path:
    """The absolute, resolved folder of a root being added; it must exist
    and be a directory."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise ValidationError(
            f"root path {resolved} does not exist",
            details={"field": "path", "path": str(resolved)},
        )
    if not resolved.is_dir():
        raise ValidationError(
            f"root path {resolved} is not a directory",
            details={"field": "path", "path": str(resolved)},
        )
    return resolved


def with_root(roots: CorpusRoots, root: CorpusRoot) -> CorpusRoots:
    """The roots with one more, validated as a whole."""
    added = CorpusRoots((*roots.roots, root))
    added.check_legacy_entries(legacy_entries(added))
    return added


def without_root(roots: CorpusRoots, name: str) -> CorpusRoots:
    """The roots without the one of this name; an unknown name is an error."""
    if roots.get(name) is None:
        raise ValidationError(
            f"no saved root is named {name!r}; saved roots: "
            + (", ".join(roots.names) or "none"),
            details={"field": "name", "name": name, "known": list(roots.names)},
        )
    return CorpusRoots(tuple(root for root in roots if root.name != name))
