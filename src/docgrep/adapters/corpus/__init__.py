"""Local corpus enumeration and content hashing."""

import os
from collections.abc import Iterator, Sequence
from fnmatch import fnmatchcase
from pathlib import Path

from blake3 import blake3

from docgrep.domain.errors import ValidationError
from docgrep.domain.models import FileType, Source

_FILE_TYPES = {
    ".txt": FileType.TXT,
    ".md": FileType.MARKDOWN,
    ".markdown": FileType.MARKDOWN,
    ".pdf": FileType.PDF,
    ".epub": FileType.EPUB,
    ".docx": FileType.DOCX,
    ".djvu": FileType.DJVU,
    ".djv": FileType.DJVU,
}


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


def _matches(path: str, patterns: Sequence[str]) -> bool:
    """Return whether a POSIX-relative path matches any glob pattern.

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
    for pattern in patterns:
        anchored = pattern.startswith("/")
        if anchored:
            pattern = pattern[1:]
        if pattern.endswith("/"):
            pattern += "**"
        if not anchored and "/" not in pattern:
            if fnmatchcase(parts[-1], pattern):
                return True
        elif _match_segments(tuple(pattern.split("/")), parts):
            return True
    return False


def _files(root: Path, skipped_dir: Path | None = None) -> Iterator[Path]:
    def raise_walk_error(error: OSError) -> None:
        raise error

    def is_skipped(directory: Path) -> bool:
        return directory.name == ".git" or (
            skipped_dir is not None and directory.resolve() == skipped_dir
        )

    for directory, subdirs, filenames in os.walk(
        root, followlinks=False, onerror=raise_walk_error
    ):
        current = Path(directory)
        subdirs[:] = sorted(
            name
            for name in subdirs
            if not (current / name).is_symlink() and not is_skipped(current / name)
        )
        for name in sorted(filenames):
            path = current / name
            if not path.is_symlink() and path.is_file():
                yield path


def _content_hash(path: Path) -> str:
    digest = blake3()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def enumerate_corpus(
    corpus_root: Path,
    include_globs: Sequence[str] = ("**/*",),
    exclude_globs: Sequence[str] = (),
    data_dir: Path | None = None,
) -> tuple[Source, ...]:
    """Return files in relative-path order, including unsupported types.

    ``.git`` directories are never entered. When ``data_dir`` lies inside the
    corpus it is skipped as well, so index files are not listed as sources; a
    corpus root that is ``data_dir`` or lies inside it is a configuration error.
    """
    skipped_dir: Path | None = None
    if data_dir is not None:
        resolved_data = data_dir.resolve()
        resolved_root = corpus_root.resolve()
        if resolved_root == resolved_data or resolved_data in resolved_root.parents:
            raise ValidationError(
                "corpus root must not be inside the data directory",
                details={"corpus_root": str(corpus_root), "data_dir": str(data_dir)},
            )
        if resolved_root in resolved_data.parents:
            skipped_dir = resolved_data
    try:
        corpus_root.stat()
        if not corpus_root.is_dir() or not os.access(corpus_root, os.R_OK | os.X_OK):
            raise OSError("corpus root is not a readable directory")
        sources = []
        for path in _files(corpus_root, skipped_dir):
            relative = path.relative_to(corpus_root).as_posix()
            if not _matches(relative, include_globs) or _matches(
                relative, exclude_globs
            ):
                continue
            stat = path.stat()
            sources.append(
                Source(
                    source_id=relative,
                    path=relative,
                    file_type=_FILE_TYPES.get(
                        path.suffix.lower(), FileType.UNSUPPORTED
                    ),
                    content_hash=_content_hash(path),
                    mtime=stat.st_mtime,
                    size=stat.st_size,
                )
            )
    except OSError as error:
        raise ValidationError(
            "corpus is not accessible",
            details={"path": str(corpus_root), "reason": str(error)},
        ) from error
    return tuple(sorted(sources, key=lambda source: source.path))
