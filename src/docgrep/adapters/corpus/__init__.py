"""Local corpus enumeration and content hashing."""

import os
from collections.abc import Collection, Iterator, Sequence
from pathlib import Path

from blake3 import blake3

from docgrep.domain.errors import ValidationError
from docgrep.domain.globs import matching_pattern
from docgrep.domain.models import FileType, ScannedFile, Source
from docgrep.domain.roots import ROOT_NOT_AVAILABLE, CorpusRoot, CorpusRoots

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


def _matching_pattern(path: str, patterns: Sequence[str]) -> str | None:
    """The first glob pattern a POSIX-relative path matches (see
    ``docgrep.domain.globs.matching_pattern``)."""
    return matching_pattern(path, patterns)


def _matches(path: str, patterns: Sequence[str]) -> bool:
    """Return whether a POSIX-relative path matches any glob pattern."""
    return _matching_pattern(path, patterns) is not None


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


def file_content_hash(path: Path) -> str:
    """The BLAKE3 content hash of a file, as recorded for every source."""
    digest = blake3()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_content_hash = file_content_hash


def as_roots(roots: CorpusRoots | Path) -> CorpusRoots:
    """The roots of an adapter; a plain folder is one legacy root."""
    return roots if isinstance(roots, CorpusRoots) else CorpusRoots.single(roots)


def _skipped_data_dir(corpus_root: Path, data_dir: Path | None) -> Path | None:
    """Resolve the data directory to skip, or reject a corpus inside it."""
    if data_dir is None:
        return None
    resolved_data = data_dir.resolve()
    resolved_root = corpus_root.resolve()
    if resolved_root == resolved_data or resolved_data in resolved_root.parents:
        raise ValidationError(
            "corpus root must not be inside the data directory",
            details={"corpus_root": str(corpus_root), "data_dir": str(data_dir)},
        )
    return resolved_data if resolved_root in resolved_data.parents else None


def _require_readable_root(corpus_root: Path) -> None:
    corpus_root.stat()
    if not corpus_root.is_dir() or not os.access(corpus_root, os.R_OK | os.X_OK):
        raise OSError("corpus root is not a readable directory")


def root_readable(path: Path) -> bool:
    """Whether a root folder exists and can be listed; nothing is walked."""
    try:
        _require_readable_root(path)
    except OSError:
        return False
    return True


def _inaccessible(
    corpus_root: Path, error: OSError, root: CorpusRoot | None = None
) -> ValidationError:
    details: dict[str, object] = {"path": str(corpus_root), "reason": str(error)}
    if root is not None:
        details["root"] = root.name
    return ValidationError("corpus is not accessible", details=details)


def require_roots(roots: CorpusRoots) -> None:
    """Fail when no corpus root is configured."""
    if not roots.roots:
        raise ValidationError(
            "no corpus roots configured; add one with `docgrep roots add NAME PATH`",
            details={"reason": "no_roots"},
        )


def require_available(roots: CorpusRoots | Path) -> None:
    """Fail before any work when a root is missing, not a directory or not
    readable, naming the root; a root is never taken for an empty one."""
    checked = as_roots(roots)
    require_roots(checked)
    for root in checked:
        try:
            _require_readable_root(root.path)
        except OSError as error:
            raise ValidationError(
                f"corpus root {root.name!r} is not accessible: {root.path}",
                details={
                    "root": root.name,
                    "path": str(root.path),
                    "reason": str(error),
                },
            ) from error


def _walk(
    roots: CorpusRoots, data_dir: Path | None
) -> Iterator[tuple[CorpusRoot, Path, str]]:
    """Every file of every root with its root and root-relative path.

    All roots are checked first, so a root that cannot be read fails the
    walk before any other root is listed.
    """
    skipped = [(root, _skipped_data_dir(root.path, data_dir)) for root in roots]
    for root, _ in skipped:
        try:
            _require_readable_root(root.path)
        except OSError as error:
            raise _inaccessible(root.path, error, root) from error
    for root, skipped_dir in skipped:
        try:
            for path in _files(root.path, skipped_dir):
                yield root, path, path.relative_to(root.path).as_posix()
        except OSError as error:
            raise _inaccessible(root.path, error, root) from error


def enumerate_corpus(
    corpus_root: CorpusRoots | Path,
    include_globs: Sequence[str] = ("**/*",),
    exclude_globs: Sequence[str] = (),
    data_dir: Path | None = None,
) -> tuple[Source, ...]:
    """Return files in stored-path order, including unsupported types.

    ``.git`` directories are never entered. When ``data_dir`` lies inside a
    root it is skipped as well, so index files are not listed as sources; a
    root that is ``data_dir`` or lies inside it is a configuration error.
    The globs match the path relative to each root.
    """
    roots = as_roots(corpus_root)
    sources = []
    for root, path, relative in _walk(roots, data_dir):
        if not _matches(relative, include_globs) or _matches(relative, exclude_globs):
            continue
        try:
            stat = path.stat()
            content_hash = _content_hash(path)
        except OSError as error:
            raise _inaccessible(root.path, error, root) from error
        stored = root.stored_path(relative)
        sources.append(
            Source(
                source_id=stored,
                path=stored,
                file_type=_FILE_TYPES.get(path.suffix.lower(), FileType.UNSUPPORTED),
                content_hash=content_hash,
                mtime=stat.st_mtime,
                size=stat.st_size,
            )
        )
    return tuple(sorted(sources, key=lambda source: source.path))


def describe_source(corpus_root: CorpusRoots | Path, path: str) -> Source:
    """The source record of one corpus file by stored path, exactly as
    ``enumerate_corpus`` gives it, content hash included."""
    roots = as_roots(corpus_root)
    root = roots.owner(path)
    file = roots.locate(path)
    if root is None or file is None:
        raise ValidationError(
            "the root of the source is not available",
            details={"path": path, "root": roots.root_of(path)},
        )
    try:
        stat = file.stat()
        content_hash = _content_hash(file)
    except OSError as error:
        raise _inaccessible(root.path, error, root) from error
    return Source(
        source_id=path,
        path=path,
        file_type=_FILE_TYPES.get(file.suffix.lower(), FileType.UNSUPPORTED),
        content_hash=content_hash,
        mtime=stat.st_mtime,
        size=stat.st_size,
    )


NOT_MATCHED_BY_INCLUDES = "not_matched_by_includes"


def scan_corpus(
    corpus_root: CorpusRoots | Path,
    include_globs: Sequence[str] = ("**/*",),
    exclude_globs: Sequence[str] = (),
    data_dir: Path | None = None,
) -> tuple[ScannedFile, ...]:
    """List every file of every root from directory entries alone.

    Nothing is read or hashed. Each file reports its stored path, size,
    extension, file type bucket (None for an unrecognised extension) and,
    when the include and exclude patterns would keep a build from enumerating
    it, the pattern responsible: ``not_matched_by_includes`` or the matching
    exclude pattern. The walk, the skipped directories and the matcher are
    those of ``enumerate_corpus``.
    """
    roots = as_roots(corpus_root)
    scanned = []
    for root, path, relative in _walk(roots, data_dir):
        excluded_by = None
        if not _matches(relative, include_globs):
            excluded_by = NOT_MATCHED_BY_INCLUDES
        else:
            excluded_by = _matching_pattern(relative, exclude_globs)
        extension = path.suffix.lower()
        try:
            stat = path.stat()
        except OSError as error:
            raise _inaccessible(root.path, error, root) from error
        scanned.append(
            ScannedFile(
                path=root.stored_path(relative),
                size=stat.st_size,
                extension=extension,
                file_type=_FILE_TYPES.get(extension),
                excluded_by=excluded_by,
                mtime=stat.st_mtime,
            )
        )
    return tuple(sorted(scanned, key=lambda item: item.path))


def scan_paths(
    corpus_root: CorpusRoots | Path,
    paths: Collection[str],
    include_globs: Sequence[str] = ("**/*",),
    exclude_globs: Sequence[str] = (),
    data_dir: Path | None = None,
) -> tuple[ScannedFile, ...]:
    """The entries ``scan_corpus`` would give for these stored paths, without
    walking any root: one stat per path.

    A path whose file is gone, is not a regular file, is a symlink, lies in a
    ``.git`` or data directory, or whose root is not configured is left out.
    A configured root that cannot be read raises, as a walk would, so that
    its files are never taken for deleted ones.
    """
    roots = as_roots(corpus_root)
    readable: set[str] = set()
    scanned = []
    for stored in sorted(set(paths)):
        root = roots.owner(stored)
        if root is None:
            continue
        if root.name not in readable:
            try:
                _require_readable_root(root.path)
            except OSError as error:
                raise _inaccessible(root.path, error, root) from error
            readable.add(root.name)
        relative = root.relative_path(stored)
        path = root.path / relative
        skipped = _skipped_data_dir(root.path, data_dir)
        if ".git" in relative.split("/") or (
            skipped is not None and skipped in path.resolve().parents
        ):
            continue
        try:
            if path.is_symlink() or not path.is_file():
                continue
            stat = path.stat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise _inaccessible(root.path, error, root) from error
        if not _matches(relative, include_globs):
            excluded_by: str | None = NOT_MATCHED_BY_INCLUDES
        else:
            excluded_by = _matching_pattern(relative, exclude_globs)
        extension = path.suffix.lower()
        scanned.append(
            ScannedFile(
                path=stored,
                size=stat.st_size,
                extension=extension,
                file_type=_FILE_TYPES.get(extension),
                excluded_by=excluded_by,
                mtime=stat.st_mtime,
            )
        )
    return tuple(scanned)


def source_file(roots: CorpusRoots, path: str) -> tuple[Path, Path]:
    """The file on disk of a stored path and the folder of its root.

    Raises ``ValidationError`` with reason ``root_not_available`` when the
    root of the path is not configured for this process.
    """
    root = roots.owner(path)
    if root is None:
        raise ValidationError(
            "the root of the source is not available",
            details={
                "path": path,
                "root": roots.root_of(path),
                "reason": ROOT_NOT_AVAILABLE,
            },
        )
    return root.path / root.relative_path(path), root.path
