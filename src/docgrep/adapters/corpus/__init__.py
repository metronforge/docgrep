"""Local corpus enumeration and content hashing."""

import os
from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath

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


def _matches(path: str, patterns: Sequence[str]) -> bool:
    relative = PurePosixPath(path)
    return any(
        relative.match(pattern)
        or (pattern.startswith("**/") and relative.match(pattern[3:]))
        for pattern in patterns
    )


def _files(root: Path) -> Iterator[Path]:
    def raise_walk_error(error: OSError) -> None:
        raise error

    for directory, subdirs, filenames in os.walk(
        root, followlinks=False, onerror=raise_walk_error
    ):
        current = Path(directory)
        subdirs[:] = sorted(
            name for name in subdirs if not (current / name).is_symlink()
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
) -> tuple[Source, ...]:
    """Return files in relative-path order, including unsupported types."""
    try:
        root_stat = corpus_root.stat()
        if not corpus_root.is_dir() or not root_stat.st_mode & 0o444:
            raise OSError("corpus root is not a readable directory")
        sources = []
        for path in _files(corpus_root):
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
