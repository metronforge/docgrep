"""File reading shared by text-based parser adapters."""

import codecs
from pathlib import Path

import chardet

from docgrep.adapters.corpus import as_roots, source_file
from docgrep.adapters.tool_output import scrub_tool_output
from docgrep.domain.errors import ParseError, ValidationError
from docgrep.domain.models import Source
from docgrep.domain.roots import CorpusRoots


def source_location(roots: CorpusRoots, source: Source) -> tuple[Path, Path]:
    """The file of a source and the folder of its root; ``ParseError`` when
    the root is not available to this process."""
    try:
        return source_file(roots, source.path)
    except ValidationError as error:
        raise ParseError(
            "source cannot be read: its root is not available",
            details={**error.details, "path": source.path},
        ) from error


def read_text(corpus_root: CorpusRoots | Path, source: Source) -> str:
    """Read UTF-8 first, then a confidently detected legacy encoding."""
    path, corpus_root = source_location(as_roots(corpus_root), source)
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ParseError(
            "source cannot be read",
            details={
                "path": source.path,
                "reason": scrub_tool_output(str(error), (path, corpus_root)),
            },
        ) from error
    return decode_text(data, source.path)


def decode_text(data: bytes, path: str, declared: str | None = None) -> str:
    """Decode with a declared encoding, else UTF-8, else a confident detection.

    A declared encoding that is unknown or does not fit the bytes is ignored,
    and the UTF-8 and detection steps apply as if nothing was declared.
    """
    if declared is not None:
        try:
            name = codecs.lookup(declared).name
        except LookupError:
            name = None
        if name is not None and name != "utf-8":
            try:
                return data.decode(name)
            except UnicodeDecodeError:
                pass
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        detected = chardet.detect(data)
        encoding = detected.get("encoding")
        confidence = detected.get("confidence")
        if not isinstance(encoding, str) or not isinstance(confidence, float):
            raise ParseError(
                "source encoding cannot be detected", details={"path": path}
            )
        if confidence < 0.5:
            raise ParseError(
                "source encoding cannot be detected", details={"path": path}
            )
        try:
            return data.decode(encoding)
        except (LookupError, UnicodeDecodeError) as error:
            raise ParseError(
                "source cannot be decoded",
                details={"path": path, "encoding": encoding},
            ) from error
