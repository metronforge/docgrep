"""File reading shared by text-based parser adapters."""

import codecs
from pathlib import Path

import chardet

from docgrep.adapters.tool_output import scrub_tool_output
from docgrep.domain.errors import ParseError
from docgrep.domain.models import Source


def read_text(corpus_root: Path, source: Source) -> str:
    """Read UTF-8 first, then a confidently detected legacy encoding."""
    path = corpus_root / source.path
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
