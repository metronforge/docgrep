"""File reading shared by text-based parser adapters."""

from pathlib import Path

import chardet

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
            details={"path": source.path, "reason": str(error)},
        ) from error
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        detected = chardet.detect(data)
        encoding = detected.get("encoding")
        confidence = detected.get("confidence")
        if not isinstance(encoding, str) or not isinstance(confidence, float):
            raise ParseError(
                "source encoding cannot be detected", details={"path": source.path}
            )
        if confidence < 0.5:
            raise ParseError(
                "source encoding cannot be detected", details={"path": source.path}
            )
        try:
            return data.decode(encoding)
        except (LookupError, UnicodeDecodeError) as error:
            raise ParseError(
                "source cannot be decoded",
                details={"path": source.path, "encoding": encoding},
            ) from error
