"""Generate small DOCX files with python-docx, so no binary fixture is committed."""

import io
import struct
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import docx
from docx.document import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

# An OLE compound file header: what Office writes for a password-protected DOCX.
OLE_HEADER = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 504


def tiny_png() -> bytes:
    """A valid 1x1 grey PNG."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    pixels = zlib.compress(b"\x00\x80")
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", pixels)
        + chunk(b"IEND", b"")
    )


def set_outline_level(element: Any, level: int) -> None:
    """Give a style or paragraph element an explicit ``w:outlineLvl``."""
    pPr = element.get_or_add_pPr()
    for old in pPr.findall(qn("w:outlineLvl")):
        pPr.remove(old)
    outline = OxmlElement("w:outlineLvl")
    outline.set(qn("w:val"), str(level))
    pPr.append(outline)


def add_heading_style(document: Document, name: str, level: int) -> str:
    """A custom paragraph style that is a heading by outline level only."""
    style = document.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    set_outline_level(style.element, level - 1)
    return name


def docx_bytes(build: Callable[[Document], None], **core: str) -> bytes:
    """A DOCX made by ``build`` on a new document, with core properties set."""
    document = docx.Document()
    for key, value in core.items():
        setattr(document.core_properties, key, value)
    build(document)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def write_docx(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
