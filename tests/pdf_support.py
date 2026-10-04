"""Generate PDFs with PyMuPDF whose text layer holds exactly the given text.

PyMuPDF's built-in fonts turn a soft hyphen into "-" on extraction and cannot
show Cyrillic. Real PDFs carry such text through a ToUnicode map, and so do
these: lines are written as single-byte codes laid out like cp1251 (ASCII,
Cyrillic, the soft hyphen at 0xAD) and every font gets a ToUnicode map for
that layout. Control characters other than the line break map to themselves,
so that broken text layers can be generated too. Text is placed line by line;
extraction ends each line with a line break.

U+FFFD cannot be generated this way: MuPDF ignores a ToUnicode entry for it
and falls back to the code, so pages of replacement characters are tested on
the classifier with plain strings.
"""

from collections.abc import Sequence
from pathlib import Path

import pymupdf

_FONT_SIZE = 9
_LINE_START = (50, 60)


def _encode(line: str) -> str:
    """The line as single-byte codes, spelled as Latin-1 for insert_text."""
    return line.encode("cp1251").decode("latin-1")


def _to_unicode_cmap() -> bytes:
    pairs = [f"<{code:02X}> <{code:04X}>" for code in range(0x20) if code != 0x0A]
    for code in range(0x20, 0x100):
        try:
            character = bytes([code]).decode("cp1251")
        except UnicodeDecodeError:
            continue
        pairs.append(f"<{code:02X}> <{ord(character):04X}>")
    blocks = [
        f"{len(pairs[i : i + 100])} beginbfchar\n"
        + "\n".join(pairs[i : i + 100])
        + "\nendbfchar"
        for i in range(0, len(pairs), 100)
    ]
    return (
        "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        "/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n"
        "1 begincodespacerange\n<00> <FF>\nendcodespacerange\n"
        + "\n".join(blocks)
        + "\nendcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n"
    ).encode()


def pdf_bytes(
    pages: Sequence[Sequence[str] | None], *, title: str | None = None
) -> bytes:
    """A PDF with one page per entry: its lines of text, or None for no text."""
    document = pymupdf.open()
    if title is not None:
        document.set_metadata({"title": title})
    for lines in pages:
        page = document.new_page()
        if lines:
            encoded = "\n".join(_encode(line) for line in lines)
            page.insert_text(_LINE_START, encoded, fontname="helv", fontsize=_FONT_SIZE)
    cmap = document.get_new_xref()
    document.update_object(cmap, "<<>>")
    document.update_stream(cmap, _to_unicode_cmap())
    for number in range(document.page_count):
        for font in document[number].get_fonts():
            document.xref_set_key(font[0], "ToUnicode", f"{cmap} 0 R")
    data = document.tobytes()
    document.close()
    return bytes(data)


def write_pdf(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


_WORDS = (
    "the archive holds letters maps and notes from the river survey of the "
    "northern valley where the expedition measured stones and water levels"
).split()


def prose(seed: int, lines: int = 6) -> list[str]:
    """Readable English lines, about 75 characters each."""
    out = []
    for line in range(lines):
        words = [
            _WORDS[(seed * 5 + line * 7 + step) % len(_WORDS)] for step in range(13)
        ]
        out.append(" ".join(words))
    out[0] = f"marker{seed} " + out[0]
    return out


def garbled(seed: int, lines: int = 6) -> list[str]:
    """An upper-case run-on text layer without spaces, as from a broken font.

    Only weak signals fire on it (long tokens), so it is never dropped by
    itself; see ``broken`` for a text layer that is.
    """
    letters = "QWERTYUIOPASDFGHJKLZXCVBNM"
    return [
        "".join(letters[(seed + line * 3 + step * 7) % 26] for step in range(70))
        for line in range(lines)
    ]


def broken(seed: int, lines: int = 6) -> list[str]:
    """A broken text layer: NUL and other control characters among letters,
    almost no spaces, as extracted from fonts without a Unicode mapping."""
    controls = "\x00\x01\x0e\x10\x11\x1f"
    letters = "abcdefghijklmnopqrstuvwxyz"
    return [
        "".join(
            controls[(seed + step) % len(controls)]
            if step % 3 == 0
            else letters[(seed * 7 + line + step) % 26]
            for step in range(70)
        )
        for line in range(lines)
    ]
