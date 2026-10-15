"""DOCX parser reading the main document body in place with python-docx.

python-docx is imported when the first DOCX is parsed, so a missing library
fails only DOCX sources (``docx_library_missing``); they are retried by the
next update like any failed source.
"""

import re
import zipfile
import zlib
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import IO, TYPE_CHECKING

from docgrep.adapters.tool_output import scrub_tool_output
from docgrep.domain.errors import ParseError
from docgrep.domain.models import FileType, ParsedDocument, ParsedSection, Source

if TYPE_CHECKING:
    from docx.document import Document
    from docx.oxml.table import CT_Tbl, CT_Tc
    from docx.oxml.text.paragraph import CT_P
    from docx.oxml.xmlchemy import BaseOxmlElement

# Reason codes of DOCX failures (ParseError details.reason).
LIBRARY_MISSING = "docx_library_missing"
NOT_A_ZIP = "not_a_zip"
NOT_OOXML = "not_ooxml"
NOT_A_WORD_DOCUMENT = "not_a_word_document"
ENCRYPTED = "encrypted_content"
MEMBER_TOO_LARGE = "member_too_large"
UNREADABLE = "unreadable"
INSTALL_HINT = (
    "install python-docx into the docgrep environment (uv pip install python-docx)"
)

# Office encrypts a password-protected DOCX into an OLE compound file, which
# starts with this signature instead of a ZIP header.
_OLE_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_CONTENT_TYPES = "[Content_Types].xml"
# Upper bound for one decompressed member, as for EPUB; python-docx reads
# parts whole, and a member never yields more than its declared size.
MAX_MEMBER_BYTES = 64 * 1024 * 1024
# Separator of the cells of one table row.
CELL_SEPARATOR = " | "
# Built-in heading styles carry an outline level, whatever the UI language;
# their ids and names are the fallback for styles that do not.
_HEADING_ID = re.compile(r"^heading\s*([1-9])$", re.IGNORECASE)
_TITLE_LEVEL = 0
# Outline level 9 is "body text": explicitly not a heading.
_BODY_TEXT_OUTLINE = 9
# Style inheritance deeper than this is treated as a cycle.
_MAX_STYLE_DEPTH = 16
# Paragraph text in document order: runs, hyperlinks and tracked insertions;
# tracked deletions are kept in w:delText and so never read.
_TEXT_NODES = "./w:r | ./w:hyperlink | ./w:ins/w:r | ./w:ins/w:hyperlink"


def _fail(source: Source, message: str, reason: str) -> ParseError:
    return ParseError(message, details={"path": source.path, "reason": reason})


def _outline_level(element: "BaseOxmlElement") -> int | None:
    values = element.xpath("./w:pPr/w:outlineLvl/@w:val")
    for value in values:
        try:
            return int(str(value))
        except ValueError:
            return None
    return None


class _Styles:
    """Heading levels of the document's paragraph styles."""

    def __init__(self, document: "Document") -> None:
        self._outline: dict[str, int | None] = {}
        self._based_on: dict[str, str] = {}
        self._named: dict[str, int] = {}
        self.default: str | None = None
        for style in document.styles.element.xpath("./w:style[@w:type='paragraph']"):
            style_id = str(style.get(_w("styleId")) or "")
            if not style_id:
                continue
            self._outline[style_id] = _outline_level(style)
            based = style.xpath("./w:basedOn/@w:val")
            if based:
                self._based_on[style_id] = str(based[0])
            names = [str(name) for name in style.xpath("./w:name/@w:val")]
            level = _named_level(style_id, names[0] if names else "")
            if level is not None:
                self._named[style_id] = level
            if style.get(_w("default")) in ("1", "true", "on"):
                self.default = style_id

    def level(self, paragraph: "CT_P") -> int | None:
        """The heading level (0 for a title) of a paragraph, None for text."""
        direct = _outline_level(paragraph)
        if direct is not None:
            return _from_outline(direct)
        style_id = paragraph.style or self.default
        current = style_id
        for _ in range(_MAX_STYLE_DEPTH):
            if current is None:
                break
            outline = self._outline.get(current)
            if outline is not None:
                return _from_outline(outline)
            current = self._based_on.get(current)
        return None if style_id is None else self._named.get(style_id)


def _w(name: str) -> str:
    from docx.oxml.ns import qn

    return qn(f"w:{name}")


def _from_outline(value: int) -> int | None:
    if 0 <= value < _BODY_TEXT_OUTLINE:
        return value + 1
    return None


def _named_level(style_id: str, name: str) -> int | None:
    for candidate in (style_id, name):
        match = _HEADING_ID.match(candidate.strip())
        if match:
            return int(match.group(1))
        if candidate.strip().lower() == "title":
            return _TITLE_LEVEL
    return None


def _paragraph_text(paragraph: "CT_P") -> str:
    return "".join(str(node.text) for node in paragraph.xpath(_TEXT_NODES)).strip()


def _cell_parts(cell: "CT_Tc") -> tuple[str, list[str]]:
    """A cell's own text, and the rows of the tables nested in it, in order."""
    from docx.oxml.text.paragraph import CT_P

    texts: list[str] = []
    nested: list[str] = []
    for element in cell.inner_content_elements:
        if isinstance(element, CT_P):
            text = _paragraph_text(element)
            if text:
                texts.append(text)
        else:
            nested.extend(_table_rows(element))
    return " ".join(texts), nested


def _table_rows(table: "CT_Tbl") -> Iterator[str]:
    """One line per non-empty row, cells joined by ``CELL_SEPARATOR``.

    A cell spanning several columns is one ``w:tc``, so it is read once; the
    continuation cells of a vertical merge hold no text of their own and are
    skipped. Rows of a nested table follow the row that contains it.
    """
    for row in table.tr_lst:
        cells: list[str] = []
        nested: list[str] = []
        for cell in row.tc_lst:
            if cell.vMerge == "continue":
                continue
            text, inner = _cell_parts(cell)
            if text:
                cells.append(text)
            nested.extend(inner)
        if cells:
            yield CELL_SEPARATOR.join(cells)
        yield from nested


class _Sections:
    """Text blocks grouped under the heading path that precedes them."""

    def __init__(self) -> None:
        self.headings: list[tuple[int, str]] = []
        self.sections: list[ParsedSection] = []
        self._blocks: list[str] = []

    def heading(self, level: int, title: str) -> None:
        self._flush()
        while self.headings and self.headings[-1][0] >= level:
            self.headings.pop()
        self.headings.append((level, title))

    def block(self, text: str) -> None:
        if text.strip():
            self._blocks.append(text)

    def _flush(self) -> None:
        if self._blocks:
            self.sections.append(
                ParsedSection(
                    text="\n\n".join(self._blocks),
                    heading_path=tuple(title for _, title in self.headings),
                )
            )
        self._blocks = []

    def finish(self) -> tuple[ParsedSection, ...]:
        self._flush()
        return tuple(self.sections)


def _body_sections(document: "Document") -> tuple[ParsedSection, ...]:
    from docx.oxml.text.paragraph import CT_P

    styles = _Styles(document)
    sections = _Sections()
    for element in document.element.body.inner_content_elements:
        if isinstance(element, CT_P):
            text = _paragraph_text(element)
            level = styles.level(element)
            if level is not None:
                title = " ".join(text.split())
                if title:
                    sections.heading(level, title)
            else:
                sections.block(text)
        else:
            sections.block("\n".join(_table_rows(element)))
    return sections.finish()


def _check_package(source: Source, stream: IO[bytes]) -> None:
    """Reject what python-docx would misreport or read without a bound."""
    if stream.read(len(_OLE_SIGNATURE)) == _OLE_SIGNATURE:
        raise _fail(
            source,
            "DOCX is encrypted or password-protected (OLE compound file)",
            ENCRYPTED,
        )
    stream.seek(0)
    with zipfile.ZipFile(stream) as archive:
        if _CONTENT_TYPES not in archive.namelist():
            raise _fail(source, "DOCX is not a valid OOXML package", NOT_OOXML)
        for info in archive.infolist():
            if info.flag_bits & 0x1:
                raise _fail(source, "DOCX content is encrypted", ENCRYPTED)
            if info.file_size > MAX_MEMBER_BYTES:
                raise _fail(source, "DOCX member is too large", MEMBER_TOO_LARGE)
    stream.seek(0)


def _metadata(document: "Document") -> tuple[str | None, str | None]:
    """Title and author from the core properties, when the package has them.

    python-docx invents defaults for a package without core properties, so
    those are read only when the part exists.
    """
    from docx.opc.constants import RELATIONSHIP_TYPE

    rels = document.part.package.rels.values()
    if not any(rel.reltype == RELATIONSHIP_TYPE.CORE_PROPERTIES for rel in rels):
        return None, None
    properties = document.core_properties
    title = " ".join((properties.title or "").split())
    author = " ".join((properties.author or "").split())
    return title or None, author or None


class DocxParser:
    """Extract DOCX body text in document order with heading paths.

    Paragraphs and tables are read as they appear in the body; table rows
    become lines with cells joined by ``CELL_SEPARATOR``. Headings are found
    by outline level, then by the built-in style ids and names, and feed the
    heading path as in EPUB. DOCX has no reliable page numbers, so they stay
    None. Headers, footers, notes, comments, text boxes and images are not
    read. A document without body text is left to OCR, as for EPUB.
    """

    # Recorded with every source it parses; a change makes `docgrep update`
    # extract the sources of this type again.
    version = "1"

    def __init__(self, corpus_root: Path) -> None:
        self.corpus_root = corpus_root

    def parse(self, source: Source) -> ParsedDocument:
        """Return text blocks per heading and the core title and author."""
        if source.file_type is not FileType.DOCX:
            raise ParseError("source is not DOCX", details={"path": source.path})
        try:
            import docx
            from docx.opc.exceptions import OpcError
        except ImportError as error:
            raise ParseError(
                "DOCX cannot be read: python-docx is not installed",
                details={
                    "path": source.path,
                    "reason": LIBRARY_MISSING,
                    "hint": INSTALL_HINT,
                },
            ) from error
        path = self.corpus_root / source.path
        try:
            with path.open("rb") as stream:
                _check_package(source, stream)
                document = docx.Document(stream)
                sections = _body_sections(document)
                title, author = _metadata(document)
        except ParseError:
            raise
        except zipfile.BadZipFile as error:
            raise _fail(source, "DOCX is not a valid ZIP archive", NOT_A_ZIP) from error
        except ValueError as error:
            if "is not a Word file" in str(error):
                raise _fail(
                    source, "DOCX holds no Word document", NOT_A_WORD_DOCUMENT
                ) from error
            raise self._unreadable(source, error) from error
        except (
            OSError,
            EOFError,
            KeyError,
            RuntimeError,
            zlib.error,
            zipfile.LargeZipFile,
            OpcError,
            # lxml's XMLSyntaxError, raised for a malformed part, is one.
            SyntaxError,
        ) as error:
            raise self._unreadable(source, error) from error
        enriched = replace(source, title=title, author=author)
        return ParsedDocument(
            source=enriched, sections=sections, requires_ocr=not sections
        )

    def _unreadable(self, source: Source, error: Exception) -> ParseError:
        path = self.corpus_root / source.path
        return ParseError(
            "DOCX cannot be parsed",
            details={
                "path": source.path,
                "reason": UNREADABLE,
                "error": scrub_tool_output(str(error), (path, self.corpus_root)),
            },
        )
