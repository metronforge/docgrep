"""EPUB parser reading the package metadata and the spine with the stdlib only."""

import posixpath
import re
import zipfile
import zlib
from dataclasses import replace
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree

from docgrep.adapters.parsers._common import decode_text
from docgrep.domain.errors import ParseError
from docgrep.domain.models import FileType, ParsedDocument, ParsedSection, Source

_CONTAINER = "META-INF/container.xml"
_ENCRYPTION = "META-INF/encryption.xml"
_PACKAGE_MEDIA_TYPE = "application/oebps-package+xml"
_NCX_MEDIA_TYPE = "application/x-dtbncx+xml"
_CONTENT_MEDIA_TYPES = frozenset({"application/xhtml+xml", "text/html"})
# Font obfuscation (IDPF and Adobe) only scrambles embedded fonts; the text is
# readable, so these entries in encryption.xml are not treated as DRM.
_FONT_OBFUSCATION = frozenset(
    {"http://www.idpf.org/2008/embedding", "http://ns.adobe.com/pdf/enc#RC"}
)
# Upper bound for one decompressed member; one is held in memory at a time.
MAX_MEMBER_BYTES = 64 * 1024 * 1024

_XML_ENCODING = re.compile(
    rb"""^\s*<\?xml[^>]*?encoding\s*=\s*["']([A-Za-z0-9._:-]+)"""
)
_META_CHARSET = re.compile(
    rb"""<meta[^>]+charset\s*=\s*["']?([A-Za-z0-9._:-]+)""", re.IGNORECASE
)
_YEAR = re.compile(r"^\s*(\d{4})")
_HTML_SPACE = re.compile(r"[ \t\n\r\f]+")

_SKIPPED = frozenset(
    {
        "script",
        "style",
        "head",
        "noscript",
        "template",
        "svg",
        "math",
        "iframe",
        "object",
        "embed",
        "audio",
        "video",
        "canvas",
    }
)
_HEADINGS = {f"h{level}": level for level in range(1, 7)}
_BLOCKS = frozenset(
    {
        "html",
        "body",
        "p",
        "li",
        "ul",
        "ol",
        "dl",
        "dt",
        "dd",
        "blockquote",
        "pre",
        "td",
        "th",
        "tr",
        "table",
        "thead",
        "tbody",
        "tfoot",
        "caption",
        "div",
        "section",
        "article",
        "aside",
        "header",
        "footer",
        "nav",
        "main",
        "figure",
        "figcaption",
        "address",
        "hr",
        "center",
        "details",
        "summary",
    }
)


def _fail(source: Source, message: str, reason: str) -> ParseError:
    return ParseError(message, details={"path": source.path, "reason": reason})


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _attribute(element: ElementTree.Element, name: str) -> str | None:
    """An attribute by local name, whatever its namespace."""
    for key, value in element.attrib.items():
        if _local(key) == name:
            return value
    return None


def _html_name(tag: str) -> str:
    """An HTML tag name without a namespace prefix such as ``xhtml:``."""
    return tag.rsplit(":", 1)[-1]


def _collapse(text: str | None) -> str:
    return _HTML_SPACE.sub(" ", text or "").strip()


class _Book:
    """Reads members of one open EPUB archive within the memory bound."""

    def __init__(self, archive: zipfile.ZipFile, source: Source) -> None:
        self.archive = archive
        self.source = source

    def read(self, name: str, *, missing: str) -> bytes:
        try:
            info = self.archive.getinfo(name)
        except KeyError:
            raise _fail(self.source, "EPUB member is missing", missing) from None
        if info.flag_bits & 0x1:
            raise _fail(self.source, "EPUB content is encrypted", "encrypted_content")
        if info.file_size > MAX_MEMBER_BYTES:
            raise _fail(self.source, "EPUB member is too large", "member_too_large")
        with self.archive.open(info) as stream:
            data = stream.read(MAX_MEMBER_BYTES + 1)
        if len(data) > MAX_MEMBER_BYTES:
            raise _fail(self.source, "EPUB member is too large", "member_too_large")
        return data

    def xml(self, name: str, *, missing: str) -> ElementTree.Element:
        data = self.read(name, missing=missing)
        try:
            return ElementTree.fromstring(data)
        except ElementTree.ParseError as error:
            raise ParseError(
                "EPUB XML is malformed",
                details={
                    "path": self.source.path,
                    "reason": "malformed_xml",
                    "member": name,
                    "error": str(error),
                },
            ) from error


class _Extractor(HTMLParser):
    """Collect text blocks under heading paths from XHTML content documents.

    The heading stack outlives one document, so a chapter that starts with a
    lower-level heading continues the path of the chapters before it.
    """

    def __init__(self, headings: list[tuple[int, str]]) -> None:
        super().__init__(convert_charrefs=True)
        self.headings = headings
        self.sections: list[ParsedSection] = []
        self._blocks: list[str] = []
        self._parts: list[str] = []
        self._has_pre = False
        self._skip = 0
        self._pre = 0
        self._heading: tuple[int, list[str]] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = _html_name(tag)
        if tag == "body":
            # A head that was never closed must not hide the body.
            self._skip = 0
        if tag in _SKIPPED:
            self._skip += 1
            return
        if self._skip:
            return
        if tag == "br":
            self._text("\n", raw=True)
        elif tag in _HEADINGS:
            self._flush_block()
            self._heading = (_HEADINGS[tag], [])
        elif tag in _BLOCKS:
            self._flush_block()
            if tag == "pre":
                self._pre += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = _html_name(tag)
        if not self._skip and tag == "br":
            self._text("\n", raw=True)
        elif not self._skip and tag in _BLOCKS:
            self._flush_block()

    def handle_endtag(self, tag: str) -> None:
        tag = _html_name(tag)
        if tag in _SKIPPED:
            self._skip = max(self._skip - 1, 0)
            return
        if self._skip:
            return
        if tag in _HEADINGS and self._heading is not None:
            level, parts = self._heading
            self._heading = None
            title = _collapse("".join(parts))
            if title:
                self._flush_section()
                while self.headings and self.headings[-1][0] >= level:
                    self.headings.pop()
                self.headings.append((level, title))
        elif tag in _BLOCKS:
            self._flush_block()
            if tag == "pre" and self._pre:
                self._pre -= 1

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._heading is not None:
            self._heading[1].append(data)
            return
        self._text(data, raw=bool(self._pre))

    def _text(self, data: str, *, raw: bool) -> None:
        if raw:
            self._has_pre = self._has_pre or bool(self._pre)
            self._parts.append(data)
        else:
            self._parts.append(_HTML_SPACE.sub(" ", data))

    def _flush_block(self) -> None:
        text = "".join(self._parts)
        if self._has_pre:
            text = text.lstrip("\r\n").rstrip()
        else:
            text = "\n".join(line.strip(" ") for line in text.split("\n")).strip("\n")
        if text.strip():
            self._blocks.append(text)
        self._parts = []
        self._has_pre = False

    def _flush_section(self) -> None:
        self._flush_block()
        if self._blocks:
            self.sections.append(
                ParsedSection(
                    text="\n\n".join(self._blocks),
                    heading_path=tuple(title for _, title in self.headings),
                )
            )
        self._blocks = []

    def finish(self) -> list[ParsedSection]:
        self.close()
        if self._heading is not None:
            # An unclosed heading: keep its words as text rather than lose them.
            self._parts.extend(self._heading[1])
            self._heading = None
        self._flush_section()
        return self.sections


def _declared_encoding(data: bytes) -> str | None:
    head = data[:2048]
    match = _XML_ENCODING.match(head) or _META_CHARSET.search(head)
    return match.group(1).decode("ascii") if match else None


def _check_encryption(book: _Book) -> None:
    if _ENCRYPTION not in book.archive.namelist():
        return
    root = book.xml(_ENCRYPTION, missing="missing_encryption")
    for element in root.iter():
        if _local(element.tag) != "EncryptionMethod":
            continue
        if _attribute(element, "Algorithm") not in _FONT_OBFUSCATION:
            raise _fail(
                book.source,
                "EPUB content is encrypted (DRM)",
                "encrypted_content",
            )


def _package_path(book: _Book) -> str:
    root = book.xml(_CONTAINER, missing="missing_container")
    rootfiles = [e for e in root.iter() if _local(e.tag) == "rootfile"]
    rootfiles.sort(key=lambda e: _attribute(e, "media-type") != _PACKAGE_MEDIA_TYPE)
    for element in rootfiles:
        full_path = _attribute(element, "full-path")
        if full_path:
            return full_path
    raise _fail(
        book.source, "EPUB names no package document", "missing_package_document"
    )


def _metadata(
    package: ElementTree.Element,
) -> tuple[str | None, str | None, int | None]:
    titles: list[str] = []
    creators: list[str] = []
    dates: list[tuple[bool, str]] = []
    metadata = next((e for e in package.iter() if _local(e.tag) == "metadata"), None)
    for element in () if metadata is None else metadata.iter():
        name = _local(element.tag)
        text = _collapse("".join(element.itertext()))
        if not text:
            continue
        if name == "title":
            titles.append(text)
        elif name == "creator":
            creators.append(text)
        elif name == "date":
            dates.append((_attribute(element, "event") == "publication", text))
    year = None
    for _, text in sorted(dates, key=lambda item: not item[0]):
        match = _YEAR.match(text)
        if match:
            year = int(match.group(1))
            break
    return (
        titles[0] if titles else None,
        "; ".join(dict.fromkeys(creators)) or None,
        year,
    )


def _spine(book: _Book, package: ElementTree.Element, base: str) -> list[str]:
    """Content document member names in reading order, without navigation."""
    items: dict[str, tuple[str, str, frozenset[str]]] = {}
    spine: ElementTree.Element | None = None
    for element in package.iter():
        name = _local(element.tag)
        if name == "item":
            item_id = _attribute(element, "id")
            href = _attribute(element, "href")
            if item_id and href:
                member = posixpath.normpath(
                    posixpath.join(base, unquote(href.split("#", 1)[0]))
                )
                items[item_id] = (
                    member,
                    (_attribute(element, "media-type") or "").strip().lower(),
                    frozenset((_attribute(element, "properties") or "").split()),
                )
        elif name == "spine" and spine is None:
            spine = element
    if spine is None:
        raise _fail(book.source, "EPUB has no spine", "no_readable_spine_items")
    toc = _attribute(spine, "toc")
    members: list[str] = []
    for itemref in spine:
        if _local(itemref.tag) != "itemref":
            continue
        idref = _attribute(itemref, "idref")
        if idref is None or idref == toc or idref not in items:
            continue
        member, media_type, properties = items[idref]
        if "nav" in properties or media_type == _NCX_MEDIA_TYPE:
            continue
        if media_type not in _CONTENT_MEDIA_TYPES:
            continue
        members.append(member)
    if not members:
        raise _fail(
            book.source,
            "EPUB has no readable spine items",
            "no_readable_spine_items",
        )
    return members


class EpubParser:
    """Extract EPUB 2 and 3 text in spine order with heading paths.

    Page numbers do not exist in reflowable books and stay None. Each content
    document is read, decoded and parsed on its own, so memory is bounded by
    one member.
    """

    # Recorded with every source it parses; a change makes `docgrep update`
    # extract the sources of this type again.
    version = "1"

    def __init__(self, corpus_root: Path) -> None:
        self.corpus_root = corpus_root

    def parse(self, source: Source) -> ParsedDocument:
        """Return text blocks per heading and the package's bibliographic data."""
        if source.file_type is not FileType.EPUB:
            raise ParseError("source is not EPUB", details={"path": source.path})
        try:
            with zipfile.ZipFile(self.corpus_root / source.path) as archive:
                return self._parse(_Book(archive, source))
        except ParseError:
            raise
        except zipfile.BadZipFile as error:
            raise ParseError(
                "EPUB is not a valid ZIP archive",
                details={"path": source.path, "reason": "not_a_zip"},
            ) from error
        except (
            OSError,
            EOFError,
            RuntimeError,
            ValueError,
            LookupError,
            zlib.error,
            zipfile.LargeZipFile,
        ) as error:
            raise ParseError(
                "EPUB cannot be parsed",
                details={
                    "path": source.path,
                    "reason": "unreadable",
                    "error": str(error),
                },
            ) from error

    def _parse(self, book: _Book) -> ParsedDocument:
        _check_encryption(book)
        package_path = _package_path(book)
        package = book.xml(package_path, missing="missing_package_document")
        title, author, year = _metadata(package)
        members = _spine(book, package, posixpath.dirname(package_path))
        headings: list[tuple[int, str]] = []
        sections: list[ParsedSection] = []
        for member in members:
            data = book.read(member, missing="missing_content_document")
            text = decode_text(data, book.source.path, _declared_encoding(data))
            extractor = _Extractor(headings)
            extractor.feed(text)
            sections.extend(extractor.finish())
        enriched = replace(book.source, title=title, author=author, year=year)
        return ParsedDocument(
            source=enriched, sections=tuple(sections), requires_ocr=not sections
        )
