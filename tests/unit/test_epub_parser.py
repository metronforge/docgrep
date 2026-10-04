"""The EPUB parser on books generated in the test (UC-03)."""

from collections.abc import Callable
from pathlib import Path

import pytest

from docgrep.adapters.chunking import TokenChunker, WhitespaceTokenizer
from docgrep.adapters.parsers import EpubParser
from docgrep.domain.errors import ParseError
from docgrep.domain.models import FileType, ParsedDocument, Source
from docgrep.domain.protocols import DocumentParser
from epub_support import (
    AES,
    FONT_OBFUSCATION,
    NAV_TEXT,
    encryption_xml,
    epub_bytes,
    write_epub,
    xhtml,
)


def source(path: str = "book.epub", file_type: FileType = FileType.EPUB) -> Source:
    return Source(source_id=path, path=path, file_type=file_type)


def parse(root: Path, data: bytes, name: str = "book.epub") -> ParsedDocument:
    write_epub(root / name, data)
    parser: DocumentParser = EpubParser(root)
    return parser.parse(source(name))


def reason(root: Path, data: bytes) -> str:
    with pytest.raises(ParseError) as raised:
        parse(root, data)
    assert raised.value.category == "parse_error"
    assert raised.value.details["path"] == "book.epub"
    return str(raised.value.details["reason"])


NESTED = (
    (
        "one.xhtml",
        xhtml(
            "<p>Before any heading.</p>"
            "<h1>Part One</h1><p>Opening of part one.</p>"
            "<h2>Chapter 1</h2><p>First chapter text.</p><p>Second paragraph.</p>"
        ),
    ),
    (
        "two.xhtml",
        xhtml(
            "<h2>Chapter 2</h2><p>Second chapter text.</p>"
            "<h3>Section 2.1</h3><blockquote>A quoted line.</blockquote>"
        ),
    ),
    ("three.xhtml", xhtml("<p>Continues section 2.1 in a new file.</p>")),
    ("four.xhtml", xhtml("<h1>Part Two</h1><ul><li>Item a</li><li>Item b</li></ul>")),
)


def test_epub3_heading_paths_continue_across_chapters(tmp_path: Path) -> None:
    document = parse(tmp_path, epub_bytes(NESTED))

    assert [(s.heading_path, s.text) for s in document.sections] == [
        ((), "Before any heading."),
        (("Part One",), "Opening of part one."),
        (("Part One", "Chapter 1"), "First chapter text.\n\nSecond paragraph."),
        (("Part One", "Chapter 2"), "Second chapter text."),
        (("Part One", "Chapter 2", "Section 2.1"), "A quoted line."),
        (
            ("Part One", "Chapter 2", "Section 2.1"),
            "Continues section 2.1 in a new file.",
        ),
        (("Part Two",), "Item a\n\nItem b"),
    ]
    assert all(s.page_number is None for s in document.sections)
    assert not document.requires_ocr


def test_epub3_metadata_and_navigation_document_is_skipped(tmp_path: Path) -> None:
    document = parse(
        tmp_path,
        epub_bytes(
            NESTED, creators=("Ada Lovelace", "Charles Babbage"), date="1843-07-01"
        ),
    )

    assert document.source.title == "A Book"
    assert document.source.author == "Ada Lovelace; Charles Babbage"
    assert document.source.year == 1843
    text = "\n".join(s.text for s in document.sections)
    assert NAV_TEXT not in text
    assert "Head title" not in text and "color" not in text and "hidden" not in text


def test_epub2_with_ncx_and_publication_date(tmp_path: Path) -> None:
    data = epub_bytes(
        NESTED[:2], version=2, date="2001-05-12", date_event="publication"
    )

    document = parse(tmp_path, data)

    assert document.source.year == 2001
    assert [s.heading_path for s in document.sections][-1] == (
        "Part One",
        "Chapter 2",
        "Section 2.1",
    )
    assert NAV_TEXT not in "\n".join(s.text for s in document.sections)


def replace_in(member: str, old: bytes, new: bytes) -> Callable[[str, bytes], bytes]:
    def edit(name: str, content: bytes) -> bytes:
        return content.replace(old, new) if name.endswith(member) else content

    return edit


def test_publication_date_wins_over_other_dates(tmp_path: Path) -> None:
    # Two dates: the publication event is preferred over an earlier entry.
    data = epub_bytes(
        NESTED[:1],
        version=2,
        date=None,
        edit=replace_in(
            "content.opf",
            b"</metadata>",
            b'<dc:date opf:event="modification">2020</dc:date>'
            b'<dc:date opf:event="publication">1999-01-01</dc:date></metadata>',
        ),
    )

    assert parse(tmp_path, data).source.year == 1999


def test_spine_order_wins_over_manifest_order(tmp_path: Path) -> None:
    chapters = [
        ("a.xhtml", xhtml("<p>Alpha</p>")),
        ("b.xhtml", xhtml("<p>Bravo</p>")),
        ("c.xhtml", xhtml("<p>Charlie</p>")),
    ]

    document = parse(
        tmp_path, epub_bytes(chapters, spine=["c.xhtml", "a.xhtml", "b.xhtml"])
    )

    assert [s.text for s in document.sections] == ["Charlie", "Alpha", "Bravo"]


def test_cyrillic_in_utf8_and_in_declared_windows_1251(tmp_path: Path) -> None:
    text = "Съешь же ещё этих мягких французских булок"
    chapters = [
        ("utf.xhtml", xhtml(f"<h1>Глава</h1><p>{text}</p>")),
        ("cp.xhtml", xhtml(f"<p>{text}, да выпей чаю.</p>", encoding="windows-1251")),
    ]

    document = parse(tmp_path, epub_bytes(chapters, title="Книга"))

    assert document.source.title == "Книга"
    assert [(s.heading_path, s.text) for s in document.sections] == [
        (("Глава",), text),
        (("Глава",), f"{text}, да выпей чаю."),
    ]


def test_undeclared_legacy_encoding_falls_back_to_detection(tmp_path: Path) -> None:
    text = "Café déjà vu, naïve façade, crème brûlée. " * 8
    chapters = [("x.xhtml", xhtml(f"<p>{text}</p>", encoding="cp1252", declare=False))]

    document = parse(tmp_path, epub_bytes(chapters))

    assert document.sections[0].text == text.strip()


def test_an_unknown_declared_encoding_is_ignored(tmp_path: Path) -> None:
    body = xhtml("<p>Plain words</p>").replace(b'encoding="utf-8"', b'encoding="nope"')

    document = parse(tmp_path, epub_bytes([("x.xhtml", body)]))

    assert document.sections[0].text == "Plain words"


def test_entities_are_decoded(tmp_path: Path) -> None:
    chapters = [
        (
            "x.xhtml",
            xhtml(
                "<p>Fish &amp; chips &lt;tag&gt; &#1046; &#x416; &mdash; a&nbsp;b</p>"
            ),
        )
    ]

    document = parse(tmp_path, epub_bytes(chapters))

    assert document.sections[0].text == "Fish & chips <tag> Ж Ж — a\u00a0b"


def test_blocks_whitespace_pre_tables_and_ignored_elements(tmp_path: Path) -> None:
    body = (
        "<div>Loose   text\n in a div</div>"
        "<p>Line one<br/>line two</p>"
        "<pre>  indented\n    code</pre>"
        "<table><tr><th>Head</th><td>Cell <b>bold</b> end</td></tr></table>"
        "<dl><dt>Term</dt><dd>Definition</dd></dl>"
        "<svg><text>vector text</text></svg>"
        "<p><img src='x.png' alt='image alt'/>After image</p>"
        "<p>   </p>"
    )

    document = parse(tmp_path, epub_bytes([("x.xhtml", xhtml(body))]))

    assert document.sections[0].text == (
        "Loose text in a div\n\n"
        "Line one\nline two\n\n"
        "  indented\n    code\n\n"
        "Head\n\nCell bold end\n\n"
        "Term\n\nDefinition\n\n"
        "After image"
    )


def test_missing_metadata_stays_none(tmp_path: Path) -> None:
    document = parse(
        tmp_path, epub_bytes(NESTED[:1], title=None, creators=(), date=None)
    )

    assert document.source.title is None
    assert document.source.author is None
    assert document.source.year is None


def test_drm_encryption_is_rejected(tmp_path: Path) -> None:
    data = epub_bytes(NESTED, encryption=encryption_xml(FONT_OBFUSCATION, AES))

    assert reason(tmp_path, data) == "encrypted_content"


def test_font_obfuscation_alone_is_ignored(tmp_path: Path) -> None:
    data = epub_bytes(
        NESTED,
        encryption=encryption_xml(FONT_OBFUSCATION, "http://ns.adobe.com/pdf/enc#RC"),
    )

    assert parse(tmp_path, data).sections[0].text == "Before any heading."


def test_corrupt_zip_is_rejected(tmp_path: Path) -> None:
    assert reason(tmp_path, b"PK\x03\x04 this is not really a zip") == "not_a_zip"
    good = epub_bytes(NESTED)
    assert reason(tmp_path, good[: len(good) // 2]) == "not_a_zip"


def test_missing_container_and_package(tmp_path: Path) -> None:
    no_container = epub_bytes(
        NESTED, edit=lambda name, content: None if "container" in name else content
    )
    assert reason(tmp_path, no_container) == "missing_container"

    no_rootfile = epub_bytes(
        NESTED, edit=replace_in("container.xml", b"<rootfile ", b"<other ")
    )
    assert reason(tmp_path, no_rootfile) == "missing_package_document"

    no_package = epub_bytes(
        NESTED, edit=lambda name, content: None if name.endswith(".opf") else content
    )
    assert reason(tmp_path, no_package) == "missing_package_document"


def test_no_readable_spine_items(tmp_path: Path) -> None:
    only_nav = epub_bytes([], nav=True)
    assert reason(tmp_path, only_nav) == "no_readable_spine_items"

    no_spine = epub_bytes(NESTED, edit=replace_in("content.opf", b"spine", b"nospine"))
    assert reason(tmp_path, no_spine) == "no_readable_spine_items"


def test_malformed_xml_and_html_never_crash(tmp_path: Path) -> None:
    broken_opf = epub_bytes(
        NESTED, edit=replace_in("content.opf", b"</manifest>", b"</manifest")
    )
    assert reason(tmp_path, broken_opf) == "malformed_xml"

    sloppy = b"<html><body><p>Unclosed <b>bold<p>Next <h2>Open heading"
    document = parse(tmp_path, epub_bytes([("x.xhtml", sloppy)]))
    # An unclosed heading never becomes a path; its words stay as text.
    assert [(s.heading_path, s.text) for s in document.sections] == [
        ((), "Unclosed bold\n\nNext\n\nOpen heading")
    ]


def test_missing_content_document_is_rejected(tmp_path: Path) -> None:
    data = epub_bytes(
        NESTED, edit=replace_in("content.opf", b'href="two.xhtml"', b'href="x.xhtml"')
    )

    assert reason(tmp_path, data) == "missing_content_document"


def test_empty_book_requires_ocr(tmp_path: Path) -> None:
    chapters = [("x.xhtml", xhtml("<p>  </p><img src='scan.png'/>"))]

    document = parse(tmp_path, epub_bytes(chapters))

    assert document.sections == ()
    assert document.requires_ocr


def test_wrong_file_type_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ParseError):
        EpubParser(tmp_path).parse(source("book.txt", FileType.TXT))


def test_unreadable_file_is_a_parse_error(tmp_path: Path) -> None:
    with pytest.raises(ParseError):
        EpubParser(tmp_path).parse(source("missing.epub"))


def test_same_bytes_give_the_same_passages(tmp_path: Path) -> None:
    data = epub_bytes(NESTED)
    chunker = TokenChunker(WhitespaceTokenizer(), chunk_size=4, chunk_overlap=1)

    first = chunker.chunk(parse(tmp_path, data, "a/book.epub"))
    write_epub(tmp_path / "b/book.epub", data)
    second = chunker.chunk(EpubParser(tmp_path).parse(source("a/book.epub")))

    assert first == second
    assert [p.passage_id for p in first] == [p.passage_id for p in second]


def test_chunked_passages_carry_heading_paths(tmp_path: Path) -> None:
    chunker = TokenChunker(WhitespaceTokenizer(), chunk_size=3, chunk_overlap=1)

    passages = chunker.chunk(parse(tmp_path, epub_bytes(NESTED)))

    by_text = {p.text: p.heading_path for p in passages}
    assert by_text["Second chapter text."] == ("Part One", "Chapter 2")
    assert by_text["A quoted line."] == ("Part One", "Chapter 2", "Section 2.1")
    assert all(p.page_number is None for p in passages)
    assert len({p.passage_id for p in passages}) == len(passages)
