from pathlib import Path

import pytest

from docgrep.adapters.parsers import MarkdownParser, TextParser
from docgrep.domain.errors import ParseError
from docgrep.domain.models import FileType, Source
from docgrep.domain.protocols import DocumentParser


def source(path: str, file_type: FileType) -> Source:
    return Source(source_id=path, path=path, file_type=file_type)


def test_txt_parser_produces_one_verbatim_section(tmp_path: Path) -> None:
    text = "First line.\n\nSecond line.\n"
    (tmp_path / "note.txt").write_text(text, encoding="utf-8")
    parser: DocumentParser = TextParser(tmp_path)

    document = parser.parse(source("note.txt", FileType.TXT))

    assert len(document.sections) == 1
    assert document.sections[0].text == text
    assert document.sections[0].heading_path == ()
    assert document.source.title is None
    assert document.source.author is None
    assert document.source.year is None


def test_txt_parser_falls_back_to_detected_encoding(tmp_path: Path) -> None:
    (tmp_path / "latin.txt").write_bytes("Café déjà vu".encode("cp1252"))

    document = TextParser(tmp_path).parse(source("latin.txt", FileType.TXT))

    assert document.sections[0].text == "Café déjà vu"


def test_txt_parser_reports_undecodable_input(tmp_path: Path) -> None:
    (tmp_path / "broken.txt").write_bytes(b"\x81")

    with pytest.raises(ParseError) as raised:
        TextParser(tmp_path).parse(source("broken.txt", FileType.TXT))

    assert raised.value.category == "parse_error"
    assert raised.value.details["path"] == "broken.txt"


def test_markdown_heading_paths_and_frontmatter(tmp_path: Path) -> None:
    markdown = (
        "---\n"
        'title: "Research note"\n'
        "author: Ada Lovelace\n"
        "year: 1843\n"
        "---\n"
        "# Main\n"
        "Opening [[related note|link]].\n\n"
        "## Detail\n"
        "Deep text.\n\n"
        "### Subdetail\n"
        "Nested text.\n\n"
        "## Sibling\n"
        "More text.\n"
    )
    (tmp_path / "note.md").write_text(markdown, encoding="utf-8")
    parser: DocumentParser = MarkdownParser(tmp_path)

    document = parser.parse(source("note.md", FileType.MARKDOWN))

    assert document.source.title == "Research note"
    assert document.source.author == "Ada Lovelace"
    assert document.source.year == 1843
    assert [section.heading_path for section in document.sections] == [
        ("Main",),
        ("Main", "Detail"),
        ("Main", "Detail", "Subdetail"),
        ("Main", "Sibling"),
    ]
    assert [section.text for section in document.sections] == [
        "Opening [[related note|link]].\n\n",
        "Deep text.\n\n",
        "Nested text.\n\n",
        "More text.\n",
    ]
    assert document.metadata == (("wikilink", "related note"),)


def test_markdown_without_frontmatter_does_not_invent_metadata(tmp_path: Path) -> None:
    (tmp_path / "plain.md").write_text("# Title\nBody.\n", encoding="utf-8")

    document = MarkdownParser(tmp_path).parse(source("plain.md", FileType.MARKDOWN))

    assert document.source.title is None
    assert document.source.author is None
    assert document.source.year is None
    assert document.sections[0].heading_path == ("Title",)
    assert document.sections[0].text == "Body.\n"
    assert document.metadata == ()


def test_markdown_ignores_heading_syntax_inside_fenced_code(tmp_path: Path) -> None:
    (tmp_path / "code.md").write_text(
        "# Real\nBefore.\n```markdown\n# Not a heading\n```\nAfter.\n",
        encoding="utf-8",
    )

    document = MarkdownParser(tmp_path).parse(source("code.md", FileType.MARKDOWN))

    assert len(document.sections) == 1
    assert document.sections[0].heading_path == ("Real",)
    assert "# Not a heading" in document.sections[0].text
