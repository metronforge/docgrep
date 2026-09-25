"""Markdown document parser with heading and frontmatter metadata."""

import re
from dataclasses import replace
from pathlib import Path

import yaml
from markdown_it import MarkdownIt

from docgrep.adapters.parsers._common import read_text
from docgrep.domain.errors import ParseError
from docgrep.domain.models import FileType, ParsedDocument, ParsedSection, Source

_WIKILINK = re.compile(r"\[\[([^\[\]\n]+)\]\]")


def _frontmatter(text: str, source: Source) -> tuple[dict[str, object], str]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, text
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() in ("---", "..."):
            try:
                loaded: object = yaml.safe_load("".join(lines[1:index]))
            except yaml.YAMLError as error:
                raise ParseError(
                    "invalid Markdown frontmatter",
                    details={"path": source.path, "reason": str(error)},
                ) from error
            if loaded is None:
                return {}, "".join(lines[index + 1 :])
            if not isinstance(loaded, dict) or not all(
                isinstance(key, str) for key in loaded
            ):
                raise ParseError(
                    "Markdown frontmatter must be a mapping",
                    details={"path": source.path},
                )
            return loaded, "".join(lines[index + 1 :])
    return {}, text


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _year(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return None


def _sections(text: str) -> tuple[ParsedSection, ...]:
    lines = text.splitlines(keepends=True)
    tokens = MarkdownIt("commonmark").parse(text)
    headings: list[tuple[int, int, int, str]] = []
    for index, token in enumerate(tokens):
        if token.type != "heading_open" or token.map is None:
            continue
        level = int(token.tag[1:])
        title = tokens[index + 1].content
        headings.append((token.map[0], token.map[1], level, title))

    if not headings:
        return (ParsedSection(text=text),) if text.strip() else ()

    sections: list[ParsedSection] = []
    preamble = "".join(lines[: headings[0][0]])
    if preamble.strip():
        sections.append(ParsedSection(text=preamble))
    path: list[tuple[int, str]] = []
    for index, (_, end, level, title) in enumerate(headings):
        while path and path[-1][0] >= level:
            path.pop()
        path.append((level, title))
        next_start = headings[index + 1][0] if index + 1 < len(headings) else len(lines)
        body = "".join(lines[end:next_start])
        if body.strip():
            sections.append(
                ParsedSection(text=body, heading_path=tuple(name for _, name in path))
            )
    return tuple(sections)


class MarkdownParser:
    """Parse Markdown structure while retaining original body text."""

    def __init__(self, corpus_root: Path) -> None:
        self.corpus_root = corpus_root

    def parse(self, source: Source) -> ParsedDocument:
        """Extract headings, known frontmatter fields, and link targets."""
        if source.file_type is not FileType.MARKDOWN:
            raise ParseError("source is not Markdown", details={"path": source.path})
        raw = read_text(self.corpus_root, source)
        frontmatter, body = _frontmatter(raw, source)
        enriched = replace(
            source,
            title=_string(frontmatter.get("title")),
            author=_string(frontmatter.get("author")),
            year=_year(frontmatter.get("year")),
        )
        metadata = tuple(
            ("wikilink", match.group(1).split("|", maxsplit=1)[0])
            for match in _WIKILINK.finditer(body)
        )
        return ParsedDocument(
            source=enriched, sections=_sections(body), metadata=metadata
        )
