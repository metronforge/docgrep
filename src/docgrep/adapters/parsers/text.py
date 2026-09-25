"""Plain-text document parser."""

from pathlib import Path

from docgrep.adapters.parsers._common import read_text
from docgrep.domain.errors import ParseError
from docgrep.domain.models import FileType, ParsedDocument, ParsedSection, Source


class TextParser:
    """Parse a TXT source as one verbatim section."""

    def __init__(self, corpus_root: Path) -> None:
        self.corpus_root = corpus_root

    def parse(self, source: Source) -> ParsedDocument:
        """Read a TXT source and preserve its full decoded text."""
        if source.file_type is not FileType.TXT:
            raise ParseError("source is not TXT", details={"path": source.path})
        text = read_text(self.corpus_root, source)
        return ParsedDocument(source=source, sections=(ParsedSection(text=text),))
