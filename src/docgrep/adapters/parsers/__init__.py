"""Document parsers."""

from docgrep.adapters.parsers.markdown import MarkdownParser
from docgrep.adapters.parsers.pdf import PdfParser
from docgrep.adapters.parsers.text import TextParser

__all__ = ["MarkdownParser", "PdfParser", "TextParser"]
