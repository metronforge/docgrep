"""Document parsers."""

from docgrep.adapters.parsers.djvu import DjvuParser
from docgrep.adapters.parsers.epub import EpubParser
from docgrep.adapters.parsers.markdown import MarkdownParser
from docgrep.adapters.parsers.pdf import PdfParser
from docgrep.adapters.parsers.text import TextParser

__all__ = ["DjvuParser", "EpubParser", "MarkdownParser", "PdfParser", "TextParser"]
