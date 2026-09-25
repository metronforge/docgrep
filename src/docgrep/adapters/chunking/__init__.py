"""Token-based, location-aware passage chunking."""

import re
from bisect import bisect_right

from docgrep.domain.models import ParsedDocument, Passage, make_passage_id
from docgrep.domain.protocols import TokenSpanTokenizer

_TOKEN = re.compile(r"\S+")
_PARAGRAPH_GAP = re.compile(r"\n\s*\n")


class WhitespaceTokenizer:
    """Count non-whitespace runs as tokens for deterministic local use."""

    def token_spans(self, text: str) -> tuple[tuple[int, int], ...]:
        """Return character boundaries for each non-whitespace run."""
        return tuple(match.span() for match in _TOKEN.finditer(text))


class TokenChunker:
    """Chunk sections independently, preferring paragraph ends when practical."""

    def __init__(
        self, tokenizer: TokenSpanTokenizer, chunk_size: int, chunk_overlap: int
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: ParsedDocument) -> tuple[Passage, ...]:
        """Return exact source-text slices with stable section-stream offsets."""
        passages: list[Passage] = []
        section_base = 0
        for section in document.sections:
            text = section.text
            spans = self.tokenizer.token_spans(text)
            if any(
                start < 0 or end > len(text) or start >= end for start, end in spans
            ) or any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
                raise ValueError("tokenizer returned invalid token spans")

            paragraph_ends = [
                index + 1
                for index in range(len(spans) - 1)
                if _PARAGRAPH_GAP.search(text[spans[index][1] : spans[index + 1][0]])
            ]
            start = 0
            while start < len(spans):
                end = min(start + self.chunk_size, len(spans))
                if end < len(spans):
                    boundary_index = bisect_right(paragraph_ends, end) - 1
                    if boundary_index >= 0:
                        boundary = paragraph_ends[boundary_index]
                        if (
                            boundary > start + self.chunk_overlap
                            and 2 * (boundary - start) >= self.chunk_size
                        ):
                            end = boundary
                first_char = spans[start][0]
                last_char = spans[end - 1][1]
                offset = section_base + first_char
                passages.append(
                    Passage(
                        passage_id=make_passage_id(document.source.source_id, offset),
                        source_id=document.source.source_id,
                        text=text[first_char:last_char],
                        chunk_offset=offset,
                        token_count=end - start,
                        page_number=section.page_number,
                        heading_path=section.heading_path,
                    )
                )
                if end == len(spans):
                    break
                start = end - self.chunk_overlap
            # The separator makes offsets unique even when adjacent sections abut.
            section_base += len(text) + 1
        return tuple(passages)
