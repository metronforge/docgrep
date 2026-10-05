"""The page policy shared by the parsers of paged formats (PDF, DjVu).

Each page's extracted text loses its soft hyphens, then the page rules of
``text_quality`` decide: a garbled page (R1) is dropped, weak signals (R2, R3)
only count. The whole document is left to OCR when the text layer as a whole
is unusable (see the two shares below).
"""

from collections.abc import Iterable

from docgrep.adapters.parsers.text_quality import classify_page, remove_soft_hyphens
from docgrep.domain.models import GARBLED_TEXT_LAYER, ParsedSection, ParseQuality

# When at least this share of the pages that have text are garbled (R1), the
# text layer as a whole is unusable and the document is left to OCR.
GARBLED_DOCUMENT_SHARE = 0.5
# When at least this share of the pages that have text fire any page rule
# (R1, or the weak signals R2 and R3), the document is left to OCR as well:
# weak signals on nearly every page mean a broken layer, not a few tables of
# contents or pages of code.
ANY_RULE_DOCUMENT_SHARE = 0.8


def apply_page_policy(
    pages: Iterable[tuple[int, str]], pages_total: int
) -> tuple[tuple[ParsedSection, ...], ParseQuality]:
    """Kept page sections and the quality figures of a paged document.

    ``pages`` yields (1-based page number, extracted text). When the text
    layer is unusable, ``skip_reason`` is ``garbled_text_layer``; the caller
    then returns no sections and requires OCR.
    """
    kept: list[ParsedSection] = []
    with_text = 0
    dropped: list[int] = []
    any_rule = long_tokens = low_alnum = 0
    soft_hyphens = 0
    for page_number, raw in pages:
        text, removed = remove_soft_hyphens(raw)
        soft_hyphens += removed
        if not text.strip():
            continue
        with_text += 1
        signals = classify_page(text)
        any_rule += signals.any_rule
        long_tokens += signals.long_tokens
        low_alnum += signals.low_alnum
        if signals.garbled:
            dropped.append(page_number)
            continue
        kept.append(ParsedSection(text=text, page_number=page_number))
    garbled = len(dropped)
    unusable = (garbled > 0 and garbled >= GARBLED_DOCUMENT_SHARE * with_text) or (
        any_rule > 0 and any_rule >= ANY_RULE_DOCUMENT_SHARE * with_text
    )
    quality = ParseQuality(
        pages_total=pages_total,
        pages_with_text=with_text,
        pages_garbled=garbled,
        soft_hyphens_removed=soft_hyphens,
        skip_reason=GARBLED_TEXT_LAYER if unusable else None,
        pages_rule_broken=garbled,
        pages_rule_long_tokens=long_tokens,
        pages_rule_low_alnum=low_alnum,
        dropped_pages=tuple(dropped),
    )
    return tuple(kept), quality
