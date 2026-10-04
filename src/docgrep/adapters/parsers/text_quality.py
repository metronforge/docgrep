"""Pure checks of extracted text layers: soft hyphens and garbled pages.

Nothing here reads files or loads models; every threshold is a named constant
so that the rules can be reviewed and tested on their own. A page is judged by
three rules. R1 (broken characters) is the only one that marks a page as
garbled, and it is deliberately conservative: a page that reads as text in any
script must never be flagged. R2 (long tokens) and R3 (few letters and digits)
are weak signals: tables of contents, code, keys and hashes, and regular
expressions trigger them on readable pages, so they never drop a page by
themselves and only count when nearly a whole document shows them.
"""

import re
import unicodedata
from dataclasses import dataclass

SOFT_HYPHEN = "\u00ad"
REPLACEMENT_CHARACTER = "\ufffd"

# A page is judged only when it has at least this many non-whitespace
# characters; shorter pages (headings, page numbers) are never garbled.
MIN_NON_WHITESPACE = 200
# Rule 1: replacement, private-use and control characters (other than newline
# and tab) and the characters of "(cid:N)" placeholders, which extractors emit
# for glyphs without a Unicode mapping, above this share of the non-whitespace
# characters.
MAX_BROKEN_SHARE = 0.10
# Rule 2: tokens longer than LONG_TOKEN_LENGTH characters (text whose spaces
# were lost), except URLs, above this share of the non-whitespace characters.
LONG_TOKEN_LENGTH = 40
MAX_LONG_TOKEN_SHARE = 0.40
# Rule 3: letters and digits of any script below this share of the
# non-whitespace characters (symbol soup).
MIN_ALNUM_SHARE = 0.40
# Dot leaders (four or more dots or ellipsis characters, with or without
# spaces between them) are ignored by R2 and R3, so that a table of contents
# whose leaders are glued to the titles is not read as long tokens or symbols.
# Only the metrics ignore them; the text itself is unchanged.
MIN_DOT_LEADER_RUN = 4

_CID = re.compile(r"\(cid:\d+\)")
_DOT_LEADER = re.compile(
    "[.\u2026](?:[ \t]*[.\u2026]){" + str(MIN_DOT_LEADER_RUN - 1) + ",}"
)
# A soft hyphen ending a line, with optional spaces before the line break.
_SOFT_HYPHEN_AT_LINE_END = re.compile("\u00ad[ ]*\r?\n")


def remove_soft_hyphens(text: str) -> tuple[str, int]:
    """Remove U+00AD and return the text with how many were removed.

    A soft hyphen that ends a line (optionally followed by spaces) joins the
    two halves of the word: the line break and those spaces go too. This is
    the only change made to extracted text; a visible hyphen is never touched.
    """
    count = text.count(SOFT_HYPHEN)
    if not count:
        return text, 0
    joined = _SOFT_HYPHEN_AT_LINE_END.sub("", text)
    return joined.replace(SOFT_HYPHEN, ""), count


def _broken(character: str) -> bool:
    if character == REPLACEMENT_CHARACTER:
        return True
    category = unicodedata.category(character)
    return category == "Co" or (category == "Cc" and character not in "\n\t")


@dataclass(frozen=True, slots=True)
class PageSignals:
    """The three rules on one page, with the metrics behind them.

    ``non_whitespace`` and ``broken_share`` are measured on the page as
    extracted; ``long_token_share`` and ``alnum_share`` on the page with dot
    leaders ignored. A rule only fires on a page with at least
    ``MIN_NON_WHITESPACE`` non-whitespace characters in the text it measures.
    """

    non_whitespace: int
    broken_share: float
    long_token_share: float
    alnum_share: float
    broken: bool
    long_tokens: bool
    low_alnum: bool

    @property
    def garbled(self) -> bool:
        """Whether the page is dropped: only R1 drops a page."""
        return self.broken

    @property
    def weak(self) -> bool:
        """Whether a weak signal (R2 or R3) fired."""
        return self.long_tokens or self.low_alnum

    @property
    def any_rule(self) -> bool:
        return self.broken or self.weak


def _non_whitespace(text: str) -> int:
    return sum(1 for character in text if not character.isspace())


def classify_page(text: str) -> PageSignals:
    """Evaluate R1, R2 and R3 on a page's extracted text."""
    non_whitespace = _non_whitespace(text)
    placeholders = sum(len(match.group()) for match in _CID.finditer(text))
    broken = placeholders + sum(
        1
        for character in _CID.sub(" ", text)
        if not character.isspace() and _broken(character)
    )
    broken_share = broken / non_whitespace if non_whitespace else 0.0

    without_leaders = _DOT_LEADER.sub(" ", text)
    measured = _non_whitespace(without_leaders)
    long_tokens = sum(
        len(token)
        for token in without_leaders.split()
        if len(token) > LONG_TOKEN_LENGTH
        and "://" not in token
        and not token.startswith("www.")
    )
    alnum = sum(1 for character in without_leaders if character.isalnum())
    long_share = long_tokens / measured if measured else 0.0
    alnum_share = alnum / measured if measured else 0.0
    judged = non_whitespace >= MIN_NON_WHITESPACE
    weak_judged = measured >= MIN_NON_WHITESPACE
    return PageSignals(
        non_whitespace=non_whitespace,
        broken_share=broken_share,
        long_token_share=long_share,
        alnum_share=alnum_share,
        broken=judged and broken > MAX_BROKEN_SHARE * non_whitespace,
        long_tokens=weak_judged and long_tokens > MAX_LONG_TOKEN_SHARE * measured,
        low_alnum=weak_judged and alnum < MIN_ALNUM_SHARE * measured,
    )


def is_garbled(text: str) -> bool:
    """Whether a page's text layer is garbled and the page is dropped (R1)."""
    return classify_page(text).garbled
