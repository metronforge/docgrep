"""Pure checks of extracted text layers: soft hyphens and garbled pages.

Nothing here reads files or loads models; every threshold is a named constant
so that the rules can be reviewed and tested on their own. The garbled-page
rules are deliberately conservative: a page that reads as text in any script
must never be flagged, even if some garbled pages are missed.
"""

import re
import unicodedata

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

_CID = re.compile(r"\(cid:\d+\)")
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


def is_garbled(text: str) -> bool:
    """Whether a page's extracted text is a garbled text layer, not prose."""
    non_whitespace = sum(1 for character in text if not character.isspace())
    if non_whitespace < MIN_NON_WHITESPACE:
        return False
    placeholders = sum(len(match.group()) for match in _CID.finditer(text))
    broken = placeholders + sum(
        1
        for character in _CID.sub(" ", text)
        if not character.isspace() and _broken(character)
    )
    if broken > MAX_BROKEN_SHARE * non_whitespace:
        return True
    long_tokens = sum(
        len(token)
        for token in text.split()
        if len(token) > LONG_TOKEN_LENGTH
        and "://" not in token
        and not token.startswith("www.")
    )
    if long_tokens > MAX_LONG_TOKEN_SHARE * non_whitespace:
        return True
    alnum = sum(1 for character in text if character.isalnum())
    return alnum < MIN_ALNUM_SHARE * non_whitespace
