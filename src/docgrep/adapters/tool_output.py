"""Tool and library output made fit to store in a failure message.

The output of an external tool or library often names the file it worked on
by its absolute path, sometimes as a percent-encoded ``file://`` URL, and can
be long or span lines. Before such text goes into an error message or error
details it is scrubbed: every spelling of the given paths becomes
``PATH_PLACEHOLDER``, whitespace runs become single spaces, and the result is
cut to ``MAX_TOOL_MESSAGE_CHARS``.
"""

import re
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import quote

# Upper bound, in characters, of a scrubbed message.
MAX_TOOL_MESSAGE_CHARS = 200
PATH_PLACEHOLDER = "<file>"

# Shorter spellings (such as "/") would replace unrelated text.
_MIN_PATH_CHARS = 2


def _spellings(path: str | Path) -> set[str]:
    plain = {str(path)}
    try:
        plain.add(str(Path(path).resolve()))
    except (OSError, RuntimeError):
        pass
    found = set(plain)
    for text in plain:
        encoded = quote(text, safe="/")
        found |= {encoded, f"file://{encoded}", f"file://localhost{encoded}"}
        found |= {f"file://{text}", f"file://localhost{text}"}
    return {text for text in found if len(text.rstrip("/")) >= _MIN_PATH_CHARS}


def scrub_tool_output(text: str, paths: Iterable[str | Path] = ()) -> str:
    """``text`` without the given paths, on one line, at most 200 characters.

    Longer spellings are replaced first, so the path of a file is replaced as
    a whole before the path of the directory that holds it. A spelling is not
    replaced inside a longer name (``/books`` in ``/bookshelf``).
    """
    spellings: set[str] = set()
    for path in paths:
        spellings |= _spellings(path)
    for spelling in sorted(spellings, key=lambda s: (-len(s), s)):
        text = re.sub(re.escape(spelling) + r"(?![\w-])", PATH_PLACEHOLDER, text)
    return " ".join(text.split())[:MAX_TOOL_MESSAGE_CHARS]
