"""Test sources spell invisible and look-alike characters as escapes."""

import unicodedata
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]

# Files allowed to keep such characters raw, each with a one-line reason.
ALLOWED: dict[str, str] = {}


def _suspicious(character: str) -> bool:
    """Invisible, zero-width or ASCII look-alike characters (visible letters of
    other scripts, such as Russian test data, are fine)."""
    if ord(character) < 128:
        return False
    if unicodedata.category(character) in (
        "Zs",
        "Zl",
        "Zp",
        "Cf",
        "Cc",
        "Co",
        "Mn",
        "Me",
    ):
        return True
    return "FULLWIDTH" in unicodedata.name(character, "")


def test_no_test_source_hides_characters() -> None:
    found = []
    for path in sorted(TESTS.rglob("*.py")):
        relative = path.relative_to(TESTS).as_posix()
        if relative in ALLOWED:
            continue
        for number, line in enumerate(path.read_text("utf-8").splitlines(), 1):
            hidden = sorted({f"U+{ord(c):04X}" for c in line if _suspicious(c)})
            if hidden:
                found.append(f"{relative}:{number}: {', '.join(hidden)}")

    assert found == [], "use escapes such as \\u00a0 instead"


def test_the_check_catches_hidden_characters() -> None:
    for character in ("\u00a0", "\u00ad", "\u200b", "\ufeff", "\uff41", "\u0301"):
        assert _suspicious(character)
    for character in ("a", "Ж", "é", "—"):
        assert not _suspicious(character)
