"""Scrubbing tool and library output before it goes into a failure message."""

from pathlib import Path
from typing import Any

import pytest

from djvu_support import FakeRunner
from docgrep.adapters import parsers
from docgrep.adapters.djvulibre import CommandResult, DjvuError, run_tool
from docgrep.adapters.parsers import DjvuParser
from docgrep.adapters.tool_output import (
    MAX_TOOL_MESSAGE_CHARS,
    PATH_PLACEHOLDER,
    scrub_tool_output,
)
from docgrep.domain.errors import ParseError
from docgrep.domain.models import FileType, Source

FILE = "/srv/My Library/books/war and peace.djvu"
ROOT = "/srv/My Library"


@pytest.mark.parametrize(
    "spelling",
    [
        FILE,
        f"'{FILE}'",
        f'"{FILE}"',
        f"{FILE}:",
        "file://localhost/srv/My%20Library/books/war%20and%20peace.djvu",
        "file:///srv/My%20Library/books/war%20and%20peace.djvu",
    ],
)
def test_every_spelling_of_the_file_path_is_replaced(spelling: str) -> None:
    scrubbed = scrub_tool_output(f"*** Failed to open {spelling} now", [FILE, ROOT])

    assert "/srv" not in scrubbed and "My" not in scrubbed
    assert scrubbed.count(PATH_PLACEHOLDER) == 1


def test_all_occurrences_and_the_corpus_root_are_replaced() -> None:
    text = f"{FILE}: bad chunk\n*** in {FILE}\n*** cwd {ROOT}/other.djvu"

    scrubbed = scrub_tool_output(text, [FILE, ROOT])

    assert scrubbed == ("<file>: bad chunk *** in <file> *** cwd <file>/other.djvu")


def test_a_path_is_not_replaced_inside_a_longer_name() -> None:
    scrubbed = scrub_tool_output("/srv/books and /srv/bookshelf", ["/srv/books"])

    assert scrubbed == "<file> and /srv/bookshelf"


def test_whitespace_is_collapsed_and_the_length_is_capped() -> None:
    text = "  first\n\tsecond   line\r\n" + "x" * 1000

    scrubbed = scrub_tool_output(text)

    assert scrubbed.startswith("first second line x")
    assert len(scrubbed) == MAX_TOOL_MESSAGE_CHARS == 200


def test_text_without_paths_is_unchanged() -> None:
    assert scrub_tool_output("djvused printed no page count", [FILE, ROOT]) == (
        "djvused printed no page count"
    )


def test_run_tool_scrubs_the_absolute_path_argument_from_standard_error() -> None:
    stderr = (f"*** [1-11711] Failed to open '{FILE}': bad.\n" * 50).encode()
    runner = FakeRunner({Path(FILE).name: CommandResult(10, b"", stderr)})

    with pytest.raises(DjvuError) as raised:
        run_tool(runner, ["djvutxt", "--detail=page", FILE])

    message = str(raised.value)
    assert message.startswith("djvutxt exited with status 10: *** [1-11711] Failed")
    assert "/srv" not in message and "<file>" in message
    assert raised.value.reason == "djvu_invalid"
    assert len(message) <= len("djvutxt exited with status 10: ") + 200


def _parse_failure(tmp_path: Path, stderr: str) -> ParseError:
    root = tmp_path / "My Library"
    (root / "books").mkdir(parents=True)
    (root / "books" / "war and peace.djvu").write_bytes(b"AT&TFORM")
    runner: Any = FakeRunner(
        {"war and peace.djvu": CommandResult(10, b"", stderr.encode())}
    )
    source = Source(
        "books/war and peace.djvu", "books/war and peace.djvu", FileType.DJVU
    )
    with pytest.raises(ParseError) as raised:
        DjvuParser(root, runner).parse(source)
    return raised.value


def test_the_parser_message_names_no_absolute_path(tmp_path: Path) -> None:
    root = (tmp_path / "My Library").resolve()
    file = root / "books" / "war and peace.djvu"
    url = "file://localhost" + str(file).replace(" ", "%20")
    stderr = (
        f"*** Failed to open document '{url}'\n"
        f"{file}: IFFByteStream not ready\n"
        f'in "{file}" under {root}\n'
    )

    error = _parse_failure(tmp_path, stderr)

    message = str(error)
    assert str(tmp_path) not in message and "My%20Library" not in message
    assert message.startswith("DjVu cannot be read (djvu_invalid): djvused exited")
    assert message.count("<file>") == 4
    assert len(message) <= MAX_TOOL_MESSAGE_CHARS
    assert error.details == {
        "path": "books/war and peace.djvu",
        "reason": "djvu_invalid",
    }


@pytest.mark.parametrize(
    ("parser_name", "file_type", "name", "key"),
    [
        ("TextParser", FileType.TXT, "gone.txt", "reason"),
        ("MarkdownParser", FileType.MARKDOWN, "gone.md", "reason"),
        ("PdfParser", FileType.PDF, "gone.pdf", "reason"),
        ("EpubParser", FileType.EPUB, "gone.epub", "error"),
    ],
)
def test_library_errors_of_other_parsers_name_no_absolute_path(
    tmp_path: Path, parser_name: str, file_type: FileType, name: str, key: str
) -> None:
    root = tmp_path / "My Library"
    root.mkdir()
    parser = getattr(parsers, parser_name)(root)

    with pytest.raises(ParseError) as raised:
        parser.parse(Source(name, name, file_type))

    detail = str(raised.value.details[key])
    assert detail and str(tmp_path) not in detail
    assert raised.value.details["path"] == name
