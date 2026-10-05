"""The DjVu parser through a fake djvulibre runner (UC-3 step 6.3, UC-4)."""

import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from djvu_support import FakeRunner, detail_output, write_fake_djvu
from docgrep.adapters.chunking import TokenChunker, WhitespaceTokenizer
from docgrep.adapters.djvulibre import (
    CommandResult,
    CommandTimeout,
    DjvuError,
    OutputTooLarge,
    SubprocessRunner,
    ToolMissing,
    parse_page_texts,
)
from docgrep.adapters.parsers import DjvuParser
from docgrep.domain.errors import ParseError
from docgrep.domain.models import FileType, ParsedDocument, Source
from page_classes import control_characters
from pdf_support import prose

SHY = "\u00ad"


def _source(name: str = "book.djvu") -> Source:
    return Source(name, name, FileType.DJVU)


def _parse(tmp_path: Path, answer: object, name: str = "book.djvu") -> ParsedDocument:
    write_fake_djvu(tmp_path / name)
    return DjvuParser(tmp_path, FakeRunner({name: answer})).parse(_source(name))


def _reason(tmp_path: Path, answer: object) -> str:
    with pytest.raises(ParseError) as raised:
        _parse(tmp_path, answer)
    assert raised.value.category == "parse_error"
    assert raised.value.details["path"] == "book.djvu"
    return str(raised.value.details["reason"])


def _page(seed: int) -> str:
    return "\n".join(prose(seed)) + "\n"


def test_the_parser_declares_version_1() -> None:
    assert DjvuParser.version == "1"


def test_pages_keep_their_numbers_around_pages_without_text(tmp_path: Path) -> None:
    document = _parse(tmp_path, [_page(1), None, "  \n", _page(4)])

    assert [(s.page_number, s.text) for s in document.sections] == [
        (1, _page(1)),
        (4, _page(4)),
    ]
    assert not document.requires_ocr
    assert document.quality is not None
    assert (document.quality.pages_total, document.quality.pages_with_text) == (4, 2)
    assert document.source.title is None and document.source.author is None


def test_the_tools_are_called_with_an_argument_list_and_an_absolute_path(
    tmp_path: Path,
) -> None:
    write_fake_djvu(tmp_path / "dir" / "book.djvu")
    runner = FakeRunner({"book.djvu": [_page(1)]})

    DjvuParser(tmp_path, runner).parse(_source("dir/book.djvu"))

    absolute = str((tmp_path / "dir" / "book.djvu").resolve())
    assert runner.calls == [
        ["djvused", "-e", "n", absolute],
        ["djvutxt", "--detail=page", absolute],
    ]


def test_escaped_text_is_decoded() -> None:
    raw = (
        b'(page 0 0 1 1 "Quote \\" back\\\\slash\\ttab\\nline \\001 \\x41 '
        b'\\320\\226 \xd0\x96")\n()\n'
    )

    assert parse_page_texts(raw) == [
        'Quote " back\\slash\ttab\nline \x01 A Ж Ж',
        "",
    ]


@pytest.mark.parametrize("pages", [[None, None], ["  ", "\n"], []])
def test_a_file_without_text_requires_ocr_as_no_text_layer(
    tmp_path: Path, pages: list[str | None]
) -> None:
    document = _parse(tmp_path, pages)

    assert document.requires_ocr and document.sections == ()
    assert document.quality is not None
    assert document.quality.skip_reason == "no_text_layer"


def test_broken_pages_are_dropped_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    broken = "\n".join(control_characters(1))
    pages: list[str | None] = [_page(1), broken, _page(3)]

    with caplog.at_level(logging.WARNING):
        document = _parse(tmp_path, pages)

    assert [s.page_number for s in document.sections] == [1, 3]
    assert document.quality is not None and document.quality.dropped_pages == (2,)
    (record,) = [r for r in caplog.records if r.getMessage() == "djvu_pages_dropped"]
    assert (record.__dict__["dropped"], record.__dict__["total"]) == (1, 3)
    assert "marker" not in caplog.text


def test_a_garbled_text_layer_requires_ocr(tmp_path: Path) -> None:
    broken = "\n".join(control_characters(1))
    document = _parse(tmp_path, [broken, broken, _page(3)])

    assert document.requires_ocr and document.sections == ()
    assert document.quality is not None
    assert document.quality.skip_reason == "garbled_text_layer"


def test_soft_hyphens_join_words(tmp_path: Path) -> None:
    document = _parse(tmp_path, [f"the hyphen{SHY}\nation of pere{SHY}nos\n"])

    assert document.sections[0].text == "the hyphenation of perenos\n"
    assert document.quality is not None
    assert document.quality.soft_hyphens_removed == 2


def test_missing_tools_fail_the_source_with_an_install_hint(tmp_path: Path) -> None:
    write_fake_djvu(tmp_path / "book.djvu")
    parser = DjvuParser(tmp_path, FakeRunner({"book.djvu": ToolMissing("djvused")}))

    with pytest.raises(ParseError) as raised:
        parser.parse(_source())

    assert raised.value.details["reason"] == "djvu_tools_missing"
    assert "djvulibre-bin" in str(raised.value)


@pytest.mark.parametrize(
    "answer",
    [
        CommandResult(10, b"", b"Failed to open document"),
        CommandResult(0, b"many\n", b""),
    ],
)
def test_tool_failures_are_invalid(tmp_path: Path, answer: CommandResult) -> None:
    assert _reason(tmp_path, answer) == "djvu_invalid"


class TextOnly(FakeRunner):
    """Correct page count, given djvutxt output."""

    def __init__(self, pages: int, output: bytes) -> None:
        super().__init__({})
        self.pages = pages
        self.output = output

    def run(self, args: Sequence[str]) -> CommandResult:
        if args[0] == "djvused":
            return CommandResult(0, f"{self.pages}\n".encode(), b"")
        return CommandResult(0, self.output, b"")


@pytest.mark.parametrize(
    ("pages", "output"),
    [
        (2, b'(page 0 0 1 1 "one")\n'),  # fewer pages than counted
        (1, b'(page 0 0 1 1 "unterminated)\n'),
        (1, b'(page 0 0 1 1 "x")\n)'),
        (1, b'(page 0 0 1 1 "bad \\q escape")\n'),
        (1, b"(page 0 0 1 1 (line"),
    ],
)
def test_unparsable_output_is_invalid(
    tmp_path: Path, pages: int, output: bytes
) -> None:
    write_fake_djvu(tmp_path / "book.djvu")

    with pytest.raises(ParseError) as raised:
        DjvuParser(tmp_path, TextOnly(pages, output)).parse(_source())

    assert raised.value.details["reason"] == "djvu_invalid"


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (CommandTimeout("djvutxt"), "djvu_timeout"),
        (OutputTooLarge("djvutxt"), "djvu_output_too_large"),
    ],
)
def test_timeouts_and_oversized_output(
    tmp_path: Path, error: Exception, reason: str
) -> None:
    assert _reason(tmp_path, error) == reason


def test_the_same_file_gives_the_same_passages(tmp_path: Path) -> None:
    pages: list[str | None] = [_page(1), None, f"split hyphen{SHY}\nation\n" + _page(3)]
    chunker = TokenChunker(WhitespaceTokenizer(), chunk_size=12, chunk_overlap=3)

    first = _parse(tmp_path, pages)
    second = _parse(tmp_path, pages)

    assert first == second
    passages = chunker.chunk(first)
    assert passages == chunker.chunk(second) and len(passages) > 2
    assert {p.page_number for p in passages} == {1, 3}


def test_a_wrong_file_type_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ParseError):
        DjvuParser(tmp_path, FakeRunner({})).parse(
            Source("a.pdf", "a.pdf", FileType.PDF)
        )


def test_the_detail_output_helper_matches_the_decoder() -> None:
    pages = ['a "b" \\ c\nd\t\x01', None, "Ж"]

    assert parse_page_texts(detail_output(pages)) == ['a "b" \\ c\nd\t\x01', "", "Ж"]


# The real subprocess runner, exercised with the Python interpreter so that no
# DjVu tool is needed.


def test_the_runner_reports_a_missing_program() -> None:
    with pytest.raises(ToolMissing):
        SubprocessRunner().run(["docgrep-no-such-program-for-tests", "x"])


def test_the_runner_captures_output_and_exit_status() -> None:
    result = SubprocessRunner().run(
        [sys.executable, "-c", "import sys; print('out'); sys.exit(3)"]
    )

    assert (result.returncode, result.stdout) == (3, b"out\n")


def test_the_runner_stops_a_slow_program() -> None:
    with pytest.raises(CommandTimeout):
        SubprocessRunner(timeout=0.5).run(
            [sys.executable, "-c", "import time; time.sleep(30)"]
        )


def test_the_runner_caps_the_output() -> None:
    with pytest.raises(OutputTooLarge):
        SubprocessRunner(max_output=1024 * 1024).run(
            [sys.executable, "-c", "import sys; sys.stdout.write('x' * 3_000_000)"]
        )


def test_djvu_errors_carry_a_reason() -> None:
    assert DjvuError("djvu_invalid", "message").reason == "djvu_invalid"
