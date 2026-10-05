"""DjVu text layers read through the djvulibre command-line tools.

The tools run as subprocesses with an argument list (never a shell), an
absolute file path, a timeout and a cap on the captured output. They are an
optional system dependency: when they are missing, only DjVu sources fail,
and they can be retried after installing djvulibre-bin.

Page text comes from ``djvutxt --detail=page``, which prints one
S-expression per page in page order, ``()`` for a page without text. Plain
``djvutxt`` output cannot be used: it ends only the pages that have text with
a form feed, so page numbers after a page without text would shift.
"""

import shutil
import subprocess
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Protocol

# Seconds one tool invocation may take before it is stopped.
COMMAND_TIMEOUT_SECONDS = 300
# Upper bound for the captured standard output of one invocation, like the
# EPUB member cap.
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
# Upper bound for captured standard error; only used in error details.
MAX_ERROR_BYTES = 64 * 1024
_READ_BLOCK = 1024 * 1024

DJVUTXT = "djvutxt"
DJVUSED = "djvused"

# Reason codes of DjVu failures (ParseError details.reason).
TOOLS_MISSING = "djvu_tools_missing"
INVALID = "djvu_invalid"
TIMEOUT = "djvu_timeout"
OUTPUT_TOO_LARGE = "djvu_output_too_large"
INSTALL_HINT = "install the djvulibre command-line tools (Debian: djvulibre-bin)"


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: bytes
    stderr: bytes


class RunnerError(Exception):
    """A command could not be run to completion."""


class ToolMissing(RunnerError):
    """The program is not on PATH."""


class CommandTimeout(RunnerError):
    """The program ran longer than the timeout and was stopped."""


class OutputTooLarge(RunnerError):
    """The program wrote more than the output cap and was stopped."""


class CommandRunner(Protocol):
    """Run a program with arguments and capture its output.

    Raises ``ToolMissing``, ``CommandTimeout`` or ``OutputTooLarge``; a
    nonzero exit status is returned, not raised.
    """

    def run(self, args: Sequence[str]) -> CommandResult: ...


def _read_capped(stream: IO[bytes], cap: int) -> tuple[bytes, bool]:
    chunks: list[bytes] = []
    size = 0
    while block := stream.read(_READ_BLOCK):
        size += len(block)
        if size > cap:
            return b"".join(chunks), True
        chunks.append(block)
    return b"".join(chunks), False


class SubprocessRunner:
    """Run the tools as child processes; nothing is written to disk."""

    def __init__(
        self,
        *,
        timeout: float = COMMAND_TIMEOUT_SECONDS,
        max_output: int = MAX_OUTPUT_BYTES,
    ) -> None:
        self.timeout = timeout
        self.max_output = max_output

    def run(self, args: Sequence[str]) -> CommandResult:
        if not args or shutil.which(args[0]) is None:
            raise ToolMissing(args[0] if args else "")
        process = subprocess.Popen(
            list(args),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None and process.stderr is not None
        timed_out = threading.Event()

        def stop() -> None:
            timed_out.set()
            process.kill()

        errors: list[bytes] = []
        stderr = process.stderr
        reader = threading.Thread(
            target=lambda: errors.append(_read_capped(stderr, MAX_ERROR_BYTES)[0]),
            daemon=True,
        )
        timer = threading.Timer(self.timeout, stop)
        timer.start()
        reader.start()
        try:
            stdout, too_large = _read_capped(process.stdout, self.max_output)
            if too_large:
                process.kill()
            process.wait()
            reader.join(timeout=5)
        finally:
            timer.cancel()
            process.stdout.close()
        if timed_out.is_set():
            raise CommandTimeout(args[0])
        if too_large:
            raise OutputTooLarge(args[0])
        return CommandResult(process.returncode, stdout, errors[0] if errors else b"")


class DjvuError(Exception):
    """A DjVu file could not be read; ``reason`` is one of the codes above."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _run(runner: CommandRunner, args: Sequence[str]) -> bytes:
    try:
        result = runner.run(args)
    except ToolMissing as error:
        raise DjvuError(
            TOOLS_MISSING, f"{args[0]} is not installed; {INSTALL_HINT}"
        ) from error
    except CommandTimeout as error:
        raise DjvuError(
            TIMEOUT, f"{args[0]} took longer than {COMMAND_TIMEOUT_SECONDS} s"
        ) from error
    except OutputTooLarge as error:
        raise DjvuError(
            OUTPUT_TOO_LARGE, f"{args[0]} wrote more than {MAX_OUTPUT_BYTES} bytes"
        ) from error
    if result.returncode != 0:
        detail = " ".join(result.stderr.decode("utf-8", "replace").split())[:200]
        raise DjvuError(
            INVALID, f"{args[0]} exited with status {result.returncode}: {detail}"
        )
    return result.stdout


def page_count(runner: CommandRunner, path: Path) -> int:
    """The number of pages, from ``djvused -e n``."""
    output = _run(runner, [DJVUSED, "-e", "n", str(path)])
    text = output.decode("ascii", "replace").strip()
    if not text.isdecimal():
        raise DjvuError(INVALID, "djvused printed no page count")
    return int(text)


_ESCAPES = {
    ord("a"): 7,
    ord("b"): 8,
    ord("t"): 9,
    ord("n"): 10,
    ord("v"): 11,
    ord("f"): 12,
    ord("r"): 13,
    ord('"'): ord('"'),
    ord("\\"): ord("\\"),
}
_OCTAL = b"01234567"
_HEX = b"0123456789abcdefABCDEF"


def _string(data: bytes, start: int) -> tuple[bytes, int]:
    """Decode the quoted string that starts after ``data[start - 1] == '"'``."""
    out = bytearray()
    index = start
    while index < len(data):
        byte = data[index]
        if byte == ord('"'):
            return bytes(out), index + 1
        if byte != ord("\\"):
            out.append(byte)
            index += 1
            continue
        if index + 1 >= len(data):
            break
        code = data[index + 1]
        if code in _ESCAPES:
            out.append(_ESCAPES[code])
            index += 2
        elif code in _OCTAL:
            end = index + 1
            while end < len(data) and end < index + 4 and data[end] in _OCTAL:
                end += 1
            out.append(int(data[index + 1 : end], 8) & 0xFF)
            index = end
        elif code == ord("x"):
            end = index + 2
            while end < len(data) and end < index + 4 and data[end] in _HEX:
                end += 1
            if end == index + 2:
                break
            out.append(int(data[index + 2 : end], 16))
            index = end
        elif code == ord("\n"):
            index += 2  # a line continuation
        else:
            raise DjvuError(INVALID, "djvutxt printed an unknown escape")
    raise DjvuError(INVALID, "djvutxt printed an unterminated string")


def parse_page_texts(data: bytes) -> list[str]:
    """Split ``djvutxt --detail=page`` output into the text of each page.

    Every top-level S-expression is one page; the strings inside it are its
    text, decoded as UTF-8 (invalid bytes become U+FFFD and count as broken
    characters for the page rules).
    """
    pages: list[str] = []
    depth = 0
    current = bytearray()
    index = 0
    while index < len(data):
        byte = data[index]
        if byte == ord('"'):
            if depth == 0:
                raise DjvuError(INVALID, "djvutxt printed text outside a page")
            text, index = _string(data, index + 1)
            current += text
            continue
        if byte == ord("("):
            depth += 1
        elif byte == ord(")"):
            if depth == 0:
                raise DjvuError(INVALID, "djvutxt printed unbalanced parentheses")
            depth -= 1
            if depth == 0:
                pages.append(current.decode("utf-8", "replace"))
                current = bytearray()
        index += 1
    if depth != 0:
        raise DjvuError(INVALID, "djvutxt output ended inside a page")
    return pages


def page_texts(runner: CommandRunner, path: Path) -> list[str]:
    """The text of every page, in page order, checked against the page count."""
    count = page_count(runner, path)
    pages = parse_page_texts(_run(runner, [DJVUTXT, "--detail=page", str(path)]))
    if len(pages) != count:
        raise DjvuError(
            INVALID,
            f"djvutxt printed {len(pages)} pages, djvused counted {count}",
        )
    return pages
