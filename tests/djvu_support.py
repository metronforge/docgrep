"""A fake djvulibre runner and helpers to generate DjVu files with the real
tools when they are installed."""

import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from docgrep.adapters.djvulibre import CommandResult

DJVU_TOOLS = ("cjb2", "djvm", "djvused", "djvutxt")
HAVE_DJVU_TOOLS = all(shutil.which(tool) for tool in DJVU_TOOLS)


def _escape(text: str) -> str:
    out = []
    for character in text:
        if character in '"\\':
            out.append("\\" + character)
        elif character == "\n":
            out.append("\\n")
        elif character == "\t":
            out.append("\\t")
        elif ord(character) < 0x20 or ord(character) == 0x7F:
            out.append(f"\\{ord(character):03o}")
        else:
            out.append(character)
    return "".join(out)


def detail_output(pages: Sequence[str | None]) -> bytes:
    """What ``djvutxt --detail=page`` prints: one S-expression per page."""
    lines = [
        "()" if text is None else f'(page 0 0 100 100 "{_escape(text)}")'
        for text in pages
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


class FakeRunner:
    """Answer djvused and djvutxt calls from prepared pages, per file name.

    ``files`` maps a file name to its pages (text, or None for no text); a
    value may instead be a CommandResult or an exception for both tools.
    """

    def __init__(self, files: Mapping[str, object]) -> None:
        self.files = dict(files)
        self.calls: list[list[str]] = []

    def run(self, args: Sequence[str]) -> CommandResult:
        self.calls.append(list(args))
        answer = self.files[Path(args[-1]).name]
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, CommandResult):
            return answer
        assert isinstance(answer, list)
        if args[0] == "djvused":
            return CommandResult(0, f"{len(answer)}\n".encode(), b"")
        return CommandResult(0, detail_output(answer), b"")


def write_fake_djvu(path: Path) -> Path:
    """A placeholder file: with a fake runner only its name matters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"AT&TFORM placeholder")
    return path


def make_djvu(path: Path, pages: Sequence[str | None]) -> Path:
    """A real DjVu file, one blank bitmap page per entry, with a hidden text
    layer on the pages that have text. Needs the djvulibre tools."""
    work = path.parent / f".{path.stem}-pages"
    work.mkdir(parents=True, exist_ok=True)
    bitmap = work / "page.pbm"
    width = height = 64
    rows = ["0" * width for _ in range(height)]
    bitmap.write_text(f"P1\n{width} {height}\n" + "\n".join(rows) + "\n")
    page_files = []
    for number in range(len(pages)):
        page = work / f"p{number + 1}.djvu"
        subprocess.run(["cjb2", str(bitmap), str(page)], check=True)
        page_files.append(str(page))
    subprocess.run(["djvm", "-c", str(path), *page_files], check=True)
    script = []
    for number, text in enumerate(pages, start=1):
        if text is None:
            continue
        layer = work / f"t{number}.txt"
        layer.write_text(
            f'(page 0 0 {width} {height} "{_escape(text)}")\n', encoding="utf-8"
        )
        script.append(f"select {number}; set-txt {layer}")
    if script:
        subprocess.run(
            ["djvused", str(path), "-e", "; ".join([*script, "save"])], check=True
        )
    shutil.rmtree(work)
    return path
