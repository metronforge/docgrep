"""Generate scans for OCR tests: pages that are images of rendered text, with
no text layer or with a broken one, as PDF or DjVu; and fakes of the OCR
ports that work on an in-memory corpus.

The real tools are only needed by the tests that say so (``HAVE_OCR_TOOLS``,
``HAVE_DJVU_ENCODER``); everything else runs on the fakes.
"""

import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import pymupdf
from blake3 import blake3

from docgrep.domain.errors import InspectionFailed, OcrFailed, ValidationError
from docgrep.domain.models import FileType, OcrMode, ScannedFile
from docgrep.domain.ocr import CANCELLED
from docgrep.domain.protocols import StopCheck
from docgrep.domain.scans import PageSample
from pdf_support import pdf_bytes

HAVE_OCR_TOOLS = all(shutil.which(tool) for tool in ("ocrmypdf", "tesseract"))
HAVE_DJVU_ENCODER = all(shutil.which(tool) for tool in ("c44", "ddjvu", "djvused"))
# A font with Latin and Cyrillic letters, when the system has one.
FONTS = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
)
CYRILLIC_FONT = next((font for font in FONTS if font.is_file()), None)
_DPI = 200


def _page_image(lines: Sequence[str], font: Path | None) -> pymupdf.Pixmap:
    """A grayscale image of an A5 page showing the lines in large type."""
    document = pymupdf.open()
    page = document.new_page(width=420, height=595)
    options: dict[str, object] = {"fontsize": 20}
    if font is not None:
        options.update(fontfile=str(font), fontname="custom")
    page.insert_text((40, 80), "\n".join(lines), **options)
    pixmap = page.get_pixmap(dpi=_DPI, colorspace=pymupdf.csGRAY)
    document.close()
    return pixmap


def image_pdf_bytes(
    pages: Sequence[Sequence[str]], *, font: Path | None = None
) -> bytes:
    """A PDF whose pages are images of the given lines, without any text."""
    document = pymupdf.open()
    for lines in pages:
        page = document.new_page(width=420, height=595)
        page.insert_image(page.rect, stream=_page_image(lines, font).tobytes("png"))
    data = document.tobytes()
    document.close()
    return bytes(data)


def garbled_pdf_bytes(
    pages: Sequence[Sequence[str]], broken: Sequence[Sequence[str]]
) -> bytes:
    """Pages that show the given lines as an image over a text layer that
    extracts as ``broken`` (the image hides it, as in a bad conversion)."""
    document = pymupdf.open("pdf", pdf_bytes(broken))
    for page, lines in zip(document, pages, strict=True):
        page.insert_image(page.rect, stream=_page_image(lines, None).tobytes("png"))
    data = document.tobytes()
    document.close()
    return bytes(data)


def image_djvu(path: Path, lines: Sequence[str]) -> Path:
    """A one-page DjVu file showing the lines, without a text layer.

    Needs ``c44`` from the djvulibre tools.
    """
    work = path.parent / f".{path.stem}-image"
    work.mkdir(parents=True, exist_ok=True)
    image = work / "page.pgm"
    _page_image(lines, None).save(str(image))
    subprocess.run(
        ["c44", "-dpi", str(_DPI), str(image), str(path)],
        check=True,
        capture_output=True,
    )
    shutil.rmtree(work)
    return path


def pdf_text(path: Path) -> str:
    with pymupdf.open(path) as document:
        return "".join(page.get_text("text") for page in document)


# Fakes of the OCR ports -----------------------------------------------------

# Page text that the fake probe reports as broken (rule R1 would drop it).
BROKEN = "\x00broken"


@dataclass
class FakeDocument:
    """A corpus file: its bytes and, for a readable document, its pages.

    ``pages`` holds the text layer of each page (None for no text); a file
    without pages cannot be opened.
    """

    data: bytes
    pages: list[str | None] | None = None
    encrypted: bool = False


@dataclass
class FakeCorpus:
    """Corpus files in memory, by relative path; the OCR files port, the scan
    probe and the corpus scanner all read it."""

    files: dict[str, FakeDocument] = field(default_factory=dict)
    free: int = 10**12
    removed: list[str] = field(default_factory=list)
    temporaries: list[str] = field(default_factory=list)

    def add(
        self,
        path: str,
        pages: list[str | None] | None,
        *,
        data: bytes | None = None,
        encrypted: bool = False,
    ) -> None:
        self.files[path] = FakeDocument(
            data if data is not None else path.encode() + repr(pages).encode(),
            pages,
            encrypted,
        )

    # CorpusScanner
    def scan(self) -> tuple[ScannedFile, ...]:
        found = []
        for path in sorted(self.files):
            suffix = PurePosixPath(path).suffix.lower()
            kind = {".pdf": FileType.PDF, ".djvu": FileType.DJVU}.get(suffix)
            found.append(ScannedFile(path, len(self.files[path].data), suffix, kind))
        return tuple(found)

    # OcrFiles
    def locate(self, path: str) -> str:
        normal = str(PurePosixPath(path))
        if normal.startswith(("/", "..")) or "/../" in normal:
            raise ValidationError(
                "outside", details={"reason": "outside_corpus", "path": path}
            )
        if normal not in self.files:
            raise ValidationError(
                "not found", details={"reason": "not_found", "path": path}
            )
        return normal

    def exists(self, path: str) -> bool:
        return path in self.files

    def size(self, path: str) -> int:
        return len(self.files[path].data)

    def content_hash(self, path: str) -> str:
        if path not in self.files:
            raise FileNotFoundError(path)
        return blake3(self.files[path].data).hexdigest()

    def free_bytes(self, path: str) -> int:
        return self.free

    def temporary(self, near: str, label: str) -> str:
        name = f"{PurePosixPath(near).parent}/.{label}-{len(self.temporaries)}.part"
        self.temporaries.append(name.removeprefix("./"))
        return self.temporaries[-1]

    def publish(self, temporary: str, final: str, *, replace: bool) -> None:
        if final in self.files and not replace:
            raise FileExistsError(final)
        self.files[final] = self.files.pop(temporary)

    def remove(self, path: str) -> None:
        if path in self.files:
            del self.files[path]
            self.removed.append(path)

    # ScanProbe
    def _document(self, path: str) -> list[str | None]:
        document = self.files.get(path)
        if document is None or document.pages is None:
            raise InspectionFailed("cannot open", details={"reason": "corrupt_header"})
        if document.encrypted:
            raise InspectionFailed("encrypted", details={"reason": "encrypted"})
        return document.pages

    def page_count(self, path: str, file_type: FileType) -> int:
        return len(self._document(path))

    def sample(
        self, path: str, file_type: FileType, pages: Sequence[int]
    ) -> tuple[PageSample | None, ...]:
        texts = self._document(path)
        return tuple(
            PageSample(
                number,
                len("".join((texts[number - 1] or "").split())),
                (texts[number - 1] or "").startswith(BROKEN),
            )
            for number in pages
        )


NATIVE_PAGE = "word " * 200


@dataclass
class FakeEngine:
    """Writes an output with the input's pages, each with recognised text.

    ``failures`` maps an input path to an exception to raise instead;
    ``block`` makes the run wait until ``should_stop`` says so (a cancel).
    """

    corpus: FakeCorpus
    installed: tuple[str, ...] = ("eng", "osd", "rus", "ukr")
    present: bool = True
    failures: dict[str, Exception] = field(default_factory=dict)
    calls: list[dict[str, object]] = field(default_factory=list)
    block: bool = False
    pages_delta: int = 0
    on_run: Callable[[str], None] | None = None
    name: str = "fake-ocr"

    def version(self) -> str:
        self._require()
        return "1.0"

    def languages(self) -> tuple[str, ...]:
        self._require()
        return self.installed

    def _require(self) -> None:
        if not self.present:
            from docgrep.domain.errors import OcrToolsMissing

            raise OcrToolsMissing("ocrmypdf is not installed")

    def run(
        self,
        source: str,
        target: str,
        *,
        languages: str,
        mode: OcrMode,
        jobs: int | None,
        deadline: float | None,
        should_stop: StopCheck,
    ) -> None:
        self.calls.append(
            {
                "source": source,
                "target": target,
                "languages": languages,
                "mode": mode,
                "jobs": jobs,
                "deadline": deadline,
            }
        )
        if self.on_run is not None:
            self.on_run(source)
        # A partial output exists while the engine works.
        self.corpus.add(target, None, data=b"partial")
        if self.block:
            for _ in range(1000):
                if should_stop():
                    raise OcrFailed("stopped", details={"reason": CANCELLED})
            raise AssertionError("the engine was never stopped")
        original = self.corpus.files[source]
        failure = self.failures.get(source)
        if failure is not None:
            raise failure
        pages = original.pages or []
        count = len(pages) + self.pages_delta
        self.corpus.add(
            target, [NATIVE_PAGE] * count, data=b"ocr:" + original.data + mode.encode()
        )


@dataclass
class FakeConverter:
    """Converts a DjVu file into a PDF of the same pages, without text."""

    corpus: FakeCorpus
    failures: dict[str, Exception] = field(default_factory=dict)
    calls: list[tuple[str, str]] = field(default_factory=list)

    def convert(
        self,
        source: str,
        target: str,
        *,
        deadline: float | None,
        should_stop: StopCheck,
    ) -> None:
        self.calls.append((source, target))
        failure = self.failures.get(source)
        if failure is not None:
            self.corpus.add(target, None, data=b"partial")
            raise failure
        pages = self.corpus.files[source].pages or []
        self.corpus.add(
            target, [None] * len(pages), data=b"pdf:" + self.corpus.files[source].data
        )


@dataclass
class TextLayerEngine:
    """An engine on real files that adds a known text layer instead of OCR:
    each output page carries ``recognised page <n> of <file stem>`` lines.

    ``on_run`` is called before the output is written; when ``should_stop``
    then says so, the run ends as cancelled with a partial output left.
    """

    corpus_root: Path
    on_run: Callable[[str], None] | None = None
    calls: list[dict[str, object]] = field(default_factory=list)
    name: str = "text-layer-fake"

    def version(self) -> str:
        return "0"

    def languages(self) -> tuple[str, ...]:
        return ("eng", "rus", "ukr")

    def run(
        self,
        source: str,
        target: str,
        *,
        languages: str,
        mode: OcrMode,
        jobs: int | None,
        deadline: float | None,
        should_stop: StopCheck,
    ) -> None:
        self.calls.append({"source": source, "mode": mode, "jobs": jobs})
        output = self.corpus_root / target
        output.write_bytes(b"partial")
        if self.on_run is not None:
            self.on_run(source)
        if should_stop():
            raise OcrFailed("stopped", details={"reason": CANCELLED})
        stem = PurePosixPath(source).name.split(".")[0].lstrip(".")
        with pymupdf.open(self.corpus_root / source) as document:
            for number, page in enumerate(document, start=1):
                lines = [f"recognised page {number} of {stem}"] + [
                    f"line {line} recognised words of {stem}" for line in range(40)
                ]
                page.insert_text((40, 40), "\n".join(lines), fontsize=6)
            document.save(output)


@dataclass
class CopyConverter:
    """A DjVu converter on real files that writes a blank PDF page per page."""

    corpus_root: Path
    pages: int = 1

    def convert(
        self,
        source: str,
        target: str,
        *,
        deadline: float | None,
        should_stop: StopCheck,
    ) -> None:
        document = pymupdf.open()
        for _ in range(self.pages):
            document.new_page()
        document.save(self.corpus_root / target)
        document.close()
