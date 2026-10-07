"""The ocrmypdf and ddjvu adapters with fake runners, the process-group
termination rules with a fake process, and the local OCR files."""

import os
import shutil
import signal
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from docgrep.adapters.djvulibre import CommandResult, CommandTimeout, ToolMissing
from docgrep.adapters.ocrmypdf import (
    DdjvuConverter,
    LocalOcrFiles,
    OcrmypdfEngine,
    Stopped,
    SubprocessSupervisor,
    stop_process_group,
)
from docgrep.adapters.tool_output import PATH_PLACEHOLDER
from docgrep.domain.errors import OcrFailed, OcrToolsMissing, ValidationError
from docgrep.domain.protocols import StopCheck


class FakeProcess:
    """A process group that exits after ``exits_after`` of the given signal."""

    def __init__(self, exits_on: int | None) -> None:
        self.exits_on = exits_on
        self.signals: list[int] = []
        self.code: int | None = None

    def poll(self) -> int | None:
        return self.code

    def signal_group(self, number: int) -> None:
        self.signals.append(number)
        if number in (self.exits_on, signal.SIGKILL):
            self.code = -number


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_a_group_that_obeys_sigterm_is_not_killed() -> None:
    process = FakeProcess(exits_on=signal.SIGTERM)
    clock = Clock()

    stop_process_group(process, grace=10, clock=clock, sleep=clock.sleep)

    assert process.signals == [signal.SIGTERM]
    assert clock.now == 0


def test_a_group_that_ignores_sigterm_is_killed_after_the_grace_period() -> None:
    process = FakeProcess(exits_on=None)
    clock = Clock()

    stop_process_group(process, grace=10, clock=clock, sleep=clock.sleep)

    assert process.signals == [signal.SIGTERM, signal.SIGKILL]
    assert 10 <= clock.now < 10.2


def test_a_finished_process_gets_no_signal() -> None:
    process = FakeProcess(exits_on=None)
    process.code = 0

    stop_process_group(process, grace=10)

    assert process.signals == []


def _group_gone(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    return False


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX shell")
def test_a_stop_request_terminates_the_whole_process_group(tmp_path: Path) -> None:
    marker = tmp_path / "pid"
    supervisor = SubprocessSupervisor(poll_seconds=0.05, grace=2)
    script = f"sleep 30 & echo $$ > {marker}; wait"

    def stop() -> bool:
        return marker.exists() and marker.read_text().strip() != ""

    with pytest.raises(Stopped):
        supervisor.run(["sh", "-c", script], deadline=None, should_stop=stop)

    group = int(marker.read_text())
    for _ in range(50):
        if _group_gone(group):
            break
        time.sleep(0.05)
    assert _group_gone(group)


def test_a_deadline_stops_the_command_as_a_timeout() -> None:
    supervisor = SubprocessSupervisor(poll_seconds=0.05, grace=2)
    started = time.monotonic()

    with pytest.raises(CommandTimeout):
        supervisor.run(
            ["sleep", "30"], deadline=time.monotonic() + 0.2, should_stop=lambda: False
        )

    assert time.monotonic() - started < 5


def test_a_finished_command_returns_its_status_and_error_output() -> None:
    supervisor = SubprocessSupervisor(poll_seconds=0.05)

    result = supervisor.run(
        ["sh", "-c", "echo problem >&2; exit 3"],
        deadline=None,
        should_stop=lambda: False,
    )

    assert result.returncode == 3 and result.stderr.strip() == b"problem"


def test_a_missing_program_is_reported_before_starting() -> None:
    with pytest.raises(ToolMissing):
        SubprocessSupervisor().run(
            ["docgrep-no-such-tool"], deadline=None, should_stop=lambda: False
        )


class FakeSupervisor:
    def __init__(self, answer: CommandResult | Exception) -> None:
        self.answer = answer
        self.calls: list[list[str]] = []

    def run(
        self, args: Sequence[str], *, deadline: float | None, should_stop: StopCheck
    ) -> CommandResult:
        self.calls.append(list(args))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


class FakeRunner:
    def __init__(self, answers: dict[str, CommandResult | Exception]) -> None:
        self.answers = answers

    def run(self, args: Sequence[str]) -> CommandResult:
        answer = self.answers[args[0]]
        if isinstance(answer, Exception):
            raise answer
        return answer


def _engine(
    tmp_path: Path, answer: CommandResult | Exception = CommandResult(0, b"", b"")
) -> tuple[OcrmypdfEngine, FakeSupervisor]:
    supervisor = FakeSupervisor(answer)
    runner = FakeRunner(
        {
            "ocrmypdf": CommandResult(0, b"16.13.0+dfsg1\n", b""),
            "tesseract": CommandResult(
                0,
                b'List of available languages in "/usr/share/tessdata/" (4):\n'
                b"eng\nosd\nrus\nukr\n",
                b"",
            ),
        }
    )
    return OcrmypdfEngine(tmp_path, supervisor=supervisor, runner=runner), supervisor


def _ocr(
    engine: OcrmypdfEngine,
    *,
    force: bool = False,
    jobs: int | None = None,
    languages: str = "rus+eng",
) -> None:
    engine.run(
        "books/scan.pdf",
        "books/.scan.part",
        languages=languages,
        mode="force-ocr" if force else "skip-text",
        jobs=jobs,
        deadline=None,
        should_stop=lambda: False,
    )


def test_the_engine_is_called_with_an_argument_list_and_absolute_paths(
    tmp_path: Path,
) -> None:
    engine, supervisor = _engine(tmp_path)

    _ocr(engine)
    _ocr(engine, force=True, jobs=2, languages="eng")

    root = tmp_path.resolve()
    assert supervisor.calls == [
        [
            "ocrmypdf",
            "--language",
            "rus+eng",
            "--output-type",
            "pdf",
            "--rotate-pages",
            "--deskew",
            "--optimize",
            "1",
            "--skip-text",
            str(root / "books/scan.pdf"),
            str(root / "books/.scan.part"),
        ],
        [
            "ocrmypdf",
            "--language",
            "eng",
            "--output-type",
            "pdf",
            "--rotate-pages",
            "--deskew",
            "--optimize",
            "1",
            "--force-ocr",
            "--jobs",
            "2",
            str(root / "books/scan.pdf"),
            str(root / "books/.scan.part"),
        ],
    ]


def test_version_and_languages_come_from_the_tools(tmp_path: Path) -> None:
    engine, _ = _engine(tmp_path)

    assert engine.version() == "16.13.0+dfsg1"
    assert engine.languages() == ("eng", "osd", "rus", "ukr")


def test_missing_tools_name_the_install_command(tmp_path: Path) -> None:
    engine = OcrmypdfEngine(
        tmp_path,
        supervisor=FakeSupervisor(ToolMissing("ocrmypdf")),
        runner=FakeRunner(
            {"ocrmypdf": ToolMissing("ocrmypdf"), "tesseract": ToolMissing("t")}
        ),
    )

    with pytest.raises(OcrToolsMissing) as raised:
        engine.version()
    with pytest.raises(OcrToolsMissing):
        engine.languages()
    with pytest.raises(OcrToolsMissing):
        _ocr(engine)

    assert "apt install ocrmypdf tesseract-ocr" in str(raised.value)
    assert "tesseract-ocr-ukr" in str(raised.value.details["hint"])


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        (Stopped("ocrmypdf"), "cancelled"),
        (CommandTimeout("ocrmypdf"), "timeout"),
        (
            CommandResult(8, b"", b"EncryptedPdfError: input PDF is encrypted\n"),
            "encrypted",
        ),
        (
            CommandResult(15, b"", b"OSError: [Errno 28] No space left on device\n"),
            "disk_full",
        ),
        (CommandResult(2, b"", b"InputFileError: not a PDF\n"), "ocr_engine_error"),
    ],
)
def test_engine_failures_map_to_reasons(
    tmp_path: Path, answer: CommandResult | Exception, reason: str
) -> None:
    engine, _ = _engine(tmp_path, answer)

    with pytest.raises(OcrFailed) as raised:
        _ocr(engine)

    assert raised.value.reason == reason


def test_engine_error_messages_are_scrubbed_of_paths(tmp_path: Path) -> None:
    long_output = (
        f"processing {tmp_path.resolve()}/books/scan.pdf\n" + "x" * 1000 + "\n"
        f"PriorOcrFoundError: page already has text in {tmp_path.resolve()}"
        "/books/scan.pdf\n"
    )
    engine, _ = _engine(tmp_path, CommandResult(6, b"", long_output.encode()))

    with pytest.raises(OcrFailed) as raised:
        _ocr(engine)

    message = str(raised.value)
    assert str(tmp_path) not in message
    assert PATH_PLACEHOLDER in message and "status 6" in message
    assert len(message) < 300


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        (ToolMissing("ddjvu"), "djvu_tools_missing"),
        (Stopped("ddjvu"), "cancelled"),
        (CommandTimeout("ddjvu"), "timeout"),
        (CommandResult(1, b"", b"corrupted file"), "djvu_conversion"),
        (CommandResult(1, b"", b"write error: No space left on device"), "disk_full"),
    ],
)
def test_djvu_conversion_failures_map_to_reasons(
    tmp_path: Path, answer: CommandResult | Exception, reason: str
) -> None:
    converter = DdjvuConverter(tmp_path, supervisor=FakeSupervisor(answer))

    with pytest.raises(OcrFailed) as raised:
        converter.convert(
            "book.djvu", ".book.part", deadline=None, should_stop=lambda: False
        )

    assert raised.value.reason == reason


def test_djvu_conversion_uses_ddjvu_with_absolute_paths(tmp_path: Path) -> None:
    supervisor = FakeSupervisor(CommandResult(0, b"", b""))

    DdjvuConverter(tmp_path, supervisor=supervisor).convert(
        "a/book.djvu", "a/.book.part", deadline=None, should_stop=lambda: False
    )

    root = tmp_path.resolve()
    assert supervisor.calls == [
        ["ddjvu", "-format=pdf", str(root / "a/book.djvu"), str(root / "a/.book.part")]
    ]


def _files(tmp_path: Path) -> tuple[LocalOcrFiles, Path]:
    corpus = tmp_path / "corpus"
    (corpus / "books").mkdir(parents=True)
    (corpus / "books" / "scan.pdf").write_bytes(b"scan")
    (tmp_path / "secret.pdf").write_bytes(b"outside")
    return LocalOcrFiles(corpus), corpus


@pytest.mark.parametrize(
    "path",
    ["../secret.pdf", "books/../../secret.pdf", "/etc/hostname", "escape.pdf"],
)
def test_paths_outside_the_corpus_are_rejected(tmp_path: Path, path: str) -> None:
    files, corpus = _files(tmp_path)
    (corpus / "escape.pdf").symlink_to(tmp_path / "secret.pdf")

    with pytest.raises(ValidationError) as raised:
        files.locate(path)

    assert raised.value.details["reason"] == "outside_corpus"


def test_paths_inside_the_corpus_are_normalized(tmp_path: Path) -> None:
    files, corpus = _files(tmp_path)

    assert files.locate("books/./scan.pdf") == "books/scan.pdf"
    assert files.locate(str(corpus / "books" / "scan.pdf")) == "books/scan.pdf"
    with pytest.raises(ValidationError) as raised:
        files.locate("books/missing.pdf")
    assert raised.value.details["reason"] == "not_found"
    with pytest.raises(ValidationError):
        files.locate("books")


def test_publishing_never_overwrites_unless_asked(tmp_path: Path) -> None:
    files, corpus = _files(tmp_path)
    temporary = files.temporary("books/scan.ocr.pdf", "ocr")
    assert temporary.startswith("books/.scan.ocr.docgrep-ocr-")
    assert temporary.endswith(".part")
    (corpus / temporary).write_bytes(b"first")
    files.publish(temporary, "books/scan.ocr.pdf", replace=False)
    assert (corpus / "books" / "scan.ocr.pdf").read_bytes() == b"first"
    assert not (corpus / temporary).exists()

    second = files.temporary("books/scan.ocr.pdf", "ocr")
    (corpus / second).write_bytes(b"second")
    with pytest.raises(FileExistsError):
        files.publish(second, "books/scan.ocr.pdf", replace=False)
    assert (corpus / "books" / "scan.ocr.pdf").read_bytes() == b"first"

    files.publish(second, "books/scan.ocr.pdf", replace=True)
    assert (corpus / "books" / "scan.ocr.pdf").read_bytes() == b"second"
    files.remove(second)
    files.remove("books/scan.ocr.pdf")
    assert not files.exists("books/scan.ocr.pdf")
    assert files.free_bytes("books/scan.pdf") > 0
    assert files.size("books/scan.pdf") == 4
    assert len(files.content_hash("books/scan.pdf")) == 64
