"""OCR through the system ``ocrmypdf`` command and DjVu conversion through
``ddjvu``, run as supervised subprocesses (ADR-0017).

Each tool runs with an argument list (never a shell) and absolute paths, in
a process group of its own, so that a cancel or a timeout can stop it with
everything it started: SIGTERM first, SIGKILL after a grace period. Standard
error is captured up to a cap, logged at DEBUG and, scrubbed of paths, used
in failure messages. The tools are an optional system dependency: only the
OCR command needs them.
"""

import errno
import logging
import os
import secrets
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import IO, Protocol

from docgrep.adapters.corpus import as_roots, file_content_hash, source_file
from docgrep.adapters.djvulibre import (
    MAX_ERROR_BYTES,
    CommandResult,
    CommandRunner,
    CommandTimeout,
    RunnerError,
    SubprocessRunner,
    ToolMissing,
)
from docgrep.adapters.tool_output import scrub_tool_output
from docgrep.domain.errors import OcrFailed, OcrToolsMissing, ValidationError
from docgrep.domain.models import OcrMode
from docgrep.domain.ocr import (
    CANCELLED,
    DISK_FULL,
    DJVU_CONVERSION,
    DJVU_TOOLS_MISSING,
    ENCRYPTED,
    ENGINE_ERROR,
    INSTALL_COMMAND,
    TERMINATE_GRACE_SECONDS,
    TIMEOUT,
)
from docgrep.domain.protocols import StopCheck
from docgrep.domain.roots import ROOT_NOT_AVAILABLE, CorpusRoots

_LOG = logging.getLogger(__name__)

OCRMYPDF = "ocrmypdf"
TESSERACT = "tesseract"
DDJVU = "ddjvu"
# Seconds between two checks of the stop condition while a tool runs.
POLL_SECONDS = 0.5
# Seconds a version or language listing may take.
QUERY_TIMEOUT_SECONDS = 60
# ocrmypdf's exit status for an encrypted input.
_EXIT_ENCRYPTED = 8
_NO_SPACE = ("no space left on device", "disk quota exceeded")
DJVU_INSTALL_HINT = "install the djvulibre command-line tools (Debian: djvulibre-bin)"


class Stopped(RunnerError):
    """The program was stopped because the caller asked it to stop."""


class ProcessHandle(Protocol):
    """A running child process that leads its own process group."""

    def poll(self) -> int | None: ...

    def signal_group(self, number: int) -> None: ...


def stop_process_group(
    process: ProcessHandle,
    *,
    grace: float = TERMINATE_GRACE_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Terminate a process group: SIGTERM, then SIGKILL after ``grace``
    seconds if the leader is still running."""
    if process.poll() is not None:
        return
    process.signal_group(signal.SIGTERM)
    deadline = clock() + grace
    while process.poll() is None:
        if clock() >= deadline:
            _LOG.warning("ocr_process_killed", extra={"grace_seconds": grace})
            process.signal_group(signal.SIGKILL)
            while process.poll() is None:
                sleep(0.05)
            return
        sleep(0.05)


class _PopenHandle:
    def __init__(self, process: "subprocess.Popen[bytes]") -> None:
        self.process = process

    def poll(self) -> int | None:
        return self.process.poll()

    def signal_group(self, number: int) -> None:
        try:
            os.killpg(self.process.pid, number)
        except ProcessLookupError:
            pass


class SupervisedRunner(Protocol):
    """Run a program until it exits, a deadline passes or the caller asks it
    to stop. Raises ``ToolMissing``, ``CommandTimeout`` or ``Stopped``; a
    nonzero exit status is returned. Standard output is not captured."""

    def run(
        self, args: Sequence[str], *, deadline: float | None, should_stop: StopCheck
    ) -> CommandResult: ...


def _drain(stream: IO[bytes], sink: list[bytes]) -> None:
    size = 0
    while block := stream.read(64 * 1024):
        if size < MAX_ERROR_BYTES:
            sink.append(block[: MAX_ERROR_BYTES - size])
            size += len(block)


class SubprocessSupervisor:
    """Run tools as child processes in their own session (process group)."""

    def __init__(
        self,
        *,
        poll_seconds: float = POLL_SECONDS,
        grace: float = TERMINATE_GRACE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.poll_seconds = poll_seconds
        self.grace = grace
        self.clock = clock

    def run(
        self, args: Sequence[str], *, deadline: float | None, should_stop: StopCheck
    ) -> CommandResult:
        if not args or shutil.which(args[0]) is None:
            raise ToolMissing(args[0] if args else "")
        process = subprocess.Popen(
            list(args),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        assert process.stderr is not None
        captured: list[bytes] = []
        reader = threading.Thread(
            target=_drain, args=(process.stderr, captured), daemon=True
        )
        reader.start()
        handle = _PopenHandle(process)
        stop: type[RunnerError] | None = None
        try:
            while process.poll() is None:
                if should_stop():
                    stop = Stopped
                elif deadline is not None and self.clock() >= deadline:
                    stop = CommandTimeout
                if stop is not None:
                    stop_process_group(handle, grace=self.grace)
                    break
                try:
                    process.wait(timeout=self.poll_seconds)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            # An interrupt of this process stops the tool as well.
            stop_process_group(handle, grace=self.grace)
            raise
        finally:
            reader.join(timeout=5)
            process.stderr.close()
        if stop is not None:
            raise stop(args[0])
        return CommandResult(process.returncode, b"", b"".join(captured))


def _decode(data: bytes) -> str:
    return data.decode("utf-8", "replace")


def _log_output(tool: str, result: CommandResult) -> None:
    if result.stderr:
        _LOG.debug(
            "ocr_tool_output",
            extra={
                "tool": tool,
                "returncode": result.returncode,
                "output": _decode(result.stderr),
            },
        )


def _disk_full(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _NO_SPACE)


def _failed(reason: str, message: str) -> OcrFailed:
    return OcrFailed(message, details={"reason": reason})


def _absolute(
    roots: CorpusRoots, path: str, *, resolve: bool = True
) -> tuple[Path, Path]:
    """The file of a stored path and the folder of its root."""
    try:
        file, root = source_file(roots, path)
    except ValidationError as error:
        raise _failed(
            ROOT_NOT_AVAILABLE, "the root of the file is not available"
        ) from error
    return (file.resolve() if resolve else file), root


class OcrmypdfEngine:
    """The ``ocrmypdf`` command, with Tesseract for the text recognition."""

    name = OCRMYPDF

    def __init__(
        self,
        corpus_root: CorpusRoots | Path,
        *,
        supervisor: SupervisedRunner | None = None,
        runner: CommandRunner | None = None,
    ) -> None:
        self.roots = as_roots(corpus_root)
        self.supervisor: SupervisedRunner = supervisor or SubprocessSupervisor()
        self.runner: CommandRunner = runner or SubprocessRunner(
            timeout=QUERY_TIMEOUT_SECONDS
        )

    def _query(self, args: Sequence[str]) -> str:
        try:
            result = self.runner.run(args)
        except ToolMissing as error:
            raise OcrToolsMissing(
                f"{args[0]} is not installed; OCR needs ocrmypdf and Tesseract "
                f"(Debian: {INSTALL_COMMAND})",
                details={"tool": args[0], "hint": INSTALL_COMMAND},
            ) from error
        except RunnerError as error:
            raise OcrToolsMissing(
                f"{args[0]} did not answer", details={"tool": args[0]}
            ) from error
        if result.returncode != 0:
            raise OcrToolsMissing(
                f"{args[0]} exited with status {result.returncode}",
                details={"tool": args[0], "hint": INSTALL_COMMAND},
            )
        # Some versions print the language listing on standard error.
        return _decode(result.stdout) + "\n" + _decode(result.stderr)

    def version(self) -> str:
        return self._query([OCRMYPDF, "--version"]).split()[0]

    def languages(self) -> tuple[str, ...]:
        text = self._query([TESSERACT, "--list-langs"])
        return tuple(
            line.strip()
            for line in text.splitlines()
            if line.strip()
            and not line.rstrip().endswith(":")
            and " " not in line.strip()
        )

    def arguments(
        self,
        source: Path,
        target: Path,
        *,
        languages: str,
        mode: OcrMode,
        jobs: int | None,
    ) -> list[str]:
        args = [
            OCRMYPDF,
            "--language",
            languages,
            "--output-type",
            "pdf",
            "--rotate-pages",
            "--deskew",
            "--optimize",
            "1",
            "--skip-text" if mode == "skip-text" else "--force-ocr",
        ]
        if jobs is not None:
            args += ["--jobs", str(jobs)]
        return [*args, str(source), str(target)]

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
        absolute_source, root = _absolute(self.roots, source)
        absolute_target, _ = _absolute(self.roots, target)
        args = self.arguments(
            absolute_source, absolute_target, languages=languages, mode=mode, jobs=jobs
        )
        try:
            result = self.supervisor.run(
                args, deadline=deadline, should_stop=should_stop
            )
        except ToolMissing as error:
            raise OcrToolsMissing(
                f"ocrmypdf is not installed (Debian: {INSTALL_COMMAND})",
                details={"tool": OCRMYPDF, "hint": INSTALL_COMMAND},
            ) from error
        except Stopped as error:
            raise _failed(CANCELLED, "OCR was cancelled") from error
        except CommandTimeout as error:
            raise _failed(TIMEOUT, "OCR took longer than the time limit") from error
        _log_output(OCRMYPDF, result)
        if result.returncode == 0:
            return
        text = _decode(result.stderr)
        if _disk_full(text):
            raise _failed(DISK_FULL, "no space left on device")
        if result.returncode == _EXIT_ENCRYPTED:
            raise _failed(ENCRYPTED, "the PDF is encrypted")
        lines = [line for line in text.splitlines() if line.strip()]
        detail = scrub_tool_output(
            lines[-1] if lines else "",
            (absolute_source, absolute_target, root),
        )
        raise _failed(
            ENGINE_ERROR, f"ocrmypdf exited with status {result.returncode}: {detail}"
        )


class DdjvuConverter:
    """``ddjvu -format=pdf``: a DjVu file as a PDF of page images."""

    def __init__(
        self,
        corpus_root: CorpusRoots | Path,
        *,
        supervisor: SupervisedRunner | None = None,
    ) -> None:
        self.roots = as_roots(corpus_root)
        self.supervisor: SupervisedRunner = supervisor or SubprocessSupervisor()

    def convert(
        self,
        source: str,
        target: str,
        *,
        deadline: float | None,
        should_stop: StopCheck,
    ) -> None:
        absolute_source, root = _absolute(self.roots, source)
        absolute_target, _ = _absolute(self.roots, target)
        args = [DDJVU, "-format=pdf", str(absolute_source), str(absolute_target)]
        try:
            result = self.supervisor.run(
                args, deadline=deadline, should_stop=should_stop
            )
        except ToolMissing as error:
            raise _failed(
                DJVU_TOOLS_MISSING, f"ddjvu is not installed; {DJVU_INSTALL_HINT}"
            ) from error
        except Stopped as error:
            raise _failed(CANCELLED, "the DjVu conversion was cancelled") from error
        except CommandTimeout as error:
            raise _failed(
                TIMEOUT, "the DjVu conversion took longer than the time limit"
            ) from error
        _log_output(DDJVU, result)
        if result.returncode == 0:
            return
        text = _decode(result.stderr)
        if _disk_full(text):
            raise _failed(DISK_FULL, "no space left on device")
        detail = scrub_tool_output(text, (absolute_source, absolute_target, root))
        raise _failed(
            DJVU_CONVERSION, f"ddjvu exited with status {result.returncode}: {detail}"
        )


class LocalOcrFiles:
    """OCR inputs and outputs inside the corpus roots on the local disk.

    Paths are stored paths: an output written next to its original is in
    the same root and carries the same prefix.
    """

    def __init__(self, corpus_root: CorpusRoots | Path) -> None:
        self.roots = as_roots(corpus_root)

    def _absolute(self, path: str) -> Path:
        return _absolute(self.roots, path, resolve=False)[0]

    def locate(self, path: str) -> str:
        given = Path(path)
        stored: str | None
        if given.is_absolute():
            resolved = given.resolve()
            stored = None
            for root in self.roots:
                base = root.path.resolve()
                if resolved.is_relative_to(base) and resolved != base:
                    stored = root.stored_path(resolved.relative_to(base).as_posix())
                    break
        else:
            stored = self._inside(given.as_posix())
        if stored is None:
            raise ValidationError(
                "the path is outside the corpus root",
                details={"reason": "outside_corpus", "path": path},
            )
        if not self._absolute(stored).is_file():
            raise ValidationError(
                "the path is not an existing file",
                details={"reason": "not_found", "path": path},
            )
        return stored

    def _inside(self, path: str) -> str | None:
        """The stored path of a stored path given by the user, normalised,
        or None when it leaves its root."""
        root = self.roots.owner(path)
        if root is None:
            return None
        base = root.path.resolve()
        resolved = (base / root.relative_path(path)).resolve()
        if not resolved.is_relative_to(base) or resolved == base:
            return None
        return root.stored_path(resolved.relative_to(base).as_posix())

    def exists(self, path: str) -> bool:
        return self._absolute(path).is_file()

    def size(self, path: str) -> int:
        return self._absolute(path).stat().st_size

    def content_hash(self, path: str) -> str:
        return file_content_hash(self._absolute(path))

    def free_bytes(self, path: str) -> int:
        return shutil.disk_usage(self._absolute(path).parent).free

    def temporary(self, near: str, label: str) -> str:
        final = Path(near)
        while True:
            name = f".{final.stem}.docgrep-{label}-{secrets.token_hex(4)}.part"
            candidate = final.with_name(name)
            if not self._absolute(str(candidate)).exists():
                return candidate.as_posix()

    def publish(self, temporary: str, final: str, *, replace: bool) -> None:
        source = self._absolute(temporary)
        target = self._absolute(final)
        if replace:
            os.replace(source, target)
            return
        try:
            # A hard link never replaces an existing file.
            os.link(source, target)
        except FileExistsError:
            raise
        except OSError as error:
            if error.errno not in (errno.EPERM, errno.EXDEV, errno.ENOTSUP):
                raise
            # No hard links on this file system: check, then rename.
            if target.exists():
                raise FileExistsError(errno.EEXIST, "output exists") from error
            os.rename(source, target)
            return
        source.unlink()

    def remove(self, path: str) -> None:
        self._absolute(path).unlink(missing_ok=True)
