"""A one-line progress display on a terminal (standard library only).

The line is drawn from ``ProgressEvent`` values, redrawn in place with a
carriage return and padded to the terminal width, at most ten times a
second. The last event of a phase stays on screen as its own line. Log
records written meanwhile first clear the line, so they never share it.
"""

import logging
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TextIO

from docgrep.domain.progress import ProgressEvent

# The fewest seconds between two redraws.
MIN_REDRAW_SECONDS = 0.1
# Never narrower, so that the fixed part of a line always has room.
MIN_WIDTH = 20
_ELLIPSIS = "…"
_VERBS = {"index": "Indexing", "update": "Updating", "ocr": "OCR"}
_PACKAGE_LOGGER = "docgrep"


def format_eta(seconds: float) -> str:
    """``42s``, ``5m 03s`` or ``1h 12m``."""
    whole = max(int(round(seconds)), 0)
    if whole < 60:
        return f"{whole}s"
    minutes, rest = divmod(whole, 60)
    if minutes < 60:
        return f"{minutes}m {rest:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def shorten_middle(text: str, width: int) -> str:
    """``text`` cut in the middle with an ellipsis to at most ``width``
    characters, so that the beginning and the extension stay visible."""
    if len(text) <= width:
        return text
    if width <= 0:
        return ""
    if width == 1:
        return _ELLIPSIS
    kept = width - 1
    head = (kept + 1) // 2
    tail = kept - head
    return text[:head] + _ELLIPSIS + (text[-tail:] if tail else "")


def _verb(event: ProgressEvent) -> str:
    if event.operation == "reencode":
        return "Scanning" if event.phase == "scanning" else "Re-encoding"
    return _VERBS.get(event.operation, event.operation.capitalize())


def _rate(rate: float) -> str:
    return f"{rate:.0f}" if rate >= 10 else f"{rate:.1f}"


def format_line(event: ProgressEvent, width: int) -> str:
    """The status line of ``event``, at most ``width`` characters.

    The label is shortened in the middle to the room that is left; when even
    the fixed part is too long, it is cut at the end.
    """
    parts = [_verb(event)]
    if event.total:
        parts.append(f"{min(event.done * 100 // event.total, 100)}%")
        parts.append(f"{event.done:,}/{event.total:,} {event.unit}")
    else:
        parts.append(f"{event.done:,} {event.unit}")
    if event.passages is not None and event.unit != "passages":
        parts.append(f"{event.passages:,} passages")
    if event.failed:
        parts.append(f"{event.failed:,} failed")
    if event.rate:
        parts.append(f"{_rate(event.rate)} p/s")
    if event.eta_seconds is not None and not event.finished:
        parts.append(f"ETA {format_eta(event.eta_seconds)}")
    fixed = "  ".join(parts)
    width = max(width, MIN_WIDTH)
    room = width - len(fixed) - 2
    if not event.label or room < 1:
        return fixed if len(fixed) <= width else fixed[: width - 1] + _ELLIPSIS
    return fixed + "  " + shorten_middle(event.label, room)


def terminal_width() -> int:
    """The columns of the terminal on standard error, 80 when unknown."""
    try:
        return os.get_terminal_size(sys.stderr.fileno()).columns
    except (OSError, ValueError, AttributeError):
        return shutil.get_terminal_size((80, 24)).columns


class _ClearBeforeRecord(logging.Filter):
    def __init__(self, line: "ProgressLine") -> None:
        super().__init__()
        self._line = line

    def filter(self, record: logging.LogRecord) -> bool:
        self._line.clear()
        return True


class ProgressLine:
    """Draw ``ProgressEvent`` values as one line on ``stream``.

    Used as the observer of a run. ``width`` and ``clock`` are injected so
    that the drawing is deterministic under test. ``close`` clears a line
    that was not finished (a cancelled or failed run).
    """

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        width: Callable[[], int] = terminal_width,
        clock: Callable[[], float] = time.monotonic,
        min_interval: float = MIN_REDRAW_SECONDS,
    ) -> None:
        self._stream = stream
        self._width = width
        self._clock = clock
        self._min_interval = min_interval
        self._last_drawn: float | None = None
        self._visible = False

    def __call__(self, event: ProgressEvent) -> None:
        now = self._clock()
        if (
            not event.finished
            and self._last_drawn is not None
            and now - self._last_drawn < self._min_interval
        ):
            return
        columns = max(self._width(), MIN_WIDTH)
        # One column stays free, so that a full line never wraps.
        line = format_line(event, columns - 1).ljust(columns - 1)
        self._write("\r" + line + ("\n" if event.finished else ""))
        self._last_drawn = None if event.finished else now
        self._visible = not event.finished

    def clear(self) -> None:
        """Wipe the line, if one is on screen; the next event redraws it."""
        self._last_drawn = None
        if not self._visible:
            return
        columns = max(self._width(), MIN_WIDTH)
        self._write("\r" + " " * (columns - 1) + "\r")
        self._visible = False

    def close(self) -> None:
        """Leave the terminal clean: wipe a line that was never finished."""
        self.clear()

    @contextmanager
    def keep_logs_apart(self) -> Iterator[None]:
        """Clear the line before each log record of the package is written."""
        guard = _ClearBeforeRecord(self)
        handlers = logging.getLogger(_PACKAGE_LOGGER).handlers
        for handler in handlers:
            handler.addFilter(guard)
        try:
            yield
        finally:
            for handler in handlers:
                handler.removeFilter(guard)

    def _write(self, text: str) -> None:
        stream = self._stream if self._stream is not None else sys.stderr
        stream.write(text)
        stream.flush()
