"""Progress of long operations: arithmetic and the event a display receives."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath


def estimate_eta_seconds(
    *, done: int, total: int, elapsed_seconds: float, carried: int = 0
) -> float | None:
    """Seconds left at the pace so far, or None until it is computable.

    ``carried`` counts sources that were already done when this process started
    (a resumed run); they say nothing about the pace.
    """
    processed = done - carried
    if processed <= 0 or elapsed_seconds <= 0:
        return None
    return max(total - done, 0) * elapsed_seconds / processed


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """Where a long operation has got to, for a display.

    ``operation`` is ``index``, ``update``, ``reencode`` or ``ocr``; ``unit``
    names what ``done`` and ``total`` count (``sources``, ``passages``,
    ``pages`` or ``files``). ``total`` is None when it is not known.
    ``passages`` and ``rate`` (passages per second) are None where they mean
    nothing (OCR). ``label`` is the name of the item being worked on, the
    last component of its path, or None. ``phase`` is ``scanning`` or
    ``writing``; ``finished`` marks the last event of a phase.
    """

    operation: str
    unit: str
    done: int
    total: int | None
    passages: int | None = None
    failed: int = 0
    elapsed_seconds: float = 0.0
    rate: float | None = None
    eta_seconds: float | None = None
    label: str | None = None
    phase: str = "writing"
    finished: bool = False


ProgressObserver = Callable[[ProgressEvent], None]


@dataclass(frozen=True, slots=True)
class ProgressOptions:
    """How a run reports its progress besides the log: an observer called on
    every advance, and the level of the ``index_progress`` log records."""

    observer: ProgressObserver | None = None
    log_level: int = logging.INFO


def item_label(path: str) -> str:
    """The name of an item for a display: the last component of its path."""
    return PurePosixPath(path).name or path
