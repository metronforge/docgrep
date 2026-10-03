"""Turn the first Ctrl-C or SIGTERM during a build into a graceful cancel."""

import signal
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import FrameType
from typing import Any

Handler = Any  # what signal.signal accepts and returns: a callable or a SIG_* value
Register = Callable[[int, Handler], Handler]

_SIGNALS = (signal.SIGINT, signal.SIGTERM)


@contextmanager
def graceful_cancel(
    request_cancel: Callable[[], None],
    notify: Callable[[str], None],
    *,
    register: Register = signal.signal,
    raise_signal: Callable[[int], None] = signal.raise_signal,
) -> Iterator[None]:
    """Route SIGINT and SIGTERM to ``request_cancel`` for the managed block.

    The first signal requests the cancel and says so; the second restores the
    previous handlers and delivers the signal again, so the default behaviour
    applies. The previous handlers are restored on exit either way.
    """
    previous: dict[int, Handler] = {}
    state = {"signals": 0}

    def restore() -> None:
        for number, handler in previous.items():
            register(number, handler)
        previous.clear()

    def handle(number: int, frame: FrameType | None) -> None:
        state["signals"] += 1
        if state["signals"] == 1:
            notify(
                "Cancel requested: the build stops after the current batch. "
                "Press Ctrl-C again to stop immediately."
            )
            request_cancel()
            return
        restore()
        raise_signal(number)

    for number in _SIGNALS:
        previous[number] = register(number, handle)
    try:
        yield
    finally:
        restore()
