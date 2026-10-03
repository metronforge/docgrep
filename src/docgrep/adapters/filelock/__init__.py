"""Exclusive index write lock backed by an advisory file lock."""

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

from docgrep.domain.errors import IndexLockHeld, StoreUnavailable


class FileIndexLock:
    """Hold a non-blocking ``flock`` on a file for the duration of a build."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _acquire(self) -> IO[bytes]:
        handle: IO[bytes] | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = self.path.open("a+b")
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            if handle is not None:
                handle.close()
            raise IndexLockHeld(
                "An index build is already running", details={"path": str(self.path)}
            ) from error
        except OSError as error:
            if handle is not None:
                handle.close()
            raise StoreUnavailable(
                "Cannot access the index lock", details={"path": str(self.path)}
            ) from error
        return handle

    @contextmanager
    def hold(self) -> Iterator[None]:
        """Acquire the lock, run the managed block, then release the lock.

        Only acquisition failures are translated; whatever the block raises
        propagates unchanged.
        """
        handle = self._acquire()
        try:
            yield
        finally:
            try:
                fcntl.flock(handle, fcntl.LOCK_UN)
            finally:
                handle.close()
