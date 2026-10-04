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

    def _acquire(self, operation: str) -> IO[bytes]:
        handle: IO[bytes] | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = self.path.open("a+b")
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            if handle is not None:
                handle.close()
            raise IndexLockHeld(
                f"Another index operation is running, so the {operation} cannot "
                "start; try again when it has finished",
                details={"path": str(self.path)},
            ) from error
        except OSError as error:
            if handle is not None:
                handle.close()
            raise StoreUnavailable(
                "Cannot access the index lock", details={"path": str(self.path)}
            ) from error
        return handle

    def is_held(self) -> bool:
        """Try the lock without waiting and release it at once.

        Creates nothing: a lock file that does not exist is not held.
        """
        try:
            handle = self.path.open("rb")
        except FileNotFoundError:
            return False
        except OSError as error:
            raise StoreUnavailable(
                "Cannot access the index lock", details={"path": str(self.path)}
            ) from error
        try:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            except OSError as error:
                raise StoreUnavailable(
                    "Cannot access the index lock", details={"path": str(self.path)}
                ) from error
            fcntl.flock(handle, fcntl.LOCK_UN)
            return False
        finally:
            handle.close()

    @contextmanager
    def hold(self, operation: str = "build") -> Iterator[None]:
        """Acquire the lock, run the managed block, then release the lock.

        Only acquisition failures are translated, naming ``operation`` as the
        one that cannot start; whatever the block raises propagates unchanged.
        """
        handle = self._acquire(operation)
        try:
            yield
        finally:
            try:
                fcntl.flock(handle, fcntl.LOCK_UN)
            finally:
                handle.close()
