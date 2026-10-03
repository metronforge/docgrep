"""The file-backed index lock against the real filesystem."""

import fcntl
from pathlib import Path

import pytest

from docgrep.adapters.filelock import FileIndexLock
from docgrep.domain.errors import IndexLockHeld, StoreUnavailable
from docgrep.domain.protocols import IndexLock


def test_lock_is_exclusive_and_released_on_exit(tmp_path: Path) -> None:
    lock: IndexLock = FileIndexLock(tmp_path / "data" / "index.lock")
    other = FileIndexLock(tmp_path / "data" / "index.lock")

    with lock.hold():
        with pytest.raises(IndexLockHeld) as held, other.hold():
            pass
        assert held.value.category == "index_lock_held"
        assert held.value.details == {"path": str(tmp_path / "data" / "index.lock")}

    with other.hold():
        pass


def test_lock_is_released_when_the_block_raises_and_the_error_is_unchanged(
    tmp_path: Path,
) -> None:
    lock = FileIndexLock(tmp_path / "index.lock")
    original = OSError(28, "No space left on device")

    with pytest.raises(OSError) as raised, lock.hold():
        raise original

    assert raised.value is original
    with lock.hold():
        pass


def test_lock_held_by_another_open_file_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "index.lock"
    path.touch()
    with path.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(IndexLockHeld), FileIndexLock(path).hold():
            pass


def test_unusable_lock_path_is_a_store_error(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory", encoding="utf-8")

    with (
        pytest.raises(StoreUnavailable) as raised,
        FileIndexLock(blocker / "index.lock").hold(),
    ):
        pass

    assert raised.value.category == "store_unavailable"
    assert raised.value.details == {"path": str(blocker / "index.lock")}
