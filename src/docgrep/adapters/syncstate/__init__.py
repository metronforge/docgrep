"""The state of automatic sync, shared with other processes through a file.

`docgrep serve` runs sync and keeps its state in memory; `docgrep status` is
another process, so the server writes the status it reports to a small JSON
file in the data directory after every cycle (ADR-0029).
"""

import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from docgrep.domain.sync import SyncStatus

_LOG = logging.getLogger(__name__)

SYNC_STATE_FILE = "sync-status.json"


def process_alive(pid: int) -> bool:
    """Whether a process with this id exists (signal 0 sends nothing)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class FileSyncState:
    """Write and read the sync status of the server of one data directory."""

    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / SYNC_STATE_FILE

    def write(self, status: SyncStatus) -> None:
        """Replace the file atomically; failures are logged, never raised,
        so a full disk cannot stop sync."""
        body = {
            "pid": os.getpid(),
            "written_at": datetime.now(UTC).isoformat(),
            "status": status.to_dict(paths=True),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle, name = tempfile.mkstemp(dir=self.path.parent, prefix=".sync-")
            try:
                with os.fdopen(handle, "w", encoding="utf-8") as stream:
                    json.dump(body, stream)
                os.replace(name, self.path)
            except BaseException:
                Path(name).unlink(missing_ok=True)
                raise
        except OSError:
            _LOG.warning("sync_state_not_written", exc_info=True)

    def read(self) -> SyncStatus | None:
        """The status the server last wrote, with ``running`` true only
        while that server process is alive; None without a readable file."""
        try:
            body = json.loads(self.path.read_text(encoding="utf-8"))
            status = SyncStatus.from_dict(body["status"])
            alive = process_alive(int(body["pid"]))
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if status.running and not alive:
            return SyncStatus.from_dict(
                {**status.to_dict(paths=True), "running": False}
            )
        return status
