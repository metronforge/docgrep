"""Pure rules of automatic sync: which names it never indexes, when a changed
file has settled, and the state it reports (ADR-0029)."""

from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any

# Defaults of DOCGREP_SYNC_INTERVAL_SECONDS, DOCGREP_SYNC_SETTLE_SECONDS and
# DOCGREP_SYNC_RECONCILE_HOURS.
DEFAULT_SYNC_INTERVAL_SECONDS = 120.0
DEFAULT_SYNC_SETTLE_SECONDS = 60.0
DEFAULT_SYNC_RECONCILE_HOURS = 24.0
# A change that has not settled after this long is reported (and watched on).
UNSETTLED_REPORT_SECONDS = 30 * 60
# A file or removal must be seen in this many consecutive scans.
SETTLE_SCANS = 2

# Names that sync never adds, changes or removes, on top of the root
# patterns: partial downloads, temporary files, lock and swap files.
IGNORED_NAMES = ("*.part", "*.crdownload", "*.tmp", "~$*", ".~lock.*#", ".#*", "*.swp")
# Folders it never enters: other tools' own data.
IGNORED_DIRS = (".obsidian", ".git", ".trash")


def is_ignored(path: str) -> bool:
    """Whether a POSIX path (stored or root-relative) is one sync leaves
    alone: its name matches ``IGNORED_NAMES`` or a folder in it is one of
    ``IGNORED_DIRS``."""
    *folders, name = path.split("/")
    return any(folder in IGNORED_DIRS for folder in folders) or any(
        fnmatchcase(name, pattern) for pattern in IGNORED_NAMES
    )


@dataclass(slots=True)
class Pending:
    """An added or modified file waiting to settle.

    ``scans`` counts the consecutive scans that saw the same size and
    modification time. ``forced`` marks a change found by a reconciliation,
    which settles without waiting for the modification time to age.
    """

    kind: str
    size: int
    mtime: float
    scans: int = 1
    first_seen: float = 0.0
    forced: bool = False
    reported: bool = False

    def observe(self, size: int, mtime: float) -> None:
        """Count one more scan that saw the file; a different size or time
        starts the wait again."""
        if (size, mtime) == (self.size, self.mtime):
            self.scans += 1
        else:
            self.size, self.mtime, self.scans, self.forced = size, mtime, 1, False

    def settled(self, wall_now: float, settle_seconds: float) -> bool:
        """Whether the file may be applied: seen unchanged in two scans and,
        unless a reconciliation found it, not modified for
        ``settle_seconds``."""
        if self.scans < SETTLE_SCANS:
            return False
        return self.forced or wall_now - self.mtime >= settle_seconds


@dataclass(frozen=True, slots=True)
class HeldRoot:
    """Removals of one root that the removal guard holds back."""

    root: str
    count: int
    paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LastApplied:
    """The latest run that sync applied."""

    run_id: str
    finished_at: str
    added: int = 0
    modified: int = 0
    removed: int = 0
    moved: int = 0
    copied: int = 0
    failed: int = 0
    cancelled: bool = False


@dataclass(frozen=True, slots=True)
class SyncStatus:
    """What sync reports about itself; paths only on request."""

    enabled: bool
    running: bool = False
    interval_seconds: float = DEFAULT_SYNC_INTERVAL_SECONDS
    settle_seconds: float = DEFAULT_SYNC_SETTLE_SECONDS
    last_scan_at: str | None = None
    last_applied: LastApplied | None = None
    pending_added: int = 0
    pending_modified: int = 0
    pending_removed: int = 0
    unsettled: int = 0
    unsettled_over_limit: int = 0
    held: tuple[HeldRoot, ...] = ()
    paused_roots: tuple[str, ...] = ()
    awaiting_first_update: tuple[str, ...] = ()
    idle_reason: str | None = None
    last_error: str | None = None
    applying: tuple[int, int] | None = None

    def to_dict(self, *, paths: bool = False) -> dict[str, Any]:
        """The ``sync`` section of the status bodies; ``paths`` adds a few
        stored paths of each held root (``docgrep status`` only)."""
        return {
            "enabled": self.enabled,
            "running": self.running,
            "interval_seconds": self.interval_seconds,
            "settle_seconds": self.settle_seconds,
            "last_scan_at": self.last_scan_at,
            "last_applied": None
            if self.last_applied is None
            else {
                "run_id": self.last_applied.run_id,
                "finished_at": self.last_applied.finished_at,
                "added": self.last_applied.added,
                "modified": self.last_applied.modified,
                "removed": self.last_applied.removed,
                "moved": self.last_applied.moved,
                "copied": self.last_applied.copied,
                "failed": self.last_applied.failed,
                "cancelled": self.last_applied.cancelled,
            },
            "pending": {
                "added": self.pending_added,
                "modified": self.pending_modified,
                "removed": self.pending_removed,
            },
            "unsettled": self.unsettled,
            "unsettled_over_limit": self.unsettled_over_limit,
            "held_removals": [
                {"root": held.root, "count": held.count}
                | ({"paths": list(held.paths)} if paths else {})
                for held in self.held
            ],
            "paused_roots": list(self.paused_roots),
            "awaiting_first_update": list(self.awaiting_first_update),
            "idle_reason": self.idle_reason,
            "last_error": self.last_error,
            "applying": None
            if self.applying is None
            else {"done": self.applying[0], "total": self.applying[1]},
        }

    @classmethod
    def from_dict(cls, body: Mapping[str, Any]) -> "SyncStatus":
        """The status written by ``to_dict`` (with paths)."""
        applied = body.get("last_applied")
        pending = body.get("pending") or {}
        applying = body.get("applying")
        return cls(
            enabled=bool(body["enabled"]),
            running=bool(body.get("running", False)),
            interval_seconds=float(body["interval_seconds"]),
            settle_seconds=float(body["settle_seconds"]),
            last_scan_at=body.get("last_scan_at"),
            last_applied=None if applied is None else LastApplied(**applied),
            pending_added=int(pending.get("added", 0)),
            pending_modified=int(pending.get("modified", 0)),
            pending_removed=int(pending.get("removed", 0)),
            unsettled=int(body.get("unsettled", 0)),
            unsettled_over_limit=int(body.get("unsettled_over_limit", 0)),
            held=tuple(
                HeldRoot(h["root"], int(h["count"]), tuple(h.get("paths", ())))
                for h in body.get("held_removals", ())
            ),
            paused_roots=tuple(body.get("paused_roots", ())),
            awaiting_first_update=tuple(body.get("awaiting_first_update", ())),
            idle_reason=body.get("idle_reason"),
            last_error=body.get("last_error"),
            applying=None
            if applying is None
            else (int(applying["done"]), int(applying["total"])),
        )
