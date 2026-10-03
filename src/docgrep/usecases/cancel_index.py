"""Ask a running index build to stop, without touching the index."""

from dataclasses import dataclass
from typing import Literal

from docgrep.domain.protocols import IndexLock, Manifest

CancelOutcome = Literal[
    "requested", "already_requested", "interrupted", "nothing_running"
]


@dataclass(frozen=True, slots=True)
class CancelResult:
    """What a cancel request found and did."""

    outcome: CancelOutcome
    run_id: str | None = None
    requested_at: str | None = None


class CancelIndex:
    """Persist a cooperative cancel flag for the build that holds the lock.

    Takes no lock and writes only the flag on the latest run. Without a
    manifest nothing has ever been built, so nothing is running.
    """

    def __init__(self, *, lock: IndexLock, manifest: Manifest | None) -> None:
        self.lock = lock
        self.manifest = manifest

    def run(self) -> CancelResult:
        latest = None if self.manifest is None else self.manifest.latest_run()
        if self.manifest is None or latest is None:
            return CancelResult("nothing_running")
        held = self.lock.is_held()
        if not held:
            if latest.status == "running":
                return CancelResult("interrupted", latest.run_id)
            return CancelResult("nothing_running", latest.run_id)
        if latest.status not in ("running", "swapped"):
            return CancelResult("nothing_running", latest.run_id)
        if latest.cancel_requested_at is not None:
            return CancelResult(
                "already_requested", latest.run_id, latest.cancel_requested_at
            )
        requested_at = self.manifest.request_cancel(latest.run_id)
        return CancelResult("requested", latest.run_id, requested_at)
