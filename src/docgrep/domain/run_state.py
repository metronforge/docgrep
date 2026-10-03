"""Pure rules about the state of an indexing run."""

from typing import Literal

from docgrep.domain.models import ManifestRunRecord

# What a reader sees: the stored status plus the one state that is derived.
RunState = Literal[
    "running", "interrupted", "swapped", "completed", "aborted", "cancelled"
]


def is_interrupted(run: ManifestRunRecord, *, lock_held: bool) -> bool:
    """Whether a run that claims to be running lost its process.

    A run stored as ``running`` while nobody holds the index lock was killed.
    The state is derived on every read and never stored.
    """
    return run.status == "running" and not lock_held


def run_state(run: ManifestRunRecord, *, lock_held: bool) -> RunState:
    """The stored status of a run, with an interrupted run reported as such."""
    if is_interrupted(run, lock_held=lock_held):
        return "interrupted"
    return run.status
