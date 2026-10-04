"""Pure rules about the state of an indexing run."""

from collections.abc import Mapping
from typing import Literal

from docgrep.domain.models import ManifestRunRecord
from docgrep.domain.protocols import ManifestReader

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


RESUMABLE_STATES: frozenset[RunState] = frozenset(
    {"aborted", "cancelled", "interrupted"}
)

# The configuration keys that decide what vectors and passages contain. A run
# may only be resumed when its recorded values equal the runtime ones; every
# other key (device, dtype, batch sizes, log level, host, port) is operational.
VECTOR_CONFIG_KEYS = (
    "embedding_model",
    "embedding_dimension",
    "chunk_size",
    "chunk_overlap",
    "tokenizer",
    "normalize_embeddings",
)


def config_drift(
    recorded: Mapping[str, str | int | float], runtime: Mapping[str, str | int | float]
) -> tuple[str, ...]:
    """The vector-affecting keys whose recorded and runtime values differ."""
    return tuple(
        key for key in VECTOR_CONFIG_KEYS if recorded.get(key) != runtime.get(key)
    )


def swap_already_happened(
    run: ManifestRunRecord, *, journal_rows: int, staging_published: bool
) -> bool:
    """Whether an unfinished run had in fact published its index.

    The alias swap is only reached after every enumerated source was journaled.
    So a running (interrupted) or aborted run whose recorded staging collection
    is already the published one died between the swap and the write of its
    ``swapped`` status, provided the journal covers the planned sources. A run
    without a recorded plan (older versions) cannot be checked against one; it
    is accepted when its journal is not empty, so that an empty journal is
    never committed over the sources of the index that was replaced.
    """
    if run.status not in ("running", "aborted") or not staging_published:
        return False
    if run.total_sources is None:
        return journal_rows > 0
    return journal_rows >= run.total_sources


def latest_build_run(manifest: ManifestReader) -> ManifestRunRecord | None:
    """The latest run of kind build: the only kind that can be resumed.

    Updates continue by running again (ADR-0013), so an update that stopped
    never makes a build resumable or hides one.
    """
    latest = manifest.latest_run()
    if latest is None or latest.kind == "build":
        return latest
    return manifest.latest_run_of_kind("build")
