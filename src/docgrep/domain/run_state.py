"""Pure rules about the state of an indexing run."""

from collections.abc import Mapping
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
