"""Indexes with the sparse weighting of release 1.20.0 (ADR-0025)."""

from typing import Any


def as_tf_v1_build(build: Any) -> Any:
    """Make a ``BuildIndex`` write what release 1.20.0 wrote: raw term
    counts in every sparse vector, no final re-encode and no weighting keys
    in the completion marker (so the index reads as ``tf-v1``)."""
    build.sparse_store = None
    return build
