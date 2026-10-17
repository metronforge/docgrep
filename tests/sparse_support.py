"""Indexes with the sparse weighting of release 1.20.0, and the corpus shape
that made it fail (ADR-0025)."""

from pathlib import Path
from typing import Any


def as_tf_v1_build(build: Any) -> Any:
    """Make a ``BuildIndex`` write what release 1.20.0 wrote: raw term
    counts in every sparse vector, no final re-encode and no weighting keys
    in the completion marker (so the index reads as ``tf-v1``)."""
    build.sparse_store = None
    return build


# The shape of GitHub issue #20: many long passages repeat one query term
# 20 to 40 times, one short passage holds both terms once.
SHAPE_QUERY = "context engineering"
SHAPE_SHORT = "short.txt"
_FILLER = (
    "goroutine channel deadline cancel value tree request server handler "
    "stream buffer select timer mutex queue worker"
).split()


def write_issue_shape(corpus: Path) -> None:
    """One file per passage (each at most 50 tokens)."""
    corpus.mkdir(parents=True, exist_ok=True)

    def filler(count: int, seed: int) -> list[str]:
        return [_FILLER[(seed + step * 7) % len(_FILLER)] for step in range(count)]

    for number in range(24):
        repeats = 20 + number % 21
        words = ["context"] * repeats + filler(50 - repeats, number)
        (corpus / f"go{number:02d}.txt").write_text(" ".join(words), "utf-8")
    for number in range(6):
        repeats = 20 + (number * 4) % 21
        words = ["engineering"] * repeats + filler(50 - repeats, number)
        (corpus / f"quality{number}.txt").write_text(" ".join(words), "utf-8")
    for number in range(100):
        words = filler(50, number) + [f"topic{number}"]
        (corpus / f"other{number:03d}.txt").write_text(" ".join(words), "utf-8")
    (corpus / SHAPE_SHORT).write_text(
        "notes on context engineering for the test platform", "utf-8"
    )
