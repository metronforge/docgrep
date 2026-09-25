"""In-memory implementations of persistence and retrieval boundaries."""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Literal

from docgrep.domain.models import (
    ManifestRunRecord,
    ManifestSourceRecord,
    Passage,
    RetrievalCandidate,
    Source,
)


class InMemoryPassageStore:
    """A small deterministic hybrid retriever with test-only write helpers."""

    def __init__(self) -> None:
        self._items: dict[
            str, tuple[Passage, Source, tuple[float, ...], dict[int, float]]
        ] = {}
        self._completion: dict[str, str | int | float] | None = None

    def upsert(
        self,
        passage: Passage,
        source: Source,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float],
    ) -> None:
        self._items[passage.passage_id] = (
            passage,
            source,
            tuple(dense_vector),
            dict(sparse_vector),
        )

    def retrieve(
        self,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float],
        top_k: int,
    ) -> tuple[RetrievalCandidate, ...]:
        scored: list[tuple[float, RetrievalCandidate]] = []
        for passage, source, stored_dense, stored_sparse in self._items.values():
            dense_score = sum(
                a * b for a, b in zip(dense_vector, stored_dense, strict=True)
            )
            sparse_score = sum(
                weight * stored_sparse.get(index, 0.0)
                for index, weight in sparse_vector.items()
            )
            scored.append(
                (
                    dense_score + sparse_score,
                    RetrievalCandidate(passage, source, dense_score),
                )
            )
        scored.sort(
            key=lambda pair: (
                -pair[0],
                -pair[1].dense_score,
                pair[1].passage.passage_id,
            )
        )
        return tuple(candidate for _, candidate in scored[:top_k])

    def read_completion_metadata(self) -> Mapping[str, str | int | float] | None:
        return dict(self._completion) if self._completion is not None else None

    def write_completion_metadata(
        self, metadata: Mapping[str, str | int | float]
    ) -> None:
        self._completion = dict(metadata)


class InMemoryManifest:
    """Keep source and run records with the same read/write port surface."""

    def __init__(self) -> None:
        self._sources: dict[str, ManifestSourceRecord] = {}
        self._runs: list[ManifestRunRecord] = []

    def get_source(self, path: str) -> ManifestSourceRecord | None:
        return self._sources.get(path)

    def latest_run(self) -> ManifestRunRecord | None:
        return self._runs[-1] if self._runs else None

    def start_run(self, config: Mapping[str, str | int | float]) -> str:
        run_id = str(len(self._runs) + 1)
        self._runs.append(
            ManifestRunRecord(run_id, "running", tuple(sorted(config.items())))
        )
        return run_id

    def record_source(
        self, source: Source, status: str, error: str | None = None
    ) -> None:
        self._sources[source.path] = ManifestSourceRecord(source, status, error)

    def finish_run(
        self,
        run_id: str,
        status: Literal["completed", "aborted"],
        totals: Mapping[str, int],
    ) -> None:
        if not self._runs or self._runs[-1].run_id != run_id:
            raise ValueError("unknown run_id")
        if status not in ("completed", "aborted"):
            raise ValueError("invalid run status")
        self._runs[-1] = replace(
            self._runs[-1], status=status, totals=tuple(sorted(totals.items()))
        )
