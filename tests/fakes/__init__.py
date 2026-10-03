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
        self._building = False
        self.orphans: list[str] = []

    def recreate_collection(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._items.clear()
        self._completion = None
        self._building = True

    def staging_name(self) -> str | None:
        return "memory-staging" if self._building else None

    def staging_collections(self) -> tuple[str, ...]:
        return tuple(self.orphans)

    def drop_collection(self, name: str) -> None:
        if name not in self.orphans:
            raise ValueError("not a staging collection of this store")
        self.orphans.remove(name)

    def delete_source_points(self, source_id: str) -> int:
        doomed = [
            pid for pid, item in self._items.items() if item[1].source_id == source_id
        ]
        for passage_id in doomed:
            del self._items[passage_id]
        return len(doomed)

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

    def count_passages(self) -> int:
        return len(self._items)

    def write_completion_metadata(
        self, metadata: Mapping[str, str | int | float]
    ) -> None:
        self._completion = dict(metadata)
        self._building = False

    def clear_completion_metadata(self) -> None:
        self._completion = None
        self._building = False


class InMemoryManifest:
    """Keep source and run records with the same read/write port surface."""

    def __init__(self) -> None:
        self._sources: dict[str, ManifestSourceRecord] = {}
        self._runs: list[ManifestRunRecord] = []
        self._journal: dict[str, dict[str, ManifestSourceRecord]] = {}

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
        self,
        source: Source,
        status: str,
        error: str | None = None,
        embedding_model: str | None = None,
        *,
        run_id: str | None = None,
        passages: int = 0,
    ) -> None:
        if run_id is not None:
            self._journal.setdefault(run_id, {})[source.path] = ManifestSourceRecord(
                source, status, error, embedding_model, passages
            )
            return
        self._sources[source.path] = ManifestSourceRecord(
            source, status, error, embedding_model
        )

    def run_sources(self, run_id: str) -> tuple[ManifestSourceRecord, ...]:
        journal = self._journal.get(run_id, {})
        return tuple(journal[path] for path in sorted(journal))

    def set_staging_collection(self, run_id: str, name: str) -> None:
        index = next(
            (i for i, run in enumerate(self._runs) if run.run_id == run_id), None
        )
        if index is None:
            raise ValueError("unknown run_id")
        self._runs[index] = replace(self._runs[index], staging_collection=name)

    def commit_run(self, run_id: str) -> None:
        if not any(run.run_id == run_id for run in self._runs):
            raise ValueError("unknown run_id")
        self._sources = {
            path: replace(record, passages=None)
            for path, record in self._journal.get(run_id, {}).items()
        }

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
