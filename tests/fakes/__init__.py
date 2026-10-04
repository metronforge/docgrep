"""In-memory implementations of persistence and retrieval boundaries."""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Literal

from docgrep.domain.errors import StagingInconsistent, StoreUnavailable, ValidationError
from docgrep.domain.models import (
    ManifestRunRecord,
    ManifestSourceRecord,
    Passage,
    RetrievalCandidate,
    Source,
    StagingInfo,
)
from docgrep.domain.run_state import VECTOR_CONFIG_KEYS


class InMemoryPassageStore:
    """A small deterministic hybrid retriever with test-only write helpers."""

    def __init__(self) -> None:
        self._items: dict[
            str, tuple[Passage, Source, tuple[float, ...], dict[int, float]]
        ] = {}
        self._staged: dict[
            str, tuple[Passage, Source, tuple[float, ...], dict[int, float]]
        ] = {}
        self._completion: dict[str, str | int | float] | None = None
        self._building = False
        self._dimension = 0
        self.orphans: list[str] = []

    def _target(
        self,
    ) -> dict[str, tuple[Passage, Source, tuple[float, ...], dict[int, float]]]:
        """Writes go to the collection being built, else to the published one."""
        return self._staged if self._building else self._items

    def recreate_collection(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._staged = {}
        self._building = True
        self._dimension = dimension

    def staging_info(self, name: str) -> StagingInfo | None:
        if name != "memory-staging":
            return None
        if self._building:
            return StagingInfo(name, self._dimension, len(self._staged))
        if self._completion is not None:  # the one collection is the published one
            return StagingInfo(name, self._dimension, len(self._items), published=True)
        return None

    def adopt_staging(self, name: str, dimension: int) -> None:
        info = self.staging_info(name)
        if info is None:
            raise StagingInconsistent(
                "The staging collection no longer exists", details={"collection": name}
            )
        if info.dimension != dimension:
            raise StagingInconsistent(
                "The staging collection has a different vector dimension",
                details={"collection": name},
            )

    def staged_source_ids(self) -> frozenset[str]:
        return frozenset(item[1].source_id for item in self._target().values())

    def staging_name(self) -> str | None:
        return "memory-staging" if self._building else None

    def staging_collections(self) -> tuple[str, ...]:
        retained = ["memory-staging"] if self._building else []
        return tuple([*self.orphans, *retained])

    def drop_collection(self, name: str) -> None:
        if name == "memory-staging" and self._building:
            self._staged = {}
            self._building = False
            return
        if name not in self.orphans:
            raise ValueError("not a staging collection of this store")
        self.orphans.remove(name)

    def delete_source_points(self, source_id: str) -> int:
        target = self._target()
        doomed = [pid for pid, item in target.items() if item[1].source_id == source_id]
        for passage_id in doomed:
            del target[passage_id]
        return len(doomed)

    def upsert(
        self,
        passage: Passage,
        source: Source,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float],
    ) -> None:
        self._target()[passage.passage_id] = (
            passage,
            source,
            tuple(dense_vector),
            dict(sparse_vector),
        )

    def upsert_batch(
        self,
        items: Sequence[tuple[Passage, Source, Sequence[float], Mapping[int, float]]],
    ) -> None:
        for passage, source, dense_vector, sparse_vector in items:
            self.upsert(passage, source, dense_vector, sparse_vector)

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
        if self._building:
            self._items = self._staged
            self._staged = {}
        self._completion = dict(metadata)
        self._building = False

    def clear_completion_metadata(self) -> None:
        """Drop the collection being built; the published one is untouched."""
        self._staged = {}
        self._building = False

    # The live side: always the published collection, never the staging one.

    def _published(
        self,
    ) -> dict[str, tuple[Passage, Source, tuple[float, ...], dict[int, float]]]:
        if self._completion is None:
            raise StoreUnavailable("No index is published")
        return self._items

    def upsert_live_batch(
        self,
        items: Sequence[tuple[Passage, Source, Sequence[float], Mapping[int, float]]],
    ) -> None:
        published = self._published()
        for passage, source, dense_vector, sparse_vector in items:
            published[passage.passage_id] = (
                passage,
                source,
                tuple(dense_vector),
                dict(sparse_vector),
            )

    def delete_live_source_points(self, source_id: str) -> int:
        return self._delete_published(source_id, ())

    def delete_live_source_points_except(
        self, source_id: str, keep: Collection[str]
    ) -> int:
        return self._delete_published(source_id, keep)

    def _delete_published(self, source_id: str, keep: Collection[str]) -> int:
        published = self._published()
        doomed = [
            passage_id
            for passage_id, item in published.items()
            if item[1].source_id == source_id and passage_id not in keep
        ]
        for passage_id in doomed:
            del published[passage_id]
        return len(doomed)

    def refresh_live_source(self, source_id: str, *, size: int, mtime: float) -> None:
        published = self._published()
        for passage_id, (passage, source, dense, sparse) in list(published.items()):
            if source.source_id == source_id:
                fresh = replace(source, size=size, mtime=mtime)
                published[passage_id] = (passage, fresh, dense, sparse)

    def merge_completion_metadata(
        self, updates: Mapping[str, str | int | float]
    ) -> None:
        if set(updates) & set(VECTOR_CONFIG_KEYS):
            raise ValidationError("Only a full build may set these keys")
        if self._completion is None:
            raise StoreUnavailable("No index is published")
        self._completion = {**self._completion, **updates}


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

    def latest_run_of_kind(self, kind: str) -> ManifestRunRecord | None:
        return next((run for run in reversed(self._runs) if run.kind == kind), None)

    def committed_sources(self) -> tuple[ManifestSourceRecord, ...]:
        return tuple(self._sources[path] for path in sorted(self._sources))

    def remove_source(self, path: str) -> None:
        self._sources.pop(path, None)

    def request_cancel(self, run_id: str) -> str:
        run = self._find(run_id)
        if run.cancel_requested_at is None:
            self._update_run(run_id, cancel_requested_at=datetime.now(UTC).isoformat())
        return self._find(run_id).cancel_requested_at or ""

    def cancel_requested_at(self, run_id: str) -> str | None:
        return self._find(run_id).cancel_requested_at

    def set_total_sources(self, run_id: str, total: int, *, carried: int = 0) -> None:
        self._update_run(run_id, total_sources=total, carried_sources=carried)

    def carry_over(
        self, from_run_id: str, to_run_id: str, paths: Collection[str]
    ) -> None:
        source = self._journal.get(from_run_id, {})
        target = self._journal.setdefault(to_run_id, {})
        for path in paths:
            if path in source:
                target[path] = source[path]

    def mark_swapped(self, run_id: str) -> None:
        run = self._find(run_id)
        if run.status not in ("running", "aborted"):
            raise ValueError("unknown or already finished run_id")
        self._update_run(run_id, status="swapped")

    def _find(self, run_id: str) -> ManifestRunRecord:
        for run in self._runs:
            if run.run_id == run_id:
                return run
        raise ValueError("unknown run_id")

    def _update_run(self, run_id: str, **changes: object) -> None:
        self._find(run_id)
        index = next(i for i, run in enumerate(self._runs) if run.run_id == run_id)
        self._runs[index] = replace(self._runs[index], **changes)

    def start_run(
        self,
        config: Mapping[str, str | int | float],
        *,
        resumed_from: str | None = None,
        kind: str = "build",
    ) -> str:
        run_id = str(len(self._runs) + 1)
        self._runs.append(
            ManifestRunRecord(
                run_id,
                "running",
                tuple(sorted(config.items())),
                kind=kind,
                resumed_from=resumed_from,
            )
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
        status: Literal["completed", "aborted", "cancelled"],
        totals: Mapping[str, int],
    ) -> None:
        if not self._runs or self._runs[-1].run_id != run_id:
            raise ValueError("unknown run_id")
        if self._runs[-1].status not in ("running", "swapped"):
            raise ValueError("unknown or finished run_id")
        if status not in ("completed", "aborted", "cancelled"):
            raise ValueError("invalid run status")
        self._runs[-1] = replace(
            self._runs[-1], status=status, totals=tuple(sorted(totals.items()))
        )
