"""Embedded Qdrant passage storage."""

import logging
import math
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from functools import wraps
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

from qdrant_client import QdrantClient, models

from docgrep.domain.errors import StoreUnavailable
from docgrep.domain.models import FileType, Passage, RetrievalCandidate, Source

_LOG = logging.getLogger(__name__)
_ALIAS = "docgrep"
_MARKER_ID = str(uuid5(NAMESPACE_URL, "docgrep:completion"))
_PASSAGE_FILTER = models.Filter(
    must=[
        models.FieldCondition(
            key="record_type", match=models.MatchValue(value="passage")
        )
    ]
)


@contextmanager
def _without_local_index_warning() -> Iterator[None]:
    """Hide only the notice that local mode ignores payload index requests.

    The indexes are still requested so the schema matches a server deployment.
    Every other warning, such as the one about large local collections, stays
    visible.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Payload indexes have no effect in the local Qdrant",
            category=UserWarning,
        )
        yield


def _stored_dense(vector: object) -> list[float]:
    if isinstance(vector, dict) and isinstance(vector.get("dense"), list):
        return [float(value) for value in vector["dense"]]
    raise TypeError("hit carries no dense vector")


def _cosine(
    query: Sequence[float], query_norm: float, stored: Sequence[float]
) -> float:
    stored_norm = math.sqrt(sum(value * value for value in stored))
    if not query_norm or not stored_norm:
        return 0.0
    dot = sum(left * right for left, right in zip(stored, query, strict=True))
    return dot / (stored_norm * query_norm)


def _domain_errors[**P, T](operation: Callable[P, T]) -> Callable[P, T]:
    @wraps(operation)
    def guarded(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return operation(*args, **kwargs)
        except StoreUnavailable:
            raise
        except Exception as error:
            raise StoreUnavailable(
                "Qdrant operation failed",
                details={"operation": operation.__name__, "reason": str(error)},
            ) from error

    return guarded


class QdrantPassageStore:
    """Build in a staging collection and publish it through one read alias."""

    def __init__(self, path: Path) -> None:
        try:
            path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(path))
        except Exception as error:
            raise StoreUnavailable(
                "Cannot open Qdrant storage",
                details={"path": str(path), "reason": str(error)},
            ) from error
        self._staging: str | None = None
        self._dimension: int | None = None

    def _active_collection(self) -> str | None:
        for alias in self._client.get_aliases().aliases:
            if alias.alias_name == _ALIAS:
                return alias.collection_name
        return None

    @_domain_errors
    def recreate_collection(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if self._staging is not None:
            self._client.delete_collection(self._staging)
        collection = f"docgrep_{uuid4().hex}"
        self._client.create_collection(
            collection_name=collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=dimension, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        with _without_local_index_warning():
            for field, schema in (
                ("source_id", models.PayloadSchemaType.KEYWORD),
                ("author", models.PayloadSchemaType.KEYWORD),
                ("year", models.PayloadSchemaType.INTEGER),
                ("file_type", models.PayloadSchemaType.KEYWORD),
                ("path", models.PayloadSchemaType.KEYWORD),
            ):
                self._client.create_payload_index(collection, field, schema)
        self._staging = collection
        self._dimension = dimension

    @_domain_errors
    def upsert(
        self,
        passage: Passage,
        source: Source,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float],
    ) -> None:
        if self._staging is None:
            raise StoreUnavailable("No collection is being built")
        if len(dense_vector) != self._dimension:
            raise ValueError("Dense vector dimension does not match collection")
        indices = sorted(sparse_vector)
        payload = {
            "record_type": "passage",
            "passage_id": passage.passage_id,
            "source_id": source.source_id,
            "text": passage.text,
            "chunk_offset": passage.chunk_offset,
            "token_count": passage.token_count,
            "page_number": passage.page_number,
            "heading_path": list(passage.heading_path),
            "path": source.path,
            "file_type": source.file_type.value,
            "title": source.title,
            "author": source.author,
            "year": source.year,
            "content_hash": source.content_hash,
            "mtime": source.mtime,
            "size": source.size,
            "indexed_at": source.indexed_at,
            "parser_version": source.parser_version,
        }
        self._client.upsert(
            collection_name=self._staging,
            points=[
                models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, passage.passage_id)),
                    vector={
                        "dense": list(dense_vector),
                        "sparse": models.SparseVector(
                            indices=indices,
                            values=[sparse_vector[index] for index in indices],
                        ),
                    },
                    payload=payload,
                )
            ],
        )

    @_domain_errors
    def write_completion_metadata(
        self, metadata: Mapping[str, str | int | float]
    ) -> None:
        if self._staging is None or self._dimension is None:
            raise StoreUnavailable("No collection is being built")
        self._client.upsert(
            collection_name=self._staging,
            points=[
                models.PointStruct(
                    id=_MARKER_ID,
                    vector={"dense": [0.0] * self._dimension},
                    payload={"record_type": "completion", "metadata": dict(metadata)},
                )
            ],
        )
        previous = self._active_collection()
        operations: list[models.CreateAliasOperation | models.DeleteAliasOperation] = []
        if previous is not None:
            operations.append(
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name=_ALIAS)
                )
            )
        operations.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=self._staging, alias_name=_ALIAS
                )
            )
        )
        self._client.update_collection_aliases(operations)
        if previous is not None:
            self._client.delete_collection(previous)
        self._staging = None
        self._dimension = None

    @_domain_errors
    def clear_completion_metadata(self) -> None:
        if self._staging is not None:
            self._client.delete_collection(self._staging)
            self._staging = None
            self._dimension = None

    @_domain_errors
    def read_completion_metadata(self) -> Mapping[str, str | int | float] | None:
        if self._active_collection() is None:
            return None
        records = self._client.retrieve(_ALIAS, [_MARKER_ID], with_payload=True)
        if not records or records[0].payload is None:
            return None
        metadata = records[0].payload.get("metadata")
        if not isinstance(metadata, dict):
            return None
        return {
            str(key): value
            for key, value in metadata.items()
            if isinstance(value, (str, int, float))
        }

    @_domain_errors
    def count_passages(self) -> int:
        if self._active_collection() is None:
            return 0
        return self._client.count(
            _ALIAS, count_filter=_PASSAGE_FILTER, exact=True
        ).count

    @_domain_errors
    def retrieve(
        self,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float],
        top_k: int,
    ) -> tuple[RetrievalCandidate, ...]:
        if self._active_collection() is None or top_k <= 0:
            return ()
        prefetch = [
            models.Prefetch(
                query=list(dense_vector),
                using="dense",
                limit=top_k,
                filter=_PASSAGE_FILTER,
            )
        ]
        if sparse_vector:
            indices = sorted(sparse_vector)
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(
                        indices=indices,
                        values=[sparse_vector[index] for index in indices],
                    ),
                    using="sparse",
                    limit=top_k,
                    filter=_PASSAGE_FILTER,
                )
            )
        fused = self._client.query_points(
            _ALIAS,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=_PASSAGE_FILTER,
            limit=top_k,
            with_payload=True,
            with_vectors=["dense"],
        )
        query_vector = list(dense_vector)
        query_norm = math.sqrt(sum(value * value for value in query_vector))
        scored: list[tuple[float, RetrievalCandidate]] = []
        for hit in fused.points:
            try:
                payload = hit.payload or {}
                source = Source(
                    source_id=str(payload["source_id"]),
                    path=str(payload["path"]),
                    file_type=FileType(payload["file_type"]),
                    title=payload.get("title"),
                    author=payload.get("author"),
                    year=payload.get("year"),
                    content_hash=payload.get("content_hash"),
                    mtime=payload.get("mtime"),
                    size=payload.get("size"),
                    indexed_at=payload.get("indexed_at"),
                    parser_version=payload.get("parser_version"),
                )
                passage = Passage(
                    passage_id=str(payload["passage_id"]),
                    source_id=source.source_id,
                    text=str(payload["text"]),
                    chunk_offset=int(payload["chunk_offset"]),
                    token_count=int(payload["token_count"]),
                    page_number=payload.get("page_number"),
                    heading_path=tuple(payload["heading_path"]),
                )
                dense_score = _cosine(
                    query_vector, query_norm, _stored_dense(hit.vector)
                )
            except (KeyError, TypeError, ValueError) as error:
                _LOG.warning(
                    "invalid_passage_payload",
                    extra={"point_id": str(hit.id), "reason": str(error)},
                )
                continue
            scored.append(
                (
                    hit.score,
                    RetrievalCandidate(passage, source, dense_score),
                )
            )
        scored.sort(
            key=lambda item: (
                -item[0],
                -item[1].dense_score,
                item[1].passage.passage_id,
            )
        )
        return tuple(
            replace(candidate, stage1_rank=rank)
            for rank, (_, candidate) in enumerate(scored)
        )
