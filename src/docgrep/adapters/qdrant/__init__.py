"""Qdrant passage storage in local mode or against a server."""

import logging
import math
import re
import threading
import warnings
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import replace
from functools import wraps
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, uuid4, uuid5

from qdrant_client import QdrantClient, models

from docgrep.domain.context import OutlineEntry
from docgrep.domain.errors import (
    DocgrepError,
    IndexIncomplete,
    StagingInconsistent,
    StoreUnavailable,
    ValidationError,
)
from docgrep.domain.evaluation import MAX_PROBE_DEPTH, ChannelLists, ProbeHit
from docgrep.domain.models import (
    FileType,
    Passage,
    RetrievalCandidate,
    Source,
    StagingInfo,
)
from docgrep.domain.protocols import StoredPassagePage
from docgrep.domain.run_state import VECTOR_CONFIG_KEYS
from docgrep.domain.search_effort import DenseSearchEffort

_LOG = logging.getLogger(__name__)
DEFAULT_ALIAS = "docgrep"
_ALIAS_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
LOCAL_POINT_LIMIT = 20_000
_LOCAL_MODE_ADVICE = (
    "local mode is intended for tests and small corpora; "
    "set DOCGREP_QDRANT_URL for larger corpora"
)
_MARKER_ID = str(uuid5(NAMESPACE_URL, "docgrep:completion"))
_PASSAGE_FILTER = models.Filter(
    must=[
        models.FieldCondition(
            key="record_type", match=models.MatchValue(value="passage")
        )
    ]
)


def _allowed_passages_filter(source_ids: Collection[str] | None) -> models.Filter:
    """Passages only, and only of the allowed sources when a list is given."""
    if source_ids is None:
        return _PASSAGE_FILTER
    return models.Filter(
        must=[
            models.FieldCondition(
                key="record_type", match=models.MatchValue(value="passage")
            ),
            models.FieldCondition(
                key="source_id", match=models.MatchAny(any=sorted(source_ids))
            ),
        ]
    )


def _source_filter(source_id: str) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="record_type", match=models.MatchValue(value="passage")
            ),
            models.FieldCondition(
                key="source_id", match=models.MatchValue(value=source_id)
            ),
        ]
    )


def _point_id(passage_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, passage_id))


# The payload fields of a passage without its source metadata.
_PASSAGE_FIELDS = [
    "record_type",
    "passage_id",
    "source_id",
    "text",
    "chunk_offset",
    "token_count",
    "page_number",
    "heading_path",
]
# The payload fields of a probe hit, without and with its text.
_PROBE_FIELDS = ["passage_id", "source_id"]
_PROBE_TEXT_FIELDS = [*_PROBE_FIELDS, "text"]
# The payload fields that order the passages of a source.
_OUTLINE_FIELDS = ["passage_id", "chunk_offset", "token_count"]


def _passage_of(payload: Mapping[str, Any]) -> Passage:
    """The passage of a point payload; raises KeyError, TypeError or
    ValueError when the payload is not a valid passage."""
    return Passage(
        passage_id=str(payload["passage_id"]),
        source_id=str(payload["source_id"]),
        text=str(payload["text"]),
        chunk_offset=int(payload["chunk_offset"]),
        token_count=int(payload["token_count"]),
        page_number=payload.get("page_number"),
        heading_path=tuple(payload["heading_path"]),
    )


def _source_of(payload: Mapping[str, Any]) -> Source:
    """The source stored with a passage; raises like ``_passage_of``."""
    return Source(
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


def warn_if_local_mode_is_large(passages: int, limit: int = LOCAL_POINT_LIMIT) -> None:
    """Log a warning when a local-mode collection outgrows what it is meant for."""
    if passages > limit:
        _LOG.warning(_LOCAL_MODE_ADVICE, extra={"passages": passages, "limit": limit})


def _valid_alias(alias: str) -> str:
    if not _ALIAS_PATTERN.fullmatch(alias):
        raise ValidationError(
            "qdrant_alias must be non-empty and use only letters, digits, "
            "underscores and hyphens",
            details={"setting": "qdrant_alias"},
        )
    return alias


def _server_address(url: str) -> str:
    """Validate an http(s) server URL and return it without credentials."""
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError:
        parts = None
        port = None
    if parts is None or parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValidationError(
            "qdrant_url must be an http or https URL with a host name",
            details={"setting": "qdrant_url"},
        )
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    return (
        f"{parts.scheme}://{host}"
        + (f":{port}" if port else "")
        + parts.path.rstrip("/")
    )


def local_passage_count(path: Path, alias: str = DEFAULT_ALIAS) -> int | None:
    """Count the passages of an existing local store, or None when unavailable.

    The store is opened only for the count and closed again, so it does not
    keep the single-process lock of local mode.
    """
    if not path.is_dir():
        return None
    try:
        store = QdrantPassageStore(path, alias=alias)
    except StoreUnavailable:
        return None
    try:
        return store.count_passages()
    except StoreUnavailable:
        return None
    finally:
        store.close()


def _stored_sparse(vector: object) -> dict[int, float]:
    """The ``sparse`` vector of a point read with its vectors; a point
    without one has an empty sparse vector."""
    if not isinstance(vector, dict) or vector.get("sparse") is None:
        return {}
    sparse = vector["sparse"]
    if not isinstance(sparse, models.SparseVector):
        raise TypeError("point carries an invalid sparse vector")
    return {
        int(index): float(value)
        for index, value in zip(sparse.indices, sparse.values, strict=True)
    }


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


class _SerializedClient:
    """A local-mode client whose calls run one at a time.

    Local mode keeps its collections in arrays that nothing synchronizes: a
    search that reads while an upsert grows them can fail with an index
    error. A server process has several threads (searches, an update by
    automatic sync), so every call of the client takes one lock. A server
    client needs none.
    """

    def __init__(self, client: QdrantClient) -> None:
        self._client = client
        self._lock = threading.RLock()

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._client, name)
        if not callable(attribute):
            return attribute

        @wraps(attribute)
        def serialized(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                return attribute(*args, **kwargs)

        return serialized


def _local_client(path: Path) -> QdrantClient:
    client: QdrantClient = _SerializedClient(QdrantClient(path=str(path)))  # type: ignore[assignment]  # same calls, one at a time
    return client


def _domain_errors[**P, T](operation: Callable[P, T]) -> Callable[P, T]:
    @wraps(operation)
    def guarded(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return operation(*args, **kwargs)
        except DocgrepError:
            raise
        except Exception as error:
            raise StoreUnavailable(
                "Qdrant operation failed",
                details={"operation": operation.__name__, "reason": str(error)},
            ) from error

    return guarded


class QdrantPassageStore:
    """Build in a staging collection and publish it through one read alias."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        url: str | None = None,
        alias: str = DEFAULT_ALIAS,
        local_point_limit: int = LOCAL_POINT_LIMIT,
        timeout: int | None = None,
        dense_hnsw_ef: DenseSearchEffort | None = None,
    ) -> None:
        """``dense_hnsw_ef`` is the search effort of the dense channel in
        ``retrieve`` and ``probe_channels``: an hnsw_ef value (raised to the
        limit of the request when that is larger) or "exact"; None leaves
        Qdrant's default (ADR-0026)."""
        if (path is None) == (url is None):
            raise ValueError("exactly one of path and url is required")
        self._alias = _valid_alias(alias)
        self._staging_pattern = re.compile(re.escape(self._alias) + r"_[0-9a-f]{32}")
        self._local = url is None
        self._path = path
        self._local_point_limit = local_point_limit
        self._dense_hnsw_ef = dense_hnsw_ef
        if url is not None:
            address = _server_address(url)
            try:
                self._client = QdrantClient(url=address, timeout=timeout)
                self._client.get_collections()
            except Exception as error:
                raise StoreUnavailable(
                    "Cannot reach the Qdrant server",
                    details={"url": address, "reason": str(error)},
                ) from error
        elif path is not None:
            try:
                path.mkdir(parents=True, exist_ok=True)
                self._client = _local_client(path)
            except Exception as error:
                raise StoreUnavailable(
                    "Cannot open Qdrant storage",
                    details={
                        "path": str(path),
                        "reason": str(error),
                        # Another process holds the single-process local store.
                        "busy": "already accessed by another instance" in str(error),
                    },
                ) from error
        self._staging: str | None = None
        self._dimension: int | None = None
        self._live_dimension: tuple[str, int] | None = None

    def close(self) -> None:
        """Release the connection or the local storage lock."""
        self._client.close()

    @property
    def dense_hnsw_ef(self) -> DenseSearchEffort | None:
        return self._dense_hnsw_ef

    def dense_search_params(self, limit: int) -> models.SearchParams | None:
        """The search parameters of a dense request returning ``limit`` hits."""
        effort = self._dense_hnsw_ef
        if effort is None:
            return None
        if effort == "exact":
            return models.SearchParams(exact=True)
        return models.SearchParams(hnsw_ef=max(int(effort), limit))

    def _active_collection(self) -> str | None:
        for alias in self._client.get_aliases().aliases:
            if alias.alias_name == self._alias:
                return alias.collection_name
        return None

    @_domain_errors
    def recreate_collection(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if self._staging is not None:
            self._client.delete_collection(self._staging)
        collection = f"{self._alias}_{uuid4().hex}"
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
        quiet: AbstractContextManager[None] = (
            _without_local_index_warning() if self._local else nullcontext()
        )
        with quiet:
            for field, schema in (
                ("record_type", models.PayloadSchemaType.KEYWORD),
                ("source_id", models.PayloadSchemaType.KEYWORD),
                ("author", models.PayloadSchemaType.KEYWORD),
                ("year", models.PayloadSchemaType.INTEGER),
                ("file_type", models.PayloadSchemaType.KEYWORD),
                ("path", models.PayloadSchemaType.KEYWORD),
            ):
                self._client.create_payload_index(collection, field, schema)
        self._staging = collection
        self._dimension = dimension

    def _point(
        self,
        passage: Passage,
        source: Source,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float],
        *,
        dimension: int | None = None,
    ) -> models.PointStruct:
        expected = self._dimension if dimension is None else dimension
        if len(dense_vector) != expected:
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
        return models.PointStruct(
            id=_point_id(passage.passage_id),
            vector={
                "dense": list(dense_vector),
                "sparse": models.SparseVector(
                    indices=indices,
                    values=[sparse_vector[index] for index in indices],
                ),
            },
            payload=payload,
        )

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
        self._client.upsert(
            collection_name=self._staging,
            points=[self._point(passage, source, dense_vector, sparse_vector)],
        )

    @_domain_errors
    def upsert_batch(
        self,
        items: Sequence[tuple[Passage, Source, Sequence[float], Mapping[int, float]]],
    ) -> None:
        """Write several passages with one client call.

        Every item is validated and converted first, so a bad item writes nothing.
        """
        if self._staging is None:
            raise StoreUnavailable("No collection is being built")
        if not items:
            return
        points = [self._point(*item) for item in items]
        self._client.upsert(collection_name=self._staging, points=points)

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
                    delete_alias=models.DeleteAlias(alias_name=self._alias)
                )
            )
        operations.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=self._staging, alias_name=self._alias
                )
            )
        )
        self._client.update_collection_aliases(operations)
        if previous is not None:
            self._client.delete_collection(previous)
        self._staging = None
        self._dimension = None
        if self._local:
            warn_if_local_mode_is_large(self.count_passages(), self._local_point_limit)

    @_domain_errors
    def clear_completion_metadata(self) -> None:
        if self._staging is not None:
            self._client.delete_collection(self._staging)
            self._staging = None
            self._dimension = None

    def staging_name(self) -> str | None:
        """Name of the collection being built, or None when no build is open."""
        return self._staging

    @_domain_errors
    def staging_collections(self) -> tuple[str, ...]:
        """Unpublished collections named like this store's staging collections.

        The match is exact (the alias, an underscore and 32 hex digits), so the
        collections of another alias that merely starts with this alias are never
        listed, and the published collection is excluded.
        """
        published = self._active_collection()
        return tuple(
            sorted(
                collection.name
                for collection in self._client.get_collections().collections
                if self._staging_pattern.fullmatch(collection.name)
                and collection.name != published
            )
        )

    @_domain_errors
    def drop_collection(self, name: str) -> None:
        """Remove one unpublished staging collection of this store."""
        if not self._staging_pattern.fullmatch(name):
            raise ValueError("not a staging collection of this store")
        if name == self._active_collection():
            raise ValueError("refusing to drop the published collection")
        self._client.delete_collection(name)
        if name == self._staging:
            self._staging = None
            self._dimension = None

    @_domain_errors
    def staging_info(self, name: str) -> StagingInfo | None:
        """Facts about one collection with the staging name pattern, if it exists."""
        if not self._staging_pattern.fullmatch(name):
            return None
        if name not in {c.name for c in self._client.get_collections().collections}:
            return None
        vectors = self._client.get_collection(name).config.params.vectors
        dense = vectors.get("dense") if isinstance(vectors, dict) else None
        passages = self._client.count(
            name, count_filter=_PASSAGE_FILTER, exact=True
        ).count
        return StagingInfo(
            name,
            None if dense is None else dense.size,
            passages,
            published=name == self._active_collection(),
        )

    @_domain_errors
    def adopt_staging(self, name: str, dimension: int) -> None:
        info = self.staging_info(name)
        if info is None:
            raise StagingInconsistent(
                "The staging collection no longer exists",
                details={"collection": name},
            )
        if info.published:
            raise StagingInconsistent(
                "The collection is the published index, not a staging collection",
                details={"collection": name},
            )
        if info.dimension != dimension:
            raise StagingInconsistent(
                "The staging collection has a different vector dimension",
                details={
                    "collection": name,
                    "expected": dimension,
                    "found": info.dimension if info.dimension is not None else "none",
                },
            )
        self._staging = name
        self._dimension = dimension

    @_domain_errors
    def staged_source_ids(self) -> frozenset[str]:
        """Source ids with passages in the collection being built."""
        collection = self._staging or self._active_collection()
        if collection is None:
            return frozenset()
        found: set[str] = set()
        offset: models.ExtendedPointId | None = None
        while True:
            points, offset = self._client.scroll(
                collection,
                scroll_filter=_PASSAGE_FILTER,
                limit=1000,
                offset=offset,
                with_payload=["source_id"],
                with_vectors=False,
            )
            found.update(
                str(point.payload["source_id"])
                for point in points
                if point.payload and "source_id" in point.payload
            )
            if offset is None:
                return frozenset(found)

    @_domain_errors
    def delete_source_points(self, source_id: str) -> int:
        """Delete all passages of a source; returns how many were removed.

        Works on the collection being built, or on the published one when no
        build is open.
        """
        collection = self._staging or self._active_collection()
        if collection is None:
            return 0
        return self._delete_source(collection, source_id)

    def _delete_source(self, collection: str, source_id: str) -> int:
        source_filter = _source_filter(source_id)
        found = self._client.count(
            collection, count_filter=source_filter, exact=True
        ).count
        if found:
            self._client.delete(
                collection,
                points_selector=models.FilterSelector(filter=source_filter),
                wait=True,
            )
        return found

    def _source_page(
        self, collection: str, source_id: str, offset: str | None, limit: int
    ) -> StoredPassagePage:
        """One scroll page of the passages of a source with their payload and
        both vectors, in point id order. A point that is not a valid passage
        raises, so a copy is never silently partial."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        points, following = self._client.scroll(
            collection,
            scroll_filter=_source_filter(source_id),
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        page = tuple(
            (
                _passage_of(point.payload or {}),
                _source_of(point.payload or {}),
                tuple(_stored_dense(point.vector)),
                _stored_sparse(point.vector),
            )
            for point in points
        )
        return page, None if following is None else str(following)

    @_domain_errors
    def staged_source_page(
        self, source_id: str, offset: str | None, limit: int
    ) -> StoredPassagePage:
        """A page of the passages of a source in the collection being built."""
        if self._staging is None:
            raise StoreUnavailable("No collection is being built")
        return self._source_page(self._staging, source_id, offset, limit)

    @_domain_errors
    def live_source_page(
        self, source_id: str, offset: str | None, limit: int
    ) -> StoredPassagePage:
        """A page of the passages of a source in the published collection."""
        collection, _ = self._live()
        return self._source_page(collection, source_id, offset, limit)

    def _live(self) -> tuple[str, int]:
        """The published collection and its dense vector dimension."""
        collection = self._active_collection()
        if collection is None:
            raise StoreUnavailable("No index is published")
        if self._live_dimension is None or self._live_dimension[0] != collection:
            vectors = self._client.get_collection(collection).config.params.vectors
            dense = vectors.get("dense") if isinstance(vectors, dict) else None
            if dense is None:
                raise StoreUnavailable(
                    "The published index has no dense vectors",
                    details={"collection": collection},
                )
            self._live_dimension = (collection, dense.size)
        return self._live_dimension

    @_domain_errors
    def upsert_live_batch(
        self,
        items: Sequence[tuple[Passage, Source, Sequence[float], Mapping[int, float]]],
    ) -> None:
        """Write passages into the published collection and wait until stored.

        Every item is validated and converted first, so a bad item writes nothing.
        """
        if not items:
            return
        collection, dimension = self._live()
        points = [self._point(*item, dimension=dimension) for item in items]
        self._client.upsert(collection_name=collection, points=points, wait=True)

    @_domain_errors
    def delete_live_source_points(self, source_id: str) -> int:
        """Delete every passage of a source from the published collection."""
        collection, _ = self._live()
        return self._delete_source(collection, source_id)

    @_domain_errors
    def delete_live_source_points_except(
        self, source_id: str, keep: Collection[str]
    ) -> int:
        """Delete the published passages of a source that are not in ``keep``."""
        collection, _ = self._live()
        kept = {_point_id(passage_id) for passage_id in keep}
        doomed: list[models.ExtendedPointId] = []
        offset: models.ExtendedPointId | None = None
        while True:
            points, offset = self._client.scroll(
                collection,
                scroll_filter=_source_filter(source_id),
                limit=1000,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            doomed.extend(point.id for point in points if str(point.id) not in kept)
            if offset is None:
                break
        for start in range(0, len(doomed), 1000):
            self._client.delete(
                collection,
                points_selector=models.PointIdsList(
                    points=doomed[start : start + 1000]
                ),
                wait=True,
            )
        return len(doomed)

    @_domain_errors
    def refresh_live_source(self, source_id: str, *, size: int, mtime: float) -> None:
        """Set size and mtime on the published passages of a source."""
        collection, _ = self._live()
        self._client.set_payload(
            collection,
            payload={"size": size, "mtime": mtime},
            points=models.FilterSelector(filter=_source_filter(source_id)),
            wait=True,
        )

    @_domain_errors
    def merge_completion_metadata(
        self, updates: Mapping[str, str | int | float]
    ) -> None:
        """Add or replace completion marker keys of the published collection."""
        refused = sorted(set(updates) & set(VECTOR_CONFIG_KEYS))
        if refused:
            raise ValidationError(
                "Only a full build may set these index metadata keys",
                details={"keys": ", ".join(refused)},
            )
        collection, _ = self._live()
        records = self._client.retrieve(collection, [_MARKER_ID], with_payload=True)
        metadata = (
            records[0].payload.get("metadata")
            if records and records[0].payload
            else None
        )
        if not isinstance(metadata, dict):
            raise StoreUnavailable(
                "The published index has no completion marker",
                details={"collection": collection},
            )
        self._client.set_payload(
            collection,
            payload={"metadata": {**metadata, **dict(updates)}},
            points=[_MARKER_ID],
            wait=True,
        )

    def _sparse_target(self) -> str:
        """The collection being built, else the published one."""
        collection = self._staging or self._active_collection()
        if collection is None:
            raise StoreUnavailable("No index is published")
        return collection

    @_domain_errors
    def target_passage_count(self) -> int:
        """The passages of the collection being built, else the published one."""
        return self._client.count(
            self._sparse_target(), count_filter=_PASSAGE_FILTER, exact=True
        ).count

    def _reload_local(self) -> None:
        """Reopen a local store, so every point is held as it was persisted.

        Local mode keeps the points written by this process with their
        dense vector normalized, but persists and loads them as given; a
        rewrite of a point persists what is held. Reloading first makes a
        sparse rewrite leave every dense vector exactly as persisted, as a
        server does.
        """
        if self._local and self._path is not None:
            self._client.close()
            self._client = _local_client(self._path)

    @_domain_errors
    def passage_text_page(
        self, offset: str | None, limit: int
    ) -> tuple[tuple[tuple[str, str], ...], str | None]:
        """One scroll page of passage ids and texts, in point id order.

        A scan from the beginning reloads a local store first (see
        ``_reload_local``).
        """
        if offset is None:
            self._reload_local()
        points, following = self._client.scroll(
            self._sparse_target(),
            scroll_filter=_PASSAGE_FILTER,
            limit=limit,
            offset=offset,
            with_payload=["passage_id", "text"],
            with_vectors=False,
        )
        page: list[tuple[str, str]] = []
        for point in points:
            payload = point.payload or {}
            try:
                page.append((str(payload["passage_id"]), str(payload["text"])))
            except KeyError as error:
                _LOG.warning(
                    "invalid_passage_payload",
                    extra={"point_id": str(point.id), "reason": str(error)},
                )
        return tuple(page), None if following is None else str(following)

    @_domain_errors
    def write_sparse_vectors(
        self, vectors: Sequence[tuple[str, Mapping[int, float]]]
    ) -> None:
        """Replace only the ``sparse`` vector of each point and wait.

        An empty vector is dropped with a debug record: a point with no
        vector to update is rejected by Qdrant, and an empty sparse vector
        stays empty under any weighting.
        """
        points = []
        for passage_id, sparse in vectors:
            if not sparse:
                _LOG.debug(
                    "empty_sparse_vector_skipped", extra={"passage_id": passage_id}
                )
                continue
            indices = sorted(sparse)
            points.append(
                models.PointVectors(
                    id=_point_id(passage_id),
                    vector={
                        "sparse": models.SparseVector(
                            indices=indices, values=[sparse[i] for i in indices]
                        )
                    },
                )
            )
        if not points:
            return
        self._client.update_vectors(self._sparse_target(), points=points, wait=True)

    @_domain_errors
    def read_completion_metadata(self) -> Mapping[str, str | int | float] | None:
        if self._active_collection() is None:
            return None
        records = self._client.retrieve(self._alias, [_MARKER_ID], with_payload=True)
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
            self._alias, count_filter=_PASSAGE_FILTER, exact=True
        ).count

    @_domain_errors
    def count_source_passages(self, source_id: str) -> int:
        """The exact passage count of one source in the published collection."""
        if self._active_collection() is None:
            raise IndexIncomplete("No index is published")
        return self._client.count(
            self._alias, count_filter=_source_filter(source_id), exact=True
        ).count

    @_domain_errors
    def retrieve(
        self,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float],
        top_k: int,
        source_ids: Collection[str] | None = None,
        *,
        pool: int | None = None,
    ) -> tuple[RetrievalCandidate, ...]:
        if self._active_collection() is None or top_k <= 0:
            return ()
        if source_ids is not None and not source_ids:
            return ()
        # The allow-list sits in every stage of the query, so the candidates
        # are the best of the allowed sources, not a remainder of the best.
        allowed = _allowed_passages_filter(source_ids)
        prefetch = [
            models.Prefetch(
                query=list(dense_vector),
                using="dense",
                limit=top_k,
                filter=allowed,
                params=self.dense_search_params(top_k),
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
                    filter=allowed,
                )
            )
        fused = self._client.query_points(
            self._alias,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=allowed,
            # Each channel keeps top_k; the fused list may keep more of
            # their union (at most 2 * top_k) for the reranker.
            limit=top_k if pool is None else pool,
            with_payload=True,
            with_vectors=["dense"],
        )
        query_vector = list(dense_vector)
        query_norm = math.sqrt(sum(value * value for value in query_vector))
        scored: list[tuple[float, RetrievalCandidate]] = []
        for hit in fused.points:
            try:
                payload = hit.payload or {}
                source = _source_of(payload)
                passage = _passage_of(payload)
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

    @_domain_errors
    def probe_channels(
        self,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float],
        depth: int,
        *,
        source_ids: Collection[str] | None = None,
        with_text: bool = False,
    ) -> ChannelLists:
        """The dense, sparse and fused lists of a query in one batch request.

        The fused list is the query ``retrieve`` sends, with ``depth`` in
        place of ``top_k`` in every stage. Only the passage id, the source id
        and, ``with_text``, the text are read; no vectors.
        """
        if not 0 < depth <= MAX_PROBE_DEPTH:
            raise ValidationError(
                f"depth must be between 1 and {MAX_PROBE_DEPTH}",
                details={"minimum": 1, "maximum": MAX_PROBE_DEPTH},
            )
        collection = self._active_collection()
        if collection is None:
            return ChannelLists((), (), ())
        info = self._client.get_collection(collection)
        sparse_channel = "sparse" in (info.config.params.sparse_vectors or {})
        if source_ids is not None and not source_ids:
            return ChannelLists((), () if sparse_channel else None, ())
        if not sparse_channel:
            # Without a sparse channel the fusion has the dense list alone.
            sparse_vector = {}
        allowed = _allowed_passages_filter(source_ids)
        fields = _PROBE_TEXT_FIELDS if with_text else _PROBE_FIELDS
        dense_params = self.dense_search_params(depth)
        dense = models.Prefetch(
            query=list(dense_vector),
            using="dense",
            limit=depth,
            filter=allowed,
            params=dense_params,
        )
        prefetch = [dense]
        sparse: models.Prefetch | None = None
        if sparse_vector:
            indices = sorted(sparse_vector)
            sparse = models.Prefetch(
                query=models.SparseVector(
                    indices=indices,
                    values=[sparse_vector[index] for index in indices],
                ),
                using="sparse",
                limit=depth,
                filter=allowed,
            )
            prefetch.append(sparse)

        def request(**query: Any) -> models.QueryRequest:
            return models.QueryRequest(
                **query,
                filter=allowed,
                limit=depth,
                with_payload=fields,
                with_vector=False,
            )

        requests = [
            request(query=dense.query, using="dense", params=dense_params),
            request(
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
            ),
        ]
        if sparse is not None:
            requests.append(request(query=sparse.query, using="sparse"))
        responses = self._client.query_batch_points(self._alias, requests=requests)
        dense_hits, fused_hits, *sparse_hits = (
            self._probe_hits(response.points, with_text) for response in responses
        )
        # Fused ties break as in ``retrieve``: by dense similarity, then by
        # passage id. A hit missing from the dense list is less similar than
        # every hit in it.
        similarity = {hit.passage_id: score for score, hit in dense_hits}
        fused_hits.sort(
            key=lambda item: (
                -item[0],
                -similarity.get(item[1].passage_id, -math.inf),
                item[1].passage_id,
            )
        )
        sparse_list: tuple[ProbeHit, ...] | None = None
        if sparse_hits:
            sparse_list = tuple(hit for _, hit in sparse_hits[0])
        elif sparse_channel:
            sparse_list = ()
        return ChannelLists(
            tuple(hit for _, hit in dense_hits),
            sparse_list,
            tuple(hit for _, hit in fused_hits),
        )

    @_domain_errors
    def dense_recall(
        self,
        dense_vector: Sequence[float],
        limit: int,
        *,
        source_ids: Collection[str] | None = None,
    ) -> float | None:
        """Recall at ``limit`` of the dense channel as search runs it,
        against exact search: the share of the exact top ``limit`` that the
        configured search effort returns. None when exact search returns
        nothing (no published index or no eligible passage)."""
        if not 0 < limit <= MAX_PROBE_DEPTH:
            raise ValidationError(
                f"limit must be between 1 and {MAX_PROBE_DEPTH}",
                details={"minimum": 1, "maximum": MAX_PROBE_DEPTH},
            )
        if self._active_collection() is None:
            return None
        if source_ids is not None and not source_ids:
            return None
        allowed = _allowed_passages_filter(source_ids)

        def request(params: models.SearchParams | None) -> models.QueryRequest:
            return models.QueryRequest(
                query=list(dense_vector),
                using="dense",
                filter=allowed,
                limit=limit,
                params=params,
                with_payload=False,
                with_vector=False,
            )

        exact, effective = self._client.query_batch_points(
            self._alias,
            requests=[
                request(models.SearchParams(exact=True)),
                request(self.dense_search_params(limit)),
            ],
        )
        expected = {point.id for point in exact.points}
        if not expected:
            return None
        found = {point.id for point in effective.points}
        return len(found & expected) / len(expected)

    @staticmethod
    def _probe_hits(
        points: Sequence[models.ScoredPoint], with_text: bool
    ) -> list[tuple[float, ProbeHit]]:
        """The hits of one probe list with their scores, best first; ties keep
        the store's order, which is the order the fusion ranks them in."""
        scored: list[tuple[float, ProbeHit]] = []
        for point in points:
            payload = point.payload or {}
            try:
                hit = ProbeHit(
                    str(payload["passage_id"]),
                    str(payload["source_id"]),
                    str(payload["text"]) if with_text else None,
                )
            except KeyError as error:
                _LOG.warning(
                    "invalid_passage_payload",
                    extra={"point_id": str(point.id), "reason": str(error)},
                )
                continue
            scored.append((point.score, hit))
        scored.sort(key=lambda item: -item[0])
        return scored

    def _published_points(
        self, passage_ids: Collection[str], fields: list[str] | bool
    ) -> list[models.Record]:
        """The passage points with these ids in the published collection."""
        if not passage_ids or self._active_collection() is None:
            return []
        records = self._client.retrieve(
            self._alias,
            [_point_id(passage_id) for passage_id in passage_ids],
            with_payload=fields,
            with_vectors=False,
        )
        return [
            record
            for record in records
            if record.payload and record.payload.get("record_type") == "passage"
        ]

    @_domain_errors
    def fetch_passages(
        self, passage_ids: Sequence[str]
    ) -> tuple[tuple[Passage, Source] | None, ...]:
        """Published passages with their stored source, in input order, in one
        retrieve; None for an id without a valid passage."""
        found: dict[str, tuple[Passage, Source]] = {}
        for record in self._published_points(set(passage_ids), True):
            payload = record.payload or {}
            try:
                passage = _passage_of(payload)
                found[passage.passage_id] = (passage, _source_of(payload))
            except (KeyError, TypeError, ValueError) as error:
                _LOG.warning(
                    "invalid_passage_payload",
                    extra={"point_id": str(record.id), "reason": str(error)},
                )
        return tuple(found.get(passage_id) for passage_id in passage_ids)

    @_domain_errors
    def source_outline(self, source_id: str) -> tuple[OutlineEntry, ...]:
        """The published passages of a source by chunk offset, read with the
        ``source_id`` filter and only the fields that order them."""
        if self._active_collection() is None:
            return ()
        entries: list[OutlineEntry] = []
        offset: models.ExtendedPointId | None = None
        while True:
            points, offset = self._client.scroll(
                self._alias,
                scroll_filter=_source_filter(source_id),
                limit=1000,
                offset=offset,
                with_payload=_OUTLINE_FIELDS,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                try:
                    entries.append(
                        OutlineEntry(
                            str(payload["passage_id"]),
                            int(payload["chunk_offset"]),
                            int(payload["token_count"]),
                        )
                    )
                except (KeyError, TypeError, ValueError) as error:
                    _LOG.warning(
                        "invalid_passage_payload",
                        extra={"point_id": str(point.id), "reason": str(error)},
                    )
            if offset is None:
                break
        entries.sort(key=lambda entry: (entry.chunk_offset, entry.passage_id))
        return tuple(entries)

    @_domain_errors
    def passage_texts(self, passage_ids: Collection[str]) -> Mapping[str, Passage]:
        """Published passages by id, in one retrieve without source metadata."""
        found: dict[str, Passage] = {}
        for record in self._published_points(set(passage_ids), _PASSAGE_FIELDS):
            try:
                passage = _passage_of(record.payload or {})
            except (KeyError, TypeError, ValueError) as error:
                _LOG.warning(
                    "invalid_passage_payload",
                    extra={"point_id": str(record.id), "reason": str(error)},
                )
                continue
            found[passage.passage_id] = passage
        return found
