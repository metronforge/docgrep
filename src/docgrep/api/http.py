"""FastAPI transport for local search and full index builds."""

import importlib.util
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
from threading import Lock
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    ValidationInfo,
    field_validator,
    model_validator,
)
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from docgrep import __version__
from docgrep.adapters.corpus import root_readable
from docgrep.adapters.qdrant import local_passage_count, warn_if_local_mode_is_large
from docgrep.api.container import (
    ContextOperation,
    SearchOperation,
    Services,
    create_container,
    create_discover_scans,
    create_process_scans,
    create_status,
    index_missing,
    open_ocr_registry_view,
    unindexed_context,
)
from docgrep.api.log_config import (
    configure_logging,
    log_server_start,
    warn_if_not_loopback,
)
from docgrep.api.roots import corpus_roots
from docgrep.api.sync_task import SyncTask
from docgrep.domain.context import ContextResult, PassageContext, PassageLocation
from docgrep.domain.errors import (
    ConfigDrift,
    DocgrepError,
    IndexIncomplete,
    IndexLockHeld,
    ManifestUnavailable,
    ModelUnavailable,
    OcrToolsMissing,
    ParseError,
    ResumeAvailable,
    StoreUnavailable,
    ValidationError,
)
from docgrep.domain.metadata import DropReason, FieldAudit, MetadataAudit
from docgrep.domain.models import FileType, RankedPassage, SearchResult
from docgrep.domain.sources import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    BrowsedSource,
    BrowseRequest,
    BrowseResult,
    HasMetadata,
    OcrStatus,
    SortKey,
    SortOrder,
    SourceFilter,
    distinct_texts,
    parse_file_types,
)
from docgrep.domain.sync import SyncStatus
from docgrep.settings import Settings
from docgrep.usecases.build_index import BuildSummary
from docgrep.usecases.process_scans import OcrRequest


class SourceResponse(BaseModel):
    """Known source metadata without inferred fields."""

    model_config = ConfigDict(from_attributes=True)

    source_id: str
    path: str
    file_type: FileType
    title: str | None
    author: str | None
    year: int | None
    content_hash: str | None
    mtime: float | None
    size: int | None
    indexed_at: str | None
    parser_version: str | None
    # The name of the source's root (ADR-0023); set in search results.
    root: str | None = Field(default=None, exclude_if=lambda value: value is None)


class PassageResponse(BaseModel):
    """A verbatim indexed passage and its location."""

    model_config = ConfigDict(from_attributes=True)

    passage_id: str
    source_id: str
    text: str
    chunk_offset: int
    token_count: int
    page_number: int | None
    heading_path: tuple[str, ...]


class DuplicateResponse(BaseModel):
    """Where another copy of a collapsed passage's text was found."""

    passage_id: str
    source_id: str
    path: str
    file_type: FileType
    page_number: int | None
    heading_path: tuple[str, ...]
    root: str | None = Field(default=None, exclude_if=lambda value: value is None)

    @classmethod
    def from_domain(
        cls, item: RankedPassage, roots: Mapping[str, str] | None = None
    ) -> "DuplicateResponse":
        return cls(
            passage_id=item.passage.passage_id,
            source_id=item.source.source_id,
            path=item.source.path,
            file_type=item.source.file_type,
            page_number=item.passage.page_number,
            heading_path=item.passage.heading_path,
            root=(roots or {}).get(item.source.source_id),
        )


class RankedPassageResponse(BaseModel):
    """One passage with source metadata and retrieval scores."""

    passage: PassageResponse
    source: SourceResponse
    score: float
    dense_score: float
    reranker_score: float | None
    # Omitted when empty, so a response without duplicates keeps its old shape.
    duplicates: tuple[DuplicateResponse, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )

    @classmethod
    def from_domain(
        cls, item: RankedPassage, roots: Mapping[str, str] | None = None
    ) -> "RankedPassageResponse":
        source = SourceResponse.model_validate(item.source)
        source.root = (roots or {}).get(item.source.source_id)
        return cls(
            passage=PassageResponse.model_validate(item.passage),
            source=source,
            score=item.score,
            dense_score=item.dense_score,
            reranker_score=item.reranker_score,
            duplicates=tuple(
                DuplicateResponse.from_domain(duplicate, roots)
                for duplicate in item.duplicates
            ),
        )


class SearchResponse(BaseModel):
    """Ranked findings and the configuration applied to them."""

    results: tuple[RankedPassageResponse, ...]
    total_hits: int
    took_ms: float
    applied_config: dict[str, str | int | float | bool]
    rerank_skipped: bool
    rerank_skip_reason: str | None
    hint: str | None
    # Omitted when collapsing is off, so the response keeps its old shape.
    collapsed_duplicates: int | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # The canonical source filter; {} for an unfiltered search.
    applied_filters: dict[str, list[str] | int] = Field(default_factory=dict)
    # Present only when the filter sets an author or a year bound: the
    # sources left out only because that metadata is null.
    sources_excluded_for_missing_metadata: int | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @classmethod
    def from_domain(cls, result: SearchResult) -> "SearchResponse":
        return cls(
            results=tuple(
                RankedPassageResponse.from_domain(item, result.source_roots)
                for item in result.results
            ),
            total_hits=result.total_hits,
            took_ms=result.took_ms,
            applied_config=dict(result.applied_config),
            rerank_skipped=result.rerank_skipped,
            rerank_skip_reason=result.rerank_skip_reason,
            hint=result.hint,
            collapsed_duplicates=result.collapsed_duplicates,
            applied_filters=dict(result.applied_filters),
            sources_excluded_for_missing_metadata=(
                result.sources_excluded_for_missing_metadata
            ),
        )


def _checked[T](check: Callable[[], T]) -> T:
    """Run a domain check; its validation error becomes a pydantic one."""
    try:
        return check()
    except ValidationError as error:
        raise ValueError(str(error)) from error


class SourceFilterBody(BaseModel):
    """The fields of the shared source filter, for browsing and search.

    Unknown fields are rejected. The values are checked by the domain's
    ``SourceFilter``, so every front applies the same rules.
    """

    model_config = ConfigDict(extra="forbid")

    source_ids: list[str] | None = None
    authors: list[str] | None = None
    year_from: int | None = None
    year_to: int | None = None
    file_types: list[str] | None = None
    path_prefixes: list[str] | None = None
    path_globs: list[str] | None = None
    # Root names (ADR-0023); checked against the known roots when resolved.
    roots: list[str] | None = None

    @field_validator("source_ids", "authors", "path_prefixes", "path_globs", "roots")
    @classmethod
    def check_texts(
        cls, values: list[str] | None, info: ValidationInfo
    ) -> list[str] | None:
        if values is not None:
            _checked(lambda: distinct_texts(info.field_name or "", values))
        return values

    @field_validator("file_types")
    @classmethod
    def check_file_types(cls, values: list[str] | None) -> list[str] | None:
        if values is not None:
            _checked(lambda: parse_file_types(values))
        return values

    @model_validator(mode="after")
    def check_years(self) -> Self:
        _checked(lambda: SourceFilter(year_from=self.year_from, year_to=self.year_to))
        return self

    def source_filter(self) -> SourceFilter:
        return SourceFilter(
            source_ids=tuple(self.source_ids or ()),
            authors=tuple(self.authors or ()),
            year_from=self.year_from,
            year_to=self.year_to,
            file_types=parse_file_types(self.file_types or ()),
            path_prefixes=tuple(self.path_prefixes or ()),
            path_globs=tuple(self.path_globs or ()),
            roots=tuple(self.roots or ()),
        )


class SearchRequest(SourceFilterBody):
    """Validated search parameters and an optional source filter (UC-02).

    Unknown fields are rejected, so a misspelt filter is never ignored.
    """

    query: str
    top_n: int | None = None
    # None applies the DOCGREP_COLLAPSE_DUPLICATES setting.
    collapse_duplicates: bool | None = None


class BrowseSourcesRequest(SourceFilterBody):
    """The filter, metadata rule, page and order of a source listing.

    Unknown fields are rejected. The values are checked by the domain's
    ``SourceFilter`` and ``BrowseRequest``, so every front applies the same
    rules.
    """

    has_metadata: HasMetadata = "any"
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE)
    sort_by: SortKey = "path"
    sort_order: SortOrder = "asc"
    # True: only sources whose metadata was dropped or inherited; false: only
    # the others; absent: both.
    metadata_dropped: StrictBool | None = None

    def to_domain(self) -> BrowseRequest:
        return BrowseRequest(
            self.source_filter(),
            has_metadata=self.has_metadata,
            page=self.page,
            page_size=self.page_size,
            sort_by=self.sort_by,
            sort_order=self.sort_order,
            metadata_dropped=self.metadata_dropped,
        )


class DroppedFieldResponse(BaseModel):
    """What sanitizing did to one metadata field (ADR-0021).

    ``raw`` is the stored value that was dropped, absent when none was
    stored; ``inherited_from`` is the original's path when the shown value
    comes from the original of an OCR output.
    """

    raw: str | int | None = Field(default=None, exclude_if=lambda value: value is None)
    reason: DropReason
    inherited_from: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @classmethod
    def from_domain(cls, item: FieldAudit) -> "DroppedFieldResponse":
        return cls(raw=item.raw, reason=item.reason, inherited_from=item.inherited_from)


class MetadataDroppedResponse(BaseModel):
    """The metadata fields of a source that were dropped or inherited."""

    title: DroppedFieldResponse | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    author: DroppedFieldResponse | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    year: DroppedFieldResponse | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @classmethod
    def from_domain(cls, audit: MetadataAudit) -> "MetadataDroppedResponse":
        return cls(
            **{
                name: DroppedFieldResponse.from_domain(item)
                for name, item in audit.fields()
            }
        )


class BrowsedSourceResponse(BaseModel):
    """One indexed source with its passage count and OCR origin.

    Title, author and year are the sanitized values (ADR-0021).
    """

    source_id: str
    path: str
    file_type: FileType
    title: str | None
    author: str | None
    year: int | None
    # Null when the count failed; the entry is then marked degraded.
    passage_count: int | None
    indexed_at: str | None
    content_hash: str | None
    ocr_status: OcrStatus
    # The name of the source's root (ADR-0023).
    root: str | None = Field(default=None, exclude_if=lambda value: value is None)
    # Present, and true, only when the passage count failed.
    degraded: bool = Field(default=False, exclude_if=lambda value: not value)
    # Present only when sanitizing dropped or inherited a field.
    metadata_dropped: MetadataDroppedResponse | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @classmethod
    def from_domain(cls, item: BrowsedSource) -> "BrowsedSourceResponse":
        source = item.source
        return cls(
            source_id=source.source_id,
            path=source.path,
            file_type=source.file_type,
            title=source.title,
            author=source.author,
            year=source.year,
            passage_count=item.passage_count,
            indexed_at=source.indexed_at,
            content_hash=source.content_hash,
            ocr_status=item.ocr_status,
            root=item.root,
            degraded=item.degraded,
            metadata_dropped=(
                None
                if item.metadata_dropped is None
                else MetadataDroppedResponse.from_domain(item.metadata_dropped)
            ),
        )


class BrowseAggregatesResponse(BaseModel):
    """Counts over every matching source, not only the page."""

    total_sources: int
    by_file_type: dict[str, int]


class BrowseSourcesResponse(BaseModel):
    """One page of indexed sources, aggregates and the applied filters."""

    sources: tuple[BrowsedSourceResponse, ...]
    total_hits: int
    page: int
    page_size: int
    aggregates: BrowseAggregatesResponse
    applied_filters: dict[str, list[str] | bool | int | str]
    # Set only when an MCP result left sources out to stay small.
    hint: str | None = Field(default=None, exclude_if=lambda value: value is None)

    @classmethod
    def from_domain(cls, result: BrowseResult) -> "BrowseSourcesResponse":
        return cls(
            sources=tuple(
                BrowsedSourceResponse.from_domain(item) for item in result.sources
            ),
            total_hits=result.total_hits,
            page=result.page,
            page_size=result.page_size,
            aggregates=BrowseAggregatesResponse(
                total_sources=result.total_hits,
                by_file_type=dict(result.by_file_type),
            ),
            applied_filters=dict(result.applied_filters),
        )


def root_status_lines(roots: Sequence[Mapping[str, Any]] | None) -> list[str]:
    """The roots of a status report as text, when there is more than one or
    one needs attention; a single readable root says nothing new."""
    roots = roots or ()
    if len(roots) < 2 and all(
        root["configured"] and root["readable"] is not False for root in roots
    ):
        return []
    return [format_root_status(root) for root in roots]


def sync_status_lines(sync: Mapping[str, Any] | None) -> list[str]:
    """The ``sync`` section of a status report as text; nothing when sync is
    off. Stored paths are listed only when the section carries them."""
    if not sync or not sync["enabled"]:
        return []
    state = "running" if sync["running"] else "enabled, but no server is running it"
    lines = [
        f"Sync: {state}, scanning every {sync['interval_seconds']:g} s, "
        f"settling {sync['settle_seconds']:g} s"
        + (f"; last scan {sync['last_scan_at']}" if sync["last_scan_at"] else "")
    ]
    applied = sync["last_applied"]
    if applied is not None:
        lines.append(
            f"  Last applied: run {applied['run_id']} at {applied['finished_at']}: "
            f"{applied['added']} added, {applied['modified']} modified, "
            f"{applied['removed']} removed, {applied['moved']} moved, "
            f"{applied['copied']} copied, {applied['failed']} failed"
            + (" (cancelled)" if applied["cancelled"] else "")
        )
    pending = sync["pending"]
    lines.append(
        f"  Pending: {pending['added']} added, {pending['modified']} modified, "
        f"{pending['removed']} removed; {sync['unsettled']} not yet settled"
        + (
            f", {sync['unsettled_over_limit']} for over 30 minutes"
            if sync["unsettled_over_limit"]
            else ""
        )
    )
    if sync["applying"] is not None:
        lines.append(
            f"  Applying: {sync['applying']['done']} of {sync['applying']['total']}"
        )
    for held in sync["held_removals"]:
        lines.append(
            f"  Held removals in root {held['root']}: {held['count']} (the removal "
            "guard; `docgrep update --yes` applies them)"
        )
        lines.extend(f"    {path}" for path in held.get("paths", ()))
    if sync["paused_roots"]:
        lines.append("  Paused roots: " + ", ".join(sync["paused_roots"]))
    if sync["awaiting_first_update"]:
        lines.append(
            "  Awaiting a first update: " + ", ".join(sync["awaiting_first_update"])
        )
    if sync["idle_reason"]:
        lines.append(f"  Idle: {sync['idle_reason']}")
    if sync["last_error"]:
        lines.append(f"  Last error: {sync['last_error']}")
    return lines


def format_root_status(root: Mapping[str, Any]) -> str:
    """One root of the status report on one line: name, flags and counts."""
    flags = ["legacy"] if root["legacy"] else []
    if not root["configured"]:
        flags.append("not configured")
    elif root["readable"] is False:
        flags.append("not readable")
    return (
        f"Root {root['name']}"
        + (f" ({', '.join(flags)})" if flags else "")
        + f": {root['indexed']} indexed, {root['skipped']} skipped, "
        f"{root['failed']} failed"
    )


def format_filters(applied: Mapping[str, object]) -> str:
    """The applied filters on one line, as ``key=value`` pairs."""
    return "; ".join(f"{key}={value}" for key, value in applied.items())


def run_search(
    operation: SearchOperation,
    query: str,
    *,
    top_n: int | None,
    collapse_duplicates: bool | None,
    source_filter: SourceFilter,
) -> SearchResult:
    """Run a search, passing the optional arguments only when they are set,
    so every front calls the operation the same way."""
    options: dict[str, Any] = {"top_n": top_n}
    if collapse_duplicates is not None:
        options["collapse_duplicates"] = collapse_duplicates
    if source_filter.applied():
        options["source_filter"] = source_filter
    return operation.run(query, **options)


def browse_sources(
    settings: Settings, services: Callable[[], Services], request: BrowseRequest
) -> BrowseSourcesResponse:
    """Run a source listing through the shared composition root.

    A data directory without an index gives an empty listing without
    creating the composition root, which would create an empty store.
    """
    if index_missing(settings):
        return BrowseSourcesResponse.from_domain(BrowseResult.empty(request))
    operation = services().browse
    if operation is None:
        raise ValidationError("Source browsing is not available in this configuration")
    return BrowseSourcesResponse.from_domain(operation.run(request))


class PassageContextRequest(BaseModel):
    """Passage ids from search or browse results and the window, in tokens,
    applied to each side of every passage (UC-12).

    Unknown fields are rejected. The number of ids, their shape and the
    window range are checked by the use case, so every front applies the
    same limits; a missing window takes DOCGREP_CONTEXT_WINDOW_TOKENS.
    """

    model_config = ConfigDict(extra="forbid")

    passage_ids: list[str]
    window_tokens: int | None = None


class ContextPassageResponse(BaseModel):
    """The requested passage, verbatim, and its location."""

    passage_id: str
    text: str
    page_number: int | None
    heading_path: tuple[str, ...]


class ContextLocationResponse(BaseModel):
    """The page and heading path of one neighbour used in a window."""

    page_number: int | None
    heading_path: tuple[str, ...]

    @classmethod
    def from_domain(cls, item: PassageLocation) -> "ContextLocationResponse":
        return cls(page_number=item.page_number, heading_path=item.heading_path)


class WindowTokensUsedResponse(BaseModel):
    """The stored token counts of the neighbours used on each side."""

    before: int
    after: int


class PassageContextOk(BaseModel):
    """A passage with the text before and after it in its source.

    ``context_before + passage.text + context_after`` is the whole window:
    verbatim text of the neighbouring passages, their overlap removed, and
    a space or a blank line where two passages do not touch.
    """

    passage_id: str
    status: Literal["ok"] = "ok"
    passage: ContextPassageResponse
    source: SourceResponse
    context_before: str
    context_after: str
    before_locations: tuple[ContextLocationResponse, ...]
    after_locations: tuple[ContextLocationResponse, ...]
    at_source_start: bool
    at_source_end: bool
    window_tokens_used: WindowTokensUsedResponse


class PassageNotFound(BaseModel):
    """No passage with this id is in the published index."""

    passage_id: str
    status: Literal["not_found"] = "not_found"
    hint: str


class PassageSourceRemoved(BaseModel):
    """The passage exists, but its source is no longer indexed."""

    passage_id: str
    status: Literal["source_removed"] = "source_removed"
    hint: str
    # The last known path of the source.
    path: str


class PassageSourceModified(BaseModel):
    """The passage exists, but its source changed after it was indexed."""

    passage_id: str
    status: Literal["source_modified"] = "source_modified"
    hint: str
    path: str
    indexed_content_hash: str | None
    current_content_hash: str | None


PassageContextEntry = Annotated[
    PassageContextOk | PassageNotFound | PassageSourceRemoved | PassageSourceModified,
    Field(discriminator="status"),
]


def _context_entry(item: PassageContext) -> PassageContextEntry:
    if item.status == "ok" and item.passage and item.source and item.context:
        passage, context = item.passage, item.context
        return PassageContextOk(
            passage_id=item.passage_id,
            passage=ContextPassageResponse(
                passage_id=passage.passage_id,
                text=passage.text,
                page_number=passage.page_number,
                heading_path=passage.heading_path,
            ),
            source=SourceResponse.model_validate(item.source),
            context_before=context.before,
            context_after=context.after,
            before_locations=tuple(
                ContextLocationResponse.from_domain(location)
                for location in context.before_locations
            ),
            after_locations=tuple(
                ContextLocationResponse.from_domain(location)
                for location in context.after_locations
            ),
            at_source_start=context.at_source_start,
            at_source_end=context.at_source_end,
            window_tokens_used=WindowTokensUsedResponse(
                before=context.tokens_before, after=context.tokens_after
            ),
        )
    hint = item.hint or ""
    if item.status == "source_removed":
        return PassageSourceRemoved(
            passage_id=item.passage_id, hint=hint, path=item.path or ""
        )
    if item.status == "source_modified":
        return PassageSourceModified(
            passage_id=item.passage_id,
            hint=hint,
            path=item.path or "",
            indexed_content_hash=item.indexed_content_hash,
            current_content_hash=item.current_content_hash,
        )
    return PassageNotFound(passage_id=item.passage_id, hint=hint)


class PassageContextResponse(BaseModel):
    """One entry per distinct requested id, in request order, each with a
    status; failed entries never fail the request."""

    results: tuple[PassageContextEntry, ...]
    window_tokens: int
    # Set only when an MCP result left entries out to stay small.
    hint: str | None = Field(default=None, exclude_if=lambda value: value is None)

    @classmethod
    def from_domain(cls, result: ContextResult) -> "PassageContextResponse":
        return cls(
            results=tuple(_context_entry(item) for item in result.entries),
            window_tokens=result.window_tokens,
        )


def expand_passage_context(
    settings: Settings,
    services: Callable[[], Services],
    request: PassageContextRequest,
) -> PassageContextResponse:
    """Run a context request through the shared composition root.

    A data directory without an index answers every id with ``not_found``
    without creating the composition root, which would create an empty
    store.
    """
    operation: ContextOperation
    if index_missing(settings):
        operation = unindexed_context(settings)
    else:
        context = services().context
        if context is None:
            raise ValidationError(
                "Passage context is not available in this configuration"
            )
        operation = context
    return PassageContextResponse.from_domain(
        operation.run(request.passage_ids, window_tokens=request.window_tokens)
    )


class BuildFailureResponse(BaseModel):
    """A source that failed during a completed build."""

    model_config = ConfigDict(from_attributes=True)

    path: str
    reason: str
    message: str


class DiscoverScansRequest(BaseModel):
    """Optional filters of a scan discovery."""

    path_glob: str | None = None
    file_types: list[Literal["pdf", "djvu"]] | None = None
    include_already_processed: bool = False


class ProcessScansRequest(BaseModel):
    """The files to process through OCR (exactly one of paths, path_glob and
    all_discovered) and the options of the batch."""

    paths: list[str] | None = None
    path_glob: str | None = None
    all_discovered: bool = False
    languages: str | None = None
    force: bool = False


class InfoRootResponse(BaseModel):
    """A configured corpus root; ``readable`` is a cheap check of its folder."""

    name: str
    path: str
    legacy: bool
    readable: bool


class InfoResponse(BaseModel):
    """Basic service information and the configured corpus roots, in order."""

    version: str
    backend: str
    roots: list[InfoRootResponse] = Field(default_factory=list)
    # The fused first-stage candidates the reranker scores (ADR-0025) and
    # the search effort of the dense channel, hnsw_ef or "exact" (ADR-0026).
    rerank_pool: int
    dense_hnsw_ef: int | Literal["exact"]
    # Automatic sync (ADR-0029): the same section as ``get_index_status``.
    sync: dict[str, Any] = Field(default_factory=dict)


def _sync_off(settings: Settings) -> SyncStatus:
    return SyncStatus(
        enabled=settings.sync,
        interval_seconds=settings.sync_interval_seconds,
        settle_seconds=settings.sync_settle_seconds,
    )


def service_info(
    settings: Settings, sync: Callable[[], SyncStatus] | None = None
) -> InfoResponse:
    """What ``GET /info`` returns; the index is not read."""
    return InfoResponse(
        sync=(sync or (lambda: _sync_off(settings)))().to_dict(),
        version=__version__,
        backend=settings.backend,
        rerank_pool=settings.effective_rerank_pool,
        dense_hnsw_ef=settings.dense_hnsw_ef,
        roots=[
            InfoRootResponse(
                name=root.name,
                path=str(root.path),
                legacy=root.legacy,
                readable=root_readable(root.path),
            )
            for root in corpus_roots(settings)
        ],
    )


class BuildRequest(BaseModel):
    """Optional body of a build request."""

    restart: bool = False


class BuildResponse(BaseModel):
    """Synchronous build outcome."""

    run_id: str
    indexed: int
    skipped: int
    failed: int
    elapsed_seconds: float
    failures: tuple[BuildFailureResponse, ...]

    @classmethod
    def from_domain(cls, summary: BuildSummary) -> "BuildResponse":
        return cls(
            run_id=summary.run_id,
            indexed=summary.indexed,
            skipped=summary.skipped,
            failed=summary.failed,
            elapsed_seconds=summary.elapsed_seconds,
            failures=tuple(
                BuildFailureResponse.model_validate(item) for item in summary.failures
            ),
        )


def _error_status(error: DocgrepError) -> int:
    if isinstance(error, (ValidationError, ParseError)):
        return 400
    if isinstance(
        error, (IndexLockHeld, ConfigDrift, IndexIncomplete, ResumeAvailable)
    ):
        return 409
    if isinstance(
        error,
        (ModelUnavailable, StoreUnavailable, ManifestUnavailable, OcrToolsMissing),
    ):
        return 503
    return 500


_LOG = logging.getLogger(__name__)


class McpEndpoint:
    """The ``/mcp`` route, served by the MCP server that the app's lifespan
    started (see ``docgrep.api.mcp``); outside a lifespan it answers 503."""

    def __init__(self) -> None:
        self.app: ASGIApp | None = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.app is None:
            response = JSONResponse(
                status_code=503,
                content={
                    "category": "mcp_unavailable",
                    "message": "The MCP server is not running",
                    "details": {},
                },
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def _mcp_sdk_available() -> bool:
    return importlib.util.find_spec("mcp") is not None


_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})
_LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


def _error_response(
    status: int, category: str, message: str, details: dict[str, object]
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"category": category, "message": message, "details": details},
    )


def validation_problems(errors: Sequence[Any]) -> list[dict[str, object]]:
    """The location and message of each pydantic validation error, as the
    ``details.errors`` of a validation error body (HTTP and MCP)."""
    return [{"location": list(item["loc"]), "message": item["msg"]} for item in errors]


def _header_hostname(value: str) -> str | None:
    """Return the lower-cased host name of a Host header value, without port."""
    host = value.strip().lower()
    if host.startswith("["):
        end = host.find("]")
        return host[1:end] if end > 0 else None
    if host.count(":") == 1:
        host = host.split(":", 1)[0]
    return host or None


def _origin_hostname(value: str) -> str | None:
    try:
        return urlsplit(value).hostname
    except ValueError:
        return None


def create_app(
    settings: Settings | None = None, *, services: Services | None = None
) -> FastAPI:
    """Create an HTTP app with one shared composition root per process."""
    effective = settings or Settings()
    configure_logging(effective.log_level)
    initialized = services
    # The MCP tools share this app's operations, so one process holds the
    # models for REST and MCP clients alike.
    mcp_endpoint = (
        McpEndpoint() if effective.mcp_http_enabled and _mcp_sdk_available() else None
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        nonlocal initialized
        log_server_start(effective)
        warn_if_not_loopback(effective.host)
        if not effective.qdrant_url:
            count = local_passage_count(
                effective.data_dir / "qdrant", effective.qdrant_alias
            )
            if count is not None:
                warn_if_local_mode_is_large(count)
        async with AsyncExitStack() as stack:
            if effective.sync:
                stack.push_async_callback(sync_task.stop)
                sync_task.start()
            if mcp_endpoint is not None:
                await stack.enter_async_context(_serve_mcp(mcp_endpoint))
            elif effective.mcp_http_enabled:
                _LOG.warning(
                    "mcp_http_unavailable",
                    extra={"reason": "the mcp package cannot be imported"},
                )
            yield
        # Release a container created on demand so its models can be freed even
        # while the framework still caches the endpoint functions.
        if services is None:
            initialized = None

    app = FastAPI(title="docgrep", version=__version__, lifespan=lifespan)
    initialization_lock = Lock()
    sync_task = SyncTask(lambda: operations(), effective.sync_interval_seconds)

    def sync_status() -> SyncStatus:
        """The live state of sync; the server answers while the container,
        and with it sync, is still being created."""
        if sync_task.sync is not None:
            return sync_task.sync.status()
        status = _sync_off(effective)
        if effective.sync:
            return replace(status, running=True, idle_reason="starting")
        return status

    def operations() -> Services:
        nonlocal initialized
        if initialized is None:
            with initialization_lock:
                if initialized is None:
                    initialized = create_container(
                        effective, refuse_resumable_by_default=True
                    )
        return initialized

    @asynccontextmanager
    async def _serve_mcp(endpoint: McpEndpoint) -> AsyncIterator[None]:
        # Imported here, at startup: the module imports this one.
        from docgrep.api.mcp import create_mcp_server

        server = create_mcp_server(
            effective,
            services=operations,
            status=lambda: create_status(effective, sync=sync_status),
        )
        mcp_app = server.streamable_http_app()
        async with server.session_manager.run():
            endpoint.app = mcp_app
            try:
                yield
            finally:
                endpoint.app = None

    if mcp_endpoint is not None:
        app.router.routes.append(
            Route("/mcp", endpoint=mcp_endpoint, methods=["GET", "POST", "DELETE"])
        )

    allowed = frozenset(
        name.lower()
        for name in (*_LOCAL_HOSTS, effective.host, *effective.allowed_hosts)
    )

    @app.middleware("http")
    async def guard_local_access(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        host = _header_hostname(request.headers.get("host", ""))
        if host not in allowed:
            return _error_response(
                403, "host_not_allowed", "Host header is not allowed", {"host": host}
            )
        origin = request.headers.get("origin")
        if origin is not None and request.method not in _SAFE_METHODS:
            if _origin_hostname(origin) not in allowed:
                return _error_response(
                    403,
                    "origin_not_allowed",
                    "Origin is not allowed",
                    {"origin": origin},
                )
        return await call_next(request)

    @app.exception_handler(DocgrepError)
    def domain_error(_request: Request, error: DocgrepError) -> JSONResponse:
        return _error_response(
            _error_status(error), error.category, str(error), error.details
        )

    @app.exception_handler(RequestValidationError)
    def request_error(_request: Request, error: RequestValidationError) -> JSONResponse:
        return _error_response(
            400,
            "validation_error",
            "Invalid request body",
            {"errors": validation_problems(error.errors())},
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/info", response_model=InfoResponse)
    def info() -> InfoResponse:
        """The version, the backend and the configured corpus roots with their
        absolute paths; the index is not read."""
        return service_info(effective, sync_status)

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        return SearchResponse.from_domain(
            run_search(
                operations().search,
                request.query,
                top_n=request.top_n,
                collapse_duplicates=request.collapse_duplicates,
                source_filter=request.source_filter(),
            )
        )

    @app.post("/sources/browse", response_model=BrowseSourcesResponse)
    def browse(request: BrowseSourcesRequest | None = None) -> BrowseSourcesResponse:
        """Read-only (UC-08): one page of indexed sources; no lock is taken."""
        body = request or BrowseSourcesRequest()
        return browse_sources(effective, operations, body.to_domain())

    @app.post("/passages/context", response_model=PassageContextResponse)
    def passage_context(request: PassageContextRequest) -> PassageContextResponse:
        """Read-only (UC-12): passages with the text around them; no lock is
        taken and the corpus is not read."""
        return expand_passage_context(effective, operations, request)

    @app.post("/scans/discover")
    def discover_scans(request: DiscoverScansRequest | None = None) -> dict[str, Any]:
        """Read-only (UC-5): the same report as `docgrep discover-scans --json`."""
        body = request or DiscoverScansRequest()
        report = create_discover_scans(effective).run(
            path_glob=body.path_glob,
            file_types=body.file_types,
            include_already_processed=body.include_already_processed,
            registry=open_ocr_registry_view(effective),
        )
        return report.as_dict()

    @app.post("/scans/process")
    def process_scans(request: ProcessScansRequest) -> dict[str, Any]:
        """OCR (UC-6), synchronous like a build: the same report as
        `docgrep ocr --json`. `docgrep cancel` stops it."""
        report = create_process_scans(effective).run(
            OcrRequest(
                paths=tuple(request.paths or ()),
                path_glob=request.path_glob,
                all_discovered=request.all_discovered,
                languages=request.languages,
                force=request.force,
            )
        )
        return report.as_dict()

    @app.post("/index/build", response_model=BuildResponse)
    def build(request: BuildRequest | None = None) -> BuildResponse:
        build_operation = operations().build
        try:
            if request is not None and request.restart:
                summary = build_operation.run(refuse_resumable=False)
            else:
                summary = build_operation.run()
        except ResumeAvailable as error:
            raise ResumeAvailable(
                "The latest build was interrupted and can be resumed: run "
                "`docgrep index --resume` on the command line to continue it, "
                'or send {"restart": true} to start from scratch and discard '
                "its retained work",
                details=error.details,
            ) from error
        return BuildResponse.from_domain(summary)

    return app


_default_app: FastAPI | None = None
_default_app_lock = Lock()


def __getattr__(name: str) -> FastAPI:
    """Build ``app`` (for ``docgrep.api.http:app``) on first access, once.

    Importing this module never reads the settings, so the CLI can report
    invalid settings with its own exit status.
    """
    global _default_app
    if name != "app":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if _default_app is None:
        with _default_app_lock:
            if _default_app is None:
                _default_app = create_app()
    return _default_app
