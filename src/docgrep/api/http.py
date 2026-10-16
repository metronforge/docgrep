"""FastAPI transport for local search and full index builds."""

import importlib.util
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from threading import Lock
from typing import Any, Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from docgrep import __version__
from docgrep.adapters.qdrant import local_passage_count, warn_if_local_mode_is_large
from docgrep.api.container import (
    Services,
    create_container,
    create_discover_scans,
    create_process_scans,
    open_ocr_registry_view,
)
from docgrep.api.log_config import (
    configure_logging,
    log_server_start,
    warn_if_not_loopback,
)
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
from docgrep.domain.models import FileType, RankedPassage, SearchResult
from docgrep.settings import Settings
from docgrep.usecases.build_index import BuildSummary
from docgrep.usecases.process_scans import OcrRequest


class SearchRequest(BaseModel):
    """Validated search parameters."""

    query: str
    top_n: int | None = None
    # None applies the DOCGREP_COLLAPSE_DUPLICATES setting.
    collapse_duplicates: bool | None = None


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

    @classmethod
    def from_domain(cls, item: RankedPassage) -> "DuplicateResponse":
        return cls(
            passage_id=item.passage.passage_id,
            source_id=item.source.source_id,
            path=item.source.path,
            file_type=item.source.file_type,
            page_number=item.passage.page_number,
            heading_path=item.passage.heading_path,
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
    def from_domain(cls, item: RankedPassage) -> "RankedPassageResponse":
        return cls(
            passage=PassageResponse.model_validate(item.passage),
            source=SourceResponse.model_validate(item.source),
            score=item.score,
            dense_score=item.dense_score,
            reranker_score=item.reranker_score,
            duplicates=tuple(
                DuplicateResponse.from_domain(duplicate)
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

    @classmethod
    def from_domain(cls, result: SearchResult) -> "SearchResponse":
        return cls(
            results=tuple(
                RankedPassageResponse.from_domain(item) for item in result.results
            ),
            total_hits=result.total_hits,
            took_ms=result.took_ms,
            applied_config=dict(result.applied_config),
            rerank_skipped=result.rerank_skipped,
            rerank_skip_reason=result.rerank_skip_reason,
            hint=result.hint,
            collapsed_duplicates=result.collapsed_duplicates,
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

        server = create_mcp_server(effective, services=operations)
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

    @app.get("/info")
    def info() -> dict[str, str]:
        return {"version": __version__, "backend": effective.backend}

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        search_operation = operations().search
        # Pass the override only when given, as the CLI does.
        if request.collapse_duplicates is None:
            result = search_operation.run(request.query, top_n=request.top_n)
        else:
            result = search_operation.run(
                request.query,
                top_n=request.top_n,
                collapse_duplicates=request.collapse_duplicates,
            )
        return SearchResponse.from_domain(result)

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
