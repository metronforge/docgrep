"""FastAPI transport for local search and full index builds."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from threading import Lock
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict

from docgrep import __version__
from docgrep.adapters.qdrant import local_passage_count, warn_if_local_mode_is_large
from docgrep.api.container import Services, create_container
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
    ParseError,
    StoreUnavailable,
    ValidationError,
)
from docgrep.domain.models import FileType, RankedPassage, SearchResult
from docgrep.settings import Settings
from docgrep.usecases.build_index import BuildSummary


class SearchRequest(BaseModel):
    """Validated search parameters."""

    query: str
    top_n: int | None = None


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


class RankedPassageResponse(BaseModel):
    """One passage with source metadata and retrieval scores."""

    passage: PassageResponse
    source: SourceResponse
    score: float
    dense_score: float
    reranker_score: float | None

    @classmethod
    def from_domain(cls, item: RankedPassage) -> "RankedPassageResponse":
        return cls(
            passage=PassageResponse.model_validate(item.passage),
            source=SourceResponse.model_validate(item.source),
            score=item.score,
            dense_score=item.dense_score,
            reranker_score=item.reranker_score,
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
        )


class BuildFailureResponse(BaseModel):
    """A source that failed during a completed build."""

    model_config = ConfigDict(from_attributes=True)

    path: str
    reason: str
    message: str


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
    if isinstance(error, (IndexLockHeld, ConfigDrift, IndexIncomplete)):
        return 409
    if isinstance(error, (ModelUnavailable, StoreUnavailable, ManifestUnavailable)):
        return 503
    return 500


_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})
_LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


def _error_response(
    status: int, category: str, message: str, details: dict[str, object]
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"category": category, "message": message, "details": details},
    )


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
                    initialized = create_container(effective)
        return initialized

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
        problems = [
            {"location": list(item["loc"]), "message": item["msg"]}
            for item in error.errors()
        ]
        return _error_response(
            400,
            "validation_error",
            "Invalid request body",
            {"errors": problems},
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/info")
    def info() -> dict[str, str]:
        return {"version": __version__, "backend": effective.backend}

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        result = operations().search.run(request.query, top_n=request.top_n)
        return SearchResponse.from_domain(result)

    @app.post("/index/build", response_model=BuildResponse)
    def build() -> BuildResponse:
        summary = operations().build.run()
        return BuildResponse.from_domain(summary)

    return app


app = create_app()
