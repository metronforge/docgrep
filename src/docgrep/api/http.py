"""FastAPI transport for local search and full index builds."""

from threading import Lock

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from docgrep import __version__
from docgrep.api.container import Services, create_container
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


def create_app(
    settings: Settings | None = None, *, services: Services | None = None
) -> FastAPI:
    """Create an HTTP app with one shared composition root per process."""
    effective = settings or Settings()
    app = FastAPI(title="docgrep", version=__version__)
    initialized = services
    initialization_lock = Lock()

    def operations() -> Services:
        nonlocal initialized
        if initialized is None:
            with initialization_lock:
                if initialized is None:
                    initialized = create_container(effective)
        return initialized

    @app.exception_handler(DocgrepError)
    def domain_error(_request: Request, error: DocgrepError) -> JSONResponse:
        return JSONResponse(
            status_code=_error_status(error),
            content={
                "category": error.category,
                "message": str(error),
                "details": error.details,
            },
        )

    @app.exception_handler(RequestValidationError)
    def request_error(_request: Request, error: RequestValidationError) -> JSONResponse:
        problems = [
            {"location": list(item["loc"]), "message": item["msg"]}
            for item in error.errors()
        ]
        return JSONResponse(
            status_code=400,
            content={
                "category": "validation_error",
                "message": "Invalid request body",
                "details": {"errors": problems},
            },
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
