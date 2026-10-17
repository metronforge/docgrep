"""Environment-backed configuration for the local service."""

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, PrivateAttr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from docgrep.domain.ocr import parse_languages
from docgrep.domain.roots import CorpusRoots
from docgrep.domain.sources import MAX_PAGE_SIZE

# The MCP tools return this many passages by default, and at most
# DOCGREP_MCP_MAX_TOP_N (default MCP_MAX_TOP_N), within DOCGREP_TOP_N_MAX.
MCP_DEFAULT_TOP_N = 10
MCP_MAX_TOP_N = 25
# The browse_sources tool returns at most DOCGREP_MCP_MAX_PAGE_SIZE sources
# per page (default MCP_MAX_PAGE_SIZE), never more than MAX_PAGE_SIZE.
MCP_MAX_PAGE_SIZE = 100
# The expand_passage_context tool takes at most DOCGREP_MCP_MAX_CONTEXT_PASSAGES
# ids per call (default MCP_MAX_CONTEXT_PASSAGES), within
# DOCGREP_CONTEXT_MAX_PASSAGES.
MCP_MAX_CONTEXT_PASSAGES = 5


class Settings(BaseSettings):
    """Configuration loaded from ``DOCGREP_`` environment variables."""

    model_config = SettingsConfigDict(env_prefix="DOCGREP_")

    # Deprecated: the corpus roots live in <data_dir>/roots.json (ADR-0023).
    # Without that file a corpus_root that is set is one legacy root.
    corpus_root: Path = Path(".")
    data_dir: Path = Path(".docgrep")
    include_globs: tuple[str, ...] = ("**/*",)
    exclude_globs: tuple[str, ...] = ()

    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=64, ge=0)
    upsert_batch_size: int = Field(default=256, gt=0)
    top_k: int = Field(default=100, gt=0)
    top_n: int = Field(default=10, gt=0)
    top_n_max: int = Field(default=50, gt=0)
    max_query_len: int = Field(default=1000, gt=0)
    # Collapse results whose text is identical after whitespace normalization
    # into one result that lists the other locations; a request may override it.
    collapse_duplicates: bool = True

    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    device: Literal["cuda", "cpu", "auto"] = "auto"
    # Embedding weight precision: auto means float16 on CUDA, float32 otherwise.
    embedding_dtype: Literal["auto", "float16", "float32"] = "auto"
    # Reranker weight precision: auto means float16 on CUDA, float32 otherwise.
    reranker_dtype: Literal["auto", "float16", "float32"] = "auto"
    backend: Literal["real", "fake"] = "real"
    # URL of a Qdrant server; without it the index lives in local mode under
    # the data directory.
    qdrant_url: str | None = None
    # Name of the read alias; give each corpus on a shared server its own.
    qdrant_alias: str = "docgrep"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    # Scan discovery: mean non-whitespace characters per sampled page below
    # which a PDF or DjVu file counts as scanned, and above which as native.
    scan_density_lower: float = Field(default=100.0, ge=0)
    scan_density_upper: float = Field(default=500.0, gt=0)
    # OCR (`docgrep ocr`): Tesseract languages joined with "+", the time limit
    # per file in seconds (0: none), the number of cores ocrmypdf may use
    # (unset: its default) and whether a cancel stops the running file
    # (terminate) or lets it complete (finish).
    ocr_languages: str = "rus+eng+ukr"
    ocr_timeout_seconds: float = Field(default=0, ge=0)
    ocr_jobs: int | None = Field(default=None, gt=0)
    ocr_cancel_mode: Literal["terminate", "finish"] = "terminate"
    # Extra host names accepted in the Host and Origin headers, as a
    # comma-separated list in the environment (for container deployments).
    allowed_hosts: Annotated[tuple[str, ...], NoDecode] = ()
    # MCP: serve the read-only tools over streamable HTTP at /mcp of
    # `docgrep serve`, and the largest number of passages one call may ask for.
    mcp_http_enabled: bool = True
    mcp_max_top_n: int = Field(default=MCP_MAX_TOP_N, gt=0)
    mcp_max_page_size: int = Field(default=MCP_MAX_PAGE_SIZE, gt=0, le=MAX_PAGE_SIZE)
    mcp_max_context_passages: int = Field(default=MCP_MAX_CONTEXT_PASSAGES, gt=0)
    # Passage context (UC-12): the most passage ids per request, and the
    # default and largest window in tokens, applied to each side.
    context_max_passages: int = Field(default=20, gt=0)
    context_window_tokens: int = Field(default=500, ge=0)
    context_max_window_tokens: int = Field(default=5000, gt=0)

    # The roots this process uses, once chosen (the roots file, the
    # deprecated corpus_root or --root parameters); never read from the
    # environment.
    _corpus_roots: CorpusRoots | None = PrivateAttr(default=None)

    @property
    def chosen_corpus_roots(self) -> CorpusRoots | None:
        return self._corpus_roots

    def with_corpus_roots(self, roots: CorpusRoots) -> Self:
        """A copy of these settings that uses the given roots."""
        chosen = self.model_copy()
        chosen._corpus_roots = roots
        return chosen

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("ocr_languages")
    @classmethod
    def check_ocr_languages(cls, value: str) -> str:
        parse_languages(value)
        return value.strip()

    @field_validator("allowed_hosts", mode="before")
    @classmethod
    def split_allowed_hosts(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        return value

    @model_validator(mode="after")
    def validate_limits(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.top_n > self.top_n_max:
            raise ValueError("top_n must not exceed top_n_max")
        if self.top_n > self.top_k:
            raise ValueError("top_n must not exceed top_k")
        if self.context_window_tokens > self.context_max_window_tokens:
            raise ValueError(
                "context_window_tokens must not exceed context_max_window_tokens"
            )
        if self.scan_density_lower >= self.scan_density_upper:
            raise ValueError("scan_density_lower must be below scan_density_upper")
        return self
