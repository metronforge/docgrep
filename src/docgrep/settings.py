"""Environment-backed configuration for the local service."""

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration loaded from ``DOCGREP_`` environment variables."""

    model_config = SettingsConfigDict(env_prefix="DOCGREP_")

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
    # Extra host names accepted in the Host and Origin headers, as a
    # comma-separated list in the environment (for container deployments).
    allowed_hosts: Annotated[tuple[str, ...], NoDecode] = ()

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

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
        if self.scan_density_lower >= self.scan_density_upper:
            raise ValueError("scan_density_lower must be below scan_density_upper")
        return self
