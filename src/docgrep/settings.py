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
    top_k: int = Field(default=100, gt=0)
    top_n: int = Field(default=10, gt=0)
    top_n_max: int = Field(default=50, gt=0)
    max_query_len: int = Field(default=1000, gt=0)

    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    device: Literal["cuda", "cpu", "auto"] = "auto"
    backend: Literal["real", "fake"] = "real"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    # Extra host names accepted in the Host and Origin headers, as a
    # comma-separated list in the environment (for container deployments).
    allowed_hosts: Annotated[tuple[str, ...], NoDecode] = ()

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
        return self
