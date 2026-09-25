"""Structured failures emitted by domain and use-case operations."""

from collections.abc import Mapping
from typing import ClassVar


class DocgrepError(Exception):
    """Base error with a stable category and machine-readable details."""

    category: ClassVar[str] = "docgrep_error"

    def __init__(
        self, message: str, *, details: Mapping[str, object] | None = None
    ) -> None:
        super().__init__(message)
        self.details = dict(details or {})


class ValidationError(DocgrepError):
    """The request or configured input is invalid."""

    category = "validation_error"


class ParseError(DocgrepError):
    """A source cannot be decoded or parsed."""

    category = "parse_error"


class ModelUnavailable(DocgrepError):
    """An embedding or reranking model cannot run."""

    category = "model_unavailable"


class StoreUnavailable(DocgrepError):
    """The passage store cannot be used."""

    category = "store_unavailable"


class ManifestUnavailable(DocgrepError):
    """The persistent indexing manifest cannot be used."""

    category = "manifest_unavailable"


class ConfigDrift(DocgrepError):
    """Runtime settings disagree with the indexed configuration."""

    category = "config_drift"


class IndexLockHeld(DocgrepError):
    """Another index operation owns the exclusive write lock."""

    category = "index_lock_held"


class BuildFailed(DocgrepError):
    """An unexpected indexing failure aborted the current build."""

    category = "build_failed"
