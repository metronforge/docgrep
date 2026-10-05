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


class IndexIncomplete(DocgrepError):
    """The passage index has no valid completion marker."""

    category = "index_incomplete"


class IndexLockHeld(DocgrepError):
    """Another index operation owns the exclusive write lock."""

    category = "index_lock_held"


class BuildFailed(DocgrepError):
    """An unexpected indexing failure aborted the current build."""

    category = "build_failed"


class ResumeAvailable(DocgrepError):
    """The latest run can be resumed, so a plain build would discard its work."""

    category = "resume_available"


class NothingToResume(DocgrepError):
    """There is no run that can be resumed."""

    category = "nothing_to_resume"


class StagingInconsistent(DocgrepError):
    """The staging collection of a run cannot be continued safely."""

    category = "staging_inconsistent"


class InspectionFailed(DocgrepError):
    """A file cannot be inspected; ``details["reason"]`` says why."""

    category = "inspection_failed"


class CorpusDrift(DocgrepError):
    """The corpus changed too much since the run for a resume to make sense."""

    category = "corpus_drift"
