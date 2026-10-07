"""Find the PDF and DjVu files whose text layer is missing or unusable (UC-5).

Read-only: the corpus is scanned from directory entries with the build's
include and exclude patterns, a few pages of each file are sampled through a
ScanProbe, and nothing is written, locked, indexed or embedded. A file that
cannot be inspected is reported, never raised.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from docgrep.domain.errors import InspectionFailed, ValidationError
from docgrep.domain.globs import matching_pattern
from docgrep.domain.models import FileType
from docgrep.domain.protocols import CorpusScanner, ScanProbe
from docgrep.domain.scans import (
    INSPECTION_FAILED,
    LARGE_FILE_BYTES,
    LARGE_FILE_PAGES,
    MIXED_OR_UNCERTAIN,
    NATIVE,
    NOT_PROCESSED,
    SCANNED,
    UNSAMPLABLE,
    classify,
    sample_pages,
    text_density,
)

_LOG = logging.getLogger(__name__)

SCHEMA_VERSION = 1
FILE_TYPES = {"pdf": FileType.PDF, "djvu": FileType.DJVU}
# Failure reasons when an unexpected error escapes a probe.
_FALLBACK_REASON = {FileType.PDF: "corrupt_header", FileType.DJVU: "djvu_invalid"}
_COUNTED = (NATIVE, SCANNED, MIXED_OR_UNCERTAIN, INSPECTION_FAILED)
_COUNT_KEYS = {
    NATIVE: "native",
    SCANNED: "scanned",
    MIXED_OR_UNCERTAIN: "uncertain",
    INSPECTION_FAILED: "inspection_failed",
}


@dataclass(frozen=True, slots=True)
class ScanEntry:
    """One inspected file; counts, sizes and page numbers only."""

    path: str
    file_type: str
    size_bytes: int
    page_count: int | None
    classification: str
    reason: str | None
    hint: str | None
    text_density: float | None
    sampled_pages: tuple[int, ...]

    @property
    def large_file(self) -> bool:
        return self.size_bytes > LARGE_FILE_BYTES or (
            self.page_count is not None and self.page_count > LARGE_FILE_PAGES
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "file_type": self.file_type,
            "size_bytes": self.size_bytes,
            "page_count": self.page_count,
            "classification": self.classification,
            "reason": self.reason,
            "hint": self.hint,
            "ocr_status": NOT_PROCESSED,
            "large_file": self.large_file,
            "text_density": self.text_density,
            "sampled_pages": list(self.sampled_pages),
        }


def _counts(entries: Sequence[ScanEntry]) -> dict[str, int]:
    counts = {"inspected": len(entries)}
    for classification in _COUNTED:
        counts[_COUNT_KEYS[classification]] = sum(
            e.classification == classification for e in entries
        )
    return counts


@dataclass(frozen=True, slots=True)
class ScanDiscoveryReport:
    path_glob: str | None
    file_types: tuple[str, ...]
    density_lower: float
    density_upper: float
    entries: tuple[ScanEntry, ...]

    def as_dict(self) -> dict[str, Any]:
        """The JSON form; its key names are stable and documented."""
        listed = [e for e in self.entries if e.classification != NATIVE]
        return {
            "schema_version": SCHEMA_VERSION,
            "filters": {
                "path_glob": self.path_glob,
                "file_types": list(self.file_types),
            },
            "thresholds": {
                "density_lower": self.density_lower,
                "density_upper": self.density_upper,
            },
            "files": [e.as_dict() for e in listed],
            "totals": _counts(self.entries),
            "by_file_type": {
                name: _counts([e for e in self.entries if e.file_type == name])
                for name in self.file_types
            },
        }


class DiscoverScans:
    """Classify PDF and DjVu files as native, scanned or uncertain."""

    def __init__(
        self,
        *,
        scanner: CorpusScanner,
        probe: ScanProbe,
        density_lower: float,
        density_upper: float,
    ) -> None:
        if not 0 <= density_lower < density_upper:
            raise ValidationError(
                "the lower text density threshold must be below the upper one",
                details={"lower": density_lower, "upper": density_upper},
            )
        self.scanner = scanner
        self.probe = probe
        self.density_lower = density_lower
        self.density_upper = density_upper

    def run(
        self,
        *,
        path_glob: str | None = None,
        file_types: Sequence[str] | None = None,
    ) -> ScanDiscoveryReport:
        """Inspect the matching files; raises only for invalid filters or an
        inaccessible corpus."""
        types = self._types(file_types)
        if path_glob is not None and not path_glob.strip():
            raise ValidationError(
                "path_glob must not be empty", details={"path_glob": path_glob}
            )
        wanted = {FILE_TYPES[name] for name in types}
        entries = []
        for item in self.scanner.scan():
            if item.excluded_by is not None or item.file_type not in wanted:
                continue
            if (
                path_glob is not None
                and matching_pattern(item.path, [path_glob]) is None
            ):
                continue
            assert item.file_type is not None
            entries.append(self._inspect(item.path, item.file_type, item.size))
        entries.sort(key=lambda e: (e.file_type, e.path))
        return ScanDiscoveryReport(
            path_glob=path_glob,
            file_types=types,
            density_lower=self.density_lower,
            density_upper=self.density_upper,
            entries=tuple(entries),
        )

    @staticmethod
    def _types(file_types: Sequence[str] | None) -> tuple[str, ...]:
        if file_types is None or not file_types:
            return tuple(sorted(FILE_TYPES))
        unknown = sorted(set(file_types) - set(FILE_TYPES))
        if unknown:
            raise ValidationError(
                "file_types may only contain pdf and djvu",
                details={"unknown": ", ".join(unknown)},
            )
        return tuple(sorted(set(file_types)))

    def inspect(self, path: str, file_type: FileType, size: int) -> ScanEntry:
        """Classify one file exactly as a discovery does; never raises for the
        file itself (a failure is an ``inspection_failed`` entry)."""
        return self._inspect(path, file_type, size)

    def _inspect(self, path: str, file_type: FileType, size: int) -> ScanEntry:
        def failed(
            reason: str, count: int | None, hint: str | None = None
        ) -> ScanEntry:
            return ScanEntry(
                path,
                file_type.value,
                size,
                count,
                INSPECTION_FAILED,
                reason,
                hint,
                None,
                (),
            )

        count: int | None = None
        try:
            count = self.probe.page_count(path, file_type)
            pages = sample_pages(path, count)
            found = self.probe.sample(path, file_type, pages) if pages else ()
        except InspectionFailed as error:
            hint = error.details.get("hint")
            return failed(
                str(error.details.get("reason", _FALLBACK_REASON[file_type])),
                count,
                None if hint is None else str(hint),
            )
        except Exception:
            _LOG.warning("scan_inspection_failed", extra={"file_type": file_type.value})
            return failed(_FALLBACK_REASON[file_type], count)
        samples = [s for s in found if s is not None]
        if not samples:
            return failed(UNSAMPLABLE, count)
        classification, reason = classify(
            samples, lower=self.density_lower, upper=self.density_upper
        )
        return ScanEntry(
            path,
            file_type.value,
            size,
            count,
            classification,
            reason,
            None,
            round(text_density(samples), 1),
            tuple(s.page_number for s in samples),
        )
