"""Read-only preview of what a build would do with the corpus (UC-19 subset).

The preview works from directory entries. Optional sampling parses and chunks a
few files to estimate passage counts and parse time; nothing is embedded,
locked or written, and no text is ever reported.
"""

import random
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from docgrep.domain.errors import ValidationError
from docgrep.domain.models import (
    FileType,
    ParsedDocument,
    ParsedSection,
    ScannedFile,
    Source,
)
from docgrep.domain.protocols import Chunker, CorpusScanner, DocumentParser

SCHEMA_VERSION = 1
MAX_LIMIT = 500
DEFAULT_LIMIT = 100
MAX_SAMPLE = 50
_EXTENSION_HISTOGRAM_SIZE = 20
_BYTES_PER_MB = 1_000_000


def _stats(sizes: Sequence[int]) -> dict[str, int]:
    count = len(sizes)
    return {
        "files": count,
        "size_bytes": sum(sizes),
        "average_size_bytes": round(sum(sizes) / count) if count else 0,
        "largest_size_bytes": max(sizes, default=0),
    }


@dataclass(frozen=True, slots=True)
class SampleTypeReport:
    """Measured numbers for the sampled files of one file type."""

    file_type: str
    sampled: tuple[str, ...]
    parsed: int
    failed: int
    requires_ocr: int
    passages: int
    sampled_bytes: int
    seconds: float
    corpus_files: int
    corpus_bytes: int

    def as_dict(self) -> dict[str, Any]:
        ok = self.parsed
        mb = self.sampled_bytes / _BYTES_PER_MB
        corpus_mb = self.corpus_bytes / _BYTES_PER_MB
        per_file_passages = self.passages / ok if ok else None
        per_mb_passages = self.passages / mb if ok and mb else None
        per_file_seconds = self.seconds / ok if ok else None
        per_mb_seconds = self.seconds / mb if ok and mb else None

        def scaled(rate: float | None, amount: float) -> float | None:
            return None if rate is None else round(rate * amount, 2)

        return {
            "type": self.file_type,
            "sampled_files": list(self.sampled),
            "parsed": self.parsed,
            "failed": self.failed,
            "requires_ocr_fraction": round(self.requires_ocr / ok, 4) if ok else None,
            "passages_per_file": _round(per_file_passages),
            "passages_per_mb": _round(per_mb_passages),
            "parse_chunk_seconds_per_file": _round(per_file_seconds, 4),
            "estimate": {
                "note": "extrapolated from the sample; not a measurement",
                "corpus_files": self.corpus_files,
                "corpus_size_bytes": self.corpus_bytes,
                "by_file_count": {
                    "passages": scaled(per_file_passages, self.corpus_files),
                    "parse_chunk_seconds": scaled(per_file_seconds, self.corpus_files),
                },
                "by_size": {
                    "passages": scaled(per_mb_passages, corpus_mb),
                    "parse_chunk_seconds": scaled(per_mb_seconds, corpus_mb),
                },
            },
        }


def _round(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(value, digits)


@dataclass(frozen=True, slots=True)
class DryRunReport:
    """The classification of a corpus and, optionally, a sampled estimate."""

    path_prefix: str | None
    files: tuple[ScannedFile, ...]
    status_by_path: Mapping[str, str]
    parser_types: frozenset[FileType]
    offset: int
    limit: int
    sample: tuple[SampleTypeReport, ...] | None = None
    sample_seed: int | None = None
    sample_requested: int | None = None

    def _included(self) -> list[ScannedFile]:
        return [f for f in self.files if f.excluded_by is None]

    def as_dict(self, *, with_files: bool = True) -> dict[str, Any]:
        """The report as plain data; the keys are the documented JSON schema."""
        included = self._included()
        excluded = [f for f in self.files if f.excluded_by is not None]
        by_type: dict[FileType, list[int]] = {}
        unsupported: list[ScannedFile] = []
        for item in included:
            if item.file_type is None:
                unsupported.append(item)
            else:
                by_type.setdefault(item.file_type, []).append(item.size)
        types = [
            {
                "type": file_type.value,
                "status": (
                    "supported"
                    if file_type in self.parser_types
                    else "recognised_without_parser"
                ),
                **_stats(sizes),
            }
            for file_type, sizes in sorted(by_type.items(), key=lambda e: e[0].value)
        ]
        extensions = Counter(item.extension or "(none)" for item in unsupported)
        top = sorted(extensions.items(), key=lambda entry: (-entry[1], entry[0]))
        indexed_sizes = [
            size
            for file_type, sizes in by_type.items()
            if file_type in self.parser_types
            for size in sizes
        ]
        pending_sizes = [
            size
            for file_type, sizes in by_type.items()
            if file_type not in self.parser_types
            for size in sizes
        ]
        patterns = Counter(item.excluded_by for item in excluded)
        report: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "scope": {"path_prefix": self.path_prefix},
            "totals": {
                "files": len(self.files),
                "size_bytes": sum(f.size for f in self.files),
                "included": len(included),
                "excluded": len(excluded),
            },
            "types": types,
            "unsupported": {
                **_stats([item.size for item in unsupported]),
                "extensions": [
                    {"extension": name, "files": count}
                    for name, count in top[:_EXTENSION_HISTOGRAM_SIZE]
                ],
            },
            "would_be_indexed": {
                "files": len(indexed_sizes),
                "size_bytes": sum(indexed_sizes),
            },
            "recognised_without_parser": {
                "files": len(pending_sizes),
                "size_bytes": sum(pending_sizes),
                "types": [
                    entry["type"]
                    for entry in types
                    if entry["status"] == "recognised_without_parser"
                ],
            },
            "excluded": {
                "files": len(excluded),
                "size_bytes": sum(f.size for f in excluded),
                "by_pattern": [
                    {"pattern": pattern, "files": count}
                    for pattern, count in sorted(
                        patterns.items(), key=lambda entry: (-entry[1], str(entry[0]))
                    )
                ],
            },
            "sample": None
            if self.sample is None
            else {
                "requested": self.sample_requested,
                "seed": self.sample_seed,
                "note": "estimates are extrapolated from the sample",
                "types": [entry.as_dict() for entry in self.sample],
            },
        }
        if with_files:
            page = self.files[self.offset : self.offset + self.limit]
            report["files"] = {
                "offset": self.offset,
                "limit": self.limit,
                "total": len(self.files),
                "items": [
                    {
                        "path": item.path,
                        "size_bytes": item.size,
                        "extension": item.extension,
                        "type": None
                        if item.file_type is None
                        else item.file_type.value,
                        "status": self.status_by_path[item.path],
                        "excluded_by": item.excluded_by,
                    }
                    for item in page
                ],
            }
        return report


class DryRun:
    """Classify the corpus the way a build would, without changing anything."""

    def __init__(
        self,
        *,
        scanner: CorpusScanner,
        parsers: Mapping[FileType, DocumentParser],
        chunker: Chunker,
    ) -> None:
        self.scanner = scanner
        self.parsers = parsers
        self.chunker = chunker

    @staticmethod
    def _validate(
        path_prefix: str | None, sample: int | None, limit: int, offset: int
    ) -> None:
        if not 1 <= limit <= MAX_LIMIT:
            raise ValidationError(
                f"limit must be between 1 and {MAX_LIMIT}",
                details={"minimum": 1, "maximum": MAX_LIMIT, "limit": limit},
            )
        if offset < 0:
            raise ValidationError(
                "offset must not be negative", details={"offset": offset}
            )
        if sample is not None and not 1 <= sample <= MAX_SAMPLE:
            raise ValidationError(
                f"sample must be between 1 and {MAX_SAMPLE}",
                details={"minimum": 1, "maximum": MAX_SAMPLE, "sample": sample},
            )
        if path_prefix is not None and (
            not path_prefix.strip()
            or path_prefix.startswith("/")
            or "\x00" in path_prefix
            or ".." in path_prefix.split("/")
        ):
            raise ValidationError(
                "path_prefix must be a relative path inside the corpus",
                details={"path_prefix": path_prefix},
            )

    def _status(self, item: ScannedFile) -> str:
        if item.excluded_by is not None:
            return "excluded"
        if item.file_type is None:
            return "unsupported"
        return (
            "indexed" if item.file_type in self.parsers else "recognised_without_parser"
        )

    def _sample(
        self, files: Sequence[ScannedFile], count: int, seed: int
    ) -> tuple[SampleTypeReport, ...]:
        pools: dict[FileType, list[ScannedFile]] = {}
        for item in files:
            if (
                item.excluded_by is None
                and item.file_type is not None
                and item.file_type in self.parsers
            ):
                pools.setdefault(item.file_type, []).append(item)
        total = sum(len(pool) for pool in pools.values())
        count = min(count, total)
        if not count:
            return ()
        # One untimed chunk so that one-off costs, such as loading a tokenizer,
        # do not count as parse time of the first sampled file.
        self.chunker.chunk(
            ParsedDocument(
                Source("warm-up", "warm-up", FileType.TXT),
                (ParsedSection("warm up the tokenizer"),),
            )
        )
        # Proportional allocation by the largest remainder, ties by type name.
        shares = {t: count * len(pool) / total for t, pool in pools.items()}
        quotas = {t: int(share) for t, share in shares.items()}
        leftover = count - sum(quotas.values())
        for file_type in sorted(
            shares, key=lambda t: (-(shares[t] - quotas[t]), t.value)
        )[:leftover]:
            quotas[file_type] += 1
        reports = []
        for file_type in sorted(pools, key=lambda t: t.value):
            if not quotas[file_type]:
                continue
            pool = pools[file_type]
            chosen = random.Random(f"{seed}:{file_type.value}").sample(
                pool, quotas[file_type]
            )
            chosen.sort(key=lambda item: item.path)
            reports.append(self._measure(file_type, chosen, pool))
        return tuple(reports)

    def _measure(
        self,
        file_type: FileType,
        chosen: Sequence[ScannedFile],
        pool: Sequence[ScannedFile],
    ) -> SampleTypeReport:
        parser = self.parsers[file_type]
        parsed = failed = ocr = passages = sampled_bytes = 0
        seconds = 0.0
        for item in chosen:
            source = Source(item.path, item.path, file_type, size=item.size)
            started = time.perf_counter()
            try:
                document = parser.parse(source)
                found = (
                    0 if document.requires_ocr else len(self.chunker.chunk(document))
                )
            except Exception:
                failed += 1
                continue
            seconds += time.perf_counter() - started
            parsed += 1
            ocr += int(document.requires_ocr)
            passages += found
            sampled_bytes += item.size
        return SampleTypeReport(
            file_type=file_type.value,
            sampled=tuple(item.path for item in chosen),
            parsed=parsed,
            failed=failed,
            requires_ocr=ocr,
            passages=passages,
            sampled_bytes=sampled_bytes,
            seconds=seconds,
            corpus_files=len(pool),
            corpus_bytes=sum(item.size for item in pool),
        )

    def run(
        self,
        *,
        path_prefix: str | None = None,
        sample: int | None = None,
        seed: int = 0,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
    ) -> DryRunReport:
        """Scan the corpus and, when asked, sample files to estimate the build."""
        self._validate(path_prefix, sample, limit, offset)
        scanned = self.scanner.scan()
        files = tuple(
            item
            for item in scanned
            if path_prefix is None or item.path.startswith(path_prefix)
        )
        return DryRunReport(
            path_prefix=path_prefix,
            files=files,
            status_by_path={item.path: self._status(item) for item in files},
            parser_types=frozenset(self.parsers),
            offset=offset,
            limit=limit,
            sample=None if sample is None else self._sample(files, sample, seed),
            sample_seed=None if sample is None else seed,
            sample_requested=sample,
        )
