"""Console transport for indexing, search, and the local HTTP server."""

import argparse
import errno
import functools
import json
import os
import signal
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from typing import Any

from pydantic import ValidationError as SettingsValidationError

from docgrep.api.container import (
    BrowseOperation,
    ContextOperation,
    Services,
    create_cancel,
    create_container,
    create_discover_scans,
    create_dry_run,
    create_process_scans,
    create_status,
    open_browse,
    open_context,
    open_ocr_registry_view,
)
from docgrep.api.http import (
    BrowseSourcesRequest,
    BrowseSourcesResponse,
    ContextLocationResponse,
    PassageContextOk,
    PassageContextRequest,
    PassageContextResponse,
    PassageSourceModified,
    PassageSourceRemoved,
    SearchResponse,
    create_app,
    format_filters,
    run_search,
    validation_problems,
)
from docgrep.api.log_config import configure_logging
from docgrep.api.signals import Register, graceful_cancel
from docgrep.domain.errors import DocgrepError, ValidationError
from docgrep.domain.sources import HAS_METADATA_VALUES, SORT_KEYS, SourceFilter
from docgrep.settings import Settings
from docgrep.usecases.cancel_index import CancelIndex, CancelResult
from docgrep.usecases.discover_scans import DiscoverScans
from docgrep.usecases.dry_run import DEFAULT_LIMIT, DryRun
from docgrep.usecases.inspect_status import InspectStatus
from docgrep.usecases.process_scans import (
    UPDATE_HINT,
    OcrFileResult,
    OcrPlan,
    OcrRequest,
    OcrSummary,
    ProcessScans,
)
from docgrep.usecases.update_index import UpdatePlan, UpdateSummary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docgrep")
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser(
        "index", help="Build a full index of the configured corpus"
    )
    mode = index.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Continue the latest aborted, cancelled or interrupted build",
    )
    mode.add_argument(
        "--restart",
        action="store_true",
        help="Discard an unfinished build and start from scratch",
    )
    index.add_argument(
        "--no-retry-failed",
        action="store_true",
        help="With --resume: keep sources that failed instead of trying them again",
    )
    update = commands.add_parser(
        "update",
        help="Apply the corpus changes since the last run to the index, in place",
    )
    update.add_argument(
        "--no-retry-failed",
        action="store_true",
        help="Keep sources that failed before instead of trying them again",
    )
    search = commands.add_parser(
        "search",
        help="Find verbatim passages",
        description="Filters restrict the search to the matching indexed "
        "sources before ranking. They combine with AND; repeated values of one "
        "option combine with OR. A source without the filtered metadata never "
        "matches.",
    )
    search.add_argument("query")
    search.add_argument("--top-n", type=int, dest="top_n")
    search.add_argument("--json", action="store_true")
    search.add_argument(
        "--no-collapse-duplicates",
        dest="collapse_duplicates",
        action="store_const",
        const=False,
        default=None,
        help="List passages with identical text from several files separately",
    )
    _add_filter_options(search)
    dry = commands.add_parser(
        "dry-run",
        help="Preview what a build would index, without changing anything",
    )
    dry.add_argument("--json", action="store_true")
    dry.add_argument("--path-prefix", dest="path_prefix")
    dry.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    dry.add_argument("--offset", type=int, default=0)
    dry.add_argument("--sample", type=int, default=None)
    dry.add_argument("--seed", type=int, default=0)
    discover = commands.add_parser(
        "discover-scans",
        help="List PDF and DjVu files whose text layer needs OCR (read-only)",
    )
    discover.add_argument("--path-glob", dest="path_glob")
    discover.add_argument(
        "--file-type",
        dest="file_types",
        action="append",
        choices=["pdf", "djvu"],
        help="Restrict to a file type; repeat for both (default: both)",
    )
    discover.add_argument(
        "--include-processed",
        action="store_true",
        help="Also list files already processed through OCR, and their outputs",
    )
    discover.add_argument("--json", action="store_true")
    ocr = commands.add_parser(
        "ocr",
        help="Make scanned PDF and DjVu files searchable through OCR",
        description="Name the files with exactly one of: paths (relative to "
        "the corpus root, or absolute), --path-glob or --all-discovered.",
    )
    ocr.add_argument("paths", nargs="*", help="PDF or DjVu files to process")
    ocr.add_argument("--path-glob", dest="path_glob")
    ocr.add_argument(
        "--all-discovered",
        action="store_true",
        help="Process every file scan discovery classifies as scanned",
    )
    ocr.add_argument(
        "--include-uncertain",
        action="store_true",
        help="With --all-discovered: also process mixed_or_uncertain files",
    )
    ocr.add_argument(
        "--languages", help="Tesseract languages joined with + (rus+eng+ukr)"
    )
    ocr.add_argument(
        "--force", action="store_true", help="Process files already processed again"
    )
    ocr.add_argument(
        "--plan",
        action="store_true",
        help="Print the work plan and exit without OCR and without the lock",
    )
    ocr.add_argument(
        "--override-native",
        action="store_true",
        help="With explicit paths: also process files classified as native, "
        "rebuilding their text layer",
    )
    ocr.add_argument("--json", action="store_true")
    sources = commands.add_parser(
        "sources",
        help="List the indexed sources with their metadata (read-only)",
        description="Filters combine with AND; repeated values of one option "
        "combine with OR. A source without the filtered metadata never matches.",
    )
    _add_filter_options(sources)
    sources.add_argument(
        "--has-metadata", dest="has_metadata", choices=HAS_METADATA_VALUES
    )
    sources.add_argument(
        "--dropped-metadata",
        action="store_true",
        help="Only sources whose title, author or year was dropped as junk or "
        "inherited from an OCR original; --json shows what and why",
    )
    sources.add_argument("--sort", dest="sort_by", choices=SORT_KEYS)
    sources.add_argument("--desc", action="store_true", help="Sort in descending order")
    sources.add_argument("--page", type=int)
    sources.add_argument("--page-size", dest="page_size", type=int)
    sources.add_argument(
        "--json", action="store_true", help="Print the body of POST /sources/browse"
    )
    context = commands.add_parser(
        "context",
        help="Show passages with the text around them in their source (read-only)",
        description="Each window adds whole neighbouring passages on each side "
        "while their tokens fit, and always the nearest one.",
    )
    context.add_argument(
        "passage_ids",
        nargs="+",
        metavar="PASSAGE_ID",
        help="Passage ids from search or browse results",
    )
    context.add_argument(
        "--window",
        dest="window_tokens",
        type=int,
        help="Tokens of context on each side (default DOCGREP_CONTEXT_WINDOW_TOKENS)",
    )
    context.add_argument(
        "--json", action="store_true", help="Print the body of POST /passages/context"
    )
    commands.add_parser(
        "cancel", help="Ask a running index build to stop after its current batch"
    )
    status = commands.add_parser(
        "status", help="Show the state of the index and of long operations"
    )
    status.add_argument("--json", action="store_true")
    commands.add_parser("serve", help="Start the local HTTP API")
    commands.add_parser(
        "mcp",
        help="Serve the read-only MCP tools on standard input and output",
    )
    return parser


def _add_filter_options(command: argparse.ArgumentParser) -> None:
    """The options of the shared source filter, for sources and search."""
    command.add_argument(
        "--author",
        dest="authors",
        action="append",
        help="Author substring, case-insensitive; repeatable",
    )
    command.add_argument("--year-from", dest="year_from", type=int)
    command.add_argument("--year-to", dest="year_to", type=int)
    command.add_argument(
        "--type",
        dest="file_types",
        action="append",
        help="File type (pdf, markdown, txt, epub, docx, djvu); repeatable",
    )
    command.add_argument(
        "--path-prefix",
        dest="path_prefixes",
        action="append",
        help="Path prefix relative to the corpus root, case-sensitive; repeatable",
    )
    command.add_argument(
        "--glob",
        dest="path_globs",
        action="append",
        help="Path glob, as in DOCGREP_INCLUDE_GLOBS; repeatable",
    )
    command.add_argument(
        "--source-id", dest="source_ids", action="append", help="Repeatable"
    )


def _source_filter(args: argparse.Namespace) -> SourceFilter:
    """The filter of the command line; invalid values raise ValidationError."""
    return SourceFilter(
        source_ids=tuple(args.source_ids or ()),
        authors=tuple(args.authors or ()),
        year_from=args.year_from,
        year_to=args.year_to,
        file_types=tuple(args.file_types or ()),
        path_prefixes=tuple(args.path_prefixes or ()),
        path_globs=tuple(args.path_globs or ()),
    )


def _size(value: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1000 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1000
    return f"{value:.1f} GB"


def _format_dry_run(report: dict[str, Any]) -> str:
    """Render the dry-run report as plain text; paths and counts only."""
    totals = report["totals"]
    scope = report["scope"]["path_prefix"] or "(whole corpus)"
    lines = [
        "Corpus dry run (read-only: nothing is indexed, locked or written)",
        f"Scope: {scope}",
        f"Files: {totals['files']} ({_size(totals['size_bytes'])}); "
        f"included {totals['included']}, excluded {totals['excluded']}",
        f"Would be indexed: {report['would_be_indexed']['files']} files "
        f"({_size(report['would_be_indexed']['size_bytes'])})",
        f"Recognised without a parser: {report['recognised_without_parser']['files']} "
        f"files ({_size(report['recognised_without_parser']['size_bytes'])})",
        "By file type:",
    ]
    for entry in report["types"]:
        lines.append(
            f"  {entry['type']:<10} {entry['status']:<26} {entry['files']:>6} files"
            f"  total {_size(entry['size_bytes'])}"
            f"  average {_size(entry['average_size_bytes'])}"
            f"  largest {_size(entry['largest_size_bytes'])}"
        )
    unsupported = report["unsupported"]
    lines.append(
        f"Unsupported: {unsupported['files']} files "
        f"({_size(unsupported['size_bytes'])})"
    )
    if unsupported["extensions"]:
        lines.append(
            "  most common extensions: "
            + ", ".join(
                f"{e['extension']} {e['files']}" for e in unsupported["extensions"]
            )
        )
    excluded = report["excluded"]
    lines.append(
        f"Excluded: {excluded['files']} files ({_size(excluded['size_bytes'])})"
    )
    for entry in excluded["by_pattern"]:
        lines.append(f"  {entry['pattern']}: {entry['files']}")
    sample = report["sample"]
    if sample is not None:
        lines.append(
            f"Sample (seed {sample['seed']}, requested {sample['requested']}); "
            "the figures below marked estimate are extrapolated, not measured:"
        )
        for entry in sample["types"]:
            ocr = entry["requires_ocr_fraction"]
            lines.append(
                f"  {entry['type']}: {entry['parsed']} parsed, "
                f"{entry['failed']} failed; "
                f"passages/file {entry['passages_per_file']}, "
                f"passages/MB {entry['passages_per_mb']}, "
                f"parse+chunk {entry['parse_chunk_seconds_per_file']} s/file, "
                f"requires OCR {'n/a' if ocr is None else f'{ocr:.1%}'}"
            )
            quality = entry["text_quality"]
            if quality["pages_sampled"]:
                lines.append(_format_text_quality("    text layer", quality))
            est = entry["estimate"]
            lines.append(
                f"    estimate for {est['corpus_files']} files: "
                f"passages {est['by_file_count']['passages']} by count, "
                f"{est['by_size']['passages']} by size; "
                f"parse+chunk {est['by_file_count']['parse_chunk_seconds']} s by "
                f"count, "
                f"{est['by_size']['parse_chunk_seconds']} s by size"
            )
        total = sample["text_quality"]
        if total["pages_sampled"]:
            lines.append(_format_text_quality("  Text layer, all sampled", total))
    return "\n".join(lines) + "\n"


def _fraction(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _format_text_quality(label: str, quality: dict[str, Any]) -> str:
    """Measured text layer figures: counts, fractions and file paths, no text."""
    files = quality["garbled_text_layer_files"]
    hyphens = quality["soft_hyphens_removed_per_file"]
    line = (
        f"{label}: {quality['pages_sampled']} pages sampled, "
        f"{quality['pages_garbled']} garbled "
        f"({_fraction(quality['pages_garbled_fraction'])}); "
        f"{len(files)} files would be skipped as requires_ocr for a garbled "
        f"text layer ({_fraction(quality['garbled_text_layer_fraction'])}); "
        f"soft hyphens removed per file {'n/a' if hyphens is None else hyphens}"
    )
    rules = quality["pages_by_rule"]
    line += (
        f"\n      pages per rule (weak signals drop nothing): "
        f"R1 broken characters {rules['r1_broken_characters']}, "
        f"R2 long tokens {rules['r2_long_tokens']}, "
        f"R3 few letters and digits {rules['r3_low_alnum']}"
    )
    line += "".join(f"\n      garbled text layer: {path}" for path in files)
    return line


def _format_discover_scans(report: dict[str, Any]) -> str:
    """Render scan discovery as plain text: paths, counts and sizes only."""
    lines = [
        "Scan discovery (read-only: nothing is indexed, locked or written)",
        f"Thresholds: scanned below {report['thresholds']['density_lower']:g}, "
        f"native above {report['thresholds']['density_upper']:g} characters "
        "per sampled page",
    ]
    files = report["files"]
    if files:
        lines.append("Files needing OCR or attention:")
    for entry in files:
        pages = entry["page_count"] if entry["page_count"] is not None else "?"
        lines.append(
            f"  {entry['path']}  {entry['file_type']}  {entry['classification']}"
            f"  {entry['reason'] or '-'}  {pages} pages  {_size(entry['size_bytes'])}"
            + ("  large file" if entry["large_file"] else "")
            + (f"  ({entry['hint']})" if entry["hint"] else "")
        )
    if report["already_processed"]:
        lines.append(
            f"Already processed through OCR (not listed): "
            f"{report['already_processed']}; use --include-processed to list them"
        )
    for label, counts in [("Total", report["totals"])] + [
        (name, counts) for name, counts in report["by_file_type"].items()
    ]:
        lines.append(
            f"{label}: {counts['inspected']} inspected, {counts['native']} native, "
            f"{counts['scanned']} scanned, {counts['uncertain']} uncertain, "
            f"{counts['inspection_failed']} inspection failed"
        )
    return "\n".join(lines) + "\n"


def _counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{name} {count}" for name, count in counts.items())


def _format_ocr_plan(plan: OcrPlan) -> str:
    """The work plan as text: counts first, then one line per file."""
    report = plan.as_dict()
    work = plan.to_process
    types = Counter(t.file_type.value for t in work)
    skipped = sum(report["to_skip"].values())
    failing = sum(report["to_fail"].values())
    lines = [
        f"OCR plan (languages {plan.languages}"
        + (", force" if plan.force else "")
        + f"): {report['to_process']} to process"
        + (f" ({_counts(dict(sorted(types.items())))})" if work else "")
        + f", {skipped} to skip"
        + (f" ({_counts(report['to_skip'])})" if skipped else "")
        + f", {failing} to fail"
        + (f" ({_counts(report['to_fail'])})" if failing else ""),
        f"Input: {_size(report['input_bytes'])}, {report['pages']} pages"
        + (
            f" ({report['pages_unknown']} files with an unknown page count)"
            if report["pages_unknown"]
            else ""
        )
        + (
            f"; {report['filtered']} matched files are not PDF or DjVu"
            if report["filtered"]
            else ""
        ),
    ]
    for target in plan.targets:
        pages = target.page_count if target.page_count is not None else "?"
        lines.append(
            f"  {target.status}  {target.path}  {target.file_type.value}"
            f"  {pages} pages  {_size(target.size)}"
            + (f"  {target.mode}" if target.mode else "")
        )
    return "\n".join(lines) + "\n"


def _format_ocr_file(result: OcrFileResult) -> str:
    pages = result.target.page_count if result.target.page_count is not None else "?"
    return (
        f"[{result.index}/{result.total}] {result.status} {result.target.path}, "
        f"{pages} pages, {_duration(result.elapsed_seconds)}, ETA "
        f"{_duration(result.eta_seconds)}"
        + (f": {result.message}" if result.message else "")
        + "\n"
    )


def _format_ocr_summary(summary: OcrSummary) -> str:
    report = summary.as_dict()
    totals = report["totals"]
    status = summary.status + (
        f" ({summary.abort_reason})" if summary.abort_reason else ""
    )
    lines = [
        f"OCR run {summary.run_id} {status} in {_duration(summary.elapsed_seconds)}: "
        f"{totals['processed']} processed, {totals['skipped']} skipped"
        + (
            f" ({_counts(report['skipped_by_reason'])})"
            if report["skipped_by_reason"]
            else ""
        )
        + f", {totals['failed']} failed"
        + (
            f" ({_counts(report['failed_by_reason'])})"
            if report["failed_by_reason"]
            else ""
        )
        + f", {totals['cancelled']} cancelled, {totals['not_started']} not started."
    ]
    lines.extend(
        f"  {entry['path']}: {entry['status']}"
        + (f": {entry['message']}" if entry["message"] else "")
        for entry in report["files"]
        if entry["status"].startswith("failed:")
    )
    if summary.status == "cancelled":
        lines.append(
            "Completed files stay registered; run the same command again to continue."
        )
    lines.append(UPDATE_HINT)
    return "\n".join(lines) + "\n"


def _format_cancel(result: CancelResult) -> tuple[str, int]:
    """Describe what a cancel request found, with the exit status to use."""
    if result.outcome == "requested":
        return (
            f"Cancel requested for run {result.run_id}: the operation is "
            "stopping after its current batch.\n",
            0,
        )
    if result.outcome == "already_requested":
        return (
            f"Cancel was already requested for run {result.run_id} at "
            f"{result.requested_at}: the operation is stopping.\n",
            0,
        )
    if result.outcome == "interrupted" and result.kind == "ocr":
        return (
            f"OCR run {result.run_id} was interrupted: its process is gone and "
            "nothing holds the index lock. Completed files stay registered; run "
            "`docgrep ocr` again to continue.\n",
            0,
        )
    if result.outcome == "interrupted" and result.kind == "update":
        return (
            f"Update run {result.run_id} was interrupted: its process is gone and "
            "nothing holds the index lock. What it applied stays live; run "
            "`docgrep update` again to continue.\n",
            0,
        )
    if result.outcome == "interrupted":
        return (
            f"Run {result.run_id} was interrupted: its process is gone and "
            "nothing holds the index lock. Its staging collection and journal "
            "are retained, so it can be resumed.\n",
            0,
        )
    return "Nothing is running.\n", 1


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    minutes, rest = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m {rest}s" if minutes else f"{rest}s"


def _format_status(report: dict[str, Any]) -> str:
    """Render the status report as plain text; counts, ids and timestamps only."""
    store = report["vector_store"]
    lines = [
        f"Index: {report['index_state']}",
        f"Vector store: {store['mode']}, {store['status']}"
        + (f" ({store['reason']})" if store["reason"] else ""),
    ]
    collection = report["collection"]
    if collection is not None:
        lines.append(
            f"Collection: {collection['passage_count']} passages, "
            f"model {collection['embedding_model']} "
            f"(dimension {collection['embedding_dimension']}, "
            f"{collection['embedding_dtype']}), docgrep "
            f"{collection['docgrep_version']}, completed {collection['completed_at']}"
            + (
                f", last update {collection['last_updated_at']}"
                if collection["last_updated_at"]
                else ""
            )
        )
    run = report["latest_run"]
    if run is None:
        lines.append("Latest run: none")
    elif run["kind"] == "ocr":
        totals = run["totals"]
        lines.append(
            f"Latest run: {run['run_id']} (ocr) {run['state']}, started "
            f"{run['started_at']}, finished {run['finished_at'] or '-'}, "
            + ", ".join(
                f"{totals.get(key, 0)} {key.replace('_', ' ')}"
                for key in (
                    "processed",
                    "skipped",
                    "failed",
                    "cancelled",
                    "not_started",
                )
            )
        )
    elif run["kind"] == "update":
        totals = run["totals"]
        lines.append(
            f"Latest run: {run['run_id']} (update) {run['state']}, started "
            f"{run['started_at']}, finished {run['finished_at'] or '-'}, "
            + ", ".join(
                f"{totals.get(key, 0)} {key}"
                for key in ("added", "modified", "removed", "retried", "failed")
            )
            + f", {totals.get('unchanged', 0)} unchanged"
        )
    else:
        totals = run["totals"]
        lines.append(
            f"Latest run: {run['run_id']} ({run['kind']}) {run['state']}, started "
            f"{run['started_at']}, finished {run['finished_at'] or '-'}, "
            f"{totals.get('indexed', 0)} indexed, {totals.get('skipped', 0)} "
            f"skipped, {totals.get('failed', 0)} failed of "
            f"{run['total_sources'] if run['total_sources'] is not None else '?'}"
            " sources"
            + (f", resumed from {run['resumed_from']}" if run["resumed_from"] else "")
        )
    active = report["active_operation"]
    if active is not None:
        total = active["sources_total"] if active["sources_total"] is not None else "?"
        lines.append(
            f"Active operation: {active['kind']} (run {active['run_id']}), "
            f"{active['sources_done']}/{total} sources, "
            f"{active['passages_written']} passages, elapsed "
            f"{_duration(active['elapsed_seconds'])}, ETA "
            f"{_duration(active['eta_seconds'])}"
            + (
                f", cancel requested at {active['cancel_requested_at']}"
                if active["cancel_requested_at"]
                else ""
            )
        )
    sources = report.get("sources")
    if sources is not None:
        states = sources["by_state"]
        changed = sources["parser_changed"]
        lines.append(
            f"Sources: {sources['total']} total, {states['indexed']} indexed, "
            f"{states['skipped']} skipped, {states['failed']} failed"
            + (
                f"; {changed} to re-extract for a changed parser version"
                if changed
                else ""
            )
        )
        lines.append(
            "Skipped by reason: "
            + ", ".join(
                f"{reason} {count}"
                for reason, count in sources["skipped_by_reason"].items()
            )
        )
        if sources["failed"]:
            listed = len(sources["failed"])
            lines.append(
                f"Failed sources ({listed} of {states['failed']} listed):"
                if listed < states["failed"]
                else "Failed sources:"
            )
            lines.extend(
                f"  {entry['path']}: {entry['category']}"
                + (f": {entry['reason']}" if entry["reason"] else "")
                for entry in sources["failed"]
            )
    for entry in report["drift"]:
        lines.append(
            f"Drift: {entry['field']} is {entry['indexed']} in the index and "
            f"{entry['runtime']} in the runtime configuration"
        )
    if report["warnings"]:
        lines.append("Warnings:")
        lines.extend(f"  {w['code']}: {w['message']}" for w in report["warnings"])
    return "\n".join(lines) + "\n"


# Column widths of the source table; longer values end with an ellipsis.
_SOURCE_COLUMNS = (
    ("PATH", 48),
    ("TYPE", 8),
    ("YEAR", 4),
    ("AUTHOR", 24),
    ("TITLE", 36),
)


def _cell(value: object, width: int) -> str:
    text = "-" if value is None else " ".join(str(value).split())
    if len(text) > width:
        text = text[: width - 1] + "…"
    return text.ljust(width)


def _format_sources(response: BrowseSourcesResponse) -> str:
    """Render a source listing as a table with one line per source."""
    count = len(response.sources)
    pages = -(-response.total_hits // response.page_size)
    if count:
        first = (response.page - 1) * response.page_size + 1
        summary = (
            f"Sources {first}-{first + count - 1} of {response.total_hits} "
            f"(page {response.page} of {pages})"
        )
    elif response.total_hits:
        summary = (
            f"No sources on page {response.page}; {response.total_hits} match "
            f"({pages} pages)"
        )
    else:
        summary = "No indexed sources match"
    types = response.aggregates.by_file_type
    if types:
        summary += "; " + ", ".join(f"{name} {n}" for name, n in types.items())
    lines = []
    if count:
        lines.append(
            "  ".join(_cell(name, width) for name, width in _SOURCE_COLUMNS)
            + "  PASSAGES"
        )
        for item in response.sources:
            values = (
                item.path,
                item.file_type.value,
                item.year,
                item.author,
                item.title,
            )
            passages = "?" if item.passage_count is None else str(item.passage_count)
            lines.append(
                "  ".join(
                    _cell(value, width)
                    for value, (_, width) in zip(values, _SOURCE_COLUMNS, strict=True)
                )
                + f"  {passages:>8}"
            )
    lines.append(summary)
    if any(item.degraded for item in response.sources):
        lines.append("? the passage count could not be read")
    return "\n".join(lines) + "\n"


def _sources(args: argparse.Namespace, browse: BrowseOperation) -> int:
    """Run ``docgrep sources``: 0 on success, also when nothing matches, and
    2 for invalid arguments."""
    names = (
        "source_ids",
        "authors",
        "year_from",
        "year_to",
        "file_types",
        "path_prefixes",
        "path_globs",
        "has_metadata",
        "page",
        "page_size",
        "sort_by",
    )
    body: dict[str, Any] = {
        name: getattr(args, name) for name in names if getattr(args, name) is not None
    }
    if args.desc:
        body["sort_order"] = "desc"
    if args.dropped_metadata:
        body["metadata_dropped"] = True
    try:
        request = BrowseSourcesRequest.model_validate(body)
    except SettingsValidationError as error:
        for problem in validation_problems(error.errors()):
            location = problem["location"]
            where = (
                ".".join(str(part) for part in location)
                if isinstance(location, list)
                else ""
            )
            sys.stderr.write(
                f"validation_error: {where + ': ' if where else ''}"
                f"{problem['message']}\n"
            )
        return 2
    response = BrowseSourcesResponse.from_domain(browse.run(request.to_domain()))
    if args.json:
        _emit(response.model_dump_json(indent=2) + "\n")
    else:
        _emit(_format_sources(response))
    return 0


def _pages(locations: Sequence[ContextLocationResponse]) -> str:
    pages = [item.page_number for item in locations if item.page_number is not None]
    if not pages:
        return ""
    first, last = min(pages), max(pages)
    return f", page {first}" if first == last else f", pages {first}-{last}"


def _format_context(response: PassageContextResponse) -> str:
    """Render passage contexts: the source, the text before, the passage
    between clear markers and the text after, all verbatim."""
    lines: list[str] = []
    for item in response.results:
        lines.append(f"== {item.passage_id} [{item.status}]")
        if not isinstance(item, PassageContextOk):
            if isinstance(item, PassageSourceRemoved | PassageSourceModified):
                lines.append(f"Source: {item.path}")
            if isinstance(item, PassageSourceModified):
                lines.append(
                    f"Indexed content hash: {item.indexed_content_hash}; "
                    f"current: {item.current_content_hash}"
                )
            lines.append(item.hint)
            lines.append("")
            continue
        source = item.source
        location = source.path
        if item.passage.page_number is not None:
            location += f":{item.passage.page_number}"
        if item.passage.heading_path:
            location += " — " + " > ".join(item.passage.heading_path)
        if source.title:
            location += f" ({source.title})"
        lines.append(f"Source: {location}")
        start = "start of source" if item.at_source_start else "more before"
        lines.append(
            f"-- before ({item.window_tokens_used.before} tokens"
            f"{_pages(item.before_locations)}; {start}) --"
        )
        lines.append(item.context_before.strip("\n"))
        lines.append(">>>>> passage >>>>>")
        lines.append(item.passage.text)
        lines.append("<<<<< passage <<<<<")
        end = "end of source" if item.at_source_end else "more after"
        lines.append(
            f"-- after ({item.window_tokens_used.after} tokens"
            f"{_pages(item.after_locations)}; {end}) --"
        )
        lines.append(item.context_after.strip("\n"))
        lines.append("")
    return "\n".join(lines) + "\n"


def _context(args: argparse.Namespace, context: ContextOperation) -> int:
    """Run ``docgrep context``: 0 on success, also when entries fail with a
    per-id status, and 2 for invalid arguments."""
    body: dict[str, Any] = {"passage_ids": args.passage_ids}
    if args.window_tokens is not None:
        body["window_tokens"] = args.window_tokens
    try:
        request = PassageContextRequest.model_validate(body)
        result = context.run(request.passage_ids, window_tokens=request.window_tokens)
    except ValidationError as error:
        sys.stderr.write(_describe(error) + "\n")
        malformed = error.details.get("malformed")
        if isinstance(malformed, list):
            for item in malformed:
                sys.stderr.write(f"  malformed passage id: {item}\n")
        return 2
    response = PassageContextResponse.from_domain(result)
    if args.json:
        _emit(response.model_dump_json(indent=2) + "\n")
    else:
        _emit(_format_context(response))
    return 0


def _format_plan(plan: UpdatePlan) -> str:
    if plan.changes == 0:
        return f"No changes detected ({plan.unchanged} sources unchanged).\n"
    return (
        f"Update plan: {plan.added} added, {plan.modified} modified, "
        f"{plan.removed} removed, {plan.retried} retried, "
        f"{plan.unchanged} unchanged.\n"
        f"Modified by reason: {plan.newly_supported} newly supported, "
        f"{plan.content_changed} content changed, "
        f"{plan.parser_changed} parser changed, {plan.retried} retried"
        # The OCR reasons are shown when they occur.
        + (
            f", {plan.delegated_to_ocr} delegated to OCR"
            if plan.delegated_to_ocr
            else ""
        )
        + (
            f", {plan.delegation_ended} OCR delegation ended"
            if plan.delegation_ended
            else ""
        )
        + ".\n"
    )


def _format_update(summary: UpdateSummary) -> str:
    counts = (
        f"{summary.added} added, {summary.modified} modified, "
        f"{summary.removed} removed, {summary.retried} retried, "
        f"{summary.unchanged} unchanged, {summary.skipped} skipped, "
        f"{summary.failed} failed; {summary.passages_written} passages written, "
        f"{summary.passages_removed} removed in {summary.elapsed_seconds:.2f}s"
    )
    if summary.cancelled:
        return (
            f"Update {summary.run_id} cancelled: {counts}. The changes applied "
            "so far stay live; run `docgrep update` again to continue.\n"
        )
    lines = [f"Update {summary.run_id}: {counts}\n"]
    lines.extend(
        f"  {failure.path}: {failure.reason}: {failure.message}\n"
        for failure in summary.failures
    )
    return "".join(lines)


def _format_search(response: SearchResponse) -> str:
    """Render search results as plain text with the verbatim passage text."""
    lines: list[str] = []
    if response.applied_filters:
        lines.append(f"Filters: {format_filters(response.applied_filters)}\n")
    if response.sources_excluded_for_missing_metadata:
        lines.append(
            "Sources left out for a missing author or year: "
            f"{response.sources_excluded_for_missing_metadata}\n"
        )
    if response.hint:
        lines.append(response.hint + "\n")
    for item in response.results:
        location = item.source.path
        if item.passage.page_number is not None:
            location += f":{item.passage.page_number}"
        if item.passage.heading_path:
            location += " — " + " > ".join(item.passage.heading_path)
        lines.append(f"{location} (score {item.score:.3f})\n")
        lines.append(item.passage.text + "\n")
        for duplicate in item.duplicates:
            if duplicate.page_number is not None:
                where = f" (page {duplicate.page_number})"
            elif duplicate.heading_path:
                where = f" ({' > '.join(duplicate.heading_path)})"
            else:
                where = ""
            lines.append(f"also in: {duplicate.path}{where}\n")
        lines.append("\n")
    return "".join(lines)


def _is_closed_pipe(error: OSError) -> bool:
    return isinstance(error, BrokenPipeError) or error.errno == errno.EPIPE


def _discard_stdout() -> None:
    """Send the rest of standard output, buffered or not, to the null device.

    The reader has gone, so nothing more can be delivered; pointing the file
    descriptor at the null device also keeps the final flush at interpreter
    shutdown from failing again.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, sys.stdout.fileno())
    except (OSError, ValueError):
        # Standard output without a file descriptor: replace it until exit.
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    finally:
        os.close(devnull)


def _emit(text: str) -> None:
    """Write command output; if the reader closed the pipe, drop it silently.

    A closed pipe never interrupts the running command: output that can no
    longer be delivered is lost, and the command finishes and records its
    result as it would have otherwise.
    """
    try:
        sys.stdout.write(text)
    except OSError as error:
        if not _is_closed_pipe(error):
            raise
        _discard_stdout()


def _flush_stdout() -> None:
    try:
        sys.stdout.flush()
    except OSError as error:
        if not _is_closed_pipe(error):
            raise
        _discard_stdout()


def _tolerate_closed_stdout[**P](command: Callable[P, int]) -> Callable[P, int]:
    """Flush command output at the end without failing on a closed pipe."""

    @functools.wraps(command)
    def run(*args: P.args, **kwargs: P.kwargs) -> int:
        try:
            return command(*args, **kwargs)
        finally:
            _flush_stdout()

    return run


def _plan_printer(plan: UpdatePlan) -> None:
    _emit(_format_plan(plan))
    _flush_stdout()


def _warn_user(message: str) -> None:
    sys.stderr.write(message + "\n")


def _describe(error: DocgrepError) -> str:
    text = f"{error.category}: {error}"
    reason = error.details.get("reason")
    return f"{text} (reason: {reason})" if reason else text


def _ocr(args: argparse.Namespace, ocr: ProcessScans, register: Register) -> int:
    """Run ``docgrep ocr``: 0 when nothing failed, 1 when a file failed or the
    run was cancelled or aborted, 2 for usage errors."""
    request = OcrRequest(
        paths=tuple(args.paths),
        path_glob=args.path_glob,
        all_discovered=args.all_discovered,
        include_uncertain=args.include_uncertain,
        languages=args.languages,
        force=args.force,
        override_native=args.override_native,
    )
    try:
        if args.plan:
            plan = ocr.plan(request)
            if args.json:
                _emit(json.dumps(plan.as_dict(), indent=2) + "\n")
            else:
                _emit(_format_ocr_plan(plan))
            return 0

        def show_plan(plan: OcrPlan) -> None:
            if not args.json:
                _emit(_format_ocr_plan(plan))
                _flush_stdout()

        def show_file(result: OcrFileResult) -> None:
            if not args.json:
                _emit(_format_ocr_file(result))
                _flush_stdout()

        terminate = ocr.cancel_mode == "terminate"
        with graceful_cancel(
            ocr.request_cancel,
            _warn_user,
            register=register,
            notice="Cancel requested: "
            + (
                "the running OCR is stopped and the batch ends."
                if terminate
                else "the batch ends when the current file is done."
            )
            + " Press Ctrl-C again to stop immediately.",
        ):
            summary = ocr.run(request, on_plan=show_plan, on_file=show_file)
    except ValidationError as error:
        sys.stderr.write(_describe(error) + "\n")
        return 2
    if args.json:
        _emit(json.dumps(summary.as_dict(), indent=2) + "\n")
    else:
        _emit(_format_ocr_summary(summary))
    return 0 if summary.status == "completed" and not summary.failed else 1


MCP_INSTALL_HINT = "uv pip install mcp"


def _mcp(settings: Settings) -> int:
    """Serve MCP on standard input and output; nothing else is written there."""
    try:
        from docgrep.api.mcp import run_stdio
    except ImportError as error:
        if error.name is None or error.name.split(".")[0] != "mcp":
            raise
        sys.stderr.write(
            "docgrep mcp needs the MCP Python SDK (package `mcp`); "
            f"install it with: {MCP_INSTALL_HINT}\n"
        )
        return 2
    run_stdio(settings)
    return 0


@_tolerate_closed_stdout
def main(
    argv: Sequence[str] | None = None,
    *,
    container_factory: Callable[[Settings], Services] = create_container,
    dry_run_factory: Callable[[Settings], DryRun] = create_dry_run,
    discover_factory: Callable[[Settings], DiscoverScans] = create_discover_scans,
    cancel_factory: Callable[[Settings], CancelIndex] = create_cancel,
    status_factory: Callable[[Settings], InspectStatus] = create_status,
    ocr_factory: Callable[[Settings], ProcessScans] = create_process_scans,
    browse_factory: Callable[
        [Settings], AbstractContextManager[BrowseOperation]
    ] = open_browse,
    context_factory: Callable[
        [Settings], AbstractContextManager[ContextOperation]
    ] = open_context,
    register_signal: Register = signal.signal,
) -> int:
    """Run one CLI command and return a process exit status.

    The status is 0 on success and 1 when the command fails, including an index
    build or update that finished with failed sources or was refused. A build or
    update stopped by a cancel request exits with 2.
    """
    args = _parser().parse_args(argv)
    try:
        settings = Settings()
        configure_logging(settings.log_level)
        if args.command == "serve":
            import uvicorn

            uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
            return 0

        if args.command == "mcp":
            return _mcp(settings)

        if args.command == "dry-run":
            report = dry_run_factory(settings).run(
                path_prefix=args.path_prefix,
                sample=args.sample,
                seed=args.seed,
                limit=args.limit,
                offset=args.offset,
            )
            if args.json:
                _emit(json.dumps(report.as_dict(), indent=2) + "\n")
            else:
                _emit(_format_dry_run(report.as_dict(with_files=False)))
            return 0

        if args.command == "index" and args.no_retry_failed and not args.resume:
            sys.stderr.write("usage: --no-retry-failed requires --resume\n")
            return 1

        if args.command == "discover-scans":
            try:
                discovery = discover_factory(settings).run(
                    path_glob=args.path_glob,
                    file_types=args.file_types,
                    include_already_processed=args.include_processed,
                    registry=open_ocr_registry_view(settings),
                )
            except ValidationError as error:
                sys.stderr.write(_describe(error) + "\n")
                return 2
            if args.json:
                _emit(json.dumps(discovery.as_dict(), indent=2) + "\n")
            else:
                _emit(_format_discover_scans(discovery.as_dict()))
            return 0

        if args.command == "ocr":
            return _ocr(args, ocr_factory(settings), register_signal)

        if args.command == "status":
            status_report = status_factory(settings).run().as_dict()
            if args.json:
                _emit(json.dumps(status_report, indent=2) + "\n")
            else:
                _emit(_format_status(status_report))
            return 0

        if args.command == "sources":
            with browse_factory(settings) as browse:
                return _sources(args, browse)

        if args.command == "context":
            with context_factory(settings) as context:
                return _context(args, context)

        if args.command == "cancel":
            text, status = _format_cancel(cancel_factory(settings).run())
            _emit(text)
            return status

        # Checked before the composition root, so a bad filter loads nothing.
        source_filter = (
            _source_filter(args) if args.command == "search" else SourceFilter()
        )
        services = container_factory(settings)
        if args.command == "update":
            update = services.update
            if update is None:
                sys.stderr.write("update is not available in this configuration\n")
                return 1
            with graceful_cancel(
                lambda: update.request_cancel(),
                _warn_user,
                register=register_signal,
                operation="update",
            ):
                outcome = update.run(
                    retry_failed=not args.no_retry_failed, on_plan=_plan_printer
                )
            _emit(_format_update(outcome))
            if outcome.cancelled:
                return 2
            return 1 if outcome.failed else 0

        if args.command == "index":
            with graceful_cancel(
                lambda: services.build.request_cancel(),
                _warn_user,
                register=register_signal,
            ):
                if args.resume:
                    summary = services.build.resume(
                        retry_failed=not args.no_retry_failed
                    )
                else:
                    summary = services.build.run(refuse_resumable=not args.restart)
            if summary.cancelled:
                _emit(
                    f"Build {summary.run_id} cancelled: {summary.indexed} indexed, "
                    f"{summary.skipped} skipped, {summary.failed} failed "
                    f"in {summary.elapsed_seconds:.2f}s. The live index is "
                    "unchanged; the staging collection and journal are kept. "
                    "Continue with `docgrep index --resume`.\n"
                )
                return 2
            if summary.recovered:
                _emit(
                    f"Run {summary.run_id} had published its index but not "
                    f"committed its sources; finished it: {summary.indexed} "
                    f"indexed, {summary.skipped} skipped, {summary.failed} failed.\n"
                )
                return 1 if summary.failed else 0
            if summary.resumed_from is not None:
                _emit(
                    f"Resumed run {summary.resumed_from} as run {summary.run_id}: "
                    f"{summary.carried_over} sources carried over, "
                    f"{summary.processed} processed ({summary.retried} retried), "
                    f"{summary.removed} removed.\n"
                )
            _emit(
                f"Build {summary.run_id}: {summary.indexed} indexed, "
                f"{summary.skipped} skipped, {summary.failed} failed "
                f"in {summary.elapsed_seconds:.2f}s\n"
            )
            for failure in summary.failures:
                _emit(f"  {failure.path}: {failure.reason}: {failure.message}\n")
            return 1 if summary.failed else 0

        result = run_search(
            services.search,
            args.query,
            top_n=args.top_n,
            collapse_duplicates=args.collapse_duplicates,
            source_filter=source_filter,
        )
        response = SearchResponse.from_domain(result)
        if args.json:
            _emit(response.model_dump_json() + "\n")
        else:
            _emit(_format_search(response))
        return 0
    except DocgrepError as error:
        sys.stderr.write(_describe(error) + "\n")
        return 1
    except SettingsValidationError as error:
        sys.stderr.write(f"invalid_settings: {error}\n")
        # Scan discovery, OCR, MCP, source browsing and passage context report
        # configuration errors with 2, like usage errors.
        return (
            2
            if args.command in ("discover-scans", "ocr", "mcp", "sources", "context")
            else 1
        )
    except OSError as error:
        sys.stderr.write(f"io_error: {error}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
