"""Console transport for indexing, search, and the local HTTP server."""

import argparse
import json
import signal
import sys
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import ValidationError as SettingsValidationError

from docgrep.api.container import (
    Services,
    create_cancel,
    create_container,
    create_discover_scans,
    create_dry_run,
    create_status,
)
from docgrep.api.http import SearchResponse, create_app
from docgrep.api.log_config import configure_logging
from docgrep.api.signals import Register, graceful_cancel
from docgrep.domain.errors import DocgrepError, ValidationError
from docgrep.settings import Settings
from docgrep.usecases.cancel_index import CancelIndex, CancelResult
from docgrep.usecases.discover_scans import DiscoverScans
from docgrep.usecases.dry_run import DEFAULT_LIMIT, DryRun
from docgrep.usecases.inspect_status import InspectStatus
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
    search = commands.add_parser("search", help="Find verbatim passages")
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
    discover.add_argument("--json", action="store_true")
    commands.add_parser(
        "cancel", help="Ask a running index build to stop after its current batch"
    )
    status = commands.add_parser(
        "status", help="Show the state of the index and of long operations"
    )
    status.add_argument("--json", action="store_true")
    commands.add_parser("serve", help="Start the local HTTP API")
    return parser


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
    for label, counts in [("Total", report["totals"])] + [
        (name, counts) for name, counts in report["by_file_type"].items()
    ]:
        lines.append(
            f"{label}: {counts['inspected']} inspected, {counts['native']} native, "
            f"{counts['scanned']} scanned, {counts['uncertain']} uncertain, "
            f"{counts['inspection_failed']} inspection failed"
        )
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


def _format_plan(plan: UpdatePlan) -> str:
    if plan.changes == 0:
        return f"No changes detected ({plan.unchanged} sources unchanged).\n"
    return (
        f"Update plan: {plan.added} added, {plan.modified} modified, "
        f"{plan.removed} removed, {plan.retried} retried, "
        f"{plan.unchanged} unchanged.\n"
        f"Modified by reason: {plan.newly_supported} newly supported, "
        f"{plan.content_changed} content changed, "
        f"{plan.parser_changed} parser changed, {plan.retried} retried.\n"
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


def _plan_printer(plan: UpdatePlan) -> None:
    sys.stdout.write(_format_plan(plan))
    sys.stdout.flush()


def _warn_user(message: str) -> None:
    sys.stderr.write(message + "\n")


def _describe(error: DocgrepError) -> str:
    text = f"{error.category}: {error}"
    reason = error.details.get("reason")
    return f"{text} (reason: {reason})" if reason else text


def main(
    argv: Sequence[str] | None = None,
    *,
    container_factory: Callable[[Settings], Services] = create_container,
    dry_run_factory: Callable[[Settings], DryRun] = create_dry_run,
    discover_factory: Callable[[Settings], DiscoverScans] = create_discover_scans,
    cancel_factory: Callable[[Settings], CancelIndex] = create_cancel,
    status_factory: Callable[[Settings], InspectStatus] = create_status,
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

        if args.command == "dry-run":
            report = dry_run_factory(settings).run(
                path_prefix=args.path_prefix,
                sample=args.sample,
                seed=args.seed,
                limit=args.limit,
                offset=args.offset,
            )
            if args.json:
                sys.stdout.write(json.dumps(report.as_dict(), indent=2) + "\n")
            else:
                sys.stdout.write(_format_dry_run(report.as_dict(with_files=False)))
            return 0

        if args.command == "index" and args.no_retry_failed and not args.resume:
            sys.stderr.write("usage: --no-retry-failed requires --resume\n")
            return 1

        if args.command == "discover-scans":
            try:
                discovery = discover_factory(settings).run(
                    path_glob=args.path_glob, file_types=args.file_types
                )
            except ValidationError as error:
                sys.stderr.write(_describe(error) + "\n")
                return 2
            if args.json:
                sys.stdout.write(json.dumps(discovery.as_dict(), indent=2) + "\n")
            else:
                sys.stdout.write(_format_discover_scans(discovery.as_dict()))
            return 0

        if args.command == "status":
            status_report = status_factory(settings).run().as_dict()
            if args.json:
                sys.stdout.write(json.dumps(status_report, indent=2) + "\n")
            else:
                sys.stdout.write(_format_status(status_report))
            return 0

        if args.command == "cancel":
            text, status = _format_cancel(cancel_factory(settings).run())
            sys.stdout.write(text)
            return status

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
            sys.stdout.write(_format_update(outcome))
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
                sys.stdout.write(
                    f"Build {summary.run_id} cancelled: {summary.indexed} indexed, "
                    f"{summary.skipped} skipped, {summary.failed} failed "
                    f"in {summary.elapsed_seconds:.2f}s. The live index is "
                    "unchanged; the staging collection and journal are kept. "
                    "Continue with `docgrep index --resume`.\n"
                )
                return 2
            if summary.recovered:
                sys.stdout.write(
                    f"Run {summary.run_id} had published its index but not "
                    f"committed its sources; finished it: {summary.indexed} "
                    f"indexed, {summary.skipped} skipped, {summary.failed} failed.\n"
                )
                return 1 if summary.failed else 0
            if summary.resumed_from is not None:
                sys.stdout.write(
                    f"Resumed run {summary.resumed_from} as run {summary.run_id}: "
                    f"{summary.carried_over} sources carried over, "
                    f"{summary.processed} processed ({summary.retried} retried), "
                    f"{summary.removed} removed.\n"
                )
            sys.stdout.write(
                f"Build {summary.run_id}: {summary.indexed} indexed, "
                f"{summary.skipped} skipped, {summary.failed} failed "
                f"in {summary.elapsed_seconds:.2f}s\n"
            )
            for failure in summary.failures:
                sys.stdout.write(
                    f"  {failure.path}: {failure.reason}: {failure.message}\n"
                )
            return 1 if summary.failed else 0

        if args.collapse_duplicates is None:
            result = services.search.run(args.query, top_n=args.top_n)
        else:
            result = services.search.run(
                args.query,
                top_n=args.top_n,
                collapse_duplicates=args.collapse_duplicates,
            )
        response = SearchResponse.from_domain(result)
        if args.json:
            sys.stdout.write(response.model_dump_json() + "\n")
        else:
            sys.stdout.write(_format_search(response))
        return 0
    except DocgrepError as error:
        sys.stderr.write(_describe(error) + "\n")
        return 1
    except SettingsValidationError as error:
        sys.stderr.write(f"invalid_settings: {error}\n")
        # Scan discovery reports configuration errors with 2, like usage errors.
        return 2 if args.command == "discover-scans" else 1
    except OSError as error:
        sys.stderr.write(f"io_error: {error}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
