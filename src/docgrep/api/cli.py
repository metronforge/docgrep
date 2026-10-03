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
    create_dry_run,
)
from docgrep.api.http import SearchResponse, create_app
from docgrep.api.log_config import configure_logging
from docgrep.api.signals import Register, graceful_cancel
from docgrep.domain.errors import DocgrepError
from docgrep.settings import Settings
from docgrep.usecases.cancel_index import CancelIndex, CancelResult
from docgrep.usecases.dry_run import DEFAULT_LIMIT, DryRun


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
    search = commands.add_parser("search", help="Find verbatim passages")
    search.add_argument("query")
    search.add_argument("--top-n", type=int, dest="top_n")
    search.add_argument("--json", action="store_true")
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
    commands.add_parser(
        "cancel", help="Ask a running index build to stop after its current batch"
    )
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
            est = entry["estimate"]
            lines.append(
                f"    estimate for {est['corpus_files']} files: "
                f"passages {est['by_file_count']['passages']} by count, "
                f"{est['by_size']['passages']} by size; "
                f"parse+chunk {est['by_file_count']['parse_chunk_seconds']} s by "
                f"count, "
                f"{est['by_size']['parse_chunk_seconds']} s by size"
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
    if result.outcome == "interrupted":
        return (
            f"Run {result.run_id} was interrupted: its process is gone and "
            "nothing holds the index lock. Its staging collection and journal "
            "are retained, so it can be resumed.\n",
            0,
        )
    return "Nothing is running.\n", 1


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
    cancel_factory: Callable[[Settings], CancelIndex] = create_cancel,
    register_signal: Register = signal.signal,
) -> int:
    """Run one CLI command and return a process exit status.

    The status is 0 on success and 1 when the command fails, including an index
    build that finished with failed sources. A build stopped by a cancel request
    exits with 2.
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

        if args.command == "cancel":
            text, status = _format_cancel(cancel_factory(settings).run())
            sys.stdout.write(text)
            return status

        services = container_factory(settings)
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

        result = services.search.run(args.query, top_n=args.top_n)
        response = SearchResponse.from_domain(result)
        if args.json:
            sys.stdout.write(response.model_dump_json() + "\n")
        else:
            if response.hint:
                sys.stdout.write(response.hint + "\n")
            for item in response.results:
                location = item.source.path
                if item.passage.page_number is not None:
                    location += f":{item.passage.page_number}"
                if item.passage.heading_path:
                    location += " — " + " > ".join(item.passage.heading_path)
                sys.stdout.write(f"{location} (score {item.score:.3f})\n")
                sys.stdout.write(item.passage.text + "\n\n")
        return 0
    except DocgrepError as error:
        sys.stderr.write(_describe(error) + "\n")
        return 1
    except SettingsValidationError as error:
        sys.stderr.write(f"invalid_settings: {error}\n")
        return 1
    except OSError as error:
        sys.stderr.write(f"io_error: {error}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
