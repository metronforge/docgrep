"""Console transport for indexing, search, and the local HTTP server."""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import ValidationError as SettingsValidationError

from docgrep.api.container import Services, create_container, create_dry_run
from docgrep.api.http import SearchResponse, create_app
from docgrep.api.log_config import configure_logging
from docgrep.domain.errors import DocgrepError
from docgrep.settings import Settings
from docgrep.usecases.dry_run import DEFAULT_LIMIT, DryRun


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docgrep")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("index", help="Build a full index of the configured corpus")
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


def _describe(error: DocgrepError) -> str:
    text = f"{error.category}: {error}"
    reason = error.details.get("reason")
    return f"{text} (reason: {reason})" if reason else text


def main(
    argv: Sequence[str] | None = None,
    *,
    container_factory: Callable[[Settings], Services] = create_container,
    dry_run_factory: Callable[[Settings], DryRun] = create_dry_run,
) -> int:
    """Run one CLI command and return a process exit status.

    The status is 0 on success and 1 when the command fails, including an index
    build that finished with failed sources.
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

        services = container_factory(settings)
        if args.command == "index":
            summary = services.build.run()
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
