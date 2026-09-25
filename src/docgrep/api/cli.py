"""Console transport for indexing, search, and the local HTTP server."""

import argparse
import sys
from collections.abc import Sequence

from pydantic import ValidationError as SettingsValidationError

from docgrep.api.container import create_container
from docgrep.api.http import SearchResponse, create_app
from docgrep.domain.errors import DocgrepError
from docgrep.settings import Settings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docgrep")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("index", help="Build a full index of the configured corpus")
    search = commands.add_parser("search", help="Find verbatim passages")
    search.add_argument("query")
    search.add_argument("--top-n", type=int, dest="top_n")
    search.add_argument("--json", action="store_true")
    commands.add_parser("serve", help="Start the local HTTP API")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one CLI command and return a process exit status."""
    args = _parser().parse_args(argv)
    try:
        settings = Settings()
        if args.command == "serve":
            import uvicorn

            uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
            return 0

        services = create_container(settings)
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
            return 0

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
        sys.stderr.write(f"{error.category}: {error}\n")
        return 1
    except SettingsValidationError as error:
        sys.stderr.write(f"invalid_settings: {error}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
