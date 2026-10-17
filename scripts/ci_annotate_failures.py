#!/usr/bin/env python3
"""Turn a pytest JUnit report into GitHub workflow annotations.

Usage: ci_annotate_failures.py JUNIT_XML [--log CHECK_LOG]

Prints one ``::error`` workflow command per failing or erroring test case, so
the failing tests can be read through the check-run annotations of the REST
API without the job log, and appends the same lines as a Markdown table to
``$GITHUB_STEP_SUMMARY``. Without a report, it names the stage of
``scripts/check.sh`` that failed, read from the check log (by default
``check.log`` next to the report). Standard library only; always exits 0.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

MAX_LINES = 20
MAX_MESSAGE = 300

# The stages of scripts/check.sh in order, with the line each prints when it
# passes. The first stage without its line is the one that failed.
_STAGES = (
    ("setup", "No virtual environment", None),
    ("ruff check", None, "All checks passed!"),
    ("ruff format", None, "already formatted"),
    ("mypy", None, "Success: no issues found"),
    ("pytest", None, None),
)


@dataclass(frozen=True)
class Finding:
    kind: str
    classname: str
    name: str
    message: str


def _escape_data(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(value: str) -> str:
    return _escape_data(value).replace(":", "%3A").replace(",", "%2C")


def _command(title: str, message: str) -> str:
    return f"::error title={_escape_property(title)}::{_escape_data(message)}"


def _first_line(element: ET.Element) -> str:
    for text in (element.get("message"), element.text):
        for line in (text or "").splitlines():
            if line.strip():
                return line.strip()[:MAX_MESSAGE]
    return ""


def findings(report: ET.Element) -> list[Finding]:
    found: list[Finding] = []
    for case in report.iter("testcase"):
        for kind in ("failure", "error"):
            problem = case.find(kind)
            if problem is not None:
                found.append(
                    Finding(
                        kind,
                        case.get("classname", ""),
                        case.get("name", ""),
                        _first_line(problem),
                    )
                )
                break
    return found


def failed_stage(log: Path) -> str:
    """The stage of check.sh that ran last, judged from its output."""
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "unknown (no check log)"
    for stage, failure_marker, success_marker in _STAGES:
        if failure_marker is not None and failure_marker in text:
            return stage
        if success_marker is not None and success_marker not in text:
            return stage
    return "pytest"


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _summary(rows: list[tuple[str, str]], heading: str) -> str:
    lines = [f"### {heading}", "", "| Title | Message |", "| --- | --- |"]
    lines += [f"| {_cell(title)} | {_cell(message)} |" for title, message in rows]
    return "\n".join(lines) + "\n\n"


def annotate(junit: Path, log: Path) -> list[tuple[str, str]]:
    """The (title, message) pairs to report for one JUnit file."""
    if not junit.is_file():
        stage = failed_stage(log)
        return [
            (
                "check.sh",
                f"no test report at {junit}; the failing stage is {stage}",
            )
        ]
    try:
        report = ET.parse(junit).getroot()
    except ET.ParseError as error:
        return [("check.sh", f"the test report {junit} is unreadable: {error}")]
    found = findings(report)
    rows = [
        (item.classname, f"{item.name}: {item.message}") for item in found[:MAX_LINES]
    ]
    if len(found) > MAX_LINES:
        failed = sum(item.kind == "failure" for item in found)
        errors = len(found) - failed
        rows.append(("failures", f"{failed} failed, {errors} errors"))
    return rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("junit", type=Path)
    parser.add_argument("--log", type=Path, default=None)
    arguments = parser.parse_args(argv)
    junit: Path = arguments.junit
    log: Path = arguments.log or junit.parent / "check.log"
    try:
        rows = annotate(junit, log)
    except Exception as error:  # the annotation step must never fail the job
        rows = [("annotate", f"could not read {junit}: {error!r}")]
    for title, message in rows:
        print(_command(title, message))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if rows and summary:
        try:
            with open(summary, "a", encoding="utf-8") as handle:
                handle.write(_summary(rows, f"Failures in {junit.name}"))
        except OSError as error:
            print(_command("annotate", f"could not write the step summary: {error}"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
