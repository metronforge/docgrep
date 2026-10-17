#!/usr/bin/env python3
"""Turn a pytest JUnit report into GitHub workflow annotations.

Usage: ci_annotate_failures.py JUNIT_XML [--log CHECK_LOG]

Prints a ``::error`` workflow command with the failure totals, then one per
failing or erroring test case, so the failing tests can be read through the
check-run annotations of the REST API without the job log. GitHub shows at
most 10 error annotations per step, so only the first 9 tests are annotated;
the Markdown table appended to ``$GITHUB_STEP_SUMMARY`` lists up to 20 and
says how many more are in the report. Without a report, it names the stage of
``scripts/check.sh`` that failed, read from the check log (by default
``check.log`` next to the report). Standard library only; always exits 0.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

# GitHub keeps at most 10 error annotations per step: the totals and 9 tests.
MAX_ANNOTATED = 9
MAX_SUMMARY = 20
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


def _summary(rows: list[tuple[str, str]], heading: str, more: int) -> str:
    lines = [f"### {heading}", "", "| Title | Message |", "| --- | --- |"]
    lines += [f"| {_cell(title)} | {_cell(message)} |" for title, message in rows]
    if more:
        lines += ["", f"{more} more failing tests are in the reports artifact."]
    return "\n".join(lines) + "\n\n"


def annotate(junit: Path, log: Path) -> tuple[list[tuple[str, str]], int]:
    """The (title, message) pairs to report for one JUnit file, totals first
    and at most ``MAX_SUMMARY`` tests, and how many failing tests they omit."""
    if not junit.is_file():
        stage = failed_stage(log)
        return [
            (
                "check.sh",
                f"no test report at {junit}; the failing stage is {stage}",
            )
        ], 0
    try:
        report = ET.parse(junit).getroot()
    except ET.ParseError as error:
        return [("check.sh", f"the test report {junit} is unreadable: {error}")], 0
    found = findings(report)
    if not found:
        return [], 0
    failed = sum(item.kind == "failure" for item in found)
    rows = [("failures", f"{failed} failed, {len(found) - failed} errors")]
    rows += [
        (item.classname, f"{item.name}: {item.message}") for item in found[:MAX_SUMMARY]
    ]
    return rows, max(len(found) - MAX_SUMMARY, 0)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("junit", type=Path)
    parser.add_argument("--log", type=Path, default=None)
    arguments = parser.parse_args(argv)
    junit: Path = arguments.junit
    log: Path = arguments.log or junit.parent / "check.log"
    try:
        rows, more = annotate(junit, log)
    except Exception as error:  # the annotation step must never fail the job
        rows, more = [("annotate", f"could not read {junit}: {error!r}")], 0
    for title, message in rows[: 1 + MAX_ANNOTATED]:
        print(_command(title, message))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if rows and summary:
        try:
            with open(summary, "a", encoding="utf-8") as handle:
                handle.write(_summary(rows, f"Failures in {junit.name}", more))
        except OSError as error:
            print(_command("annotate", f"could not write the step summary: {error}"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
