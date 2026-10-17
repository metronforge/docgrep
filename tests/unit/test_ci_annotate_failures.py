"""The CI script that turns a JUnit report into workflow annotations (#34)."""

import os
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import quoteattr

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ci_annotate_failures.py"


def _case(name: str, kind: str | None = None, message: str = "boom") -> str:
    inner = f"<{kind} message={quoteattr(message)}>trace</{kind}>" if kind else ""
    return f'<testcase classname="tests.unit.test_x" name="{name}">{inner}</testcase>'


def _junit(path: Path, *cases: str) -> Path:
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?><testsuites>'
        f'<testsuite name="pytest">{"".join(cases)}</testsuite></testsuites>',
        encoding="utf-8",
    )
    return path


def _run(tmp_path: Path, *args: str) -> tuple[list[str], str]:
    summary = tmp_path / "summary.md"
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GITHUB")
    }
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        env={**environment, "GITHUB_STEP_SUMMARY": str(summary)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    written = summary.read_text(encoding="utf-8") if summary.exists() else ""
    return completed.stdout.splitlines(), written


def test_the_totals_come_first_then_one_line_per_test(tmp_path: Path) -> None:
    report = _junit(
        tmp_path / "junit.xml",
        _case("test_ok"),
        _case("test_bad", "failure", "assert 1 == 2\n  where 1 = f()"),
        _case("test_setup", "error", "failed on setup with ValueError"),
    )

    lines, summary = _run(tmp_path, str(report))

    assert lines == [
        "::error title=failures::1 failed, 1 errors",
        "::error title=tests.unit.test_x::test_bad: assert 1 == 2",
        "::error title=tests.unit.test_x::test_setup: failed on setup with ValueError",
    ]
    assert "| failures | 1 failed, 1 errors |" in summary
    assert "| tests.unit.test_x | test_bad: assert 1 == 2 |" in summary
    assert "test_setup: failed on setup with ValueError" in summary
    assert "more failing tests" not in summary


def test_a_passing_report_prints_nothing(tmp_path: Path) -> None:
    report = _junit(tmp_path / "junit.xml", _case("test_ok"), _case("test_fine"))

    lines, summary = _run(tmp_path, str(report))

    assert lines == []
    assert summary == ""


def test_ten_failures_fit_the_annotation_limit_of_ten(tmp_path: Path) -> None:
    cases = [_case(f"test_f{n:02}", "failure") for n in range(10)]
    report = _junit(tmp_path / "junit.xml", *cases)

    lines, summary = _run(tmp_path, str(report))

    assert lines == [
        "::error title=failures::10 failed, 0 errors",
        *(f"::error title=tests.unit.test_x::test_f{n:02}: boom" for n in range(9)),
    ]
    assert "| tests.unit.test_x | test_f09: boom |" in summary
    assert "more failing tests" not in summary


def test_more_than_twenty_failures_list_twenty_in_the_summary(
    tmp_path: Path,
) -> None:
    cases = [_case(f"test_f{n:02}", "failure") for n in range(23)]
    cases += [_case(f"test_e{n}", "error") for n in range(2)]
    report = _junit(tmp_path / "junit.xml", *cases)

    lines, summary = _run(tmp_path, str(report))

    assert len(lines) == 10
    assert lines[0] == "::error title=failures::23 failed, 2 errors"
    assert lines[1] == "::error title=tests.unit.test_x::test_f00: boom"
    assert lines[-1] == "::error title=tests.unit.test_x::test_f08: boom"
    rows = [line for line in summary.splitlines() if line.startswith("| ")]
    assert rows[:3] == [
        "| Title | Message |",
        "| --- | --- |",
        "| failures | 23 failed, 2 errors |",
    ]
    assert len(rows) == 3 + 20
    assert rows[-1] == "| tests.unit.test_x | test_f19: boom |"
    assert "5 more failing tests are in the reports artifact." in summary


def test_special_characters_are_escaped_and_long_messages_cut(
    tmp_path: Path,
) -> None:
    message = "100% wrong: a, b\r\nsecond line"
    report = _junit(
        tmp_path / "junit.xml",
        _case("test_odd[a:b,c]", "failure", message),
        _case("test_long", "failure", "x" * 500),
    )
    report.write_text(
        report.read_text(encoding="utf-8").replace(
            'classname="tests.unit.test_x" name="test_odd',
            'classname="tests.unit.test_x:odd,cls" name="test_odd',
        ),
        encoding="utf-8",
    )

    lines, summary = _run(tmp_path, str(report))

    assert lines[1] == (
        "::error title=tests.unit.test_x%3Aodd%2Ccls::"
        "test_odd[a:b,c]: 100%25 wrong: a, b"
    )
    assert lines[2] == "::error title=tests.unit.test_x::test_long: " + "x" * 300
    assert "second line" not in summary


def test_a_multiline_message_keeps_its_first_line_only(tmp_path: Path) -> None:
    report = _junit(
        tmp_path / "junit.xml", _case("test_x", "failure", "\nfirst\nsecond\r\nthird")
    )

    lines, _ = _run(tmp_path, str(report))

    assert lines[1:] == ["::error title=tests.unit.test_x::test_x: first"]


def test_a_missing_report_names_the_stage_that_failed(tmp_path: Path) -> None:
    log = tmp_path / "check.log"
    stages = {
        "src/a.py:1:1: F401 unused import\nFound 1 error.\n": "ruff check",
        "All checks passed!\nWould reformat: src/a.py\n": "ruff format",
        "All checks passed!\n80 files already formatted\n"
        "src/a.py:3: error: bad\nFound 1 error in 1 file\n": "mypy",
        "No virtual environment: run 'uv sync --extra cpu' first.\n": "setup",
    }
    for output, stage in stages.items():
        log.write_text(output, encoding="utf-8")

        lines, summary = _run(tmp_path, str(tmp_path / "junit.xml"))

        assert len(lines) == 1
        assert lines[0].startswith("::error title=check.sh::no test report at ")
        assert lines[0].endswith(f"; the failing stage is {stage}")
        assert f"the failing stage is {stage}" in summary
        (tmp_path / "summary.md").unlink()


def test_a_missing_report_and_log_still_exit_zero(tmp_path: Path) -> None:
    lines, _ = _run(tmp_path, str(tmp_path / "junit.xml"))

    assert lines == [
        f"::error title=check.sh::no test report at {tmp_path / 'junit.xml'}; "
        "the failing stage is unknown (no check log)"
    ]


def test_an_explicit_log_and_an_unreadable_report(tmp_path: Path) -> None:
    log = tmp_path / "elsewhere.log"
    log.write_text(
        "All checks passed!\n1 file already formatted\n"
        "Success: no issues found in 1 source file\n",
        encoding="utf-8",
    )
    missing, _ = _run(tmp_path, str(tmp_path / "junit.xml"), "--log", str(log))
    (tmp_path / "broken.xml").write_text("<testsuites><testcase", encoding="utf-8")

    broken, _ = _run(tmp_path, str(tmp_path / "broken.xml"))

    assert missing[0].endswith("the failing stage is pytest")
    assert broken[0].startswith("::error title=check.sh::the test report ")
    assert "is unreadable" in broken[0]
