"""`docgrep eval --dense-hnsw-ef` and `--hnsw-recall` (ADR-0026)."""

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from docgrep.api.cli import main
from docgrep.api.container import create_container
from mcp_support import environment
from resume_support import close
from search_filter_support import TOP_K, filter_settings, write_filter_library

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "eval-cases.example.jsonl"
CASE_IDS = [
    "arnold-title",
    "arnold-comet",
    "late-lectures",
    "kepler-mars",
    "punctuation",
]


@pytest.fixture
def cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    write_filter_library(tmp_path / "corpus")
    services = create_container(filter_settings(tmp_path))
    assert services.build.run().indexed == 16
    close(services)
    for name, value in environment(
        tmp_path,
        DOCGREP_TOP_K=str(TOP_K),
        DOCGREP_TOP_N=str(TOP_K),
        DOCGREP_LOG_LEVEL="WARNING",
    ).items():
        if name.startswith("DOCGREP_"):
            monkeypatch.setenv(name, value)
    copy = tmp_path / "cases.private.jsonl"
    shutil.copyfile(EXAMPLE, copy)
    return copy


def _eval(capsys: pytest.CaptureFixture[str], *args: str) -> dict[str, Any]:
    status = main(["eval", *args, "--json"])
    captured = capsys.readouterr()
    assert status == 0, captured.err
    report: dict[str, Any] = json.loads(captured.out)
    return report


def test_the_effort_is_the_setting_unless_overridden(
    cases: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    default = _eval(capsys, str(cases))
    exact = _eval(capsys, str(cases), "--dense-hnsw-ef", "exact")
    monkeypatch.setenv("DOCGREP_DENSE_HNSW_EF", "256")
    configured = _eval(capsys, str(cases))

    assert default["settings"]["dense_hnsw_ef"] == 1024
    assert exact["settings"]["dense_hnsw_ef"] == "exact"
    assert configured["settings"]["dense_hnsw_ef"] == 256
    # Local mode searches exactly, so the effort cannot change the ranks.
    assert exact["cases"] == default["cases"] == configured["cases"]


@pytest.mark.parametrize("value", ["fast", "0", "16385"])
def test_an_invalid_effort_is_a_usage_error(
    cases: Path, capsys: pytest.CaptureFixture[str], value: str
) -> None:
    status = main(["eval", str(cases), "--dense-hnsw-ef", value])

    assert status == 2
    assert "--dense-hnsw-ef" in capsys.readouterr().err


def test_the_recall_is_reported_only_on_request(
    cases: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plain = _eval(capsys, str(cases))
    measured = _eval(capsys, str(cases), "--hnsw-recall")

    assert list(plain) == ["settings", "cases", "aggregates"]
    assert list(measured) == ["settings", "cases", "aggregates", "hnsw_recall"]
    recall = measured["hnsw_recall"]
    assert recall["limit"] == TOP_K
    assert [entry["id"] for entry in recall["cases"]] == CASE_IDS
    # Exact search in local mode: the configured effort finds all of it.
    assert all(entry["recall"] == 1.0 for entry in recall["cases"])
    assert (recall["mean"], recall["min"]) == (1.0, 1.0)
    # Measuring changes nothing else in the report.
    assert {key: measured[key] for key in plain} == plain


def test_the_text_report_shows_the_effort_and_the_recall(
    cases: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = main(["eval", str(cases), "--hnsw-recall"])
    out = capsys.readouterr().out

    assert status == 0
    assert "dense hnsw_ef 1024" in out.splitlines()[3]
    assert f"Dense recall@{TOP_K} against exact search: mean 1.000, min 1.000" in out
    assert "  arnold-title: 1.000" in out.splitlines()
