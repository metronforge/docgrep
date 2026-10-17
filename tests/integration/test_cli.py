"""Console commands over the real local stores and deterministic models."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from docgrep.api.cli import main
from docgrep.api.container import Services
from docgrep.api.http import create_app
from docgrep.domain.errors import StoreUnavailable
from docgrep.settings import Settings


def _run_cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        DOCGREP_CORPUS_ROOT=str(tmp_path / "corpus"),
        DOCGREP_DATA_DIR=str(tmp_path / "data"),
        DOCGREP_BACKEND="fake",
        DOCGREP_CHUNK_SIZE="10",
        DOCGREP_CHUNK_OVERLAP="2",
    )
    return subprocess.run(
        [sys.executable, "-m", "docgrep.api.cli", *args],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
        timeout=30,
    )


def test_console_index_and_search_match_http_schema(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.txt").write_text(
        "A rare blue comet crosses the sky.", encoding="utf-8"
    )

    built = _run_cli(tmp_path, "index")
    assert built.returncode == 0, built.stderr
    assert "1 indexed" in built.stdout

    searched = _run_cli(tmp_path, "search", "blue comet", "--top-n", "1", "--json")
    assert searched.returncode == 0, searched.stderr
    cli_result = json.loads(searched.stdout)

    readable = _run_cli(tmp_path, "search", "blue comet")
    assert readable.returncode == 0
    assert "note.txt" in readable.stdout
    assert "A rare blue comet crosses the sky." in readable.stdout

    settings = Settings(
        corpus_root=corpus,
        data_dir=tmp_path / "data",
        backend="fake",
        chunk_size=10,
        chunk_overlap=2,
    )
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        http_response = client.post("/search", json={"query": "blue comet", "top_n": 1})
    assert http_response.status_code == 200
    http_result = http_response.json()
    assert cli_result.keys() == http_result.keys()
    assert cli_result.pop("took_ms") >= 0
    assert http_result.pop("took_ms") >= 0
    assert cli_result == http_result
    assert cli_result["results"][0]["passage"]["text"] == (
        "A rare blue comet crosses the sky."
    )


def test_console_reports_structured_failures(tmp_path: Path) -> None:
    (tmp_path / "corpus").mkdir()
    missing_index = _run_cli(tmp_path, "search", "comet", "--json")
    assert missing_index.returncode != 0
    assert "index_incomplete" in missing_index.stderr

    invalid_query = _run_cli(tmp_path, "search", "   ")
    assert invalid_query.returncode != 0
    assert "validation_error" in invalid_query.stderr


def test_index_with_failed_sources_exits_non_zero_but_reports_everything(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "good.txt").write_text("A rare blue comet.", encoding="utf-8")
    (corpus / "broken.pdf").write_bytes(b"this is not a pdf")

    built = _run_cli(tmp_path, "index")

    assert built.returncode == 1
    assert "1 indexed, 0 skipped, 1 failed" in built.stdout
    assert "broken.pdf: parse_error" in built.stdout

    clean = tmp_path / "clean"
    (clean / "corpus").mkdir(parents=True)
    (clean / "corpus" / "good.txt").write_text("A rare blue comet.", "utf-8")
    assert _run_cli(clean, "index").returncode == 0


def test_error_output_includes_the_underlying_reason(tmp_path: Path) -> None:
    (tmp_path / "corpus").mkdir()
    blocker = tmp_path / "data"
    blocker.write_text("a file, not a directory", encoding="utf-8")

    result = _run_cli(tmp_path, "index")

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.startswith(
        "store_unavailable: Cannot open Qdrant storage (reason: "
    )
    assert "Traceback" not in result.stderr


class _Boom:
    def run(self, *args: object, **kwargs: object) -> None:
        raise OSError(28, "No space left on device")


def test_unexpected_os_errors_are_reported_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DOCGREP_CORPUS_ROOT", str(tmp_path))
    monkeypatch.setenv("DOCGREP_DATA_DIR", str(tmp_path / "data"))

    def factory(_settings: Settings) -> Services:
        return Services(search=_Boom(), build=_Boom())

    assert main(["index"], container_factory=factory) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    # The deprecated DOCGREP_CORPUS_ROOT adds one warning line (ADR-0023).
    assert captured.err.startswith("io_error: [Errno 28] No space left on device\n")


def test_domain_error_reason_is_printed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DOCGREP_CORPUS_ROOT", str(tmp_path))
    monkeypatch.setenv("DOCGREP_DATA_DIR", str(tmp_path / "data"))

    def factory(_settings: Settings) -> Services:
        raise StoreUnavailable(
            "Cannot open Qdrant storage", details={"reason": "already accessed"}
        )

    assert main(["search", "x"], container_factory=factory) == 1
    # The deprecated DOCGREP_CORPUS_ROOT adds one warning line (ADR-0023).
    assert capsys.readouterr().err.startswith(
        "store_unavailable: Cannot open Qdrant storage (reason: already accessed)\n"
    )


def test_stdout_is_unchanged_by_logging_and_logs_go_to_stderr(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.txt").write_text("A rare blue comet crosses the sky.", "utf-8")

    built = _run_cli(tmp_path, "index")

    assert re.fullmatch(
        r"Build 1: 1 indexed, 0 skipped, 0 failed in \d+\.\d\ds\n", built.stdout
    )
    searched = _run_cli(tmp_path, "search", "blue comet")
    assert searched.stdout == (
        "note.txt (score 0.731)\nA rare blue comet crosses the sky.\n\n"
    )
    assert "INFO" not in searched.stdout


def test_info_records_reach_stderr_while_stdout_stays_the_same(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.txt").write_text("A rare blue comet crosses the sky.", "utf-8")

    built = _run_cli(tmp_path, "index")
    searched = _run_cli(tmp_path, "search", "blue comet")

    assert re.fullmatch(
        r"Build 1: 1 indexed, 0 skipped, 0 failed in \d+\.\d\ds\n", built.stdout
    )
    assert searched.stdout == (
        "note.txt (score 0.731)\nA rare blue comet crosses the sky.\n\n"
    )
    assert " INFO docgrep.usecases.build_index index_build_started " in built.stderr
    assert " index_build_finished " in built.stderr
    assert " INFO docgrep.usecases.find_passages search_completed " in searched.stderr
    assert "blue comet" not in searched.stderr
