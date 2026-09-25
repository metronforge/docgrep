"""Console commands over the real local stores and deterministic models."""

import json
import os
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from docgrep.api.http import create_app
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
        [str(Path(sys.executable).with_name("docgrep")), *args],
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
    with TestClient(create_app(settings)) as client:
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
