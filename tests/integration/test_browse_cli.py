"""`docgrep sources`: the table, --json, filters and exit statuses (UC-08)."""

import json
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from docgrep.api.cli import main
from docgrep.api.container import BrowseOperation, create_container
from docgrep.api.http import create_app
from docgrep.domain.errors import StoreUnavailable
from docgrep.domain.models import FileType, Source
from docgrep.domain.sources import BrowsedSource, BrowseRequest, BrowseResult
from docgrep.settings import Settings
from mcp_support import built_library, environment, snapshot
from resume_support import close


def _cli(root: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "docgrep.api.cli", "sources", *args],
        env=environment(root, **env),
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_the_table_lists_one_line_per_source(tmp_path: Path) -> None:
    built_library(tmp_path)

    done = _cli(tmp_path)

    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    assert lines[0].split() == ["PATH", "TYPE", "YEAR", "AUTHOR", "TITLE", "PASSAGES"]
    assert [line.split()[0] for line in lines[1:5]] == [
        "memo.docx",
        "notes/a.txt",
        "notes/b.txt",
        "paper.pdf",
    ]
    assert "Sea Memo" in lines[1]
    assert lines[5] == "Sources 1-4 of 4 (page 1 of 1); docx 1, pdf 1, txt 2"
    assert len({len(line) for line in lines[:5]}) == 1


def test_json_prints_the_body_of_the_http_endpoint(tmp_path: Path) -> None:
    settings = built_library(tmp_path)
    arguments = (
        "--type",
        "TXT",
        "--type",
        "pdf",
        "--glob",
        "**/*.txt",
        "--path-prefix",
        "notes/",
        "--sort",
        "path",
        "--desc",
        "--page",
        "1",
        "--page-size",
        "1",
    )

    done = _cli(tmp_path, *arguments, "--json")

    assert done.returncode == 0, done.stderr
    services = create_container(settings)
    try:
        client = TestClient(
            create_app(settings, services=services), base_url="http://127.0.0.1"
        )
        rest = client.post(
            "/sources/browse",
            json={
                "file_types": ["TXT", "pdf"],
                "path_globs": ["**/*.txt"],
                "path_prefixes": ["notes/"],
                "sort_by": "path",
                "sort_order": "desc",
                "page": 1,
                "page_size": 1,
            },
        ).json()
    finally:
        close(services)
    body = json.loads(done.stdout)
    assert body == rest
    assert [item["path"] for item in body["sources"]] == ["notes/b.txt"]
    assert body["total_hits"] == 2


def test_every_filter_option_reaches_the_request(tmp_path: Path) -> None:
    built_library(tmp_path)

    done = _cli(
        tmp_path,
        "--author",
        "PYTHON",
        "--author",
        "nobody",
        "--year-from",
        "1900",
        "--year-to",
        "2100",
        "--source-id",
        "memo.docx",
        "--has-metadata",
        "with_author",
        "--json",
    )

    assert done.returncode == 0, done.stderr
    body = json.loads(done.stdout)
    assert body["applied_filters"] == {
        "source_ids": ["memo.docx"],
        "authors": ["PYTHON", "nobody"],
        "year_from": 1900,
        "year_to": 2100,
        "has_metadata": "with_author",
    }
    # The memo has an author but no year, so the year bounds exclude it.
    assert body["sources"] == [] and body["total_hits"] == 0


def test_long_values_are_cut_with_an_ellipsis_never_wrapped(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    long_name = "Ландау [Том 1] (1958) " + "with a very long name " * 5 + ".txt"
    (corpus / long_name).write_text("heron", encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "-m", "docgrep.api.cli", "index"],
        env=environment(tmp_path),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr

    listed = _cli(tmp_path)

    lines = listed.stdout.splitlines()
    assert len(lines) == 3
    # The path column is 48 characters wide.
    assert lines[1][:50] == long_name[:47] + "…  "
    assert len(lines[1]) == len(lines[0])


def test_an_empty_result_and_a_missing_index_exit_0_and_create_nothing(
    tmp_path: Path,
) -> None:
    (tmp_path / "corpus").mkdir()

    done = _cli(tmp_path)

    assert done.returncode == 0, done.stderr
    assert done.stdout == "No indexed sources match\n"
    assert not (tmp_path / "data").exists()


def test_browsing_writes_nothing(tmp_path: Path) -> None:
    built_library(tmp_path)
    before = snapshot(tmp_path / "corpus", tmp_path / "data")

    assert _cli(tmp_path, "--sort", "passage_count").returncode == 0

    assert snapshot(tmp_path / "corpus", tmp_path / "data") == before


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--page", "0"), "validation_error: page: Input should be greater"),
        (("--page-size", "501"), "validation_error: page_size: Input should be less"),
        (("--type", "ocr_pdf"), "validation_error: file_types: Value error, unknown"),
        (("--author", " "), "validation_error: authors: Value error, authors must"),
        (("--year-from", "3", "--year-to", "1"), "validation_error: Value error, year"),
        (("--sort", "relevance"), "invalid choice: 'relevance'"),
        (("--year-from", "soon"), "invalid int value: 'soon'"),
    ],
)
def test_invalid_arguments_exit_2(
    tmp_path: Path, arguments: tuple[str, ...], message: str
) -> None:
    (tmp_path / "corpus").mkdir()

    done = _cli(tmp_path, *arguments)

    assert done.returncode == 2
    assert done.stdout == ""
    assert message in done.stderr


def test_invalid_settings_exit_2(tmp_path: Path) -> None:
    (tmp_path / "corpus").mkdir()

    done = _cli(tmp_path, DOCGREP_MCP_MAX_PAGE_SIZE="501")

    assert done.returncode == 2
    assert done.stderr.startswith("invalid_settings:")


class _Down:
    def run(self, request: BrowseRequest) -> BrowseResult:
        raise StoreUnavailable("Qdrant operation failed", details={"reason": "down"})


def test_a_runtime_failure_exits_1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name, value in environment(tmp_path).items():
        if name.startswith("DOCGREP_"):
            monkeypatch.setenv(name, value)

    @contextmanager
    def down(_settings: Settings) -> Iterator[BrowseOperation]:
        yield _Down()

    status = main(["sources"], browse_factory=down)

    out, err = capsys.readouterr()
    assert status == 1 and out == ""
    # The deprecated DOCGREP_CORPUS_ROOT adds one warning line (ADR-0023).
    assert err.startswith("store_unavailable: Qdrant operation failed (reason: down)\n")


def test_a_degraded_count_is_shown_as_a_question_mark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _Degraded:
        def run(self, request: BrowseRequest) -> BrowseResult:
            source = Source("a.txt", "a.txt", FileType.TXT, year=1999)
            return BrowseResult(
                (BrowsedSource(source, None, degraded=True),),
                1,
                request.page,
                request.page_size,
                (("txt", 1),),
                request.applied_filters(),
            )

    @contextmanager
    def degraded(_settings: Settings) -> Iterator[BrowseOperation]:
        yield _Degraded()

    for name, value in environment(tmp_path).items():
        if name.startswith("DOCGREP_"):
            monkeypatch.setenv(name, value)

    assert main(["sources"], browse_factory=degraded) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].split() == ["a.txt", "txt", "1999", "-", "-", "?"]
    assert lines[-1] == "? the passage count could not be read"
