"""`docgrep context`: the human output, --json and exit statuses (UC-12)."""

import json
import subprocess
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from context_support import MISSING, source_passages
from docgrep.api.cli import main
from docgrep.api.container import ContextOperation, create_container
from docgrep.api.http import create_app
from docgrep.domain.context import ContextResult
from docgrep.domain.errors import StoreUnavailable
from docgrep.settings import Settings
from mcp_support import built_library, environment, snapshot
from resume_support import close


def _cli(root: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "docgrep.api.cli", "context", *args],
        env=environment(root, **env),
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_json_prints_the_body_of_the_http_endpoint(tmp_path: Path) -> None:
    settings = built_library(tmp_path)
    paper = source_passages(settings, "paper.pdf")
    ids = [paper[1].passage_id, MISSING, paper[0].passage_id]
    services = create_container(settings)
    try:
        with TestClient(
            create_app(settings, services=services), base_url="http://127.0.0.1"
        ) as client:
            rest = client.post(
                "/passages/context", json={"passage_ids": ids, "window_tokens": 70}
            ).json()
    finally:
        close(services)

    done = _cli(tmp_path, *ids, "--window", "70", "--json")

    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == rest


def test_the_human_output_shows_the_source_the_passage_and_its_context(
    tmp_path: Path,
) -> None:
    settings = built_library(tmp_path)
    first, second, *_ = source_passages(settings, "paper.pdf")
    memo = source_passages(settings, "memo.docx")[-1]

    done = _cli(tmp_path, second.passage_id, memo.passage_id, MISSING, "--window", "1")

    assert done.returncode == 0, done.stderr
    out = done.stdout
    assert f"== {second.passage_id} [ok]" in out
    assert "Source: paper.pdf:1" in out
    assert out.index("-- before") < out.index(">>>>> passage >>>>>")
    assert f">>>>> passage >>>>>\n{second.text}\n<<<<< passage <<<<<" in out
    # The text before is the first passage without its overlap.
    assert first.text[: second.chunk_offset - first.chunk_offset].strip() in out
    assert "start of source" in out
    assert "Source: memo.docx — Expedition > Storm (Sea Memo)" in out
    assert "end of source" in out
    assert f"== {MISSING} [not_found]" in out
    assert "Search again for current passage ids." in out


@pytest.mark.parametrize(
    ("arguments", "env", "message"),
    [
        (("abc",), {}, "malformed passage id: abc"),
        ((MISSING, "--window", "-1"), {}, "window_tokens must be between 0 and 5000"),
        ((MISSING, "--window", "5001"), {}, "window_tokens must be between 0 and 5000"),
        (
            (MISSING, MISSING),
            {"DOCGREP_CONTEXT_MAX_PASSAGES": "1"},
            "passage_ids must hold between 1 and 1 ids",
        ),
        ((), {}, "the following arguments are required: PASSAGE_ID"),
        ((MISSING, "--window", "wide"), {}, "invalid int value: 'wide'"),
    ],
)
def test_invalid_arguments_exit_2(
    tmp_path: Path, arguments: tuple[str, ...], env: dict[str, str], message: str
) -> None:
    (tmp_path / "corpus").mkdir()

    done = _cli(tmp_path, *arguments, **env)

    assert done.returncode == 2
    assert done.stdout == ""
    assert message in done.stderr


def test_invalid_settings_exit_2(tmp_path: Path) -> None:
    (tmp_path / "corpus").mkdir()

    done = _cli(tmp_path, MISSING, DOCGREP_CONTEXT_WINDOW_TOKENS="6000")

    assert done.returncode == 2
    assert done.stderr.startswith("invalid_settings:")


def test_a_missing_index_exits_0_and_creates_nothing(tmp_path: Path) -> None:
    (tmp_path / "corpus").mkdir()

    done = _cli(tmp_path, MISSING, "--json")

    assert done.returncode == 0, done.stderr
    body = json.loads(done.stdout)
    assert [item["status"] for item in body["results"]] == ["not_found"]
    assert not (tmp_path / "data").exists()


def test_expanding_writes_nothing(tmp_path: Path) -> None:
    settings = built_library(tmp_path)
    ids = [item.passage_id for item in source_passages(settings, "memo.docx")]
    before = snapshot(tmp_path / "corpus", tmp_path / "data")

    assert _cli(tmp_path, *ids).returncode == 0

    assert snapshot(tmp_path / "corpus", tmp_path / "data") == before


class _Down:
    def run(
        self, passage_ids: Sequence[str], *, window_tokens: int | None = None
    ) -> ContextResult:
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
    def down(_settings: Settings) -> Iterator[ContextOperation]:
        yield _Down()

    status = main(["context", MISSING], context_factory=down)

    out, err = capsys.readouterr()
    assert status == 1 and out == ""
    # The deprecated DOCGREP_CORPUS_ROOT adds one warning line (ADR-0023).
    assert err.startswith("store_unavailable: Qdrant operation failed (reason: down)\n")
