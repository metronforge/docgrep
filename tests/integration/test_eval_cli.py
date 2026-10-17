"""`docgrep eval`: text and --json output, options, exit statuses and
read-only use over the generated filter library (ADR-0024)."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from docgrep.adapters.filelock import FileIndexLock
from docgrep.api.cli import main
from docgrep.api.container import create_container
from mcp_support import environment, snapshot
from resume_support import close
from search_filter_support import TOP_K, filter_settings, write_filter_library
from sparse_support import as_tf_v1_build

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "eval-cases.example.jsonl"


def _built(root: Path) -> Path:
    write_filter_library(root / "corpus")
    services = create_container(filter_settings(root))
    # The expected ranks were recorded on a tf-v1 index (ADR-0025).
    summary = as_tf_v1_build(services.build).run()
    assert (summary.indexed, summary.failed) == (16, 0), summary
    close(services)
    cases = root / "cases.private.jsonl"
    shutil.copyfile(EXAMPLE, cases)
    return cases


@pytest.fixture
def library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cases = _built(tmp_path)
    for name, value in environment(
        tmp_path,
        DOCGREP_TOP_K=str(TOP_K),
        DOCGREP_TOP_N=str(TOP_K),
        DOCGREP_LOG_LEVEL="WARNING",
    ).items():
        if name.startswith("DOCGREP_"):
            monkeypatch.setenv(name, value)
    return cases


def _run(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    status = main(["eval", *args])
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def _json(capsys: pytest.CaptureFixture[str], *args: str) -> dict[str, Any]:
    status, out, err = _run(capsys, *args, "--json")
    assert status == 0, err
    report: dict[str, Any] = json.loads(out)
    return report


def test_the_text_report(library: Path, capsys: pytest.CaptureFixture[str]) -> None:
    status, out, _ = _run(capsys, str(library))

    assert status == 0
    lines = out.splitlines()
    assert lines[:3] == [
        "Evaluation of 5 cases over 16 passages",
        "Models: deterministic-hash-v1 (embedding), lexical-overlap-v1 (reranker)",
        "top_k 4; depth 2000; control on; duplicates collapsed",
    ]
    block = lines[lines.index("== arnold-comet (line 5)") :][:12]
    assert block[1:3] == [
        "Query: comet orbit",
        "Note: decoys outrank it in the first stage",
    ]
    assert block[4].split() == ["LIST", "RANK", "SOURCE"]
    assert [row.split()[0] for row in block[5:10]] == [
        "dense",
        "sparse",
        "fused_deep",
        "stage1",
        "final",
    ]
    assert block[8].split() == ["stage1", "-", "-"]
    assert block[10] == (
        "  within top_k: no; rerank score: -; control: rank 1, score 0.622"
    )
    assert "  sparse           -       -  (no sparse terms)" in lines
    assert lines[-3:] == [
        "Lost in the first stage (found at depth, not within top_k): arnold-comet",
        "Lost in rerank (within top_k, final rank above 10): -",
        "Not found in any list: -",
    ]


def test_the_json_report_has_fixed_keys_in_file_order(
    library: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = _json(capsys, str(library))

    assert list(report) == ["settings", "cases", "aggregates"]
    assert report["settings"] == {
        "passages": 16,
        "embedding_model": "deterministic-hash-v1",
        "reranker_model": "lexical-overlap-v1",
        "top_k": TOP_K,
        "depth": 2000,
        "collapse_duplicates": True,
        "sparse_channel": True,
        "control": True,
    }
    assert [case["id"] for case in report["cases"]] == [
        "arnold-title",
        "arnold-comet",
        "late-lectures",
        "kepler-mars",
        "punctuation",
    ]
    comet = report["cases"][1]
    assert list(comet["ranks"]) == ["dense", "sparse", "fused_deep", "stage1", "final"]
    assert comet["expected"]["roots"] == ["corpus"]
    assert comet["within_top_k"] is False
    assert comet["control_rank"] == 1
    assert report["aggregates"]["lost_in_first_stage"] == ["arnold-comet"]
    # No timings or timestamps anywhere.
    assert "took_ms" not in json.dumps(report)


def test_top_k_replaces_the_configured_first_stage(
    library: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = _json(capsys, str(library), "--top-k", "16")

    assert report["settings"]["top_k"] == 16
    comet = report["cases"][1]
    assert comet["within_top_k"] is True
    assert comet["ranks"]["final"]["rank"] > TOP_K
    assert comet["rerank_score"] is not None


def test_depth_and_no_control(
    library: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = _json(capsys, str(library), "--depth", "3", "--no-control")

    assert report["settings"]["depth"] == 3
    assert report["settings"]["control"] is False
    comet = report["cases"][1]
    assert comet["ranks"]["dense"]["rank"] is None
    assert (comet["control_rank"], comet["control_score"]) == (None, None)


def test_a_case_not_found_still_completes(
    library: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    library.write_text(
        json.dumps({"id": "ghost", "query": "comet", "fragment": "no such words"})
        + "\n",
        encoding="utf-8",
    )

    report = _json(capsys, str(library))

    assert report["aggregates"]["not_found"] == ["ghost"]
    assert report["aggregates"]["mrr"] == 0.0


def test_long_queries_and_notes_are_cut_in_text_only(
    library: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    query = "comet " * 30
    library.write_text(
        json.dumps({"query": query, "note": "n" * 100, "source": "books/late.md"})
        + "\n",
        encoding="utf-8",
    )

    _, out, _ = _run(capsys, str(library))
    report = _json(capsys, str(library))

    query_line = next(line for line in out.splitlines() if line.startswith("Query:"))
    assert query_line.endswith("…") and len(query_line) == len("Query: ") + 72
    assert "Note: " + "n" * 71 + "…" in out
    assert report["cases"][0]["query"] == query
    assert report["cases"][0]["note"] == "n" * 100


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (
            '{"query": "x", "source": "a", "extra": 1}\n',
            "line 1: unknown fields: extra",
        ),
        ('# c\n{"query": "x"}\n', "line 2: one of source, source_glob or fragment"),
        ('{"query": "x", "source"\n', "line 1: invalid JSON"),
        (
            '{"query": "x", "fragment": "y", "filters": {"roots": ["nowhere"]}}\n',
            "nowhere",
        ),
        ('{"query": "' + "x" * 1001 + '", "fragment": "y"}\n', "line 1: query must"),
    ],
)
def test_a_malformed_file_exits_with_2_naming_the_problem(
    library: Path, capsys: pytest.CaptureFixture[str], content: str, message: str
) -> None:
    library.write_text(content, encoding="utf-8")

    status, out, err = _run(capsys, str(library))

    assert status == 2
    assert out == ""
    assert message in err


@pytest.mark.parametrize(
    "arguments",
    [["--depth", "0"], ["--depth", "10001"], ["--top-k", "0"], ["--top-k", "10001"]],
)
def test_bounds_exit_with_2(
    library: Path, capsys: pytest.CaptureFixture[str], arguments: list[str]
) -> None:
    status, _, err = _run(capsys, str(library), *arguments)

    assert status == 2
    assert "must be between 1 and 10000" in err


def test_a_missing_file_exits_with_2(
    library: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status, _, err = _run(capsys, str(library.parent / "missing.jsonl"))

    assert status == 2
    assert "cannot read the cases file" in err


def test_invalid_settings_exit_with_2(
    library: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCGREP_TOP_K", "0")

    status, _, err = _run(capsys, str(library))

    assert status == 2
    assert "invalid_settings" in err


def test_no_index_exits_with_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in environment(tmp_path).items():
        if name.startswith("DOCGREP_"):
            monkeypatch.setenv(name, value)
    (tmp_path / "corpus").mkdir()
    cases = tmp_path / "cases.jsonl"
    shutil.copyfile(EXAMPLE, cases)

    status, out, err = _run(capsys, str(cases))

    assert status == 1
    assert out == ""
    assert err.startswith("index_incomplete:")
    assert not (tmp_path / "data").exists()


def test_an_unreachable_store_exits_with_1(
    library: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCGREP_QDRANT_URL", "http://127.0.0.1:9")

    status, _, err = _run(capsys, str(library))

    assert status == 1
    assert err.startswith("store_unavailable:")


def test_nothing_is_written_and_no_lock_is_taken(
    library: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = library.parent / "data"
    before = snapshot(data, library.parent / "corpus")

    status, _, _ = _run(capsys, str(library))

    assert status == 0
    assert snapshot(data, library.parent / "corpus") == before


def test_it_runs_while_another_operation_holds_the_index_lock(
    library: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lock = FileIndexLock(library.parent / "data" / "index.lock")

    with lock.hold("build"):
        status, out, err = _run(capsys, str(library), "--json")

    assert status == 0, err
    assert json.loads(out)["aggregates"]["cases"] == 5


def test_two_processes_print_identical_json(tmp_path: Path) -> None:
    cases = _built(tmp_path)
    env = environment(tmp_path, DOCGREP_TOP_K=str(TOP_K), DOCGREP_TOP_N=str(TOP_K))

    outputs = [
        subprocess.run(
            [sys.executable, "-m", "docgrep.api.cli", "eval", str(cases), "--json"],
            env=env,
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout
        for _ in range(2)
    ]

    assert outputs[0] == outputs[1]
    assert json.loads(outputs[0])["settings"]["passages"] == 16
