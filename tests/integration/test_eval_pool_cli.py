"""`docgrep eval --rerank-pool`, the sparse weighting in its settings, and
the rerank pool in `GET /info` (ADR-0025)."""

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from docgrep.api.cli import main
from docgrep.api.container import create_container
from docgrep.api.http import create_app
from docgrep.domain.search_effort import default_rerank_pool
from mcp_support import environment
from resume_support import close
from search_filter_support import TOP_K, filter_settings, write_filter_library
from sparse_support import as_tf_v1_build

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "eval-cases.example.jsonl"


def _library(
    root: Path, monkeypatch: pytest.MonkeyPatch, *, bm25: bool, **extra: str
) -> Path:
    write_filter_library(root / "corpus")
    services = create_container(filter_settings(root))
    build = services.build if bm25 else as_tf_v1_build(services.build)
    assert build.run().indexed == 16
    close(services)
    for name, value in environment(
        root,
        DOCGREP_TOP_K=str(TOP_K),
        DOCGREP_TOP_N=str(TOP_K),
        DOCGREP_LOG_LEVEL="WARNING",
        **extra,
    ).items():
        if name.startswith("DOCGREP_"):
            monkeypatch.setenv(name, value)
    cases = root / "cases.private.jsonl"
    shutil.copyfile(EXAMPLE, cases)
    return cases


def _eval(capsys: pytest.CaptureFixture[str], *args: str) -> dict[str, Any]:
    status = main(["eval", *args, "--json"])
    captured = capsys.readouterr()
    assert status == 0, captured.err
    report: dict[str, Any] = json.loads(captured.out)
    return report


def test_the_rerank_pool_replaces_the_setting_for_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cases = _library(tmp_path, monkeypatch, bm25=False)
    default = _eval(capsys, str(cases))

    pooled = _eval(capsys, str(cases), "--rerank-pool", str(2 * TOP_K))

    # The default pool is 150, at most twice top_k (ADR-0026).
    assert default["settings"]["rerank_pool"] == default_rerank_pool(TOP_K)
    assert pooled["settings"]["rerank_pool"] == 2 * TOP_K
    assert pooled["settings"]["top_k"] == TOP_K
    # More candidates reach the reranker, never fewer.
    assert (
        pooled["aggregates"]["recall_stage1_top_k"]
        >= (default["aggregates"]["recall_stage1_top_k"])
    )
    stage1 = [
        case["ranks"]["stage1"]["rank"]
        for case in pooled["cases"]
        if case["ranks"]["stage1"]["rank"] is not None
    ]
    assert stage1 and max(stage1) <= 2 * TOP_K


def test_the_setting_is_the_default_of_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cases = _library(tmp_path, monkeypatch, bm25=False, DOCGREP_RERANK_POOL="6")

    report = _eval(capsys, str(cases))

    assert report["settings"]["rerank_pool"] == 6
    status = main(["eval", str(cases)])
    out = capsys.readouterr().out
    assert status == 0
    assert out.splitlines()[3] == "rerank pool 6; dense hnsw_ef 1024; sparse tf-v1"


@pytest.mark.parametrize("pool", ["0", str(2 * TOP_K + 1)])
def test_a_pool_out_of_bounds_is_a_usage_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    pool: str,
) -> None:
    cases = _library(tmp_path, monkeypatch, bm25=False)

    status = main(["eval", str(cases), "--rerank-pool", pool])

    assert status == 2
    assert "rerank" in capsys.readouterr().err


def test_the_settings_carry_the_sparse_weighting_of_the_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cases = _library(tmp_path, monkeypatch, bm25=True)

    settings = _eval(capsys, str(cases))["settings"]

    assert settings["sparse_encoder"] == "bm25-v2"
    assert (settings["bm25_k1"], settings["bm25_b"]) == (1.2, 0.75)
    assert settings["bm25_avgdl"] > 0
    status = main(["eval", str(cases)])
    line = capsys.readouterr().out.splitlines()[3]
    assert status == 0
    assert line.startswith(
        f"rerank pool {default_rerank_pool(TOP_K)}; dense hnsw_ef 1024; "
        "sparse bm25-v2 (k1 1.2, b 0.75, "
    )


def test_info_reports_the_rerank_pool(tmp_path: Path) -> None:
    settings = filter_settings(tmp_path, rerank_pool=7)
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        assert client.get("/info").json()["rerank_pool"] == 7
    default = filter_settings(tmp_path)
    with TestClient(create_app(default), base_url="http://127.0.0.1") as client:
        assert client.get("/info").json()["rerank_pool"] == default_rerank_pool(TOP_K)
