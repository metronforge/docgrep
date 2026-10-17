"""Dry-run counts per page rule and per flagged file on generated PDFs (UC-19
subset, UC-5)."""

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import pytest

from docgrep.api.cli import main
from docgrep.api.container import create_dry_run
from docgrep.settings import Settings
from page_classes import (
    class_names,
    control_characters,
    regular_expressions,
    toc_glued,
)
from pdf_support import garbled, pdf_bytes, prose, write_pdf


def _corpus(root: Path) -> None:
    # Pages 3 and 6 are broken (R1, and R2: no spaces); class names fire R2
    # and regular expressions R3, both weak; the table of contents fires none.
    write_pdf(
        root / "mixed.pdf",
        pdf_bytes(
            [
                prose(1),
                toc_glued(),
                control_characters(1),
                class_names(),
                regular_expressions(),
                control_characters(2),
                prose(2),
            ]
        ),
    )
    # 25 broken pages: the document is skipped and 20 page numbers are listed.
    write_pdf(
        root / "many.pdf",
        pdf_bytes([control_characters(i) for i in range(25)] + [prose(3)]),
    )
    # Three weak-only pages out of ten: flagged, nothing dropped.
    write_pdf(
        root / "weak.pdf",
        pdf_bytes([garbled(i) for i in range(3)] + [prose(i) for i in range(7)]),
    )
    write_pdf(root / "clean.pdf", pdf_bytes([prose(4), prose(5)]))
    (root / "notes.txt").write_text("marker10 plain words", encoding="utf-8")


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "corpus"
    _corpus(root)
    monkeypatch.setenv("DOCGREP_CORPUS_ROOT", str(root))
    monkeypatch.setenv("DOCGREP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DOCGREP_BACKEND", "fake")
    return root


def _run(*args: str) -> tuple[int, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(["dry-run", *args])
    return code, out.getvalue()


def _pdf(report: dict[str, Any]) -> dict[str, Any]:
    (entry,) = [e for e in report["sample"]["types"] if e["type"] == "pdf"]
    quality: dict[str, Any] = entry["text_quality"]
    return quality


def _by_rule(r1: int, r2: int, r3: int) -> dict[str, int]:
    return {"r1_broken_characters": r1, "r2_long_tokens": r2, "r3_low_alnum": r3}


def test_rule_counts_and_flagged_files_match_the_generated_pages(
    corpus: Path,
) -> None:
    code, out = _run("--json", "--sample", "5")

    assert code == 0
    report = json.loads(out)
    quality = _pdf(report)
    assert quality["pages_sampled"] == 7 + 26 + 10 + 2
    assert quality["pages_garbled"] == 2 + 25
    assert quality["pages_by_rule"] == _by_rule(27, 3 + 25 + 3, 1)
    assert quality["garbled_text_layer_files"] == ["many.pdf"]
    assert quality["flagged_files"] == [
        {
            "path": "many.pdf",
            "pages_dropped": 25,
            "pages_by_rule": _by_rule(25, 25, 0),
            "dropped_pages": list(range(1, 21)),
        },
        {
            "path": "mixed.pdf",
            "pages_dropped": 2,
            "pages_by_rule": _by_rule(2, 3, 1),
            "dropped_pages": [3, 6],
        },
        {
            "path": "weak.pdf",
            "pages_dropped": 0,
            "pages_by_rule": _by_rule(0, 3, 0),
            "dropped_pages": [],
        },
    ]
    total = report["sample"]["text_quality"]
    assert total["pages_by_rule"] == quality["pages_by_rule"]
    assert total["flagged_files"] == quality["flagged_files"]
    (txt,) = [e for e in report["sample"]["types"] if e["type"] == "txt"]
    assert txt["text_quality"]["pages_by_rule"] == _by_rule(0, 0, 0)
    assert txt["text_quality"]["flagged_files"] == []


def test_the_text_report_shows_rule_counts_and_no_text(corpus: Path) -> None:
    code, out = _run("--sample", "5")

    assert code == 0
    assert (
        "pages per rule (weak signals drop nothing): R1 broken characters 27, "
        "R2 long tokens 31, R3 few letters and digits 1"
    ) in out
    assert "garbled text layer: many.pdf" in out
    assert "mixed.pdf" not in out and "weak.pdf" not in out
    for content in ("marker", "import", "grep", "Spring", "QWERTY"):
        assert content not in out


def test_the_same_seed_gives_the_same_rule_figures(corpus: Path) -> None:
    first = json.loads(_run("--json", "--sample", "3", "--seed", "7")[1])
    second = json.loads(_run("--json", "--sample", "3", "--seed", "7")[1])

    assert first["sample"]["text_quality"] == second["sample"]["text_quality"]
    assert [e["sampled_files"] for e in first["sample"]["types"]] == [
        e["sampled_files"] for e in second["sample"]["types"]
    ]


def test_rule_sampling_writes_nothing_and_wires_no_embedder(
    corpus: Path, tmp_path: Path
) -> None:
    before = {
        str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in corpus.rglob("*")
    }

    assert _run("--json", "--sample", "5")[0] == 0

    after = {
        str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in corpus.rglob("*")
    }
    assert after == before
    assert not (tmp_path / "data").exists()
    assert sorted(vars(create_dry_run(Settings()))) == [
        "chunker",
        "parsers",
        "roots",
        "scanner",
    ]
