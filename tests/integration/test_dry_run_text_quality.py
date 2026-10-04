"""Dry-run text layer figures on generated PDFs (UC-19 subset, UC-5)."""

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import pytest

from docgrep.api.cli import main
from docgrep.api.container import create_dry_run
from docgrep.settings import Settings
from pdf_support import garbled, pdf_bytes, prose, write_pdf

SHY = "­"


def _corpus(root: Path) -> None:
    hyphenated = [f"hyphen{SHY}", "ation", f"mid{SHY}word and {SHY}more"]
    write_pdf(root / "a-clean.pdf", pdf_bytes([prose(1), hyphenated]))
    write_pdf(root / "b-partly.pdf", pdf_bytes([prose(2), garbled(3), prose(4)]))
    write_pdf(root / "c-broken.pdf", pdf_bytes([garbled(5), garbled(6), prose(7)]))
    write_pdf(root / "d-half.pdf", pdf_bytes([garbled(8), prose(9)]))
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


def _types(report: dict[str, Any]) -> dict[str, Any]:
    return {entry["type"]: entry for entry in report["sample"]["types"]}


def test_the_sample_counts_the_generated_garbled_pages(corpus: Path) -> None:
    code, out = _run("--json", "--sample", "5")

    assert code == 0
    report = json.loads(out)
    pdf = _types(report)["pdf"]
    assert pdf["parsed"] == 4
    assert pdf["text_quality"] == {
        "pages_sampled": 2 + 3 + 3 + 2,
        "pages_garbled": 1 + 2 + 1,
        "pages_garbled_fraction": 0.4,
        "garbled_text_layer_files": ["c-broken.pdf", "d-half.pdf"],
        "garbled_text_layer_fraction": 0.5,
        "soft_hyphens_removed_per_file": 0.75,
    }
    assert pdf["estimate"]["by_file_count"]["garbled_text_layer_files"] == 2.0
    txt = _types(report)["txt"]["text_quality"]
    assert (txt["pages_sampled"], txt["pages_garbled_fraction"]) == (0, None)
    total = report["sample"]["text_quality"]
    assert (total["pages_sampled"], total["pages_garbled"]) == (10, 4)
    assert total["garbled_text_layer_files"] == ["c-broken.pdf", "d-half.pdf"]
    assert total["garbled_text_layer_fraction"] == 0.4
    assert total["soft_hyphens_removed_per_file"] == 0.6


def test_the_text_report_shows_the_figures_and_paths_but_no_text(
    corpus: Path,
) -> None:
    code, out = _run("--sample", "5")

    assert code == 0
    assert (
        "    text layer: 10 pages sampled, 4 garbled (40.0%); 2 files would be "
        "skipped as requires_ocr for a garbled text layer (50.0%); soft hyphens "
        "removed per file 0.75"
    ) in out
    assert "  Text layer, all sampled: 10 pages sampled, 4 garbled (40.0%)" in out
    assert "      garbled text layer: c-broken.pdf" in out
    assert "      garbled text layer: d-half.pdf" in out
    assert "a-clean.pdf" not in out and "b-partly.pdf" not in out
    for content in ("marker", "QWERTY", "hyphenation", "midword", "archive"):
        assert content not in out


def _measured(report: dict[str, Any]) -> list[Any]:
    return [
        (entry["type"], entry["sampled_files"], entry["text_quality"])
        for entry in report["sample"]["types"]
    ]


def test_the_same_seed_gives_the_same_figures(corpus: Path) -> None:
    first = json.loads(_run("--json", "--sample", "3", "--seed", "4")[1])
    second = json.loads(_run("--json", "--sample", "3", "--seed", "4")[1])

    assert _measured(first) == _measured(second)
    assert first["sample"]["text_quality"] == second["sample"]["text_quality"]


def test_sampling_pdfs_writes_nothing_and_wires_no_embedder(
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
    dry_run = create_dry_run(Settings())
    assert sorted(vars(dry_run)) == ["chunker", "parsers", "scanner"]


def _keys(value: object, found: set[str] | None = None) -> set[str]:
    found = set() if found is None else found
    if isinstance(value, dict):
        for key, inner in value.items():
            found.add(key)
            _keys(inner, found)
    elif isinstance(value, list):
        for inner in value:
            _keys(inner, found)
    return found


def test_every_key_of_a_pdf_sample_is_documented(corpus: Path) -> None:
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text("utf-8")
    report = json.loads(_run("--json", "--sample", "5")[1])

    assert report["sample"]["text_quality"]["garbled_text_layer_files"]
    assert {key for key in _keys(report) if f"`{key}`" not in readme} == set()
