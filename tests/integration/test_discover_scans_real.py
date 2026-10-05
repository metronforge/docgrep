"""Scan discovery on generated PDFs and DjVu files, through the CLI and HTTP
(UC-5)."""

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import pymupdf
import pytest
from fastapi.testclient import TestClient

from djvu_support import HAVE_DJVU_TOOLS, FakeRunner, make_djvu, write_fake_djvu
from docgrep.adapters.djvulibre import CommandResult, ToolMissing
from docgrep.adapters.scans import FileScanProbe
from docgrep.api.cli import main
from docgrep.api.container import create_discover_scans
from docgrep.api.http import create_app
from docgrep.domain.errors import InspectionFailed
from docgrep.domain.models import FileType
from docgrep.settings import Settings
from page_classes import control_characters
from pdf_support import pdf_bytes, prose, write_pdf


def _heavy(seed: int) -> list[str]:
    return prose(seed, lines=30)


def _corpus(root: Path) -> None:
    write_pdf(root / "books/native.pdf", pdf_bytes([_heavy(i) for i in range(12)]))
    write_pdf(root / "books/scan.pdf", pdf_bytes([None] * 12))
    write_pdf(
        root / "books/garbled.pdf",
        pdf_bytes(["\n".join(control_characters(i)).split("\n") for i in range(12)]),
    )
    # Text pages with blank pages in between: still native.
    write_pdf(
        root / "papers/mixed.pdf",
        pdf_bytes([_heavy(i) if i % 3 else None for i in range(12)]),
    )
    (root / "papers/corrupt.pdf").parent.mkdir(parents=True, exist_ok=True)
    (root / "papers/corrupt.pdf").write_bytes(b"%PDF-1.7 broken")
    encrypted = pymupdf.open()
    encrypted.new_page().insert_text((72, 72), "secret text")
    encrypted.save(
        root / "papers/locked.pdf",
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw="user",
        owner_pw="owner",
    )
    encrypted.close()
    (root / "notes.txt").write_text("not inspected", encoding="utf-8")


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "corpus"
    _corpus(root)
    monkeypatch.setenv("DOCGREP_CORPUS_ROOT", str(root))
    monkeypatch.setenv("DOCGREP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DOCGREP_BACKEND", "fake")
    return root


def _cli(*args: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()

    def no_container(settings: Settings) -> Any:
        raise AssertionError("discovery must not create the index services")

    with redirect_stdout(out), redirect_stderr(err):
        code = main(["discover-scans", *args], container_factory=no_container)
    return code, out.getvalue(), err.getvalue()


def test_generated_pdfs_are_classified(corpus: Path) -> None:
    code, out, _ = _cli("--json")

    assert code == 0
    report = json.loads(out)
    entries = {e["path"]: e for e in report["files"]}
    assert set(entries) == {
        "books/garbled.pdf",
        "books/scan.pdf",
        "papers/corrupt.pdf",
        "papers/locked.pdf",
    }
    assert (
        entries["books/scan.pdf"]["classification"],
        entries["books/scan.pdf"]["reason"],
    ) == (
        "scanned",
        "low_text_density",
    )
    assert entries["books/garbled.pdf"]["reason"] == "garbled_text_layer"
    assert entries["papers/corrupt.pdf"]["reason"] == "corrupt_header"
    assert entries["papers/locked.pdf"]["reason"] == "encrypted"
    assert entries["books/scan.pdf"]["page_count"] == 12
    assert report["totals"] == {
        "inspected": 6,
        "native": 2,
        "scanned": 2,
        "uncertain": 0,
        "inspection_failed": 2,
    }
    assert report["by_file_type"]["djvu"]["inspected"] == 0


def test_the_text_output_lists_files_and_counts(corpus: Path) -> None:
    code, out, _ = _cli()

    assert code == 0
    assert "  books/scan.pdf  pdf  scanned  low_text_density  12 pages  " in out
    assert "  papers/locked.pdf  pdf  inspection_failed  encrypted" in out
    assert (
        "Total: 6 inspected, 2 native, 2 scanned, 0 uncertain, 2 inspection failed"
        in out
    )
    assert "native.pdf" not in out and "marker" not in out


def test_two_runs_are_equal_and_write_nothing(corpus: Path, tmp_path: Path) -> None:
    before = sorted(
        (str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in tmp_path.rglob("*")
    )

    first = _cli("--json")[1]
    second = _cli("--json")[1]
    _cli()

    after = sorted(
        (str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in tmp_path.rglob("*")
    )
    assert first == second
    assert after == before
    assert not (tmp_path / "data").exists()


def test_the_wiring_has_no_store_manifest_lock_or_model(corpus: Path) -> None:
    discovery = create_discover_scans(Settings())

    assert sorted(vars(discovery)) == [
        "density_lower",
        "density_upper",
        "probe",
        "scanner",
    ]


def test_filters_from_the_command_line(corpus: Path) -> None:
    report = json.loads(
        _cli("--json", "--path-glob", "books/**", "--file-type", "pdf")[1]
    )

    assert {e["path"] for e in report["files"]} == {
        "books/garbled.pdf",
        "books/scan.pdf",
    }
    assert report["filters"] == {"path_glob": "books/**", "file_types": ["pdf"]}


def test_usage_and_configuration_errors_exit_with_2(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SystemExit) as usage:
        _cli("--file-type", "epub")
    assert usage.value.code == 2

    monkeypatch.setenv("DOCGREP_SCAN_DENSITY_LOWER", "600")
    assert _cli()[0] == 2
    monkeypatch.delenv("DOCGREP_SCAN_DENSITY_LOWER")

    monkeypatch.setenv("DOCGREP_CORPUS_ROOT", str(corpus / "missing"))
    code, out, err = _cli()
    assert code == 2 and out == "" and err.startswith("validation_error: ")


def test_http_returns_the_cli_json(corpus: Path) -> None:
    cli = json.loads(_cli("--json", "--file-type", "pdf")[1])

    with TestClient(create_app(Settings()), base_url="http://127.0.0.1") as client:
        response = client.post("/scans/discover", json={"file_types": ["pdf"]})
        empty_body = client.post("/scans/discover")
        bad = client.post("/scans/discover", json={"file_types": ["epub"]})

    assert response.status_code == 200 and response.json() == cli
    assert empty_body.status_code == 200
    assert empty_body.json()["filters"]["file_types"] == ["djvu", "pdf"]
    assert bad.status_code == 400
    assert bad.json()["category"] == "validation_error"


def test_the_pdf_probe_skips_failing_pages(tmp_path: Path) -> None:
    write_pdf(tmp_path / "a.pdf", pdf_bytes([_heavy(1), None, ["x"]]))
    probe = FileScanProbe(tmp_path)

    samples = probe.sample("a.pdf", FileType.PDF, [1, 2, 3, 99])

    assert samples[3] is None
    assert samples[1] is not None and samples[1].non_whitespace == 0
    assert samples[0] is not None and samples[0].non_whitespace > 1000
    assert probe.page_count("a.pdf", FileType.PDF) == 3


def test_the_djvu_probe_maps_tool_failures(tmp_path: Path) -> None:
    write_fake_djvu(tmp_path / "a.djvu")
    missing = FileScanProbe(tmp_path, FakeRunner({"a.djvu": ToolMissing("djvused")}))
    invalid = FileScanProbe(
        tmp_path, FakeRunner({"a.djvu": CommandResult(10, b"", b"broken")})
    )
    good = FileScanProbe(
        tmp_path, FakeRunner({"a.djvu": ["one two", None, "x " * 300]})
    )

    with pytest.raises(InspectionFailed) as raised:
        missing.page_count("a.djvu", FileType.DJVU)
    assert raised.value.details["reason"] == "djvu_tools_missing"
    assert "djvulibre-bin" in str(raised.value.details["hint"])
    with pytest.raises(InspectionFailed) as raised:
        invalid.page_count("a.djvu", FileType.DJVU)
    assert raised.value.details["reason"] == "djvu_invalid"
    assert good.page_count("a.djvu", FileType.DJVU) == 3


@pytest.mark.skipif(not HAVE_DJVU_TOOLS, reason="the djvulibre tools are not installed")
def test_real_djvu_files_are_classified(corpus: Path) -> None:
    make_djvu(corpus / "djvu/book.djvu", ["\n".join(_heavy(i)) for i in range(7)])
    make_djvu(corpus / "djvu/scan.djv", [None] * 8)

    report = json.loads(_cli("--json", "--file-type", "djvu")[1])

    assert [e["path"] for e in report["files"]] == ["djvu/scan.djv"]
    assert report["files"][0]["reason"] == "low_text_density"
    assert report["files"][0]["page_count"] == 8
    assert report["by_file_type"]["djvu"] == {
        "inspected": 2,
        "native": 1,
        "scanned": 1,
        "uncertain": 0,
        "inspection_failed": 0,
    }


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


def test_every_json_key_is_documented_in_the_readme(corpus: Path) -> None:
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text("utf-8")
    report = json.loads(_cli("--json")[1])

    assert report["files"]
    assert {key for key in _keys(report) if f"`{key}`" not in readme} == set()
    for flag in ("--path-glob", "--file-type", "--json"):
        assert flag in readme
