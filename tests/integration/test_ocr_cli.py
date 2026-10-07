"""`docgrep ocr` on generated corpora with the real manifest, registry, lock,
probe and files; the OCR engine is a fake that adds a known text layer, or
the real ocrmypdf where the tests say so."""

import io
import json
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import pymupdf
import pytest
from fastapi.testclient import TestClient

from docgrep.adapters.filelock import FileIndexLock
from docgrep.adapters.manifest import SQLiteManifest, SQLiteOcrRegistry
from docgrep.api.cli import main
from docgrep.api.container import create_cancel, create_process_scans, create_status
from docgrep.api.http import create_app
from docgrep.settings import Settings
from docgrep.usecases.process_scans import ProcessScans
from ocr_support import (
    CYRILLIC_FONT,
    HAVE_OCR_TOOLS,
    CopyConverter,
    FakeCorpus,
    FakeEngine,
    TextLayerEngine,
    garbled_pdf_bytes,
    image_pdf_bytes,
    pdf_text,
)
from pdf_support import broken, pdf_bytes, prose, write_pdf

needs_ocr_tools = pytest.mark.skipif(
    not HAVE_OCR_TOOLS, reason="needs ocrmypdf and tesseract"
)


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "corpus"
    write_pdf(root / "books" / "scan.pdf", pdf_bytes([None, None]))
    write_pdf(root / "books" / "other scan.pdf", pdf_bytes([None]))
    write_pdf(
        root / "books" / "native.pdf", pdf_bytes([prose(n, 30) for n in range(3)])
    )
    (root / "books" / "notes.md").write_text("# notes\n", encoding="utf-8")
    (tmp_path / "outside.pdf").write_bytes(pdf_bytes([None]))
    monkeypatch.setenv("DOCGREP_CORPUS_ROOT", str(root))
    monkeypatch.setenv("DOCGREP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DOCGREP_BACKEND", "fake")
    return root


def _factory(
    on_run: Callable[[str], None] | None = None,
) -> Callable[[Settings], ProcessScans]:
    def create(settings: Settings) -> ProcessScans:
        use_case = create_process_scans(settings)
        use_case.engine = TextLayerEngine(settings.corpus_root, on_run=on_run)
        use_case.converter = CopyConverter(settings.corpus_root)
        use_case.poll_seconds = 0
        return use_case

    return create


def _cli(
    *args: str, factory: Callable[[Settings], ProcessScans] | None = None
) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        status = main(["ocr", *args], ocr_factory=factory or _factory())
    return status, out.getvalue(), err.getvalue()


def test_plan_run_and_rerun(corpus: Path, tmp_path: Path) -> None:
    original = (corpus / "books" / "scan.pdf").read_bytes()

    status, plan, _ = _cli("--all-discovered", "--plan")
    assert status == 0
    assert "OCR plan (languages rus+eng+ukr): 2 to process (pdf 2)" in plan
    assert "to_process  books/scan.pdf  pdf  2 pages" in plan
    assert not (tmp_path / "data" / "manifest.sqlite3").exists() or (
        SQLiteManifest(tmp_path / "data").latest_run() is None
    )

    status, out, _ = _cli("--all-discovered")
    assert status == 0
    assert "[1/2] processed books/other scan.pdf, 1 pages" in out
    assert "[2/2] processed books/scan.pdf, 2 pages" in out
    assert "2 processed, 0 skipped, 0 failed, 0 cancelled, 0 not started." in out
    assert out.rstrip().endswith(
        "Run `docgrep update` so that the OCR outputs are indexed."
    )
    assert (corpus / "books" / "scan.pdf").read_bytes() == original
    assert "recognised page 2 of scan" in pdf_text(corpus / "books" / "scan.ocr.pdf")
    assert not [p for p in (corpus / "books").iterdir() if p.name.startswith(".")]
    registry = SQLiteOcrRegistry(tmp_path / "data", corpus)
    delegation = registry.live_delegation("books/scan.pdf")
    assert delegation is not None and delegation.output_path == "books/scan.ocr.pdf"
    run = SQLiteManifest(tmp_path / "data").latest_run()
    assert run is not None and (run.kind, run.status) == ("ocr", "completed")

    status, out, _ = _cli("books/scan.pdf", "books/other scan.pdf")
    assert status == 0
    assert "2 skipped (already_processed 2)" in out


def test_the_plan_takes_no_lock(corpus: Path, tmp_path: Path) -> None:
    lock = FileIndexLock(tmp_path / "data" / "index.lock")
    with lock.hold("build"):
        assert _cli("books/scan.pdf", "--plan")[0] == 0
        status, _, err = _cli("books/scan.pdf")
    assert status == 1
    assert "index_lock_held" in err and "the ocr cannot start" in err
    assert not (corpus / "books" / "scan.ocr.pdf").exists()


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((), "targets"),
        (("books/scan.pdf", "--all-discovered"), "targets"),
        (("books/scan.pdf", "../outside.pdf"), "outside_corpus"),
        (("books/scan.pdf", "books/notes.md"), ".md"),
        (("books/missing.pdf",), "not_found"),
        (("--all-discovered", "--override-native"), "override_native"),
        (("books/scan.pdf", "--languages", "rus eng"), "invalid language"),
    ],
)
def test_usage_errors_exit_with_2_before_any_work(
    corpus: Path, args: tuple[str, ...], expected: str
) -> None:
    status, out, err = _cli(*args)

    assert status == 2
    assert err.startswith("validation_error:") and expected in err
    assert out == ""
    assert not (corpus / "books" / "scan.ocr.pdf").exists()


def test_invalid_ocr_settings_exit_with_2(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCGREP_OCR_LANGUAGES", "rus,eng")

    status, _, err = _cli("books/scan.pdf")

    assert status == 2 and err.startswith("invalid_settings:")


def test_missing_tools_exit_with_1_and_the_install_command(corpus: Path) -> None:
    def create(settings: Settings) -> ProcessScans:
        use_case = create_process_scans(settings)
        use_case.engine = FakeEngine(FakeCorpus(), present=False)
        return use_case

    status, out, err = _cli("--all-discovered", factory=create)

    assert status == 1 and out == ""
    assert err.startswith("ocr_tools_missing:")
    assert not (corpus / "books" / "scan.ocr.pdf").exists()


def test_a_failed_file_makes_the_exit_status_1(corpus: Path) -> None:
    (corpus / "books" / "scan.ocr.pdf").write_bytes(b"someone else's file")

    status, out, _ = _cli("--all-discovered")

    assert status == 1
    assert "books/scan.pdf: failed:output_collision" in out
    assert (corpus / "books" / "scan.ocr.pdf").read_bytes() == b"someone else's file"
    assert (corpus / "books" / "other scan.ocr.pdf").exists()


def test_json_output_of_plan_and_run(corpus: Path) -> None:
    status, out, _ = _cli("--path-glob", "books/*", "--plan", "--json")
    plan = json.loads(out)
    assert status == 0
    assert plan["filtered"] == 1 and plan["to_process"] == 2
    assert plan["to_skip"] == {"does_not_need_ocr": 1}

    status, out, _ = _cli("--path-glob", "books/*", "--json")
    report = json.loads(out)
    assert status == 0
    assert report["status"] == "completed"
    assert report["totals"]["processed"] == 2
    assert {f["path"]: f["status"] for f in report["files"]} == {
        "books/native.pdf": "skipped:does_not_need_ocr",
        "books/other scan.pdf": "processed",
        "books/scan.pdf": "processed",
    }


def test_status_and_cancel_see_the_running_ocr(corpus: Path, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def during(source: str) -> None:
        settings = Settings()
        seen["status"] = create_status(settings).run().as_dict()
        seen["cancel"] = create_cancel(settings).run()

    status, out, _ = _cli("--all-discovered", factory=_factory(on_run=during))

    active = seen["status"]["active_operation"]
    assert active["kind"] == "ocr" and active["sources_total"] == 2
    assert seen["status"]["latest_run"]["kind"] == "ocr"
    assert seen["cancel"].outcome == "requested" and seen["cancel"].kind == "ocr"
    assert status == 1
    assert "[1/2] cancelled books/other scan.pdf" in out
    assert "0 processed" in out and "1 cancelled, 1 not started" in out
    assert "run the same command again to continue" in out
    assert not [p for p in (corpus / "books").iterdir() if p.name.startswith(".")]
    run = SQLiteManifest(tmp_path / "data").latest_run()
    assert run is not None and run.status == "cancelled"
    report = create_status(Settings()).run().as_dict()
    assert report["latest_run"]["totals"]["cancelled"] == 1
    assert report["index_state"] == "not_built"

    status, out, _ = _cli("--all-discovered")
    assert status == 0 and "2 processed" in out


def test_a_signal_cancels_like_docgrep_cancel(corpus: Path) -> None:
    handlers: dict[int, Any] = {}

    def register(number: int, handler: Any) -> Any:
        previous = handlers.get(number)
        handlers[number] = handler
        return previous

    def interrupt(source: str) -> None:
        handlers[2](2, None)

    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        status = main(
            ["ocr", "--all-discovered"],
            ocr_factory=_factory(on_run=interrupt),
            register_signal=register,
        )

    assert status == 1
    assert "the running OCR is stopped" in err.getvalue()
    assert "1 cancelled, 1 not started" in out.getvalue()


def test_http_rejects_invalid_requests(corpus: Path) -> None:
    client = TestClient(create_app(Settings()), base_url="http://127.0.0.1")

    traversal = client.post("/scans/process", json={"paths": ["../outside.pdf"]})
    nothing = client.post("/scans/process", json={})
    both = client.post(
        "/scans/process", json={"paths": ["books/scan.pdf"], "all_discovered": True}
    )

    assert traversal.status_code == 400
    assert traversal.json()["details"]["reason"] == "outside_corpus"
    assert nothing.status_code == 400 and both.status_code == 400
    assert not (corpus / "books" / "scan.ocr.pdf").exists()


@needs_ocr_tools
def test_http_runs_a_batch_and_returns_the_summary(corpus: Path) -> None:
    client = TestClient(create_app(Settings()), base_url="http://127.0.0.1")

    response = client.post(
        "/scans/process", json={"path_glob": "books/native.pdf", "languages": "eng"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["skipped_by_reason"] == {"does_not_need_ocr": 1}


def _real(corpus: Path, monkeypatch: pytest.MonkeyPatch, *args: str) -> str:
    monkeypatch.setenv("DOCGREP_OCR_JOBS", "2")
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        status = main(["ocr", *args])
    assert status == 0, out.getvalue() + err.getvalue()
    return out.getvalue()


@needs_ocr_tools
def test_real_ocr_adds_a_text_layer_and_keeps_the_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "corpus"
    scan = write_pdf(
        root / "scan.pdf",
        image_pdf_bytes([["Harbour lighthouse keeper", "logbook of 1887 storms"]]),
    )
    original = scan.read_bytes()
    monkeypatch.setenv("DOCGREP_CORPUS_ROOT", str(root))
    monkeypatch.setenv("DOCGREP_DATA_DIR", str(tmp_path / "data"))

    out = _real(root, monkeypatch, "scan.pdf", "--languages", "eng")

    assert "[1/1] processed scan.pdf" in out
    text = pdf_text(root / "scan.ocr.pdf").lower()
    assert "lighthouse" in text and "logbook" in text
    assert scan.read_bytes() == original
    registry = SQLiteOcrRegistry(tmp_path / "data", root)
    delegation = registry.live_delegation("scan.pdf")
    assert delegation is not None and delegation.output_path == "scan.ocr.pdf"
    record = registry.find_by_input_hash(delegation.input_hash)
    assert record is not None and record.engine == "ocrmypdf"
    assert (record.languages, record.mode) == ("eng", "skip-text")
    with pymupdf.open(root / "scan.ocr.pdf") as document:
        assert document.page_count == 1


@needs_ocr_tools
def test_real_ocr_replaces_a_garbled_text_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "corpus"
    write_pdf(
        root / "garbled.pdf",
        garbled_pdf_bytes(
            [["Meadow orchard survey", "apple trees of 1912"]], [broken(3)]
        ),
    )
    monkeypatch.setenv("DOCGREP_CORPUS_ROOT", str(root))
    monkeypatch.setenv("DOCGREP_DATA_DIR", str(tmp_path / "data"))

    out = _real(root, monkeypatch, "--all-discovered", "--languages", "eng")

    assert "force-ocr" in out and "[1/1] processed garbled.pdf" in out
    text = pdf_text(root / "garbled.ocr.pdf").lower()
    assert "orchard" in text and "\x00" not in text


@needs_ocr_tools
@pytest.mark.skipif(CYRILLIC_FONT is None, reason="needs a font with Cyrillic")
def test_real_ocr_reads_cyrillic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "corpus"
    write_pdf(
        root / "letter.pdf",
        image_pdf_bytes(
            [["Маяк на северном берегу", "журнал наблюдений"]], font=CYRILLIC_FONT
        ),
    )
    monkeypatch.setenv("DOCGREP_CORPUS_ROOT", str(root))
    monkeypatch.setenv("DOCGREP_DATA_DIR", str(tmp_path / "data"))

    _real(root, monkeypatch, "letter.pdf")

    assert "маяк" in pdf_text(root / "letter.ocr.pdf").lower()
