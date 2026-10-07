"""OCR outputs in the index: an original delegated to its OCR'd sibling is
skipped, the sibling is indexed, and removing the sibling brings the
original back. Real SQLite and local-mode Qdrant with the deterministic
backend; the OCR engine is a fake that adds a known text layer."""

import os
from pathlib import Path
from typing import Any

import pytest

from docgrep.adapters.manifest import SQLiteManifest
from docgrep.api.container import (
    create_discover_scans,
    create_process_scans,
    create_status,
    open_ocr_registry_view,
)
from docgrep.settings import Settings
from docgrep.usecases.process_scans import OcrRequest, ProcessScans
from docgrep.usecases.update_index import UpdateSummary
from ocr_support import CopyConverter, TextLayerEngine
from pdf_support import pdf_bytes, prose, write_pdf
from resume_support import close, dump_points, settings_for, write_corpus
from update_support import sources_table, update_services

STAMP = 1_700_000_000


def _corpus(root: Path) -> None:
    write_corpus(root, 3)
    write_pdf(root / "scan.pdf", pdf_bytes([None, None]))
    # Some text on the first page only: indexed, but worth an OCR.
    write_pdf(root / "partial.pdf", pdf_bytes([prose(7, 4), None, None]))
    for path in root.rglob("*"):
        os.utime(path, (STAMP, STAMP))


def _build(settings: Settings) -> None:
    services = update_services(settings)
    try:
        services.build.run()
    finally:
        close(services)


def _update(settings: Settings) -> UpdateSummary:
    services = update_services(settings)
    try:
        assert services.update is not None
        return services.update.run()
    finally:
        close(services)


def _search(settings: Settings, query: str) -> list[str]:
    services = update_services(settings)
    try:
        return [h.source.path for h in services.search.run(query, top_n=10).results]
    finally:
        close(services)


def _ocr(settings: Settings, *paths: str) -> None:
    use_case: ProcessScans = create_process_scans(settings)
    use_case.engine = TextLayerEngine(settings.corpus_root)
    use_case.converter = CopyConverter(settings.corpus_root)
    summary = use_case.run(OcrRequest(paths=paths, override_native=True))
    assert summary.status == "completed" and summary.failed == 0


def _status(settings: Settings, path: str) -> str:
    record = SQLiteManifest(settings.data_dir).get_source(path)
    assert record is not None
    return record.status


def _points_of(settings: Settings, path: str) -> int:
    return sum(1 for _, payload, _ in dump_points(settings) if payload["path"] == path)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    corpus = tmp_path / "corpus"
    _corpus(corpus)
    return settings_for(corpus, tmp_path / "data")


def test_ocr_then_update_indexes_the_sibling_instead_of_the_original(
    settings: Settings,
) -> None:
    _build(settings)
    assert _status(settings, "scan.pdf") == "skipped:requires_ocr"
    assert _status(settings, "partial.pdf") == "indexed"
    assert _points_of(settings, "partial.pdf") > 0

    _ocr(settings, "scan.pdf", "partial.pdf")
    summary = _update(settings)

    plan = summary.plan
    assert (plan.added, plan.modified, plan.delegated_to_ocr) == (2, 2, 2)
    assert (plan.delegation_ended, plan.content_changed, plan.parser_changed) == (
        0,
        0,
        0,
    )
    assert summary.failed == 0
    assert _status(settings, "scan.pdf") == "skipped:delegated_to_ocr"
    assert _status(settings, "partial.pdf") == "skipped:delegated_to_ocr"
    assert _status(settings, "scan.ocr.pdf") == "indexed"
    assert _status(settings, "partial.ocr.pdf") == "indexed"
    assert _points_of(settings, "partial.pdf") == 0
    assert _points_of(settings, "scan.ocr.pdf") > 0
    assert "scan.ocr.pdf" in _search(settings, "recognised page 2 of scan")

    sources = create_status(settings).run().as_dict()["sources"]
    assert sources["skipped_by_reason"]["delegated_to_ocr"] == 2
    assert sources["parser_changed"] == 0

    again = _update(settings)
    assert again.no_changes
    assert again.plan.unchanged == len(
        SQLiteManifest(settings.data_dir).committed_sources()
    )


def test_the_updated_index_equals_a_clean_build(settings: Settings) -> None:
    _build(settings)
    _ocr(settings, "scan.pdf", "partial.pdf")
    _update(settings)
    points = dump_points(settings)
    table = sources_table(settings)

    services = update_services(settings)
    try:
        services.build.run(refuse_resumable=False)
    finally:
        close(services)

    assert dump_points(settings) == points
    assert sources_table(settings) == table


def test_removing_the_output_makes_the_original_reprocessable(
    settings: Settings,
) -> None:
    _build(settings)
    _ocr(settings, "scan.pdf", "partial.pdf")
    _update(settings)

    (settings.corpus_root / "partial.ocr.pdf").unlink()
    summary = _update(settings)

    assert (summary.plan.removed, summary.plan.delegation_ended) == (1, 1)
    assert _status(settings, "partial.pdf") == "indexed"
    assert _points_of(settings, "partial.pdf") > 0
    assert _points_of(settings, "partial.ocr.pdf") == 0
    assert _status(settings, "scan.pdf") == "skipped:delegated_to_ocr"


def test_a_changed_original_is_no_longer_delegated(settings: Settings) -> None:
    _build(settings)
    _ocr(settings, "partial.pdf")
    _update(settings)

    write_pdf(settings.corpus_root / "partial.pdf", pdf_bytes([prose(8, 4)]))
    summary = _update(settings)

    assert summary.plan.content_changed == 1
    assert _status(settings, "partial.pdf") == "indexed"


def _discover(settings: Settings, include: bool = False) -> dict[str, Any]:
    return (
        create_discover_scans(settings)
        .run(
            include_already_processed=include, registry=open_ocr_registry_view(settings)
        )
        .as_dict()
    )


def test_discovery_before_and_after_ocr(settings: Settings) -> None:
    before = _discover(settings)
    assert {f["path"]: f["ocr_status"] for f in before["files"]} == {
        "partial.pdf": "not_processed",
        "scan.pdf": "not_processed",
    }
    assert before["already_processed"] == 0

    _ocr(settings, "scan.pdf", "partial.pdf")
    after = _discover(settings)
    everything = _discover(settings, include=True)

    assert after["files"] == [] and after["already_processed"] == 2
    assert after["totals"]["inspected"] == 0
    listed = {f["path"]: f["ocr_status"] for f in everything["files"]}
    assert listed == {
        "partial.pdf": "processed_via_ocr_pdf",
        "scan.pdf": "processed_via_ocr_pdf",
    }
    assert everything["include_already_processed"] is True
    # The outputs are native, so only counted; their status is processed.
    statuses = {
        e.path: e.ocr_status
        for e in create_discover_scans(settings)
        .run(include_already_processed=True, registry=open_ocr_registry_view(settings))
        .entries
    }
    assert statuses["scan.ocr.pdf"] == "processed"

    (settings.corpus_root / "scan.ocr.pdf").unlink()
    assert [f["path"] for f in _discover(settings)["files"]] == ["scan.pdf"]
