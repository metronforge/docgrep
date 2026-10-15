"""DjVu sources through the real tools, builds, updates and the dry run
(UC-3 step 6.3, UC-4)."""

from pathlib import Path
from typing import Any

import pytest

from djvu_support import HAVE_DJVU_TOOLS, FakeRunner, make_djvu, write_fake_djvu
from docgrep.adapters.djvulibre import ToolMissing
from docgrep.adapters.manifest import SQLiteManifest
from docgrep.adapters.parsers import DjvuParser
from docgrep.api.container import create_dry_run
from docgrep.domain.models import FileType, Source
from pdf_support import pdf_bytes, prose, write_pdf
from resume_support import close, dump_points, settings_for
from update_support import update_services


def _page(seed: int) -> str:
    return "\n".join(prose(seed)) + "\n"


@pytest.mark.skipif(not HAVE_DJVU_TOOLS, reason="the djvulibre tools are not installed")
def test_the_real_tools_read_a_generated_text_layer(tmp_path: Path) -> None:
    make_djvu(
        tmp_path / "book.djvu",
        [
            _page(1),
            None,
            "\u041f\u0435\u0440\u0435\u00ad\n\u043d\u043e\u0441 " + _page(3),
        ],
    )
    make_djvu(tmp_path / "scan.djvu", [None, None])
    parser = DjvuParser(tmp_path)

    book = parser.parse(Source("book.djvu", "book.djvu", FileType.DJVU))
    scan = parser.parse(Source("scan.djvu", "scan.djvu", FileType.DJVU))

    assert [s.page_number for s in book.sections] == [1, 3]
    assert book.sections[0].text == _page(1)
    assert book.sections[1].text.startswith(
        "\u041f\u0435\u0440\u0435\u043d\u043e\u0441 marker3"
    )
    assert scan.requires_ocr and scan.quality is not None
    assert scan.quality.skip_reason == "no_text_layer"


def _corpus(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "note.txt").write_text("marker1 plain words here", encoding="utf-8")
    write_pdf(root / "paper.pdf", pdf_bytes([prose(2)]))
    write_fake_djvu(root / "book.djvu")
    write_fake_djvu(root / "scan.djv")


def _runner() -> FakeRunner:
    return FakeRunner({"book.djvu": [_page(4), None, _page(5)], "scan.djv": [None]})


def _built_without_djvu(tmp_path: Path) -> Any:
    _corpus(tmp_path / "corpus")
    settings = settings_for(tmp_path / "corpus", tmp_path / "data")
    services = update_services(settings)
    build: Any = services.build
    build.parsers = {t: p for t, p in build.parsers.items() if t is not FileType.DJVU}
    assert build.run().failed == 0
    close(services)
    return settings


def _update(settings: Any, runner: FakeRunner) -> Any:
    services = update_services(settings)
    update: Any = services.update
    update.parsers = {
        **update.parsers,
        FileType.DJVU: DjvuParser(settings.corpus_root, runner),
    }
    try:
        return update.run()
    finally:
        close(services)


def test_djvu_skipped_as_unsupported_is_picked_up_by_update(tmp_path: Path) -> None:
    settings = _built_without_djvu(tmp_path)
    manifest = SQLiteManifest(settings.data_dir)
    for path in ("book.djvu", "scan.djv"):
        record = manifest.get_source(path)
        assert record is not None and record.status == "skipped:unsupported"
    pdf_version = manifest.get_source("paper.pdf")

    summary = _update(settings, _runner())

    plan = summary.plan
    assert (plan.newly_supported, plan.modified) == (2, 2)
    assert (plan.content_changed, plan.parser_changed, plan.retried) == (0, 0, 0)
    assert (plan.added, plan.removed, plan.unchanged) == (0, 0, 2)
    book = manifest.get_source("book.djvu")
    assert book is not None and book.status == "indexed"
    assert book.source.parser_version == "1"
    scan = manifest.get_source("scan.djv")
    assert scan is not None
    assert (scan.status, scan.error) == ("skipped:requires_ocr", "no_text_layer")
    assert manifest.get_source("paper.pdf") == pdf_version
    assert pdf_version is not None and pdf_version.source.parser_version == "3"
    pages = {
        p[1]["page_number"]
        for p in dump_points(settings)
        if p[1]["path"] == "book.djvu"
    }
    assert pages == {1, 3}
    assert _update(settings, _runner()).no_changes


def test_missing_tools_in_a_build_fail_the_source_and_update_retries_it(
    tmp_path: Path,
) -> None:
    _corpus(tmp_path / "corpus")
    settings = settings_for(tmp_path / "corpus", tmp_path / "data")
    services = update_services(settings)
    build: Any = services.build
    missing = FakeRunner({"book.djvu": ToolMissing("djvused"), "scan.djv": [None]})
    build.parsers = {
        **build.parsers,
        FileType.DJVU: DjvuParser(settings.corpus_root, missing),
    }
    summary = build.run()
    close(services)

    assert (summary.failed, summary.indexed) == (1, 2)
    record = SQLiteManifest(settings.data_dir).get_source("book.djvu")
    assert record is not None and record.status == "failed:parse_error"
    assert record.error is not None and "djvu_tools_missing" in record.error

    retried = _update(settings, _runner())

    assert retried.plan.retried == 1 and retried.indexed == 1
    record = SQLiteManifest(settings.data_dir).get_source("book.djvu")
    assert record is not None and record.status == "indexed"


def test_missing_tools_in_an_update_keep_the_source_for_the_next_update(
    tmp_path: Path,
) -> None:
    settings = _built_without_djvu(tmp_path)
    missing = FakeRunner({"book.djvu": ToolMissing("djvused"), "scan.djv": [None]})

    first = _update(settings, missing)

    assert [(f.path, f.reason) for f in first.failures] == [
        ("book.djvu", "parse_error")
    ]
    # A recorded source that fails in an update is recorded as failed, so
    # the next update retries it.
    record = SQLiteManifest(settings.data_dir).get_source("book.djvu")
    assert record is not None and record.status == "failed:parse_error"

    second = _update(settings, _runner())

    assert second.plan.retried == 1 and second.indexed == 1


def test_the_dry_run_lists_djvu_as_supported(tmp_path: Path) -> None:
    _corpus(tmp_path / "corpus")
    settings = settings_for(tmp_path / "corpus", tmp_path / "data")

    report = create_dry_run(settings).run().as_dict()

    types = {entry["type"]: entry for entry in report["types"]}
    assert types["djvu"]["status"] == "supported"
    assert types["djvu"]["files"] == 2
    assert report["recognised_without_parser"]["files"] == 0
