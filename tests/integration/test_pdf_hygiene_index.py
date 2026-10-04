"""PDF text hygiene through builds and updates over the real stores (UC-3, UC-4)."""

from pathlib import Path

from docgrep.adapters.manifest import SQLiteManifest
from docgrep.domain.models import FileType
from pdf_support import garbled, pdf_bytes, prose, write_pdf
from resume_support import close, dump_points, settings_for
from update_support import sources_table, update_services, with_version

SHY = "\u00ad"


def _corpus(root: Path) -> None:
    write_pdf(root / "clean.pdf", pdf_bytes([prose(1), [f"hyphen{SHY}", "ation"]]))
    write_pdf(root / "partly.pdf", pdf_bytes([prose(2), garbled(3), prose(4)]))
    write_pdf(root / "broken.pdf", pdf_bytes([garbled(5), garbled(6), prose(7)]))
    (root / "note.txt").write_text("marker9 plain words here", encoding="utf-8")


def test_a_garbled_text_layer_is_recorded_as_requiring_ocr(tmp_path: Path) -> None:
    _corpus(tmp_path / "corpus")
    settings = settings_for(tmp_path / "corpus", tmp_path / "data")
    services = update_services(settings)

    summary = services.build.run()
    close(services)

    assert (summary.indexed, summary.skipped, summary.failed) == (3, 1, 0)
    manifest = SQLiteManifest(settings.data_dir)
    broken = manifest.get_source("broken.pdf")
    assert broken is not None
    assert (broken.status, broken.error) == (
        "skipped:requires_ocr",
        "garbled_text_layer",
    )
    assert broken.source.parser_version == "2"
    pages = {
        p[1]["page_number"]
        for p in dump_points(settings)
        if p[1]["path"] == "partly.pdf"
    }
    assert pages == {1, 3}
    texts = " ".join(p[1]["text"] for p in dump_points(settings))
    assert "hyphenation" in texts and SHY not in texts


def test_an_index_from_pdf_parser_1_is_re_extracted_by_update(tmp_path: Path) -> None:
    _corpus(tmp_path / "corpus")
    clean = settings_for(tmp_path / "corpus", tmp_path / "clean")
    services = update_services(clean)
    services.build.run()
    close(services)
    live = settings_for(tmp_path / "corpus", tmp_path / "live")
    services = update_services(live)
    with_version(services.build, FileType.PDF, "1")
    services.build.run()
    close(services)

    services = update_services(live)
    assert services.update is not None
    summary = services.update.run()
    close(services)

    assert (summary.plan.parser_changed, summary.plan.modified) == (3, 3)
    assert summary.plan.unchanged == 1
    assert dump_points(live) == dump_points(clean)
    assert sources_table(live) == sources_table(clean)
