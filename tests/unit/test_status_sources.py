"""The sources block of the status report, on in-memory fakes (UC-7)."""

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

from docgrep.api.cli import main
from docgrep.domain.models import FileType, Source
from docgrep.usecases.inspect_status import InspectStatus
from fakes import InMemoryManifest

_TYPES = {
    "txt": FileType.TXT,
    "md": FileType.MARKDOWN,
    "pdf": FileType.PDF,
    "epub": FileType.EPUB,
}


class Lock:
    def is_held(self) -> bool:
        return False


def _source(path: str, version: str | None = None) -> Source:
    suffix = path.rsplit(".", 1)[-1] if "." in path else ""
    file_type = _TYPES.get(suffix, FileType.UNSUPPORTED)
    return Source(path, path, file_type, parser_version=version)


def _record(
    manifest: InMemoryManifest,
    path: str,
    status: str,
    error: str | None = None,
    version: str | None = None,
) -> None:
    manifest.record_source(_source(path, version), status, error)


def _status(manifest: Any, versions: Any = None) -> InspectStatus:
    return InspectStatus(
        lock=Lock(),
        open_manifest=lambda: manifest,
        open_store=lambda: None,
        store_mode="local",
        runtime_config={},
        parser_versions=versions,
    )


def _mixed() -> InMemoryManifest:
    manifest = InMemoryManifest()
    _record(manifest, "a.txt", "indexed")
    _record(manifest, "b.md", "indexed")
    _record(manifest, "scan.pdf", "skipped:requires_ocr")
    _record(manifest, "broken.pdf", "skipped:requires_ocr", "garbled_text_layer")
    _record(manifest, "c.docx", "skipped:unsupported")
    _record(manifest, "d.docx", "skipped:unsupported")
    _record(manifest, "README", "skipped:unsupported")
    _record(manifest, "odd.txt", "skipped:something_new")
    _record(
        manifest, "bad.epub", "failed:parse_error", "EPUB is not a valid\nZIP  archive"
    )
    return manifest


def test_counts_per_state_reason_and_extension() -> None:
    sources = _status(_mixed()).run().as_dict()["sources"]

    assert sources["total"] == 9
    assert sources["by_state"] == {"indexed": 2, "skipped": 6, "failed": 1}
    assert sources["skipped_by_reason"] == {
        "requires_ocr": 1,
        "garbled_text_layer": 1,
        "unsupported": 3,
        "other": 1,
    }
    assert sources["skipped_by_extension"] == [
        {"extension": ".docx", "sources": 2},
        {"extension": ".pdf", "sources": 2},
        {"extension": "(none)", "sources": 1},
        {"extension": ".txt", "sources": 1},
    ]
    assert sources["failed"] == [
        {
            "path": "bad.epub",
            "category": "parse_error",
            "reason": "EPUB is not a valid ZIP archive",
        }
    ]
    assert sources["parser_changed"] is None


def test_only_the_ten_most_common_skipped_extensions_are_listed() -> None:
    manifest = InMemoryManifest()
    for index in range(12):
        for copy in range(12 - index):
            _record(manifest, f"f{index}-{copy}.x{index:02d}", "skipped:unsupported")

    listed = _status(manifest).run().as_dict()["sources"]["skipped_by_extension"]

    assert [entry["extension"] for entry in listed] == [
        f".x{index:02d}" for index in range(10)
    ]
    assert [entry["sources"] for entry in listed] == list(range(12, 2, -1))


def test_the_failed_list_stops_at_twenty_in_path_order() -> None:
    manifest = InMemoryManifest()
    for index in range(25):
        _record(manifest, f"f{index:02d}.pdf", "failed:store_error", "x" * 300)

    sources = _status(manifest).run().as_dict()["sources"]

    assert sources["by_state"]["failed"] == 25
    assert [f["path"] for f in sources["failed"]] == [
        f"f{index:02d}.pdf" for index in range(20)
    ]
    assert all(len(f["reason"]) == 200 for f in sources["failed"])


def test_parser_changes_follow_the_update_rule_with_legacy_rows() -> None:
    manifest = InMemoryManifest()
    _record(manifest, "legacy.txt", "indexed", version=None)  # counts as 1
    _record(manifest, "current.txt", "indexed", version="1")
    _record(manifest, "old.pdf", "indexed", version="2")  # changed
    _record(manifest, "legacy.pdf", "indexed", version="")  # 1 != 3: changed
    _record(manifest, "current.pdf", "indexed", version="3")
    _record(manifest, "scan.pdf", "skipped:requires_ocr", version="2")  # changed
    _record(manifest, "failed.pdf", "failed:parse_error", "x", version="1")
    _record(manifest, "c.docx", "skipped:unsupported")
    _record(manifest, "gone.md", "indexed", version="1")  # parser gone: changed
    versions = {FileType.TXT: "1", FileType.PDF: "3", FileType.EPUB: None}

    sources = _status(manifest, versions).run().as_dict()["sources"]

    assert sources["parser_changed"] == 4


def test_an_unreadable_manifest_gives_no_block_and_no_error() -> None:
    def broken() -> Any:
        raise OSError("disk error")

    report = _status(None).run()
    assert report.sources is None and report.as_dict()["sources"] is None
    unreadable = InspectStatus(
        lock=Lock(),
        open_manifest=broken,
        open_store=lambda: None,
        store_mode="local",
        runtime_config={},
    ).run()
    assert unreadable.index_state == "manifest_unreadable"
    assert unreadable.as_dict()["sources"] is None


class SourcesFail(InMemoryManifest):
    def committed_sources(self) -> Any:
        raise OSError("table is corrupt")


def test_unreadable_sources_alone_give_no_block_and_no_warning() -> None:
    report = _status(SourcesFail()).run()

    assert report.sources is None
    assert report.index_state != "manifest_unreadable"
    assert report.warnings == _status(InMemoryManifest()).run().warnings


def _cli(status: InspectStatus, *args: str) -> str:
    out = io.StringIO()
    with redirect_stdout(out), redirect_stderr(io.StringIO()):
        assert main(["status", *args], status_factory=lambda settings: status) == 0
    return out.getvalue()


def test_the_text_output_shows_counts_reasons_and_failures() -> None:
    manifest = _mixed()
    for index in range(22):
        _record(manifest, f"z{index:02d}.pdf", "failed:store_error", "store down")

    text = _cli(_status(manifest, {FileType.PDF: "3"}))

    assert "Sources: 31 total, 2 indexed, 6 skipped, 23 failed" in text
    assert (
        "Skipped by reason: requires_ocr 1, garbled_text_layer 1, unsupported 3, "
        "other 1"
    ) in text
    assert "Failed sources (20 of 23 listed):" in text
    assert "  bad.epub: parse_error: EPUB is not a valid ZIP archive" in text
    assert "  z18.pdf: store_error: store down" in text
    assert "z19.pdf" not in text


def test_the_text_output_names_parser_changes_only_when_there_are_some() -> None:
    manifest = InMemoryManifest()
    _record(manifest, "old.pdf", "indexed", version="2")

    assert "1 to re-extract for a changed parser version" in _cli(
        _status(manifest, {FileType.PDF: "3"})
    )
    assert "re-extract" not in _cli(_status(manifest, {FileType.PDF: "2"}))


def test_the_json_output_carries_the_block() -> None:
    data = json.loads(_cli(_status(_mixed(), {}), "--json"))

    assert list(data["sources"]) == [
        "total",
        "by_state",
        "skipped_by_reason",
        "skipped_by_extension",
        "failed",
        "parser_changed",
    ]
    assert data["sources"]["by_state"]["indexed"] == 2
