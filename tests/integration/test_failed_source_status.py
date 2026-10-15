"""An update persists every per-source failure on the committed manifest row,
whatever the row said before, so status counts it as failed and the next
update retries it (UC-4, UC-7)."""

from pathlib import Path
from typing import Any

from docgrep.adapters.manifest import SQLiteManifest
from docgrep.api.container import create_status
from docgrep.domain.models import FileType, ParsedDocument, Source
from epub_support import epub_bytes, xhtml
from resume_support import close, dump_points, settings_for
from update_support import NEW, OLD, update_services, write_files

CORRUPT = bytes(range(256)) * 8
BOOK = epub_bytes([("a.xhtml", xhtml("<h1>Start</h1><p>marker5 readable words</p>"))])
BLANK = epub_bytes([("a.xhtml", xhtml("<p> </p>"))])


def _settings(tmp_path: Path, files: dict[str, tuple[bytes, int]]) -> Any:
    corpus = tmp_path / "corpus"
    write_files(corpus, {"note.txt": (b"marker1 plain words here", OLD), **files})
    return settings_for(corpus, tmp_path / "data")


def _build(settings: Any, *, epub: bool = True) -> Any:
    services = update_services(settings)
    build: Any = services.build
    if not epub:
        build.parsers = {
            t: p for t, p in build.parsers.items() if t is not FileType.EPUB
        }
    try:
        return build.run()
    finally:
        close(services)


def _update(settings: Any, parser: Any = None) -> Any:
    services = update_services(settings)
    update: Any = services.update
    if parser is not None:
        update.parsers = {**update.parsers, FileType.EPUB: parser}
    try:
        return update.run()
    finally:
        close(services)


def _row(settings: Any, path: str) -> Any:
    record = SQLiteManifest(settings.data_dir).get_source(path)
    assert record is not None
    return record


def _plan(summary: Any) -> dict[str, int]:
    plan = summary.plan
    return {
        key: getattr(plan, key)
        for key in (
            "added",
            "modified",
            "removed",
            "unchanged",
            "retried",
            "content_changed",
            "parser_changed",
            "newly_supported",
        )
    }


def _sources(settings: Any) -> dict[str, Any]:
    sources: dict[str, Any] = create_status(settings).run().as_dict()["sources"]
    return sources


def _assert_failed_once(settings: Any, summary: Any, path: str) -> None:
    (failure,) = summary.failures
    assert (failure.path, failure.reason) == (path, "parse_error")
    record = _row(settings, path)
    assert record.status == "failed:parse_error"
    assert record.error == failure.message
    assert str(settings.corpus_root) not in record.error


def test_an_unsupported_row_that_now_fails_is_recorded_as_failed(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, {"broken.epub": (CORRUPT, OLD)})
    _build(settings, epub=False)
    assert _row(settings, "broken.epub").status == "skipped:unsupported"

    first = _update(settings)

    assert _plan(first) == {
        **dict.fromkeys(_plan(first), 0),
        "modified": 1,
        "newly_supported": 1,
        "unchanged": 1,
    }
    _assert_failed_once(settings, first, "broken.epub")
    assert "not a valid ZIP archive" in _row(settings, "broken.epub").error
    sources = _sources(settings)
    assert sources["total"] == 2
    assert sources["by_state"] == {"indexed": 1, "skipped": 0, "failed": 1}
    assert [f["path"] for f in sources["failed"]] == ["broken.epub"]
    assert sources["skipped_by_reason"]["unsupported"] == 0

    second = _update(settings)

    assert _plan(second) == {
        **dict.fromkeys(_plan(second), 0),
        "retried": 1,
        "unchanged": 1,
    }
    assert (second.indexed, second.skipped, second.failed) == (0, 0, 1)
    assert _row(settings, "broken.epub").status == "failed:parse_error"


def test_a_requires_ocr_row_whose_reparse_fails_is_recorded_as_failed(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, {"blank.epub": (BLANK, OLD)})
    _build(settings)
    assert _row(settings, "blank.epub").status == "skipped:requires_ocr"
    write_files(settings.corpus_root, {"blank.epub": (CORRUPT, NEW)})

    summary = _update(settings)

    assert (summary.plan.modified, summary.plan.content_changed) == (1, 1)
    _assert_failed_once(settings, summary, "blank.epub")
    assert _row(settings, "blank.epub").source.mtime == NEW
    assert _update(settings).plan.retried == 1


def test_an_indexed_source_modified_into_a_corrupt_file_is_recorded_as_failed(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, {"book.epub": (BOOK, OLD)})
    _build(settings)
    assert _row(settings, "book.epub").status == "indexed"
    assert [p for p in dump_points(settings) if p[1]["path"] == "book.epub"]
    write_files(settings.corpus_root, {"book.epub": (CORRUPT, NEW)})

    summary = _update(settings)

    assert (summary.plan.modified, summary.plan.content_changed) == (1, 1)
    _assert_failed_once(settings, summary, "book.epub")
    assert _row(settings, "book.epub").source.indexed_at is None
    # The old passages stay searchable (ADR-0013) until an update succeeds.
    assert [p for p in dump_points(settings) if p[1]["path"] == "book.epub"]
    sources = _sources(settings)
    assert sources["by_state"] == {"indexed": 1, "skipped": 0, "failed": 1}
    second = _update(settings)
    assert (second.plan.retried, second.plan.modified) == (1, 0)


class _FailsWith:
    """An EPUB parser that always fails with the given message."""

    version = "1"

    def __init__(self, message: str) -> None:
        self.message = message

    def parse(self, source: Source) -> ParsedDocument:
        raise ValueError(self.message)


def test_a_failed_row_that_fails_again_keeps_one_refreshed_row(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, {"book.epub": (CORRUPT, OLD)})
    _build(settings)
    assert _row(settings, "book.epub").status == "failed:parse_error"
    assert "not a valid ZIP archive" in _row(settings, "book.epub").error

    summary = _update(settings, _FailsWith("second failure text"))

    assert summary.plan.retried == 1
    record = _row(settings, "book.epub")
    assert (record.status, record.error) == (
        "failed:parse_error",
        "second failure text",
    )
    paths = [
        r.source.path for r in SQLiteManifest(settings.data_dir).committed_sources()
    ]
    assert sorted(paths) == ["book.epub", "note.txt"]


def test_an_added_corrupt_source_is_recorded_as_failed(tmp_path: Path) -> None:
    settings = _settings(tmp_path, {})
    _build(settings)
    write_files(settings.corpus_root, {"new.epub": (CORRUPT, NEW)})

    summary = _update(settings)

    assert summary.plan.added == 1
    _assert_failed_once(settings, summary, "new.epub")
    second = _update(settings)
    assert (second.plan.retried, second.plan.added) == (1, 0)
