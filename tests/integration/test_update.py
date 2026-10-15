"""Incremental update against real SQLite and local-mode Qdrant (UC-4, ADR-0013).

The key invariant: building the initial corpus without an EPUB parser, then
changing the corpus and updating with the parser registered, gives the same
stored points, search results and committed sources as a clean build of the
final corpus, also after a cancel or a failure followed by another update.
"""

from pathlib import Path
from typing import Any

import pytest

from docgrep.adapters.filelock import FileIndexLock
from docgrep.adapters.manifest import SQLiteManifest
from docgrep.api.container import create_status
from docgrep.domain.errors import (
    BuildFailed,
    ConfigDrift,
    IndexIncomplete,
    IndexLockHeld,
    NothingToResume,
    ResumeAvailable,
)
from docgrep.domain.models import FileType, ParsedDocument, Source
from docgrep.settings import Settings
from docgrep.usecases.update_index import UpdateSummary
from resume_support import (
    Crash,
    HookedEmbedder,
    cancel_at_embed_call,
    close,
    dump_points,
    search_snapshot,
    settings_for,
)
from update_support import (
    CHANGES,
    FINAL,
    INITIAL,
    PLAN,
    QUERIES,
    HookedManifest,
    cancel_after_sources,
    change_corpus,
    fail_live_upsert_at,
    sources_table,
    update_services,
    without_epub,
    write_files,
)

Reference = tuple[list[Any], list[Any], list[Any]]


def _searched(settings: Settings) -> list[Any]:
    """Search in a fresh container, so both sides read vectors back from disk."""
    services = update_services(settings)
    try:
        return search_snapshot(services, QUERIES)
    finally:
        close(services)


def _snapshot(settings: Settings) -> Reference:
    return dump_points(settings), _searched(settings), sources_table(settings)


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory) -> Reference:
    """A clean full build of the final corpus, EPUB parser included."""
    corpus = tmp_path_factory.mktemp("final")
    write_files(corpus, FINAL)
    settings = settings_for(corpus, tmp_path_factory.mktemp("clean"))
    services = update_services(settings)
    services.build.run()
    close(services)
    snapshot = _snapshot(settings)
    points, searched, sources = snapshot
    assert len(points) > 20 and all(searched)
    assert {row[0] for row in sources} == set(FINAL)
    assert any(row[2] is FileType.EPUB and row[10] == "indexed" for row in sources)
    return snapshot


def _initial_index(tmp_path: Path) -> Settings:
    """Build INITIAL without an EPUB parser, then change the files to FINAL."""
    corpus = tmp_path / "corpus"
    write_files(corpus, INITIAL)
    settings = settings_for(corpus, tmp_path / "data")
    services = update_services(settings)
    without_epub(services)
    summary = services.build.run()
    close(services)
    assert (summary.indexed, summary.skipped, summary.failed) == (6 + 2, 3, 0)
    change_corpus(corpus)
    return settings


def _update(settings: Settings, **kwargs: Any) -> UpdateSummary:
    services = update_services(settings)
    try:
        assert services.update is not None
        return services.update.run(**kwargs)
    finally:
        close(services)


def _plan(summary: UpdateSummary) -> dict[str, int]:
    plan = summary.plan
    return {
        "added": plan.added,
        "modified": plan.modified,
        "removed": plan.removed,
        "unchanged": plan.unchanged,
        "retried": plan.retried,
        "refreshed": plan.refreshed,
    }


def test_an_update_equals_a_clean_build_of_the_final_corpus(
    tmp_path: Path, reference: Reference
) -> None:
    settings = _initial_index(tmp_path)

    summary = _update(settings)

    assert _plan(summary) == PLAN
    assert not summary.cancelled and summary.failed == 0
    assert (summary.added, summary.modified, summary.removed) == (2, 4, 2)
    assert (summary.indexed, summary.skipped, summary.unchanged) == (6, 0, 5)
    assert summary.passages_written > 0 and summary.passages_removed > 0
    assert _snapshot(settings) == reference
    run = SQLiteManifest(settings.data_dir).latest_run()
    assert run is not None and (run.kind, run.status) == ("update", "completed")
    assert run.total_sources == CHANGES
    totals = dict(run.totals)
    assert (totals["added"], totals["removed"], totals["failed"]) == (2, 2, 0)


def test_a_second_update_is_a_no_op(tmp_path: Path, reference: Reference) -> None:
    settings = _initial_index(tmp_path)
    _update(settings)
    before = dump_points(settings)

    again = _update(settings)

    assert again.no_changes
    assert _plan(again) == {**dict.fromkeys(PLAN, 0), "unchanged": len(FINAL)}
    assert again.passages_written == again.passages_removed == 0
    assert dump_points(settings) == before
    assert _snapshot(settings) == reference
    metadata = _metadata(settings)
    assert "last_updated_at" in metadata
    run = SQLiteManifest(settings.data_dir).latest_run()
    assert run is not None and run.status == "completed" and run.total_sources == 0


def _metadata(settings: Settings) -> dict[str, Any]:
    services = update_services(settings)
    try:
        build: Any = services.build
        return dict(build.passage_writer.read_completion_metadata())
    finally:
        close(services)


def test_the_marker_keeps_its_build_keys_and_records_the_update(
    tmp_path: Path,
) -> None:
    settings = _initial_index(tmp_path)
    built = _metadata(settings)

    _update(settings)

    updated = _metadata(settings)
    assert set(updated) == set(built) | {"last_updated_at"}
    assert {k: updated[k] for k in built} == built


@pytest.mark.parametrize("after", range(CHANGES + 1))
def test_a_cancelled_update_continues_to_the_same_index(
    tmp_path: Path, reference: Reference, after: int
) -> None:
    settings = _initial_index(tmp_path)
    services = update_services(settings)
    cancel_after_sources(services, after)
    assert services.update is not None

    cancelled = services.update.run()
    close(services)

    assert cancelled.cancelled
    processed = cancelled.added + cancelled.modified + cancelled.removed
    assert processed == after
    run = SQLiteManifest(settings.data_dir).latest_run()
    assert run is not None and (run.kind, run.status) == ("update", "cancelled")
    assert "last_updated_at" not in _metadata(settings)

    finished = _update(settings)

    assert not finished.cancelled
    assert finished.plan.changes == CHANGES - after
    assert _snapshot(settings) == reference


def test_a_cancel_between_embedding_batches_of_a_source(
    tmp_path: Path, reference: Reference
) -> None:
    settings = _initial_index(tmp_path)
    services = update_services(settings)
    update: Any = services.update
    update.batch_size = 1
    # Removals embed nothing; the second embed call is inside the first
    # modified source, so the cancel lands between two of its batches.
    update.embedder = HookedEmbedder(update.embedder, 2, update.request_cancel)

    cancelled = update.run()
    close(services)

    assert cancelled.cancelled and cancelled.modified == 0 and cancelled.removed == 2
    assert _update(settings).plan.changes == CHANGES - 2
    assert _snapshot(settings) == reference


@pytest.mark.parametrize("call", [1, 2, 4, 7])
def test_a_failure_inside_a_source_is_repaired_by_the_next_update(
    tmp_path: Path, reference: Reference, call: int
) -> None:
    settings = _initial_index(tmp_path)
    services = update_services(settings)
    update: Any = services.update
    update.upsert_batch_size = 2  # several writes per source: fail mid-source
    fail_live_upsert_at(services, call)

    failed = update.run()
    close(services)

    assert not failed.cancelled and failed.failed > 0
    assert {f.reason for f in failed.failures} == {"store_error"}
    run = SQLiteManifest(settings.data_dir).latest_run()
    assert run is not None and run.status == "completed"

    repaired = _update(settings)

    assert repaired.failed == 0
    assert _snapshot(settings) == reference


def test_a_crash_inside_a_source_is_repaired_by_the_next_update(
    tmp_path: Path, reference: Reference
) -> None:
    settings = _initial_index(tmp_path)
    services = update_services(settings)
    update: Any = services.update
    update.upsert_batch_size = 2
    fail_live_upsert_at(services, 3, Crash())

    with pytest.raises(Crash):
        update.run()
    close(services)
    dead = SQLiteManifest(settings.data_dir).latest_run()
    assert dead is not None and (dead.kind, dead.status) == ("update", "running")

    _update(settings)

    manifest = SQLiteManifest(settings.data_dir)
    closed = manifest.latest_run_of_kind("update")
    assert closed is not None and closed.status == "completed"
    assert _snapshot(settings) == reference


class _Versioned:
    """The real parser of a type, declaring a parser version."""

    def __init__(self, inner: Any, version: str) -> None:
        self.inner = inner
        self.version = version

    def parse(self, source: Source) -> ParsedDocument:
        document: ParsedDocument = self.inner.parse(source)
        return document


def test_a_parser_version_change_reprocesses_only_its_file_type(
    tmp_path: Path,
) -> None:
    settings = _initial_index(tmp_path)
    _update(settings)
    services = update_services(settings)
    update: Any = services.update
    markdown = update.parsers[FileType.MARKDOWN]
    update.parsers = {**update.parsers, FileType.MARKDOWN: _Versioned(markdown, "2")}

    summary = update.run()
    close(services)

    assert _plan(summary) == {
        **dict.fromkeys(PLAN, 0),
        "modified": 1,
        "unchanged": len(FINAL) - 1,
    }
    record = SQLiteManifest(settings.data_dir).get_source("notes/a.md")
    assert record is not None and record.source.parser_version == "2"
    points = [p for p in dump_points(settings) if p[1]["path"] == "notes/a.md"]
    assert points and {p[1]["parser_version"] for p in points} == {"2"}

    # Unchanged version: nothing to do.
    services = update_services(settings)
    update = services.update
    update.parsers = {**update.parsers, FileType.MARKDOWN: _Versioned(markdown, "2")}
    assert update.run().no_changes
    close(services)


def test_skipped_unsupported_sources_are_indexed_once_a_parser_exists(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    write_files(corpus, INITIAL)
    settings = settings_for(corpus, tmp_path / "data")
    services = update_services(settings)
    without_epub(services)
    services.build.run()
    close(services)
    manifest = SQLiteManifest(settings.data_dir)
    record = manifest.get_source("books/one.epub")
    assert record is not None and record.status == "skipped:unsupported"

    summary = _update(settings)

    assert _plan(summary) == {
        **dict.fromkeys(PLAN, 0),
        "modified": 2,
        "unchanged": len(INITIAL) - 2,
    }
    record = manifest.get_source("books/one.epub")
    assert record is not None and record.status == "indexed"
    assert record.source.title == "One"
    docx = manifest.get_source("misc/letter.doc")
    assert docx is not None and docx.status == "skipped:unsupported"


class _FailsFor:
    """The real parser, except that it fails for one path."""

    def __init__(self, inner: Any, path: str) -> None:
        self.inner = inner
        self.path = path
        # It stands in for the real parser, declared version included.
        self.version = inner.version

    def parse(self, source: Source) -> ParsedDocument:
        if source.path == self.path:
            raise ValueError("injected parse failure")
        document: ParsedDocument = self.inner.parse(source)
        return document


def _built_with_a_failure(tmp_path: Path) -> Settings:
    corpus = tmp_path / "corpus"
    write_files(corpus, FINAL)
    settings = settings_for(corpus, tmp_path / "data")
    services = update_services(settings)
    build: Any = services.build
    text = build.parsers[FileType.TXT]
    build.parsers = {**build.parsers, FileType.TXT: _FailsFor(text, "doc03.txt")}
    assert build.run().failed == 1
    close(services)
    return settings


def test_failed_sources_are_retried(tmp_path: Path, reference: Reference) -> None:
    settings = _built_with_a_failure(tmp_path)

    summary = _update(settings)

    assert _plan(summary) == {
        **dict.fromkeys(PLAN, 0),
        "retried": 1,
        "unchanged": len(FINAL) - 1,
    }
    assert summary.retried == 1 and summary.indexed == 1
    assert _snapshot(settings) == reference


def test_retrying_failed_sources_can_be_disabled(tmp_path: Path) -> None:
    settings = _built_with_a_failure(tmp_path)

    summary = _update(settings, retry_failed=False)

    assert summary.no_changes and summary.plan.unchanged == len(FINAL)
    record = SQLiteManifest(settings.data_dir).get_source("doc03.txt")
    assert record is not None and record.status == "failed:parse_error"


def test_a_modified_source_that_fails_stays_searchable(tmp_path: Path) -> None:
    settings = _initial_index(tmp_path)
    old_hash = SQLiteManifest(settings.data_dir).get_source("doc01.txt")
    services = update_services(settings)
    update: Any = services.update
    text = update.parsers[FileType.TXT]
    update.parsers = {**update.parsers, FileType.TXT: _FailsFor(text, "doc01.txt")}

    summary = update.run()

    assert summary.failed == 1
    assert [(f.path, f.reason) for f in summary.failures] == [
        ("doc01.txt", "parse_error")
    ]
    hits = services.search.run("marker1", top_n=3).results
    assert hits and hits[0].passage.source_id == "doc01.txt"
    assert hits[0].passage.text.startswith("marker1 ")
    close(services)
    manifest = SQLiteManifest(settings.data_dir)
    assert manifest.get_source("doc01.txt") == old_hash
    run = manifest.latest_run()
    assert run is not None
    (journaled,) = [
        r for r in manifest.run_sources(run.run_id) if r.source.path == "doc01.txt"
    ]
    assert journaled.status == "failed:parse_error"

    again = _update(settings)

    assert again.plan.modified == 1 and again.failed == 0
    record = manifest.get_source("doc01.txt")
    assert record is not None and record.source.content_hash != (
        old_hash.source.content_hash if old_hash else None
    )


def test_refused_without_a_completed_index(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    write_files(corpus, INITIAL)
    settings = settings_for(corpus, tmp_path / "data")

    with pytest.raises(IndexIncomplete):
        _update(settings)

    assert SQLiteManifest(settings.data_dir).latest_run() is None


def test_refused_when_a_setting_that_decides_vectors_changed(tmp_path: Path) -> None:
    settings = _initial_index(tmp_path)
    before = SQLiteManifest(settings.data_dir).latest_run()

    with pytest.raises(ConfigDrift) as refused:
        _update(settings.model_copy(update={"chunk_size": 12}))

    assert refused.value.category == "config_drift"
    assert refused.value.details["fields"] == "chunk_size"
    assert "docgrep index" in str(refused.value)
    assert SQLiteManifest(settings.data_dir).latest_run() == before


def test_refused_while_the_latest_build_can_be_resumed(tmp_path: Path) -> None:
    settings = _initial_index(tmp_path)
    services = update_services(settings)
    cancel_at_embed_call(services, 2)
    assert services.build.run().cancelled
    close(services)
    before = SQLiteManifest(settings.data_dir).latest_run()

    with pytest.raises(ResumeAvailable) as refused:
        _update(settings)

    assert refused.value.category == "resume_available"
    assert "docgrep index --resume" in str(refused.value)
    assert "--restart" in str(refused.value)
    assert SQLiteManifest(settings.data_dir).latest_run() == before


def test_refused_while_another_operation_holds_the_lock(tmp_path: Path) -> None:
    settings = _initial_index(tmp_path)

    with FileIndexLock(settings.data_dir / "index.lock").hold():
        with pytest.raises(IndexLockHeld):
            _update(settings)

    assert SQLiteManifest(settings.data_dir).latest_run_of_kind("update") is None


def _disk_error() -> None:
    raise OSError("manifest disk error")


@pytest.mark.parametrize("how", ["cancelled", "aborted", "interrupted"])
def test_an_unfinished_update_blocks_nothing(tmp_path: Path, how: str) -> None:
    settings = _initial_index(tmp_path)
    services = update_services(settings)
    update: Any = services.update
    if how == "cancelled":
        cancel_after_sources(services, 1)
        assert update.run().cancelled
    elif how == "aborted":
        # A manifest failure is not a per-source failure: it aborts the run.
        update.manifest = HookedManifest(update.manifest, 1, _disk_error)
        with pytest.raises(BuildFailed):
            update.run()
    else:
        fail_live_upsert_at(services, 1, Crash())
        with pytest.raises(Crash):
            update.run()
    close(services)
    run = SQLiteManifest(settings.data_dir).latest_run()
    expected = "running" if how == "interrupted" else how
    assert run is not None and (run.kind, run.status) == ("update", expected)
    report = create_status(settings).run()
    assert report.index_state == "ready"
    assert "resume_available" not in {w.code for w in report.warnings}

    services = update_services(settings)
    try:
        with pytest.raises(NothingToResume):
            services.build.resume()
        build: Any = services.build
        assert not build.run(refuse_resumable=True).cancelled
    finally:
        close(services)
    assert _update(settings).no_changes
