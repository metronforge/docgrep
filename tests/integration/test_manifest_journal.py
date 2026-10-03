"""Per-run journal, committed sources and the additive manifest migration."""

import sqlite3
from pathlib import Path

import pytest

from docgrep.adapters.manifest import SQLiteManifest
from docgrep.domain.models import FileType, Source

OLD_SCHEMA = """
CREATE TABLE runs (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    config_snapshot TEXT NOT NULL,
    totals TEXT NOT NULL
);
CREATE TABLE sources (
    path TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    hash TEXT,
    mtime REAL,
    size INTEGER,
    file_type TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    parser_version TEXT,
    embedding_model TEXT,
    indexed_at TEXT,
    title TEXT,
    author TEXT,
    year INTEGER
);
"""


def _source(path: str, title: str | None = None, year: int | None = None) -> Source:
    return Source(path, path, FileType.TXT, title=title, year=year)


def _columns(db: Path, table: str) -> set[str]:
    with sqlite3.connect(db) as connection:
        return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def test_migration_keeps_rows_of_the_previous_schema(tmp_path: Path) -> None:
    db = tmp_path / "manifest.sqlite3"
    with sqlite3.connect(db) as connection:
        connection.executescript(OLD_SCHEMA)
        connection.execute(
            "INSERT INTO runs (id, kind, status, started_at, finished_at,"
            " config_snapshot, totals) VALUES (7, 'build', 'completed',"
            " '2026-09-25T00:00:00+00:00', '2026-09-25T00:01:00+00:00',"
            " '{\"chunk_size\": 10}', '{\"indexed\": 1}')"
        )
        connection.execute(
            "INSERT INTO sources (path, source_id, hash, file_type, status,"
            " embedding_model, title, year) VALUES ('old.txt', 'old.txt', 'h1',"
            " 'txt', 'indexed', 'm', 'Old', 1999)"
        )
    assert "staging_collection" not in _columns(db, "runs")

    manifest = SQLiteManifest(tmp_path)

    assert {"staging_collection", "resumed_from", "cancel_requested_at"} <= _columns(
        db, "runs"
    )
    assert "run_sources" in {
        row[0]
        for row in sqlite3.connect(db).execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    record = manifest.get_source("old.txt")
    assert record is not None
    assert (record.status, record.source.title, record.source.year) == (
        "indexed",
        "Old",
        1999,
    )
    assert record.source.content_hash == "h1"
    run = manifest.latest_run()
    assert run is not None
    assert (run.run_id, run.status, run.totals) == ("7", "completed", (("indexed", 1),))
    assert run.config == (("chunk_size", 10),)
    assert (run.staging_collection, run.resumed_from, run.cancel_requested_at) == (
        None,
        None,
        None,
    )
    # The migrated file keeps working for new runs and for legacy writes.
    next_run = manifest.start_run({"chunk_size": 10})
    assert next_run == "8"
    manifest.set_staging_collection(next_run, "docgrep_abc")
    manifest.record_source(_source("legacy.txt"), "indexed")
    assert manifest.get_source("legacy.txt") is not None


def test_migration_is_idempotent(tmp_path: Path) -> None:
    SQLiteManifest(tmp_path)
    before = _columns(tmp_path / "manifest.sqlite3", "runs")

    SQLiteManifest(tmp_path)
    SQLiteManifest(tmp_path)

    assert _columns(tmp_path / "manifest.sqlite3", "runs") == before


def test_journal_entries_do_not_touch_the_committed_sources(tmp_path: Path) -> None:
    manifest = SQLiteManifest(tmp_path)
    manifest.record_source(_source("live.txt"), "indexed", embedding_model="m")
    run_id = manifest.start_run({})

    manifest.record_source(
        _source("live.txt", title="New"),
        "indexed",
        embedding_model="m2",
        run_id=run_id,
        passages=4,
    )
    manifest.record_source(
        _source("fresh.txt"), "failed:parse_error", "boom", run_id=run_id
    )

    live = manifest.get_source("live.txt")
    assert (
        live is not None and live.embedding_model == "m" and live.source.title is None
    )
    assert manifest.get_source("fresh.txt") is None
    journal = manifest.run_sources(run_id)
    assert [(r.source.path, r.status, r.error, r.passages) for r in journal] == [
        ("fresh.txt", "failed:parse_error", "boom", 0),
        ("live.txt", "indexed", None, 4),
    ]
    assert journal[1].source.title == "New" and journal[1].embedding_model == "m2"
    assert manifest.run_sources("999") == ()


def test_recording_a_source_twice_in_one_run_keeps_the_last_outcome(
    tmp_path: Path,
) -> None:
    manifest = SQLiteManifest(tmp_path)
    run_id = manifest.start_run({})

    manifest.record_source(
        _source("a.txt"), "failed:embedding_error", "x", run_id=run_id
    )
    manifest.record_source(_source("a.txt"), "indexed", run_id=run_id, passages=2)

    [only] = manifest.run_sources(run_id)
    assert (only.status, only.error, only.passages) == ("indexed", None, 2)


def test_commit_replaces_the_committed_sources_and_keeps_the_journal(
    tmp_path: Path,
) -> None:
    manifest = SQLiteManifest(tmp_path)
    manifest.record_source(_source("gone.txt"), "indexed")
    manifest.record_source(_source("kept.txt"), "indexed", embedding_model="old")
    run_id = manifest.start_run({})
    manifest.record_source(
        _source("kept.txt"), "indexed", embedding_model="new", run_id=run_id, passages=3
    )
    manifest.record_source(_source("added.txt"), "skipped:unsupported", run_id=run_id)
    other = manifest.start_run({})
    manifest.record_source(_source("other-run.txt"), "indexed", run_id=other)

    manifest.commit_run(run_id)

    assert manifest.get_source("gone.txt") is None
    assert manifest.get_source("other-run.txt") is None
    kept = manifest.get_source("kept.txt")
    assert kept is not None and kept.embedding_model == "new"
    added = manifest.get_source("added.txt")
    assert added is not None and added.status == "skipped:unsupported"
    assert [r.source.path for r in manifest.run_sources(run_id)] == [
        "added.txt",
        "kept.txt",
    ]
    manifest.commit_run(run_id)
    assert manifest.get_source("added.txt") is not None


def test_committing_an_empty_journal_empties_the_sources(tmp_path: Path) -> None:
    manifest = SQLiteManifest(tmp_path)
    manifest.record_source(_source("old.txt"), "indexed")
    run_id = manifest.start_run({})

    manifest.commit_run(run_id)

    assert manifest.get_source("old.txt") is None


def test_unknown_runs_are_rejected(tmp_path: Path) -> None:
    manifest = SQLiteManifest(tmp_path)
    manifest.record_source(_source("old.txt"), "indexed")

    with pytest.raises(ValueError, match="unknown run_id"):
        manifest.commit_run("42")
    with pytest.raises(ValueError, match="unknown run_id"):
        manifest.set_staging_collection("42", "docgrep_x")
    assert manifest.get_source("old.txt") is not None


def test_staging_collection_is_stored_on_the_run(tmp_path: Path) -> None:
    manifest = SQLiteManifest(tmp_path)
    run_id = manifest.start_run({})

    manifest.set_staging_collection(run_id, "docgrep_" + "a" * 32)

    run = manifest.latest_run()
    assert run is not None and run.staging_collection == "docgrep_" + "a" * 32
    assert run.resumed_from is None and run.cancel_requested_at is None
    assert SQLiteManifest(tmp_path).latest_run() == run


def test_invalid_statuses_are_rejected_for_the_journal_too(tmp_path: Path) -> None:
    manifest = SQLiteManifest(tmp_path)
    run_id = manifest.start_run({})

    with pytest.raises(ValueError, match="invalid source status"):
        manifest.record_source(_source("a.txt"), "done", run_id=run_id)
