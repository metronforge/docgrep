"""SQLite manifest persistence across connections and process lifetimes."""

import sqlite3
from pathlib import Path

import pytest

from docgrep.adapters.manifest import SQLiteManifest
from docgrep.domain.models import FileType, Source
from docgrep.domain.protocols import ManifestReader, ManifestWriter


def test_manifest_creates_schema_and_wal_idempotently(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    SQLiteManifest(data_dir)
    SQLiteManifest(data_dir)

    with sqlite3.connect(data_dir / "manifest.sqlite3") as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {"runs", "sources"} <= tables


def test_manifest_round_trips_runs_and_sources(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    manifest = SQLiteManifest(data_dir)
    reader: ManifestReader = manifest
    writer: ManifestWriter = manifest
    assert reader.latest_run() is None
    assert reader.get_source("papers/a.pdf") is None

    first_id = writer.start_run({"model": "fake", "chunk_size": 512})
    running = reader.latest_run()
    assert running is not None
    assert running.run_id == first_id
    assert running.status == "running"
    assert running.kind == "build"
    assert running.config == (("chunk_size", 512), ("model", "fake"))
    assert running.started_at is not None
    assert running.finished_at is None

    source = Source(
        "source-a",
        "papers/a.pdf",
        FileType.PDF,
        content_hash="abc123",
        mtime=123.5,
        size=42,
        parser_version="pdf-v1",
    )
    writer.record_source(source, "indexed", embedding_model="fake")
    writer.finish_run(first_id, "completed", {"indexed": 1, "failed": 0})

    reopened = SQLiteManifest(data_dir)
    record = reopened.get_source("papers/a.pdf")
    assert record is not None
    assert record.source.source_id == "source-a"
    assert record.source.file_type is FileType.PDF
    assert record.source.content_hash == "abc123"
    assert record.source.mtime == 123.5
    assert record.source.size == 42
    assert record.source.parser_version == "pdf-v1"
    assert record.source.indexed_at is not None
    assert record.embedding_model == "fake"
    assert record.status == "indexed"
    assert record.error is None

    completed = reopened.latest_run()
    assert completed is not None
    assert completed.run_id == first_id
    assert completed.status == "completed"
    assert completed.finished_at is not None
    assert completed.totals == (("failed", 0), ("indexed", 1))


def test_manifest_checkpoints_and_preserves_latest_outcome(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    manifest = SQLiteManifest(data_dir)
    first_id = manifest.start_run({"model": "fake"})
    source = Source("source-a", "a.txt", FileType.TXT, content_hash="first")
    manifest.record_source(source, "failed:parse_error", "bad input")
    assert SQLiteManifest(data_dir).get_source("a.txt") is not None

    manifest.finish_run(first_id, "aborted", {"failed": 1})
    second_id = manifest.start_run({"model": "fake"})
    manifest.record_source(
        Source("source-a", "a.txt", FileType.TXT, content_hash="second"),
        "skipped:requires_ocr",
    )
    manifest.finish_run(second_id, "completed", {"skipped": 1})

    reopened = SQLiteManifest(data_dir)
    record = reopened.get_source("a.txt")
    assert record is not None
    assert record.source.content_hash == "second"
    assert record.status == "skipped:requires_ocr"
    assert record.error is None
    latest = reopened.latest_run()
    assert latest is not None
    assert latest.run_id == second_id
    assert latest.status == "completed"


def test_manifest_rejects_unknown_or_repeated_run_finish(tmp_path: Path) -> None:
    manifest = SQLiteManifest(tmp_path)
    with pytest.raises(ValueError, match="unknown or finished run_id"):
        manifest.finish_run("99", "completed", {})

    run_id = manifest.start_run({})
    manifest.finish_run(run_id, "completed", {})
    with pytest.raises(ValueError, match="unknown or finished run_id"):
        manifest.finish_run(run_id, "aborted", {})
