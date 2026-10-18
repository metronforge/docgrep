"""The run journal records the source a copy was made from (ADR-0028); the
committed sources do not."""

import sqlite3
from pathlib import Path

import pytest

from docgrep.adapters.manifest import SQLiteManifest
from docgrep.domain.models import FileType, Source


def test_the_journal_keeps_where_passages_were_reused_from(tmp_path: Path) -> None:
    manifest = SQLiteManifest(tmp_path)
    source = Source("b.txt", "b.txt", FileType.TXT, content_hash="h", size=1)
    run_id = manifest.start_run({}, kind="update")

    manifest.record_source(source, "indexed", run_id=run_id, reused_from="a.txt")
    resumed = manifest.start_run({})
    manifest.carry_over(run_id, resumed, ["b.txt"])
    manifest.commit_run(resumed)

    (journaled,) = manifest.run_sources(run_id)
    (carried,) = manifest.run_sources(resumed)
    committed = manifest.get_source("b.txt")
    assert journaled.reused_from == carried.reused_from == "a.txt"
    assert committed is not None and committed.reused_from is None
    with pytest.raises(ValueError):
        manifest.record_source(source, "indexed", reused_from="a.txt")


def test_an_older_journal_gets_the_column(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "manifest.sqlite3")
    connection.execute(
        "CREATE TABLE run_sources (run_id INTEGER NOT NULL, path TEXT NOT NULL, "
        "source_id TEXT NOT NULL, hash TEXT, mtime REAL, size INTEGER, "
        "file_type TEXT NOT NULL, status TEXT NOT NULL, error TEXT, "
        "parser_version TEXT, embedding_model TEXT, indexed_at TEXT, title TEXT, "
        "author TEXT, year INTEGER, passages INTEGER NOT NULL DEFAULT 0, "
        "PRIMARY KEY (run_id, path))"
    )
    connection.execute(
        "INSERT INTO run_sources (run_id, path, source_id, file_type, status) "
        "VALUES (1, 'a.txt', 'a.txt', 'txt', 'indexed')"
    )
    connection.commit()
    connection.close()

    assert SQLiteManifest(tmp_path).run_sources("1")[0].reused_from is None
