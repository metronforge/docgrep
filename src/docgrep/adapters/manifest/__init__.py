"""SQLite-backed durable indexing manifest."""

import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from docgrep.domain.errors import ManifestUnavailable
from docgrep.domain.models import (
    FileType,
    ManifestRunRecord,
    ManifestSourceRecord,
    Source,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SQLiteManifest:
    """Commit each run or source outcome to a SQLite file in the data directory."""

    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / "manifest.sqlite3"
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ManifestUnavailable(
                "Cannot create manifest directory", details={"path": str(data_dir)}
            ) from exc
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    config_snapshot TEXT NOT NULL,
                    totals TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sources (
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
            )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(self.path)
            connection.row_factory = sqlite3.Row
            with connection:
                yield connection
        except sqlite3.Error as exc:
            raise ManifestUnavailable(
                "Manifest database operation failed", details={"path": str(self.path)}
            ) from exc
        finally:
            if connection is not None:
                connection.close()

    def start_run(self, config: Mapping[str, str | int | float]) -> str:
        snapshot = json.dumps(dict(config), sort_keys=True)
        with self._connection() as connection:
            cursor = connection.execute(
                "INSERT INTO runs (kind, status, started_at, config_snapshot, totals) "
                "VALUES (?, ?, ?, ?, ?)",
                ("build", "running", _now(), snapshot, "{}"),
            )
            assert cursor.lastrowid is not None
            return str(cursor.lastrowid)

    def record_source(
        self,
        source: Source,
        status: str,
        error: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        if status != "indexed" and not status.startswith(("skipped:", "failed:")):
            raise ValueError("invalid source status")
        indexed_at = (source.indexed_at or _now()) if status == "indexed" else None
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO sources (
                    path, source_id, hash, mtime, size, file_type, status, error,
                    parser_version, embedding_model, indexed_at, title, author, year
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    source_id=excluded.source_id, hash=excluded.hash,
                    mtime=excluded.mtime, size=excluded.size,
                    file_type=excluded.file_type, status=excluded.status,
                    error=excluded.error, parser_version=excluded.parser_version,
                    embedding_model=excluded.embedding_model,
                    indexed_at=excluded.indexed_at, title=excluded.title,
                    author=excluded.author, year=excluded.year
                """,
                (
                    source.path,
                    source.source_id,
                    source.content_hash,
                    source.mtime,
                    source.size,
                    source.file_type.value,
                    status,
                    error,
                    source.parser_version,
                    embedding_model,
                    indexed_at,
                    source.title,
                    source.author,
                    source.year,
                ),
            )

    def finish_run(
        self,
        run_id: str,
        status: Literal["completed", "aborted"],
        totals: Mapping[str, int],
    ) -> None:
        if status not in ("completed", "aborted"):
            raise ValueError("invalid run status")
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE runs SET status=?, finished_at=?, totals=? "
                "WHERE id=? AND status='running'",
                (status, _now(), json.dumps(dict(totals), sort_keys=True), run_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("unknown or finished run_id")

    def get_source(self, path: str) -> ManifestSourceRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM sources WHERE path=?", (path,)
            ).fetchone()
        if row is None:
            return None
        source = Source(
            source_id=cast(str, row["source_id"]),
            path=cast(str, row["path"]),
            file_type=FileType(cast(str, row["file_type"])),
            title=cast(str | None, row["title"]),
            author=cast(str | None, row["author"]),
            year=cast(int | None, row["year"]),
            content_hash=cast(str | None, row["hash"]),
            mtime=cast(float | None, row["mtime"]),
            size=cast(int | None, row["size"]),
            indexed_at=cast(str | None, row["indexed_at"]),
            parser_version=cast(str | None, row["parser_version"]),
        )
        return ManifestSourceRecord(
            source,
            cast(str, row["status"]),
            cast(str | None, row["error"]),
            cast(str | None, row["embedding_model"]),
        )

    def latest_run(self) -> ManifestRunRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        config = cast(dict[str, str | int | float], json.loads(row["config_snapshot"]))
        totals = cast(dict[str, int], json.loads(row["totals"]))
        return ManifestRunRecord(
            run_id=str(row["id"]),
            status=cast(Literal["running", "completed", "aborted"], row["status"]),
            config=tuple(sorted(config.items())),
            totals=tuple(sorted(totals.items())),
            kind=cast(str, row["kind"]),
            started_at=cast(str, row["started_at"]),
            finished_at=cast(str | None, row["finished_at"]),
        )
