"""SQLite-backed durable indexing manifest."""

import json
import sqlite3
from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from docgrep.domain.errors import ManifestUnavailable
from docgrep.domain.models import (
    FileType,
    ManifestRunRecord,
    ManifestSourceRecord,
    RunStatus,
    Source,
)

# Columns added to the runs table after the first schema; added when missing.
_RUN_COLUMNS = (
    ("staging_collection", "TEXT"),
    ("resumed_from", "TEXT"),
    ("cancel_requested_at", "TEXT"),
    ("total_sources", "INTEGER"),
    ("carried_sources", "INTEGER"),
)

_SOURCE_COLUMNS = (
    "path, source_id, hash, mtime, size, file_type, status, error, "
    "parser_version, embedding_model, indexed_at, title, author, year"
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SQLiteManifest:
    """Commit each run or source outcome to a SQLite file in the data directory."""

    def __init__(self, data_dir: Path, *, read_only: bool = False) -> None:
        """Open the manifest; ``read_only`` neither creates nor migrates anything.

        A read-only manifest only runs SELECT statements and tolerates a schema
        that predates the newest columns.
        """
        self.path = data_dir / "manifest.sqlite3"
        if read_only:
            if not self.path.is_file():
                raise ManifestUnavailable(
                    "No manifest exists", details={"path": str(self.path)}
                )
            return
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
                CREATE TABLE IF NOT EXISTS run_sources (
                    run_id INTEGER NOT NULL,
                    path TEXT NOT NULL,
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
                    year INTEGER,
                    passages INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (run_id, path)
                );
                """
            )
            existing = {
                row["name"] for row in connection.execute("PRAGMA table_info(runs)")
            }
            for column, kind in _RUN_COLUMNS:
                if column not in existing:
                    connection.execute(f"ALTER TABLE runs ADD COLUMN {column} {kind}")

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

    def start_run(
        self,
        config: Mapping[str, str | int | float],
        *,
        resumed_from: str | None = None,
    ) -> str:
        snapshot = json.dumps(dict(config), sort_keys=True)
        with self._connection() as connection:
            cursor = connection.execute(
                "INSERT INTO runs (kind, status, started_at, config_snapshot, totals,"
                " resumed_from) VALUES (?, ?, ?, ?, ?, ?)",
                ("build", "running", _now(), snapshot, "{}", resumed_from),
            )
            assert cursor.lastrowid is not None
            return str(cursor.lastrowid)

    def record_source(
        self,
        source: Source,
        status: str,
        error: str | None = None,
        embedding_model: str | None = None,
        *,
        run_id: str | None = None,
        passages: int = 0,
    ) -> None:
        if status != "indexed" and not status.startswith(("skipped:", "failed:")):
            raise ValueError("invalid source status")
        indexed_at = (source.indexed_at or _now()) if status == "indexed" else None
        values = (
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
        )
        with self._connection() as connection:
            if run_id is not None:
                connection.execute(
                    f"""
                    INSERT INTO run_sources (run_id, {_SOURCE_COLUMNS}, passages)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, path) DO UPDATE SET
                        source_id=excluded.source_id, hash=excluded.hash,
                        mtime=excluded.mtime, size=excluded.size,
                        file_type=excluded.file_type, status=excluded.status,
                        error=excluded.error, parser_version=excluded.parser_version,
                        embedding_model=excluded.embedding_model,
                        indexed_at=excluded.indexed_at, title=excluded.title,
                        author=excluded.author, year=excluded.year,
                        passages=excluded.passages
                    """,
                    (run_id, *values, passages),
                )
                return
            connection.execute(
                f"""
                INSERT INTO sources ({_SOURCE_COLUMNS})
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    source_id=excluded.source_id, hash=excluded.hash,
                    mtime=excluded.mtime, size=excluded.size,
                    file_type=excluded.file_type, status=excluded.status,
                    error=excluded.error, parser_version=excluded.parser_version,
                    embedding_model=excluded.embedding_model,
                    indexed_at=excluded.indexed_at, title=excluded.title,
                    author=excluded.author, year=excluded.year
                """,
                values,
            )

    def set_staging_collection(self, run_id: str, name: str) -> None:
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE runs SET staging_collection=? WHERE id=?", (name, run_id)
            )
            if cursor.rowcount != 1:
                raise ValueError("unknown run_id")

    def set_total_sources(self, run_id: str, total: int, *, carried: int = 0) -> None:
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE runs SET total_sources=?, carried_sources=? WHERE id=?",
                (total, carried, run_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("unknown run_id")

    def carry_over(
        self, from_run_id: str, to_run_id: str, paths: Collection[str]
    ) -> None:
        wanted = sorted(paths)
        with self._connection() as connection:
            for start in range(0, len(wanted), 500):
                group = wanted[start : start + 500]
                marks = ", ".join("?" for _ in group)
                connection.execute(
                    f"INSERT OR REPLACE INTO run_sources "
                    f"(run_id, {_SOURCE_COLUMNS}, passages) "
                    f"SELECT ?, {_SOURCE_COLUMNS}, passages FROM run_sources "
                    f"WHERE run_id=? AND path IN ({marks})",
                    (to_run_id, from_run_id, *group),
                )

    def request_cancel(self, run_id: str) -> str:
        with self._connection() as connection:
            connection.execute(
                "UPDATE runs SET cancel_requested_at=? "
                "WHERE id=? AND cancel_requested_at IS NULL",
                (_now(), run_id),
            )
            row = connection.execute(
                "SELECT cancel_requested_at FROM runs WHERE id=?", (run_id,)
            ).fetchone()
        if row is None:
            raise ValueError("unknown run_id")
        return cast(str, row["cancel_requested_at"])

    def cancel_requested_at(self, run_id: str) -> str | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT cancel_requested_at FROM runs WHERE id=?", (run_id,)
            ).fetchone()
        return None if row is None else cast(str | None, row["cancel_requested_at"])

    def mark_swapped(self, run_id: str) -> None:
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE runs SET status='swapped' "
                "WHERE id=? AND status IN ('running', 'aborted')",
                (run_id,),
            )
            if cursor.rowcount != 1:
                raise ValueError("unknown or already finished run_id")

    def commit_run(self, run_id: str) -> None:
        """Replace the committed sources with the run's journal in one transaction."""
        with self._connection() as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM runs WHERE id=?", (run_id,)
                ).fetchone()
                is None
            ):
                raise ValueError("unknown run_id")
            connection.execute("DELETE FROM sources")
            connection.execute(
                f"INSERT INTO sources ({_SOURCE_COLUMNS}) "
                f"SELECT {_SOURCE_COLUMNS} FROM run_sources WHERE run_id=?",
                (run_id,),
            )

    def finish_run(
        self,
        run_id: str,
        status: Literal["completed", "aborted", "cancelled"],
        totals: Mapping[str, int],
    ) -> None:
        if status not in ("completed", "aborted", "cancelled"):
            raise ValueError("invalid run status")
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE runs SET status=?, finished_at=?, totals=? "
                "WHERE id=? AND status IN ('running', 'swapped')",
                (status, _now(), json.dumps(dict(totals), sort_keys=True), run_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("unknown or finished run_id")

    @staticmethod
    def _optional(row: sqlite3.Row, name: str) -> object:
        return row[name] if name in row.keys() else None

    @staticmethod
    def _record(row: sqlite3.Row, passages: int | None = None) -> ManifestSourceRecord:
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
            passages,
        )

    def get_source(self, path: str) -> ManifestSourceRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM sources WHERE path=?", (path,)
            ).fetchone()
        return None if row is None else self._record(row)

    def run_sources(self, run_id: str) -> tuple[ManifestSourceRecord, ...]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM run_sources WHERE run_id=? ORDER BY path", (run_id,)
            ).fetchall()
        return tuple(self._record(row, cast(int, row["passages"])) for row in rows)

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
            status=cast(RunStatus, row["status"]),
            config=tuple(sorted(config.items())),
            totals=tuple(sorted(totals.items())),
            kind=cast(str, row["kind"]),
            started_at=cast(str, row["started_at"]),
            finished_at=cast(str | None, row["finished_at"]),
            staging_collection=cast(
                str | None, self._optional(row, "staging_collection")
            ),
            resumed_from=cast(str | None, self._optional(row, "resumed_from")),
            cancel_requested_at=cast(
                str | None, self._optional(row, "cancel_requested_at")
            ),
            total_sources=cast(int | None, self._optional(row, "total_sources")),
            carried_sources=cast(int | None, self._optional(row, "carried_sources"))
            or 0,
        )
