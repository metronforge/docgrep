"""SQLite-backed durable indexing manifest."""

import json
import logging
import sqlite3
from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from docgrep.adapters.corpus import as_roots, file_content_hash
from docgrep.domain.errors import ManifestUnavailable
from docgrep.domain.models import (
    FileType,
    ManifestRunRecord,
    ManifestSourceRecord,
    OcrDelegation,
    OcrMode,
    OcrOutput,
    OcrRecord,
    RunStatus,
    Source,
)
from docgrep.domain.roots import CorpusRoots

_LOG = logging.getLogger(__name__)

# The kinds of run recorded in the runs table.
RUN_KINDS = ("build", "update", "sync", "ocr", "reencode_sparse")

# Columns added to the runs table after the first schema; added when missing.
_RUN_COLUMNS = (
    ("staging_collection", "TEXT"),
    ("resumed_from", "TEXT"),
    ("cancel_requested_at", "TEXT"),
    ("total_sources", "INTEGER"),
    ("carried_sources", "INTEGER"),
    ("checkpoint", "TEXT"),
)

# Columns added to the run journal after the first schema; added when missing.
_JOURNAL_COLUMNS = (("reused_from", "TEXT"),)

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
                CREATE TABLE IF NOT EXISTS ocr_registry (
                    input_path TEXT PRIMARY KEY,
                    input_format TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    output_hash TEXT NOT NULL,
                    engine TEXT NOT NULL,
                    engine_version TEXT NOT NULL,
                    languages TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ocr_registry_input_hash
                    ON ocr_registry (input_hash);
                CREATE INDEX IF NOT EXISTS ocr_registry_output_path
                    ON ocr_registry (output_path);
                CREATE TABLE IF NOT EXISTS ocr_delegations (
                    input_path TEXT PRIMARY KEY,
                    input_hash TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            existing = {
                row["name"] for row in connection.execute("PRAGMA table_info(runs)")
            }
            for column, kind in _RUN_COLUMNS:
                if column not in existing:
                    connection.execute(f"ALTER TABLE runs ADD COLUMN {column} {kind}")
            journal = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(run_sources)")
            }
            for column, kind in _JOURNAL_COLUMNS:
                if column not in journal:
                    connection.execute(
                        f"ALTER TABLE run_sources ADD COLUMN {column} {kind}"
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

    def start_run(
        self,
        config: Mapping[str, str | int | float],
        *,
        resumed_from: str | None = None,
        kind: str = "build",
    ) -> str:
        if kind not in RUN_KINDS:
            raise ValueError("invalid run kind")
        snapshot = json.dumps(dict(config), sort_keys=True)
        with self._connection() as connection:
            cursor = connection.execute(
                "INSERT INTO runs (kind, status, started_at, config_snapshot, totals,"
                " resumed_from) VALUES (?, ?, ?, ?, ?, ?)",
                (kind, "running", _now(), snapshot, "{}", resumed_from),
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
        reused_from: str | None = None,
    ) -> None:
        if reused_from is not None and run_id is None:
            raise ValueError("reused_from is journaled only")
        if status not in ("indexed", "removed") and not status.startswith(
            ("skipped:", "failed:")
        ):
            # An OCR run journals its own outcomes for the progress report.
            if run_id is None or status not in ("processed", "cancelled"):
                raise ValueError("invalid source status")
        if status == "removed" and run_id is None:
            raise ValueError("a removal is journaled; use remove_source")
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
                    INSERT INTO run_sources
                        (run_id, {_SOURCE_COLUMNS}, passages, reused_from)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, path) DO UPDATE SET
                        source_id=excluded.source_id, hash=excluded.hash,
                        mtime=excluded.mtime, size=excluded.size,
                        file_type=excluded.file_type, status=excluded.status,
                        error=excluded.error, parser_version=excluded.parser_version,
                        embedding_model=excluded.embedding_model,
                        indexed_at=excluded.indexed_at, title=excluded.title,
                        author=excluded.author, year=excluded.year,
                        passages=excluded.passages,
                        reused_from=excluded.reused_from
                    """,
                    (run_id, *values, passages, reused_from),
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

    def remove_source(self, path: str) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM sources WHERE path=?", (path,))

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
                    f"(run_id, {_SOURCE_COLUMNS}, passages, reused_from) "
                    f"SELECT ?, {_SOURCE_COLUMNS}, passages, reused_from "
                    f"FROM run_sources "
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

    def set_checkpoint(self, run_id: str, checkpoint: str) -> None:
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE runs SET checkpoint=? WHERE id=?", (checkpoint, run_id)
            )
            if cursor.rowcount != 1:
                raise ValueError("unknown run_id")

    @staticmethod
    def _optional(row: sqlite3.Row, name: str) -> object:
        return row[name] if name in row.keys() else None

    @classmethod
    def _record(
        cls, row: sqlite3.Row, passages: int | None = None
    ) -> ManifestSourceRecord:
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
            # Journal rows only; absent from a journal older than the column.
            cast(str | None, cls._optional(row, "reused_from")),
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

    def committed_sources(self) -> tuple[ManifestSourceRecord, ...]:
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM sources ORDER BY path").fetchall()
        return tuple(self._record(row) for row in rows)

    def latest_run(self) -> ManifestRunRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return None if row is None else self._run(row)

    def latest_run_of_kind(self, kind: str) -> ManifestRunRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE kind=? ORDER BY id DESC LIMIT 1", (kind,)
            ).fetchone()
        return None if row is None else self._run(row)

    def _run(self, row: sqlite3.Row) -> ManifestRunRecord:
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
            checkpoint=cast(str | None, self._optional(row, "checkpoint")),
        )


class SQLiteOcrRegistry:
    """The OCR registry and the delegation table in the manifest database.

    Liveness is checked on disk through the corpus roots: an entry needs its
    output file, a delegation also needs the original's unchanged content. A
    file whose root is not configured here is not live. Reads never raise;
    they log at DEBUG and read as nothing found.
    """

    def __init__(
        self,
        data_dir: Path,
        corpus_root: CorpusRoots | Path,
        *,
        read_only: bool = False,
    ) -> None:
        """Open the registry; unless ``read_only``, create or migrate the
        database first, exactly as the manifest does."""
        self.path = data_dir / "manifest.sqlite3"
        self.roots = as_roots(corpus_root)
        if not read_only:
            SQLiteManifest(data_dir)

    def _read(self, query: str, args: tuple[object, ...]) -> list[sqlite3.Row]:
        if not self.path.is_file():
            return []
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(self.path)
            connection.row_factory = sqlite3.Row
            return connection.execute(query, args).fetchall()
        except sqlite3.Error:
            _LOG.debug("ocr_registry_unreadable", exc_info=True)
            return []
        finally:
            if connection is not None:
                connection.close()

    def _exists(self, path: str) -> bool:
        file = self.roots.locate(path)
        try:
            return file is not None and file.is_file()
        except OSError:
            return False

    @staticmethod
    def _record(row: sqlite3.Row) -> OcrRecord:
        return OcrRecord(
            input_path=cast(str, row["input_path"]),
            input_format=cast(Literal["pdf", "djvu"], row["input_format"]),
            input_hash=cast(str, row["input_hash"]),
            output_path=cast(str, row["output_path"]),
            output_hash=cast(str, row["output_hash"]),
            engine=cast(str, row["engine"]),
            engine_version=cast(str, row["engine_version"]),
            languages=cast(str, row["languages"]),
            mode=cast(OcrMode, row["mode"]),
            created_at=cast(str, row["created_at"]),
        )

    def find_by_input_hash(self, content_hash: str) -> OcrRecord | None:
        rows = self._read(
            "SELECT * FROM ocr_registry WHERE input_hash=? ORDER BY input_path",
            (content_hash,),
        )
        for row in rows:
            if self._exists(cast(str, row["output_path"])):
                return self._record(row)
        return None

    def output_owner(self, output_path: str) -> str | None:
        rows = self._read(
            "SELECT input_path FROM ocr_registry WHERE output_path=? "
            "ORDER BY input_path",
            (output_path,),
        )
        if not rows or not self._exists(output_path):
            return None
        return cast(str, rows[0]["input_path"])

    def live_outputs(self) -> dict[str, OcrOutput]:
        rows = self._read(
            "SELECT output_path, input_path, created_at FROM ocr_registry "
            "ORDER BY output_path, input_path",
            (),
        )
        outputs: dict[str, OcrOutput] = {}
        for row in rows:
            output = cast(str, row["output_path"])
            if output not in outputs and self._exists(output):
                outputs[output] = OcrOutput(
                    output,
                    cast(str, row["input_path"]),
                    cast(str | None, row["created_at"]),
                )
        return outputs

    def live_delegation(
        self, path: str, content_hash: str | None = None
    ) -> OcrDelegation | None:
        rows = self._read("SELECT * FROM ocr_delegations WHERE input_path=?", (path,))
        if not rows:
            return None
        row = rows[0]
        delegation = OcrDelegation(
            input_path=cast(str, row["input_path"]),
            input_hash=cast(str, row["input_hash"]),
            output_path=cast(str, row["output_path"]),
            created_at=cast(str, row["created_at"]),
        )
        if not self._exists(delegation.output_path):
            return None
        if content_hash is None:
            file = self.roots.locate(path)
            if file is None:
                return None
            try:
                content_hash = file_content_hash(file)
            except OSError:
                return None
        return delegation if content_hash == delegation.input_hash else None

    def register(self, record: OcrRecord) -> None:
        created = record.created_at or _now()
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(self.path)
            with connection:
                connection.execute(
                    "INSERT OR REPLACE INTO ocr_registry (input_path, input_format, "
                    "input_hash, output_path, output_hash, engine, engine_version, "
                    "languages, mode, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.input_path,
                        record.input_format,
                        record.input_hash,
                        record.output_path,
                        record.output_hash,
                        record.engine,
                        record.engine_version,
                        record.languages,
                        record.mode,
                        created,
                    ),
                )
                connection.execute(
                    "INSERT OR REPLACE INTO ocr_delegations (input_path, input_hash, "
                    "output_path, created_at) VALUES (?, ?, ?, ?)",
                    (record.input_path, record.input_hash, record.output_path, created),
                )
        except sqlite3.Error as exc:
            raise ManifestUnavailable(
                "OCR registry write failed", details={"path": str(self.path)}
            ) from exc
        finally:
            if connection is not None:
                connection.close()
