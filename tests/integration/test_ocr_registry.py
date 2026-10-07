"""The OCR registry and delegations: SQLite adapter and in-memory fake alike."""

import sqlite3
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from blake3 import blake3

from docgrep.adapters.manifest import SQLiteManifest, SQLiteOcrRegistry
from docgrep.domain.errors import ManifestUnavailable
from docgrep.domain.models import FileType, OcrRecord, Source
from docgrep.domain.protocols import OcrRegistry
from fakes import InMemoryOcrRegistry

Factory = Callable[[Path, Path], OcrRegistry]


def _hash(path: Path) -> str:
    return blake3(path.read_bytes()).hexdigest()


def _sqlite(data_dir: Path, corpus: Path) -> OcrRegistry:
    return SQLiteOcrRegistry(data_dir, corpus)


def _memory(data_dir: Path, corpus: Path) -> OcrRegistry:
    return InMemoryOcrRegistry(
        exists=lambda path: (corpus / path).is_file(),
        content_hash=lambda path: _hash(corpus / path),
    )


@pytest.fixture(params=[_sqlite, _memory], ids=["sqlite", "memory"])
def factory(request: pytest.FixtureRequest) -> Factory:
    chosen: Factory = request.param
    return chosen


def _corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    (corpus / "books").mkdir(parents=True)
    (corpus / "books" / "scan.pdf").write_bytes(b"%PDF scan original")
    (corpus / "books" / "scan.ocr.pdf").write_bytes(b"%PDF searchable output")
    return corpus


def _record(corpus: Path, **changes: str) -> OcrRecord:
    values = {
        "input_path": "books/scan.pdf",
        "output_path": "books/scan.ocr.pdf",
        "engine": "ocrmypdf",
        "engine_version": "16.0.0",
        "languages": "rus+eng+ukr",
        **changes,
    }
    return OcrRecord(
        input_path=values["input_path"],
        input_format="pdf",
        input_hash=_hash(corpus / values["input_path"]),
        output_path=values["output_path"],
        output_hash=_hash(corpus / values["output_path"]),
        engine=values["engine"],
        engine_version=values["engine_version"],
        languages=values["languages"],
        mode="skip-text",
    )


def test_insert_and_lookup_by_input_hash(tmp_path: Path, factory: Factory) -> None:
    corpus = _corpus(tmp_path)
    registry = factory(tmp_path / "data", corpus)
    record = _record(corpus)

    assert registry.find_by_input_hash(record.input_hash) is None
    registry.register(record)

    found = registry.find_by_input_hash(record.input_hash)
    assert found is not None and found.created_at
    assert found == replace(record, created_at=found.created_at)
    assert registry.find_by_input_hash("0" * 64) is None
    assert registry.output_owner("books/scan.ocr.pdf") == "books/scan.pdf"
    assert registry.output_owner("books/scan.pdf") is None
    delegation = registry.live_delegation("books/scan.pdf")
    assert delegation is not None
    assert (delegation.input_hash, delegation.output_path) == (
        record.input_hash,
        "books/scan.ocr.pdf",
    )
    assert registry.live_delegation("books/scan.pdf", record.input_hash) == delegation
    assert registry.live_delegation("books/other.pdf") is None


def test_removing_the_output_ends_entry_and_delegation(
    tmp_path: Path, factory: Factory
) -> None:
    corpus = _corpus(tmp_path)
    registry = factory(tmp_path / "data", corpus)
    record = _record(corpus)
    registry.register(record)

    (corpus / "books" / "scan.ocr.pdf").unlink()

    assert registry.find_by_input_hash(record.input_hash) is None
    assert registry.output_owner("books/scan.ocr.pdf") is None
    assert registry.live_delegation("books/scan.pdf") is None


def test_a_changed_original_is_no_longer_delegated(
    tmp_path: Path, factory: Factory
) -> None:
    corpus = _corpus(tmp_path)
    registry = factory(tmp_path / "data", corpus)
    record = _record(corpus)
    registry.register(record)

    (corpus / "books" / "scan.pdf").write_bytes(b"%PDF a different scan")

    assert registry.live_delegation("books/scan.pdf") is None
    assert registry.live_delegation("books/scan.pdf", "f" * 64) is None
    # The entry itself stays findable by the content it was made from.
    assert registry.find_by_input_hash(record.input_hash) is not None


def test_registering_again_replaces_instead_of_duplicating(
    tmp_path: Path, factory: Factory
) -> None:
    corpus = _corpus(tmp_path)
    registry = factory(tmp_path / "data", corpus)
    registry.register(_record(corpus))
    (corpus / "books" / "scan.ocr.pdf").write_bytes(b"%PDF searchable output, redone")
    again = _record(corpus, engine_version="16.1.0", languages="eng")

    registry.register(again)
    registry.register(again)

    found = registry.find_by_input_hash(again.input_hash)
    assert found is not None
    assert (found.engine_version, found.languages, found.output_hash) == (
        "16.1.0",
        "eng",
        again.output_hash,
    )
    if isinstance(registry, SQLiteOcrRegistry):
        with sqlite3.connect(registry.path) as connection:
            counts = [
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("ocr_registry", "ocr_delegations")
            ]
        assert counts == [1, 1]


def test_same_content_at_two_paths_is_found_by_the_first_path(
    tmp_path: Path, factory: Factory
) -> None:
    corpus = _corpus(tmp_path)
    (corpus / "books" / "copy.pdf").write_bytes(b"%PDF scan original")
    (corpus / "books" / "copy.ocr.pdf").write_bytes(b"%PDF copy output")
    registry = factory(tmp_path / "data", corpus)
    registry.register(_record(corpus))
    registry.register(
        _record(corpus, input_path="books/copy.pdf", output_path="books/copy.ocr.pdf")
    )

    found = registry.find_by_input_hash(_hash(corpus / "books" / "scan.pdf"))

    assert found is not None and found.input_path == "books/copy.pdf"
    assert registry.live_delegation("books/scan.pdf") is not None
    assert registry.live_delegation("books/copy.pdf") is not None


def test_reads_of_a_missing_or_broken_database_find_nothing(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    missing = SQLiteOcrRegistry(tmp_path / "none", corpus, read_only=True)
    assert missing.find_by_input_hash("0" * 64) is None
    assert missing.live_delegation("books/scan.pdf") is None
    assert missing.output_owner("books/scan.ocr.pdf") is None
    assert not (tmp_path / "none").exists()

    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    (broken_dir / "manifest.sqlite3").write_bytes(b"not a database at all" * 100)
    broken = SQLiteOcrRegistry(broken_dir, corpus, read_only=True)
    assert broken.find_by_input_hash("0" * 64) is None
    assert broken.live_delegation("books/scan.pdf") is None


def test_a_failed_write_raises_manifest_unavailable(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    registry = SQLiteOcrRegistry(tmp_path / "data", corpus)
    with sqlite3.connect(registry.path) as connection:
        connection.execute("DROP TABLE ocr_delegations")

    with pytest.raises(ManifestUnavailable):
        registry.register(_record(corpus))

    # One transaction: the entry was not written without its delegation.
    assert registry.find_by_input_hash(_hash(corpus / "books" / "scan.pdf")) is None


def test_migration_from_the_previous_schema_keeps_every_row(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    data_dir = tmp_path / "data"
    manifest = SQLiteManifest(data_dir)
    run_id = manifest.start_run({"chunk_size": 512})
    source = Source("a.txt", "a.txt", FileType.TXT, content_hash="h", size=1)
    manifest.record_source(source, "indexed", run_id=run_id, passages=3)
    manifest.record_source(source, "indexed", embedding_model="m")
    manifest.finish_run(run_id, "completed", {"indexed": 1})
    with sqlite3.connect(data_dir / "manifest.sqlite3") as connection:
        # The schema of the previous release had no OCR tables.
        for name in ("ocr_registry", "ocr_delegations"):
            connection.execute(f"DROP TABLE {name}")
        before = {
            table: connection.execute(f"SELECT * FROM {table}").fetchall()
            for table in ("runs", "sources", "run_sources")
        }

    registry = SQLiteOcrRegistry(data_dir, corpus)
    registry.register(_record(corpus))

    with sqlite3.connect(data_dir / "manifest.sqlite3") as connection:
        after = {
            table: connection.execute(f"SELECT * FROM {table}").fetchall()
            for table in ("runs", "sources", "run_sources")
        }
    assert after == before
    reopened = SQLiteManifest(data_dir)
    latest = reopened.latest_run()
    assert latest is not None and latest.status == "completed"
    assert reopened.get_source("a.txt") is not None
    assert registry.live_delegation("books/scan.pdf") is not None


def test_ocr_runs_and_their_journal_are_accepted(tmp_path: Path) -> None:
    manifest = SQLiteManifest(tmp_path / "data")
    run_id = manifest.start_run({"languages": "eng"}, kind="ocr")
    source = Source("scan.pdf", "scan.pdf", FileType.PDF, content_hash="h")
    manifest.record_source(source, "processed", run_id=run_id)
    manifest.record_source(source, "cancelled", run_id=run_id)
    with pytest.raises(ValueError):
        manifest.record_source(source, "processed")
    with pytest.raises(ValueError):
        manifest.start_run({}, kind="something")

    run = manifest.latest_run_of_kind("ocr")
    assert run is not None and run.kind == "ocr"
    assert [r.status for r in manifest.run_sources(run_id)] == ["cancelled"]
