"""Parser versions recorded by builds and updates, and re-extraction on a
version change, against real SQLite and local-mode Qdrant (UC-3, UC-4)."""

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from docgrep.adapters.filelock import FileIndexLock
from docgrep.adapters.manifest import SQLiteManifest
from docgrep.domain.errors import IndexLockHeld
from docgrep.domain.models import FileType
from docgrep.settings import Settings
from resume_support import close, dump_points, search_snapshot, settings_for
from update_support import (
    FINAL,
    INITIAL,
    QUERIES,
    cancel_after_sources,
    change_corpus,
    sources_table,
    update_services,
    with_version,
    without_epub,
    write_files,
)

TEXT_FILES = sorted(name for name in FINAL if name.endswith(".txt"))


def _snapshot(settings: Settings) -> tuple[Any, ...]:
    services = update_services(settings)
    try:
        searched = search_snapshot(services, QUERIES)
    finally:
        close(services)
    return dump_points(settings), searched, sources_table(settings)


def _build(settings: Settings, **versions: str | None) -> None:
    services = update_services(settings)
    for name, version in versions.items():
        with_version(services.build, FileType(name), version)
    try:
        assert services.build.run().failed == 0
    finally:
        close(services)


def _update(settings: Settings, after: int | None = None, **versions: str) -> Any:
    services = update_services(settings)
    assert services.update is not None
    for name, version in versions.items():
        with_version(services.update, FileType(name), version)
    if after is not None:
        cancel_after_sources(services, after)
    try:
        return services.update.run()
    finally:
        close(services)


def _final(tmp_path: Path, name: str) -> Settings:
    corpus = tmp_path / f"{name}-corpus"
    write_files(corpus, FINAL)
    return settings_for(corpus, tmp_path / f"{name}-data")


@pytest.fixture(scope="module")
def bumped_reference(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, ...]:
    """A clean build of the final corpus with the text parser at version 2."""
    settings = _final(tmp_path_factory.mktemp("ref"), "clean")
    _build(settings, txt="2")
    return _snapshot(settings)


def test_a_build_records_the_declared_versions(tmp_path: Path) -> None:
    settings = _final(tmp_path, "built")

    _build(settings)

    versions = {
        record.source.path: (record.status, record.source.parser_version)
        for record in SQLiteManifest(settings.data_dir).committed_sources()
    }
    assert versions.pop("misc/letter.doc") == ("skipped:unsupported", None)
    assert set(versions.values()) == {("indexed", "1")}
    assert {point[1]["parser_version"] for point in dump_points(settings)} == {"1"}


def test_an_update_records_the_declared_versions(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    write_files(corpus, INITIAL)
    settings = settings_for(corpus, tmp_path / "data")
    services = update_services(settings)
    without_epub(services)
    services.build.run()
    close(services)
    change_corpus(corpus)

    _update(settings)

    manifest = SQLiteManifest(settings.data_dir)
    for path in ("books/one.epub", "books/three.epub", "doc10.txt", "notes/a.md"):
        record = manifest.get_source(path)
        assert record is not None and record.source.parser_version == "1", path


def test_a_version_bump_reprocesses_only_its_file_type(
    tmp_path: Path, bumped_reference: tuple[Any, ...]
) -> None:
    settings = _final(tmp_path, "live")
    _build(settings)

    summary = _update(settings, txt="2")

    plan = summary.plan
    assert (plan.parser_changed, plan.modified) == (len(TEXT_FILES), len(TEXT_FILES))
    assert (plan.content_changed, plan.newly_supported, plan.retried) == (0, 0, 0)
    assert (plan.added, plan.removed) == (0, 0)
    assert plan.unchanged == len(FINAL) - len(TEXT_FILES)
    assert summary.failed == 0
    assert _snapshot(settings) == bumped_reference
    assert _update(settings, txt="2").no_changes


@pytest.mark.parametrize("after", [0, 1, 3, len(TEXT_FILES)])
def test_a_cancelled_version_bump_continues_on_the_next_update(
    tmp_path: Path, bumped_reference: tuple[Any, ...], after: int
) -> None:
    settings = _final(tmp_path, "live")
    _build(settings)

    cancelled = _update(settings, after=after, txt="2")
    finished = _update(settings, txt="2")

    assert cancelled.cancelled and cancelled.modified == after
    assert finished.plan.parser_changed == len(TEXT_FILES) - after
    assert _snapshot(settings) == bumped_reference


def test_the_plan_counts_each_reason(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    write_files(corpus, INITIAL)
    settings = settings_for(corpus, tmp_path / "data")
    services = update_services(settings)
    without_epub(services)
    services.build.run()
    close(services)
    change_corpus(corpus)

    summary = _update(settings, txt="2")

    plan = summary.plan
    # doc01 changed; doc02 (touched only), doc03, doc04 and doc05 have the
    # bumped parser; the two EPUBs only gained a parser.
    assert (plan.content_changed, plan.parser_changed, plan.newly_supported) == (
        2,
        4,
        2,
    )
    assert plan.modified == 8 and (plan.added, plan.removed) == (2, 2)


def test_legacy_rows_without_a_version_are_not_reprocessed(tmp_path: Path) -> None:
    settings = _final(tmp_path, "legacy")
    # Parsers that declare nothing, as before versions existed.
    _build(settings, txt=None, markdown=None, epub=None)
    manifest_file = settings.data_dir / "manifest.sqlite3"
    with sqlite3.connect(manifest_file) as connection:
        connection.execute(
            "UPDATE sources SET parser_version='' WHERE path LIKE 'notes/%'"
        )
    recorded = {
        r.source.parser_version
        for r in SQLiteManifest(settings.data_dir).committed_sources()
        if r.status == "indexed"
    }
    assert recorded == {None, ""}

    summary = _update(settings)

    assert summary.no_changes
    assert summary.plan.unchanged == len(FINAL)


@pytest.mark.parametrize("operation", ["build", "update"])
def test_the_lock_refusal_names_the_blocked_operation(
    tmp_path: Path, operation: str
) -> None:
    settings = _final(tmp_path, "locked")
    _build(settings)
    services = update_services(settings)
    try:
        with FileIndexLock(settings.data_dir / "index.lock").hold():
            with pytest.raises(IndexLockHeld) as refused:
                if operation == "build":
                    services.build.run()
                else:
                    assert services.update is not None
                    services.update.run()
    finally:
        close(services)

    assert f"so the {operation} cannot start" in str(refused.value)
