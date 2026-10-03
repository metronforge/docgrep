"""A build that died right after the alias swap is finished, not rebuilt."""

import logging
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from docgrep.adapters.manifest import SQLiteManifest
from docgrep.api.container import create_status
from docgrep.domain.errors import BuildFailed, NothingToResume, StagingInconsistent
from resume_support import (
    Crash,
    close,
    dump_points,
    fail_at_swapped_mark,
    new_services,
    search_snapshot,
    settings_for,
    write_corpus,
)

SOURCES = 5
VARIANTS = [("running", True), ("aborted", False)]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("corpus")
    write_corpus(root, SOURCES)
    return root


def _searched(settings: Any) -> list[Any]:
    services = new_services(settings)
    try:
        return search_snapshot(services)
    finally:
        close(services)


@pytest.fixture(scope="module")
def reference(
    corpus: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[list[Any], list[Any]]:
    settings = settings_for(corpus, tmp_path_factory.mktemp("clean"))
    services = new_services(settings)
    services.build.run()
    close(services)
    return dump_points(settings), _searched(settings)


def _died_after_swap(settings: Any, *, crash: bool) -> None:
    services = new_services(settings)
    fail_at_swapped_mark(services, crash=crash)
    with pytest.raises(Crash if crash else BuildFailed):
        services.build.run()
    close(services)


def _listing(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


@pytest.mark.parametrize(("status", "crash"), VARIANTS)
def test_status_then_resume_finish_a_run_that_died_after_the_swap(
    corpus: Path,
    tmp_path: Path,
    reference: tuple[list[Any], list[Any]],
    status: str,
    crash: bool,
) -> None:
    settings = settings_for(corpus, tmp_path / "data")
    _died_after_swap(settings, crash=crash)
    manifest = SQLiteManifest(settings.data_dir)
    run = manifest.latest_run()
    assert run is not None and run.status == status
    assert manifest.get_source("doc000.txt") is None  # not committed yet

    before = _listing(tmp_path)
    report = create_status(settings).run().as_dict()
    assert _listing(tmp_path) == before  # status wrote nothing
    assert report["index_state"] == "swap_pending_commit"
    assert report["latest_run"]["state"] == "swapped"
    assert report["latest_run"]["status"] == status  # still what is stored
    assert "swap_pending_commit" in [w["code"] for w in report["warnings"]]
    assert report["collection"]["passage_count"] > SOURCES
    latest = SQLiteManifest(settings.data_dir).latest_run()
    assert latest is not None and latest.status == status

    services = new_services(settings)
    summary = services.build.resume()
    close(services)

    assert summary.recovered and summary.run_id == run.run_id
    assert (summary.indexed, summary.failed) == (SOURCES, 0)
    done = SQLiteManifest(settings.data_dir)
    finished = done.latest_run()
    assert finished is not None and finished.status == "completed"
    journal = {r.source.path: r for r in done.run_sources(run.run_id)}
    assert len(journal) == SOURCES
    for path, record in journal.items():
        committed = done.get_source(path)
        assert committed is not None and committed.status == record.status
        assert committed.source.content_hash == record.source.content_hash
    assert dump_points(settings) == reference[0]
    assert _searched(settings) == reference[1]
    assert create_status(settings).run().index_state == "ready"

    again = new_services(settings)
    with pytest.raises(NothingToResume):
        again.build.resume()
    close(again)


@pytest.mark.parametrize(("status", "crash"), VARIANTS)
def test_plain_index_recovers_the_run_the_same_way_and_proceeds(
    corpus: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    status: str,
    crash: bool,
) -> None:
    settings = settings_for(corpus, tmp_path / "data")
    _died_after_swap(settings, crash=crash)

    services = new_services(settings)
    with caplog.at_level(logging.WARNING, logger="docgrep"):
        summary = services.build.run(refuse_resumable=True)
    close(services)

    recovered = [r for r in caplog.records if r.getMessage() == "run_recovered"]
    assert len(recovered) == 1 and recovered[0].levelno == logging.WARNING
    assert recovered[0].__dict__["reason"] == "alias_target_reached"
    assert recovered[0].__dict__["previous_status"] == status
    manifest = SQLiteManifest(settings.data_dir)
    assert summary.run_id == "2" and not summary.cancelled
    assert manifest.latest_run() is not None
    first = SQLiteManifest(settings.data_dir)
    with first._connection() as connection:
        rows = connection.execute("SELECT id, status FROM runs ORDER BY id").fetchall()
    assert [tuple(r) for r in rows] == [(1, "completed"), (2, "completed")]


def test_a_journal_shorter_than_the_plan_is_still_rejected(
    corpus: Path, tmp_path: Path
) -> None:
    settings = settings_for(corpus, tmp_path / "data")
    _died_after_swap(settings, crash=True)
    with sqlite3.connect(settings.data_dir / "manifest.sqlite3") as connection:
        connection.execute("DELETE FROM run_sources WHERE path='doc004.txt'")

    report = create_status(settings).run()
    assert report.index_state == "interrupted_pending_resume"

    services = new_services(settings)
    with pytest.raises(StagingInconsistent) as raised:
        services.build.resume()
    close(services)
    assert "status" in str(raised.value) and "--restart" in str(raised.value)
    run = SQLiteManifest(settings.data_dir).latest_run()
    assert run is not None and run.status == "running"  # nothing was changed


def test_a_run_without_a_recorded_plan_needs_a_non_empty_journal(
    corpus: Path, tmp_path: Path
) -> None:
    settings = settings_for(corpus, tmp_path / "data")
    _died_after_swap(settings, crash=True)
    with sqlite3.connect(settings.data_dir / "manifest.sqlite3") as connection:
        connection.execute("UPDATE runs SET total_sources=NULL")
    services = new_services(settings)
    assert services.build.resume().recovered  # complete journal: recovered
    close(services)

    other = tmp_path / "other"
    other_settings = settings_for(corpus, other / "data")
    _died_after_swap(other_settings, crash=True)
    with sqlite3.connect(other_settings.data_dir / "manifest.sqlite3") as connection:
        connection.execute("UPDATE runs SET total_sources=NULL")
        connection.execute("DELETE FROM run_sources")
    empty = new_services(other_settings)
    with pytest.raises(StagingInconsistent):
        empty.build.resume()
    close(empty)
