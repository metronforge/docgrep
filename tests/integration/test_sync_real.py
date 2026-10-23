"""Automatic sync over the real local-mode store, manifest and corpus adapters
with the deterministic backend (ADR-0029). The clock is injected and each
cycle is run explicitly."""

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from docgrep.adapters.filelock import FileIndexLock
from docgrep.adapters.syncstate import FileSyncState
from sync_support import BOOKS, Rig, write


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[Rig]:
    rig = Rig(tmp_path)
    yield rig
    rig.close()


def _search(rig: Rig, query: str) -> list[str]:
    assert rig.services is not None
    result = rig.services.search.run(query)
    return [hit.source.path for hit in result.results]


def test_added_changed_removed_and_moved_files_are_applied(rig: Rig) -> None:
    rig.open()
    assert rig.cycle().applied is None  # start-up: nothing changed
    write(rig.books, {"fresh.txt": "A lighthouse stands above the northern cliffs."})
    write(rig.books, {"book01.txt": "Book one was rewritten about glaciers."})
    (rig.books / "book02.txt").unlink()
    (rig.books / "book03.txt").rename(rig.books / "book03-renamed.txt")
    embedder: Any = rig.services.update.embedder  # type: ignore[union-attr]  # container internals
    embedded = []
    inner = embedder.embed
    embedder.embed = lambda texts: embedded.append(len(texts)) or inner(texts)

    assert rig.cycle().applied is None
    result = rig.cycle()

    assert result.applied is not None
    applied = result.applied
    assert (applied.added, applied.modified, applied.removed, applied.moved) == (
        1,
        1,
        1,
        1,
    )
    sources = rig.sources()
    assert "fresh.txt" in sources and "book02.txt" not in sources
    assert "book03.txt" not in sources and "book03-renamed.txt" in sources
    # Two files were embedded (the new and the changed one); the move reused.
    assert len(embedded) == 2
    assert "fresh.txt" in _search(rig, "lighthouse northern cliffs")
    manifest: Any = rig.services.update.manifest  # type: ignore[union-attr]  # container internals
    run = manifest.latest_run_of_kind("sync")
    assert run is not None and run.status == "completed"
    assert manifest.latest_run_of_kind("update") is None


def test_start_up_reconciliation_catches_a_change_made_while_stopped(
    rig: Rig,
) -> None:
    write(
        rig.books,
        {"book01.txt": "Rewritten while docgrep was stopped."},
        mtime=rig.clock.wall,
    )
    (rig.books / "book05.txt").unlink()
    rig.open()

    first = rig.cycle()

    # The change settles at once; the removal waits for the second scan.
    assert first.applied is not None and first.applied.modified == 1
    assert first.applied.removed == 0
    second = rig.cycle()
    assert second.applied is not None and second.applied.removed == 1


def test_a_lock_held_by_another_operation_keeps_the_entries(rig: Rig) -> None:
    rig.open()
    rig.cycle()
    write(rig.books, {"fresh.txt": "Waiting for the index lock."})
    rig.cycle()

    with FileIndexLock(rig.data / "index.lock").hold("update"):
        blocked = rig.cycle()
    assert blocked.skipped == "index_lock_held" and blocked.applied is None
    assert "fresh.txt" not in rig.sources()
    assert rig.sync.status().pending_added == 1

    assert rig.cycle().applied is not None
    assert "fresh.txt" in rig.sources()


def test_an_unmounted_root_keeps_its_sources_and_is_reconciled_when_back(
    rig: Rig, tmp_path: Path
) -> None:
    rig.open()
    rig.cycle()
    away = tmp_path / "away"
    rig.notes.rename(away)

    for _ in range(3):
        assert rig.cycle().applied is None

    assert rig.sync.status().paused_roots == ("notes",)
    assert {"notes/x.txt", "notes/y.txt"} <= set(rig.sources())

    away.rename(rig.notes)
    write(rig.notes, {"y.txt": "Edited while the disk was away."}, mtime=rig.clock.wall)
    result = rig.cycle()

    assert rig.sync.status().paused_roots == ()
    assert result.applied is not None and result.applied.modified == 1


def test_an_unreadable_folder_of_a_root_is_paused_not_emptied(
    rig: Rig, tmp_path: Path
) -> None:
    rig.open()
    rig.cycle()
    # The root's folder is replaced by a plain file: not a directory.
    rig.notes.rename(tmp_path / "away")
    rig.notes.write_text("not a folder")

    for _ in range(3):
        assert rig.cycle().applied is None

    assert rig.sync.status().paused_roots == ("notes",)
    assert "notes/x.txt" in rig.sources()


def test_a_new_root_waits_for_its_first_manual_update(rig: Rig, tmp_path: Path) -> None:
    from roots_support import save_roots

    extra = tmp_path / "extra"
    write(extra, {"e1.txt": "Extra root: a map of the old canal.", "e2.txt": "More."})
    save_roots(
        rig.data,
        ("books", rig.books, True),
        ("notes", rig.notes, False),
        ("extra", extra, False),
    )
    rig.open()

    for _ in range(3):
        assert rig.cycle().applied is None
    assert rig.sync.status().awaiting_first_update == ("extra",)
    assert not any(path.startswith("extra/") for path in rig.sources())

    summary = rig.services.update.run()  # type: ignore[union-attr]  # the first manual update
    assert summary.added == 2

    write(extra, {"e3.txt": "A third file after the first update."})
    rig.cycle()
    result = rig.cycle()
    assert rig.sync.status().awaiting_first_update == ()
    assert result.applied is not None and result.applied.added == 1


def test_emptying_a_root_holds_its_removals_but_applies_additions(rig: Rig) -> None:
    rig.open()
    rig.cycle()
    for name in BOOKS:
        (rig.books / name).unlink()
    write(rig.books, {"fresh.txt": "The only file left in the root."})
    rig.cycle()

    result = rig.cycle()

    assert result.applied is not None
    assert (result.applied.added, result.applied.removed) == (1, 0)
    assert len(result.applied.held_removals) == len(BOOKS)
    status = rig.sync.status()
    assert [(h.root, h.count) for h in status.held] == [("books", len(BOOKS))]
    assert set(BOOKS) <= set(rig.sources())

    # A manual update with --yes applies them.
    summary = rig.services.update.run(override_guard=True)  # type: ignore[union-attr]  # container
    assert summary.removed == len(BOOKS)
    assert rig.cycle().applied is None
    assert rig.sync.status().held == ()


def test_held_removals_are_dropped_when_the_files_reappear(rig: Rig) -> None:
    rig.open()
    rig.cycle()
    for name in BOOKS:
        (rig.books / name).unlink()
    rig.cycle()
    rig.cycle()
    assert rig.sync.status().held

    write(rig.books, BOOKS)
    rig.cycle()

    assert rig.sync.status().held == ()
    assert set(BOOKS) <= set(rig.sources())


def test_moving_a_whole_root_into_another_folder_passes_the_guard(rig: Rig) -> None:
    rig.open()
    rig.cycle()
    target = rig.notes / "from-books"
    target.mkdir()
    for name in BOOKS:
        (rig.books / name).rename(target / name)
    rig.cycle()

    result = rig.cycle()

    assert result.applied is not None
    assert (result.applied.moved, result.applied.held_removals) == (len(BOOKS), ())
    assert not any(path in rig.sources() for path in BOOKS)
    assert rig.sync.status().held == ()


def test_the_status_is_written_for_other_processes(rig: Rig) -> None:
    rig.open()
    rig.cycle()

    status = FileSyncState(rig.data).read()

    assert status is not None and status.enabled and status.running
    assert status.last_scan_at is not None
    assert status.interval_seconds == 120


def test_ocr_outputs_are_picked_up_as_added_files(rig: Rig) -> None:
    rig.open()
    rig.cycle()
    # What `docgrep ocr` writes next to an original is an ordinary new file.
    write(rig.books, {"scan.ocr.txt": "Text recognised from a scanned page."})
    rig.cycle()

    result = rig.cycle()

    assert result.applied is not None and result.applied.added == 1
    assert "scan.ocr.txt" in rig.sources()


def test_files_the_settle_time_has_not_aged_wait(rig: Rig) -> None:
    rig.open()
    rig.cycle()
    path = rig.books / "recent.txt"
    path.write_text("Written a moment ago.", encoding="utf-8")
    os.utime(path, (rig.clock.wall, rig.clock.wall))

    assert rig.cycle(advance=10).applied is None
    assert rig.cycle(advance=10).applied is None
    assert rig.cycle(advance=120).applied is not None


def test_a_failed_file_is_not_retried_until_it_changes(rig: Rig) -> None:
    from docgrep.domain.models import FileType

    rig.open()
    rig.cycle()
    update: Any = rig.services.update  # type: ignore[union-attr]  # container
    real = update.parsers[FileType.TXT]

    class Failing:
        version = real.version
        attempts = 0

        def parse(self, source: Any) -> Any:
            if source.path == "broken.txt":
                Failing.attempts += 1
                raise ValueError("injected parse failure")
            return real.parse(source)

    update.parsers = {**update.parsers, FileType.TXT: Failing()}
    write(rig.books, {"broken.txt": "This file cannot be parsed."})
    rig.cycle()
    assert rig.cycle().applied is not None and Failing.attempts == 1

    for _ in range(4):
        assert rig.cycle().applied is None
    assert Failing.attempts == 1

    write(rig.books, {"broken.txt": "This file still cannot be parsed, but longer."})
    rig.cycle()
    assert rig.cycle().applied is not None and Failing.attempts == 2
    assert rig.sources()["broken.txt"].startswith("failed:")
