"""Automatic sync: scan, settle and apply with an update scoped to the settled
paths (ADR-0029), through domain ports and in-memory fakes. The clock is
injected and a cycle is run explicitly; nothing sleeps."""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from test_plan_update import Harness, _files

from docgrep.domain.errors import ConfigDrift, IndexLockHeld, ValidationError
from docgrep.domain.models import ScannedFile, Source
from docgrep.domain.roots import CorpusRoot, CorpusRoots
from docgrep.domain.sync import IGNORED_DIRS, IGNORED_NAMES, is_ignored
from docgrep.domain.update_plan import RemovalGuard
from docgrep.usecases.plan_update import PlanUpdate
from docgrep.usecases.sync_index import CycleResult, SyncIndex, UpdateApplier

ROOT = CorpusRoots.single(Path("/corpus"), "docs")
WALL = 1_000_000.0


class Clock:
    def __init__(self) -> None:
        self.mono = 0.0
        self.wall = WALL

    def advance(self, seconds: float) -> None:
        self.mono += seconds
        self.wall += seconds


class Catalog:
    def __init__(self, roots: CorpusRoots = ROOT) -> None:
        self._roots = roots

    def roots(self) -> CorpusRoots:
        return self._roots


class FakeSyncCorpus:
    """The files of a harness corpus, root by root."""

    def __init__(self, harness: Harness, roots: CorpusRoots = ROOT) -> None:
        self.harness = harness
        self.roots = roots
        self.unreadable: set[str] = set()
        self.scans = 0

    def root_readable(self, name: str) -> bool:
        return name not in self.unreadable

    def scan_root(self, name: str) -> tuple[ScannedFile, ...]:
        if name in self.unreadable:
            raise ValidationError("corpus is not accessible")
        self.scans += 1
        return tuple(
            item
            for item in self.harness.corpus.scan()
            if self.roots.root_of(item.path) == name
        )

    def describe(self, path: str) -> Source:
        return self.harness.corpus.describe(path)


class HeldLock:
    """An index lock that another operation may hold."""

    def __init__(self) -> None:
        self.held = False

    @contextmanager
    def hold(self, operation: str = "build") -> Iterator[None]:
        if self.held:
            raise IndexLockHeld("Another index operation is running")
        yield

    def is_held(self) -> bool:
        return self.held


class Rig:
    def __init__(
        self,
        files: dict[str, tuple[str, float]] | None = None,
        *,
        reuse: bool = True,
        guard: RemovalGuard | None = None,
        settle: float = 60.0,
        roots: CorpusRoots = ROOT,
    ) -> None:
        self.harness = Harness(files if files is not None else _files(6))
        self.harness.build()
        self.lock = HeldLock()
        self.harness.lock = self.lock  # type: ignore[assignment]  # test double
        self.clock = Clock()
        self.corpus = FakeSyncCorpus(self.harness, roots)
        self.catalog = Catalog(roots)
        self.update = self.harness.update(
            roots=self.catalog,
            reuse_by_hash=reuse,
            guard=guard or RemovalGuard(),
        )
        self.slept: list[float] = []
        self.statuses: list[Any] = []
        self.sync = SyncIndex(
            corpus=self.corpus,
            roots=self.catalog,
            manifest=self.harness.manifest,
            updates=UpdateApplier(self.update),
            planner=lambda view: PlanUpdate(
                corpus=view,
                manifest=self.harness.manifest,
                parsers=self.harness.parsers,
                reuse_by_hash=reuse,
                embedding_model=self.harness.embedder.model_name,
                guard=guard or RemovalGuard(),
            ),
            interval_seconds=120,
            settle_seconds=settle,
            clock=lambda: self.clock.mono,
            wall_clock=lambda: self.clock.wall,
            now=lambda: datetime(2026, 10, 23, tzinfo=UTC),
            sleep=self.slept.append,
            on_change=self.statuses.append,
        )

    @property
    def files(self) -> dict[str, tuple[str, float]]:
        return self.harness.corpus.files

    def cycle(self, advance: float = 120) -> CycleResult:
        self.clock.advance(advance)
        return self.sync.run_cycle()

    def start(self) -> None:
        """The first cycle reconciles; run it on an unchanged corpus."""
        assert self.cycle().applied is None

    def recorded(self) -> set[str]:
        return {r.source.path for r in self.harness.manifest.committed_sources()}

    def runs(self, kind: str) -> int:
        return sum(1 for run in self.harness.manifest._runs if run.kind == kind)

    def aged(self, text: str) -> tuple[str, float]:
        """A file last written long before the settle time."""
        return text, self.clock.wall - 3600


def test_the_ignore_list_matches_names_and_folders() -> None:
    for name in ("a.part", "b.crdownload", "c.tmp", "~$d.docx", ".~lock.e.odt#"):
        assert is_ignored(name) and is_ignored(f"sub/{name}")
    assert is_ignored(".#f.txt") and is_ignored("g.swp")
    for folder in IGNORED_DIRS:
        assert is_ignored(f"books/{folder}/notes.md")
    assert not is_ignored("notes.md") and not is_ignored("obsidian/notes.md")
    assert len(IGNORED_NAMES) == 7


def test_a_changed_file_is_applied_when_seen_unchanged_in_two_scans() -> None:
    rig = Rig()
    rig.start()
    rig.files["new.txt"] = rig.aged("a brand new document")
    rig.files["doc01.txt"] = rig.aged("rewritten words of doc one")

    first = rig.cycle()
    assert first.applied is None and rig.recorded() == set(_files(6))
    status = rig.sync.status()
    assert (status.pending_added, status.pending_modified, status.unsettled) == (
        1,
        1,
        2,
    )

    second = rig.cycle()

    assert second.applied is not None
    assert (second.applied.added, second.applied.modified) == (1, 1)
    assert "new.txt" in rig.recorded()
    assert rig.sync.status().pending_added == 0
    assert rig.runs("sync") == 1 and rig.runs("update") == 0


def test_a_file_growing_across_scans_is_not_applied_until_stable() -> None:
    rig = Rig()
    rig.start()
    for step in range(1, 4):
        rig.files["big.txt"] = rig.aged("growing " * step)
        assert rig.cycle().applied is None
    assert "big.txt" not in rig.recorded()

    assert rig.cycle().applied is not None
    assert "big.txt" in rig.recorded()


def test_a_file_written_a_moment_ago_waits_for_its_time_to_age() -> None:
    rig = Rig(settle=60)
    rig.start()
    rig.files["fresh.txt"] = ("just written", rig.clock.wall - 30)

    # Seen twice, but modified less than 60 seconds before the second scan.
    assert rig.cycle(advance=10).applied is None
    assert rig.cycle(advance=10).applied is None
    assert rig.cycle(advance=90).applied is not None


def test_a_file_deleted_before_settling_is_dropped() -> None:
    rig = Rig()
    rig.start()
    rig.files["tmp.txt"] = rig.aged("here for a moment")
    rig.cycle()
    del rig.files["tmp.txt"]

    assert rig.cycle().applied is None
    status = rig.sync.status()
    assert (status.pending_added, status.unsettled) == (0, 0)
    assert rig.runs("sync") == 0


def test_ignored_names_are_never_applied_or_removed() -> None:
    rig = Rig({**_files(6), ".obsidian/n.txt": ("obsidian", 1.0)})
    # Recorded before the ignore list existed; sync leaves it alone.
    assert ".obsidian/n.txt" in rig.recorded()
    rig.start()
    for name in ("movie.part", "doc.tmp", "~$x.txt", "a.swp", ".git/c.txt"):
        rig.files[name] = rig.aged("partial " + name)
    rig.files["sub/.trash/old.txt"] = rig.aged("trash")
    del rig.files[".obsidian/n.txt"]

    for _ in range(4):
        assert rig.cycle().applied is None

    assert ".obsidian/n.txt" in rig.recorded()
    assert rig.sync.status().pending_added == 0 and rig.runs("sync") == 0


def test_a_removal_needs_two_scans_without_the_file() -> None:
    rig = Rig()
    rig.start()
    del rig.files["doc02.txt"]

    assert rig.cycle().applied is None
    assert rig.sync.status().pending_removed == 1
    assert "doc02.txt" in rig.recorded()

    result = rig.cycle()

    assert result.applied is not None and result.applied.removed == 1
    assert "doc02.txt" not in rig.recorded()


def test_a_file_that_comes_back_cancels_its_removal() -> None:
    rig = Rig()
    rig.start()
    saved = rig.files.pop("doc02.txt")
    rig.cycle()
    rig.files["doc02.txt"] = saved

    assert rig.cycle().applied is None
    assert rig.sync.status().pending_removed == 0
    assert "doc02.txt" in rig.recorded() and rig.runs("sync") == 0


def test_a_held_lock_ends_the_cycle_and_keeps_the_entries() -> None:
    rig = Rig()
    rig.start()
    rig.files["new.txt"] = rig.aged("waiting for the lock")
    del rig.files["doc03.txt"]
    rig.cycle()
    rig.lock.held = True

    result = rig.cycle()

    assert result.skipped == "index_lock_held" and result.applied is None
    assert rig.runs("sync") == 0 and "new.txt" not in rig.recorded()
    status = rig.sync.status()
    assert (status.pending_added, status.pending_removed) == (1, 1)

    rig.lock.held = False
    result = rig.cycle()

    assert result.applied is not None
    assert (result.applied.added, result.applied.removed) == (1, 1)


def test_a_failed_file_is_retried_only_when_it_changes_or_at_reconciliation() -> None:
    rig = Rig()
    rig.start()
    rig.harness.parser.failing.add("bad.txt")
    rig.files["bad.txt"] = rig.aged("this one fails to parse")
    rig.cycle()
    failed = rig.cycle()
    assert failed.applied is not None and failed.applied.failed == 1
    attempts = rig.harness.parser.parsed.count("bad.txt")

    for _ in range(5):
        assert rig.cycle().applied is None
    assert rig.harness.parser.parsed.count("bad.txt") == attempts

    # A change of the file starts it again.
    rig.files["bad.txt"] = rig.aged("this one still fails to parse, longer")
    rig.cycle()
    assert rig.cycle().applied is not None
    assert rig.harness.parser.parsed.count("bad.txt") == attempts + 1

    # The daily reconciliation retries it once more.
    rig.harness.parser.failing.clear()
    result = rig.cycle(advance=24 * 3600)
    assert result.applied is not None and result.applied.failed == 0
    record = rig.harness.manifest.get_source("bad.txt")
    assert record is not None and record.status == "indexed"


def test_start_up_reconciliation_catches_a_change_made_while_stopped() -> None:
    rig = Rig()
    # Written moments ago, so the settle time has not passed; and a deletion.
    rig.files["doc00.txt"] = ("edited while docgrep was stopped", rig.clock.wall)
    rig.files["added.txt"] = ("added while docgrep was stopped", rig.clock.wall)
    del rig.files["doc05.txt"]

    first = rig.cycle()

    # Additions and changes settle at once; the removal waits for a second scan.
    assert first.applied is not None
    assert (first.applied.added, first.applied.modified, first.applied.removed) == (
        1,
        1,
        0,
    )
    second = rig.cycle()
    assert second.applied is not None and second.applied.removed == 1


def test_a_moved_file_reuses_its_passages_without_a_model_call() -> None:
    rig = Rig()
    rig.start()
    rig.files["moved/doc00.txt"] = rig.files.pop("doc00.txt")
    before = rig.harness.embedder.embedded

    rig.cycle()
    result = rig.cycle()

    assert result.applied is not None and result.applied.moved == 1
    assert rig.harness.embedder.embedded == before
    assert rig.harness.parser.parsed == []
    assert "doc00.txt" not in rig.recorded() and "moved/doc00.txt" in rig.recorded()


def test_emptying_a_root_holds_its_removals_and_applies_additions() -> None:
    rig = Rig(_files(40))
    rig.start()
    for path in list(rig.files):
        del rig.files[path]
    rig.files["fresh.txt"] = rig.aged("a fresh document")
    rig.cycle()

    result = rig.cycle()

    assert result.applied is not None
    assert (result.applied.added, result.applied.removed) == (1, 0)
    assert len(result.applied.held_removals) == 40
    assert len(rig.recorded()) == 41
    status = rig.sync.status()
    assert [(h.root, h.count) for h in status.held] == [("docs", 40)]
    assert len(status.held[0].paths) == 5 and status.pending_removed == 0
    # Nothing more is applied while they stay held.
    runs = rig.runs("sync")
    for _ in range(3):
        assert rig.cycle().applied is None
    assert rig.runs("sync") == runs


def test_held_removals_are_applied_by_a_manual_update_with_yes() -> None:
    rig = Rig(_files(40))
    rig.start()
    rig.files.clear()
    rig.cycle()
    rig.cycle()
    assert len(rig.sync.status().held[0].paths) == 5

    summary = rig.update.run(override_guard=True)

    assert summary.removed == 40
    assert rig.cycle().applied is None
    status = rig.sync.status()
    assert status.held == () and status.pending_removed == 0


def test_files_that_reappear_drop_their_held_removals() -> None:
    rig = Rig(_files(40))
    rig.start()
    saved = dict(rig.files)
    rig.files.clear()
    rig.cycle()
    rig.cycle()
    assert rig.sync.status().held

    rig.files.update(saved)
    rig.cycle()

    status = rig.sync.status()
    assert status.held == () and status.pending_removed == 0
    assert len(rig.recorded()) == 40


def test_removals_beyond_the_threshold_in_later_cycles_stay_together() -> None:
    rig = Rig(_files(60))
    rig.start()
    for index in range(10):
        del rig.files[f"doc{index:02d}.txt"]
    rig.cycle()
    rig.cycle()
    assert len(rig.sync.status().held[0].paths) == 5
    held = rig.sync.status().held[0].count
    assert held == 10

    # One more deletion does not slip under the threshold on its own.
    del rig.files["doc50.txt"]
    rig.cycle()
    result = rig.cycle()

    assert result.applied is not None and result.applied.removed == 0
    assert rig.sync.status().held[0].count == 11


def test_moving_more_files_than_the_threshold_passes_the_guard() -> None:
    rig = Rig(_files(40))
    rig.start()
    for path in list(rig.files):
        rig.files[f"archive/{path}"] = rig.files.pop(path)
    rig.cycle()

    result = rig.cycle()

    assert result.applied is not None
    assert (result.applied.moved, result.applied.held_removals) == (40, ())
    assert rig.recorded() == {f"archive/{p}" for p in _files(40)}
    assert rig.sync.status().held == ()


def test_the_same_number_of_deletions_without_copies_is_held() -> None:
    rig = Rig(_files(40))
    rig.start()
    rig.files.clear()
    rig.cycle()

    result = rig.cycle()

    assert result.applied is not None and len(result.applied.held_removals) == 40
    assert len(rig.recorded()) == 40


def test_moves_start_up_together_with_the_removal_of_their_old_path() -> None:
    rig = Rig(_files(40))
    for path in list(rig.files):
        rig.files[f"archive/{path}"] = rig.files.pop(path)

    first = rig.cycle()
    # The targets wait for the second scan, which confirms the removals.
    assert first.applied is None

    second = rig.cycle()

    assert second.applied is not None
    assert (second.applied.moved, second.applied.held_removals) == (40, ())


def test_sync_stays_idle_and_says_why_when_the_configuration_drifted() -> None:
    rig = Rig()
    rig.start()

    def drifted() -> None:
        raise ConfigDrift("The configuration differs from the index")

    rig.update.check_ready = drifted  # type: ignore[method-assign]  # test double
    rig.files["new.txt"] = rig.aged("never indexed")
    scans = rig.corpus.scans

    result = rig.cycle()

    assert result.skipped == "not_ready" and rig.corpus.scans == scans
    status = rig.sync.status()
    assert status.idle_reason is not None
    assert status.idle_reason.startswith("config_drift: ")
    assert rig.runs("sync") == 0


def test_a_run_gives_way_between_sources_and_reports_progress() -> None:
    rig = Rig()
    rig.start()
    for index in range(3):
        rig.files[f"new{index}.txt"] = rig.aged(f"new document number {index}")
    rig.cycle()

    result = rig.cycle()

    assert result.applied is not None and result.applied.added == 3
    assert len(rig.slept) >= 3 and set(rig.slept) == {0.05}
    assert rig.sync.status().applying is None


def test_stopping_starts_no_new_cycle_and_cancels_a_run() -> None:
    rig = Rig()
    rig.start()
    rig.files["new.txt"] = rig.aged("after the stop")
    rig.cycle()

    rig.sync.request_stop()

    assert rig.sync.stopping
    result = rig.cycle()
    assert result.skipped == "stopping" and "new.txt" not in rig.recorded()


def test_the_status_lists_its_sections_without_paths_unless_asked() -> None:
    rig = Rig(_files(40))
    rig.start()
    rig.files.clear()
    rig.cycle()
    rig.cycle()

    status = rig.sync.status()
    body = status.to_dict()
    with_paths = status.to_dict(paths=True)

    assert set(body) == {
        "enabled",
        "running",
        "interval_seconds",
        "settle_seconds",
        "last_scan_at",
        "last_applied",
        "pending",
        "unsettled",
        "unsettled_over_limit",
        "held_removals",
        "paused_roots",
        "awaiting_first_update",
        "idle_reason",
        "last_error",
        "applying",
    }
    assert body["held_removals"] == [{"root": "docs", "count": 40}]
    assert len(with_paths["held_removals"][0]["paths"]) == 5
    assert type(status).from_dict(with_paths) == status


def test_invalid_settings_are_refused() -> None:
    rig = Rig()
    for name, value in (
        ("interval_seconds", 0),
        ("settle_seconds", -1),
        ("reconcile_hours", 0),
    ):
        with pytest.raises(ValueError, match=name):
            SyncIndex(
                corpus=rig.corpus,
                roots=rig.catalog,
                manifest=rig.harness.manifest,
                updates=UpdateApplier(rig.update),
                planner=lambda view: None,  # type: ignore[arg-type,return-value]  # never called
                **{name: value},
            )


def test_paused_roots_keep_their_sources_and_are_reconciled_when_back() -> None:
    books, notes = Path("/books"), Path("/notes")
    roots = CorpusRoots(
        (CorpusRoot("books", books, legacy=True), CorpusRoot("notes", notes))
    )
    files = {
        **_files(5),
        **{f"notes/n{i}.txt": (f"note number {i} here", 1.0) for i in range(5)},
    }
    rig = Rig(files, roots=roots)
    rig.start()
    saved = {p: v for p, v in rig.files.items() if p.startswith("notes/")}
    for path in saved:
        del rig.files[path]
    rig.corpus.unreadable.add("notes")

    for _ in range(3):
        assert rig.cycle().applied is None

    assert rig.sync.status().paused_roots == ("notes",)
    assert {p for p in rig.recorded() if p.startswith("notes/")} == set(saved)

    # The folder comes back with a change made while it was away.
    rig.corpus.unreadable.clear()
    rig.files.update(saved)
    rig.files["notes/n0.txt"] = ("note zero was edited while away", rig.clock.wall)

    result = rig.cycle()

    assert rig.sync.status().paused_roots == ()
    assert result.applied is not None and result.applied.modified == 1
    assert "notes/n0.txt" in rig.recorded()


def test_a_cancelled_run_keeps_what_it_did_not_reach() -> None:
    rig = Rig()
    rig.start()
    for index in range(3):
        rig.files[f"new{index}.txt"] = rig.aged(f"new document number {index}")
    rig.cycle()
    parse = rig.harness.parser.parse

    def parse_and_cancel(source: Source) -> Any:
        rig.update.request_cancel()
        return parse(source)

    rig.harness.parser.parse = parse_and_cancel  # type: ignore[method-assign]  # test double

    cancelled = rig.cycle()

    assert cancelled.applied is not None and cancelled.applied.cancelled
    done = {p for p in rig.recorded() if p.startswith("new")}
    assert len(done) == 1
    assert rig.sync.status().last_applied is not None
    assert rig.sync.status().last_applied.cancelled  # type: ignore[union-attr]  # checked just above

    rig.harness.parser.parse = parse  # type: ignore[method-assign]  # test double
    again = rig.cycle()

    # The two files it did not reach are applied; the first is not repeated.
    assert again.applied is not None and again.applied.added == 2
    assert {p for p in rig.recorded() if p.startswith("new")} == {
        "new0.txt",
        "new1.txt",
        "new2.txt",
    }


def test_an_unexpected_failure_is_reported_by_category_and_the_next_cycle_runs() -> (
    None
):
    rig = Rig()
    rig.start()
    real = rig.harness.manifest.committed_sources

    def broken() -> Any:
        raise RuntimeError("cannot read /srv/secret/path")

    rig.harness.manifest.committed_sources = broken  # type: ignore[method-assign]  # test double

    result = rig.cycle()

    assert result.skipped == "cycle_failed"
    assert rig.sync.status().last_error == "unexpected"

    rig.harness.manifest.committed_sources = real  # type: ignore[method-assign]  # test double
    assert rig.cycle().skipped is None
