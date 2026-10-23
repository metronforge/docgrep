"""An update limited to a set of stored paths, the unit of automatic sync
(ADR-0029), through domain ports and in-memory fakes."""

from test_plan_update import Harness, _files

from docgrep.domain.update_plan import UpdateScope


def _scope(*paths: str) -> UpdateScope:
    return UpdateScope(frozenset(paths))


def test_only_the_paths_of_the_scope_change() -> None:
    harness = Harness(_files(6))
    harness.build()
    files = harness.corpus.files
    files["doc00.txt"] = ("changed words of doc zero", 2.0)
    files["doc01.txt"] = ("changed words of doc one", 2.0)
    files["new_a.txt"] = ("a new document in scope", 2.0)
    files["new_b.txt"] = ("a new document out of scope", 2.0)
    del files["doc02.txt"]
    del files["doc03.txt"]

    summary = harness.update().run(
        scope=_scope("doc00.txt", "new_a.txt", "doc02.txt"), run_kind="sync"
    )

    assert (summary.added, summary.modified, summary.removed) == (1, 1, 1)
    # Everything else counts as unchanged, whatever its state on disk.
    assert summary.plan.unchanged == 4
    assert harness.parser.parsed == ["doc00.txt", "new_a.txt"]
    paths = {r.source.path for r in harness.manifest.committed_sources()}
    assert paths == {
        "doc00.txt",
        "doc01.txt",
        "doc03.txt",
        "doc04.txt",
        "doc05.txt",
        "new_a.txt",
    }
    stale = harness.manifest.get_source("doc01.txt")
    assert stale is not None and stale.source.mtime == 1.0


def test_a_scoped_update_is_recorded_as_a_sync_run() -> None:
    harness = Harness(_files(3))
    harness.build()
    harness.corpus.files["doc00.txt"] = ("changed words of doc zero", 2.0)

    summary = harness.update().run(scope=_scope("doc00.txt"), run_kind="sync")

    run = harness.manifest.latest_run()
    assert run is not None and (run.kind, run.run_id) == ("sync", summary.run_id)
    assert run.status == "completed"


def test_an_empty_scope_changes_nothing() -> None:
    harness = Harness(_files(3))
    harness.build()
    harness.corpus.files["extra.txt"] = ("not in scope", 2.0)
    del harness.corpus.files["doc00.txt"]

    summary = harness.update().run(scope=_scope(), run_kind="sync")

    assert summary.plan.changes == 0 and summary.plan.unchanged == 3
    assert harness.parser.parsed == []


def test_a_move_in_scope_reuses_the_passages_without_a_model_call() -> None:
    harness = Harness(_files(4))
    harness.build()
    harness.corpus.files["moved/doc00.txt"] = harness.corpus.files.pop("doc00.txt")

    summary = harness.update(reuse_by_hash=True).run(
        scope=_scope("doc00.txt", "moved/doc00.txt"), run_kind="sync"
    )

    assert summary.moved == 1 and summary.failed == 0
    assert harness.parser.parsed == [] and harness.embedder.embedded == 0
    assert harness.manifest.get_source("doc00.txt") is None
    assert harness.manifest.get_source("moved/doc00.txt") is not None


def test_scoped_removals_that_trip_the_guard_are_held_and_the_rest_applied() -> None:
    harness = Harness(_files(40))
    harness.build()
    files = harness.corpus.files
    gone = [f"doc{i:02d}.txt" for i in range(30)]
    for name in gone:
        del files[name]
    files["fresh.txt"] = ("a fresh document", 2.0)

    summary = harness.update().run(scope=_scope(*gone, "fresh.txt"), run_kind="sync")

    assert summary.removed == 0 and summary.added == 1
    assert [(g.root, g.removals) for g in summary.held] == [("_legacy", 30)]
    assert set(summary.held_removals) == set(gone)
    assert harness.manifest.get_source("fresh.txt") is not None
    assert all(harness.manifest.get_source(name) is not None for name in gone)


def test_moves_in_scope_pass_the_guard_beside_held_deletions() -> None:
    harness = Harness(_files(40))
    harness.build()
    files = harness.corpus.files
    moved = [f"doc{i:02d}.txt" for i in range(30)]
    for name in moved:
        files[f"to/{name}"] = files.pop(name)

    summary = harness.update(reuse_by_hash=True).run(
        scope=_scope(*moved, *(f"to/{n}" for n in moved)), run_kind="sync"
    )

    assert summary.held == () and summary.moved == 30
