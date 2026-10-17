"""Indexing, updating and reporting across several corpus roots (ADR-0023)."""

import shutil
from pathlib import Path
from typing import Any

import pytest

from docgrep.adapters.manifest import SQLiteManifest
from docgrep.api.container import Services, create_container, create_status
from docgrep.api.roots import parameter_roots
from docgrep.domain.errors import ConfigDrift, ValidationError
from docgrep.domain.models import make_passage_id
from docgrep.domain.roots import CorpusRoot, CorpusRoots
from docgrep.domain.sources import SourceFilter
from docgrep.usecases.update_index import UpdateIndex
from roots_support import save_roots, settings, two_roots

# make_passage_id("a.txt", 0), computed before corpus roots existed.
LEGACY_A_PASSAGE_ID = "4ce5b267e4de9962352ac53f6bb5d896032fb281269d3631eb11031629d601f1"


def _committed(data_dir: Path) -> dict[str, str]:
    manifest = SQLiteManifest(data_dir, read_only=True)
    return {r.source.path: r.status for r in manifest.committed_sources()}


def _passage_ids(services: Services, source_id: str) -> set[str]:
    store: Any = getattr(services.build, "passage_writer")
    return {entry.passage_id for entry in store.source_outline(source_id)}


def _container(
    data_dir: Path,
    previous: Services | None = None,
    roots: CorpusRoots | None = None,
    **overrides: Any,
) -> Services:
    """A composition root; the local store of the previous one is closed
    first, as only one client may open it."""
    if previous is not None:
        getattr(previous.build, "passage_writer").close()
    chosen = settings(data_dir, **overrides)
    return create_container(
        chosen if roots is None else chosen.with_corpus_roots(roots)
    )


def _update(services: Services) -> UpdateIndex:
    assert isinstance(services.update, UpdateIndex)
    return services.update


def test_the_legacy_root_keeps_its_identity(tmp_path: Path) -> None:
    books, _ = two_roots(tmp_path)
    data = tmp_path / "data"
    # The deprecated variable alone: one legacy root, exactly as before.
    services = _container(data, corpus_root=books)

    summary = services.build.run()

    assert summary.indexed == 2
    assert set(_committed(data)) == {"a.txt", "shelf/b.md"}
    assert LEGACY_A_PASSAGE_ID in _passage_ids(services, "a.txt")
    assert make_passage_id("a.txt", 0) == LEGACY_A_PASSAGE_ID


def test_two_roots_with_the_same_relative_path_are_distinct(tmp_path: Path) -> None:
    books, notes = two_roots(tmp_path)
    data = tmp_path / "data"
    save_roots(data, ("books", books, True), ("notes", notes, False))
    services = _container(data)

    summary = services.build.run()

    assert summary.indexed == 4
    assert set(_committed(data)) == {
        "a.txt",
        "shelf/b.md",
        "notes/a.txt",
        "notes/c.txt",
    }
    legacy = _passage_ids(services, "a.txt")
    prefixed = _passage_ids(services, "notes/a.txt")
    assert LEGACY_A_PASSAGE_ID in legacy
    assert prefixed and not legacy & prefixed

    found = services.search.run("yellow kite meadow", top_n=5)
    assert found.source_roots == {
        item.source.source_id: (
            "notes" if item.source.source_id.startswith("notes/") else "books"
        )
        for item in found.results
    }
    assert found.source_roots["notes/a.txt"] == "notes"
    only_notes = services.search.run(
        "comet kite", top_n=5, source_filter=SourceFilter(roots=("notes",))
    )
    assert only_notes.applied_filters == {"roots": ["notes"]}
    assert {item.source.source_id for item in only_notes.results} <= {
        "notes/a.txt",
        "notes/c.txt",
    }
    with pytest.raises(ValidationError) as raised:
        services.search.run("kite", source_filter=SourceFilter(roots=("music",)))
    assert str(raised.value) == "unknown root 'music'; known roots: books, notes"


def test_update_changes_only_what_changed_in_each_root(tmp_path: Path) -> None:
    books, notes = two_roots(tmp_path)
    data = tmp_path / "data"
    save_roots(data, ("books", books, True), ("notes", notes, False))
    services = _container(data)
    services.build.run()

    (notes / "a.txt").write_text("Field notes: a red kite over the hills.", "utf-8")
    (books / "shelf" / "b.md").unlink()
    summary = _update(services).run()

    assert (summary.added, summary.modified, summary.removed) == (0, 1, 1)
    assert set(_committed(data)) == {"a.txt", "notes/a.txt", "notes/c.txt"}
    assert _passage_ids(services, "shelf/b.md") == set()
    assert LEGACY_A_PASSAGE_ID in _passage_ids(services, "a.txt")


def test_a_missing_root_fails_the_update_before_any_work(tmp_path: Path) -> None:
    books, notes = two_roots(tmp_path)
    data = tmp_path / "data"
    save_roots(data, ("books", books, True), ("notes", notes, False))
    services = _container(data)
    services.build.run()
    before = _committed(data)
    runs = SQLiteManifest(data, read_only=True).latest_run()
    shutil.rmtree(notes)

    with pytest.raises(ValidationError) as raised:
        _update(services).run()

    assert str(raised.value) == f"corpus root 'notes' is not accessible: {notes}"
    assert raised.value.details["root"] == "notes"
    assert _committed(data) == before
    assert SQLiteManifest(data, read_only=True).latest_run() == runs
    report = create_status(settings(data)).run().as_dict()
    notes_view = next(root for root in report["roots"] if root["name"] == "notes")
    assert notes_view == {
        "name": "notes",
        "legacy": False,
        "configured": True,
        "readable": False,
        "indexed": 2,
        "skipped": 0,
        "failed": 0,
    }
    assert "root_unreadable" in {w["code"] for w in report["warnings"]}


def test_a_root_that_is_not_configured_keeps_its_sources(tmp_path: Path) -> None:
    books, notes = two_roots(tmp_path)
    data = tmp_path / "data"
    save_roots(data, ("books", books, True), ("notes", notes, False))
    services = _container(data)
    services.build.run()
    # The root is removed from the list and its folder is gone as well.
    save_roots(data, ("books", books, True))
    shutil.rmtree(notes)

    services = _container(data, services)
    summary = _update(services).run()

    assert summary.plan.outside_run == (("notes", 2),)
    assert summary.removed == 0
    assert set(_committed(data)) == {
        "a.txt",
        "shelf/b.md",
        "notes/a.txt",
        "notes/c.txt",
    }
    report = create_status(settings(data)).run().as_dict()
    assert report["roots"] == [
        {
            "name": "books",
            "legacy": True,
            "configured": True,
            "readable": True,
            "indexed": 2,
            "skipped": 0,
            "failed": 0,
        },
        {
            "name": "notes",
            "legacy": False,
            "configured": False,
            "readable": None,
            "indexed": 2,
            "skipped": 0,
            "failed": 0,
        },
    ]
    assert "root_not_configured" in {w["code"] for w in report["warnings"]}
    # A full build would drop them, so it refuses.
    with pytest.raises(
        ValidationError, match=r"roots that this build does not visit: notes \(2\)"
    ):
        services.build.run()


def test_forget_root_removes_only_that_root(tmp_path: Path) -> None:
    books, notes = two_roots(tmp_path)
    data = tmp_path / "data"
    save_roots(data, ("books", books, True), ("notes", notes, False))
    services = _container(data)
    services.build.run()
    save_roots(data, ("books", books, True))
    services = _container(data, services)
    update = _update(services)

    planned = update.forget_plan("notes")
    assert update.passages_of(planned) == len(
        _passage_ids(services, "notes/a.txt")
    ) + len(_passage_ids(services, "notes/c.txt"))
    assert [r.source.path for r in planned] == ["notes/a.txt", "notes/c.txt"]
    with pytest.raises(ValidationError, match="is the legacy root"):
        update.forget_plan("books")
    with pytest.raises(ValidationError, match="unknown root 'music'"):
        update.forget_plan("music")
    summary = update.forget_root("notes")

    assert summary.removed == 2 and summary.passages_removed > 0
    assert set(_committed(data)) == {"a.txt", "shelf/b.md"}
    assert _passage_ids(services, "notes/a.txt") == set()
    assert LEGACY_A_PASSAGE_ID in _passage_ids(services, "a.txt")
    report = create_status(settings(data)).run().as_dict()
    assert [root["name"] for root in report["roots"]] == ["books"]


def test_without_a_legacy_root_unprefixed_sources_stay(tmp_path: Path) -> None:
    books, notes = two_roots(tmp_path)
    data = tmp_path / "data"
    services = _container(data, corpus_root=books)
    services.build.run()
    # A roots file without the legacy root: the old sources are reported.
    save_roots(data, ("notes", notes, False))

    services = _container(data, services)
    update = _update(services)
    summary = update.run()

    assert summary.added == 2 and summary.removed == 0
    assert summary.plan.outside_run == (("_legacy", 2),)
    assert {"a.txt", "shelf/b.md"} <= set(_committed(data))
    report = create_status(settings(data)).run().as_dict()
    legacy = next(root for root in report["roots"] if root["name"] == "_legacy")
    assert legacy["configured"] is False and legacy["legacy"] is True
    with pytest.raises(ValidationError, match="is the legacy root"):
        update.forget_plan("_legacy")


def test_parameter_roots_index_only_themselves(tmp_path: Path) -> None:
    books, notes = two_roots(tmp_path)
    data = tmp_path / "data"
    save_roots(data, ("books", books, True))
    services = _container(data)
    services.build.run()
    run_roots = parameter_roots(settings(data), [f"notes={notes}"]).roots
    assert run_roots.names == ("notes",)

    services = _container(data, services, roots=run_roots)
    summary = _update(services).run()

    # The saved legacy root is not part of the run: its sources stay.
    assert summary.added == 2 and summary.removed == 0
    assert summary.plan.outside_run == (("books", 2),)
    assert set(_committed(data)) == {
        "a.txt",
        "shelf/b.md",
        "notes/a.txt",
        "notes/c.txt",
    }
    # The roots file is unchanged.
    assert (data / "roots.json").read_text("utf-8").count('"name"') == 1


def test_a_resume_refuses_other_roots(tmp_path: Path) -> None:
    books, notes = two_roots(tmp_path)
    data = tmp_path / "data"
    save_roots(data, ("books", books, True))
    services = _container(data)
    services.build.request_cancel()
    assert services.build.run().cancelled
    save_roots(data, ("books", books, True), ("notes", notes, False))

    services = _container(data, services)
    with pytest.raises(ConfigDrift) as raised:
        services.build.resume()

    assert "The corpus roots differ from the roots the run started with" in str(
        raised.value
    )
    same = CorpusRoots((CorpusRoot("books", books.resolve(), legacy=True),))
    resumed = _container(data, services, roots=same).build.resume()
    assert resumed.indexed == 2
