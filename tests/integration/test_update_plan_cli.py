"""`docgrep update --plan`, the removal guard and `update --yes`, over the
real local-mode store and manifest with the deterministic backend (ADR-0027)."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from docgrep.adapters.manifest import SQLiteManifest
from docgrep.api.cli import main
from docgrep.api.container import Services, create_container, create_update_plan
from docgrep.domain.models import FileType, ParsedDocument, Source
from docgrep.settings import Settings
from docgrep.usecases.plan_update import PlanUpdate
from roots_support import environment, save_roots
from update_support import with_version

OLD = 1_700_000_000
NEW = 1_700_100_000

BOOKS = {
    "a.txt": "A rare blue comet crosses the winter sky above the harbour.",
    "c.txt": "Recipe for bread with rye flour and caraway seeds.",
    "d.txt": "Field notes on the migration of cranes in autumn.",
    "e.txt": "Letters from a lighthouse keeper to his sister.",
    "f.txt": "A ledger of the village mill, written in pencil.",
    "h.txt": "Instructions for tuning an old upright piano.",
    "shelf/b.md": "# Moss\n\nGarden notes about green moss on old stones.",
    "shelf/g.md": "# Tides\n\nTables of the spring tides for the bay.",
}
NOTES = {
    "x.txt": "A yellow kite drifts over the quiet meadow.",
    "y.txt": "Notes on a walk along the river after rain.",
}


def _write(root: Path, files: dict[str, str], mtime: int = OLD) -> None:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        os.utime(path, (mtime, mtime))


def _cli(capsys: pytest.CaptureFixture[str], *args: str, **factories: Any) -> Any:
    code = main(list(args), **factories)
    out, err = capsys.readouterr()
    return code, out, err


def _snapshot(data: Path) -> dict[str, str]:
    """Every file under the data directory with a hash of its content."""
    return {
        str(path.relative_to(data)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(data.rglob("*"))
        if path.is_file()
    }


class _FailsFor:
    def __init__(self, inner: Any, path: str) -> None:
        self.inner = inner
        self.path = path
        self.version = inner.version

    def parse(self, source: Source) -> ParsedDocument:
        if source.path == self.path:
            raise ValueError("injected parse failure")
        document: ParsedDocument = self.inner.parse(source)
        return document


def _failing_build(settings: Settings) -> Services:
    services = create_container(settings)
    build: Any = services.build
    build.parsers = {
        **build.parsers,
        FileType.TXT: _FailsFor(build.parsers[FileType.TXT], "h.txt"),
    }
    return services


def _new_markdown(settings: Settings) -> Services:
    services = create_container(settings)
    with_version(services.update, FileType.MARKDOWN, "99")
    update: Any = services.update
    update.sleep = lambda seconds: None
    return services


def _new_markdown_plan(settings: Settings) -> PlanUpdate:
    plan = create_update_plan(settings)
    with_version(plan, FileType.MARKDOWN, "99")
    return plan


@pytest.fixture
def library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    books, notes, data = tmp_path / "books", tmp_path / "notes", tmp_path / "data"
    _write(books, BOOKS)
    _write(notes, NOTES)
    for name, value in {**environment(data), "DOCGREP_LOG_LEVEL": "WARNING"}.items():
        monkeypatch.setenv(name, value)
    save_roots(data, ("books", books, True), ("notes", notes, False))
    return books.resolve(), notes.resolve(), data


@pytest.fixture
def indexed(
    library: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> tuple[Path, Path, Path]:
    code, _, err = _cli(capsys, "index", container_factory=_failing_build)
    assert code == 1, err  # h.txt failed on purpose
    return library


def _change_every_kind(books: Path, data: Path, notes: Path) -> None:
    _write(books, {"c.txt": "Recipe for bread with wheat flour and honey."}, NEW)
    os.utime(books / "e.txt", (NEW, NEW))  # touched, same content
    (books / "moved").mkdir()
    (books / "f.txt").rename(books / "moved" / "f.txt")
    (books / "d.txt").unlink()
    _write(books, {"new.txt": "A brand new pamphlet about bees."}, NEW)
    # notes stays indexed but is not part of the run.
    save_roots(data, ("books", books, True))


def test_the_plan_writes_nothing_and_predicts_the_update(
    indexed: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    books, notes, data = indexed
    _change_every_kind(books, data, notes)
    before = _snapshot(data)

    code, out, err = _cli(
        capsys, "update", "--plan", "--json", "--paths", plan_factory=_new_markdown_plan
    )

    assert code == 0, err
    assert _snapshot(data) == before
    body = json.loads(out)
    assert set(body) == {"note", "roots", "total", "outside_run", "guarded_roots"}
    (root,) = body["roots"]
    assert root["root"] == "books" and root["recorded"] == 8
    total = body["total"]
    assert {k: v for k, v in total.items() if k != "modified_by_reason"} == {
        "added": 2,
        "modified": 3,
        "removed": 2,
        "retried": 1,
        "refreshed": 1,
        "unchanged": 2,
        "moved": 1,
    }
    assert total["modified_by_reason"]["content_changed"] == 1
    assert total["modified_by_reason"]["parser_changed"] == 2
    assert body["outside_run"] == [{"root": "notes", "sources": 2}]
    assert body["guarded_roots"] == []
    assert root["paths"]["moved"] == [{"removed": "f.txt", "added": "moved/f.txt"}]
    assert root["paths"]["refreshed"] == ["e.txt"]
    assert root["paths"]["retried"] == ["h.txt"]

    code, out, err = _cli(capsys, "update", container_factory=_new_markdown)

    assert code == 0, err
    assert out.splitlines()[0] == (
        "Update plan: 2 added, 3 modified, 2 removed, 1 retried, 2 unchanged, 1 moved."
    )
    run = SQLiteManifest(data, read_only=True).latest_run()
    assert run is not None
    totals = dict(run.totals)
    assert {k: totals[k] for k in ("added", "modified", "removed", "retried")} == {
        k: total[k] for k in ("added", "modified", "removed", "retried")
    }
    assert totals["unchanged"] == total["unchanged"]
    refreshed = SQLiteManifest(data, read_only=True).get_source("e.txt")
    assert refreshed is not None and refreshed.source.mtime == NEW


def test_the_plan_text_counts_first_and_paths_on_request(
    indexed: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    books, notes, data = indexed
    _change_every_kind(books, data, notes)

    code, out, _ = _cli(capsys, "update", "--plan")

    assert code == 0
    assert out.splitlines() == [
        "Update plan. Nothing was changed. The corpus or the index can change "
        "before `docgrep update` runs.",
        "Root books: 2 added, 1 modified, 2 removed, 1 retried, 1 refreshed, "
        "4 unchanged, 1 moved.",
        "  Modified by reason: 1 content changed.",
        "Kept 2 sources of root notes, which is not part of this run.",
    ]

    code, out, _ = _cli(capsys, "update", "--plan", "--paths", "--no-retry-failed")

    assert code == 0
    assert "  moved:\n    f.txt -> moved/f.txt\n" in out
    assert "  removed:\n    d.txt\n    f.txt\n" in out
    assert "  modified:\n    c.txt (content changed)\n" in out
    assert "retried:" not in out and ", 0 retried, " in out


def test_the_plan_runs_while_another_process_holds_the_lock(
    indexed: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    books, _, data = indexed
    (books / "d.txt").unlink()
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl, sys\n"
            "handle = open(sys.argv[1], 'a+b')\n"
            "fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
            "print('ready', flush=True)\n"
            "sys.stdin.read()\n",
            str(data / "index.lock"),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline() == "ready\n"

        code, out, err = _cli(capsys, "update", "--plan", "--json")

        assert code == 0, err
        assert json.loads(out)["total"]["removed"] == 1
        code, _, err = _cli(capsys, "update")
        assert code == 1 and err.startswith("index_lock_held: ")
    finally:
        assert holder.stdin is not None
        holder.stdin.close()
        holder.wait(timeout=30)


def test_json_and_paths_need_plan(
    indexed: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    for option in ("--json", "--paths"):
        code, out, err = _cli(capsys, "update", option)
        assert (code, out) == (2, "")
        assert err == "usage: --json and --paths require --plan\n"


def test_the_plan_without_an_index(
    library: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    _, _, data = library
    before = _snapshot(data)

    code, out, err = _cli(capsys, "update", "--plan")

    assert (code, out) == (1, "")
    assert err.startswith("index_incomplete: ")
    assert _snapshot(data) == before


def test_the_guard_stops_an_update_that_empties_a_root(
    indexed: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    # Default thresholds: a root of 2 sources is far below them.
    books, notes, data = indexed
    for path in notes.iterdir():
        path.unlink()  # an emptied (or unmounted) folder
    os.utime(books / "e.txt", (NEW, NEW))  # a pending refresh
    before = _snapshot(data)

    code, out, err = _cli(capsys, "update")

    assert code == 1
    assert out.startswith("Update plan: 0 added, 0 modified, 2 removed, 1 retried")
    assert err.startswith("removal_guard: Update stopped before any change: ")
    assert "root 'notes' would lose all its 2 sources." in err
    assert "docgrep update --yes" in err
    assert _snapshot(data) == before

    code, out, _ = _cli(capsys, "update", "--plan")
    assert code == 0
    assert out.splitlines()[-1] == (
        "Removal guard: root notes would lose all its 2 sources; `docgrep update` "
        "would stop without changing anything unless run with --yes."
    )
    code, out, _ = _cli(capsys, "update", "--plan", "--json")
    assert json.loads(out)["guarded_roots"] == [
        {
            "root": "notes",
            "removals": 2,
            "recorded": 2,
            "threshold": 50,
            "reason": "empties_root",
        }
    ]
    assert _snapshot(data) == before

    code, out, err = _cli(capsys, "update", "--yes")

    assert code == 0, err
    manifest = SQLiteManifest(data, read_only=True)
    assert manifest.get_source("notes/x.txt") is None
    assert manifest.get_source("notes/y.txt") is None
    record = manifest.get_source("e.txt")
    assert record is not None and record.source.mtime == NEW


def test_the_guard_stops_removals_above_the_threshold(
    indexed: tuple[Path, Path, Path],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    books, _, data = indexed
    monkeypatch.setenv("DOCGREP_UPDATE_GUARD_COUNT", "1")
    (books / "c.txt").unlink()
    (books / "d.txt").unlink()
    before = _snapshot(data)

    code, _, err = _cli(capsys, "update")

    assert code == 1
    assert "root 'books' would lose 2 of 8 sources (at most 1 allowed)" in err
    assert _snapshot(data) == before
    code, out, _ = _cli(capsys, "update", "--plan", "--json")
    assert [(g["root"], g["reason"]) for g in json.loads(out)["guarded_roots"]] == [
        ("books", "threshold")
    ]


def test_removing_one_of_two_sources_is_applied(
    indexed: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    _, notes, data = indexed
    (notes / "x.txt").unlink()

    code, _, err = _cli(capsys, "update")

    assert code == 0, err
    manifest = SQLiteManifest(data, read_only=True)
    assert manifest.get_source("notes/x.txt") is None
    assert manifest.get_source("notes/y.txt") is not None


def test_forget_root_is_not_guarded(
    indexed: tuple[Path, Path, Path],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DOCGREP_UPDATE_GUARD_COUNT", "0")

    code, out, err = _cli(capsys, "forget-root", "notes", "--yes")

    assert code == 0, err
    assert out.startswith("Forgot root notes: 2 sources and ")


def test_plan_and_yes_exclude_each_other(
    indexed: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["update", "--plan", "--yes"])
    assert stopped.value.code == 2
