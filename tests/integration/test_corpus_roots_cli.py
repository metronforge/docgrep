"""The corpus roots on the command line: `docgrep roots`, one-off --root
runs and the offer to save them, forget-root and the root filter (ADR-0023)."""

import io
import json
from pathlib import Path
from typing import Any

import pytest

from docgrep.api.cli import main
from roots_support import environment, save_roots, two_roots


def _env(monkeypatch: pytest.MonkeyPatch, data: Path, **extra: str) -> None:
    # Only warnings are logged, so standard error holds the messages tested.
    values = {**environment(data), "DOCGREP_LOG_LEVEL": "WARNING", **extra}
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def _cli(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    status = main(list(args))
    out, err = capsys.readouterr()
    return status, out, err


def _saved(data: Path) -> list[dict[str, Any]]:
    roots: list[dict[str, Any]] = json.loads((data / "roots.json").read_text("utf-8"))[
        "roots"
    ]
    return roots


@pytest.fixture
def folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    books, notes = two_roots(tmp_path)
    data = tmp_path / "data"
    _env(monkeypatch, data)
    return books.resolve(), notes.resolve(), data


def test_roots_are_added_listed_and_removed(
    folders: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    books, notes, data = folders

    code, out, _ = _cli(capsys, "roots", "add", "books", str(books), "--legacy")
    assert code == 0
    assert out == (
        f"Added root books ({books}) to {data / 'roots.json'} as the legacy root: "
        "its sources keep their stored paths. Run `docgrep update` to index it.\n"
    )
    assert _cli(capsys, "roots", "add", "notes", str(notes))[0] == 0
    assert _saved(data) == [
        {"name": "books", "path": str(books), "legacy": True},
        {"name": "notes", "path": str(notes)},
    ]

    code, out, _ = _cli(capsys, "roots", "list")
    assert code == 0
    assert out == (
        f"books  {books}  (legacy)  0 indexed sources\n"
        f"notes  {notes}  0 indexed sources\n"
    )
    listing = json.loads(_cli(capsys, "roots", "list", "--json")[1])
    assert listing["origin"] == "file"
    assert listing["roots"][1] == {
        "name": "notes",
        "path": str(notes),
        "legacy": False,
        "readable": True,
        "indexed_sources": 0,
    }

    code, out, _ = _cli(capsys, "roots", "remove", "notes")
    assert code == 0 and out.startswith(
        f"Removed root notes from {data / 'roots.json'}."
    )
    assert [root["name"] for root in _saved(data)] == ["books"]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("NOTES", "{notes}"), "root name 'NOTES' is already used by root 'notes'"),
        (("bad name", "{books}"), "invalid root name 'bad name'"),
        (("sub", "{notes}/sub"), "is inside root 'notes'"),
        (("parent", "{parent}"), "is inside root 'parent'"),
        (("other", "{books}", "--legacy"), "only one root can be the legacy root"),
        (("shelf", "{shelf}"), "equals a top-level entry of the legacy root 'books'"),
        (("gone", "{parent}/missing"), "does not exist"),
    ],
)
def test_invalid_roots_are_refused_with_exit_2(
    folders: tuple[Path, Path, Path],
    capsys: pytest.CaptureFixture[str],
    args: tuple[str, ...],
    message: str,
) -> None:
    books, notes, data = folders
    (notes / "sub").mkdir()
    shelf = books.parent / "shelf"
    shelf.mkdir()
    save_roots(data, ("books", books, True), ("notes", notes, False))
    before = (data / "roots.json").read_text("utf-8")
    values = {"books": books, "notes": notes, "parent": books.parent, "shelf": shelf}
    if "--legacy" in args:
        # Another folder, so that only the second legacy root is wrong.
        values["books"] = shelf

    code, out, err = _cli(capsys, "roots", "add", *(a.format(**values) for a in args))

    assert (code, out) == (2, "")
    assert err.startswith("validation_error: ") and message in err
    assert (data / "roots.json").read_text("utf-8") == before


def test_removing_a_root_leaves_the_index_alone(
    folders: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    books, notes, data = folders
    save_roots(data, ("books", books, True), ("notes", notes, False))
    assert _cli(capsys, "index")[0] == 0
    manifest = (data / "manifest.sqlite3").read_bytes()

    assert _cli(capsys, "roots", "remove", "notes")[0] == 0

    assert (data / "manifest.sqlite3").read_bytes() == manifest
    status = json.loads(_cli(capsys, "status", "--json")[1])
    assert {r["name"]: r["configured"] for r in status["roots"]} == {
        "books": True,
        "notes": False,
    }
    code, out, _ = _cli(capsys, "status")
    assert code == 0
    assert "Root books (legacy): 2 indexed, 0 skipped, 0 failed" in out
    assert "Root notes (not configured): 2 indexed, 0 skipped, 0 failed" in out
    assert "root_not_configured: Sources of roots that are not configured" in out


def test_the_deprecated_variable_is_one_legacy_root_with_a_warning(
    folders: tuple[Path, Path, Path],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    books, notes, data = folders
    monkeypatch.setenv("DOCGREP_CORPUS_ROOT", str(books))

    code, out, err = _cli(capsys, "index")

    assert code == 0 and "2 indexed" in out
    assert err == (
        "warning: DOCGREP_CORPUS_ROOT is deprecated; save it as a root with: "
        f"docgrep roots add books {books} --legacy\n"
    )
    assert not (data / "roots.json").exists()
    listing = json.loads(_cli(capsys, "roots", "list", "--json")[1])
    assert listing["origin"] == "bootstrap"
    assert listing["roots"][0]["indexed_sources"] == 2

    # With a roots file the variable is ignored, and says so.
    save_roots(data, ("books", books, True), ("notes", notes, False))
    code, out, err = _cli(capsys, "update")
    assert code == 0 and "2 added" in out
    assert err == (
        "warning: DOCGREP_CORPUS_ROOT is ignored because "
        f"{data / 'roots.json'} exists\n"
    )


def test_without_roots_indexing_fails_and_reading_works(
    folders: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    books, _, data = folders
    save_roots(data, ("books", books, True))
    assert _cli(capsys, "index")[0] == 0
    (data / "roots.json").unlink()

    for command in (("index",), ("update",), ("dry-run",)):
        code, out, err = _cli(capsys, *command)
        assert code == 1 and out == "", command
        assert err.startswith("validation_error: no corpus roots configured"), command
    assert _cli(capsys, "discover-scans")[0] == 2

    code, out, _ = _cli(capsys, "search", "comet", "--json")
    assert code == 0 and json.loads(out)["results"]
    assert _cli(capsys, "sources")[0] == 0
    status = json.loads(_cli(capsys, "status", "--json")[1])
    assert status["index_state"] == "ready"
    assert status["roots"][0]["name"] == "_legacy"


def test_a_one_off_root_is_used_alone_and_saved_only_when_asked(
    folders: tuple[Path, Path, Path],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    books, notes, data = folders
    save_roots(data, ("books", books, True))
    saved = (data / "roots.json").read_text("utf-8")
    assert _cli(capsys, "index")[0] == 0

    code, out, err = _cli(capsys, "update", "--root", f"notes={notes}")

    assert code == 0 and "2 added" in out and "0 removed" in out
    assert "Kept 2 sources of root books, which is not part of this run." in out
    assert err == (
        f"Root notes ({notes}) is not saved, so future updates will not visit "
        f"it; save it with: docgrep roots add notes {notes}\n"
    )
    assert (data / "roots.json").read_text("utf-8") == saved

    (notes / "c.txt").write_text("Changed bread recipe.", "utf-8")
    code, out, err = _cli(
        capsys, "update", "--root", f"notes={notes}", "--no-save-roots"
    )
    assert code == 0 and "1 modified" in out and err == ""

    code, out, err = _cli(capsys, "update", "--root", str(notes), "--save-roots")
    assert code == 0 and f"Saved root notes ({notes}) in {data / 'roots.json'}." in out
    assert _saved(data)[1] == {"name": "notes", "path": str(notes)}


class _Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_on_a_terminal_the_user_is_asked(
    folders: tuple[Path, Path, Path],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    books, notes, data = folders
    save_roots(data, ("books", books, True))
    assert _cli(capsys, "index")[0] == 0
    answers = _Terminal("n\ny\n")
    monkeypatch.setattr("sys.stdin", answers)
    terminal = _Terminal()
    monkeypatch.setattr("sys.stdout", terminal)

    assert main(["update", "--root", f"notes={notes}"]) == 0
    assert [root["name"] for root in _saved(data)] == ["books"]
    assert main(["update", "--root", f"notes={notes}"]) == 0

    prompt = (
        f"Add root notes ({notes}) to the saved list so future updates include "
        "it? [y/N] "
    )
    assert terminal.getvalue().count(prompt) == 2
    assert _saved(data)[1] == {"name": "notes", "path": str(notes)}
    assert "legacy" not in _saved(data)[1]


def test_one_off_roots_are_checked_against_the_saved_ones(
    folders: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    books, notes, data = folders
    save_roots(data, ("books", books, True))

    code, _, err = _cli(capsys, "dry-run", "--root", f"books={notes}")
    assert code == 1
    assert err.startswith(
        f"validation_error: root 'books' is saved with the path {books}, but "
        f"--root gives {notes}"
    )
    code, out, _ = _cli(
        capsys, "dry-run", "--json", "--root", f"books={books}", "--root", str(notes)
    )
    report = json.loads(out)
    assert code == 0
    assert report["unsaved_roots"] == [{"name": "notes", "path": str(notes)}]
    assert [(r["root"], r["files"]) for r in report["by_root"]] == [
        ("books", 2),
        ("notes", 2),
    ]
    assert {item["path"] for item in report["files"]["items"]} == {
        "a.txt",
        "shelf/b.md",
        "notes/a.txt",
        "notes/c.txt",
    }
    assert not (data / "manifest.sqlite3").exists()


def test_forget_root_shows_counts_and_needs_yes(
    folders: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    books, notes, data = folders
    save_roots(data, ("books", books, True), ("notes", notes, False))
    assert _cli(capsys, "index")[0] == 0
    save_roots(data, ("books", books, True))

    code, out, _ = _cli(capsys, "forget-root", "notes")
    assert code == 1
    assert out.startswith(
        "forget-root notes would remove 2 sources (2 indexed, 0 skipped, 0 failed) and "
    )
    assert out.endswith(
        "passages from the index; the files are not touched. Run it again with "
        "--yes to remove them.\n"
    )
    code, out, _ = _cli(capsys, "forget-root", "notes", "--yes")
    assert code == 0 and out.startswith("Forgot root notes: 2 sources and ")
    code, _, err = _cli(capsys, "forget-root", "books", "--yes")
    assert code == 1 and "is the legacy root" in err
    assert (notes / "a.txt").is_file()


def test_the_root_filter_on_the_command_line(
    folders: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    books, notes, data = folders
    save_roots(data, ("books", books, True), ("notes", notes, False))
    assert _cli(capsys, "index")[0] == 0

    code, out, _ = _cli(capsys, "search", "kite comet", "--in-root", "notes", "--json")
    body = json.loads(out)
    assert code == 0 and body["applied_filters"] == {"roots": ["notes"]}
    assert {r["source"]["root"] for r in body["results"]} == {"notes"}
    code, out, _ = _cli(capsys, "sources", "--in-root", "books", "--json")
    assert [s["source_id"] for s in json.loads(out)["sources"]] == [
        "a.txt",
        "shelf/b.md",
    ]
    code, out, _ = _cli(capsys, "search", "kite", "--in-root", "music")
    assert code == 1


def test_a_missing_root_stops_the_update_on_the_command_line(
    folders: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    books, notes, data = folders
    save_roots(data, ("books", books, True), ("notes", notes, False))
    assert _cli(capsys, "index")[0] == 0
    for path in notes.iterdir():
        path.unlink()
    notes.rmdir()

    code, out, err = _cli(capsys, "update")

    assert (code, out) == (1, "")
    assert err == (
        f"validation_error: corpus root 'notes' is not accessible: {notes} "
        f"(reason: [Errno 2] No such file or directory: '{notes}')\n"
    )
    status = json.loads(_cli(capsys, "status", "--json")[1])
    assert status["roots"][1]["readable"] is False
    assert status["roots"][1]["indexed"] == 2
