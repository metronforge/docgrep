"""`docgrep diff`: a folder, such as a backup copy, against the indexed
sources of a root, over the real manifest (ADR-0027)."""

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from docgrep.api.cli import main
from roots_support import environment, save_roots

OLD = 1_700_000_000
NEW = 1_700_100_000

BOOKS = {
    "a.txt": "A rare blue comet crosses the winter sky above the harbour.",
    "b.txt": "Recipe for bread with rye flour and caraway seeds.",
    "c.txt": "Field notes on the migration of cranes in autumn.",
    "d.txt": "Letters from a lighthouse keeper to his sister.",
    "shelf/e.md": "# Moss\n\nGarden notes about green moss on old stones.",
}
NOTES = {"x.txt": "A yellow kite drifts over the quiet meadow."}


def _write(root: Path, files: dict[str, str], mtime: int = OLD) -> None:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        os.utime(path, (mtime, mtime))


def _cli(capsys: pytest.CaptureFixture[str], *args: str) -> Any:
    code = main(list(args))
    out, err = capsys.readouterr()
    return code, out, err


def _snapshot(data: Path) -> dict[str, str]:
    return {
        str(path.relative_to(data)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(data.rglob("*"))
        if path.is_file()
    }


@pytest.fixture
def backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> tuple[Path, Path]:
    """An indexed library of two roots and a backup copy of ``books`` with
    one file of each kind of difference."""
    books, notes, data = tmp_path / "books", tmp_path / "notes", tmp_path / "data"
    _write(books, BOOKS)
    _write(notes, NOTES)
    for name, value in {**environment(data), "DOCGREP_LOG_LEVEL": "WARNING"}.items():
        monkeypatch.setenv(name, value)
    save_roots(data, ("books", books, True), ("notes", notes, False))
    assert _cli(capsys, "index")[0] == 0
    copy = tmp_path / "backup"
    shutil.copytree(books, copy)  # times preserved
    (copy / "c.txt").unlink()  # only in the index
    _write(copy, {"z.txt": "Only the backup has this pamphlet."})  # only in folder
    _write(copy, {"b.txt": "Recipe for cake."}, NEW)  # different, other size
    # Same size and time, other content: only a hash tells.
    same_stat = BOOKS["d.txt"].replace("sister", "cousin")
    _write(copy, {"d.txt": same_stat})
    return copy, data


def test_diff_by_stat_and_hash(
    backup: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    copy, data = backup
    before = _snapshot(data)

    code, out, err = _cli(capsys, "diff", str(copy), "--root", "books", "--json")

    assert code == 0, err
    assert json.loads(out) == {
        "folder": str(copy),
        "root": "books",
        "compared_by": "stat",
        "hashed": 1,
        "same": 3,
        "different": 1,
        "only_in_index": 1,
        "only_in_folder": 1,
        "paths": None,
    }

    code, out, _ = _cli(
        capsys, "diff", str(copy), "--root", "books", "--hash", "--json", "--paths"
    )

    assert code == 0
    body = json.loads(out)
    assert (body["compared_by"], body["hashed"], body["same"]) == ("hash", 4, 2)
    assert body["paths"] == {
        "different": ["b.txt", "d.txt"],
        "only_in_index": ["c.txt"],
        "only_in_folder": ["z.txt"],
    }
    assert _snapshot(data) == before


def test_diff_text(
    backup: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    copy, _ = backup

    code, out, _ = _cli(capsys, "diff", str(copy), "--root", "books", "--paths")

    assert code == 0
    assert out.splitlines() == [
        f"Folder {copy} against root books, compared by size and modification "
        "time, hashing only the files whose stat differs (1 files hashed):",
        "  same: 3",
        "  different: 1",
        "    b.txt",
        "  only in the index: 1",
        "    c.txt",
        "  only in the folder: 1",
        "    z.txt",
    ]


def test_a_prefixed_root_is_compared_by_relative_path(
    backup: tuple[Path, Path], capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    copy = tmp_path / "notes-copy"
    shutil.copytree(tmp_path / "notes", copy)

    code, out, _ = _cli(capsys, "diff", str(copy), "--root", "notes", "--json")

    assert code == 0
    body = json.loads(out)
    assert (body["same"], body["only_in_index"], body["only_in_folder"]) == (1, 0, 0)


def test_the_root_must_be_named_when_several_are_known(
    backup: tuple[Path, Path], capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    copy, data = backup

    code, out, err = _cli(capsys, "diff", str(copy))

    assert (code, out) == (2, "")
    assert err == (
        "validation_error: several roots are configured (books, notes); choose one "
        "with --root NAME (reason: ambiguous_root)\n"
    )
    code, _, err = _cli(capsys, "diff", str(copy), "--root", "music")
    assert code == 2 and "unknown root 'music'; known roots: books, notes" in err

    # One configured root is the default, even when the index knows others.
    save_roots(data, ("books", tmp_path / "books", True))
    code, out, err = _cli(capsys, "diff", str(copy), "--json")
    assert code == 0, err
    assert json.loads(out)["root"] == "books"


def test_diff_without_an_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name, value in environment(tmp_path / "data").items():
        monkeypatch.setenv(name, value)

    code, out, err = _cli(capsys, "diff", str(tmp_path))

    assert (code, out) == (1, "")
    assert err.startswith("index_incomplete: ")
    assert not (tmp_path / "data").exists()
