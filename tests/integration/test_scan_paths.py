"""Stat-only scanning of chosen stored paths, as a scoped update reads them."""

from pathlib import Path

import pytest

from docgrep.adapters.corpus import scan_corpus, scan_paths
from docgrep.domain.errors import ValidationError
from docgrep.domain.roots import CorpusRoot, CorpusRoots


def _roots(tmp_path: Path) -> CorpusRoots:
    books, notes = tmp_path / "books", tmp_path / "notes"
    for folder, name in ((books, "a.txt"), (notes, "x.txt")):
        folder.mkdir()
        (folder / name).write_text(name, encoding="utf-8")
    (books / "sub").mkdir()
    (books / "sub" / "b.md").write_text("b", encoding="utf-8")
    (books / "skip.tmp").write_text("t", encoding="utf-8")
    return CorpusRoots(
        (CorpusRoot("books", books, legacy=True), CorpusRoot("notes", notes))
    )


def test_it_reports_what_a_full_scan_reports_for_the_same_paths(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    full = {f.path: f for f in scan_corpus(roots, exclude_globs=("*.tmp",))}
    wanted = ["a.txt", "sub/b.md", "notes/x.txt", "skip.tmp"]

    found = scan_paths(roots, wanted, exclude_globs=("*.tmp",))

    assert {f.path: f for f in found} == {p: full[p] for p in wanted}
    assert next(f for f in found if f.path == "skip.tmp").excluded_by == "*.tmp"


def test_missing_files_and_unknown_roots_are_left_out(tmp_path: Path) -> None:
    roots = _roots(tmp_path)

    found = scan_paths(roots, ["gone.txt", "sub", "other/x.txt"] + ["a.txt"])

    assert [f.path for f in found] == ["a.txt"]


def test_an_unreadable_root_raises_instead_of_reporting_files_gone(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    (tmp_path / "notes").rename(tmp_path / "unmounted")

    with pytest.raises(ValidationError):
        scan_paths(roots, ["notes/x.txt"])
    assert [f.path for f in scan_paths(roots, ["a.txt"])] == ["a.txt"]
