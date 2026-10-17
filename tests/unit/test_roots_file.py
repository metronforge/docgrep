"""The roots file: atomic writes and validation on add and on load."""

import json
from pathlib import Path

import pytest

from docgrep.adapters.roots import (
    checked_folder,
    read_roots_file,
    roots_file,
    with_root,
    without_root,
    write_roots_file,
)
from docgrep.domain.errors import ValidationError
from docgrep.domain.roots import CorpusRoot, CorpusRoots


def _roots(tmp_path: Path) -> CorpusRoots:
    books = tmp_path / "books"
    notes = tmp_path / "notes"
    books.mkdir(exist_ok=True)
    notes.mkdir(exist_ok=True)
    return CorpusRoots(
        (CorpusRoot("books", books, legacy=True), CorpusRoot("notes", notes))
    )


def test_a_missing_file_reads_as_none(tmp_path: Path) -> None:
    assert read_roots_file(tmp_path / "data") is None


def test_roots_round_trip_in_the_documented_shape(tmp_path: Path) -> None:
    data = tmp_path / "data"
    roots = _roots(tmp_path)

    write_roots_file(data, roots)

    assert json.loads(roots_file(data).read_text("utf-8")) == {
        "version": 1,
        "roots": [
            {"name": "books", "path": str(tmp_path / "books"), "legacy": True},
            {"name": "notes", "path": str(tmp_path / "notes")},
        ],
    }
    assert read_roots_file(data) == roots
    assert not [p for p in data.iterdir() if p.name != "roots.json"]


def test_a_failed_write_leaves_the_old_file_intact(tmp_path: Path) -> None:
    data = tmp_path / "data"
    roots = _roots(tmp_path)
    write_roots_file(data, CorpusRoots(roots.roots[:1]))
    before = roots_file(data).read_text("utf-8")

    def fail(_source: str, _target: Path) -> None:
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        write_roots_file(data, roots, replace=fail)

    assert roots_file(data).read_text("utf-8") == before
    assert [p.name for p in data.iterdir()] == ["roots.json"]


def test_adding_checks_the_folder(tmp_path: Path) -> None:
    with pytest.raises(ValidationError) as raised:
        checked_folder(tmp_path / "missing")
    assert str(raised.value) == f"root path {tmp_path / 'missing'} does not exist"
    (tmp_path / "file").write_text("x", "utf-8")
    with pytest.raises(ValidationError) as raised:
        checked_folder(tmp_path / "file")
    assert str(raised.value) == f"root path {tmp_path / 'file'} is not a directory"
    assert checked_folder(tmp_path / "." / "") == tmp_path.resolve()


def test_adding_validates_the_whole_list(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    (tmp_path / "books" / "shelf").mkdir()
    (tmp_path / "shelf").mkdir()
    (tmp_path / "other").mkdir()

    with pytest.raises(ValidationError, match="is already used by root 'notes'"):
        with_root(roots, CorpusRoot("NOTES", tmp_path / "other"))
    with pytest.raises(ValidationError, match="only one root can be the legacy"):
        with_root(roots, CorpusRoot("other", tmp_path / "other", legacy=True))
    with pytest.raises(ValidationError, match="path prefix filter, not a new root"):
        with_root(roots, CorpusRoot("sub", tmp_path / "books" / "shelf"))
    with pytest.raises(ValidationError, match="top-level entry of the legacy root"):
        with_root(roots, CorpusRoot("shelf", tmp_path / "shelf"))
    added = with_root(roots, CorpusRoot("other", tmp_path / "other"))
    assert added.names == ("books", "notes", "other")


def test_removing_an_unknown_root_is_an_error(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    assert without_root(roots, "notes").names == ("books",)
    with pytest.raises(ValidationError) as raised:
        without_root(roots, "music")
    assert (
        str(raised.value) == "no saved root is named 'music'; saved roots: books, notes"
    )


@pytest.mark.parametrize(
    ("document", "reason"),
    [
        ("not json", "not JSON"),
        ('{"version": 2, "roots": []}', "expected version 1"),
        ('{"version": 1, "roots": {}}', "roots must be a list"),
        ('{"version": 1, "roots": [{"name": "a"}]}', "needs a name and a path"),
        ('{"version": 1, "roots": [{"name": "a", "path": "rel"}]}', "absolute"),
    ],
)
def test_a_malformed_file_is_reported(
    tmp_path: Path, document: str, reason: str
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    roots_file(data).write_text(document, "utf-8")
    with pytest.raises(ValidationError, match=reason):
        read_roots_file(data)


def test_loading_validates_again(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    books = tmp_path / "books"
    books.mkdir()
    for entries, message in [
        (
            [
                {"name": "a", "path": str(books), "legacy": True},
                {"name": "b", "path": str(books / "x"), "legacy": True},
            ],
            "only one root can be the legacy root",
        ),
        (
            [{"name": "a", "path": str(books)}, {"name": "b", "path": str(books)}],
            "path prefix filter",
        ),
        ([{"name": "a b", "path": str(books)}], "invalid root name"),
    ]:
        roots_file(data).write_text(
            json.dumps({"version": 1, "roots": entries}), "utf-8"
        )
        with pytest.raises(ValidationError, match=message):
            read_roots_file(data)


def test_a_root_that_is_gone_still_loads(tmp_path: Path) -> None:
    data = tmp_path / "data"
    roots = _roots(tmp_path)
    write_roots_file(data, roots)
    (tmp_path / "notes").rmdir()
    (tmp_path / "books").rmdir()
    assert read_roots_file(data) == roots
