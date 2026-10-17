"""Corpus roots: names, the stored identity of their sources and validation."""

from pathlib import Path

import pytest

from docgrep.domain.errors import ValidationError
from docgrep.domain.models import make_passage_id
from docgrep.domain.roots import (
    UNCONFIGURED_LEGACY_ROOT,
    CorpusRoot,
    CorpusRoots,
    known_root_names,
    prefixes_value,
    recorded_prefixes,
    root_name_of,
)
from docgrep.domain.run_state import roots_drift

BOOKS = CorpusRoot("books", Path("/srv/books"), legacy=True)
NOTES = CorpusRoot("notes", Path("/srv/notes"))


def test_the_legacy_root_keeps_unprefixed_paths_and_others_are_prefixed() -> None:
    assert BOOKS.prefix == ""
    assert BOOKS.stored_path("a.txt") == "a.txt"
    assert NOTES.prefix == "notes/"
    assert NOTES.stored_path("a.txt") == "notes/a.txt"
    assert NOTES.relative_path("notes/a.txt") == "a.txt"


# make_passage_id("shelf/a.txt", 0), computed before corpus roots existed.
LEGACY_PASSAGE_ID = "3112da2fdb4d39cfa52c47976354193605bbf6f2d63e34fd2a23155f23f024ef"


def test_a_legacy_source_keeps_its_passage_id() -> None:
    stored = BOOKS.stored_path("shelf/a.txt")
    assert stored == "shelf/a.txt"
    assert make_passage_id(stored, 0) == LEGACY_PASSAGE_ID
    assert make_passage_id(NOTES.stored_path("shelf/a.txt"), 0) != LEGACY_PASSAGE_ID


def test_the_same_relative_path_in_two_roots_is_two_sources() -> None:
    roots = CorpusRoots((BOOKS, NOTES))
    legacy, prefixed = BOOKS.stored_path("a.txt"), NOTES.stored_path("a.txt")
    assert legacy != prefixed
    assert make_passage_id(legacy, 0) != make_passage_id(prefixed, 0)
    assert roots.root_of(legacy) == "books"
    assert roots.root_of(prefixed) == "notes"


def test_the_root_of_a_source_is_derived_from_the_prefix() -> None:
    roots = CorpusRoots((BOOKS, NOTES))
    assert roots.root_of("notes/sub/b.md") == "notes"
    # A legacy folder named like no prefixed root stays legacy.
    assert roots.root_of("other/b.md") == "books"
    assert roots.root_of("b.md") == "books"
    assert root_name_of("x/y", {"x"}, "legacy") == "x"
    assert root_name_of("x", {"x"}, "legacy") == "legacy"


def test_recorded_prefixes_keep_unconfigured_roots_apart() -> None:
    roots = CorpusRoots((BOOKS,), recorded=frozenset({"notes"}))
    assert roots.root_of("notes/a.txt") == "notes"
    assert roots.owner("notes/a.txt") is None
    assert roots.locate("notes/a.txt") is None
    assert roots.locate("shelf/a.txt") == Path("/srv/books/shelf/a.txt")


def test_without_a_legacy_root_unprefixed_sources_have_a_reserved_root() -> None:
    roots = CorpusRoots((NOTES,))
    assert roots.root_of("a.txt") == UNCONFIGURED_LEGACY_ROOT
    assert roots.owner("a.txt") is None
    assert known_root_names(roots, ["a.txt", "notes/b.txt"]) == (
        UNCONFIGURED_LEGACY_ROOT,
        "notes",
    )


def test_stored_paths_map_back_from_files() -> None:
    roots = CorpusRoots((BOOKS, NOTES))
    assert roots.stored_path(Path("/srv/notes/sub/a.txt")) == "notes/sub/a.txt"
    assert roots.stored_path(Path("/srv/books/a.txt")) == "a.txt"
    assert roots.stored_path(Path("/srv/other/a.txt")) is None
    assert roots.stored_path(Path("/srv/notes")) is None


@pytest.mark.parametrize("name", ["", "-x", ".x", "a/b", "a b", "_legacy", "ä"])
def test_invalid_names_are_refused(name: str) -> None:
    with pytest.raises(ValidationError, match="invalid root name"):
        CorpusRoot(name, Path("/srv/x"))


def test_names_are_unique_case_insensitively() -> None:
    with pytest.raises(ValidationError) as raised:
        CorpusRoots((NOTES, CorpusRoot("Notes", Path("/srv/other"))))
    assert str(raised.value) == (
        "root name 'Notes' is already used by root 'notes' "
        "(names are compared case-insensitively)"
    )


def test_two_legacy_roots_are_refused() -> None:
    with pytest.raises(ValidationError) as raised:
        CorpusRoots((BOOKS, CorpusRoot("old", Path("/srv/old"), legacy=True)))
    assert str(raised.value) == (
        "only one root can be the legacy root; 'books' already is"
    )


@pytest.mark.parametrize(
    ("first", "second", "inner", "outer"),
    [
        (Path("/srv/books"), Path("/srv/books/sub"), "second", "first"),
        (Path("/srv/books/sub"), Path("/srv/books"), "first", "second"),
        (Path("/srv/books"), Path("/srv/books"), "second", "first"),
    ],
)
def test_nested_roots_are_refused_in_both_directions(
    first: Path, second: Path, inner: str, outer: str
) -> None:
    with pytest.raises(ValidationError) as raised:
        CorpusRoots((CorpusRoot("first", first), CorpusRoot("second", second)))
    message = str(raised.value)
    assert f"root {inner!r}" in message and f"is inside root {outer!r}" in message
    assert message.endswith(
        "a sub-folder of an existing root is a path prefix filter, not a new root"
    )


def test_a_prefixed_name_may_not_equal_a_legacy_top_level_entry() -> None:
    roots = CorpusRoots((BOOKS, NOTES))
    roots.check_legacy_entries({"shelf", "a.txt"})
    with pytest.raises(ValidationError) as raised:
        roots.check_legacy_entries({"notes"})
    assert str(raised.value) == (
        "root name 'notes' equals a top-level entry of the legacy root 'books'; "
        "its source ids would collide with that entry's. Choose another name"
    )


def test_run_configuration_values_round_trip() -> None:
    assert recorded_prefixes({"root_prefixes": prefixes_value(["b", "a"])}) == {
        "a",
        "b",
    }
    assert recorded_prefixes({}) == frozenset()
    assert recorded_prefixes({"root_prefixes": "not json"}) == frozenset()


def test_a_resume_compares_the_roots_of_the_run() -> None:
    roots = CorpusRoots((BOOKS, NOTES))
    recorded = {"corpus_roots": roots.snapshot()}
    assert not roots_drift(recorded, roots.snapshot(), single_legacy=False)
    other = CorpusRoots((BOOKS,))
    assert roots_drift(recorded, other.snapshot(), single_legacy=True)
    # A run recorded before roots had names matches one legacy root only.
    assert not roots_drift({}, other.snapshot(), single_legacy=True)
    assert roots_drift({}, roots.snapshot(), single_legacy=False)
