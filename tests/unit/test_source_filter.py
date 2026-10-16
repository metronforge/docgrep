"""The shared source filter, metadata rules, sorting and request bounds (UC-08)."""

import pytest

from docgrep.domain.errors import ValidationError
from docgrep.domain.models import FileType, Source
from docgrep.domain.sources import (
    MAX_PAGE_SIZE,
    BrowseRequest,
    SourceFilter,
    has_metadata,
    matches,
    sort_sources,
)


def _source(
    path: str,
    *,
    author: str | None = None,
    year: int | None = None,
    title: str | None = None,
    file_type: FileType | None = None,
    indexed_at: str | None = None,
) -> Source:
    suffix = path.rsplit(".", 1)[-1]
    kind = file_type or (FileType.MARKDOWN if suffix == "md" else FileType(suffix))
    return Source(
        source_id=path,
        path=path,
        file_type=kind,
        title=title,
        author=author,
        year=year,
        indexed_at=indexed_at,
    )


ARNOLD = _source(
    "Books/Math/Arnold - Mathematical Methods (1989).pdf",
    author="V. I. Arnold",
    year=1989,
    title="Mathematical Methods of Classical Mechanics",
)
LANDAU = _source(
    "Books/Physics/Ландау [Том 1].djvu",
    author="Лев Ландау",
    year=1958,
    title="Механика",
)
NOTES = _source("notes/read me.md")


def test_an_empty_filter_matches_everything() -> None:
    assert all(matches(SourceFilter(), s) for s in (ARNOLD, LANDAU, NOTES))
    assert SourceFilter().applied() == {}


def test_fields_combine_with_and_and_lists_with_or() -> None:
    both = SourceFilter(authors=("arnold", "ландау"), file_types=("pdf",))
    either = SourceFilter(authors=("arnold", "ландау"))

    assert [matches(both, s) for s in (ARNOLD, LANDAU)] == [True, False]
    assert [matches(either, s) for s in (ARNOLD, LANDAU)] == [True, True]


def test_authors_match_case_insensitive_substrings_in_any_script() -> None:
    assert matches(SourceFilter(authors=("ARNOLD",)), ARNOLD)
    assert matches(SourceFilter(authors=("i. arn",)), ARNOLD)
    assert matches(SourceFilter(authors=("ЛАНДАУ",)), LANDAU)
    assert not matches(SourceFilter(authors=("Arnolds",)), ARNOLD)


def test_null_metadata_never_matches_a_filter_on_that_field() -> None:
    assert not matches(SourceFilter(authors=("a",)), NOTES)
    assert not matches(SourceFilter(year_from=0), NOTES)
    assert not matches(SourceFilter(year_to=3000), NOTES)


def test_year_bounds_are_inclusive_and_open_ended() -> None:
    assert matches(SourceFilter(year_from=1989, year_to=1989), ARNOLD)
    assert matches(SourceFilter(year_from=1950), LANDAU)
    assert not matches(SourceFilter(year_from=1959), LANDAU)
    assert matches(SourceFilter(year_to=1958), LANDAU)
    assert not matches(SourceFilter(year_to=1988), ARNOLD)


def test_year_from_after_year_to_is_rejected() -> None:
    with pytest.raises(ValidationError) as raised:
        SourceFilter(year_from=2000, year_to=1999)
    assert raised.value.details == {"year_from": 2000, "year_to": 1999}


def test_file_types_are_case_insensitive_and_unknown_ones_name_the_allowed() -> None:
    assert matches(SourceFilter(file_types=("PDF",)), ARNOLD)
    assert matches(SourceFilter(file_types=(" DjVu ",)), LANDAU)
    for unknown in ("ocr_pdf", "unsupported", "doc", ""):
        with pytest.raises(ValidationError) as raised:
            SourceFilter(file_types=(unknown,))
        allowed = ["djvu", "docx", "epub", "markdown", "pdf", "txt"]
        assert raised.value.details == {"field": "file_types", "allowed": allowed}
        assert ", ".join(allowed) in str(raised.value)


@pytest.mark.parametrize(
    "name", ["authors", "source_ids", "path_prefixes", "path_globs"]
)
@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_values_are_rejected(name: str, blank: str) -> None:
    with pytest.raises(ValidationError) as raised:
        SourceFilter(**{name: ("ok", blank)})  # type: ignore[arg-type]  # kwargs by name
    assert raised.value.details == {"field": name}


def test_source_ids_match_exactly_and_case_sensitively() -> None:
    assert matches(SourceFilter(source_ids=(NOTES.source_id,)), NOTES)
    assert not matches(SourceFilter(source_ids=("notes/READ ME.md",)), NOTES)
    assert not matches(SourceFilter(source_ids=("notes/read",)), NOTES)


def test_path_prefixes_are_plain_case_sensitive_prefixes_of_the_stored_path() -> None:
    assert matches(SourceFilter(path_prefixes=("Books/Math/",)), ARNOLD)
    assert matches(SourceFilter(path_prefixes=("Books/Ma",)), ARNOLD)
    assert matches(SourceFilter(path_prefixes=("/Books/Math/",)), ARNOLD)
    assert matches(SourceFilter(path_prefixes=("Books\\Math\\",)), ARNOLD)
    assert matches(SourceFilter(path_prefixes=("notes/read ",)), NOTES)
    assert matches(SourceFilter(path_prefixes=("Books/Physics/Ландау [",)), LANDAU)
    assert not matches(SourceFilter(path_prefixes=("books/",)), ARNOLD)


def test_path_globs_use_the_shared_glob_rules() -> None:
    assert matches(SourceFilter(path_globs=("Books/**/*.pdf",)), ARNOLD)
    assert matches(SourceFilter(path_globs=("*.djvu",)), LANDAU)
    assert matches(SourceFilter(path_globs=("Books/",)), LANDAU)
    assert matches(SourceFilter(path_globs=("notes/read me.md",)), NOTES)
    assert matches(SourceFilter(path_globs=("*(1989).pdf",)), ARNOLD)
    # Brackets are a character class, as in every docgrep glob.
    assert matches(SourceFilter(path_globs=("*[[]Том 1].djvu",)), LANDAU)
    assert not matches(SourceFilter(path_globs=("Books/*.pdf",)), ARNOLD)
    assert not matches(SourceFilter(path_globs=("books/**",)), ARNOLD)


def test_stored_backslashes_are_compared_as_forward_slashes() -> None:
    stored = _source("Books\\Math\\a.pdf")

    assert matches(SourceFilter(path_prefixes=("Books/Math/",)), stored)
    assert matches(SourceFilter(path_globs=("Books/**",)), stored)


def test_applied_form_is_canonical() -> None:
    applied = SourceFilter(
        source_ids=("b", "a", "b"),
        authors=("Zed", "arnold", "Zed"),
        year_from=1900,
        file_types=("PDF", "txt", "pdf"),
        path_prefixes=("x\\y/", "a/"),
        path_globs=("*.pdf",),
    ).applied()

    assert applied == {
        "source_ids": ["a", "b"],
        "authors": ["Zed", "arnold"],
        "year_from": 1900,
        "file_types": ["pdf", "txt"],
        "path_prefixes": ["a/", "x/y/"],
        "path_globs": ["*.pdf"],
    }
    assert list(applied) == [
        "source_ids",
        "authors",
        "year_from",
        "file_types",
        "path_prefixes",
        "path_globs",
    ]


def test_has_metadata_values() -> None:
    year_only = _source("y.txt", year=2001)
    author_only = _source("a.txt", author="A")
    blank_author = _source("b.txt", author="  ")
    rows = {
        "any": [True, True, True, True, True],
        "with_year": [True, True, False, False, False],
        "with_author": [True, False, True, False, False],
        "with_both": [True, False, False, False, False],
        "without_any": [False, False, False, True, True],
    }
    for mode, expected in rows.items():
        found = [
            has_metadata(mode, s)  # type: ignore[arg-type]  # every literal value
            for s in (ARNOLD, year_only, author_only, NOTES, blank_author)
        ]
        assert found == expected, mode


def test_applied_filters_include_has_metadata_unless_any() -> None:
    assert BrowseRequest().applied_filters() == {}
    request = BrowseRequest(SourceFilter(year_to=2000), has_metadata="without_any")
    assert request.applied_filters() == {"year_to": 2000, "has_metadata": "without_any"}


@pytest.mark.parametrize(
    ("arguments", "field"),
    [
        ({"page": 0}, "page"),
        ({"page": -1}, "page"),
        ({"page_size": 0}, "page_size"),
        ({"page_size": MAX_PAGE_SIZE + 1}, "page_size"),
        ({"sort_by": "relevance"}, "sort_by"),
        ({"sort_order": "up"}, "sort_order"),
        ({"has_metadata": "some"}, "has_metadata"),
    ],
)
def test_request_bounds_and_enums(arguments: dict[str, object], field: str) -> None:
    with pytest.raises(ValidationError) as raised:
        BrowseRequest(**arguments)  # type: ignore[arg-type]  # invalid on purpose
    assert raised.value.details["field"] == field


def test_page_size_500_is_allowed() -> None:
    assert BrowseRequest(page_size=MAX_PAGE_SIZE).page_size == MAX_PAGE_SIZE


def _ids(sources: list[Source]) -> list[str]:
    return [source.source_id for source in sources]


TIED = [
    _source("d.txt", year=2000, title="beta"),
    _source("b.txt", year=1990, title="Alpha"),
    _source("c.txt", year=2000, title="alpha"),
    _source("a.txt", title="Gamma"),
    _source("e.txt", year=1990),
]


def test_sorting_breaks_ties_by_source_id_and_puts_nulls_last() -> None:
    assert _ids(sort_sources(TIED, "year", "asc")) == [
        "b.txt",
        "e.txt",
        "c.txt",
        "d.txt",
        "a.txt",
    ]
    assert _ids(sort_sources(TIED, "year", "desc")) == [
        "c.txt",
        "d.txt",
        "b.txt",
        "e.txt",
        "a.txt",
    ]


def test_titles_sort_case_insensitively_with_nulls_last() -> None:
    assert _ids(sort_sources(TIED, "title", "asc")) == [
        "b.txt",
        "c.txt",
        "d.txt",
        "a.txt",
        "e.txt",
    ]
    assert _ids(sort_sources(TIED, "title", "desc")) == [
        "a.txt",
        "d.txt",
        "b.txt",
        "c.txt",
        "e.txt",
    ]


def test_paths_and_index_times_sort_as_text() -> None:
    stamped = [
        _source("b.txt", indexed_at="2026-01-02T00:00:00+00:00"),
        _source("a.txt", indexed_at="2026-01-02T00:00:00+00:00"),
        _source("c.txt", indexed_at="2025-12-31T00:00:00+00:00"),
        _source("B.txt"),
    ]
    assert _ids(sort_sources(stamped, "path", "asc")) == [
        "B.txt",
        "a.txt",
        "b.txt",
        "c.txt",
    ]
    assert _ids(sort_sources(stamped, "path", "desc")) == [
        "c.txt",
        "b.txt",
        "a.txt",
        "B.txt",
    ]
    assert _ids(sort_sources(stamped, "indexed_at", "desc")) == [
        "a.txt",
        "b.txt",
        "c.txt",
        "B.txt",
    ]


def test_passage_counts_sort_with_failed_counts_last() -> None:
    counts: dict[str, int | None] = {
        "a.txt": 3,
        "b.txt": None,
        "c.txt": 3,
        "d.txt": 0,
        "e.txt": 9,
    }
    assert _ids(sort_sources(TIED, "passage_count", "asc", counts)) == [
        "d.txt",
        "a.txt",
        "c.txt",
        "e.txt",
        "b.txt",
    ]
    assert _ids(sort_sources(TIED, "passage_count", "desc", counts)) == [
        "e.txt",
        "a.txt",
        "c.txt",
        "d.txt",
        "b.txt",
    ]
