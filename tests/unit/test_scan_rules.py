"""Sampling and classification rules of scan discovery (UC-5)."""

import pytest

from docgrep.domain.scans import (
    GARBLED_TEXT_LAYER,
    LOW_TEXT_DENSITY,
    MIXED_OR_UNCERTAIN,
    NATIVE,
    SCANNED,
    UNCERTAIN_TEXT_DENSITY,
    PageSample,
    classify,
    sample_pages,
    text_density,
)


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, ()),
        (1, (1,)),
        (2, (1, 2)),
        (3, (1, 2, 3)),
        (4, (1, 2, 3, 4)),
        (5, (1, 2, 3, 4, 5)),
    ],
)
def test_small_files_are_sampled_whole(count: int, expected: tuple[int, ...]) -> None:
    assert sample_pages("book.pdf", count) == expected


def test_large_files_sample_first_last_and_two_middle_pages() -> None:
    pages = sample_pages("books/long.pdf", 1000)

    assert len(pages) == 6 and pages == tuple(sorted(set(pages)))
    assert pages[:3] == (1, 2, 3) and pages[-1] == 1000
    assert all(3 < page < 1000 for page in pages[3:5])


def test_the_sample_is_stable_and_depends_on_path_and_page_count() -> None:
    first = sample_pages("books/long.pdf", 1000)

    assert sample_pages("books/long.pdf", 1000) == first
    others = {sample_pages(f"books/other{i}.pdf", 1000)[3:5] for i in range(20)}
    assert len(others) > 10  # different files, different middle pages
    changed = {sample_pages("books/long.pdf", count)[3:5] for count in range(990, 1000)}
    assert len(changed) > 5  # the page count is part of the seed


def test_the_sample_does_not_use_python_hash() -> None:
    # The value is fixed by SHA-256 of the path and page count.
    assert sample_pages("a.pdf", 100) == sample_pages("a.pdf", 100)
    assert sample_pages("a.pdf", 100) == (1, 2, 3, 17, 43, 100)


def _pages(*counts: int, dropped: tuple[int, ...] = ()) -> list[PageSample]:
    return [
        PageSample(index + 1, count, index + 1 in dropped)
        for index, count in enumerate(counts)
    ]


def test_density_counts_dropped_pages_as_zero() -> None:
    assert text_density(_pages(1000, 1000, 1000, 1000, dropped=(1, 2))) == 500.0
    assert text_density([]) == 0.0


@pytest.mark.parametrize(
    ("density", "expected"),
    [
        (99.9, (SCANNED, LOW_TEXT_DENSITY)),
        (100, (MIXED_OR_UNCERTAIN, UNCERTAIN_TEXT_DENSITY)),
        (300, (MIXED_OR_UNCERTAIN, UNCERTAIN_TEXT_DENSITY)),
        (500, (MIXED_OR_UNCERTAIN, UNCERTAIN_TEXT_DENSITY)),
        (500.1, (NATIVE, None)),
    ],
)
def test_the_threshold_boundaries(
    density: float, expected: tuple[str, str | None]
) -> None:
    # Ten pages, one carrying all the characters: the mean is the density.
    samples = [PageSample(1, round(density * 10), False)] + [
        PageSample(page, 0, False) for page in range(2, 11)
    ]
    assert text_density(samples) == pytest.approx(density)

    assert classify(samples, lower=100, upper=500) == expected


def test_blank_pages_do_not_tip_a_text_heavy_file() -> None:
    samples = _pages(1800, 0, 1700, 0, 1900, 1600)

    assert classify(samples, lower=100, upper=500) == (NATIVE, None)


def test_mostly_broken_pages_make_a_garbled_scan() -> None:
    samples = _pages(2000, 2000, 2000, 2000, 2000, 2000, dropped=(1, 2, 3))

    assert classify(samples, lower=100, upper=500) == (SCANNED, GARBLED_TEXT_LAYER)
    fewer = _pages(2000, 2000, 2000, 2000, 2000, 2000, dropped=(1, 2))
    assert classify(fewer, lower=100, upper=500) == (NATIVE, None)


def test_a_title_page_does_not_make_a_scan_native() -> None:
    samples = _pages(400, 0, 0, 0, 12, 0)

    assert classify(samples, lower=100, upper=500) == (SCANNED, LOW_TEXT_DENSITY)
