"""The pure rules of an update plan: move pairs and the removal guard
(ADR-0027), and the guard settings."""

import pytest
from pydantic import ValidationError as SettingsValidationError

from docgrep.domain.models import FileType, Source
from docgrep.domain.update_plan import (
    GUARD_SMALL_ROOT,
    GuardedRoot,
    Move,
    RemovalGuard,
    guarded_roots,
    pair_moves,
)
from docgrep.settings import Settings


def _source(path: str, content: str | None, size: int | None = 10) -> Source:
    return Source(
        source_id=path,
        path=path,
        file_type=FileType.TXT,
        content_hash=content,
        size=size,
    )


def test_a_removed_source_pairs_with_an_added_file_of_the_same_content() -> None:
    removed = [_source("old/a.txt", "h1"), _source("old/b.txt", "h2")]
    added = [_source("new/b.txt", "h2"), _source("new/c.txt", "h3")]

    assert pair_moves(removed, added) == (Move("old/b.txt", "new/b.txt"),)


def test_pairs_are_one_to_one_in_stored_path_order() -> None:
    removed = [_source("z.txt", "same"), _source("a.txt", "same")]
    added = [
        _source("y.txt", "same"),
        _source("b.txt", "same"),
        _source("c.txt", "same"),
    ]

    assert pair_moves(removed, added) == (
        Move("a.txt", "b.txt"),
        Move("z.txt", "c.txt"),
    )
    # The input order does not matter.
    assert pair_moves(reversed(removed), reversed(added)) == pair_moves(removed, added)


def test_identical_content_in_two_new_places_pairs_only_once() -> None:
    removed = [_source("old.txt", "h")]
    added = [_source("copy2.txt", "h"), _source("copy1.txt", "h")]

    assert pair_moves(removed, added) == (Move("old.txt", "copy1.txt"),)


def test_the_size_must_match_and_a_missing_hash_never_pairs() -> None:
    assert pair_moves([_source("a", "h", 10)], [_source("b", "h", 11)]) == ()
    assert pair_moves([_source("a", None)], [_source("b", None)]) == ()
    assert pair_moves([_source("a", "h", None)], [_source("b", "h", None)]) == ()


def test_the_threshold_is_the_smaller_of_the_ratio_and_the_count() -> None:
    guard = RemovalGuard(ratio=0.05, count=50)

    assert guard.threshold(100) == 5
    assert guard.threshold(60) == 3
    assert guard.threshold(10_000) == 50
    # A small root is guarded by the count alone.
    assert guard.threshold(GUARD_SMALL_ROOT - 1) == 50
    assert guard.threshold(GUARD_SMALL_ROOT) == 1
    # Rounded down: 1.6 allows one removal.
    assert guard.threshold(32) == 1


def test_only_roots_above_the_threshold_are_guarded() -> None:
    guard = RemovalGuard(ratio=0.05, count=50)

    found = guarded_roots(
        {"books": 6, "notes": 5, "small": 18, "empty": 0},
        {"books": 100, "notes": 100, "small": 19},
        guard,
    )

    assert found == (GuardedRoot("books", 6, 100, 5),)


def test_a_small_root_beyond_the_count_is_guarded() -> None:
    guard = RemovalGuard(ratio=0.05, count=3)

    assert guarded_roots({"small": 4}, {"small": 5}, guard) == (
        GuardedRoot("small", 4, 5, 3, "threshold"),
    )
    assert guarded_roots({"small": 3}, {"small": 5}, guard) == ()


@pytest.mark.parametrize("recorded", [1, 2, 19, 20, 5000])
def test_a_root_that_would_be_emptied_is_always_guarded(recorded: int) -> None:
    # Even when the thresholds would allow every removal.
    guard = RemovalGuard(ratio=1, count=10_000)

    assert guarded_roots({"r": recorded}, {"r": recorded}, guard) == (
        GuardedRoot("r", recorded, recorded, guard.threshold(recorded), "empties_root"),
    )
    assert guarded_roots({"r": recorded - 1}, {"r": recorded}, guard) == ()


def test_emptying_a_root_is_the_reason_even_above_the_threshold() -> None:
    found = guarded_roots({"a": 40, "b": 3}, {"a": 40, "b": 40}, RemovalGuard())

    assert [(g.root, g.reason) for g in found] == [
        ("a", "empties_root"),
        ("b", "threshold"),
    ]


def test_a_root_without_removals_or_sources_is_not_guarded() -> None:
    assert guarded_roots({"new": 0}, {}, RemovalGuard(ratio=0, count=0)) == ()


def test_the_count_caps_the_ratio_for_a_large_root() -> None:
    guard = RemovalGuard()

    assert guarded_roots({"lib": 50}, {"lib": 5000}, guard) == ()
    assert guarded_roots({"lib": 51}, {"lib": 5000}, guard) == (
        GuardedRoot("lib", 51, 5000, 50),
    )


@pytest.mark.parametrize(("ratio", "count"), [(-0.1, 1), (1.5, 1), (0.1, -1)])
def test_the_guard_refuses_invalid_limits(ratio: float, count: int) -> None:
    with pytest.raises(ValueError):
        RemovalGuard(ratio=ratio, count=count)


def test_the_guard_settings_have_defaults_and_are_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert Settings().removal_guard == RemovalGuard(0.05, 50)

    monkeypatch.setenv("DOCGREP_UPDATE_GUARD_RATIO", "0.2")
    monkeypatch.setenv("DOCGREP_UPDATE_GUARD_COUNT", "7")
    assert Settings().removal_guard == RemovalGuard(0.2, 7)

    for name, value in (
        ("DOCGREP_UPDATE_GUARD_RATIO", "1.5"),
        ("DOCGREP_UPDATE_GUARD_RATIO", "-0.1"),
        ("DOCGREP_UPDATE_GUARD_COUNT", "-1"),
    ):
        with monkeypatch.context() as patch:
            patch.setenv(name, value)
            with pytest.raises(SettingsValidationError):
                Settings()


def test_paired_removals_are_reported_and_never_counted() -> None:
    guard = RemovalGuard(ratio=0.05, count=3)

    # Only the unpaired part is compared with the threshold.
    assert guarded_roots({"r": 3}, {"r": 100}, guard, {"r": 40}) == ()
    assert guarded_roots({"r": 4}, {"r": 100}, guard, {"r": 40}) == (
        GuardedRoot("r", 4, 100, 3, "threshold", 40),
    )


def test_a_root_whose_sources_all_moved_is_not_emptied() -> None:
    guard = RemovalGuard(ratio=1, count=10_000)

    assert guarded_roots({"r": 0}, {"r": 5}, guard, {"r": 5}) == ()
    assert guarded_roots({"r": 5}, {"r": 5}, guard, {"r": 0}) == (
        GuardedRoot("r", 5, 5, 10_000, "empties_root"),
    )
