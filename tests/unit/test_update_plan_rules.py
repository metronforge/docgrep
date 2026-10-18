"""The removal guard (ADR-0027) and its settings."""

import pytest
from pydantic import ValidationError as SettingsValidationError

from docgrep.domain.update_plan import (
    GUARD_SMALL_ROOT,
    GuardedRoot,
    RemovalGuard,
    guarded_roots,
)
from docgrep.settings import Settings


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
        {"books": 6, "notes": 5, "small": 19, "empty": 0},
        {"books": 100, "notes": 100, "small": 19},
        guard,
    )

    assert found == (GuardedRoot("books", 6, 100, 5),)


def test_a_small_root_emptied_beyond_the_count_is_guarded() -> None:
    guard = RemovalGuard(ratio=0.05, count=3)

    assert guarded_roots({"small": 4}, {"small": 4}, guard) == (
        GuardedRoot("small", 4, 4, 3),
    )
    assert guarded_roots({"small": 3}, {"small": 4}, guard) == ()


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
