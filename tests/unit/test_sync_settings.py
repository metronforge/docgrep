"""The settings of automatic sync (ADR-0029)."""

import pytest
from pydantic import ValidationError

from docgrep.settings import Settings


def test_sync_is_off_with_the_documented_defaults() -> None:
    settings = Settings()

    assert settings.sync is False
    assert settings.sync_interval_seconds == 120
    assert settings.sync_settle_seconds == 60
    assert settings.sync_reconcile_hours == 24


def test_sync_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCGREP_SYNC", "true")
    monkeypatch.setenv("DOCGREP_SYNC_INTERVAL_SECONDS", "30")
    monkeypatch.setenv("DOCGREP_SYNC_SETTLE_SECONDS", "0")
    monkeypatch.setenv("DOCGREP_SYNC_RECONCILE_HOURS", "6")

    settings = Settings()

    assert settings.sync is True
    assert (settings.sync_interval_seconds, settings.sync_settle_seconds) == (30, 0)
    assert settings.sync_reconcile_hours == 6


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("DOCGREP_SYNC_INTERVAL_SECONDS", "0"),
        ("DOCGREP_SYNC_SETTLE_SECONDS", "-1"),
        ("DOCGREP_SYNC_RECONCILE_HOURS", "0"),
    ],
)
def test_invalid_sync_settings_are_refused(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        Settings()
