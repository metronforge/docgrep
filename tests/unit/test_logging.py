"""Logging setup: formatter, handler installation and the exposure warning."""

import logging
import sys

import pytest
from fastapi.testclient import TestClient

from docgrep.api.http import create_app
from docgrep.api.log_config import (
    KeyValueFormatter,
    configure_logging,
    is_loopback_host,
    warn_if_not_loopback,
)
from docgrep.settings import Settings


def _record(message: str, **extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        "docgrep.test", logging.WARNING, __file__, 1, message, None, None
    )
    record.created = 0
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_formatter_prints_extra_fields_as_sorted_key_value_pairs() -> None:
    line = KeyValueFormatter().format(
        _record("index_source_failed", path="notes/a b.md", reason="parse_error", n=3)
    )

    assert line == (
        "1970-01-01T00:00:00Z WARNING docgrep.test index_source_failed "
        'n=3 path="notes/a b.md" reason=parse_error'
    )


def test_formatter_quotes_empty_and_ambiguous_values_and_keeps_plain_records() -> None:
    formatter = KeyValueFormatter()

    assert formatter.format(_record("plain")) == (
        "1970-01-01T00:00:00Z WARNING docgrep.test plain"
    )
    assert formatter.format(_record("m", a="", b="x=y", c='say "hi"')) == (
        '1970-01-01T00:00:00Z WARNING docgrep.test m a="" b="x=y" c="say \\"hi\\""'
    )


def test_formatter_appends_the_traceback() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        record = _record("failed")
        record.exc_info = sys.exc_info()

    text = KeyValueFormatter().format(record)

    assert text.splitlines()[0].endswith("failed")
    assert "ValueError: boom" in text


def test_configure_logging_installs_one_handler_and_sets_the_level(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO")
    configure_logging("DEBUG")
    package = logging.getLogger("docgrep")

    assert len(package.handlers) == 1
    assert package.level == logging.DEBUG
    logging.getLogger("docgrep.usecases.build_index").info(
        "index_started", extra={"sources": 4}
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.endswith(
        "INFO docgrep.usecases.build_index index_started sources=4\n"
    )

    configure_logging("WARNING")
    logging.getLogger("docgrep.usecases.build_index").info("hidden")
    assert capsys.readouterr().err == ""


def test_log_level_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DOCGREP_LOG_LEVEL", raising=False)
    assert Settings().log_level == "INFO"

    monkeypatch.setenv("DOCGREP_LOG_LEVEL", "debug")
    assert Settings().log_level == "DEBUG"


@pytest.mark.parametrize(
    ("host", "loopback"),
    [
        ("127.0.0.1", True),
        ("127.5.5.5", True),
        ("::1", True),
        ("[::1]", True),
        ("localhost", True),
        ("LOCALHOST", True),
        ("0.0.0.0", False),
        ("::", False),
        ("192.168.1.10", False),
        ("docgrep.lan", False),
    ],
)
def test_loopback_detection(host: str, loopback: bool) -> None:
    assert is_loopback_host(host) is loopback


def test_warning_is_logged_for_a_non_loopback_host(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="docgrep"):
        warn_if_not_loopback("127.0.0.1")
        assert caplog.records == []
        warn_if_not_loopback("0.0.0.0")

    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert record.getMessage() == "server_host_not_loopback"
    assert record.__dict__["host"] == "0.0.0.0"


@pytest.mark.parametrize(("host", "warned"), [("127.0.0.1", False), ("0.0.0.0", True)])
def test_server_start_warns_only_for_exposed_hosts(
    host: str, warned: bool, caplog: pytest.LogCaptureFixture
) -> None:
    settings = Settings(host=host, backend="fake")
    with caplog.at_level(logging.WARNING, logger="docgrep"):
        with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
            assert client.get("/health").status_code == 200

    messages = [record.getMessage() for record in caplog.records]
    assert ("server_host_not_loopback" in messages) is warned
