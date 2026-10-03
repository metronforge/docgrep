"""Process-wide logging setup for the console and HTTP transports."""

import ipaddress
import json
import logging
import sys
import time
from typing import IO

_PACKAGE_LOGGER = "docgrep"
_STANDARD_FIELDS = frozenset(
    {*logging.makeLogRecord({}).__dict__, "message", "asctime", "taskName"}
)


class KeyValueFormatter(logging.Formatter):
    """Render a record as one line with its extra fields as key=value pairs."""

    converter = staticmethod(time.gmtime)

    def format(self, record: logging.LogRecord) -> str:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", self.converter(record.created))
        line = f"{stamp} {record.levelname} {record.name} {record.getMessage()}"
        for key in sorted(set(record.__dict__) - _STANDARD_FIELDS):
            line += f" {key}={_render(record.__dict__[key])}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def _render(value: object) -> str:
    text = str(value)
    if text and not any(char.isspace() or char in '"=' for char in text):
        return text
    return json.dumps(text, ensure_ascii=False)


class _StderrHandler(logging.StreamHandler[IO[str]]):
    """Write to whatever sys.stderr is when a record is emitted."""

    def __init__(self) -> None:
        super().__init__(sys.stderr)

    @property
    def stream(self) -> IO[str]:
        return sys.stderr

    @stream.setter
    def stream(self, _value: IO[str]) -> None:
        return None


def configure_logging(level: str) -> None:
    """Install one stderr handler on the package logger; safe to call repeatedly."""
    logger = logging.getLogger(_PACKAGE_LOGGER)
    for handler in list(logger.handlers):
        if isinstance(handler, _StderrHandler):
            logger.removeHandler(handler)
    handler = _StderrHandler()
    handler.setFormatter(KeyValueFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)


def is_loopback_host(host: str) -> bool:
    """Return whether a bind host only accepts connections from this machine."""
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def warn_if_not_loopback(host: str) -> None:
    """Log a warning when the server is configured to listen beyond loopback."""
    if not is_loopback_host(host):
        logging.getLogger(_PACKAGE_LOGGER).warning(
            "server_host_not_loopback",
            extra={"host": host},
        )
