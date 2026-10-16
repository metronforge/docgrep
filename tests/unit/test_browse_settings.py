"""The MCP page size cap of source browsing (UC-08)."""

import pytest
from pydantic import ValidationError

from docgrep.domain.sources import MAX_PAGE_SIZE
from docgrep.settings import MCP_MAX_PAGE_SIZE, Settings


def test_the_cap_defaults_to_100() -> None:
    assert Settings(_env_file=None).mcp_max_page_size == MCP_MAX_PAGE_SIZE == 100


def test_the_cap_reaches_the_page_size_ceiling_and_never_more() -> None:
    assert Settings(mcp_max_page_size=MAX_PAGE_SIZE).mcp_max_page_size == 500
    for invalid in (0, MAX_PAGE_SIZE + 1):
        with pytest.raises(ValidationError):
            Settings(mcp_max_page_size=invalid)


def test_the_cap_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCGREP_MCP_MAX_PAGE_SIZE", "7")

    assert Settings().mcp_max_page_size == 7
