"""The limits of passage context (UC-12)."""

import pytest
from pydantic import ValidationError

from docgrep.api.mcp import effective_max_context_passages
from docgrep.settings import MCP_MAX_CONTEXT_PASSAGES, Settings


def test_the_limits_default_to_20_ids_and_500_of_at_most_5000_tokens() -> None:
    settings = Settings(_env_file=None)

    assert settings.context_max_passages == 20
    assert settings.context_window_tokens == 500
    assert settings.context_max_window_tokens == 5000
    assert settings.mcp_max_context_passages == MCP_MAX_CONTEXT_PASSAGES == 5


def test_the_limits_are_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DOCGREP_CONTEXT_MAX_PASSAGES", "7")
    monkeypatch.setenv("DOCGREP_CONTEXT_WINDOW_TOKENS", "0")
    monkeypatch.setenv("DOCGREP_CONTEXT_MAX_WINDOW_TOKENS", "100")
    monkeypatch.setenv("DOCGREP_MCP_MAX_CONTEXT_PASSAGES", "3")

    settings = Settings()

    assert (
        settings.context_max_passages,
        settings.context_window_tokens,
        settings.context_max_window_tokens,
        settings.mcp_max_context_passages,
    ) == (7, 0, 100, 3)


@pytest.mark.parametrize(
    "values",
    [
        {"context_max_passages": 0},
        {"context_window_tokens": -1},
        {"context_max_window_tokens": 0},
        {"mcp_max_context_passages": 0},
        {"context_window_tokens": 501, "context_max_window_tokens": 500},
    ],
)
def test_invalid_limits_are_refused(values: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        Settings(**values)  # type: ignore[arg-type]  # limits passed by name


def test_the_mcp_cap_never_exceeds_the_http_maximum() -> None:
    assert effective_max_context_passages(Settings()) == 5
    assert effective_max_context_passages(Settings(context_max_passages=2)) == 2
    assert effective_max_context_passages(Settings(mcp_max_context_passages=40)) == 20
