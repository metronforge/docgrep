from pathlib import Path

import pytest
from pydantic import ValidationError

from docgrep.settings import Settings


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DOCGREP_HOST", raising=False)
    monkeypatch.delenv("DOCGREP_TOP_N", raising=False)

    settings = Settings(_env_file=None)

    assert settings.corpus_root == Path(".")
    assert settings.data_dir == Path(".docgrep")
    assert settings.top_k == 100
    assert settings.top_n == 10
    assert settings.top_n_max == 50
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert settings.device == "auto"
    assert settings.backend == "real"
    assert settings.host == "127.0.0.1"


def test_settings_load_prefixed_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCGREP_CORPUS_ROOT", "/tmp/corpus")
    monkeypatch.setenv("DOCGREP_INCLUDE_GLOBS", '["*.txt", "**/*.md"]')
    monkeypatch.setenv("DOCGREP_EXCLUDE_GLOBS", '["**/private/**"]')
    monkeypatch.setenv("DOCGREP_TOP_N", "20")
    monkeypatch.setenv("DOCGREP_BACKEND", "fake")

    settings = Settings(_env_file=None)

    assert settings.corpus_root == Path("/tmp/corpus")
    assert settings.include_globs == ("*.txt", "**/*.md")
    assert settings.exclude_globs == ("**/private/**",)
    assert settings.top_n == 20
    assert settings.backend == "fake"


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"chunk_size": 20, "chunk_overlap": 20}, "chunk_overlap"),
        ({"top_n": 51}, "top_n"),
        ({"top_k": 5}, "top_k"),
    ],
)
def test_settings_reject_inconsistent_limits(
    overrides: dict[str, int], field: str
) -> None:
    with pytest.raises(ValidationError, match=field):
        Settings(_env_file=None, **overrides)


def test_allowed_hosts_are_read_as_a_comma_separated_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DOCGREP_ALLOWED_HOSTS", raising=False)
    assert Settings().allowed_hosts == ()

    monkeypatch.setenv("DOCGREP_ALLOWED_HOSTS", " docgrep.lan, box.internal ,,")
    assert Settings().allowed_hosts == ("docgrep.lan", "box.internal")
