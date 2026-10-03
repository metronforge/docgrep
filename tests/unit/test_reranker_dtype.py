"""Reranker precision: resolution matrix, fallback warning and settings."""

import logging
from collections.abc import Sequence

import pytest
from pydantic import ValidationError as SettingsValidationError

from docgrep.adapters.models.real import BgeReranker, resolve_reranker_dtype
from docgrep.settings import Settings


class _Cross:
    def predict(
        self, pairs: Sequence[tuple[str, str]], *, show_progress_bar: bool
    ) -> list[float]:
        return [1.0 for _ in pairs]


class _Factory:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, name: str, device: str, dtype: str) -> _Cross:
        self.calls.append((name, device, dtype))
        return _Cross()


MATRIX = [
    # (setting, cuda available, expected dtype, warns)
    ("auto", True, "float16", False),
    ("auto", False, "float32", False),
    ("float16", True, "float16", False),
    ("float16", False, "float32", True),
    ("float32", True, "float32", False),
    ("float32", False, "float32", False),
]


@pytest.mark.parametrize(("setting", "cuda", "expected", "warns"), MATRIX)
def test_dtype_resolution_through_the_injected_probe(
    setting: str,
    cuda: bool,
    expected: str,
    warns: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    factory = _Factory()
    reranker = BgeReranker(
        "r", "auto", cuda_available=lambda: cuda, dtype=setting, factory=factory
    )

    with caplog.at_level(logging.INFO, logger="docgrep"):
        assert reranker.score("q", ["a", "b"]) == (1.0, 1.0)
        reranker.score("q", ["c"])

    assert factory.calls == [("r", "cuda" if cuda else "cpu", expected)]
    fallbacks = [
        r for r in caplog.records if r.getMessage() == "reranker_dtype_fallback"
    ]
    assert bool(fallbacks) is warns
    if warns:
        [record] = fallbacks
        assert record.levelno == logging.WARNING
        assert record.__dict__["requested"] == "float16"
        assert record.__dict__["dtype"] == "float32"
    [loaded] = [r for r in caplog.records if r.getMessage() == "model_loaded"]
    assert loaded.__dict__["dtype"] == expected
    assert loaded.__dict__["device"] == ("cuda" if cuda else "cpu")


@pytest.mark.parametrize(
    ("setting", "expected"), [("auto", "float32"), ("float16", "float32")]
)
def test_an_explicit_cpu_device_never_gets_float16(
    setting: str, expected: str, caplog: pytest.LogCaptureFixture
) -> None:
    factory = _Factory()
    reranker = BgeReranker(
        "r", "cpu", cuda_available=lambda: True, dtype=setting, factory=factory
    )

    with caplog.at_level(logging.WARNING, logger="docgrep"):
        reranker.score("q", ["a"])

    assert factory.calls == [("r", "cpu", expected)]
    assert any(r.getMessage() == "reranker_dtype_fallback" for r in caplog.records) is (
        setting == "float16"
    )


@pytest.mark.parametrize(
    ("setting", "device", "expected"),
    [
        ("auto", "cuda", "float16"),
        ("auto", "cuda:1", "float16"),
        ("auto", "cpu", "float32"),
        ("float16", "cuda:0", "float16"),
        ("float16", "cpu", "float32"),
        ("float32", "cuda", "float32"),
    ],
)
def test_resolution_function(setting: str, device: str, expected: str) -> None:
    dtype, _ = resolve_reranker_dtype(setting, device)
    assert dtype == expected


def test_the_embedder_log_record_has_no_dtype(caplog: pytest.LogCaptureFixture) -> None:
    from docgrep.adapters.models.real import BgeEmbedder

    class Tok:
        def __call__(self, text: str, **_: object) -> dict[str, list[tuple[int, int]]]:
            return {"offset_mapping": []}

    class Sentence:
        tokenizer = Tok()

        def get_sentence_embedding_dimension(self) -> int:
            return 2

    with caplog.at_level(logging.INFO, logger="docgrep"):
        _ = BgeEmbedder("e", "cpu", lambda *_: Sentence()).dimension

    [loaded] = [r for r in caplog.records if r.getMessage() == "model_loaded"]
    assert "dtype" not in loaded.__dict__


def test_reranker_dtype_setting_defaults_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DOCGREP_RERANKER_DTYPE", raising=False)
    assert Settings().reranker_dtype == "auto"

    monkeypatch.setenv("DOCGREP_RERANKER_DTYPE", "float16")
    assert Settings().reranker_dtype == "float16"


@pytest.mark.parametrize("value", ["fp16", "float64", "bfloat16", ""])
def test_invalid_dtype_is_a_settings_error(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCGREP_RERANKER_DTYPE", value)

    with pytest.raises(SettingsValidationError):
        Settings()
