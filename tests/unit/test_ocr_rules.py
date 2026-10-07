"""Pure OCR rules and the OCR settings."""

import pytest
from pydantic import ValidationError as SettingsValidationError

from docgrep.domain.models import FileType
from docgrep.domain.ocr import (
    eligible_type,
    eta_seconds,
    missing_languages,
    ocr_mode,
    output_path,
    parse_languages,
)
from docgrep.settings import Settings


def test_outputs_go_next_to_the_original_as_a_pdf() -> None:
    assert output_path("books/scan.pdf") == "books/scan.ocr.pdf"
    assert output_path("books/Book.DJVU") == "books/Book.ocr.pdf"
    assert output_path("a.b.djv") == "a.b.ocr.pdf"


def test_only_pdf_and_djvu_are_eligible() -> None:
    assert eligible_type("x/A.PDF") is FileType.PDF
    assert eligible_type("x/a.djv") is FileType.DJVU
    assert eligible_type("x/a.md") is None and eligible_type("x/pdf") is None


def test_a_garbled_layer_or_an_override_forces_ocr() -> None:
    assert ocr_mode("garbled_text_layer", override_native=False) == "force-ocr"
    assert ocr_mode(None, override_native=True) == "force-ocr"
    assert ocr_mode("low_text_density", override_native=False) == "skip-text"
    assert ocr_mode("uncertain_text_density", override_native=False) == "skip-text"


def test_languages_are_validated_and_compared() -> None:
    assert parse_languages("rus+eng+ukr") == ("rus", "eng", "ukr")
    assert parse_languages("chi_sim") == ("chi_sim",)
    for bad in ("", "rus+", "rus eng", "Rus", "rus,eng"):
        with pytest.raises(ValueError):
            parse_languages(bad)
    assert missing_languages(("rus", "deu", "eng"), ("eng", "rus")) == ("deu",)


def test_the_eta_follows_the_bytes_done() -> None:
    assert eta_seconds(done_bytes=0, total_bytes=10, elapsed_seconds=5) is None
    assert eta_seconds(done_bytes=25, total_bytes=100, elapsed_seconds=10) == 30


def test_ocr_settings_defaults_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings()
    assert settings.ocr_languages == "rus+eng+ukr"
    assert settings.ocr_timeout_seconds == 0
    assert settings.ocr_jobs is None
    assert settings.ocr_cancel_mode == "terminate"

    monkeypatch.setenv("DOCGREP_OCR_JOBS", "2")
    monkeypatch.setenv("DOCGREP_OCR_CANCEL_MODE", "finish")
    monkeypatch.setenv("DOCGREP_OCR_TIMEOUT_SECONDS", "1800")
    changed = Settings()
    assert (changed.ocr_jobs, changed.ocr_cancel_mode) == (2, "finish")
    assert changed.ocr_timeout_seconds == 1800

    for name, value in (
        ("DOCGREP_OCR_JOBS", "0"),
        ("DOCGREP_OCR_CANCEL_MODE", "now"),
        ("DOCGREP_OCR_TIMEOUT_SECONDS", "-1"),
        ("DOCGREP_OCR_LANGUAGES", "rus,eng"),
    ):
        with monkeypatch.context() as patched:
            patched.setenv(name, value)
            with pytest.raises(SettingsValidationError):
                Settings()
