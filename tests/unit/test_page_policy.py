"""The page policy on the classes of pages seen in real books (UC-3, UC-5).

Only broken characters (R1) drop a page. Long tokens (R2) and few letters and
digits (R3) are weak signals; dot leaders are ignored by them.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import pytest

from docgrep.adapters.chunking import TokenChunker, WhitespaceTokenizer
from docgrep.adapters.parsers import PdfParser
from docgrep.adapters.parsers.text_quality import (
    MIN_DOT_LEADER_RUN,
    classify_page,
    remove_soft_hyphens,
)
from docgrep.domain.models import FileType, ParsedDocument, Source
from page_classes import (
    BROKEN,
    READABLE,
    class_names,
    control_characters,
    keys_and_hashes,
    regular_expressions,
    toc_glued,
    toc_spaced,
)
from pdf_support import garbled, pdf_bytes, prose, write_pdf

SHY = "\u00ad"


def _text(lines: list[str]) -> str:
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("name", sorted(READABLE))
def test_readable_classes_are_never_dropped(name: str) -> None:
    signals = classify_page(_text(READABLE[name]()))

    assert signals.non_whitespace >= 200
    assert not signals.broken and not signals.garbled


@pytest.mark.parametrize("make", [toc_glued, toc_spaced])
def test_tables_of_contents_fire_no_rule_at_all(
    make: Callable[[], list[str]],
) -> None:
    signals = classify_page(_text(make()))

    assert not signals.any_rule
    assert signals.long_token_share < 0.1 and signals.alnum_share > 0.9


def test_dot_leaders_only_change_the_metrics_not_the_text() -> None:
    page = _text(toc_glued())

    assert "." * MIN_DOT_LEADER_RUN in page
    assert remove_soft_hyphens(page) == (page, 0)
    three_dots = _text(["Title" + "..." + "12" + "x" * 40] * 6)
    assert classify_page(three_dots).long_tokens  # three dots are no leader


def test_the_weak_classes_fire_their_weak_rule_only() -> None:
    assert classify_page(_text(class_names())).long_tokens
    assert classify_page(_text(keys_and_hashes())).long_tokens
    assert classify_page(_text(regular_expressions())).low_alnum
    for make in (class_names, keys_and_hashes, regular_expressions):
        signals = classify_page(_text(make()))
        assert signals.weak and not signals.broken


def test_keys_reach_135_characters() -> None:
    longest = max(len(token) for line in keys_and_hashes() for token in line.split())

    assert longest == 135


@pytest.mark.parametrize("name", sorted(BROKEN))
def test_broken_classes_are_dropped(name: str) -> None:
    signals = classify_page(_text(BROKEN[name]()))

    assert signals.broken and signals.garbled


def test_the_classification_is_deterministic() -> None:
    pages = [_text(make()) for make in (*READABLE.values(), *BROKEN.values())]

    assert [classify_page(p) for p in pages] == [classify_page(p) for p in pages]


def _parse(root: Path, pages: list[list[str] | None]) -> ParsedDocument:
    write_pdf(root / "doc.pdf", pdf_bytes(pages))
    return PdfParser(root).parse(Source("doc.pdf", "doc.pdf", FileType.PDF))


def test_readable_classes_in_a_pdf_keep_every_page(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pages: list[list[str] | None] = [
        prose(1),
        toc_glued(),
        toc_spaced(),
        class_names(),
        keys_and_hashes(),
        regular_expressions(),
        prose(2),
    ]

    with caplog.at_level(logging.WARNING):
        document = _parse(tmp_path, pages)

    assert [s.page_number for s in document.sections] == list(range(1, 8))
    assert not document.requires_ocr
    quality = document.quality
    assert quality is not None
    assert (quality.pages_garbled, quality.dropped_pages) == (0, ())
    assert (quality.pages_rule_broken, quality.pages_rule_long_tokens) == (0, 2)
    assert quality.pages_rule_low_alnum == 1
    assert "pdf_pages_dropped" not in caplog.text


def test_a_broken_page_is_dropped_with_a_warning_and_counts_only(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pages: list[list[str] | None] = [prose(1), control_characters(2), class_names()]

    with caplog.at_level(logging.WARNING):
        document = _parse(tmp_path, pages)

    assert [s.page_number for s in document.sections] == [1, 3]
    quality = document.quality
    assert quality is not None
    assert (quality.pages_garbled, quality.dropped_pages) == (1, (2,))
    assert (quality.pages_rule_broken, quality.pages_rule_long_tokens) == (1, 2)
    (record,) = [r for r in caplog.records if r.getMessage() == "pdf_pages_dropped"]
    assert (record.__dict__["dropped"], record.__dict__["total"]) == (1, 3)
    assert "import" not in caplog.text and "marker" not in caplog.text


def test_a_document_of_broken_pages_requires_ocr(tmp_path: Path) -> None:
    pages: list[list[str] | None] = [control_characters(i) for i in range(9)]
    pages.append(prose(1))

    document = _parse(tmp_path, pages)

    assert document.requires_ocr and document.sections == ()
    assert document.quality is not None
    assert document.quality.skip_reason == "garbled_text_layer"
    assert document.quality.pages_garbled == 9


def test_mostly_weak_only_pages_skip_the_document(tmp_path: Path) -> None:
    pages: list[list[str] | None] = [garbled(i) for i in range(4)] + [prose(1)]

    document = _parse(tmp_path, pages)

    assert document.requires_ocr and document.sections == ()
    assert document.quality is not None
    assert document.quality.skip_reason == "garbled_text_layer"
    assert document.quality.pages_garbled == 0
    assert document.quality.pages_rule_long_tokens == 4


def test_a_minority_of_weak_only_pages_is_kept(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pages: list[list[str] | None] = [garbled(i) for i in range(3)]
    pages += [prose(i) for i in range(7)]

    with caplog.at_level(logging.WARNING):
        document = _parse(tmp_path, pages)

    assert not document.requires_ocr
    assert [s.page_number for s in document.sections] == list(range(1, 11))
    assert document.quality is not None and document.quality.pages_garbled == 0
    assert "pdf_pages_dropped" not in caplog.text


def test_the_same_file_gives_the_same_passages_and_signals(tmp_path: Path) -> None:
    pages: list[list[str] | None] = [
        prose(1, lines=12),
        control_characters(2),
        toc_glued(),
        [f"split hyphen{SHY}", "ation"] + prose(3),
    ]
    chunker = TokenChunker(WhitespaceTokenizer(), chunk_size=20, chunk_overlap=4)

    first = _parse(tmp_path, pages)
    second = _parse(tmp_path, pages)

    assert first.quality == second.quality
    assert chunker.chunk(first) == chunker.chunk(second)
    assert {s.page_number for s in first.sections} == {1, 3, 4}
    assert any("hyphenation" in s.text for s in first.sections)
    assert first.quality is not None and first.quality.soft_hyphens_removed == 1
