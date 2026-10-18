"""The progress line: formatting, shortening, throttling, final line, clearing."""

import io
import logging
from dataclasses import replace

import pytest

from docgrep.api.progress_line import (
    ProgressLine,
    format_eta,
    format_line,
    shorten_middle,
)
from docgrep.domain.progress import ProgressEvent


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _event(**changes: object) -> ProgressEvent:
    base = ProgressEvent(
        operation="index",
        unit="sources",
        done=612,
        total=1650,
        passages=210_400,
        rate=95.0,
        eta_seconds=4320.0,
        label="Rust.Atomics.and.Locks.pdf",
    )
    return replace(base, **changes)


def test_the_line_shows_percent_counts_passages_rate_eta_and_name() -> None:
    assert format_line(_event(), 200) == (
        "Indexing  37%  612/1,650 sources  210,400 passages  95 p/s  ETA 1h 12m  "
        "Rust.Atomics.and.Locks.pdf"
    )


def test_an_unknown_total_shows_counts_only() -> None:
    line = format_line(_event(total=None, eta_seconds=None, label=None), 200)

    assert line == "Indexing  612 sources  210,400 passages  95 p/s"


def test_percent_never_exceeds_one_hundred_and_rounds_down() -> None:
    assert "99%" in format_line(_event(done=1649, label=None), 200)
    assert "100%" in format_line(_event(done=1650, label=None), 200)


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (0, "0s"),
        (42.4, "42s"),
        (59.6, "1m 00s"),
        (303, "5m 03s"),
        (3599, "59m 59s"),
        (4320, "1h 12m"),
        (90_000, "25h 00m"),
    ],
)
def test_eta_formatting(seconds: float, text: str) -> None:
    assert format_eta(seconds) == text


def test_a_long_label_is_shortened_in_the_middle_to_the_room_left() -> None:
    event = _event(label="A.Very.Long.Book.Title.About.Rust.Atomics.and.Locks.pdf")

    line = format_line(event, 100)

    assert len(line) <= 100
    assert line.startswith("Indexing  37%")
    assert "…" in line
    assert line.endswith("Locks.pdf")


def test_shorten_middle_keeps_both_ends() -> None:
    assert shorten_middle("abcdefghij", 10) == "abcdefghij"
    assert shorten_middle("abcdefghij", 7) == "abc…hij"
    assert shorten_middle("abcdefghij", 6) == "abc…ij"
    assert shorten_middle("abcdefghij", 1) == "…"
    assert shorten_middle("abcdefghij", 0) == ""


def test_the_fixed_part_is_cut_when_even_it_does_not_fit() -> None:
    line = format_line(_event(), 30)

    assert len(line) == 30
    assert line.startswith("Indexing  37%  612/1,650")
    assert line.endswith("…")


def test_only_the_last_component_of_a_name_is_expected_from_the_event() -> None:
    # The use cases pass names, never paths; the line shows what it is given.
    assert "books/" not in format_line(_event(label="Atomics.pdf"), 200)


def test_finished_lines_carry_no_eta_and_failures_show() -> None:
    line = format_line(_event(done=1650, failed=3, finished=True, label=None), 200)

    assert "ETA" not in line
    assert "3 failed" in line


def test_a_scan_phase_and_a_writing_phase_have_their_own_verbs() -> None:
    scan = _event(operation="reencode", unit="passages", phase="scanning")
    write = _event(operation="reencode", unit="passages", passages=None)

    assert format_line(scan, 200).startswith("Scanning")
    assert format_line(write, 200).startswith("Re-encoding")
    assert format_line(write, 200).count("passages") == 1


def test_ocr_counts_pages_without_passages_or_rate() -> None:
    event = _event(
        operation="ocr", unit="pages", passages=None, rate=None, done=40, total=100
    )

    assert format_line(event, 200).startswith("OCR  40%  40/100 pages  ETA 1h 12m")


def _line(
    clock: Clock, *, width: int = 60, stream: io.StringIO | None = None
) -> tuple[ProgressLine, io.StringIO]:
    out = stream if stream is not None else io.StringIO()
    return ProgressLine(out, width=lambda: width, clock=clock), out


def test_redraws_happen_in_place_and_fill_the_width() -> None:
    clock = Clock()
    line, out = _line(clock)

    line(_event(label=None))

    drawn = out.getvalue()
    assert drawn.startswith("\r")
    assert "\n" not in drawn
    assert len(drawn) == 1 + 59


def test_redraws_are_throttled_to_ten_a_second() -> None:
    clock = Clock()
    line, out = _line(clock)

    line(_event(done=1))
    clock.now += 0.05
    line(_event(done=2))
    clock.now += 0.04
    line(_event(done=3))
    clock.now += 0.02
    line(_event(done=4))

    assert out.getvalue().count("\r") == 2
    assert "2/" not in out.getvalue() and "3/" not in out.getvalue()


def test_the_finished_event_is_always_drawn_and_stays_on_its_own_line() -> None:
    clock = Clock()
    line, out = _line(clock)

    line(_event(done=1))
    line(_event(done=1650, finished=True, label=None))

    assert out.getvalue().endswith("\n")
    assert "1,650/1,650" in out.getvalue()
    out.truncate(0)
    out.seek(0)
    line.close()
    assert out.getvalue() == ""


def test_clear_wipes_the_line_and_the_next_event_draws_at_once() -> None:
    clock = Clock()
    line, out = _line(clock)
    line(_event(done=1))
    out.truncate(0)
    out.seek(0)

    line.clear()

    assert out.getvalue() == "\r" + " " * 59 + "\r"
    out.truncate(0)
    out.seek(0)
    line(_event(done=2))
    assert "2/1,650" in out.getvalue()


def test_clear_writes_nothing_when_no_line_is_on_screen() -> None:
    line, out = _line(Clock())

    line.clear()
    line.close()

    assert out.getvalue() == ""


def test_close_wipes_a_line_that_was_never_finished() -> None:
    line, out = _line(Clock())
    line(_event())
    out.truncate(0)
    out.seek(0)

    line.close()

    assert out.getvalue() == "\r" + " " * 59 + "\r"


def test_a_log_record_clears_the_line_first() -> None:
    clock = Clock()
    line, out = _line(clock)
    logger = logging.getLogger("docgrep")
    handler = logging.StreamHandler(out)
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(logging.INFO)
    try:
        line(_event())
        with line.keep_logs_apart():
            logging.getLogger("docgrep.test").info("a record")
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)

    text = out.getvalue()
    wiped = "\r" + " " * 59 + "\r"
    assert text.index(wiped) < text.index("a record")
