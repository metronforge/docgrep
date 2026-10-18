"""The progress observer: events on every advance, the log cadence untouched."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from docgrep.adapters.chunking import TokenChunker, WhitespaceTokenizer
from docgrep.adapters.filelock import FileIndexLock
from docgrep.adapters.models.deterministic import DeterministicEmbedder
from docgrep.adapters.models.sparse import SparseEncoderSet, TermFrequencyEncoder
from docgrep.domain.models import (
    FileType,
    ParsedDocument,
    ParsedSection,
    Passage,
    Source,
    make_passage_id,
)
from docgrep.domain.progress import ProgressEvent, ProgressOptions, item_label
from docgrep.usecases.build_index import BuildIndex, Progress
from docgrep.usecases.reencode_sparse import ReencodeSparse
from fakes import InMemoryManifest, InMemoryPassageStore


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Sources:
    def __init__(self, *sources: Source) -> None:
        self.sources = sources

    def enumerate(self) -> tuple[Source, ...]:
        return self.sources


class TickingParser:
    def __init__(self, clock: Clock) -> None:
        self.clock = clock

    def parse(self, source: Source) -> ParsedDocument:
        self.clock.now += 1.0
        return ParsedDocument(source, (ParsedSection("alpha beta gamma delta"),))


class Sparse:
    def encode(self, text: str) -> dict[int, float]:
        return {len(text): 1.0}


def _build(tmp_path: Path, count: int, *, every_sources: int = 3) -> BuildIndex:
    clock = Clock()
    return BuildIndex(
        lock=FileIndexLock(tmp_path / "index.lock"),
        enumerator=Sources(
            *(
                Source(f"d/doc{i:02d}.txt", f"d/doc{i:02d}.txt", FileType.TXT)
                for i in range(count)
            )
        ),
        parsers={FileType.TXT: TickingParser(clock)},
        chunker=TokenChunker(WhitespaceTokenizer(), 2, 0),
        embedder=DeterministicEmbedder(),
        sparse_encoder=Sparse(),
        passage_writer=InMemoryPassageStore(),
        manifest=InMemoryManifest(),
        config={"chunk_size": 2, "chunk_overlap": 0},
        docgrep_version="0.0.0",
        sleep=lambda seconds: None,
        clock=clock,
        progress_every_sources=every_sources,
    )


def _records(caplog: pytest.LogCaptureFixture) -> list[tuple[int, str, object]]:
    return [
        (r.levelno, r.getMessage(), r.__dict__.get("sources_done"))
        for r in caplog.records
        if r.getMessage() == "index_progress"
    ]


def test_the_observer_sees_every_source_and_the_end_of_the_phase(
    tmp_path: Path,
) -> None:
    events: list[ProgressEvent] = []

    _build(tmp_path, 4).run(progress=ProgressOptions(events.append))

    assert all(e.operation == "index" and e.unit == "sources" for e in events)
    assert all(e.total == 4 for e in events)
    # Start, then a begin and an advance for each source, then the end.
    assert [e.done for e in events] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert [e.label for e in events[1:3]] == ["doc00.txt", "doc00.txt"]
    assert [e.finished for e in events] == [False] * 9 + [True]
    assert events[-1].label is None
    assert events[-1].passages == 8
    assert events[-1].eta_seconds == 0.0


def test_the_log_records_are_the_same_with_and_without_an_observer(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="docgrep"):
        _build(tmp_path / "a", 7).run()
    plain = _records(caplog)
    caplog.clear()

    with caplog.at_level(logging.INFO, logger="docgrep"):
        _build(tmp_path / "b", 7).run(progress=ProgressOptions(lambda event: None))

    assert _records(caplog) == plain
    assert [done for _, _, done in plain] == [3, 6, 7]


def test_the_log_level_of_the_records_can_be_lowered(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG, logger="docgrep"):
        _build(tmp_path, 4).run(progress=ProgressOptions(log_level=logging.DEBUG))

    assert {level for level, _, _ in _records(caplog)} == {logging.DEBUG}


def test_labels_carry_the_last_path_component_only() -> None:
    assert item_label("books/rust/Atomics.pdf") == "Atomics.pdf"
    assert item_label("plain.txt") == "plain.txt"


def test_progress_counts_a_resumed_run_from_what_was_carried() -> None:
    events: list[ProgressEvent] = []
    clock = Clock()
    progress = Progress(
        run_id="r",
        total=10,
        clock=clock,
        started=clock.now,
        every_sources=25,
        every_seconds=60,
        carried=6,
        options=ProgressOptions(events.append),
    )

    progress.start()
    clock.now += 2
    progress.advance(5, 0)

    assert [e.done for e in events] == [6, 7]
    assert events[1].eta_seconds == pytest.approx(6.0)


TEXTS = ["alpha beta", "beta gamma delta", "alpha", "gamma", "delta epsilon"]


class Lock:
    @contextmanager
    def hold(self, operation: str = "build") -> Iterator[None]:
        yield

    def is_held(self) -> bool:
        return False


def _published() -> InMemoryPassageStore:
    store = InMemoryPassageStore()
    store.recreate_collection(2)
    for number, text in enumerate(TEXTS):
        source = Source(f"s{number}.txt", f"s{number}.txt", FileType.TXT)
        passage = Passage(
            make_passage_id(source.source_id, 0),
            source.source_id,
            text,
            0,
            len(text.split()),
        )
        store.upsert(
            passage, source, (1.0, float(number)), TermFrequencyEncoder().encode(text)
        )
    store.write_completion_metadata({"embedding_model": "m", "completed_at": "t"})
    return store


def test_a_reencode_reports_the_scan_and_the_writing_as_separate_phases() -> None:
    events: list[ProgressEvent] = []
    reencode = ReencodeSparse(
        lock=Lock(),
        index=_published(),
        manifest=InMemoryManifest(),
        encoders=SparseEncoderSet(),
        page_size=2,
    )

    reencode.run(progress=ProgressOptions(events.append))

    scanning = [e for e in events if e.phase == "scanning"]
    writing = [e for e in events if e.phase == "writing"]
    assert events == scanning + writing
    assert [e.done for e in scanning] == [0, 2, 4, 5, 5]
    assert scanning[-1].finished and not any(e.finished for e in scanning[:-1])
    assert all(e.operation == "reencode" and e.unit == "passages" for e in events)
    assert all(e.total == len(TEXTS) for e in events)
    assert writing[0].done == 0
    assert writing[-1].done == len(TEXTS) and writing[-1].finished


def test_a_reencode_plan_reports_its_scan() -> None:
    events: list[ProgressEvent] = []
    reencode = ReencodeSparse(
        lock=Lock(),
        index=_published(),
        manifest=InMemoryManifest(),
        encoders=SparseEncoderSet(),
        page_size=2,
    )

    reencode.plan(progress=ProgressOptions(events.append))

    assert {e.phase for e in events} == {"scanning"}
