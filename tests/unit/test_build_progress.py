"""Progress records of a build: cadence, ETA arithmetic and what they never carry."""

import logging
from pathlib import Path

import pytest

from docgrep.adapters.chunking import TokenChunker, WhitespaceTokenizer
from docgrep.adapters.filelock import FileIndexLock
from docgrep.adapters.models.deterministic import DeterministicEmbedder
from docgrep.api.log_config import KeyValueFormatter
from docgrep.domain.models import FileType, ParsedDocument, ParsedSection, Source
from docgrep.domain.progress import estimate_eta_seconds
from docgrep.usecases.build_index import BuildIndex
from fakes import InMemoryManifest, InMemoryPassageStore

SECRET = "private-title-of-a-book"


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
    """Parse each source and let a fixed amount of fake time pass."""

    def __init__(self, clock: Clock, seconds: float) -> None:
        self.clock = clock
        self.seconds = seconds

    def parse(self, source: Source) -> ParsedDocument:
        self.clock.now += self.seconds
        if source.path.startswith("bad"):
            raise ValueError("unparseable " + SECRET)
        return ParsedDocument(source, (ParsedSection("alpha beta gamma delta"),))


class Sparse:
    def encode(self, text: str) -> dict[int, float]:
        return {len(text): 1.0}


def _build(
    tmp_path: Path,
    paths: list[str],
    *,
    every_sources: int = 25,
    every_seconds: float = 60.0,
    seconds_per_source: float = 1.0,
) -> BuildIndex:
    clock = Clock()
    return BuildIndex(
        lock=FileIndexLock(tmp_path / "index.lock"),
        enumerator=Sources(
            *(Source(p, p, FileType.TXT, title=SECRET, content_hash="h") for p in paths)
        ),
        parsers={FileType.TXT: TickingParser(clock, seconds_per_source)},
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
        progress_every_seconds=every_seconds,
    )


def _progress(caplog: pytest.LogCaptureFixture) -> list[dict[str, object]]:
    return [
        {
            key: value
            for key, value in record.__dict__.items()
            if key not in {*logging.makeLogRecord({}).__dict__, "message"}
        }
        for record in caplog.records
        if record.getMessage() == "index_progress"
    ]


def _names(count: int) -> list[str]:
    return [f"doc{i:02d}.txt" for i in range(count)]


def test_records_follow_the_source_count_cadence_and_end_the_loop(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    build = _build(tmp_path, _names(7), every_sources=3)

    with caplog.at_level(logging.INFO, logger="docgrep"):
        build.run()

    assert [r["sources_done"] for r in _progress(caplog)] == [3, 6, 7]


def test_the_default_cadence_is_every_25_sources(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    build = _build(tmp_path, _names(51))

    with caplog.at_level(logging.INFO, logger="docgrep"):
        build.run()

    assert [r["sources_done"] for r in _progress(caplog)] == [25, 50, 51]
    assert build.progress_every_seconds == 60.0


def test_records_follow_the_time_cadence_when_it_comes_first(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    build = _build(
        tmp_path, _names(7), every_sources=100, every_seconds=10, seconds_per_source=4
    )

    with caplog.at_level(logging.INFO, logger="docgrep"):
        build.run()

    # 12 s pass before the third source ends, then again before the sixth.
    assert [r["sources_done"] for r in _progress(caplog)] == [3, 6, 7]


def test_no_duplicate_record_when_the_last_source_hits_the_cadence(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    build = _build(tmp_path, _names(6), every_sources=3)

    with caplog.at_level(logging.INFO, logger="docgrep"):
        build.run()

    assert [r["sources_done"] for r in _progress(caplog)] == [3, 6]


def test_an_empty_corpus_still_logs_the_end_of_the_loop(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="docgrep"):
        _build(tmp_path, []).run()

    (record,) = _progress(caplog)
    assert record["sources_done"] == 0 and record["sources_total"] == 0
    assert record["eta_seconds"] is None


def test_record_fields_and_arithmetic(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    names = ["doc0.txt", "bad1.txt", "doc2.txt", "doc3.txt"]
    build = _build(tmp_path, names, every_sources=2, seconds_per_source=5)

    with caplog.at_level(logging.INFO, logger="docgrep"):
        summary = build.run()

    first, last = _progress(caplog)
    assert first["run_id"] == summary.run_id
    assert first["sources_done"] == 2 and first["sources_total"] == 4
    # Two passages for the one parsed source, none for the unparseable one.
    assert first["passages_written"] == 2 and first["failed"] == 1
    assert first["elapsed_seconds"] == 10.0
    assert first["passages_per_second"] == 0.2
    assert first["eta_seconds"] == 10  # 2 left at 5 s each
    assert last["sources_done"] == 4 and last["eta_seconds"] == 0
    assert last["passages_written"] == 6 and last["failed"] == 1


def test_eta_is_null_until_it_is_computable_and_renders_as_null() -> None:
    assert estimate_eta_seconds(done=0, total=10, elapsed_seconds=5) is None
    assert estimate_eta_seconds(done=3, total=10, elapsed_seconds=0) is None
    assert estimate_eta_seconds(done=4, total=10, elapsed_seconds=8) == 12
    assert estimate_eta_seconds(done=10, total=10, elapsed_seconds=8) == 0
    assert estimate_eta_seconds(done=11, total=10, elapsed_seconds=8) == 0

    record = logging.makeLogRecord(
        {"msg": "index_progress", "eta_seconds": None, "failed": 0}
    )
    assert "eta_seconds=null" in KeyValueFormatter().format(record)


def test_records_never_carry_paths_titles_or_text(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    build = _build(tmp_path, ["bad-secret-path.txt", "doc.txt"], every_sources=1)

    with caplog.at_level(logging.INFO, logger="docgrep"):
        build.run()

    records = [r for r in caplog.records if r.getMessage() == "index_progress"]
    assert records
    for record in records:
        line = KeyValueFormatter().format(record)
        assert "secret-path" not in line and SECRET not in line
        assert "alpha beta" not in line and ".txt" not in line
    assert set(_progress(caplog)[0]) == {
        "run_id",
        "sources_done",
        "sources_total",
        "passages_written",
        "elapsed_seconds",
        "passages_per_second",
        "eta_seconds",
        "failed",
    }


@pytest.mark.parametrize("name", ["progress_every_sources", "progress_every_seconds"])
def test_a_non_positive_cadence_is_rejected(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match=name):
        _build(tmp_path, [], **{name.replace("progress_", ""): 0})
