"""Re-encoding passages without any sparse token, and the progress of a
re-encode, through in-memory fakes (GitHub issue #48)."""

import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import pytest

from docgrep.adapters.models.sparse import (
    BM25DocumentEncoder,
    SparseEncoderSet,
    TermFrequencyEncoder,
)
from docgrep.domain.models import FileType, Passage, Source, make_passage_id
from docgrep.usecases.reencode_sparse import ReencodeSparse
from fakes import InMemoryManifest, InMemoryPassageStore
from sparse_support import NO_TOKEN_TEXT, record_aborted_1_21_run

TEXTS = [
    "alpha beta gamma",
    "alpha alpha delta",
    "beta",
    "gamma delta epsilon zeta",
    NO_TOKEN_TEXT,
    "zeta zeta",
    "eta theta iota kappa",
]
AVGDL = sum(len(text.split()) for text in TEXTS[:4] + TEXTS[5:]) / len(TEXTS)
COMPLETED_AT = "2026-10-01"


class Lock:
    @contextmanager
    def hold(self, operation: str = "build") -> Iterator[None]:
        yield

    def is_held(self) -> bool:
        return False


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class StrictStore(InMemoryPassageStore):
    """Rejects a point without a vector to update, as a Qdrant server does,
    and records the passages of every write."""

    def __init__(self) -> None:
        super().__init__()
        self.writes: list[list[str]] = []
        self.clock: Clock | None = None
        self.scan_pages = 0

    def passage_text_page(
        self, offset: str | None, limit: int
    ) -> tuple[tuple[tuple[str, str], ...], str | None]:
        # The planning scan reads without writing: make it slow.
        if self.clock is not None and not self.writes and self.scan_pages > 0:
            self.scan_pages -= 1
            self.clock.now += 100.0
        return super().passage_text_page(offset, limit)

    def write_sparse_vectors(
        self, vectors: Sequence[tuple[str, Mapping[int, float]]]
    ) -> None:
        for passage_id, sparse in vectors:
            if not sparse:
                raise RuntimeError(f"must specify vectors to update for {passage_id}")
        self.writes.append([passage_id for passage_id, _ in vectors])
        if self.clock is not None:
            self.clock.now += 1.0
        super().write_sparse_vectors(vectors)


def _ids() -> list[str]:
    return [make_passage_id(f"s{n}.txt", 0) for n in range(len(TEXTS))]


EMPTY_ID = _ids()[TEXTS.index(NO_TOKEN_TEXT)]


def _harness() -> tuple[StrictStore, InMemoryManifest]:
    store = StrictStore()
    store.recreate_collection(2)
    for number, text in enumerate(TEXTS):
        source = Source(f"s{number}.txt", f"s{number}.txt", FileType.TXT)
        passage = Passage(
            make_passage_id(source.source_id, 0), source.source_id, text, 0, 1
        )
        store.upsert(
            passage, source, (1.0, float(number)), TermFrequencyEncoder().encode(text)
        )
    store.write_completion_metadata(
        {"embedding_model": "m", "completed_at": COMPLETED_AT}
    )
    return store, InMemoryManifest()


def _use_case(
    store: StrictStore, manifest: InMemoryManifest, **options: Any
) -> ReencodeSparse:
    return ReencodeSparse(
        lock=Lock(),
        index=store,
        manifest=manifest,
        encoders=SparseEncoderSet(),
        page_size=2,
        **options,
    )


def _assert_bm25(store: StrictStore, avgdl: float, ids: Sequence[str]) -> None:
    encoder = BM25DocumentEncoder(1.2, 0.75, avgdl)
    for passage_id in ids:
        text = TEXTS[_ids().index(passage_id)]
        assert store.sparse_vector(passage_id) == pytest.approx(encoder.encode(text))


def test_a_passage_without_tokens_is_skipped_counted_and_left_empty() -> None:
    store, manifest = _harness()
    assert store.sparse_vector(EMPTY_ID) == {}

    summary = _use_case(store, manifest).run()

    assert not summary.cancelled
    assert (summary.written, summary.skipped) == (len(TEXTS) - 1, 1)
    assert EMPTY_ID not in sum(store.writes, [])
    assert store.sparse_vector(EMPTY_ID) == {}
    _assert_bm25(store, AVGDL, [i for i in _ids() if i != EMPTY_ID])
    metadata = store.read_completion_metadata()
    assert metadata is not None and metadata["sparse_encoder"] == "bm25-v2"
    assert metadata["bm25_avgdl"] == pytest.approx(AVGDL)
    run = manifest.latest_run()
    assert run is not None and run.status == "completed"
    assert dict(run.totals) == {"passages": len(TEXTS), "skipped_empty": 1}
    checkpoint = json.loads(run.checkpoint or "")
    assert (checkpoint["done"], checkpoint["skipped"]) == (len(TEXTS), 1)


def test_an_aborted_1_21_run_resumes_from_its_checkpoint_with_its_average() -> None:
    store, manifest = _harness()
    order = sorted(_ids())
    position = order.index(EMPTY_ID)
    assert position >= 2, "the checkpoint must lie before the empty passage"
    # 1.21.0 wrote the first page, then failed on the page holding the empty
    # passage; the next page starts at ``order[2]``.
    recorded = 201.17
    stopped = record_aborted_1_21_run(
        manifest,
        completed_at=COMPLETED_AT,
        avgdl=recorded,
        total=len(TEXTS),
        offset=order[2],
        done=2,
    )

    use_case = _use_case(store, manifest)
    plan = use_case.plan()
    assert (plan.resumes, plan.done) == (stopped, 2)
    summary = use_case.run()

    assert summary.plan.resumes == stopped
    assert (summary.written, summary.skipped) == (len(TEXTS) - 3, 1)
    assert sorted(sum(store.writes, [])) == [i for i in order[2:] if i != EMPTY_ID]
    # The passages before the checkpoint are not written again.
    for passage_id in order[:2]:
        text = TEXTS[_ids().index(passage_id)]
        assert store.sparse_vector(passage_id) == TermFrequencyEncoder().encode(text)
    _assert_bm25(store, recorded, [i for i in order[2:] if i != EMPTY_ID])
    assert store.sparse_vector(EMPTY_ID) == {}
    metadata = store.read_completion_metadata()
    assert metadata is not None and metadata["bm25_avgdl"] == recorded
    run = manifest.latest_run()
    assert run is not None and run.status == "completed"
    assert (run.resumed_from, run.carried_sources) == (stopped, 2)
    assert dict(run.totals) == {"passages": len(TEXTS), "skipped_empty": 1}


def _progress(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in record.__dict__.items()
            if key not in {*logging.makeLogRecord({}).__dict__, "message"}
        }
        for record in caplog.records
        if record.getMessage() == "index_progress"
    ]


def test_progress_counts_passages_and_measures_the_writing_phase_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store, manifest = _harness()
    clock = Clock()
    store.clock = clock
    # The planning scan reads every page once: 4 pages of 2, 100 s each.
    store.scan_pages = 4
    use_case = _use_case(store, manifest, clock=clock, progress_every_passages=1)

    with caplog.at_level(logging.INFO, logger="docgrep"):
        summary = use_case.run()

    records = _progress(caplog)
    assert records, "no progress record"
    for record in records:
        assert "sources_done" not in record and "sources_total" not in record
        assert record["passages_total"] == len(TEXTS)
    assert [r["passages_done"] for r in records] == [2, 4, 6, 7]
    first = records[0]
    # One page written in 1 s after a 400 s scan: 2 passages/s and a few
    # seconds left, not 0.005 passages/s and an ETA of about half an hour.
    assert first["elapsed_seconds"] == 1.0
    assert first["passages_written"] == 2
    assert first["passages_per_second"] == 2.0
    assert first["eta_seconds"] is not None and first["eta_seconds"] <= 3
    # The page holding the empty passage writes one passage but does two.
    assert sorted(_ids()).index(EMPTY_ID) in (2, 3)
    assert (records[1]["passages_done"], records[1]["passages_written"]) == (4, 3)
    assert summary.elapsed_seconds >= 400.0
