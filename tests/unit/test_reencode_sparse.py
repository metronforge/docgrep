"""Re-encoding the stored sparse vectors through in-memory fakes (ADR-0025)."""

import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import pytest

from docgrep.adapters.models.sparse import (
    BM25DocumentEncoder,
    SparseEncoderSet,
    TermFrequencyEncoder,
)
from docgrep.domain.errors import (
    BuildFailed,
    IndexIncomplete,
    IndexLockHeld,
    ResumeAvailable,
    ValidationError,
)
from docgrep.domain.models import FileType, Passage, Source, make_passage_id
from docgrep.domain.run_state import unfinished_reencode
from docgrep.domain.sparse import SparseWeighting
from docgrep.usecases.reencode_sparse import (
    ReencodePlan,
    ReencodeSparse,
    reencode_progress,
)
from fakes import InMemoryManifest, InMemoryPassageStore

TEXTS = [
    "alpha beta gamma",
    "alpha alpha alpha alpha alpha alpha delta",
    "beta",
    "gamma delta epsilon zeta eta theta",
    "alpha",
    "zeta zeta",
    "eta theta iota kappa",
]
AVGDL = sum(len(text.split()) for text in TEXTS) / len(TEXTS)


class Lock:
    def __init__(self) -> None:
        self.held = False
        self.operations: list[str] = []

    @contextmanager
    def hold(self, operation: str = "build") -> Iterator[None]:
        if self.held:
            raise IndexLockHeld(f"held, {operation} refused")
        self.operations.append(operation)
        yield

    def is_held(self) -> bool:
        return self.held


class CountingStore(InMemoryPassageStore):
    """Records every sparse write; can fail or ask for a cancel on a page."""

    def __init__(self) -> None:
        super().__init__()
        self.writes: list[list[str]] = []
        self.fail_on_write: int | None = None
        self.on_write: Any = None

    def write_sparse_vectors(
        self, vectors: Sequence[tuple[str, Mapping[int, float]]]
    ) -> None:
        if self.fail_on_write == len(self.writes):
            raise RuntimeError("injected store failure")
        self.writes.append([passage_id for passage_id, _ in vectors])
        super().write_sparse_vectors(vectors)
        if self.on_write is not None:
            self.on_write(len(self.writes))


def _published(store: InMemoryPassageStore, **metadata: str | int | float) -> None:
    """A published tf-v1 index of TEXTS, as 1.20.0 wrote it."""
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
        store.upsert(passage, source, (1.0, float(number)), _tf(text))
    store.write_completion_metadata(
        {"embedding_model": "m", "completed_at": "2026-10-01", **metadata}
    )


def _tf(text: str) -> dict[int, float]:
    return TermFrequencyEncoder().encode(text)


class Harness:
    def __init__(self, **metadata: str | int | float) -> None:
        self.store = CountingStore()
        self.manifest = InMemoryManifest()
        self.lock = Lock()
        _published(self.store, **metadata)

    def use_case(self, page_size: int = 2) -> ReencodeSparse:
        return ReencodeSparse(
            lock=self.lock,
            index=self.store,
            manifest=self.manifest,
            encoders=SparseEncoderSet(),
            page_size=page_size,
        )

    def points(self) -> dict[str, tuple[Any, ...]]:
        """Every point without its sparse vector."""
        return {passage_id: item[:3] for passage_id, item in self.store._items.items()}

    def vectors(self) -> dict[str, dict[int, float]]:
        return {
            passage_id: self.store.sparse_vector(passage_id)
            for passage_id in self.store._items
        }

    def texts(self) -> dict[str, str]:
        return {
            passage_id: item[0].text for passage_id, item in self.store._items.items()
        }


def _assert_bm25(harness: Harness, k1: float, b: float, avgdl: float) -> None:
    encoder = BM25DocumentEncoder(k1, b, avgdl)
    for passage_id, text in harness.texts().items():
        assert harness.store.sparse_vector(passage_id) == pytest.approx(
            encoder.encode(text)
        )


def test_the_plan_reports_and_writes_nothing() -> None:
    harness = Harness()
    before = harness.vectors()

    plan = harness.use_case().plan()

    assert plan == ReencodePlan(
        len(TEXTS), SparseWeighting(), SparseWeighting.bm25(1.2, 0.75, AVGDL)
    )
    assert harness.vectors() == before
    assert harness.store.writes == []
    assert harness.manifest.latest_run() is None
    metadata = harness.store.read_completion_metadata()
    assert metadata is not None and "sparse_encoder" not in metadata


def test_a_run_rewrites_only_the_sparse_vectors_and_records_the_weighting() -> None:
    harness = Harness()
    points = harness.points()
    seen: list[ReencodePlan] = []

    summary = harness.use_case().run(on_plan=seen.append)

    assert summary.written == len(TEXTS) and not summary.cancelled
    assert seen == [summary.plan]
    assert harness.points() == points
    _assert_bm25(harness, 1.2, 0.75, AVGDL)
    assert sorted(sum(harness.store.writes, [])) == sorted(points)
    metadata = harness.store.read_completion_metadata()
    assert metadata is not None
    assert {k: metadata[k] for k in ("sparse_encoder", "bm25_k1", "bm25_b")} == {
        "sparse_encoder": "bm25-v2",
        "bm25_k1": 1.2,
        "bm25_b": 0.75,
    }
    assert metadata["bm25_avgdl"] == pytest.approx(AVGDL)
    assert metadata["completed_at"] == "2026-10-01"
    run = harness.manifest.latest_run()
    assert run is not None
    assert (run.kind, run.status, run.total_sources) == (
        "reencode_sparse",
        "completed",
        len(TEXTS),
    )
    assert reencode_progress(run) == (len(TEXTS), len(TEXTS))
    assert harness.lock.operations == ["sparse re-encode"]


def test_a_second_run_with_the_same_parameters_is_a_no_op() -> None:
    harness = Harness()
    harness.use_case().run()
    runs = harness.manifest.latest_run()
    writes = len(harness.store.writes)

    summary = harness.use_case().run()

    assert summary.plan.noop and summary.run_id is None
    assert summary.plan.current == summary.plan.target
    assert harness.use_case().plan().noop
    assert len(harness.store.writes) == writes
    assert harness.manifest.latest_run() == runs


def test_other_parameters_re_encode_again() -> None:
    harness = Harness()
    harness.use_case().run()

    summary = harness.use_case().run(k1=0.9, b=0.3)

    assert summary.written == len(TEXTS)
    assert summary.plan.current == SparseWeighting.bm25(1.2, 0.75, AVGDL)
    _assert_bm25(harness, 0.9, 0.3, AVGDL)
    metadata = harness.store.read_completion_metadata()
    assert metadata is not None
    assert (metadata["bm25_k1"], metadata["bm25_b"]) == (0.9, 0.3)


def _cancel_after(harness: Harness, pages: int) -> None:
    def cancel(written: int) -> None:
        if written == pages:
            run = harness.manifest.latest_run()
            assert run is not None
            harness.manifest.request_cancel(run.run_id)

    harness.store.on_write = cancel


def test_a_cancelled_run_continues_from_its_checkpoint_with_its_average() -> None:
    harness = Harness()
    _cancel_after(harness, 2)

    first = harness.use_case().run()

    assert first.cancelled and first.written == 4
    stopped = harness.manifest.latest_run()
    assert stopped is not None and stopped.status == "cancelled"
    checkpoint = json.loads(stopped.checkpoint or "")
    assert checkpoint["done"] == 4 and checkpoint["offset"] is not None
    assert unfinished_reencode(harness.manifest, lock_held=False) == stopped
    metadata = harness.store.read_completion_metadata()
    assert metadata is not None and "sparse_encoder" not in metadata
    done = sum(harness.store.writes, [])

    # A passage not re-encoded yet grows, so a new average would differ;
    # the run keeps the one it started with.
    pending = sorted(set(harness.points()) - set(done))[0]
    passage, source, dense, sparse = harness.store._items[pending]
    longer = Passage(
        passage.passage_id, passage.source_id, passage.text + " omega" * 40, 0, 1
    )
    harness.store.upsert(longer, source, dense, sparse)
    harness.store.on_write = None
    plan = harness.use_case().plan()
    assert (plan.resumes, plan.done) == (stopped.run_id, 4)
    assert plan.target == SparseWeighting.bm25(1.2, 0.75, AVGDL)

    second = harness.use_case().run()

    assert not second.cancelled and second.written == len(TEXTS) - 4
    rewritten = sum(harness.store.writes, [])[len(done) :]
    assert set(rewritten).isdisjoint(done)
    assert sorted(done + rewritten) == sorted(harness.points())
    _assert_bm25(harness, 1.2, 0.75, AVGDL)
    run = harness.manifest.latest_run()
    assert run is not None and run.status == "completed"
    assert run.resumed_from == stopped.run_id
    assert run.carried_sources == 4
    assert unfinished_reencode(harness.manifest, lock_held=False) is None


def test_a_crashed_run_is_aborted_and_continued_by_the_next_one() -> None:
    harness = Harness()
    harness.store.fail_on_write = 2

    with pytest.raises(BuildFailed):
        harness.use_case().run()

    stopped = harness.manifest.latest_run()
    assert stopped is not None and stopped.status == "aborted"
    harness.store.fail_on_write = None
    summary = harness.use_case().run()

    assert summary.plan.resumes == stopped.run_id
    assert summary.written == len(TEXTS) - 4
    _assert_bm25(harness, 1.2, 0.75, AVGDL)


def test_an_interrupted_run_is_closed_and_continued() -> None:
    harness = Harness()
    _cancel_after(harness, 1)
    harness.use_case().run()
    # A process killed while running leaves its run as running.
    stopped = harness.manifest.latest_run()
    assert stopped is not None
    harness.manifest._runs[-1] = type(stopped)(
        **{
            **{f: getattr(stopped, f) for f in stopped.__dataclass_fields__},
            "status": "running",
            "cancel_requested_at": None,
        }
    )
    harness.store.on_write = None

    summary = harness.use_case().run()

    assert summary.plan.resumes == stopped.run_id
    statuses = [run.status for run in harness.manifest._runs]
    assert statuses == ["aborted", "completed"]


def test_an_unfinished_run_of_other_parameters_is_not_a_no_op() -> None:
    harness = Harness()
    harness.use_case().run()
    _cancel_after(harness, 4)
    harness.use_case().run(k1=0.5, b=0.5)
    harness.store.on_write = None

    # The index still records k1 1.2, but some vectors are at k1 0.5.
    plan = harness.use_case().plan()
    assert not plan.noop and plan.resumes is None

    harness.use_case().run()

    _assert_bm25(harness, 1.2, 0.75, AVGDL)


def test_a_held_lock_refuses_before_anything_is_written() -> None:
    harness = Harness()
    harness.lock.held = True

    with pytest.raises(IndexLockHeld):
        harness.use_case().run()

    assert harness.store.writes == []
    assert harness.manifest.latest_run() is None


@pytest.mark.parametrize(("k1", "b"), [(-0.1, 0.75), (1.2, 1.5), (1.2, -0.1)])
def test_invalid_parameters_are_refused(k1: float, b: float) -> None:
    harness = Harness()

    with pytest.raises(ValidationError):
        harness.use_case().plan(k1=k1, b=b)
    with pytest.raises(ValidationError):
        harness.use_case().run(k1=k1, b=b)


def test_an_unfinished_build_refuses() -> None:
    harness = Harness()
    run_id = harness.manifest.start_run({})
    harness.manifest.finish_run(run_id, "cancelled", {})

    with pytest.raises(ResumeAvailable):
        harness.use_case().run()


def test_without_a_published_index_it_refuses() -> None:
    use_case = ReencodeSparse(
        lock=Lock(),
        index=InMemoryPassageStore(),
        manifest=InMemoryManifest(),
        encoders=SparseEncoderSet(),
    )

    with pytest.raises(IndexIncomplete):
        use_case.plan()
