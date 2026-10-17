"""`docgrep reencode-sparse` and what status says about the sparse weighting,
against local-mode Qdrant and SQLite (ADR-0025)."""

import json
import signal
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from docgrep.adapters.filelock import FileIndexLock
from docgrep.adapters.manifest import SQLiteManifest
from docgrep.adapters.models.sparse import SparseEncoderSet
from docgrep.adapters.qdrant import QdrantPassageStore
from docgrep.api.cli import _format_cancel, main
from docgrep.api.container import create_container
from docgrep.settings import Settings
from docgrep.usecases.cancel_index import CancelResult
from docgrep.usecases.reencode_sparse import ReencodeSparse
from resume_support import close, dump_points, write_corpus
from sparse_support import as_tf_v1_build


class Signals:
    """A stand-in for signal.signal that only remembers the handlers."""

    def __init__(self) -> None:
        self.handlers: dict[int, Any] = {}

    def __call__(self, number: int, handler: Any) -> Any:
        previous = self.handlers.get(number, signal.SIG_DFL)
        self.handlers[number] = handler
        return previous


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    for name, value in {
        "DOCGREP_CORPUS_ROOT": str(tmp_path / "corpus"),
        "DOCGREP_DATA_DIR": str(tmp_path / "data"),
        "DOCGREP_BACKEND": "fake",
        "DOCGREP_CHUNK_SIZE": "10",
        "DOCGREP_CHUNK_OVERLAP": "2",
        "DOCGREP_LOG_LEVEL": "WARNING",
    }.items():
        monkeypatch.setenv(name, value)
    return Settings()


def _tf_v1_index(settings: Settings) -> None:
    write_corpus(settings.corpus_root, 5, words=31)
    services = create_container(settings)
    assert as_tf_v1_build(services.build).run().indexed == 5
    close(services)


def _run(
    capsys: pytest.CaptureFixture[str], *args: str, **options: Any
) -> tuple[int, str, str]:
    status = main(list(args), **options)
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def _status(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    status, out, _ = _run(capsys, "status", "--json")
    assert status == 0
    report: dict[str, Any] = json.loads(out)
    return report


def _metadata(settings: Settings) -> Mapping[str, Any]:
    store = QdrantPassageStore(settings.data_dir / "qdrant")
    try:
        metadata = store.read_completion_metadata()
    finally:
        store.close()
    assert metadata is not None
    return metadata


def test_the_plan_shows_the_weightings_and_writes_nothing(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    _tf_v1_index(settings)
    before = dump_points(settings)

    status, out, _ = _run(capsys, "reencode-sparse")

    assert status == 0
    lines = out.splitlines()
    assert lines[0] == f"Sparse re-encode of {len(before)} passages"
    assert lines[1] == "  current: tf-v1"
    assert lines[2].startswith("  new:     bm25-v2 (k1 1.2, b 0.75, avgdl ")
    assert "No model is loaded (CPU only)" in out
    assert lines[-1] == "Run again with --yes to re-encode."
    assert dump_points(settings) == before
    manifest = SQLiteManifest(settings.data_dir)
    assert manifest.latest_run_of_kind("reencode_sparse") is None


def test_yes_re_encodes_then_a_second_run_is_a_no_op(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    _tf_v1_index(settings)
    report = _status(capsys)
    assert report["sparse"]["encoder"] == "tf-v1"
    assert "reencode-sparse" in report["sparse"]["hint"]
    assert not [w for w in report["warnings"] if w["code"] == "reencode_unfinished"]

    status, out, _ = _run(capsys, "reencode-sparse", "--yes")

    assert status == 0
    assert "Sparse re-encode run " in out and "passages written" in out
    metadata = _metadata(settings)
    assert metadata["sparse_encoder"] == "bm25-v2"
    report = _status(capsys)
    sparse = report["sparse"]
    assert (sparse["encoder"], sparse["bm25_k1"], sparse["bm25_b"]) == (
        "bm25-v2",
        1.2,
        0.75,
    )
    assert sparse["bm25_avgdl"] == metadata["bm25_avgdl"]
    assert (sparse["reencode"], sparse["hint"]) == (None, None)
    assert report["latest_run"]["kind"] == "reencode_sparse"
    _, text, _ = _run(capsys, "status")
    assert "Sparse encoder: bm25-v2 (k1 1.2, b 0.75, avgdl " in text

    status, out, _ = _run(capsys, "reencode-sparse", "--yes")

    assert status == 0
    assert out.startswith("The index already has bm25-v2 (k1 1.2, b 0.75")
    assert "nothing to re-encode" in out
    runs = SQLiteManifest(settings.data_dir).latest_run()
    assert runs is not None and runs.run_id == report["latest_run"]["run_id"]


def test_other_parameters_re_encode_again(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    _tf_v1_index(settings)
    assert _run(capsys, "reencode-sparse", "--yes")[0] == 0

    status, out, _ = _run(
        capsys, "reencode-sparse", "--yes", "--k1", "0.9", "--b", "0.4"
    )

    assert status == 0
    assert "current: bm25-v2 (k1 1.2, b 0.75" in out
    assert "new:     bm25-v2 (k1 0.9, b 0.4" in out
    metadata = _metadata(settings)
    assert (metadata["bm25_k1"], metadata["bm25_b"]) == (0.9, 0.4)


@pytest.mark.parametrize("args", [["--b", "1.5"], ["--k1", "-1"]])
def test_invalid_parameters_are_usage_errors(
    settings: Settings, capsys: pytest.CaptureFixture[str], args: list[str]
) -> None:
    _tf_v1_index(settings)
    before = dump_points(settings)

    status, _, err = _run(capsys, "reencode-sparse", "--yes", *args)

    assert status == 2
    assert "usage: validation_error" in err
    assert dump_points(settings) == before


def test_without_an_index_nothing_is_created(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    status, _, err = _run(capsys, "reencode-sparse", "--yes")

    assert status == 1
    assert "index_incomplete" in err
    assert not settings.data_dir.exists()


def test_another_index_operation_refuses_it(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    _tf_v1_index(settings)
    before = dump_points(settings)

    with FileIndexLock(settings.data_dir / "index.lock").hold("update"):
        status, _, err = _run(capsys, "reencode-sparse", "--yes")

    assert status == 1
    assert "index_lock_held" in err and "sparse re-encode" in err
    assert dump_points(settings) == before


class _InterruptingStore(QdrantPassageStore):
    """Delivers a SIGINT through the CLI's handler after the first page."""

    signals: Signals
    pages = 0

    def write_sparse_vectors(
        self, vectors: Sequence[tuple[str, Mapping[int, float]]]
    ) -> None:
        super().write_sparse_vectors(vectors)
        self.pages += 1
        if self.pages == 1:
            self.signals.handlers[signal.SIGINT](signal.SIGINT, None)


def test_ctrl_c_stops_after_a_page_and_the_next_run_continues(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    _tf_v1_index(settings)
    signals = Signals()

    @contextmanager
    def interrupted(chosen: Settings) -> Iterator[ReencodeSparse]:
        store = _InterruptingStore(chosen.data_dir / "qdrant")
        store.signals = signals
        try:
            yield ReencodeSparse(
                lock=FileIndexLock(chosen.data_dir / "index.lock"),
                index=store,
                manifest=SQLiteManifest(chosen.data_dir),
                encoders=SparseEncoderSet(),
                page_size=3,
            )
        finally:
            store.close()

    status, out, err = _run(
        capsys,
        "reencode-sparse",
        "--yes",
        reencode_factory=interrupted,
        register_signal=signals,
    )

    assert status == 2
    assert "Cancel requested: the sparse re-encode stops after the current page" in err
    assert "cancelled after 3 passages" in out
    report = _status(capsys)
    reencode = report["sparse"]["reencode"]
    assert (reencode["state"], reencode["passages_done"]) == ("cancelled", 3)
    assert "reencode-sparse --yes" in report["sparse"]["hint"]
    assert [w["code"] for w in report["warnings"]] == ["reencode_unfinished"]
    _, text, _ = _run(capsys, "status")
    assert "re-encode run" in text and "cancelled at 3 of" in text

    status, out, _ = _run(capsys, "reencode-sparse", "--yes")

    assert status == 0
    assert "continues run" in out and "from 3 of" in out
    report = _status(capsys)
    assert report["sparse"]["encoder"] == "bm25-v2"
    assert report["sparse"]["reencode"] is None
    assert report["warnings"] == []


def test_the_cancel_message_of_an_interrupted_re_encode() -> None:
    text, status = _format_cancel(
        CancelResult("interrupted", "7", kind="reencode_sparse")
    )

    assert status == 0
    assert text.startswith("Sparse re-encode run 7 was interrupted")
    assert "docgrep reencode-sparse --yes" in text
