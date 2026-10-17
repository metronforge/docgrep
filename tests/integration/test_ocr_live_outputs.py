"""The OCR registry's output-to-original map, read in one query (ADR-0021):
SQLite adapter and in-memory fake alike."""

from collections.abc import Callable
from pathlib import Path

import pytest
from blake3 import blake3

from docgrep.adapters.manifest import SQLiteOcrRegistry
from docgrep.domain.models import OcrOutput, OcrRecord
from docgrep.domain.protocols import OcrRegistry
from fakes import InMemoryOcrRegistry

Factory = Callable[[Path, Path], OcrRegistry]


def _sqlite(data_dir: Path, corpus: Path) -> OcrRegistry:
    return SQLiteOcrRegistry(data_dir, corpus)


def _memory(data_dir: Path, corpus: Path) -> OcrRegistry:
    return InMemoryOcrRegistry(exists=lambda path: (corpus / path).is_file())


@pytest.fixture(params=[_sqlite, _memory], ids=["sqlite", "memory"])
def factory(request: pytest.FixtureRequest) -> Factory:
    chosen: Factory = request.param
    return chosen


def _write(corpus: Path, path: str) -> str:
    (corpus / path).parent.mkdir(parents=True, exist_ok=True)
    (corpus / path).write_bytes(f"%PDF {path}".encode())
    return blake3((corpus / path).read_bytes()).hexdigest()


def _record(corpus: Path, input_path: str, output_path: str, stamp: str) -> OcrRecord:
    return OcrRecord(
        input_path=input_path,
        input_format="djvu" if input_path.endswith(".djvu") else "pdf",
        input_hash=_write(corpus, input_path),
        output_path=output_path,
        output_hash=_write(corpus, output_path),
        engine="ocrmypdf",
        engine_version="16.0.0",
        languages="rus+eng",
        mode="skip-text",
        created_at=stamp,
    )


def test_live_outputs_map_each_output_to_its_original(
    tmp_path: Path, factory: Factory
) -> None:
    corpus = tmp_path / "corpus"
    registry = factory(tmp_path / "data", corpus)
    assert registry.live_outputs() == {}
    registry.register(
        _record(corpus, "a/book.djvu", "a/book.ocr.pdf", "2025-03-01T00:00:00+00:00")
    )
    registry.register(
        _record(corpus, "b/scan.pdf", "b/scan.ocr.pdf", "2026-01-02T00:00:00+00:00")
    )
    registry.register(
        _record(corpus, "c/gone.pdf", "c/gone.ocr.pdf", "2026-01-03T00:00:00+00:00")
    )
    (corpus / "c" / "gone.ocr.pdf").unlink()

    assert dict(registry.live_outputs()) == {
        "a/book.ocr.pdf": OcrOutput(
            "a/book.ocr.pdf", "a/book.djvu", "2025-03-01T00:00:00+00:00"
        ),
        "b/scan.ocr.pdf": OcrOutput(
            "b/scan.ocr.pdf", "b/scan.pdf", "2026-01-02T00:00:00+00:00"
        ),
    }


def test_an_output_of_several_inputs_maps_to_the_first_by_path(
    tmp_path: Path, factory: Factory
) -> None:
    corpus = tmp_path / "corpus"
    registry = factory(tmp_path / "data", corpus)
    registry.register(_record(corpus, "z/copy.pdf", "a/out.pdf", "2026-01-01"))
    registry.register(_record(corpus, "m/first.pdf", "a/out.pdf", "2026-01-01"))

    outputs = registry.live_outputs()

    assert outputs["a/out.pdf"].input_path == "m/first.pdf"
    assert registry.output_owner("a/out.pdf") == "m/first.pdf"


def test_a_missing_database_reads_as_no_outputs(tmp_path: Path) -> None:
    registry = SQLiteOcrRegistry(tmp_path / "data", tmp_path, read_only=True)

    assert registry.live_outputs() == {}
