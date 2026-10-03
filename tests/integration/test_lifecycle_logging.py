"""INFO-level lifecycle records: events, fields and what must never be logged."""

import logging
from collections.abc import Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from docgrep.adapters.models.real import BgeEmbedder, BgeReranker
from docgrep.api.container import Services, create_container
from docgrep.api.http import create_app
from docgrep.api.log_config import KeyValueFormatter
from docgrep.settings import Settings

PASSAGE = "A rare blue comet crosses the quiet sky."
QUERY = "blue comet crossing"


def _settings(tmp_path: Path) -> Settings:
    corpus = tmp_path / "library"
    corpus.mkdir()
    (corpus / "note.txt").write_text(PASSAGE, encoding="utf-8")
    (corpus / "broken.pdf").write_bytes(b"this is not a pdf")
    return Settings(
        corpus_root=corpus,
        data_dir=tmp_path / "data",
        backend="fake",
        chunk_size=10,
        chunk_overlap=2,
    )


def _events(caplog: pytest.LogCaptureFixture) -> dict[str, logging.LogRecord]:
    return {record.getMessage(): record for record in caplog.records}


def _fields(record: logging.LogRecord) -> dict[str, object]:
    standard = set(logging.makeLogRecord({}).__dict__) | {"message", "taskName"}
    return {k: v for k, v in record.__dict__.items() if k not in standard}


def _everything_logged(caplog: pytest.LogCaptureFixture) -> str:
    formatter = KeyValueFormatter()
    parts: list[str] = []
    for record in caplog.records:
        parts.append(formatter.format(record))
        parts.append(record.getMessage())
        parts.extend(str(value) for value in _fields(record).values())
    return "\n".join(parts)


@pytest.fixture
def services(tmp_path: Path) -> Services:
    return create_container(_settings(tmp_path))


def test_build_logs_start_failures_and_finish(
    services: Services, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="docgrep"):
        summary = services.build.run()

    events = _events(caplog)
    started = events["index_build_started"]
    assert started.levelno == logging.INFO
    assert _fields(started) == {
        "run_id": summary.run_id,
        "corpus_root": "library",
        "backend": "fake",
    }
    failed = events["index_source_failed"]
    assert failed.levelno == logging.WARNING
    assert _fields(failed) == {"source_id": "broken.pdf", "category": "parse_error"}
    finished = events["index_build_finished"]
    fields = _fields(finished)
    assert finished.levelno == logging.INFO
    assert {k: v for k, v in fields.items() if k != "duration_ms"} == {
        "run_id": summary.run_id,
        "indexed": 1,
        "skipped": 0,
        "failed": 1,
    }
    assert isinstance(fields["duration_ms"], float) and fields["duration_ms"] >= 0
    order = [record.getMessage() for record in caplog.records]
    assert order.index("index_build_started") < order.index("index_build_finished")


def test_build_log_names_only_the_corpus_directory(
    services: Services, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="docgrep"):
        services.build.run()

    assert str(tmp_path) not in _everything_logged(caplog)


def test_search_logs_one_record_without_query_or_passage_text(
    services: Services, caplog: pytest.LogCaptureFixture
) -> None:
    services.build.run()
    caplog.clear()

    with caplog.at_level(logging.INFO, logger="docgrep"):
        result = services.search.run(QUERY, top_n=3)

    assert result.results[0].passage.text == PASSAGE
    [record] = [r for r in caplog.records if r.getMessage() == "search_completed"]
    fields = _fields(record)
    assert set(fields) == {"top_n", "total_hits", "took_ms", "rerank_skipped"}
    assert fields["top_n"] == 3
    assert fields["total_hits"] == result.total_hits
    assert isinstance(fields["took_ms"], float)
    assert fields["rerank_skipped"] is False
    logged = _everything_logged(caplog)
    for forbidden in (QUERY, "blue comet", PASSAGE, "rare blue", "quiet sky"):
        assert forbidden not in logged


def test_empty_result_search_is_logged_too(
    services: Services, caplog: pytest.LogCaptureFixture
) -> None:
    services.build.run()
    caplog.clear()

    with caplog.at_level(logging.INFO, logger="docgrep"):
        services.search.run("zzzzunmatched")

    [record] = [r for r in caplog.records if r.getMessage() == "search_completed"]
    assert "zzzzunmatched" not in _everything_logged(caplog)
    assert _fields(record)["top_n"] == 10


class _Tokenizer:
    def __call__(
        self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool
    ) -> dict[str, list[tuple[int, int]]]:
        return {"offset_mapping": []}


class _Sentence:
    tokenizer = _Tokenizer()

    def get_sentence_embedding_dimension(self) -> int:
        return 2

    def encode(self, texts: Sequence[str], **_: object) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class _Cross:
    def predict(self, pairs: Sequence[tuple[str, str]], **_: object) -> list[float]:
        return [1.0 for _ in pairs]


def test_model_loading_is_logged_once_per_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    embedder = BgeEmbedder("embed-x", "auto", lambda *_: _Sentence(), lambda: False)
    reranker = BgeReranker("rerank-y", "cpu", lambda *_: _Cross())

    with caplog.at_level(logging.INFO, logger="docgrep"):
        embedder.embed(["secret text"])
        embedder.embed(["more secret text"])
        reranker.score("secret query", ["secret passage"])

    loads = [r for r in caplog.records if r.getMessage() == "model_loaded"]
    assert [(_fields(r)["model"], _fields(r)["device"]) for r in loads] == [
        ("embed-x", "cpu"),
        ("rerank-y", "cpu"),
    ]
    assert all(
        isinstance(_fields(r)["load_ms"], float) and _fields(r)["load_ms"] >= 0
        for r in loads
    )
    assert "secret" not in _everything_logged(caplog)


def test_server_start_logs_host_port_backend_and_data_directory(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = _settings(tmp_path)

    with caplog.at_level(logging.INFO, logger="docgrep"):
        with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
            assert client.get("/health").status_code == 200

    [record] = [r for r in caplog.records if r.getMessage() == "server_started"]
    assert record.levelno == logging.INFO
    assert _fields(record) == {
        "host": "127.0.0.1",
        "port": 8000,
        "backend": "fake",
        "data_dir": str(tmp_path / "data"),
    }
