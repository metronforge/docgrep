"""`docgrep reencode-sparse --yes` over a passage without any sparse token
(GitHub issue #48). A Qdrant server rejects a point with no vector to
update; runs in local mode, and in server mode against
DOCGREP_TEST_QDRANT_URL."""

import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient, models

from docgrep.adapters.manifest import SQLiteManifest
from docgrep.adapters.models.sparse import (
    BM25DocumentEncoder,
    SparseEncoderSet,
    TermFrequencyEncoder,
)
from docgrep.adapters.qdrant import QdrantPassageStore
from docgrep.api.cli import main
from docgrep.api.container import create_container
from docgrep.settings import Settings
from resume_support import close, dump_points, write_corpus
from sparse_support import NO_TOKEN_TEXT, as_tf_v1_build, record_aborted_1_21_run

URL = os.environ.get("DOCGREP_TEST_QDRANT_URL")
ALIAS = "reempty"
SOURCES = 6


def _wipe(client: QdrantClient) -> None:
    for alias in client.get_aliases().aliases:
        if alias.alias_name == ALIAS:
            client.update_collection_aliases(
                [
                    models.DeleteAliasOperation(
                        delete_alias=models.DeleteAlias(alias_name=ALIAS)
                    )
                ]
            )
    for collection in client.get_collections().collections:
        if collection.name.startswith(f"{ALIAS}_"):
            client.delete_collection(collection.name)


@pytest.fixture(
    params=[
        "local",
        pytest.param(
            "server",
            marks=[
                pytest.mark.qdrant_server,
                pytest.mark.skipif(
                    not URL, reason="DOCGREP_TEST_QDRANT_URL is not set"
                ),
            ],
        ),
    ]
)
def settings(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Settings]:
    server = request.param == "server"
    environment = {
        "DOCGREP_CORPUS_ROOT": str(tmp_path / "corpus"),
        "DOCGREP_DATA_DIR": str(tmp_path / "data"),
        "DOCGREP_BACKEND": "fake",
        "DOCGREP_CHUNK_SIZE": "64",
        "DOCGREP_CHUNK_OVERLAP": "8",
        "DOCGREP_LOG_LEVEL": "WARNING",
        "DOCGREP_QDRANT_ALIAS": ALIAS,
        **({"DOCGREP_QDRANT_URL": URL} if server else {}),
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value or "")
    client = QdrantClient(url=URL) if server else None
    if client is not None:
        _wipe(client)
    try:
        yield Settings()
    finally:
        if client is not None:
            _wipe(client)
            client.close()


def _tf_v1_index(settings: Settings) -> list[Any]:
    """A tf-v1 index of one-passage sources, one of them without tokens."""
    write_corpus(settings.corpus_root, SOURCES, words=20)
    (settings.corpus_root / "symbols.txt").write_text(NO_TOKEN_TEXT, "utf-8")
    services = create_container(settings)
    try:
        assert as_tf_v1_build(services.build).run().indexed == SOURCES + 1
    finally:
        close(services)
    points = dump_points(settings)
    assert len(points) == SOURCES + 1
    assert [p for p in points if p[1]["text"] == NO_TOKEN_TEXT]
    return points


def _store(settings: Settings) -> QdrantPassageStore:
    if settings.qdrant_url:
        return QdrantPassageStore(url=settings.qdrant_url, alias=settings.qdrant_alias)
    return QdrantPassageStore(settings.data_dir / "qdrant", alias=settings.qdrant_alias)


def _sparse(vector: Any) -> dict[int, float]:
    sparse = vector.get("sparse")
    if sparse is None:
        return {}
    return dict(zip(sparse.indices, sparse.values, strict=True))


def _average(points: list[Any]) -> float:
    lengths = [SparseEncoderSet().document_length(p[1]["text"]) for p in points]
    return sum(lengths) / len(lengths)


def _run(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    status = main(list(args))
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def _metadata(settings: Settings) -> Any:
    store = _store(settings)
    try:
        return store.read_completion_metadata()
    finally:
        store.close()


def test_the_re_encode_skips_the_passage_without_tokens(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    before = _tf_v1_index(settings)

    status, out, err = _run(capsys, "reencode-sparse", "--yes")

    assert status == 0, err
    assert f"{SOURCES} passages written (1 skipped" in out
    after = dump_points(settings)
    encoder = BM25DocumentEncoder(1.2, 0.75, _average(before))
    for _, payload, vector in after:
        if payload["text"] == NO_TOKEN_TEXT:
            assert _sparse(vector) == {}
        else:
            assert _sparse(vector) == pytest.approx(encoder.encode(payload["text"]))
            assert _sparse(vector)
    metadata = _metadata(settings)
    assert metadata["sparse_encoder"] == "bm25-v2"
    run = SQLiteManifest(settings.data_dir).latest_run()
    assert run is not None and run.status == "completed"
    assert dict(run.totals) == {"passages": SOURCES + 1, "skipped_empty": 1}


def test_an_aborted_1_21_run_resumes_from_its_checkpoint(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    before = _tf_v1_index(settings)
    store = _store(settings)
    try:
        order: list[str] = []
        offset: str | None = None
        while True:
            page, offset = store.passage_text_page(offset, 1)
            order.extend(text for _, text in page)
            if offset is None:
                break
        head, following = store.passage_text_page(None, 2)
        metadata = store.read_completion_metadata()
        assert metadata is not None
        completed_at = str(metadata["completed_at"])
    finally:
        store.close()
    assert following is not None
    assert order.index(NO_TOKEN_TEXT) >= 2, "the checkpoint must precede it"
    recorded = 201.17
    stopped = record_aborted_1_21_run(
        SQLiteManifest(settings.data_dir),
        completed_at=completed_at,
        avgdl=recorded,
        total=SOURCES + 1,
        offset=following,
        done=2,
    )

    status, out, err = _run(capsys, "reencode-sparse", "--yes")

    assert status == 0, err
    assert "continues run" in out and "from 2 of" in out
    assert f"{SOURCES - 2} passages written (1 skipped" in out
    tf = TermFrequencyEncoder()
    encoder = BM25DocumentEncoder(1.2, 0.75, recorded)
    done = {text for _, text in head}
    assert done == set(order[:2])
    for _, payload, vector in dump_points(settings):
        text = payload["text"]
        if text == NO_TOKEN_TEXT:
            assert _sparse(vector) == {}
        elif text in done:
            # Before the checkpoint: not written again.
            assert _sparse(vector) == pytest.approx(tf.encode(text))
        else:
            assert _sparse(vector) == pytest.approx(encoder.encode(text))
    assert recorded != pytest.approx(_average(before))
    metadata = _metadata(settings)
    assert metadata["sparse_encoder"] == "bm25-v2"
    assert metadata["bm25_avgdl"] == recorded
    run = SQLiteManifest(settings.data_dir).latest_run()
    assert run is not None and run.status == "completed"
    assert (run.resumed_from, run.carried_sources) == (stopped, 2)
    assert dict(run.totals) == {"passages": SOURCES + 1, "skipped_empty": 1}


def test_the_store_drops_an_empty_vector_it_is_given(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    before = _tf_v1_index(settings)
    empty = next(p for p in before if p[1]["text"] == NO_TOKEN_TEXT)
    other = next(p for p in before if p[1]["text"] != NO_TOKEN_TEXT)
    store = _store(settings)
    try:
        with caplog.at_level(logging.DEBUG, logger="docgrep"):
            store.write_sparse_vectors(
                [(empty[1]["passage_id"], {}), (other[1]["passage_id"], {7: 0.5})]
            )
            store.write_sparse_vectors([(empty[1]["passage_id"], {})])
    finally:
        store.close()

    after = {p[0]: p for p in dump_points(settings)}
    assert _sparse(after[empty[0]][2]) == {}
    assert _sparse(after[other[0]][2]) == {7: 0.5}
    skipped = [
        r for r in caplog.records if r.getMessage() == "empty_sparse_vector_skipped"
    ]
    assert len(skipped) == 2
