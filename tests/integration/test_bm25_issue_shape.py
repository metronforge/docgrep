"""The shape of GitHub issue #20, end to end: a short passage with both
query terms is buried by long passages that repeat one term under tf-v1,
and rises to the top of the sparse channel after `docgrep reencode-sparse
--yes` (ADR-0025). Runs in local mode, and in server mode against
DOCGREP_TEST_QDRANT_URL."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from docgrep.adapters.models.deterministic import DeterministicEmbedder
from docgrep.adapters.models.sparse import TermFrequencyEncoder, UniqueTermEncoder
from docgrep.adapters.qdrant import QdrantPassageStore
from docgrep.api.cli import main
from docgrep.api.container import create_container
from docgrep.domain.protocols import SparseEncoder
from docgrep.settings import Settings
from resume_support import close, dump_points
from sparse_support import SHAPE_QUERY, SHAPE_SHORT, as_tf_v1_build, write_issue_shape

URL = os.environ.get("DOCGREP_TEST_QDRANT_URL")
ALIAS = "bm25shape"


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
        **(
            {"DOCGREP_QDRANT_URL": URL, "DOCGREP_QDRANT_ALIAS": ALIAS} if server else {}
        ),
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


def _sparse_rank(settings: Settings, encoder: SparseEncoder) -> int | None:
    """The 1-based rank of the short passage in the sparse channel alone."""
    store = (
        QdrantPassageStore(url=settings.qdrant_url, alias=settings.qdrant_alias)
        if settings.qdrant_url
        else QdrantPassageStore(settings.data_dir / "qdrant")
    )
    try:
        dense = DeterministicEmbedder().embed((SHAPE_QUERY,))[0]
        lists = store.probe_channels(dense, encoder.encode(SHAPE_QUERY), 200)
    finally:
        store.close()
    assert lists.sparse is not None
    sources = [hit.source_id for hit in lists.sparse]
    return sources.index(SHAPE_SHORT) + 1 if SHAPE_SHORT in sources else None


def test_bm25_lifts_the_short_passage_into_the_sparse_top_three(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    write_issue_shape(settings.corpus_root)
    services = create_container(settings)
    assert as_tf_v1_build(services.build).run().indexed == 131
    close(services)

    before = _sparse_rank(settings, TermFrequencyEncoder())
    assert before is None or before > 10
    points = dump_points(settings)

    assert main(["reencode-sparse"]) == 0
    assert _sparse_rank(settings, TermFrequencyEncoder()) == before
    assert main(["reencode-sparse", "--yes"]) == 0
    output = capsys.readouterr().out
    assert "current: tf-v1" in output and "131 passages written" in output

    after = _sparse_rank(settings, UniqueTermEncoder())
    assert after is not None and after <= 3
    # Only the sparse vectors changed: payloads and dense vectors are equal.
    rewritten = dump_points(settings)
    assert [(p[0], p[1], p[2]["dense"]) for p in rewritten] == [
        (p[0], p[1], p[2]["dense"]) for p in points
    ]
    assert [p[2]["sparse"] for p in rewritten] != [p[2]["sparse"] for p in points]
    # Search sees it too, through the query encoder of the recorded weighting.
    assert main(["search", SHAPE_QUERY, "--json", "--top-n", "3"]) == 0
    assert SHAPE_SHORT in capsys.readouterr().out
