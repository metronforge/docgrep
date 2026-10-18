"""Reuse by content hash against a real Qdrant server (set
DOCGREP_TEST_QDRANT_URL): vectors are copied, not recomputed (ADR-0028)."""

import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient, models

from docgrep.api.container import create_container
from resume_support import close, dump_points, settings_for

URL = os.environ.get("DOCGREP_TEST_QDRANT_URL")

pytestmark = [
    pytest.mark.qdrant_server,
    pytest.mark.skipif(not URL, reason="DOCGREP_TEST_QDRANT_URL is not set"),
]

ALIAS = "reuse"
MTIME = 1_700_000_000
FILES = {
    "a.txt": "A rare blue comet crosses the winter sky above the harbour at "
    "night, and the keeper of the lighthouse writes it down in his ledger.",
    "b.txt": "Recipe for bread with rye flour and caraway seeds, baked slowly "
    "in a stone oven until the crust is dark.",
    "c.txt": "Notes on a walk along the river after the rain.",
}


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


@pytest.fixture(autouse=True)
def server() -> Iterator[QdrantClient]:
    assert URL is not None
    client = QdrantClient(url=URL)
    _wipe(client)
    yield client
    _wipe(client)
    client.close()


class _Embeds:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.texts: list[str] = []

    def embed(self, texts: Sequence[str]) -> Any:
        self.texts.extend(texts)
        return self.inner.embed(texts)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def _by_offset(points: Any, path: str) -> dict[int, tuple[str, Any]]:
    return {
        payload["chunk_offset"]: (point_id, vectors)
        for point_id, payload, vectors in points
        if payload["path"] == path
    }


def test_a_move_and_a_copy_copy_the_stored_vectors(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, text in FILES.items():
        (corpus / name).write_text(text, encoding="utf-8")
        os.utime(corpus / name, (MTIME, MTIME))
    settings = settings_for(corpus, tmp_path / "data", url=URL, alias=ALIAS)
    services = create_container(settings)
    services.build.run()
    close(services)
    before = dump_points(settings)
    (corpus / "moved").mkdir()
    (corpus / "a.txt").rename(corpus / "moved" / "a.txt")
    (corpus / "copy.txt").write_text(FILES["b.txt"], encoding="utf-8")

    services = create_container(settings)
    update: Any = services.update
    embedder = update.embedder = _Embeds(update.embedder)
    summary = services.update.run()
    close(services)
    after = dump_points(settings)

    assert embedder.texts == []
    assert (summary.moved, summary.copied, summary.failed) == (1, 1, 0)
    assert not _by_offset(after, "a.txt")
    for old, new in (("a.txt", "moved/a.txt"), ("b.txt", "copy.txt")):
        original, copy = _by_offset(before, old), _by_offset(after, new)
        assert original and sorted(original) == sorted(copy)
        for offset, (point_id, vectors) in copy.items():
            assert point_id != original[offset][0]
            # Exactly the stored vectors: nothing was computed again.
            assert vectors == original[offset][1]
    # The copied original is untouched.
    assert _by_offset(after, "b.txt") == _by_offset(before, "b.txt")
