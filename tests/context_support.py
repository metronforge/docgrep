"""Passages of a built library for the passage context tests (UC-12)."""

from pathlib import Path

from docgrep.adapters.qdrant import QdrantPassageStore
from docgrep.domain.models import Passage, make_passage_id
from docgrep.settings import Settings

MISSING = make_passage_id("gone.txt", 0)


def source_passages(settings: Settings, source_id: str) -> list[Passage]:
    """The published passages of one source in reading order; the store is
    closed again, so another process or client can open it."""
    store = QdrantPassageStore(Path(settings.data_dir) / "qdrant")
    try:
        outline = store.source_outline(source_id)
        texts = store.passage_texts({entry.passage_id for entry in outline})
    finally:
        store.close()
    return [texts[entry.passage_id] for entry in outline]
