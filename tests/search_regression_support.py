"""The public search body for a fixed programmatic corpus, recorded before
the diagnostic seam was added to ``FindPassages`` (ADR-0024).

``search_bodies.json`` next to this module holds the bodies produced on
``main`` at release 1.19.0; the regression test compares the current ones
with it byte for byte. Only values that differ between two runs of the same
code are masked: the elapsed time and the indexing timestamp.
"""

import json
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from docgrep.adapters.models.deterministic import DeterministicEmbedder
from docgrep.adapters.models.sparse import BM25SparseEncoder
from docgrep.api.container import create_container
from docgrep.api.http import SearchResponse, create_app
from docgrep.domain.errors import ModelUnavailable
from docgrep.domain.models import (
    FileType,
    Passage,
    RetrievalCandidate,
    Source,
    make_passage_id,
)
from docgrep.domain.protocols import Reranker
from docgrep.settings import Settings
from docgrep.usecases.find_passages import FindPassages
from resume_support import close
from search_filter_support import filter_settings, write_filter_library
from sparse_support import as_tf_v1_build

FIXTURE = Path(__file__).resolve().parent / "search_bodies.json"

# Requests to POST /search over the generated library.
HTTP_REQUESTS: tuple[dict[str, Any], ...] = (
    {"query": "comet orbit"},
    {"query": "comet orbit", "top_n": 2},
    {"query": "comet orbit", "collapse_duplicates": False},
    {"query": "orbit tables"},
    {"query": "orbit tables", "collapse_duplicates": False},
    {"query": "comet", "authors": ["arnold"]},
    {"query": "comet", "file_types": ["txt"]},
    {"query": "comet", "year_from": 1980, "year_to": 2001},
    {"query": "comet", "authors": ["nobody"]},
    {"query": "unheard-of zzzz"},
)

_MTIME = 1_700_000_000
_TABLES = "Orbit tables for the comet of 1910, copied from the observatory log."
_MASKS = (
    (re.compile(rb'"took_ms":[-+0-9.eE]+'), b'"took_ms":0'),
    (re.compile(rb'"indexed_at":"[^"]*"'), b'"indexed_at":"<masked>"'),
)


def mask(body: bytes) -> bytes:
    for pattern, replacement in _MASKS:
        body = pattern.sub(replacement, body)
    return body


def write_library(corpus: Path) -> None:
    """The filter library plus two files with the same text, all with one
    fixed modification time."""
    write_filter_library(corpus)
    for name in ("copies/a.txt", "copies/b.txt"):
        path = corpus / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_TABLES + "\n", encoding="utf-8")
    for path in corpus.rglob("*"):
        if path.is_file():
            os.utime(path, (_MTIME, _MTIME))


@contextmanager
def http_client(root: Path) -> Iterator[TestClient]:
    write_library(root / "corpus")
    settings: Settings = filter_settings(root)
    services = create_container(settings)
    # The bodies were recorded on a tf-v1 index (ADR-0025).
    summary = as_tf_v1_build(services.build).run()
    assert (summary.indexed, summary.failed) == (18, 0), summary
    try:
        yield TestClient(
            create_app(settings, services=services), base_url="http://127.0.0.1"
        )
    finally:
        close(services)


def http_bodies(api: TestClient) -> list[str]:
    bodies = []
    for request in HTTP_REQUESTS:
        response = api.post("/search", json=request)
        assert response.status_code == 200, response.text
        bodies.append(mask(response.content).decode("utf-8"))
    return bodies


# The use case alone, with scripted first-stage candidates.

_METADATA: dict[str, str | int | float] = {
    "embedding_model": "deterministic-hash-v1",
    "embedding_dimension": 64,
}


def _source(path: str, author: str | None = None) -> Source:
    return Source(path, path, FileType.TXT, title="T", author=author, year=1999)


def _candidate(path: str, offset: int, text: str, dense: float) -> RetrievalCandidate:
    source = _source(path)
    passage = Passage(
        make_passage_id(path, offset), path, text, offset, len(text.split()), None, ()
    )
    return RetrievalCandidate(passage, source, dense)


CANDIDATES: tuple[RetrievalCandidate, ...] = tuple(
    RetrievalCandidate(item.passage, item.source, item.dense_score, stage1_rank=rank)
    for rank, item in enumerate(
        (
            _candidate("a.txt", 0, "comet orbit near the sun", 0.9),
            _candidate("b.txt", 0, "comet orbit near the sun", 0.9),
            _candidate("c.txt", 0, "an orbit of a comet", 0.5),
            _candidate("d.txt", 0, "nothing relevant here", 0.7),
            _candidate("e.txt", 4, "comet", 0.2),
            _candidate("e.txt", 9, "orbit", 0.2),
        )
    )
)


class StaticRetriever:
    def retrieve(
        self,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float],
        top_k: int,
        source_ids: Any = None,
    ) -> tuple[RetrievalCandidate, ...]:
        return CANDIDATES[:top_k]

    def read_completion_metadata(self) -> Mapping[str, str | int | float] | None:
        return _METADATA

    def count_passages(self) -> int:
        return len(CANDIDATES)


class ScriptedReranker:
    model_name = "scripted-reranker"
    _SCORES = {
        "comet orbit near the sun": 2.5,
        "an orbit of a comet": 2.5,
        "nothing relevant here": -1.0,
        "comet": 0.0,
        "orbit": 0.0,
    }

    def score(self, query: str, passages: Sequence[str]) -> tuple[float, ...]:
        return tuple(self._SCORES[text] for text in passages)


class FailingReranker:
    model_name = "broken-reranker"

    def score(self, query: str, passages: Sequence[str]) -> tuple[float, ...]:
        raise ModelUnavailable("reranker unavailable")


def use_case(reranker: Reranker, *, top_k: int = 6) -> FindPassages:
    return FindPassages(
        embedder=DeterministicEmbedder(),
        sparse_encoder=BM25SparseEncoder(),
        retriever=StaticRetriever(),
        reranker=reranker,
        top_k=top_k,
        default_top_n=3,
        top_n_max=10,
        max_query_len=100,
        now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )


def use_case_bodies() -> list[str]:
    runs: list[tuple[Reranker, int, dict[str, Any]]] = [
        (ScriptedReranker(), 6, {}),
        (ScriptedReranker(), 6, {"top_n": 10}),
        (ScriptedReranker(), 6, {"top_n": 10, "collapse_duplicates": False}),
        (ScriptedReranker(), 3, {"top_n": 10}),
        (FailingReranker(), 6, {"top_n": 10}),
        (FailingReranker(), 6, {"top_n": 10, "collapse_duplicates": False}),
    ]
    bodies = []
    for reranker, top_k, options in runs:
        result = use_case(reranker, top_k=top_k).run("comet orbit", **options)
        body = SearchResponse.from_domain(result).model_dump_json()
        bodies.append(mask(body.encode("utf-8")).decode("utf-8"))
    return bodies


def load_fixture() -> dict[str, list[str]]:
    data: dict[str, list[str]] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data
