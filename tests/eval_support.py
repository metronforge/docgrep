"""A programmatic index whose passages rank differently in the dense, the
sparse and the fused first-stage lists (ADR-0024).

The query vector is (1, 0) and the query has one sparse term. Dense
similarity falls with the angle of each passage vector, so the dense order
is a, b, c, d, e, f; the sparse weights of the term order the sparse list
f, c, e, b; a and d lack the term.
"""

import math
from collections.abc import Sequence
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from docgrep.adapters.qdrant import QdrantPassageStore
from docgrep.domain.evaluation import ProbeHit
from docgrep.domain.models import FileType, Passage, Source, make_passage_id

# The alias the server-mode tests own (see qdrant_server_guard).
ALIAS = "alpha"
TERM = 7
QUERY_DENSE = (1.0, 0.0)
QUERY_SPARSE = {TERM: 1.0}
# Text, source, angle in degrees and the weight of TERM (0: absent).
LAYOUT: tuple[tuple[str, str, int, float], ...] = (
    ("a alpha", "one.txt", 0, 0.0),
    ("b beta", "two.txt", 10, 1.0),
    ("c gamma", "one.txt", 20, 3.0),
    ("d delta", "three.txt", 30, 0.0),
    ("e epsilon", "two.txt", 40, 2.0),
    ("f zeta", "three.txt", 50, 4.0),
)
DENSE_ORDER = ("a alpha", "b beta", "c gamma", "d delta", "e epsilon", "f zeta")
SPARSE_ORDER = ("f zeta", "c gamma", "e epsilon", "b beta")


def source(name: str) -> Source:
    return Source(name, name, FileType.TXT)


def passage_of(text: str) -> Passage:
    offset = next(i for i, item in enumerate(LAYOUT) if item[0] == text)
    name = LAYOUT[offset][1]
    return Passage(make_passage_id(name, offset), name, text, offset, 2)


def build(store: QdrantPassageStore) -> None:
    store.recreate_collection(2)
    for text, name, angle, weight in LAYOUT:
        radians = math.radians(angle)
        vector = (math.cos(radians), math.sin(radians))
        # A passage without the term still needs one sparse entry of its own.
        sparse = {TERM: weight} if weight else {100 + len(text): 1.0}
        store.upsert(passage_of(text), source(name), vector, sparse)
    store.write_completion_metadata({"embedding_model": "m", "embedding_dimension": 2})


def dense_only_index(path: Path, alias: str = "docgrep") -> None:
    """A published local collection without a sparse channel."""
    client = QdrantClient(path=str(path))
    try:
        name = f"{alias}_{'0' * 32}"
        client.create_collection(
            name,
            vectors_config={
                "dense": models.VectorParams(size=2, distance=models.Distance.COSINE)
            },
        )
        points = []
        for text, name_, angle, _ in LAYOUT:
            passage = passage_of(text)
            radians = math.radians(angle)
            points.append(
                models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, passage.passage_id)),
                    vector={"dense": [math.cos(radians), math.sin(radians)]},
                    payload={
                        "record_type": "passage",
                        "passage_id": passage.passage_id,
                        "source_id": name_,
                        "text": text,
                    },
                )
            )
        client.upsert(name, points=points)
        client.update_collection_aliases(
            [
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=name, alias_name=alias
                    )
                )
            ]
        )
    finally:
        client.close()


def texts(hits: Sequence[ProbeHit] | None) -> list[str]:
    by_id = {passage_of(text).passage_id: text for text, *_ in LAYOUT}
    return [by_id[hit.passage_id] for hit in hits or ()]


def wipe(client: QdrantClient) -> None:
    """Remove the suite's alias and collections from a test server."""
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
