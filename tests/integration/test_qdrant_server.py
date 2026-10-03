"""The Qdrant adapter against a real server (set DOCGREP_TEST_QDRANT_URL)."""

import os
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from docgrep.adapters.qdrant import QdrantPassageStore
from docgrep.api.container import create_container
from docgrep.domain.models import FileType, Passage, Source, make_passage_id
from docgrep.settings import Settings

URL = os.environ.get("DOCGREP_TEST_QDRANT_URL")

pytestmark = [
    pytest.mark.qdrant_server,
    pytest.mark.skipif(not URL, reason="DOCGREP_TEST_QDRANT_URL is not set"),
]


def _wipe(client: QdrantClient) -> None:
    for alias in client.get_aliases().aliases:
        client.update_collection_aliases(
            [
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name=alias.alias_name)
                )
            ]
        )
    for collection in client.get_collections().collections:
        if collection.name.startswith(("docgrep_", "alpha_", "beta_")):
            client.delete_collection(collection.name)


@pytest.fixture
def client() -> Iterator[QdrantClient]:
    assert URL is not None
    raw = QdrantClient(url=URL)
    _wipe(raw)
    yield raw
    _wipe(raw)
    raw.close()


def _store() -> QdrantPassageStore:
    return QdrantPassageStore(url=URL)


def _source(name: str = "notes.txt", author: str | None = "A. Reader") -> Source:
    return Source(name, name, FileType.TXT, title="Notes", author=author, year=2024)


def _passage(source: Source, text: str, offset: int) -> Passage:
    return Passage(
        make_passage_id(source.source_id, offset),
        source.source_id,
        text,
        offset,
        len(text.split()),
    )


def _build(
    store: QdrantPassageStore, model: str, texts: dict[str, tuple[float, float]]
) -> None:
    source = _source()
    store.recreate_collection(2)
    for offset, (text, vector) in enumerate(texts.items()):
        sparse = {777: 1.0} if "zebra" in text else {offset + 1: 1.0}
        store.upsert(_passage(source, text, offset), source, vector, sparse)
    store.write_completion_metadata({"embedding_model": model})


TEXTS = {
    "near one": (1.0, 0.0),
    "near two": (0.9, 0.435889894),
    "zebra lexical": (0.6, 0.8),
}


def test_build_and_hybrid_search_with_true_dense_scores(
    client: QdrantClient,
) -> None:
    store = _store()
    _build(store, "m1", TEXTS)

    assert store.read_completion_metadata() == {"embedding_model": "m1"}
    assert store.count_passages() == 3
    candidates = store.retrieve((1.0, 0.0), {777: 1.0}, 2)

    by_text = {item.passage.text: item for item in candidates}
    assert "zebra lexical" in by_text and "near two" not in by_text
    assert by_text["zebra lexical"].dense_score == pytest.approx(0.6, abs=1e-6)
    assert by_text["near one"].dense_score == pytest.approx(1.0, abs=1e-6)
    assert [item.stage1_rank for item in candidates] == [0, 1]


def test_alias_swap_keeps_the_previous_index_until_completion(
    client: QdrantClient,
) -> None:
    reader = _store()
    writer = _store()
    _build(writer, "old", {"old text": (1.0, 0.0)})
    first = {a.alias_name: a.collection_name for a in client.get_aliases().aliases}

    writer.recreate_collection(2)
    source = _source()
    writer.upsert(_passage(source, "new text", 0), source, (1.0, 0.0), {1: 1.0})
    during = {a.alias_name: a.collection_name for a in client.get_aliases().aliases}
    assert during == first
    assert reader.read_completion_metadata() == {"embedding_model": "old"}
    assert [c.passage.text for c in reader.retrieve((1.0, 0.0), {1: 1.0}, 5)] == [
        "old text"
    ]

    writer.write_completion_metadata({"embedding_model": "new"})

    after = {a.alias_name: a.collection_name for a in client.get_aliases().aliases}
    assert after["docgrep"] != first["docgrep"]
    assert reader.read_completion_metadata() == {"embedding_model": "new"}
    assert [c.passage.text for c in reader.retrieve((1.0, 0.0), {1: 1.0}, 5)] == [
        "new text"
    ]
    names = {c.name for c in client.get_collections().collections}
    assert first["docgrep"] not in names


def test_aborted_rebuild_preserves_the_previous_index(client: QdrantClient) -> None:
    store = _store()
    _build(store, "kept", TEXTS)
    before = {a.alias_name: a.collection_name for a in client.get_aliases().aliases}

    store.recreate_collection(2)
    source = _source()
    store.upsert(_passage(source, "half built", 0), source, (1.0, 0.0), {1: 1.0})
    store.clear_completion_metadata()

    assert {a.alias_name: a.collection_name for a in client.get_aliases().aliases} == (
        before
    )
    assert store.read_completion_metadata() == {"embedding_model": "kept"}
    assert store.count_passages() == 3
    staging = [
        c.name
        for c in client.get_collections().collections
        if c.name != before["docgrep"]
    ]
    assert staging == []


def test_payload_indexes_exist_and_filtered_counts_work(client: QdrantClient) -> None:
    store = _store()
    source_a = _source("a.txt", author="Ann")
    source_b = _source("b.txt", author="Bob")
    store.recreate_collection(2)
    for source, count in ((source_a, 2), (source_b, 1)):
        for offset in range(count):
            store.upsert(
                _passage(source, f"{source.path} {offset}", offset),
                source,
                (1.0, 0.0),
                {offset + 1: 1.0},
            )
    store.write_completion_metadata({"embedding_model": "m"})

    schema = client.get_collection("docgrep").payload_schema
    assert {
        "record_type",
        "source_id",
        "author",
        "year",
        "file_type",
        "path",
    } <= set(schema)
    assert schema["year"].data_type == models.PayloadSchemaType.INTEGER
    assert schema["source_id"].data_type == models.PayloadSchemaType.KEYWORD
    assert all(item.points >= 0 for item in schema.values())

    def count(source_id: str) -> int:
        return client.count(
            "docgrep",
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_id", match=models.MatchValue(value=source_id)
                    )
                ]
            ),
            exact=True,
        ).count

    assert (count("a.txt"), count("b.txt"), count("missing.txt")) == (2, 1, 0)


def test_two_clients_work_on_the_same_server_at_the_same_time(
    client: QdrantClient,
) -> None:
    server, other = _store(), _store()
    _build(server, "v1", TEXTS)
    errors: list[BaseException] = []
    seen: list[int] = []
    done = threading.Event()

    def reader() -> None:
        try:
            while not done.is_set():
                assert other.read_completion_metadata() is not None
                seen.append(len(other.retrieve((1.0, 0.0), {777: 1.0}, 5)))
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        _build(server, "v2", {f"text {i}": (1.0, 0.0) for i in range(40)})
    finally:
        done.set()
        thread.join(timeout=30)

    assert errors == []
    assert seen and min(seen) > 0
    assert other.read_completion_metadata() == {"embedding_model": "v2"}
    assert other.count_passages() == 40


def test_container_builds_and_searches_while_a_second_container_reads(
    tmp_path: Path, client: QdrantClient
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.txt").write_text("A rare blue comet crosses the sky.", "utf-8")

    def settings() -> Settings:
        return Settings(
            corpus_root=corpus,
            data_dir=tmp_path / "data",
            backend="fake",
            chunk_size=10,
            chunk_overlap=2,
            qdrant_url=URL,
        )

    server = create_container(settings())
    cli = create_container(settings())

    summary = server.build.run()
    result = cli.search.run("blue comet", top_n=1)

    assert (summary.indexed, summary.failed) == (1, 0)
    assert result.results[0].passage.text == "A rare blue comet crosses the sky."
    assert not (tmp_path / "data" / "qdrant").exists()


def _alias_targets(client: QdrantClient) -> dict[str, str]:
    return {a.alias_name: a.collection_name for a in client.get_aliases().aliases}


def test_two_aliases_on_one_server_build_and_search_independently(
    client: QdrantClient,
) -> None:
    alpha = QdrantPassageStore(url=URL, alias="alpha")
    beta = QdrantPassageStore(url=URL, alias="beta")
    _build(alpha, "model-a", {"alpha text": (1.0, 0.0)})
    _build(beta, "model-b", {"beta text": (1.0, 0.0), "beta more": (0.9, 0.435889894)})

    targets = _alias_targets(client)
    assert set(targets) == {"alpha", "beta"}
    assert targets["alpha"].startswith("alpha_") and targets["beta"].startswith("beta_")
    assert alpha.read_completion_metadata() == {"embedding_model": "model-a"}
    assert beta.read_completion_metadata() == {"embedding_model": "model-b"}
    assert alpha.count_passages() == 1 and beta.count_passages() == 2
    assert [c.passage.text for c in alpha.retrieve((1.0, 0.0), {1: 1.0}, 5)] == [
        "alpha text"
    ]
    assert {c.passage.text for c in beta.retrieve((1.0, 0.0), {1: 1.0}, 5)} == {
        "beta text",
        "beta more",
    }


def test_swapping_one_alias_leaves_the_other_untouched(client: QdrantClient) -> None:
    alpha = QdrantPassageStore(url=URL, alias="alpha")
    beta = QdrantPassageStore(url=URL, alias="beta")
    _build(alpha, "a1", {"alpha old": (1.0, 0.0)})
    _build(beta, "b1", {"beta only": (1.0, 0.0)})
    before = _alias_targets(client)

    _build(alpha, "a2", {"alpha new": (1.0, 0.0), "alpha extra": (0.9, 0.435889894)})

    after = _alias_targets(client)
    assert after["beta"] == before["beta"]
    assert after["alpha"] != before["alpha"]
    names = {c.name for c in client.get_collections().collections}
    assert before["alpha"] not in names and before["beta"] in names
    assert beta.read_completion_metadata() == {"embedding_model": "b1"}
    assert [c.passage.text for c in beta.retrieve((1.0, 0.0), {1: 1.0}, 5)] == [
        "beta only"
    ]
    assert alpha.count_passages() == 2
    assert alpha.read_completion_metadata() == {"embedding_model": "a2"}
