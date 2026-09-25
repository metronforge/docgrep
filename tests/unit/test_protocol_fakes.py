"""In-memory boundary implementations and ranking behavior."""

from docgrep.adapters.models.deterministic import DeterministicEmbedder, LexicalReranker
from docgrep.domain.models import FileType, Passage, Source
from docgrep.domain.protocols import (
    EmbeddingProvider,
    ManifestReader,
    ManifestWriter,
    PassageRetriever,
    Reranker,
)
from fakes import InMemoryManifest, InMemoryPassageStore


def test_fakes_satisfy_runtime_protocols() -> None:
    embedder: EmbeddingProvider = DeterministicEmbedder()
    reranker: Reranker = LexicalReranker()
    retriever: PassageRetriever = InMemoryPassageStore()
    manifest = InMemoryManifest()
    reader: ManifestReader = manifest
    writer: ManifestWriter = manifest
    assert embedder.dimension == 64
    assert reranker.model_name
    assert retriever.read_completion_metadata() is None
    assert reader.latest_run() is None
    assert writer is manifest


def test_hybrid_retriever_uses_sparse_match_and_stable_ties() -> None:
    store = InMemoryPassageStore()
    source = Source("s", "/a.txt", FileType.TXT)
    for passage_id, dense, sparse in (
        ("b", (1.0, 0.0), {7: 1.0}),
        ("a", (1.0, 0.0), {7: 1.0}),
        ("c", (0.0, 1.0), {9: 2.0}),
    ):
        store.upsert(Passage(passage_id, "s", passage_id, 0, 1), source, dense, sparse)
    hits = store.retrieve((1.0, 0.0), {7: 1.0}, 3)
    assert [hit.passage.passage_id for hit in hits] == ["a", "b", "c"]
    assert [hit.dense_score for hit in hits] == [1.0, 1.0, 0.0]


def test_manifest_round_trip_and_completion_metadata() -> None:
    manifest = InMemoryManifest()
    run_id = manifest.start_run({"model": "fake"})
    source = Source("s", "/a.txt", FileType.TXT)
    manifest.record_source(source, "indexed")
    manifest.finish_run(run_id, "completed", {"indexed": 1})
    record = manifest.get_source("/a.txt")
    run = manifest.latest_run()
    assert record is not None and record.status == "indexed"
    assert run is not None and run.status == "completed"
    assert run.totals == (("indexed", 1),)

    store = InMemoryPassageStore()
    assert store.read_completion_metadata() is None
    store.write_completion_metadata({"embedding_model": "fake"})
    assert store.read_completion_metadata() == {"embedding_model": "fake"}


def test_ranked_passages_follow_required_tie_break() -> None:
    from docgrep.domain.models import RankedPassage
    from docgrep.domain.protocols import ranked_passage_key

    source = Source("s", "/a.txt", FileType.TXT)
    ranked = [
        RankedPassage(Passage(name, "s", name, 0, 1), source, 0.5, dense, rerank)
        for name, dense, rerank in (
            ("z", 0.9, 0.8),
            ("b", 0.8, 0.9),
            ("a", 0.8, 0.9),
        )
    ]
    assert [
        item.passage.passage_id for item in sorted(ranked, key=ranked_passage_key)
    ] == ["a", "b", "z"]
