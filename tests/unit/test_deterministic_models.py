"""Behavior of the CPU-only model adapters."""

import json
import os
import subprocess
import sys

from docgrep.adapters.models.deterministic import DeterministicEmbedder, LexicalReranker


def test_embeddings_are_stable_across_hash_seeds() -> None:
    script = (
        "import json; from docgrep.adapters.models.deterministic import "
        "DeterministicEmbedder; "
        "print(json.dumps(DeterministicEmbedder().embed(['alpha beta'])))"
    )
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
        )
        for seed in ("1", "137")
    ]
    assert json.loads(outputs[0]) == json.loads(outputs[1])
    assert len(DeterministicEmbedder().embed(["alpha"])[0]) == 64


def test_embedder_distinguishes_text_and_preserves_batch_order() -> None:
    embedder = DeterministicEmbedder()
    first, second, repeat = embedder.embed(["alpha", "beta", "alpha"])
    assert first != second
    assert first == repeat


def test_reranker_scores_lexical_overlap() -> None:
    reranker = LexicalReranker()
    scores = reranker.score("blue river", ["blue river", "blue mountain", "red desert"])
    assert scores == (1.0, 0.5, 0.0)
