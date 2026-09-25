"""Stable term-frequency sparse encoding for documents and queries."""

import json
import os
import subprocess
import sys

from docgrep.adapters.models.sparse import BM25SparseEncoder
from docgrep.domain.protocols import SparseEncoder


def test_term_frequencies_and_shared_query_indices() -> None:
    encoder: SparseEncoder = BM25SparseEncoder()
    document = encoder.encode("River river, РЕКА река; bank")
    query = encoder.encode("река RIVER")

    assert len(document) == 3
    assert len(query) == 2
    assert all(0 <= index < 2**31 for index in document)
    assert all(document[index] == 2.0 for index in query)
    assert encoder.encode("bank").items() <= document.items()


def test_unicode_normalization_and_token_boundaries() -> None:
    encoder = BM25SparseEncoder()

    assert encoder.encode("Ｃａｆｅ́") == encoder.encode("café")
    assert encoder.encode("Привет, мир!") == encoder.encode("привет МИР")
    assert encoder.encode("foo_bar") == encoder.encode("foo bar")
    assert encoder.encode("  !?") == {}


def test_sparse_indices_are_stable_across_hash_seeds() -> None:
    script = (
        "import json; from docgrep.adapters.models.sparse import "
        "BM25SparseEncoder; "
        "print(json.dumps(BM25SparseEncoder().encode('river река river'), "
        "sort_keys=True))"
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
