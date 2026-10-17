"""BM25 sparse weights and the weighting recorded in the index (ADR-0025)."""

import math

import pytest

from docgrep.adapters.models.sparse import (
    BM25DocumentEncoder,
    BM25SparseEncoder,
    SparseEncoderSet,
    TermFrequencyEncoder,
    UniqueTermEncoder,
)
from docgrep.domain.sparse import (
    BM25_V2,
    TF_V1,
    SparseWeighting,
    bm25_term_weight,
)


def _weight(encoder: BM25DocumentEncoder, text: str, term: str) -> float:
    (index,) = TermFrequencyEncoder().encode(term)
    return encoder.encode(text)[index]


def test_bm25_weights_match_the_formula_by_hand() -> None:
    encoder = BM25DocumentEncoder(1.2, 0.75, 4.0)
    # "a" twice and "b" once in a passage of 3 tokens, average length 4:
    # norm = 1 - 0.75 + 0.75 * 3 / 4 = 0.8125
    a = 2 * 2.2 / (2 + 1.2 * 0.8125)
    b = 1 * 2.2 / (1 + 1.2 * 0.8125)

    assert _weight(encoder, "a A, b", "a") == pytest.approx(a)
    assert _weight(encoder, "a A, b", "b") == pytest.approx(b)
    assert len(encoder.encode("a A, b")) == 2


def test_term_frequency_saturates() -> None:
    encoder = BM25DocumentEncoder(1.2, 0.75, 31.0)
    once = _weight(encoder, "go " + "x " * 30, "go")
    thirty = _weight(encoder, "go " * 30 + "x", "go")

    assert once < thirty < 30 * once
    # The factor never exceeds k1 + 1.
    assert thirty < 2.2


def test_the_same_count_weighs_more_in_a_shorter_passage() -> None:
    encoder = BM25DocumentEncoder(1.2, 0.75, 20.0)
    short = _weight(encoder, "context engineering", "context")
    long = _weight(encoder, "context " + "filler " * 60, "context")

    assert short > long


def test_without_length_normalization_the_length_does_not_matter() -> None:
    encoder = BM25DocumentEncoder(1.2, 0.0, 20.0)

    assert _weight(encoder, "go", "go") == pytest.approx(
        _weight(encoder, "go " + "x " * 99, "go")
    )


def test_query_terms_weigh_one_each_and_repeats_count_once() -> None:
    query = UniqueTermEncoder().encode("Context context ENGINEERING")

    assert sorted(query.values()) == [1.0, 1.0]


def test_every_encoder_produces_the_indices_of_the_term_frequency_encoder() -> None:
    text = "River river, РЕКА река; bank ﬁve café"
    counts = TermFrequencyEncoder().encode(text)

    assert BM25SparseEncoder is TermFrequencyEncoder
    assert set(BM25DocumentEncoder(1.2, 0.75, 7.0).encode(text)) == set(counts)
    assert set(UniqueTermEncoder().encode(text)) == set(counts)
    assert SparseEncoderSet().document_length(text) == sum(counts.values())


def test_the_encoder_set_follows_the_weighting() -> None:
    encoders = SparseEncoderSet()
    tf = SparseWeighting()
    bm25 = SparseWeighting.bm25(1.2, 0.75, 3.0)

    assert encoders.document_encoder(tf).encode("a a") == {
        next(iter(TermFrequencyEncoder().encode("a"))): 2.0
    }
    assert encoders.query_encoder(tf).encode("a a") == TermFrequencyEncoder().encode(
        "a a"
    )
    assert encoders.query_encoder(bm25).encode("a a") == UniqueTermEncoder().encode("a")
    weights = encoders.document_encoder(bm25).encode("a a")
    assert list(weights.values()) == [
        pytest.approx(bm25_term_weight(2, 2, 1.2, 0.75, 3.0))
    ]


def test_an_index_without_the_key_is_tf_v1() -> None:
    assert SparseWeighting.from_metadata(None) == SparseWeighting()
    assert SparseWeighting.from_metadata({"embedding_model": "m"}).encoder == TF_V1
    assert SparseWeighting.from_metadata({"sparse_encoder": "tf-v1"}).encoder == TF_V1


def test_bm25_weighting_round_trips_through_the_metadata() -> None:
    weighting = SparseWeighting.bm25(1.5, 0.5, 211.25)
    metadata = weighting.metadata()

    assert metadata == {
        "sparse_encoder": BM25_V2,
        "bm25_k1": 1.5,
        "bm25_b": 0.5,
        "bm25_avgdl": 211.25,
    }
    assert SparseWeighting.from_metadata(metadata) == weighting
    assert SparseWeighting().metadata() == {"sparse_encoder": TF_V1}
    assert weighting.same_parameters(1.5, 0.5)
    assert not weighting.same_parameters(1.2, 0.5)
    assert not SparseWeighting().same_parameters(1.5, 0.5)


@pytest.mark.parametrize(
    "metadata",
    [
        {"sparse_encoder": "bm99"},
        {"sparse_encoder": "bm25-v2", "bm25_k1": 1.2, "bm25_b": 0.75},
        {"sparse_encoder": "bm25-v2", "bm25_k1": 1.2, "bm25_b": 2, "bm25_avgdl": 9},
        {"sparse_encoder": "bm25-v2", "bm25_k1": 1.2, "bm25_b": 0, "bm25_avgdl": 0},
        {"sparse_encoder": "bm25-v2", "bm25_k1": -1, "bm25_b": 0, "bm25_avgdl": 9},
    ],
)
def test_an_invalid_recorded_weighting_is_refused(
    metadata: dict[str, str | int | float],
) -> None:
    with pytest.raises(ValueError):
        SparseWeighting.from_metadata(metadata)


def test_tf_v1_takes_no_parameters() -> None:
    with pytest.raises(ValueError):
        SparseWeighting(TF_V1, 1.2, 0.75, 3.0)
    assert math.isclose(bm25_term_weight(1, 4, 1.2, 0.75, 4.0), 1.0)
