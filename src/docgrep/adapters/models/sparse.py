"""Unicode sparse term vectors for Qdrant's sparse IDF modifier (ADR-0025).

Every encoder shares one tokenizer (NFKC, casefold, runs of letters and
digits) and one stable hash of each token to an index, so the indices of a
passage never depend on the weighting; only the values do.
"""

import re
import unicodedata
from collections import Counter
from hashlib import blake2b

from docgrep.domain.protocols import SparseEncoder
from docgrep.domain.sparse import TF_V1, SparseWeighting, bm25_term_weight

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_INDEX_SPACE = 2**31


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(unicodedata.normalize("NFKC", text).casefold())


def _term_indices(text: str) -> list[int]:
    """The index of every token of the text, in order, repeats included."""
    return [
        int.from_bytes(blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")
        % _INDEX_SPACE
        for token in _tokens(text)
    ]


class TermFrequencyEncoder:
    """``tf-v1``: the count of each term, for documents and queries alike."""

    def encode(self, text: str) -> dict[int, float]:
        weights: dict[int, float] = {}
        for index in _term_indices(text):
            weights[index] = weights.get(index, 0.0) + 1.0
        return weights


# The name of the term-frequency encoder before the BM25 weighting existed.
BM25SparseEncoder = TermFrequencyEncoder


class BM25DocumentEncoder:
    """``bm25-v2`` passages: the BM25 term-frequency factor of each term.

    The length of a passage is its number of tokens under the shared
    tokenizer; the inverse document frequency is left to Qdrant.
    """

    def __init__(self, k1: float, b: float, avgdl: float) -> None:
        SparseWeighting.bm25(k1, b, avgdl)
        self.k1 = k1
        self.b = b
        self.avgdl = avgdl

    def encode(self, text: str) -> dict[int, float]:
        indices = _term_indices(text)
        length = len(indices)
        return {
            index: bm25_term_weight(tf, length, self.k1, self.b, self.avgdl)
            for index, tf in Counter(indices).items()
        }


class UniqueTermEncoder:
    """``bm25-v2`` queries: weight 1.0 for each distinct term."""

    def encode(self, text: str) -> dict[int, float]:
        return dict.fromkeys(_term_indices(text), 1.0)


class SparseEncoderSet:
    """The document and query encoders of every weighting."""

    def document_encoder(self, weighting: SparseWeighting) -> SparseEncoder:
        if weighting.encoder == TF_V1:
            return TermFrequencyEncoder()
        assert weighting.k1 is not None and weighting.b is not None
        assert weighting.avgdl is not None
        return BM25DocumentEncoder(weighting.k1, weighting.b, weighting.avgdl)

    def query_encoder(self, weighting: SparseWeighting) -> SparseEncoder:
        if weighting.encoder == TF_V1:
            return TermFrequencyEncoder()
        return UniqueTermEncoder()

    def document_length(self, text: str) -> int:
        return len(_tokens(text))
