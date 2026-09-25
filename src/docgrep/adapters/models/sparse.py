"""Unicode term-frequency vectors for Qdrant's sparse IDF modifier."""

import re
import unicodedata
from hashlib import blake2b

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_INDEX_SPACE = 2**31


class BM25SparseEncoder:
    """Map normalized word counts to stable nonnegative sparse indices."""

    def encode(self, text: str) -> dict[int, float]:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        weights: dict[int, float] = {}
        for token in _TOKEN.findall(normalized):
            digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % _INDEX_SPACE
            weights[index] = weights.get(index, 0.0) + 1.0
        return weights
