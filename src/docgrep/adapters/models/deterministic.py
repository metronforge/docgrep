"""Stable ML-free model adapters for development and tests."""

import re
from collections.abc import Sequence
from hashlib import blake2b
from math import sqrt

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


class DeterministicEmbedder:
    """Hash token counts into a fixed-size, normalized dense vector."""

    model_name = "deterministic-hash-v1"
    dimension = 64

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        vectors: list[tuple[float, ...]] = []
        for text in texts:
            values = [0.0] * self.dimension
            for token in _tokens(text):
                digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest, "big") % self.dimension
                values[index] += 1.0
            magnitude = sqrt(sum(value * value for value in values))
            vectors.append(
                tuple(value / magnitude for value in values)
                if magnitude
                else tuple(values)
            )
        return tuple(vectors)


class LexicalReranker:
    """Score passages by the share of distinct query tokens they contain."""

    model_name = "lexical-overlap-v1"

    def score(self, query: str, passages: Sequence[str]) -> tuple[float, ...]:
        query_tokens = set(_tokens(query))
        if not query_tokens:
            return (0.0,) * len(passages)
        return tuple(
            len(query_tokens & set(_tokens(passage))) / len(query_tokens)
            for passage in passages
        )
