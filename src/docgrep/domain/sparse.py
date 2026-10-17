"""Which weighting the sparse vectors of an index carry (ADR-0025).

The weighting is index metadata: every passage of one collection is encoded
with the weighting its completion marker records, and queries are encoded to
match it. Qdrant applies the inverse document frequency on its side, so only
the term-frequency part is decided here.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

# Raw term counts, the only weighting before release 1.21.0.
TF_V1: Final = "tf-v1"
# BM25 term weights with saturation (k1) and length normalization (b).
BM25_V2: Final = "bm25-v2"
SparseEncoderName = Literal["tf-v1", "bm25-v2"]
SPARSE_ENCODERS: tuple[SparseEncoderName, ...] = (TF_V1, BM25_V2)

DEFAULT_K1 = 1.2
DEFAULT_B = 0.75
# Written while a full build runs and corrected by the re-encode that ends
# it, so its value never reaches a published index.
PROVISIONAL_AVGDL = 256.0

# The completion metadata keys of the weighting.
SPARSE_ENCODER_KEY = "sparse_encoder"
BM25_K1_KEY = "bm25_k1"
BM25_B_KEY = "bm25_b"
BM25_AVGDL_KEY = "bm25_avgdl"
SPARSE_METADATA_KEYS = (SPARSE_ENCODER_KEY, BM25_K1_KEY, BM25_B_KEY, BM25_AVGDL_KEY)


@dataclass(frozen=True, slots=True)
class SparseWeighting:
    """The sparse encoder of an index and, for BM25, its parameters.

    ``tf-v1`` has no parameters. ``bm25-v2`` needs ``k1`` >= 0, ``b`` in
    [0, 1] and a positive ``avgdl`` (average passage length in the tokens of
    the sparse tokenizer).
    """

    encoder: SparseEncoderName = TF_V1
    k1: float | None = None
    b: float | None = None
    avgdl: float | None = None

    def __post_init__(self) -> None:
        if self.encoder == TF_V1:
            if (self.k1, self.b, self.avgdl) != (None, None, None):
                raise ValueError("tf-v1 takes no parameters")
            return
        if self.encoder != BM25_V2:
            raise ValueError(f"unknown sparse encoder {self.encoder!r}")
        check_bm25_parameters(self.k1, self.b)
        if self.avgdl is None or not self.avgdl > 0:
            raise ValueError("bm25 avgdl must be positive")

    @classmethod
    def bm25(cls, k1: float, b: float, avgdl: float) -> "SparseWeighting":
        return cls(BM25_V2, float(k1), float(b), float(avgdl))

    @classmethod
    def from_metadata(
        cls, metadata: Mapping[str, str | int | float] | None
    ) -> "SparseWeighting":
        """The weighting a completion marker records; an index without the
        key (or without a marker) is ``tf-v1``. Raises ``ValueError`` for a
        recorded weighting that is not valid."""
        if metadata is None:
            return cls()
        encoder = metadata.get(SPARSE_ENCODER_KEY, TF_V1)
        if encoder == TF_V1:
            return cls()
        if encoder != BM25_V2:
            raise ValueError(f"unknown sparse encoder {encoder!r}")
        k1 = metadata.get(BM25_K1_KEY)
        b = metadata.get(BM25_B_KEY)
        avgdl = metadata.get(BM25_AVGDL_KEY)
        if (
            not isinstance(k1, int | float)
            or not isinstance(b, int | float)
            or not isinstance(avgdl, int | float)
        ):
            raise ValueError("bm25 parameters are missing from the index metadata")
        return cls.bm25(k1, b, avgdl)

    def metadata(self) -> dict[str, str | int | float]:
        """The completion metadata keys of this weighting; ``tf-v1`` records
        only its name."""
        if self.encoder == TF_V1:
            return {SPARSE_ENCODER_KEY: TF_V1}
        assert self.k1 is not None and self.b is not None and self.avgdl is not None
        return {
            SPARSE_ENCODER_KEY: BM25_V2,
            BM25_K1_KEY: self.k1,
            BM25_B_KEY: self.b,
            BM25_AVGDL_KEY: self.avgdl,
        }

    def same_parameters(self, k1: float, b: float) -> bool:
        """Whether this is ``bm25-v2`` with these saturation and length
        parameters, whatever its average length."""
        return self.encoder == BM25_V2 and (self.k1, self.b) == (k1, b)


def check_bm25_parameters(k1: float | None, b: float | None) -> None:
    """Raise ``ValueError`` unless ``k1`` >= 0 and 0 <= ``b`` <= 1."""
    if k1 is None or not k1 >= 0:
        raise ValueError("bm25 k1 must be zero or positive")
    if b is None or not 0 <= b <= 1:
        raise ValueError("bm25 b must be between 0 and 1")


def bm25_term_weight(tf: int, length: int, k1: float, b: float, avgdl: float) -> float:
    """The BM25 term-frequency factor of a term ``tf`` times in a passage of
    ``length`` tokens: tf·(k1+1) / (tf + k1·(1 − b + b·length/avgdl))."""
    return tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avgdl))
