"""Lazy local embedding and cross-encoder adapters."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from importlib import import_module
from math import isfinite
from typing import Protocol, cast

from docgrep.domain.errors import ModelUnavailable


class OffsetTokenizer(Protocol):
    """Expose tokenizer offsets without binding the domain to Transformers."""

    def __call__(
        self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool
    ) -> Mapping[str, object]: ...


class SentenceModel(Protocol):
    """Sentence Transformer operations used for embedding and chunking."""

    @property
    def tokenizer(self) -> OffsetTokenizer: ...

    def get_sentence_embedding_dimension(self) -> int | None: ...

    def encode(
        self,
        texts: Sequence[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> Iterable[Iterable[float]]: ...


class CrossModel(Protocol):
    """Cross Encoder operation used for reranking."""

    def predict(
        self, pairs: Sequence[tuple[str, str]], *, show_progress_bar: bool
    ) -> Iterable[float]: ...


def _load_sentence_model(name: str, device: str) -> SentenceModel:
    model_type = import_module("sentence_transformers").SentenceTransformer
    return cast(SentenceModel, model_type(name, device=device))


def _load_cross_model(name: str, device: str) -> CrossModel:
    model_type = import_module("sentence_transformers").CrossEncoder
    return cast(CrossModel, model_type(name, device=device))


def _torch_cuda_available() -> bool:
    return bool(import_module("torch").cuda.is_available())


def _device(setting: str, cuda_available: Callable[[], bool]) -> str:
    if setting != "auto":
        return setting
    return "cuda" if cuda_available() else "cpu"


def _unavailable(name: str, error: Exception) -> ModelUnavailable:
    return ModelUnavailable(
        f"Model {name} is unavailable",
        details={"model": name, "reason": str(error)},
    )


class ModelTokenizer:
    """Convert Hugging Face offset mappings into chunker token spans."""

    def __init__(self, tokenizer: OffsetTokenizer, model_name: str) -> None:
        self._tokenizer = tokenizer
        self._model_name = model_name

    def token_spans(self, text: str) -> tuple[tuple[int, int], ...]:
        """Return ordered, disjoint, non-empty source-character spans.

        Empty spans (special tokens) are dropped. Tokens that cover the same
        characters, such as a lone word-boundary piece followed by the word
        piece that starts at the same offset, or the byte-fallback pieces of
        one emoji, are merged into one token spanning their union.
        """
        try:
            offsets = self._tokenizer(
                text, add_special_tokens=False, return_offsets_mapping=True
            )["offset_mapping"]
            if not isinstance(offsets, Sequence):
                raise ValueError("tokenizer returned no offset sequence")
            spans: list[tuple[int, int]] = []
            for offset in offsets:
                if (
                    not isinstance(offset, (tuple, list))
                    or len(offset) != 2
                    or not isinstance(offset[0], int)
                    or not isinstance(offset[1], int)
                ):
                    raise ValueError("tokenizer returned invalid offsets")
                start, end = offset
                if start >= end:
                    continue
                merged = (start, end)
                while spans and merged[0] < spans[-1][1]:
                    previous = spans.pop()
                    merged = (min(previous[0], merged[0]), max(previous[1], merged[1]))
                spans.append(merged)
            return tuple(spans)
        except Exception as error:
            raise _unavailable(self._model_name, error) from error


class BgeEmbedder:
    """Load a Sentence Transformer on first use and return normalized vectors."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        loader: Callable[[str, str], SentenceModel] | None = None,
        cuda_available: Callable[[], bool] | None = None,
    ) -> None:
        self.model_name = model_name
        self._device = device
        self._loader = loader or _load_sentence_model
        self._cuda_available = cuda_available or _torch_cuda_available
        self._model: SentenceModel | None = None

    def _get_model(self) -> SentenceModel:
        if self._model is None:
            try:
                self._model = self._loader(
                    self.model_name, _device(self._device, self._cuda_available)
                )
            except Exception as error:
                raise _unavailable(self.model_name, error) from error
        return self._model

    @property
    def dimension(self) -> int:
        """Return the loaded model's fixed embedding dimension."""
        try:
            dimension = self._get_model().get_sentence_embedding_dimension()
            if dimension is None or dimension <= 0:
                raise ValueError("model has no embedding dimension")
            return dimension
        except ModelUnavailable:
            raise
        except Exception as error:
            raise _unavailable(self.model_name, error) from error

    @property
    def tokenizer(self) -> ModelTokenizer:
        """Expose the embedding model tokenizer for token-aware chunking."""
        try:
            return ModelTokenizer(self._get_model().tokenizer, self.model_name)
        except ModelUnavailable:
            raise
        except Exception as error:
            raise _unavailable(self.model_name, error) from error

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        """Embed text in input order with unit-length vectors."""
        if not texts:
            return ()
        try:
            vectors = tuple(
                tuple(float(value) for value in vector)
                for vector in self._get_model().encode(
                    texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            )
            dimension = self.dimension
            if len(vectors) != len(texts) or any(
                len(vector) != dimension or not all(isfinite(value) for value in vector)
                for vector in vectors
            ):
                raise ValueError("model returned invalid embeddings")
            return vectors
        except ModelUnavailable:
            raise
        except Exception as error:
            raise _unavailable(self.model_name, error) from error


class BgeReranker:
    """Load a Cross Encoder on first use and return raw relevance scores."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        loader: Callable[[str, str], CrossModel] | None = None,
        cuda_available: Callable[[], bool] | None = None,
    ) -> None:
        self.model_name = model_name
        self._device = device
        self._loader = loader or _load_cross_model
        self._cuda_available = cuda_available or _torch_cuda_available
        self._model: CrossModel | None = None

    def _get_model(self) -> CrossModel:
        if self._model is None:
            try:
                self._model = self._loader(
                    self.model_name, _device(self._device, self._cuda_available)
                )
            except Exception as error:
                raise _unavailable(self.model_name, error) from error
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> tuple[float, ...]:
        """Score each query-passage pair in input order."""
        if not passages:
            return ()
        try:
            scores = tuple(
                float(score)
                for score in self._get_model().predict(
                    [(query, passage) for passage in passages],
                    show_progress_bar=False,
                )
            )
            if len(scores) != len(passages) or not all(
                isfinite(score) for score in scores
            ):
                raise ValueError("model returned invalid scores")
            return scores
        except ModelUnavailable:
            raise
        except Exception as error:
            raise _unavailable(self.model_name, error) from error
