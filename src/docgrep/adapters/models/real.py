"""Lazy local embedding and cross-encoder adapters."""

import logging
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from importlib import import_module
from math import isfinite, sqrt
from typing import Protocol, cast

from docgrep.domain.errors import ModelUnavailable

_LOG = logging.getLogger(__name__)


class OffsetTokenizer(Protocol):
    """Expose tokenizer offsets without binding the domain to Transformers."""

    def __call__(
        self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool
    ) -> Mapping[str, object]: ...


class SentenceModel(Protocol):
    """Sentence Transformer operations used for embedding and chunking."""

    @property
    def tokenizer(self) -> OffsetTokenizer: ...

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


def _load_sentence_model(
    name: str, device: str, dtype: str = "float32"
) -> SentenceModel:
    model_type = import_module("sentence_transformers").SentenceTransformer
    if dtype == "float32":
        return cast(SentenceModel, model_type(name, device=device))
    return cast(
        SentenceModel, model_type(name, device=device, model_kwargs={"dtype": dtype})
    )


def _load_cross_model(name: str, device: str, dtype: str = "float32") -> CrossModel:
    model_type = import_module("sentence_transformers").CrossEncoder
    if dtype == "float32":
        return cast(CrossModel, model_type(name, device=device))
    return cast(
        CrossModel, model_type(name, device=device, model_kwargs={"dtype": dtype})
    )


def _torch_cuda_available() -> bool:
    return bool(import_module("torch").cuda.is_available())


def _device(setting: str, cuda_available: Callable[[], bool]) -> str:
    if setting != "auto":
        return setting
    return "cuda" if cuda_available() else "cpu"


def resolve_dtype(setting: str, device: str) -> tuple[str, bool]:
    """Pick the weight precision for a device.

    Returns the effective dtype and whether an explicit float16 request had to
    fall back to float32 because the device is not a CUDA device. ``auto`` means
    float16 on CUDA and float32 elsewhere.
    """
    if setting == "float32":
        return "float32", False
    if device.startswith("cuda"):
        return "float16", False
    return "float32", setting == "float16"


resolve_reranker_dtype = resolve_dtype


def _unit_length(vector: Sequence[float]) -> tuple[float, ...]:
    norm = sqrt(sum(value * value for value in vector))
    return tuple(vector) if norm == 0 else tuple(value / norm for value in vector)


def _load_logged[T](
    loader: Callable[[str, str], T],
    name: str,
    device_setting: str,
    cuda_available: Callable[[], bool],
    extra: Mapping[str, str] | None = None,
) -> T:
    device = _device(device_setting, cuda_available)
    started = time.monotonic()
    model = loader(name, device)
    _LOG.info(
        "model_loaded",
        extra={
            "model": name,
            "device": device,
            "load_ms": round((time.monotonic() - started) * 1000, 1),
            **(extra or {}),
        },
    )
    return model


def _unavailable(name: str, error: Exception) -> ModelUnavailable:
    return ModelUnavailable(
        f"Model {name} is unavailable",
        details={"model": name, "reason": str(error)},
    )


def _embedding_dimension(model: object) -> int | None:
    """Read the vector size with whichever accessor the library version has.

    Newer sentence-transformers releases renamed the accessor and warn when the
    old name is used, so the new name is preferred when it exists.
    """
    for name in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        accessor = getattr(model, name, None)
        if callable(accessor):
            return cast(int | None, accessor())
    raise AttributeError("model has no embedding dimension accessor")


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
        dtype: str = "auto",
        factory: Callable[[str, str, str], SentenceModel] | None = None,
    ) -> None:
        self.model_name = model_name
        self.dtype_setting = dtype
        self._device = device
        self._loader = loader
        self._factory = factory or _load_sentence_model
        self._cuda_available = cuda_available or _torch_cuda_available
        self._model: SentenceModel | None = None

    @property
    def dtype(self) -> str:
        """The precision the model runs in on the configured device."""
        device = _device(self._device, self._cuda_available)
        return resolve_dtype(self.dtype_setting, device)[0]

    def _get_model(self) -> SentenceModel:
        if self._model is None:
            try:
                device = _device(self._device, self._cuda_available)
                dtype, fell_back = resolve_dtype(self.dtype_setting, device)
                if fell_back:
                    _LOG.warning(
                        "embedding_dtype_fallback",
                        extra={
                            "requested": self.dtype_setting,
                            "device": device,
                            "dtype": dtype,
                        },
                    )
                injected = self._loader
                factory = self._factory
                self._model = _load_logged(
                    (lambda n, d: injected(n, d))
                    if injected is not None
                    else (lambda n, d: factory(n, d, dtype)),
                    self.model_name,
                    device,
                    self._cuda_available,
                    None if injected is not None else {"dtype": dtype},
                )
            except Exception as error:
                raise _unavailable(self.model_name, error) from error
        return self._model

    @property
    def dimension(self) -> int:
        """Return the loaded model's fixed embedding dimension."""
        try:
            dimension = _embedding_dimension(self._get_model())
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
            half = self.dtype == "float16"
            vectors = tuple(
                # Half-precision output is unit length only to about 1e-3, so it is
                # normalized again in float32 and callers always get float vectors.
                (_unit_length if half else tuple)([float(value) for value in vector])
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
        dtype: str = "auto",
        factory: Callable[[str, str, str], CrossModel] | None = None,
    ) -> None:
        self.model_name = model_name
        self.dtype = dtype
        self._device = device
        self._loader = loader
        self._factory = factory or _load_cross_model
        self._cuda_available = cuda_available or _torch_cuda_available
        self._model: CrossModel | None = None

    def _get_model(self) -> CrossModel:
        if self._model is None:
            try:
                device = _device(self._device, self._cuda_available)
                dtype, fell_back = resolve_reranker_dtype(self.dtype, device)
                if fell_back:
                    _LOG.warning(
                        "reranker_dtype_fallback",
                        extra={
                            "requested": self.dtype,
                            "device": device,
                            "dtype": dtype,
                        },
                    )
                injected = self._loader
                factory = self._factory
                self._model = _load_logged(
                    (lambda n, d: injected(n, d))
                    if injected is not None
                    else (lambda n, d: factory(n, d, dtype)),
                    self.model_name,
                    device,
                    self._cuda_available,
                    {"dtype": dtype},
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
