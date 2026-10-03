"""Real-model adapters tested with loader fakes, without model downloads."""

import math
import subprocess
import sys
from collections.abc import Sequence

import pytest

from docgrep.adapters.models.real import BgeEmbedder, BgeReranker
from docgrep.domain.errors import ModelUnavailable
from docgrep.domain.protocols import EmbeddingProvider, Reranker, TokenSpanTokenizer


class FakeTokenizer:
    def __call__(
        self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool
    ) -> dict[str, list[tuple[int, int]]]:
        assert not add_special_tokens
        assert return_offsets_mapping
        return {"offset_mapping": [(0, 5), (6, len(text))]}


class FakeSentenceModel:
    tokenizer = FakeTokenizer()

    def get_sentence_embedding_dimension(self) -> int:
        return 2

    def encode(
        self,
        texts: Sequence[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> list[list[float]]:
        assert normalize_embeddings and convert_to_numpy and not show_progress_bar
        return [[1.0, 0.0] for _ in texts]


class FakeCrossModel:
    def predict(
        self, pairs: Sequence[tuple[str, str]], *, show_progress_bar: bool
    ) -> list[float]:
        assert not show_progress_bar
        return [float(len(passage)) for _, passage in pairs]


def test_models_load_lazily_and_keep_batch_order() -> None:
    loaded: list[tuple[str, str]] = []

    def load_embedding(name: str, device: str) -> FakeSentenceModel:
        loaded.append((name, device))
        return FakeSentenceModel()

    def load_reranker(name: str, device: str) -> FakeCrossModel:
        loaded.append((name, device))
        return FakeCrossModel()

    embedder: EmbeddingProvider = BgeEmbedder("embedding-test", "cpu", load_embedding)
    reranker: Reranker = BgeReranker("reranker-test", "cpu", load_reranker)
    assert loaded == []
    assert embedder.dimension == 2
    assert embedder.embed(["first", "second"]) == ((1.0, 0.0), (1.0, 0.0))
    assert reranker.score("query", ["a", "longer"]) == (1.0, 6.0)
    assert loaded == [("embedding-test", "cpu"), ("reranker-test", "cpu")]


def test_embedding_tokenizer_exposes_character_spans() -> None:
    embedder = BgeEmbedder("embedding-test", "cpu", lambda *_: FakeSentenceModel())
    tokenizer: TokenSpanTokenizer = embedder.tokenizer

    assert tokenizer.token_spans("hello world") == ((0, 5), (6, 11))


@pytest.mark.parametrize("adapter", ["embedding", "reranker"])
def test_load_failure_is_structured(adapter: str) -> None:
    def fail(_name: str, _device: str) -> FakeSentenceModel:
        raise OSError("weights unavailable")

    if adapter == "embedding":
        model = BgeEmbedder("missing", "cpu", fail)
        with pytest.raises(ModelUnavailable) as caught:
            _ = model.dimension
    else:
        model = BgeReranker("missing", "cpu", fail)
        with pytest.raises(ModelUnavailable) as caught:
            model.score("query", ["passage"])
    assert caught.value.category == "model_unavailable"
    assert caught.value.details["model"] == "missing"


def test_inference_failure_is_structured() -> None:
    class BrokenSentenceModel(FakeSentenceModel):
        def encode(
            self,
            texts: Sequence[str],
            *,
            normalize_embeddings: bool,
            convert_to_numpy: bool,
            show_progress_bar: bool,
        ) -> list[list[float]]:
            raise RuntimeError("device lost")

    embedder = BgeEmbedder("broken", "cpu", lambda *_: BrokenSentenceModel())
    with pytest.raises(ModelUnavailable) as caught:
        embedder.embed(["text"])
    assert caught.value.details["model"] == "broken"


def test_fake_backend_import_does_not_load_ml_libraries() -> None:
    script = (
        "import sys; "
        "from docgrep.adapters.models.deterministic import DeterministicEmbedder; "
        "from docgrep.adapters.models.real import BgeEmbedder; "
        "assert len(DeterministicEmbedder().embed(['text'])) == 1; "
        "assert 'sentence_transformers' not in sys.modules; "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize(
    ("setting", "cuda", "expected", "probed"),
    [
        ("auto", True, "cuda", True),
        ("auto", False, "cpu", True),
        ("cpu", True, "cpu", False),
        ("cuda", False, "cuda", False),
    ],
)
def test_device_selection_uses_the_injected_availability_probe(
    setting: str, cuda: bool, expected: str, probed: bool
) -> None:
    probes: list[bool] = []

    def probe() -> bool:
        probes.append(True)
        return cuda

    chosen: list[str] = []

    def load_embedding(_name: str, device: str) -> FakeSentenceModel:
        chosen.append(device)
        return FakeSentenceModel()

    def load_reranker(_name: str, device: str) -> FakeCrossModel:
        chosen.append(device)
        return FakeCrossModel()

    _ = BgeEmbedder("e", setting, load_embedding, probe).dimension
    BgeReranker("r", setting, load_reranker, probe).score("q", ["p"])

    assert chosen == [expected, expected]
    assert bool(probes) is probed


@pytest.mark.gpu
@pytest.mark.usefixtures("require_cuda")
def test_real_bge_embedding_and_reranking() -> None:
    embedder = BgeEmbedder("BAAI/bge-m3", "auto")
    vectors = embedder.embed(["a cat sleeps", "a dog runs"])
    assert embedder.dimension == 1024
    assert len(vectors) == 2
    assert all(
        math.isclose(math.dist(vector, (0.0,) * 1024), 1.0, abs_tol=1e-4)
        for vector in vectors
    )

    reranker = BgeReranker("BAAI/bge-reranker-v2-m3", "auto")
    relevant, irrelevant = reranker.score(
        "cat sleeping",
        ["A cat is sleeping on the sofa.", "Quantum mechanics studies matter."],
    )
    assert relevant > irrelevant
