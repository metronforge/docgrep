"""Shared fixtures for GPU-marked tests."""

import gc
import sys
from collections.abc import Iterator

import pytest


@pytest.fixture
def require_cuda() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


@pytest.fixture(autouse=True)
def release_gpu_memory(request: pytest.FixtureRequest) -> Iterator[None]:
    """Free model weights between GPU tests so a small card can run them all."""
    yield
    if request.node.get_closest_marker("gpu") is None:
        return
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
