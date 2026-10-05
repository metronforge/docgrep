"""Shared fixtures: environment isolation and GPU-marked test support."""

import gc
import os
import sys
from collections.abc import Iterator

import pytest

from qdrant_server_guard import check_server


@pytest.fixture(autouse=True)
def isolated_docgrep_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove the caller's DOCGREP_ settings so they cannot change any test.

    DOCGREP_TEST_ variables configure the test run itself (such as the URL of a
    throwaway Qdrant server) and are kept. A test that needs a setting sets it
    with monkeypatch.
    """
    for name in list(os.environ):
        if name.startswith("DOCGREP_") and not name.startswith("DOCGREP_TEST_"):
            monkeypatch.delenv(name)


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


def pytest_collection_finish(session: pytest.Session) -> None:
    """Stop before any qdrant_server test if the server holds foreign data.

    Runs once, after collection and deselection and before the first test, so
    nothing on the server has been touched when the run is refused.
    """
    url = os.environ.get("DOCGREP_TEST_QDRANT_URL")
    if not url:
        return
    if not any(item.get_closest_marker("qdrant_server") for item in session.items):
        return
    refusal = check_server(url)
    if refusal is not None:
        pytest.exit(refusal, returncode=pytest.ExitCode.USAGE_ERROR)
