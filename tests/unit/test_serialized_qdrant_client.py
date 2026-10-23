"""Calls of a local-mode Qdrant client run one at a time."""

import threading

from docgrep.adapters.qdrant import _SerializedClient


class Slow:
    """A client whose `slow` call waits until the test lets it finish."""

    attribute = "value"

    def __init__(self) -> None:
        self.inside = threading.Event()
        self.release = threading.Event()
        self.order: list[str] = []

    def slow(self) -> str:
        self.order.append("slow:start")
        self.inside.set()
        assert self.release.wait(timeout=10)
        self.order.append("slow:end")
        return "slow"

    def fast(self) -> str:
        self.order.append("fast")
        return "fast"


def test_a_call_waits_for_the_one_in_progress() -> None:
    inner = Slow()
    client = _SerializedClient(inner)  # type: ignore[arg-type]  # a stand-in with the same calls
    results: list[str] = []
    first = threading.Thread(target=lambda: results.append(client.slow()))
    first.start()
    assert inner.inside.wait(timeout=10)
    second = threading.Thread(target=lambda: results.append(client.fast()))
    second.start()
    # `fast` cannot have run while `slow` holds the client.
    assert inner.order == ["slow:start"]

    inner.release.set()
    first.join(10)
    second.join(10)

    assert inner.order == ["slow:start", "slow:end", "fast"]
    assert sorted(results) == ["fast", "slow"]


def test_attributes_pass_through() -> None:
    client = _SerializedClient(Slow())  # type: ignore[arg-type]  # a stand-in with the same calls

    assert client.attribute == "value"
