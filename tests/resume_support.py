"""Helpers shared by the resume tests: corpora, interruptions and index dumps."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from docgrep.adapters.qdrant import QdrantPassageStore
from docgrep.api.container import Services, create_container
from docgrep.domain.errors import StoreUnavailable
from docgrep.settings import Settings

_WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega"
).split()


def write_corpus(root: Path, count: int, words: int = 25) -> list[str]:
    """Write ``count`` small text files with deterministic, distinct content."""
    root.mkdir(parents=True, exist_ok=True)
    names = []
    for index in range(count):
        name = f"doc{index:03d}.txt"
        body = [f"marker{index}"] + [
            _WORDS[(index * 7 + step * 3) % len(_WORDS)] for step in range(words)
        ]
        (root / name).write_text(" ".join(body), encoding="utf-8")
        names.append(name)
    return names


def settings_for(
    corpus: Path,
    data: Path,
    *,
    url: str | None = None,
    alias: str = "docgrep",
    **extra: Any,
) -> Settings:
    return Settings(
        corpus_root=corpus,
        data_dir=data,
        backend="fake",
        chunk_size=10,
        chunk_overlap=2,
        qdrant_url=url,
        qdrant_alias=alias,
        **extra,
    )


def new_services(
    settings: Settings, *, refuse_resumable_by_default: bool = False
) -> Services:
    """A fresh container, as a new process would have; no sleeping on retries."""
    services = create_container(
        settings, refuse_resumable_by_default=refuse_resumable_by_default
    )
    build: Any = services.build
    build.sleep = lambda seconds: None
    return services


def close(services: Services) -> None:
    build: Any = services.build
    store = build.passage_writer
    inner = getattr(store, "inner", store)
    assert isinstance(inner, QdrantPassageStore)
    inner.close()


class HookedEmbedder:
    """Delegate to the real embedder and run a hook before the n-th embed call."""

    def __init__(self, inner: Any, at: int, hook: Any) -> None:
        self.inner = inner
        self.at = at
        self.hook = hook
        self.calls = 0
        self.model_name = inner.model_name
        self.dimension = inner.dimension
        self.dtype = inner.dtype

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        self.calls += 1
        if self.calls == self.at:
            self.hook()
        result: tuple[tuple[float, ...], ...] = self.inner.embed(texts)
        return result


def cancel_at_embed_call(services: Services, call: int) -> None:
    """Request a cancel while the ``call``-th source is embedded (0: right away)."""
    build: Any = services.build
    if call == 0:
        build.request_cancel()
        return
    build.embedder = HookedEmbedder(build.embedder, call, build.request_cancel)


class FailingUpserts:
    """Delegate to the real writer and fail every upsert_batch from the ``at``-th."""

    def __init__(self, inner: Any, at: int) -> None:
        self.inner = inner
        self.at = at
        self.calls = 0

    def upsert_batch(self, items: Any) -> None:
        self.calls += 1
        if self.calls >= self.at:  # stays down, so the build's retries fail too
            raise StoreUnavailable("injected failure")
        self.inner.upsert_batch(items)

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)


def fail_upsert_at(services: Services, call: int) -> None:
    build: Any = services.build
    build.passage_writer = FailingUpserts(build.passage_writer, call)


def dump_points(settings: Settings) -> list[tuple[str, dict[str, Any], Any]]:
    """Every passage point of the published index: id, payload and vectors.

    Opens its own client, so no other local-mode store may be open on the data
    directory.
    """
    client = (
        QdrantClient(url=settings.qdrant_url)
        if settings.qdrant_url
        else QdrantClient(path=str(settings.data_dir / "qdrant"))
    )
    try:
        points, _ = client.scroll(
            settings.qdrant_alias, limit=100_000, with_payload=True, with_vectors=True
        )
        rows = [
            (str(p.id), dict(p.payload or {}), p.vector)
            for p in points
            if (p.payload or {}).get("record_type") == "passage"
        ]
    finally:
        client.close()
    return sorted(rows, key=lambda row: row[0])


def search_snapshot(
    services: Services, queries: Sequence[str] = ("alpha", "marker2", "omega beta")
) -> list[list[tuple[str, str, float]]]:
    snapshot = []
    for query in queries:
        result = services.search.run(query, top_n=10)
        snapshot.append(
            [(h.passage.passage_id, h.passage.text, h.score) for h in result.results]
        )
    return snapshot


class Crash(BaseException):
    """A process death: no handler of the build sees it."""


class FailsMarkingSwapped:
    """Delegate to the real manifest but fail exactly at the swapped status write.

    The alias swap has happened by then. A BaseException leaves the run running
    (a killed process); an ordinary exception makes the build abort it.
    """

    def __init__(self, inner: Any, error: BaseException) -> None:
        self.inner = inner
        self.error = error

    def mark_swapped(self, run_id: str) -> None:
        raise self.error

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def fail_at_swapped_mark(services: Services, *, crash: bool) -> None:
    build: Any = services.build
    error: BaseException = Crash() if crash else OSError("manifest disk error")
    build.manifest = FailsMarkingSwapped(build.manifest, error)
