"""Planning an update without writing, the removal guard and stat-only
refreshes, through domain ports and in-memory fakes (ADR-0027)."""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

import pytest

from docgrep.adapters.chunking import TokenChunker, WhitespaceTokenizer
from docgrep.adapters.models.deterministic import DeterministicEmbedder
from docgrep.domain.errors import RemovalGuardTripped
from docgrep.domain.models import (
    FileType,
    ParsedDocument,
    ParsedSection,
    ScannedFile,
    Source,
)
from docgrep.domain.update_plan import Move, RemovalGuard
from docgrep.usecases.build_index import BuildIndex
from docgrep.usecases.plan_update import PlanUpdate, UpdatePlan
from docgrep.usecases.update_index import UpdateIndex
from fakes import InMemoryManifest, InMemoryPassageStore

CONFIG = {"chunk_size": 3, "chunk_overlap": 1, "tokenizer": "whitespace"}
_TYPES = {".txt": FileType.TXT, ".md": FileType.MARKDOWN}
MANIFEST_WRITES = {
    "start_run",
    "record_source",
    "remove_source",
    "set_total_sources",
    "finish_run",
    "request_cancel",
    "commit_run",
    "set_staging_collection",
    "set_checkpoint",
    "mark_swapped",
    "carry_over",
}
STORE_WRITES = {
    "upsert_live_batch",
    "delete_live_source_points",
    "delete_live_source_points_except",
    "refresh_live_source",
    "merge_completion_metadata",
    "recreate_collection",
    "upsert_batch",
    "write_completion_metadata",
}


class Corpus:
    """Files as (text, mtime) by path."""

    def __init__(self, files: Mapping[str, tuple[str, float]]) -> None:
        self.files = dict(files)

    def _source(self, path: str) -> Source:
        text, mtime = self.files[path]
        return Source(
            source_id=path,
            path=path,
            file_type=_TYPES.get(path[path.rfind(".") :], FileType.UNSUPPORTED),
            content_hash=sha256(text.encode()).hexdigest(),
            mtime=mtime,
            size=len(text.encode()),
        )

    def enumerate(self) -> tuple[Source, ...]:
        return tuple(self._source(path) for path in sorted(self.files))

    def scan(self) -> tuple[ScannedFile, ...]:
        return tuple(
            ScannedFile(
                path=path,
                size=len(text.encode()),
                extension=path[path.rfind(".") :],
                mtime=mtime,
            )
            for path, (text, mtime) in sorted(self.files.items())
        )

    def describe(self, path: str) -> Source:
        return self._source(path)


class Calls:
    def __init__(self) -> None:
        self.names: list[str] = []


class Spy:
    """Delegate to a fake and log the name of every method called."""

    def __init__(self, inner: Any, calls: Calls) -> None:
        self._inner = inner
        self._calls = calls

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._inner, name)
        if not callable(value):
            return value

        def called(*args: Any, **kwargs: Any) -> Any:
            self._calls.names.append(name)
            return value(*args, **kwargs)

        return called


class CountingParser:
    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus
        self.parsed: list[str] = []
        self.failing: set[str] = set()

    def parse(self, source: Source) -> ParsedDocument:
        self.parsed.append(source.path)
        if source.path in self.failing:
            raise ValueError("injected parse failure")
        return ParsedDocument(
            source, (ParsedSection(self.corpus.files[source.path][0]),)
        )


class CountingEmbedder(DeterministicEmbedder):
    def __init__(self) -> None:
        self.embedded = 0

    def embed(self, texts: Any) -> tuple[tuple[float, ...], ...]:
        self.embedded += len(texts)
        return super().embed(texts)


class Sparse:
    def encode(self, text: str) -> dict[int, float]:
        return {len(text): 1.0}


class Lock:
    def __init__(self) -> None:
        self.taken = 0

    @contextmanager
    def hold(self, operation: str = "build") -> Iterator[None]:
        self.taken += 1
        yield

    def is_held(self) -> bool:
        return False


class Harness:
    def __init__(self, files: Mapping[str, tuple[str, float]]) -> None:
        self.corpus = Corpus(files)
        self.calls = Calls()
        self.store = InMemoryPassageStore()
        self.manifest = InMemoryManifest()
        self.parser = CountingParser(self.corpus)
        self.embedder = CountingEmbedder()
        self.lock = Lock()

    @property
    def parsers(self) -> dict[FileType, Any]:
        return {FileType.TXT: self.parser, FileType.MARKDOWN: self.parser}

    def build(self) -> None:
        BuildIndex(
            lock=self.lock,
            enumerator=self.corpus,
            parsers=self.parsers,
            chunker=TokenChunker(WhitespaceTokenizer(), 3, 1),
            embedder=self.embedder,
            sparse_encoder=Sparse(),
            passage_writer=self.store,
            manifest=self.manifest,
            config=CONFIG,
            docgrep_version="test",
            sleep=lambda seconds: None,
        ).run()
        self.parser.parsed.clear()
        self.embedder.embedded = 0
        self.lock.taken = 0

    def planner(self, **extra: Any) -> PlanUpdate:
        return PlanUpdate(
            corpus=self.corpus,
            manifest=Spy(self.manifest, self.calls),
            parsers=self.parsers,
            **extra,
        )

    def update(self, **extra: Any) -> UpdateIndex:
        return UpdateIndex(
            lock=self.lock,
            corpus=self.corpus,
            parsers=self.parsers,
            chunker=TokenChunker(WhitespaceTokenizer(), 3, 1),
            embedder=self.embedder,
            sparse_encoder=Sparse(),
            index=Spy(self.store, self.calls),
            manifest=Spy(self.manifest, self.calls),
            config=CONFIG,
            sleep=lambda seconds: None,
            now=lambda: datetime(2026, 10, 18, tzinfo=UTC),
            **extra,
        )

    def writes(self) -> list[str]:
        return [n for n in self.calls.names if n in MANIFEST_WRITES | STORE_WRITES]

    def state(self) -> tuple[Any, ...]:
        """Everything the fakes hold, to compare before and after."""
        return (
            dict(self.manifest._sources),
            list(self.manifest._runs),
            {k: dict(v) for k, v in self.manifest._journal.items()},
            dict(self.store._items),
            self.store.read_completion_metadata(),
        )


def _files(count: int) -> dict[str, tuple[str, float]]:
    return {
        f"doc{i:02d}.txt": (f"words of document {i} here", 1.0) for i in range(count)
    }


@pytest.fixture
def harness() -> Harness:
    harness = Harness(_files(4))
    harness.build()
    return harness


def _counts(plan: UpdatePlan) -> tuple[int, ...]:
    return (
        plan.added,
        plan.modified,
        plan.removed,
        plan.retried,
        plan.refreshed,
        plan.unchanged,
        plan.moved,
    )


def _change_every_kind(harness: Harness) -> None:
    files = harness.corpus.files
    files["doc00.txt"] = ("changed words of document zero", 2.0)  # modified
    files["doc01.txt"] = (files["doc01.txt"][0], 3.0)  # touched only
    files["moved/doc02.txt"] = files.pop("doc02.txt")  # moved
    del files["doc03.txt"]  # removed
    files["new.md"] = ("a new note", 2.0)  # added


def test_the_plan_reads_only(harness: Harness) -> None:
    _change_every_kind(harness)
    before = harness.state()

    preview = harness.planner(reuse_by_hash=True).run()

    # The touched file is refreshed and still counts as unchanged; the moved
    # file is one move (ADR-0028), neither an addition nor a removal.
    assert _counts(preview.plan) == (1, 1, 1, 0, 1, 1, 1)
    assert harness.state() == before
    assert harness.writes() == []
    assert harness.parser.parsed == [] and harness.embedder.embedded == 0
    assert harness.lock.taken == 0
    # Planning twice gives the same plan.
    assert harness.planner(reuse_by_hash=True).run() == preview


def test_the_plan_lists_the_paths_of_each_group(harness: Harness) -> None:
    _change_every_kind(harness)

    (root,) = harness.planner(reuse_by_hash=True).run().roots

    assert root.name == "_legacy" and root.recorded == 4
    assert root.added == ("new.md",)
    assert root.modified == (("doc00.txt", "content_changed"),)
    assert root.removed == ("doc03.txt",)
    assert root.refreshed == ("doc01.txt",)
    assert root.moved == (Move("doc02.txt", "moved/doc02.txt"),)


def test_the_plan_predicts_the_update(harness: Harness) -> None:
    _change_every_kind(harness)
    harness.parser.failing.add("doc00.txt")
    assert harness.update().run().failed == 1
    harness.parser.failing.clear()
    harness.corpus.files["late.txt"] = ("added after the failure", 4.0)

    preview = harness.planner().run()
    summary = harness.update().run()

    assert preview.plan == summary.plan
    assert (preview.plan.retried, preview.plan.added) == (1, 1)
    assert (summary.added, summary.modified, summary.removed, summary.retried) == (
        preview.plan.added,
        preview.plan.modified,
        preview.plan.removed,
        preview.plan.retried,
    )


def test_a_stat_only_change_is_refreshed_without_parsing_or_embedding(
    harness: Harness,
) -> None:
    harness.corpus.files["doc01.txt"] = (harness.corpus.files["doc01.txt"][0], 9.0)
    before = harness.state()

    plan = harness.planner().run().plan

    assert _counts(plan) == (0, 0, 0, 0, 1, 4, 0)
    assert harness.state() == before

    summary = harness.update().run()

    assert summary.plan.refreshed == 1 and summary.no_changes
    assert harness.parser.parsed == [] and harness.embedder.embedded == 0
    record = harness.manifest.get_source("doc01.txt")
    assert record is not None and record.source.mtime == 9.0
    stored = [s for _, s, *_ in harness.store._items.values() if s.path == "doc01.txt"]
    assert stored and all(source.mtime == 9.0 for source in stored)
    # The refresh is the first write of the update, after the run started.
    writes = harness.writes()
    assert writes[:3] == ["start_run", "set_total_sources", "refresh_live_source"]
    assert writes[3] == "record_source"
    assert "upsert_live_batch" not in writes


def test_identical_new_copies_pair_with_one_removal_only() -> None:
    harness = Harness({"a.txt": ("same words", 1.0), "b.txt": ("other words", 1.0)})
    harness.build()
    files = harness.corpus.files
    files["copy2.txt"] = files["copy1.txt"] = files.pop("a.txt")

    preview = harness.planner(reuse_by_hash=True).run()

    assert preview.roots[0].moved == (Move("a.txt", "copy1.txt"),)
    assert (preview.plan.moved, preview.plan.removed, preview.plan.added) == (1, 0, 1)


def _guarded_harness(count: int, remove: int) -> Harness:
    harness = Harness(_files(count))
    harness.build()
    for index in range(remove):
        del harness.corpus.files[f"doc{index:02d}.txt"]
    harness.corpus.files["doc_touched.txt"] = ("touched later", 1.0)
    return harness


def test_the_guard_refuses_an_emptied_root_and_writes_nothing() -> None:
    harness = _guarded_harness(40, 40)
    before = harness.state()
    seen: list[UpdatePlan] = []

    with pytest.raises(RemovalGuardTripped) as refused:
        harness.update().run(on_plan=seen.append)

    assert refused.value.category == "removal_guard"
    assert "--yes" in str(refused.value)
    assert refused.value.details["roots"] == [
        {
            "root": "_legacy",
            "removals": 40,
            "recorded": 40,
            "threshold": 2,
            "reason": "empties_root",
        }
    ]
    assert seen and seen[0].removed == 40
    assert harness.state() == before and harness.writes() == []
    preview = harness.planner().run()
    assert [g.root for g in preview.guarded] == ["_legacy"]

    summary = harness.update().run(override_guard=True)

    assert summary.removed == 40 and summary.added == 1
    assert len(harness.manifest.committed_sources()) == 1


def test_a_refused_update_does_not_refresh_either() -> None:
    harness = _guarded_harness(40, 3)
    harness.corpus.files["doc10.txt"] = (harness.corpus.files["doc10.txt"][0], 5.0)
    before = harness.state()

    with pytest.raises(RemovalGuardTripped):
        harness.update().run()

    assert harness.state() == before


def test_removals_up_to_the_threshold_proceed() -> None:
    harness = _guarded_harness(40, 2)

    assert harness.planner().run().guarded == ()
    assert harness.update().run().removed == 2


@pytest.mark.parametrize(("count", "refused"), [(3, False), (2, True)])
def test_a_small_root_is_guarded_by_the_count(count: int, refused: bool) -> None:
    harness = _guarded_harness(4, 3)
    update = harness.update(guard=RemovalGuard(ratio=0.05, count=count))
    run: Callable[[], Any] = update.run

    if refused:
        with pytest.raises(RemovalGuardTripped):
            run()
    else:
        assert run().removed == 3


def test_a_two_source_root_emptied_is_refused_and_yes_applies_it() -> None:
    harness = Harness(_files(2))
    harness.build()
    harness.corpus.files.clear()
    before = harness.state()

    with pytest.raises(RemovalGuardTripped) as refused:
        harness.update().run()

    assert refused.value.details["roots"][0]["reason"] == "empties_root"
    assert "would lose all its 2 sources" in str(refused.value)
    assert harness.state() == before and harness.writes() == []
    assert harness.update().run(override_guard=True).removed == 2
    assert harness.manifest.committed_sources() == ()


def test_removing_one_of_two_sources_is_allowed() -> None:
    harness = Harness(_files(2))
    harness.build()
    del harness.corpus.files["doc00.txt"]

    assert harness.planner().run().guarded == ()
    assert harness.update().run().removed == 1


def _move_all(harness: Harness, count: int) -> None:
    for index in range(count):
        harness.corpus.files[f"moved/doc{index:02d}.txt"] = harness.corpus.files.pop(
            f"doc{index:02d}.txt"
        )


@pytest.mark.parametrize("reuse", [True, False])
def test_moving_every_source_of_a_root_passes_the_guard(reuse: bool) -> None:
    harness = Harness(_files(40))
    harness.build()
    _move_all(harness, 40)

    preview = harness.planner(reuse_by_hash=reuse).run()

    # Whether or not the passages are reused, nothing is lost.
    assert preview.guarded == ()
    assert preview.roots[0].paired_removals == 40
    assert preview.roots[0].unpaired_removals == 0
    summary = harness.update(reuse_by_hash=reuse).run()
    assert summary.failed == 0
    assert {r.source.path for r in harness.manifest.committed_sources()} == {
        f"moved/doc{i:02d}.txt" for i in range(40)
    }


def test_deleting_as_many_sources_without_copies_still_trips_the_guard() -> None:
    harness = _guarded_harness(40, 40)

    with pytest.raises(RemovalGuardTripped) as refused:
        harness.update().run()

    assert "paired" not in refused.value.details["roots"][0]


def test_the_guard_counts_only_the_unpaired_part_of_a_mix() -> None:
    harness = Harness(_files(60))
    harness.build()
    # Threshold of 60 sources: 3. Two moved, four deleted.
    _move_all(harness, 2)
    for index in range(2, 6):
        del harness.corpus.files[f"doc{index:02d}.txt"]

    with pytest.raises(RemovalGuardTripped) as refused:
        harness.update().run()

    [root] = refused.value.details["roots"]
    assert (root["removals"], root["paired"], root["reason"]) == (4, 2, "threshold")
    assert "4 of 60" in str(refused.value) and "2 paired" in str(refused.value)


def test_a_mix_within_the_threshold_proceeds() -> None:
    harness = Harness(_files(60))
    harness.build()
    _move_all(harness, 10)
    del harness.corpus.files["doc20.txt"]

    assert harness.planner(reuse_by_hash=True).run().guarded == ()
    summary = harness.update(reuse_by_hash=True).run()
    assert (summary.moved, summary.removed) == (10, 1)
