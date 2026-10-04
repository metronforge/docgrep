"""Incremental update rules through domain ports and in-memory fakes (UC-4)."""

from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Literal, cast

import pytest

from docgrep.adapters.chunking import TokenChunker, WhitespaceTokenizer
from docgrep.adapters.models.deterministic import DeterministicEmbedder
from docgrep.domain.errors import (
    ConfigDrift,
    IndexIncomplete,
    IndexLockHeld,
    ResumeAvailable,
    StoreUnavailable,
)
from docgrep.domain.models import (
    FileType,
    ParsedDocument,
    ParsedSection,
    Passage,
    ScannedFile,
    Source,
)
from docgrep.usecases.build_index import BuildIndex
from docgrep.usecases.update_index import UpdateIndex
from fakes import InMemoryManifest, InMemoryPassageStore

Finished = Literal["completed", "aborted", "cancelled"]
CONFIG = {"chunk_size": 3, "chunk_overlap": 1, "tokenizer": "whitespace"}
_TYPES = {".txt": FileType.TXT, ".md": FileType.MARKDOWN, ".epub": FileType.EPUB}


class Corpus:
    """Files as (text, mtime) by path; counts how often content is read."""

    def __init__(self, files: Mapping[str, tuple[str, float]]) -> None:
        self.files = dict(files)
        self.described: list[str] = []

    def _source(self, path: str) -> Source:
        text, mtime = self.files[path]
        suffix = path[path.rfind(".") :]
        return Source(
            source_id=path,
            path=path,
            file_type=_TYPES.get(suffix, FileType.UNSUPPORTED),
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
        self.described.append(path)
        return self._source(path)


class Parser:
    """Reads the fake corpus; fails for the paths in ``failing``."""

    def __init__(self, corpus: Corpus, version: str | None = None) -> None:
        self.corpus = corpus
        self.failing: set[str] = set()
        if version is not None:
            self.version = version

    def parse(self, source: Source) -> ParsedDocument:
        if source.path in self.failing:
            raise ValueError("injected parse failure")
        text = self.corpus.files[source.path][0]
        if not text.strip():
            return ParsedDocument(source, (), requires_ocr=True)
        return ParsedDocument(source, (ParsedSection(text),))


class Sparse:
    def encode(self, text: str) -> dict[int, float]:
        return {len(text): 1.0}


class Lock:
    def __init__(self, held: bool = False) -> None:
        self.held = held

    @contextmanager
    def hold(self, operation: str = "build") -> Iterator[None]:
        if self.held:
            raise IndexLockHeld(f"held, {operation} refused")
        yield

    def is_held(self) -> bool:
        return self.held


class Events:
    def __init__(self) -> None:
        self.log: list[tuple[str, str]] = []


class LoggingStore(InMemoryPassageStore):
    def __init__(self, events: Events) -> None:
        super().__init__()
        self.events = events
        self.fail_removals = 0
        self.fail_upserts = False

    def upsert_live_batch(self, items: Sequence[Any]) -> None:
        if self.fail_upserts:
            raise StoreUnavailable("injected")
        self.events.log.append(("upsert", items[0][0].source_id))
        super().upsert_live_batch(items)

    def delete_live_source_points_except(
        self, source_id: str, keep: Collection[str]
    ) -> int:
        self.events.log.append(("prune", source_id))
        return super().delete_live_source_points_except(source_id, keep)

    def delete_live_source_points(self, source_id: str) -> int:
        if self.fail_removals:
            self.fail_removals -= 1
            raise StoreUnavailable("injected")
        self.events.log.append(("delete", source_id))
        return super().delete_live_source_points(source_id)


class LoggingManifest(InMemoryManifest):
    def __init__(self, events: Events) -> None:
        super().__init__()
        self.events = events

    def record_source(self, source: Source, status: str, *args: Any, **kw: Any) -> None:
        where = "journal" if kw.get("run_id") else "sources"
        self.events.log.append((f"{where}:{status}", source.path))
        super().record_source(source, status, *args, **kw)

    def remove_source(self, path: str) -> None:
        self.events.log.append(("sources:removed", path))
        super().remove_source(path)


class Harness:
    def __init__(self, files: Mapping[str, tuple[str, float]]) -> None:
        self.events = Events()
        self.corpus = Corpus(files)
        self.store = LoggingStore(self.events)
        self.manifest = LoggingManifest(self.events)
        self.parser = Parser(self.corpus)
        self.parsers: dict[FileType, Any] = {
            FileType.TXT: self.parser,
            FileType.MARKDOWN: self.parser,
        }
        self.lock = Lock()

    def common(self) -> dict[str, Any]:
        return {
            "lock": self.lock,
            "chunker": TokenChunker(WhitespaceTokenizer(), 3, 1),
            "embedder": DeterministicEmbedder(),
            "sparse_encoder": Sparse(),
            "manifest": self.manifest,
            "config": CONFIG,
            "sleep": lambda seconds: None,
        }

    def build(self) -> BuildIndex:
        return BuildIndex(
            enumerator=self.corpus,
            parsers=dict(self.parsers),
            passage_writer=self.store,
            docgrep_version="test",
            **self.common(),
        )

    def update(self, **extra: Any) -> UpdateIndex:
        return UpdateIndex(
            corpus=self.corpus,
            parsers=dict(self.parsers),
            index=self.store,
            now=lambda: datetime(2026, 10, 4, tzinfo=UTC),
            **{**self.common(), **extra},
        )

    def passages(self, path: str) -> list[Passage]:
        return [p for p in self._published() if p.source_id == path]

    def _published(self) -> list[Passage]:
        return [
            item[0]
            for item in self.store._items.values()  # the fake's published side
        ]


FILES = {
    "a.txt": ("one two three four five", 1.0),
    "b.txt": ("six seven eight nine", 1.0),
    "c.md": ("ten eleven twelve", 1.0),
    "d.epub": ("thirteen fourteen", 1.0),
}


@pytest.fixture
def harness() -> Harness:
    harness = Harness(FILES)
    harness.build().run()
    harness.events.log.clear()
    return harness


def test_the_plan_classifies_every_file(harness: Harness) -> None:
    files = harness.corpus.files
    del files["a.txt"]
    files["b.txt"] = ("six seven changed", 2.0)
    files["c.md"] = (files["c.md"][0], 3.0)  # touched, same content
    files["e.txt"] = ("new words here", 2.0)
    harness.parsers[FileType.EPUB] = harness.parser  # d.epub was unsupported
    harness.corpus.described.clear()

    summary = harness.update().run()

    plan = summary.plan
    assert (plan.added, plan.modified, plan.removed) == (1, 2, 1)
    assert (plan.unchanged, plan.retried, plan.refreshed) == (1, 0, 1)
    # Only files that may have changed are read and hashed.
    assert sorted(harness.corpus.described) == ["b.txt", "c.md", "d.epub", "e.txt"]
    record = harness.manifest.get_source("c.md")
    assert record is not None and record.source.mtime == 3.0
    assert {p.passage_id for p in harness.passages("c.md")}
    assert all(
        item[1].mtime == 3.0
        for item in harness.store._items.values()
        if item[1].path == "c.md"
    )
    assert harness.manifest.get_source("a.txt") is None
    assert harness.passages("a.txt") == []
    assert harness.manifest.get_source("d.epub") is not None


def test_changes_run_removed_then_modified_then_added_and_insert_first(
    harness: Harness,
) -> None:
    files = harness.corpus.files
    del files["a.txt"]
    files["b.txt"] = ("six seven changed words", 2.0)
    files["e.txt"] = ("new words here", 2.0)

    harness.update().run()

    log = [entry for entry in harness.events.log if entry[0] != "journal:indexed"]
    assert log == [
        ("delete", "a.txt"),
        ("sources:removed", "a.txt"),
        ("journal:removed", "a.txt"),
        ("upsert", "b.txt"),
        ("prune", "b.txt"),
        ("sources:indexed", "b.txt"),
        ("upsert", "e.txt"),
        ("prune", "e.txt"),
        ("sources:indexed", "e.txt"),
    ]
    assert [p.text for p in harness.passages("b.txt")] == [
        "six seven changed",
        "changed words",
    ]


def test_a_modified_source_that_fails_keeps_its_passages_and_record(
    harness: Harness,
) -> None:
    before = harness.passages("b.txt")
    record = harness.manifest.get_source("b.txt")
    harness.corpus.files["b.txt"] = ("six seven changed", 2.0)
    harness.parser.failing.add("b.txt")

    summary = harness.update().run()

    assert summary.failed == 1 and summary.failures[0].reason == "parse_error"
    assert harness.passages("b.txt") == before
    assert harness.manifest.get_source("b.txt") == record
    run = harness.manifest.latest_run()
    assert run is not None and run.status == "completed"
    (journaled,) = harness.manifest.run_sources(run.run_id)
    assert journaled.status == "failed:parse_error"

    harness.parser.failing.clear()
    again = harness.update().run()

    assert again.plan.modified == 1 and again.indexed == 1


def test_a_store_failure_while_writing_keeps_the_record(harness: Harness) -> None:
    record = harness.manifest.get_source("b.txt")
    harness.corpus.files["b.txt"] = ("six seven changed", 2.0)
    harness.corpus.files["e.txt"] = ("new words here", 2.0)
    harness.store.fail_upserts = True

    summary = harness.update().run()

    assert [(f.path, f.reason) for f in summary.failures] == [
        ("b.txt", "store_error"),
        ("e.txt", "store_error"),
    ]
    assert harness.manifest.get_source("b.txt") == record
    added = harness.manifest.get_source("e.txt")
    assert added is not None and added.status == "failed:store_error"
    assert harness.passages("e.txt") == []


def test_an_added_source_that_fails_is_recorded_and_retried(harness: Harness) -> None:
    harness.corpus.files["e.txt"] = ("new words here", 2.0)
    harness.parser.failing.add("e.txt")

    first = harness.update().run()

    record = harness.manifest.get_source("e.txt")
    assert first.failed == 1
    assert record is not None and record.status == "failed:parse_error"

    harness.parser.failing.clear()
    assert harness.update().run(retry_failed=False).no_changes
    retried = harness.update().run()

    assert retried.plan.retried == 1 and retried.retried == 1
    record = harness.manifest.get_source("e.txt")
    assert record is not None and record.status == "indexed"


def test_a_changed_failed_source_is_modified_even_without_retries(
    harness: Harness,
) -> None:
    harness.corpus.files["e.txt"] = ("new words here", 2.0)
    harness.parser.failing.add("e.txt")
    harness.update().run()
    harness.parser.failing.clear()
    harness.corpus.files["e.txt"] = ("new words changed", 3.0)

    summary = harness.update().run(retry_failed=False)

    assert (summary.plan.modified, summary.plan.retried) == (1, 0)


def test_a_failed_removal_is_retried_then_journaled_and_kept(
    harness: Harness,
) -> None:
    del harness.corpus.files["a.txt"]
    harness.store.fail_removals = 3

    summary = harness.update().run()

    assert [(f.path, f.reason) for f in summary.failures] == [
        ("a.txt", "removal_failed")
    ]
    assert harness.manifest.get_source("a.txt") is not None
    run = harness.manifest.latest_run()
    assert run is not None
    (journaled,) = harness.manifest.run_sources(run.run_id)
    assert journaled.status == "failed:removal_failed"

    again = harness.update().run()

    assert again.removed == 1 and again.failed == 0
    assert harness.manifest.get_source("a.txt") is None


def test_a_removal_succeeds_within_the_retry_policy(harness: Harness) -> None:
    del harness.corpus.files["a.txt"]
    harness.store.fail_removals = 2

    summary = harness.update().run()

    assert summary.removed == 1 and summary.failed == 0


def test_no_changes_records_a_completed_run_and_the_check_time(
    harness: Harness,
) -> None:
    summary = harness.update().run()

    assert summary.no_changes and summary.plan.unchanged == 4
    assert harness.events.log == []
    run = harness.manifest.latest_run()
    assert run is not None and (run.kind, run.status) == ("update", "completed")
    assert run.total_sources == 0
    assert dict(run.totals)["passages_written"] == 0
    metadata = harness.store.read_completion_metadata()
    assert metadata is not None
    assert metadata["last_updated_at"] == "2026-10-04T00:00:00+00:00"
    assert metadata["chunk_size"] == 3


def test_a_parser_version_reprocesses_its_type_only(harness: Harness) -> None:
    harness.parsers[FileType.MARKDOWN] = Parser(harness.corpus, version="7")

    summary = harness.update().run()

    assert (summary.plan.modified, summary.plan.unchanged) == (1, 3)
    record = harness.manifest.get_source("c.md")
    assert record is not None and record.source.parser_version == "7"
    assert harness.update().run().no_changes


def test_a_removed_parser_drops_the_passages_of_its_type(harness: Harness) -> None:
    del harness.parsers[FileType.MARKDOWN]

    summary = harness.update().run()

    assert summary.plan.modified == 1 and summary.skipped == 1
    record = harness.manifest.get_source("c.md")
    assert record is not None and record.status == "skipped:unsupported"
    assert harness.passages("c.md") == []


def test_a_source_that_now_needs_ocr_loses_its_passages(harness: Harness) -> None:
    harness.corpus.files["b.txt"] = ("   ", 2.0)

    summary = harness.update().run()

    assert summary.skipped == 1
    record = harness.manifest.get_source("b.txt")
    assert record is not None and record.status == "skipped:requires_ocr"
    assert harness.passages("b.txt") == []


def test_cancel_before_the_first_change_applies_nothing(harness: Harness) -> None:
    harness.corpus.files["e.txt"] = ("new words here", 2.0)
    update = harness.update()
    update.request_cancel()

    summary = update.run()

    assert summary.cancelled and summary.added == 0
    assert harness.manifest.get_source("e.txt") is None
    run = harness.manifest.latest_run()
    assert run is not None and run.status == "cancelled"
    assert run.cancel_requested_at is not None
    metadata = harness.store.read_completion_metadata()
    assert metadata is not None and "last_updated_at" not in metadata

    assert harness.update().run().added == 1


def test_a_dead_update_run_is_closed_by_the_next(harness: Harness) -> None:
    dead = harness.manifest.start_run({}, kind="update")

    harness.update().run()

    runs = harness.manifest._runs  # the fake's own list
    assert [(r.run_id, r.kind, r.status) for r in runs[-2:]] == [
        (dead, "update", "aborted"),
        (str(int(dead) + 1), "update", "completed"),
    ]


def test_refused_without_a_published_index() -> None:
    harness = Harness(FILES)

    with pytest.raises(IndexIncomplete):
        harness.update().run()

    assert harness.manifest.latest_run() is None


def test_refused_on_config_drift(harness: Harness) -> None:
    update = harness.update(config={**CONFIG, "chunk_overlap": 2})
    before = harness.manifest.latest_run()

    with pytest.raises(ConfigDrift) as refused:
        update.run()

    assert refused.value.details["fields"] == "chunk_overlap"
    assert harness.manifest.latest_run() == before


@pytest.mark.parametrize("status", ["aborted", "cancelled", "running", "swapped"])
def test_refused_while_the_latest_build_is_unfinished(
    harness: Harness, status: str
) -> None:
    run_id = harness.manifest.start_run({})
    if status == "swapped":
        harness.manifest.mark_swapped(run_id)
    elif status != "running":
        harness.manifest.finish_run(run_id, cast(Finished, status), {})

    with pytest.raises(ResumeAvailable) as refused:
        harness.update().run()

    assert refused.value.details["run_id"] == run_id
    assert harness.manifest.latest_run_of_kind("update") is None


def test_an_unfinished_update_does_not_make_a_build_resumable(
    harness: Harness,
) -> None:
    harness.corpus.files["e.txt"] = ("new words here", 2.0)
    update = harness.update()
    update.request_cancel()
    update.run()
    build = harness.build()

    summary = build.run(refuse_resumable=True)

    assert not summary.cancelled
    assert (summary.indexed, summary.skipped) == (4, 1)


def test_refused_while_the_lock_is_held(harness: Harness) -> None:
    harness.lock.held = True

    with pytest.raises(IndexLockHeld):
        harness.update().run()

    assert harness.manifest.latest_run_of_kind("update") is None


def test_the_marker_refuses_build_keys(harness: Harness) -> None:
    from docgrep.domain.errors import ValidationError

    with pytest.raises(ValidationError):
        harness.store.merge_completion_metadata({"chunk_size": 1})


def test_build_records_a_declared_parser_version() -> None:
    harness = Harness(FILES)
    harness.parsers[FileType.TXT] = Parser(harness.corpus, version="3")

    harness.build().run()

    record = harness.manifest.get_source("a.txt")
    assert record is not None and record.source.parser_version == "3"
    markdown = harness.manifest.get_source("c.md")
    assert markdown is not None and markdown.source.parser_version is None
    stored = {item[1].parser_version for item in harness.store._items.values()}
    assert stored == {"3", None}


def test_the_plan_splits_modified_by_reason(harness: Harness) -> None:
    files = harness.corpus.files
    files["b.txt"] = ("six seven changed", 2.0)  # content changed
    harness.parsers[FileType.MARKDOWN] = Parser(harness.corpus, version="2")
    harness.parsers[FileType.EPUB] = harness.parser  # d.epub was unsupported

    plan = harness.update().run().plan

    assert (plan.content_changed, plan.parser_changed, plan.newly_supported) == (
        1,
        1,
        1,
    )
    assert plan.modified == 3 and plan.retried == 0


@pytest.mark.parametrize("recorded", [None, ""])
def test_a_missing_recorded_version_counts_as_version_1(
    harness: Harness, recorded: str | None
) -> None:
    for path in ("a.txt", "b.txt", "c.md"):
        record = harness.manifest.get_source(path)
        assert record is not None
        stale = replace(record.source, parser_version=recorded)
        harness.manifest.record_source(stale, record.status, None, "model")
    versioned = Parser(harness.corpus, version="1")
    harness.parsers[FileType.TXT] = versioned
    harness.parsers[FileType.MARKDOWN] = versioned

    assert harness.update().run().no_changes

    harness.parsers[FileType.TXT] = Parser(harness.corpus, version="2")
    plan = harness.update().run().plan
    assert (plan.parser_changed, plan.unchanged) == (2, 2)


def test_a_parser_without_a_version_matches_a_recorded_1(harness: Harness) -> None:
    record = harness.manifest.get_source("a.txt")
    assert record is not None
    harness.manifest.record_source(
        replace(record.source, parser_version="1"), record.status, None, "model"
    )

    assert harness.update().run().no_changes
