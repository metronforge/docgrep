"""Reusing the stored passages of a file with the same content (ADR-0028),
through domain ports and in-memory fakes."""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

import pytest

from docgrep.adapters.chunking import TokenChunker, WhitespaceTokenizer
from docgrep.adapters.models.deterministic import DeterministicEmbedder
from docgrep.domain.errors import StoreUnavailable
from docgrep.domain.models import (
    FileType,
    ManifestSourceRecord,
    OcrDelegation,
    OcrRecord,
    ParsedDocument,
    ParsedSection,
    Passage,
    ScannedFile,
    Source,
    make_passage_id,
)
from docgrep.domain.update_plan import (
    Copy,
    Move,
    can_reuse,
    reused_passage,
    reused_source,
)
from docgrep.usecases.build_index import BuildIndex
from docgrep.usecases.plan_update import PlanUpdate
from docgrep.usecases.update_index import UpdateIndex
from fakes import InMemoryManifest, InMemoryOcrRegistry, InMemoryPassageStore

CONFIG = {"chunk_size": 3, "chunk_overlap": 1, "tokenizer": "whitespace"}
BOOK = "the comet crosses the winter sky above the quiet harbour at night"
OTHER = "letters from a lighthouse keeper to his sister in the city"
NOTE = "a short note"


class Corpus:
    """Files as (text, mtime) by path."""

    def __init__(self, files: Mapping[str, tuple[str, float]]) -> None:
        self.files = dict(files)

    def _source(self, path: str) -> Source:
        text, mtime = self.files[path]
        return Source(
            source_id=path,
            path=path,
            file_type=FileType.TXT,
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
                extension=".txt",
                file_type=FileType.TXT,
                mtime=mtime,
            )
            for path, (text, mtime) in sorted(self.files.items())
        )

    def describe(self, path: str) -> Source:
        return self._source(path)


class CountingParser:
    def __init__(self, corpus: Corpus, version: str = "1") -> None:
        self.corpus = corpus
        self.version = version
        self.parsed: list[str] = []

    def parse(self, source: Source) -> ParsedDocument:
        self.parsed.append(source.path)
        text = self.corpus.files[source.path][0]
        return ParsedDocument(
            replace(source, title=f"Title of {text.split()[1]}"),
            (ParsedSection(text),),
        )


class CountingEmbedder(DeterministicEmbedder):
    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        self.texts.extend(texts)
        return super().embed(texts)


class Sparse:
    def encode(self, text: str) -> dict[int, float]:
        return {len(word): 1.0 for word in text.split()}


class Lock:
    @contextmanager
    def hold(self, operation: str = "build") -> Iterator[None]:
        yield

    def is_held(self) -> bool:
        return False


class Harness:
    def __init__(
        self,
        files: Mapping[str, tuple[str, float]],
        store: InMemoryPassageStore | None = None,
    ) -> None:
        self.corpus = Corpus(files)
        self.store = store or InMemoryPassageStore()
        self.manifest = InMemoryManifest()
        self.parser = CountingParser(self.corpus)
        self.embedder = CountingEmbedder()
        self.registry = InMemoryOcrRegistry()

    def _chunker(self) -> TokenChunker:
        return TokenChunker(WhitespaceTokenizer(), 3, 1)

    def build(self, **extra: Any) -> Any:
        summary = BuildIndex(
            lock=Lock(),
            enumerator=self.corpus,
            parsers={FileType.TXT: self.parser},
            chunker=self._chunker(),
            embedder=self.embedder,
            sparse_encoder=Sparse(),
            passage_writer=self.store,
            manifest=self.manifest,
            config=CONFIG,
            docgrep_version="test",
            sleep=lambda seconds: None,
            delegations=self.registry,
            ocr_outputs=self.registry,
            **({"reuse_by_hash": True} | extra),
        ).run()
        self.forget_calls()
        return summary

    def forget_calls(self) -> None:
        self.parser.parsed.clear()
        self.embedder.texts.clear()

    def update(self, **extra: Any) -> UpdateIndex:
        return UpdateIndex(
            lock=Lock(),
            corpus=self.corpus,
            parsers={FileType.TXT: self.parser},
            chunker=self._chunker(),
            embedder=self.embedder,
            sparse_encoder=Sparse(),
            index=self.store,
            manifest=self.manifest,
            config=CONFIG,
            sleep=lambda seconds: None,
            now=lambda: datetime(2026, 10, 18, tzinfo=UTC),
            delegations=self.registry,
            ocr_outputs=self.registry,
            **({"reuse_by_hash": True} | extra),
        )

    def planner(self, **extra: Any) -> PlanUpdate:
        return PlanUpdate(
            corpus=self.corpus,
            manifest=self.manifest,
            parsers={FileType.TXT: self.parser},
            delegations=self.registry,
            ocr_outputs=self.registry,
            **({"reuse_by_hash": True} | extra),
        )

    def points(self, path: str) -> dict[int, tuple[Passage, Source, Any, Any]]:
        """The stored passages of a source by chunk offset."""
        return {
            item[0].chunk_offset: item
            for item in self.store._items.values()
            if item[1].path == path
        }

    def hits(self, query: str) -> list[tuple[str, int, str, float]]:
        vector = DeterministicEmbedder().embed([query])[0]
        found = self.store.retrieve(vector, Sparse().encode(query), 50)
        return [
            (
                hit.source.path,
                hit.passage.chunk_offset,
                hit.passage.text,
                hit.dense_score,
            )
            for hit in found
        ]


def _library() -> Harness:
    harness = Harness(
        {"a.txt": (BOOK, 1.0), "b.txt": (OTHER, 1.0), "c.txt": (NOTE, 1.0)}
    )
    harness.build()
    return harness


def test_a_moved_file_is_copied_without_parsing_or_embedding() -> None:
    harness = _library()
    before = harness.hits("comet winter harbour")
    old = harness.points("a.txt")
    harness.corpus.files["shelf/a.txt"] = harness.corpus.files.pop("a.txt")

    summary = harness.update().run()

    assert harness.parser.parsed == [] and harness.embedder.texts == []
    assert (summary.moved, summary.copied, summary.added, summary.removed) == (
        1,
        0,
        0,
        0,
    )
    assert (summary.indexed, summary.failed) == (1, 0)
    assert summary.passages_written == len(old)
    assert summary.passages_removed == len(old)
    # Search finds the same passages with the same scores, under the new path;
    # equal scores are ordered by passage id, which follows the path.
    assert sorted(harness.hits("comet winter harbour")) == sorted(
        ("shelf/a.txt" if path == "a.txt" else path, offset, text, score)
        for path, offset, text, score in before
    )
    assert harness.points("a.txt") == {}
    assert harness.manifest.get_source("a.txt") is None
    new = harness.points("shelf/a.txt")
    assert sorted(new) == sorted(old)
    for offset, (passage, source, dense, sparse) in new.items():
        assert passage.passage_id == make_passage_id("shelf/a.txt", offset)
        assert passage.text == old[offset][0].text
        assert (dense, sparse) == (old[offset][2], old[offset][3])
        assert (source.source_id, source.path) == ("shelf/a.txt", "shelf/a.txt")
        assert source.title == "Title of comet"
    record = harness.manifest.get_source("shelf/a.txt")
    assert record is not None and record.status == "indexed"
    assert record.embedding_model == DeterministicEmbedder.model_name
    assert record.source.title == "Title of comet"
    assert record.source.parser_version == "1"
    assert record.reused_from is None
    journal = {r.source.path: r for r in harness.manifest.run_sources(summary.run_id)}
    assert journal["shelf/a.txt"].reused_from == "a.txt"
    assert journal["shelf/a.txt"].passages == len(old)
    assert journal["a.txt"].status == "removed"
    run = harness.manifest.latest_run()
    assert run is not None
    assert dict(run.totals)["moved"] == 1 and dict(run.totals)["copied"] == 0
    # Nothing is left to do.
    assert harness.planner().run().plan.changes == 0


class RecordingStore(InMemoryPassageStore):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, str]] = []

    def upsert_live_batch(self, items: Any) -> None:
        self.calls.extend(("upsert", item[1].path) for item in items[:1])
        super().upsert_live_batch(items)

    def delete_live_source_points(self, source_id: str) -> int:
        self.calls.append(("delete", source_id))
        return super().delete_live_source_points(source_id)


def test_the_copy_is_written_before_the_old_source_is_removed() -> None:
    store = RecordingStore()
    harness = Harness({"a.txt": (BOOK, 1.0), "b.txt": (OTHER, 1.0)}, store=store)
    harness.build()
    harness.corpus.files["z.txt"] = harness.corpus.files.pop("a.txt")

    harness.update().run()

    assert store.calls == [("upsert", "z.txt"), ("delete", "a.txt")]


def test_a_copied_file_gives_two_sources_with_the_same_passages() -> None:
    harness = _library()
    harness.corpus.files["backup/a.txt"] = harness.corpus.files["a.txt"]

    preview = harness.planner().run()
    summary = harness.update().run()

    assert preview.roots[0].copied == (Copy("a.txt", "backup/a.txt"),)
    assert preview.plan.copied == 1 and preview.plan.added == 0
    assert preview.plan == summary.plan
    assert harness.parser.parsed == [] and harness.embedder.texts == []
    assert (summary.copied, summary.moved, summary.indexed) == (1, 0, 1)
    original, copy = harness.points("a.txt"), harness.points("backup/a.txt")
    assert sorted(original) == sorted(copy)
    for offset, (passage, _, dense, sparse) in copy.items():
        assert passage.text == original[offset][0].text
        assert (dense, sparse) == (original[offset][2], original[offset][3])
    journal = {r.source.path: r for r in harness.manifest.run_sources(summary.run_id)}
    assert journal["backup/a.txt"].reused_from == "a.txt"
    assert set(journal) == {"backup/a.txt"}


def test_a_file_whose_content_became_another_s_is_a_copy() -> None:
    harness = _library()
    harness.corpus.files["c.txt"] = (BOOK, 2.0)

    summary = harness.update().run()

    assert harness.parser.parsed == [] and harness.embedder.texts == []
    assert (summary.copied, summary.modified) == (1, 0)
    copy = harness.points("c.txt")
    assert sorted(copy) == sorted(harness.points("a.txt"))
    assert all(item[0].text in BOOK for item in copy.values())
    record = harness.manifest.get_source("c.txt")
    assert record is not None and record.source.mtime == 2.0
    assert record.source.content_hash == sha256(BOOK.encode()).hexdigest()


def test_a_full_build_embeds_identical_files_once() -> None:
    files = {"a.txt": (BOOK, 1.0), "b.txt": (OTHER, 1.0), "copy.txt": (BOOK, 2.0)}
    harness = Harness(files)
    embedder = harness.embedder

    summary = BuildIndex(
        lock=Lock(),
        enumerator=harness.corpus,
        parsers={FileType.TXT: harness.parser},
        chunker=harness._chunker(),
        embedder=embedder,
        sparse_encoder=Sparse(),
        passage_writer=harness.store,
        manifest=harness.manifest,
        config=CONFIG,
        docgrep_version="test",
        sleep=lambda seconds: None,
        reuse_by_hash=True,
    ).run()

    passages_a = harness.points("a.txt")
    passages_b = harness.points("b.txt")
    assert len(embedder.texts) == len(passages_a) + len(passages_b)
    assert harness.parser.parsed == ["a.txt", "b.txt"]
    assert (summary.indexed, summary.copied) == (3, 1)
    copy = harness.points("copy.txt")
    assert {k: (v[0].text, v[2]) for k, v in copy.items()} == {
        k: (v[0].text, v[2]) for k, v in passages_a.items()
    }
    journal = {r.source.path: r for r in harness.manifest.run_sources(summary.run_id)}
    assert journal["copy.txt"].reused_from == "a.txt"
    assert journal["a.txt"].reused_from is None
    record = harness.manifest.get_source("copy.txt")
    assert record is not None and record.status == "indexed"
    assert record.source.mtime == 2.0 and record.reused_from is None
    run = harness.manifest.latest_run()
    assert run is not None and dict(run.totals)["copied"] == 1


def test_a_full_build_never_reuses_the_previous_index() -> None:
    harness = _library()
    first = len(harness.store._items)

    harness.build()
    harness.forget_calls()
    summary = BuildIndex(
        lock=Lock(),
        enumerator=harness.corpus,
        parsers={FileType.TXT: harness.parser},
        chunker=harness._chunker(),
        embedder=harness.embedder,
        sparse_encoder=Sparse(),
        passage_writer=harness.store,
        manifest=harness.manifest,
        config=CONFIG,
        docgrep_version="test",
        sleep=lambda seconds: None,
        reuse_by_hash=True,
    ).run()

    assert summary.copied == 0
    assert len(harness.embedder.texts) == first
    assert harness.parser.parsed == ["a.txt", "b.txt", "c.txt"]


def test_without_reuse_a_move_is_a_removal_and_an_addition() -> None:
    harness = _library()
    harness.corpus.files["shelf/a.txt"] = harness.corpus.files.pop("a.txt")

    preview = harness.planner(reuse_by_hash=False).run()
    summary = harness.update(reuse_by_hash=False).run()

    assert (preview.plan.added, preview.plan.removed, preview.plan.moved) == (1, 1, 0)
    assert (summary.added, summary.removed, summary.moved) == (1, 1, 0)
    assert harness.parser.parsed == ["shelf/a.txt"]
    assert harness.embedder.texts
    journal = {r.source.path: r for r in harness.manifest.run_sources(summary.run_id)}
    assert journal["shelf/a.txt"].reused_from is None


def test_a_full_build_without_reuse_embeds_every_file() -> None:
    harness = Harness({"a.txt": (BOOK, 1.0), "copy.txt": (BOOK, 1.0)})

    summary = harness.build(reuse_by_hash=False)

    assert summary.copied == 0
    run = harness.manifest.latest_run()
    assert run is not None and "copied" not in dict(run.totals)


def _move_a(harness: Harness) -> None:
    harness.corpus.files["shelf/a.txt"] = harness.corpus.files.pop("a.txt")


def _parser_changed(harness: Harness) -> None:
    harness.parser.version = "2"
    _move_a(harness)


def _model_changed(harness: Harness) -> None:
    record = harness.manifest.get_source("a.txt")
    assert record is not None
    harness.manifest.record_source(record.source, "indexed", None, "another-model")
    _move_a(harness)


def _not_indexed(harness: Harness) -> None:
    record = harness.manifest.get_source("a.txt")
    assert record is not None
    harness.manifest.record_source(record.source, "skipped:requires_ocr")
    _move_a(harness)


def _ocr_output(harness: Harness) -> None:
    harness.registry.register(
        OcrRecord(
            input_path="scan.pdf",
            input_format="pdf",
            input_hash="scan",
            output_path="a.txt",
            output_hash="out",
            engine="ocrmypdf",
            engine_version="1",
            languages="eng",
            mode="skip-text",
        )
    )
    harness.corpus.files["copy.txt"] = harness.corpus.files["a.txt"]


@pytest.mark.parametrize(
    "change",
    [_parser_changed, _model_changed, _not_indexed, _ocr_output],
    ids=["parser_version", "embedding_model", "status", "ocr_output"],
)
def test_a_source_that_cannot_be_reused_is_processed_normally(change: Any) -> None:
    harness = _library()
    change(harness)

    preview = harness.planner().run()
    summary = harness.update().run()

    assert (preview.plan.moved, preview.plan.copied) == (0, 0)
    assert (summary.moved, summary.copied) == (0, 0)
    new = "copy.txt" if "copy.txt" in harness.corpus.files else "shelf/a.txt"
    assert new in harness.parser.parsed
    assert harness.embedder.texts
    record = harness.manifest.get_source(new)
    assert record is not None and record.status == "indexed"


def test_a_file_delegated_to_its_ocr_output_is_not_a_copy() -> None:
    harness = _library()
    harness.corpus.files["scan.txt"] = harness.corpus.files["a.txt"]
    digest = sha256(BOOK.encode()).hexdigest()
    harness.registry.delegations["scan.txt"] = OcrDelegation(
        "scan.txt", digest, "scan.ocr.pdf"
    )

    summary = harness.update().run()

    assert (summary.copied, summary.added) == (0, 1)
    assert harness.parser.parsed == ["scan.txt"]


class FailingCopies(InMemoryPassageStore):
    """Fail every live upsert of one source after the first, retries
    included."""

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        self.calls = 0
        self.failing = True

    def upsert_live_batch(self, items: Any) -> None:
        if self.failing and items and items[0][1].path == self.path:
            self.calls += 1
            if self.calls >= 2:
                raise StoreUnavailable("disk full")
        super().upsert_live_batch(items)


def test_a_failed_copy_leaves_no_passages_and_keeps_the_old_source() -> None:
    store = FailingCopies("shelf/a.txt")
    harness = Harness(
        {"a.txt": (BOOK, 1.0), "b.txt": (OTHER, 1.0)},
        store=store,
    )
    harness.build()
    old = harness.points("a.txt")
    assert len(old) > 1
    _move_a(harness)

    summary = harness.update(upsert_batch_size=1).run()

    # The second page failed, and its two retries too.
    assert store.calls == 4
    assert (summary.failed, summary.indexed) == (1, 0)
    assert summary.failures[0].reason == "store_error"
    assert harness.points("shelf/a.txt") == {}
    assert harness.points("a.txt") == old
    kept = harness.manifest.get_source("a.txt")
    assert kept is not None and kept.status == "indexed"
    failed = harness.manifest.get_source("shelf/a.txt")
    assert failed is not None and failed.status == "failed:store_error"

    # The failed file is retried, through the normal pipeline (ADR-0028
    # reuses for added files and content changes only).
    store.failing = False
    summary = harness.update().run()

    assert (summary.retried, summary.removed, summary.failed) == (1, 1, 0)
    assert harness.parser.parsed == ["shelf/a.txt"]
    assert harness.points("a.txt") == {}
    assert sorted(harness.points("shelf/a.txt")) == sorted(old)


class FailingRemoval(InMemoryPassageStore):
    def __init__(self) -> None:
        super().__init__()
        self.failing = True

    def delete_live_source_points(self, source_id: str) -> int:
        if self.failing:
            raise StoreUnavailable("store gone")
        return super().delete_live_source_points(source_id)


def test_a_move_whose_old_source_stays_is_finished_by_the_next_update() -> None:
    store = FailingRemoval()
    harness = Harness({"a.txt": (BOOK, 1.0), "b.txt": (OTHER, 1.0)}, store=store)
    harness.build()
    _move_a(harness)

    summary = harness.update().run()

    # The copy is stored and recorded; the old source stays searchable.
    assert summary.failed == 1 and summary.failures[0].reason == "removal_failed"
    assert harness.points("a.txt") and harness.points("shelf/a.txt")
    assert harness.manifest.get_source("shelf/a.txt") is not None
    assert harness.manifest.get_source("a.txt") is not None

    store.failing = False
    preview = harness.planner().run()
    summary = harness.update().run()

    assert (preview.plan.removed, preview.plan.changes) == (1, 1)
    assert summary.removed == 1 and summary.failed == 0
    assert harness.points("a.txt") == {}
    assert harness.manifest.get_source("a.txt") is None


def test_moves_do_not_count_against_the_removal_guard() -> None:
    harness = _library()
    _move_a(harness)

    detected = harness.planner().detect(retry_failed=True)

    assert detected.moves == (Move("a.txt", "shelf/a.txt"),)
    assert detected.removals() == {"_legacy": 0}
    assert detected.paired_removals() == {"_legacy": 1}


def test_the_plan_takes_the_model_of_the_latest_build_by_default() -> None:
    harness = _library()
    _move_a(harness)

    assert harness.planner().run().plan.moved == 1
    assert harness.planner(embedding_model="another-model").run().plan.moved == 0


def _record(**changes: Any) -> ManifestSourceRecord:
    source = Source("a.txt", "a.txt", FileType.TXT, content_hash="h", size=3)
    record = ManifestSourceRecord(
        replace(source, parser_version="2"), "indexed", embedding_model="m"
    )
    return replace(record, **changes)


def _file(**changes: Any) -> Source:
    return replace(
        Source("b.txt", "b.txt", FileType.TXT, content_hash="h", size=3), **changes
    )


@pytest.mark.parametrize(
    ("record", "file", "reusable"),
    [
        (_record(), _file(), True),
        (_record(status="skipped:unsupported"), _file(), False),
        (_record(), _file(content_hash="other"), False),
        (_record(), _file(size=4), False),
        (_record(), _file(file_type=FileType.MARKDOWN), False),
        (_record(embedding_model="other"), _file(), False),
        (_record(embedding_model=None), _file(), False),
        (_record(source=_file(content_hash=None)), _file(content_hash=None), False),
    ],
)
def test_the_reuse_rule(
    record: ManifestSourceRecord, file: Source, reusable: bool
) -> None:
    assert can_reuse(record, file, parser_version="2", embedding_model="m") is reusable


def test_the_reuse_rule_compares_effective_parser_versions() -> None:
    legacy = _record(source=_file(parser_version=None))

    assert can_reuse(legacy, _file(), parser_version="1", embedding_model="m")
    assert not can_reuse(legacy, _file(), parser_version="2", embedding_model="m")
    assert not can_reuse(_record(), _file(), parser_version="3", embedding_model="m")


def test_a_reused_passage_and_source_take_the_new_identity() -> None:
    original = Source(
        "a.txt",
        "a.txt",
        FileType.TXT,
        title="T",
        author="A",
        year=1901,
        content_hash="h",
        mtime=1.0,
        size=3,
        indexed_at="then",
        parser_version="2",
    )
    file = Source(
        "n/b.txt", "n/b.txt", FileType.TXT, content_hash="h", mtime=5.0, size=3
    )
    passage = Passage(make_passage_id("a.txt", 7), "a.txt", "text", 7, 1, 2, ("H",))

    source = reused_source(original, file)
    copied = reused_passage(passage, source.source_id)

    assert source == replace(
        original, source_id="n/b.txt", path="n/b.txt", mtime=5.0, indexed_at=None
    )
    assert copied == replace(
        passage, passage_id=make_passage_id("n/b.txt", 7), source_id="n/b.txt"
    )
