"""Corpus roots in the indexing use cases, through domain ports and in-memory
fakes (ADR-0023)."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from docgrep.adapters.chunking import TokenChunker, WhitespaceTokenizer
from docgrep.adapters.models.deterministic import DeterministicEmbedder
from docgrep.domain.errors import ConfigDrift, ValidationError
from docgrep.domain.models import (
    FileType,
    ManifestSourceRecord,
    ParsedDocument,
    ParsedSection,
    ScannedFile,
    Source,
)
from docgrep.domain.roots import CorpusRoot, CorpusRoots
from docgrep.domain.sources import BrowseRequest, SourceFilter
from docgrep.domain.status import RootStatusView
from docgrep.usecases.browse_sources import BrowseSources
from docgrep.usecases.build_index import BuildIndex
from docgrep.usecases.inspect_status import InspectStatus
from docgrep.usecases.update_index import UpdateIndex
from fakes import InMemoryManifest, InMemoryPassageStore

BOOKS = CorpusRoot("books", Path("/srv/books"), legacy=True)
NOTES = CorpusRoot("notes", Path("/srv/notes"))
CONFIG = {"chunk_size": 3, "chunk_overlap": 1, "tokenizer": "whitespace"}


class Corpus:
    """Files by stored path; ``readable`` is false for an unmounted root."""

    def __init__(self, files: Mapping[str, str]) -> None:
        self.files = dict(files)
        self.unreadable: str | None = None
        self.checked = 0

    def require_available(self) -> None:
        self.checked += 1
        if self.unreadable is not None:
            raise ValidationError(
                f"corpus root {self.unreadable!r} is not accessible",
                details={"root": self.unreadable},
            )

    def _source(self, path: str) -> Source:
        text = self.files[path]
        return Source(
            source_id=path,
            path=path,
            file_type=FileType.TXT,
            content_hash=sha256(text.encode()).hexdigest(),
            mtime=1.0,
            size=len(text),
        )

    def enumerate(self) -> tuple[Source, ...]:
        return tuple(self._source(path) for path in sorted(self.files))

    def scan(self) -> tuple[ScannedFile, ...]:
        return tuple(
            ScannedFile(path=path, size=len(text), extension=".txt", mtime=1.0)
            for path, text in sorted(self.files.items())
        )

    def describe(self, path: str) -> Source:
        return self._source(path)


class Parser:
    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus

    def parse(self, source: Source) -> ParsedDocument:
        return ParsedDocument(source, (ParsedSection(self.corpus.files[source.path]),))


class Lock:
    @contextmanager
    def hold(self, operation: str = "build") -> Iterator[None]:
        yield

    def is_held(self) -> bool:
        return False


class Catalog:
    def __init__(self, roots: CorpusRoots) -> None:
        self.current = roots

    def roots(self) -> CorpusRoots:
        return self.current


class Harness:
    def __init__(self, roots: CorpusRoots, files: Mapping[str, str]) -> None:
        self.corpus = Corpus(files)
        self.catalog = Catalog(roots)
        self.manifest = InMemoryManifest()
        self.store = InMemoryPassageStore()

    def _common(self) -> dict[str, Any]:
        return {
            "lock": Lock(),
            "parsers": {FileType.TXT: Parser(self.corpus)},
            "chunker": TokenChunker(WhitespaceTokenizer(), 3, 1),
            "embedder": DeterministicEmbedder(),
            "sparse_encoder": _Sparse(),
            "manifest": self.manifest,
            "config": CONFIG,
            "sleep": lambda seconds: None,
            "roots": self.catalog,
            "availability": self.corpus,
        }

    def build(self) -> BuildIndex:
        return BuildIndex(
            enumerator=self.corpus,
            passage_writer=self.store,
            docgrep_version="test",
            **self._common(),
        )

    def update(self) -> UpdateIndex:
        return UpdateIndex(corpus=self.corpus, index=self.store, **self._common())

    def committed(self) -> list[str]:
        return [r.source.path for r in self.manifest.committed_sources()]


class _Sparse:
    def encode(self, text: str) -> dict[int, float]:
        return {len(text): 1.0}


FILES = {
    "a.txt": "rare blue comet",
    "notes/a.txt": "yellow kite meadow",
    "notes/c.txt": "rye bread seeds",
}


def test_runs_record_their_roots_and_the_prefixes_of_the_index() -> None:
    harness = Harness(CorpusRoots((BOOKS, NOTES)), FILES)
    harness.build().run()

    run = dict(harness.manifest._runs[-1].config)
    assert run["corpus_roots"] == CorpusRoots((BOOKS, NOTES)).snapshot()
    assert run["root_prefixes"] == '["notes"]'
    assert harness.committed() == ["a.txt", "notes/a.txt", "notes/c.txt"]


def test_an_update_leaves_the_sources_of_other_roots_alone() -> None:
    harness = Harness(CorpusRoots((BOOKS, NOTES)), FILES)
    harness.build().run()
    # The notes root leaves the configuration and its files are gone.
    harness.catalog.current = CorpusRoots((BOOKS,))
    harness.corpus.files = {"a.txt": FILES["a.txt"]}

    summary = harness.update().run()

    assert summary.removed == 0
    assert summary.plan.outside_run == (("notes", 2),)
    assert harness.committed() == ["a.txt", "notes/a.txt", "notes/c.txt"]
    # The update still records the prefix, so the next one knows it too.
    assert dict(harness.manifest._runs[-1].config)["root_prefixes"] == '["notes"]'
    assert harness.update().run().plan.outside_run == (("notes", 2),)


def test_an_unreadable_root_stops_a_run_before_it_starts() -> None:
    harness = Harness(CorpusRoots((BOOKS, NOTES)), FILES)
    harness.build().run()
    runs = len(harness.manifest._runs)
    harness.corpus.unreadable = "notes"
    harness.corpus.files = {"a.txt": FILES["a.txt"]}

    for operation in (harness.update().run, harness.build().run):
        with pytest.raises(ValidationError, match="corpus root 'notes'"):
            operation()

    assert len(harness.manifest._runs) == runs
    assert harness.committed() == ["a.txt", "notes/a.txt", "notes/c.txt"]


def test_a_full_build_refuses_to_drop_another_root() -> None:
    harness = Harness(CorpusRoots((BOOKS, NOTES)), FILES)
    harness.build().run()
    harness.catalog.current = CorpusRoots((BOOKS,))

    with pytest.raises(ValidationError) as raised:
        harness.build().run()

    assert raised.value.details == {"roots": {"notes": 2}}


def test_forget_root_removes_one_prefix_through_the_update_path() -> None:
    harness = Harness(CorpusRoots((BOOKS, NOTES)), FILES)
    harness.build().run()
    harness.catalog.current = CorpusRoots((BOOKS,))
    update = harness.update()

    with pytest.raises(ValidationError, match="is the legacy root"):
        update.forget_plan("books")
    planned = update.forget_plan("notes")
    assert update.passages_of(planned) == 2
    summary = update.forget_root("notes")

    assert (summary.removed, summary.passages_removed) == (2, 2)
    assert harness.committed() == ["a.txt"]
    assert dict(harness.manifest._runs[-1].config)["root_prefixes"] == "[]"
    with pytest.raises(ValidationError, match="unknown root 'notes'"):
        update.forget_plan("notes")


def test_a_resume_needs_the_roots_of_its_run() -> None:
    harness = Harness(CorpusRoots((BOOKS,)), {"a.txt": FILES["a.txt"]})
    build = harness.build()
    build.request_cancel()
    assert build.run().cancelled
    harness.catalog.current = CorpusRoots((BOOKS, NOTES))

    with pytest.raises(ConfigDrift, match="The corpus roots differ"):
        harness.build().resume()

    harness.catalog.current = CorpusRoots((BOOKS,))
    assert harness.build().resume().indexed == 1


def test_browse_names_and_filters_roots() -> None:
    harness = Harness(CorpusRoots((BOOKS, NOTES)), FILES)
    harness.build().run()
    browse = BrowseSources(
        manifest=harness.manifest,
        store=harness.store,
        ocr=_NoOcr(),
        roots=harness.catalog,
    )

    result = browse.run(BrowseRequest(SourceFilter(roots=("notes",))))

    assert [(item.source.path, item.root) for item in result.sources] == [
        ("notes/a.txt", "notes"),
        ("notes/c.txt", "notes"),
    ]
    assert result.applied_filters == {"roots": ["notes"]}
    with pytest.raises(ValidationError, match="known roots: books, notes"):
        browse.run(BrowseRequest(SourceFilter(roots=("music",))))


def test_status_counts_sources_per_root() -> None:
    harness = Harness(CorpusRoots((BOOKS, NOTES)), FILES)
    harness.build().run()
    harness.catalog.current = CorpusRoots((BOOKS,))
    status = InspectStatus(
        lock=Lock(),
        open_manifest=lambda: harness.manifest,
        open_store=lambda: None,
        store_mode="local",
        runtime_config={},
        roots=harness.catalog,
        root_readable=lambda root: False,
    )

    report = status.run()

    assert report.roots == (
        RootStatusView("books", True, True, False, 1, 0, 0),
        RootStatusView("notes", False, False, None, 2, 0, 0),
    )
    codes = {warning.code for warning in report.warnings}
    assert {"root_not_configured", "root_unreadable"} <= codes


class _NoOcr:
    def live_outputs(self) -> dict[str, Any]:
        return {}


def test_records_without_roots_read_as_before() -> None:
    record = ManifestSourceRecord(
        Source(source_id="a.txt", path="a.txt", file_type=FileType.TXT), "indexed"
    )
    assert CorpusRoots((BOOKS,)).root_of(record.source.source_id) == "books"
