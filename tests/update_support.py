"""A generated corpus with known changes, and helpers for the update tests.

Every file gets a fixed modification time, so a corpus written twice, in two
directories, is identical down to the stored payloads.
"""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from docgrep.adapters.manifest import SQLiteManifest
from docgrep.api.container import Services
from docgrep.domain.errors import StoreUnavailable
from docgrep.domain.models import FileType
from docgrep.settings import Settings
from epub_support import epub_bytes, xhtml
from resume_support import new_services

OLD = 1_700_000_000
NEW = 1_700_100_000

_WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega"
).split()


def _text(seed: int, words: int = 30) -> bytes:
    body = [f"marker{seed}"] + [
        _WORDS[(seed * 7 + step * 3) % len(_WORDS)] for step in range(words)
    ]
    return " ".join(body).encode()


def _book(seed: int, title: str) -> bytes:
    return epub_bytes(
        [
            (
                "one.xhtml",
                xhtml(f"<h1>{title}</h1><p>{_text(seed).decode()}</p>"),
            ),
            (
                "two.xhtml",
                xhtml(f"<h2>Later</h2><p>{_text(seed + 50).decode()}</p>"),
            ),
        ],
        title=title,
        creators=(f"Author {seed}",),
        date=f"{1900 + seed}",
    )


def _note(seed: int) -> bytes:
    return (
        f"---\ntitle: Note {seed}\n---\n# Note {seed}\n\n"
        f"{_text(seed).decode()}\n\n## Part\n\n{_text(seed + 100).decode()}\n"
    ).encode()


# path: (content, mtime)
INITIAL: dict[str, tuple[bytes, int]] = {
    **{f"doc{i:02d}.txt": (_text(i), OLD) for i in range(6)},
    "notes/a.md": (_note(20), OLD),
    "notes/b.md": (_note(21), OLD),
    "books/one.epub": (_book(30, "One"), OLD),
    "books/two.epub": (_book(31, "Two"), OLD),
    "misc/letter.docx": (b"not parsed yet", OLD),
}

# The final corpus: two removed, two edited, one touched (same content, new
# time), two added, and two EPUBs that only a newly registered parser reads.
FINAL: dict[str, tuple[bytes, int]] = {
    **{k: v for k, v in INITIAL.items() if k not in ("doc00.txt", "notes/b.md")},
    "doc01.txt": (_text(91), NEW),
    "notes/a.md": (_note(92), NEW),
    "doc02.txt": (INITIAL["doc02.txt"][0], NEW),
    "doc10.txt": (_text(10), NEW),
    "books/three.epub": (_book(32, "Three"), NEW),
}

# What an update from INITIAL to FINAL plans, given an EPUB parser that the
# initial build did not have.
PLAN = {
    "added": 2,
    "modified": 4,
    "removed": 2,
    "unchanged": 5,
    "retried": 0,
    "refreshed": 1,
}
CHANGES = PLAN["added"] + PLAN["modified"] + PLAN["removed"] + PLAN["retried"]
QUERIES = ("marker91 alpha", "marker30", "Later omega", "marker21", "beta gamma")


def write_files(root: Path, files: Mapping[str, tuple[bytes, int]]) -> None:
    for name, (content, mtime) in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        os.utime(path, (mtime, mtime))


def change_corpus(root: Path) -> None:
    """Turn a directory holding INITIAL into one holding FINAL."""
    for name in set(INITIAL) - set(FINAL):
        (root / name).unlink()
    write_files(root, {k: v for k, v in FINAL.items() if INITIAL.get(k) != v})


def without_epub(services: Services) -> None:
    """Make the build of this container lack the EPUB parser."""
    build: Any = services.build
    build.parsers = {
        file_type: parser
        for file_type, parser in build.parsers.items()
        if file_type is not FileType.EPUB
    }


def update_services(settings: Settings) -> Services:
    """A fresh container whose update does not sleep between retries."""
    services = new_services(settings)
    update: Any = services.update
    update.sleep = lambda seconds: None
    return services


def sources_table(settings: Settings) -> list[tuple[Any, ...]]:
    """The committed sources without the time each was written."""
    return [
        (
            r.source.path,
            r.source.source_id,
            r.source.file_type,
            r.source.title,
            r.source.author,
            r.source.year,
            r.source.content_hash,
            r.source.mtime,
            r.source.size,
            r.source.parser_version,
            r.status,
            r.error,
            r.embedding_model,
        )
        for r in SQLiteManifest(settings.data_dir).committed_sources()
    ]


class HookedManifest:
    """Delegate to the real manifest; call a hook after the n-th journal write."""

    def __init__(self, inner: Any, at: int, hook: Any) -> None:
        self.inner = inner
        self.at = at
        self.hook = hook
        self.journaled = 0

    def record_source(self, *args: Any, **kwargs: Any) -> None:
        self.inner.record_source(*args, **kwargs)
        if kwargs.get("run_id") is not None:
            self.journaled += 1
            if self.journaled == self.at:
                self.hook()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def cancel_after_sources(services: Services, count: int) -> None:
    """Request a cancel once ``count`` changes are journaled (0: right away)."""
    update: Any = services.update
    if count == 0:
        update.request_cancel()
        return
    update.manifest = HookedManifest(update.manifest, count, update.request_cancel)


class FailingLiveUpserts:
    """Delegate to the real store; fail every live upsert from the ``at``-th."""

    def __init__(self, inner: Any, at: int, error: BaseException | None = None) -> None:
        self.inner = inner
        self.at = at
        self.calls = 0
        self.error = error

    def upsert_live_batch(self, items: Any) -> None:
        self.calls += 1
        if self.calls >= self.at:
            raise self.error or StoreUnavailable("injected failure")
        self.inner.upsert_live_batch(items)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def fail_live_upsert_at(
    services: Services, call: int, error: BaseException | None = None
) -> None:
    update: Any = services.update
    update.index = FailingLiveUpserts(update.index, call, error)


class Versioned:
    """A real parser declaring another version (None: declaring none)."""

    def __init__(self, inner: Any, version: str | None) -> None:
        self.inner = inner
        if version is not None:
            self.version = version

    def parse(self, source: Any) -> Any:
        return self.inner.parse(source)


def with_version(operation: Any, file_type: FileType, version: str | None) -> None:
    """Make a build or update of a container see another parser version."""
    operation.parsers = {
        **operation.parsers,
        file_type: Versioned(operation.parsers[file_type], version),
    }
