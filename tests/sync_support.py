"""A real container (fake backend, local Qdrant, SQLite) with automatic sync
under an injected clock, for the sync tests."""

import os
import time
from pathlib import Path
from typing import Any

from docgrep.api.container import Services, create_container
from docgrep.settings import Settings
from docgrep.usecases.sync_index import SyncIndex
from resume_support import close
from roots_support import save_roots, settings

OLD = 1_700_000_000
BOOKS = {
    f"book{i:02d}.txt": f"Book number {i}: a long tale of rivers, harbours, {i} lamps."
    for i in range(12)
}
NOTES = {
    "x.txt": "A yellow kite drifts over the quiet meadow.",
    "y.txt": "Notes on a walk along the river after rain.",
}


def write(root: Path, files: dict[str, str], mtime: float = OLD) -> None:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        os.utime(path, (mtime, mtime))


class Clock:
    """Time that moves only when told to; wall time starts now."""

    def __init__(self) -> None:
        self.mono = 0.0
        self.wall = time.time()

    def advance(self, seconds: float = 120) -> None:
        self.mono += seconds
        self.wall += seconds


class Rig:
    """Roots ``books`` (legacy) and ``notes``, built once, then synced."""

    def __init__(self, tmp_path: Path, *, notes: bool = True, **overrides: Any) -> None:
        self.books = tmp_path / "books"
        self.notes = tmp_path / "notes"
        self.data = tmp_path / "data"
        write(self.books, BOOKS)
        roots = [("books", self.books, True)]
        if notes:
            write(self.notes, NOTES)
            roots.append(("notes", self.notes, False))
        save_roots(self.data, *roots)
        self.overrides = overrides
        self.clock = Clock()
        self.services: Services | None = None
        self.build()

    def settings(self, **extra: Any) -> Settings:
        return settings(self.data, **{**self.overrides, **extra})

    def build(self) -> None:
        services = create_container(self.settings())
        services.build.run()
        close(services)

    def open(self, **extra: Any) -> Services:
        """A new process with sync on and the clock under control."""
        services = create_container(self.settings(sync=True, **extra))
        update: Any = services.update
        update.sleep = lambda seconds: None
        sync = services.sync
        assert sync is not None
        sync.clock = lambda: self.clock.mono
        sync.wall_clock = lambda: self.clock.wall
        sync.sleep = lambda seconds: None
        self.services = services
        return services

    @property
    def sync(self) -> SyncIndex:
        assert self.services is not None and self.services.sync is not None
        return self.services.sync

    def cycle(self, advance: float = 120) -> Any:
        self.clock.advance(advance)
        return self.sync.run_cycle()

    def close(self) -> None:
        if self.services is not None:
            close(self.services)
            self.services = None

    def sources(self) -> dict[str, str]:
        """The committed sources, path to status."""
        assert self.services is not None
        manifest: Any = self.services.update.manifest  # type: ignore[attr-defined]  # container internals
        return {r.source.path: r.status for r in manifest.committed_sources()}
