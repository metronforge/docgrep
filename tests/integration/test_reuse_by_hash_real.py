"""Reusing stored passages for files with the same content (ADR-0028), over
real SQLite, local-mode Qdrant and the deterministic backend."""

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from docgrep.adapters.manifest import SQLiteManifest
from docgrep.api.cli import main
from docgrep.api.container import Services, create_container
from docgrep.domain.models import FileType, Source
from docgrep.settings import Settings
from resume_support import close, dump_points
from roots_support import environment, save_roots, settings

MTIME = 1_700_000_000
BOOKS = {
    "a.txt": "A rare blue comet crosses the winter sky above the harbour at "
    "night, and the keeper of the lighthouse writes it down in his ledger.",
    "shelf/b.md": "# Moss\n\nGarden notes about green moss on old stones.",
}
NOTES = {
    "c.txt": "Recipe for bread with rye flour and caraway seeds, baked slowly "
    "in a stone oven until the crust is dark.",
    # A second source, so that renaming the first does not trip the guard
    # for a root it would empty (ADR-0027 counts moves).
    "d.txt": "Notes on a walk along the river after the rain.",
}


def _write(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        os.utime(path, (MTIME, MTIME))
    return root.resolve()


def _move(root: Path, old: str, new: str, other: Path | None = None) -> None:
    target = (other or root) / new
    target.parent.mkdir(parents=True, exist_ok=True)
    (root / old).rename(target)


class _Counting:
    """Count the texts embedded and the files parsed, delegating the rest."""

    def __init__(self, inner: Any, seen: list[str]) -> None:
        self._inner = inner
        self._seen = seen

    def embed(self, texts: Sequence[str]) -> Any:
        self._seen.extend(texts)
        return self._inner.embed(texts)

    def parse(self, source: Source) -> Any:
        self._seen.append(source.path)
        return self._inner.parse(source)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _services(settings_: Settings) -> tuple[Services, list[str], list[str]]:
    """A container whose update counts what it parses and embeds."""
    services = create_container(settings_)
    update: Any = services.update
    parsed: list[str] = []
    embedded: list[str] = []
    update.embedder = _Counting(update.embedder, embedded)
    update.parsers = {
        file_type: _Counting(parser, parsed)
        for file_type, parser in update.parsers.items()
    }
    update.sleep = lambda seconds: None
    return services, parsed, embedded


def _library(tmp_path: Path, **overrides: Any) -> tuple[Path, Path, Settings]:
    books = _write(tmp_path / "books", BOOKS)
    notes = _write(tmp_path / "notes", NOTES)
    data = tmp_path / "data"
    save_roots(data, ("books", books, True), ("notes", notes, False))
    chosen = settings(data, **overrides)
    services = create_container(chosen)
    try:
        assert services.build.run().failed == 0
    finally:
        close(services)
    return books, notes, chosen


def _search(services: Services, query: str) -> list[tuple[str, str, float, float]]:
    found = services.search.run(query, top_n=10, collapse_duplicates=False)
    return [
        (h.source.path, h.passage.text, h.score, round(h.dense_score, 9))
        for h in found.results
    ]


def _committed(data: Path) -> dict[str, str]:
    manifest = SQLiteManifest(data, read_only=True)
    return {r.source.path: r.status for r in manifest.committed_sources()}


def _journal(data: Path, run_id: str) -> dict[str, tuple[str, str | None]]:
    manifest = SQLiteManifest(data, read_only=True)
    return {
        r.source.path: (r.status, r.reused_from) for r in manifest.run_sources(run_id)
    }


def test_a_moved_book_is_found_under_its_new_path_with_the_same_scores(
    tmp_path: Path,
) -> None:
    books, _, chosen = _library(tmp_path)
    services, parsed, embedded = _services(chosen)
    before = _search(services, "blue comet lighthouse ledger")
    _move(books, "a.txt", "archive/comet.txt")

    summary = services.update.run()
    after = _search(services, "blue comet lighthouse ledger")
    close(services)

    assert parsed == [] and embedded == []
    assert (summary.moved, summary.added, summary.removed, summary.failed) == (
        1,
        0,
        0,
        0,
    )
    assert after == [
        ("archive/comet.txt" if path == "a.txt" else path, text, score, dense)
        for path, text, score, dense in before
    ]
    committed = _committed(chosen.data_dir)
    assert "a.txt" not in committed
    assert committed["archive/comet.txt"] == "indexed"
    assert _journal(chosen.data_dir, summary.run_id) == {
        "a.txt": ("removed", None),
        "archive/comet.txt": ("indexed", "a.txt"),
    }
    assert not [p for p in dump_points(chosen) if p[1]["path"] == "a.txt"]


def test_a_renamed_file_in_a_prefixed_root_and_a_move_between_roots(
    tmp_path: Path,
) -> None:
    books, notes, chosen = _library(tmp_path)
    moss = [p[1] for p in dump_points(chosen) if p[1]["path"] == "shelf/b.md"]
    _move(notes, "c.txt", "kitchen/bread.txt")
    _move(books, "shelf/b.md", "moss.md", other=notes)
    services, parsed, embedded = _services(chosen)

    summary = services.update.run()
    found = services.search.run("green moss stones", top_n=3)
    close(services)

    assert parsed == [] and embedded == []
    assert (summary.moved, summary.added, summary.removed) == (2, 0, 0)
    committed = _committed(chosen.data_dir)
    assert {"notes/kitchen/bread.txt", "notes/moss.md"} <= set(committed)
    assert not {"notes/c.txt", "shelf/b.md"} & set(committed)
    assert found.results[0].source.source_id == "notes/moss.md"
    assert found.source_roots["notes/moss.md"] == "notes"
    journal = _journal(chosen.data_dir, summary.run_id)
    assert journal["notes/kitchen/bread.txt"] == ("indexed", "notes/c.txt")
    assert journal["notes/moss.md"] == ("indexed", "shelf/b.md")
    # Everything but the identity, the passage id and the file's time stays.
    identity = {"passage_id", "source_id", "path", "mtime"}
    moved = [p[1] for p in dump_points(chosen) if p[1]["path"] == "notes/moss.md"]
    assert sorted(
        ({k: v for k, v in p.items() if k not in identity} for p in moved),
        key=lambda p: p["chunk_offset"],
    ) == sorted(
        ({k: v for k, v in p.items() if k not in identity} for p in moss),
        key=lambda p: p["chunk_offset"],
    )
    assert moved[0]["file_type"] == FileType.MARKDOWN.value


def test_a_copy_is_one_result_with_the_other_location(tmp_path: Path) -> None:
    books, _, chosen = _library(tmp_path)
    _write(books, {"backup/a.txt": BOOKS["a.txt"]})
    services, parsed, embedded = _services(chosen)

    summary = services.update.run()
    found = services.search.run("blue comet lighthouse ledger", top_n=3)
    close(services)

    assert parsed == [] and embedded == []
    assert (summary.copied, summary.added) == (1, 0)
    top = found.results[0]
    locations = {top.source.path, *(d.source.path for d in top.duplicates)}
    assert locations == {"a.txt", "backup/a.txt"}
    points = dump_points(chosen)
    original = {p[1]["chunk_offset"]: p for p in points if p[1]["path"] == "a.txt"}
    copy = {p[1]["chunk_offset"]: p for p in points if p[1]["path"] == "backup/a.txt"}
    assert original and sorted(original) == sorted(copy)
    for offset, (point_id, payload, vectors) in copy.items():
        assert point_id != original[offset][0]
        assert payload["text"] == original[offset][1]["text"]
        assert vectors["sparse"] == original[offset][2]["sparse"]
        assert vectors["dense"] == pytest.approx(original[offset][2]["dense"])


def test_a_full_build_indexes_a_copy_inside_the_corpus_once(tmp_path: Path) -> None:
    books = _write(tmp_path / "books", {**BOOKS, "copy/a.txt": BOOKS["a.txt"]})
    data = tmp_path / "data"
    save_roots(data, ("books", books, True))
    services = create_container(settings(data))
    build: Any = services.build
    embedded: list[str] = []
    build.embedder = _Counting(build.embedder, embedded)

    summary = services.build.run()
    close(services)

    assert (summary.indexed, summary.copied) == (3, 1)
    points = dump_points(settings(data))
    per_path: dict[str, int] = {}
    for _, payload, _ in points:
        per_path[payload["path"]] = per_path.get(payload["path"], 0) + 1
    assert per_path["copy/a.txt"] == per_path["a.txt"]
    assert len(embedded) == per_path["a.txt"] + per_path["shelf/b.md"]
    assert _journal(data, summary.run_id)["copy/a.txt"] == ("indexed", "a.txt")
    record = SQLiteManifest(data, read_only=True).get_source("copy/a.txt")
    assert record is not None and record.reused_from is None


def _cli(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str]:
    code = main(list(args))
    out, err = capsys.readouterr()
    assert code == 0, err
    return code, out


def test_the_setting_turns_reuse_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    books, _, chosen = _library(tmp_path)
    for name, value in {
        **environment(chosen.data_dir),
        "DOCGREP_LOG_LEVEL": "WARNING",
    }.items():
        monkeypatch.setenv(name, value)
    _move(books, "a.txt", "archive/a.txt")

    _, out = _cli(capsys, "update", "--plan", "--json", "--paths")
    total = json.loads(out)["total"]
    assert (total["moved"], total["added"], total["removed"]) == (1, 0, 0)
    assert json.loads(out)["roots"][0]["paths"]["moved"] == [
        {"removed": "a.txt", "added": "archive/a.txt"}
    ]

    monkeypatch.setenv("DOCGREP_REUSE_BY_HASH", "false")
    _, out = _cli(capsys, "update", "--plan", "--json")
    total = json.loads(out)["total"]
    assert (total["moved"], total["added"], total["removed"]) == (0, 1, 1)

    services, parsed, embedded = _services(Settings())
    summary = services.update.run()
    close(services)

    assert (summary.moved, summary.added, summary.removed) == (0, 1, 1)
    assert parsed == ["archive/a.txt"] and embedded
    assert _journal(chosen.data_dir, summary.run_id)["archive/a.txt"] == (
        "indexed",
        None,
    )


def test_the_update_line_and_the_plan_text_name_moves_and_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    books, _, chosen = _library(tmp_path)
    for name, value in {
        **environment(chosen.data_dir),
        "DOCGREP_LOG_LEVEL": "WARNING",
    }.items():
        monkeypatch.setenv(name, value)
    _move(books, "a.txt", "archive/a.txt")
    _write(books, {"backup/b.md": BOOKS["shelf/b.md"]})

    _, plan = _cli(capsys, "update", "--plan", "--paths")
    _, out = _cli(capsys, "update")

    assert (
        "Root books: 0 added, 0 modified, 0 removed, 0 retried, 0 refreshed, " in plan
    )
    assert "1 moved, 1 copied.\n" in plan
    assert "  moved:\n    a.txt -> archive/a.txt\n" in plan
    assert "  copied:\n    shelf/b.md -> backup/b.md\n" in plan
    assert out.splitlines()[0] == (
        "Update plan: 0 added, 0 modified, 0 removed, 0 retried, 3 unchanged, "
        "1 moved, 1 copied."
    )
    assert (
        ", 0 retried, 1 moved, 1 copied, 3 unchanged, 0 skipped, 0 failed;"
        in (out.splitlines()[-1])
    )
