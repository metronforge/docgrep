"""Two corpus roots that hold the same relative path, for root tests."""

import json
from pathlib import Path
from typing import Any

from docgrep.settings import Settings

BOOKS_TEXT = {
    "a.txt": "A rare blue comet crosses the winter sky above the harbour.",
    "shelf/b.md": "# Moss\n\nGarden notes about green moss on old stones.",
}
NOTES_TEXT = {
    "a.txt": "Field notes: a yellow kite drifts over the quiet meadow.",
    "c.txt": "Recipe for bread with rye flour and caraway seeds.",
}


def write_tree(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def two_roots(tmp_path: Path) -> tuple[Path, Path]:
    """``books`` and ``notes``, both with an ``a.txt``."""
    return (
        write_tree(tmp_path / "books", BOOKS_TEXT),
        write_tree(tmp_path / "notes", NOTES_TEXT),
    )


def save_roots(data_dir: Path, *roots: tuple[str, Path, bool]) -> None:
    """Write a roots file of (name, path, legacy) entries."""
    data_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, path, legacy in roots:
        entry: dict[str, object] = {"name": name, "path": str(path.resolve())}
        if legacy:
            entry["legacy"] = True
        entries.append(entry)
    (data_dir / "roots.json").write_text(
        json.dumps({"version": 1, "roots": entries}), encoding="utf-8"
    )


def settings(data_dir: Path, **overrides: Any) -> Settings:
    """Settings without the deprecated corpus root, as with a roots file."""
    return Settings(
        data_dir=data_dir, backend="fake", chunk_size=10, chunk_overlap=2, **overrides
    )


def environment(data_dir: Path) -> dict[str, str]:
    """The environment of a CLI run that uses the roots file."""
    return {
        "DOCGREP_DATA_DIR": str(data_dir),
        "DOCGREP_BACKEND": "fake",
        "DOCGREP_CHUNK_SIZE": "10",
        "DOCGREP_CHUNK_OVERLAP": "2",
    }
