"""Builds with the index stored inside the corpus directory."""

from pathlib import Path

import pytest

from docgrep.adapters.manifest import SQLiteManifest
from docgrep.api.container import create_container
from docgrep.settings import Settings


def _settings(corpus: Path, data_dir: Path) -> Settings:
    return Settings(
        corpus_root=corpus,
        data_dir=data_dir,
        backend="fake",
        chunk_size=10,
        chunk_overlap=2,
    )


def _populate(corpus: Path) -> None:
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "a.txt").write_text("A rare blue comet crosses the sky.", "utf-8")
    (corpus / "b.md").write_text("# Title\n\nGarden notes about moss.", "utf-8")
    (corpus / ".git").mkdir()
    (corpus / ".git" / "config").write_text("[core]", "utf-8")


@pytest.mark.parametrize("name", [".docgrep", "custom-index"])
def test_consecutive_builds_ignore_the_index_inside_the_corpus(
    tmp_path: Path, name: str
) -> None:
    corpus = tmp_path / "corpus"
    _populate(corpus)
    data_dir = corpus / name
    services = create_container(_settings(corpus, data_dir))

    summaries = [services.build.run() for _ in range(3)]

    for summary in summaries:
        assert (summary.indexed, summary.skipped, summary.failed) == (2, 0, 0)
    manifest = SQLiteManifest(data_dir)
    assert manifest.get_source("a.txt") is not None
    assert manifest.get_source("b.md") is not None
    assert manifest.get_source(f"{name}/manifest.sqlite3") is None
    assert manifest.get_source(".git/config") is None
    assert (data_dir / "manifest.sqlite3").exists()


def test_default_relative_layout_does_not_index_its_own_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _populate(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DOCGREP_BACKEND", "fake")
    monkeypatch.delenv("DOCGREP_CORPUS_ROOT", raising=False)
    monkeypatch.delenv("DOCGREP_DATA_DIR", raising=False)
    services = create_container(Settings())

    first = services.build.run()
    second = services.build.run()

    assert (first.indexed, first.skipped) == (2, 0)
    assert (second.indexed, second.skipped) == (2, 0)
