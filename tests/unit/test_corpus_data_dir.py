from pathlib import Path

import pytest

from docgrep.adapters.corpus import enumerate_corpus
from docgrep.domain.errors import ValidationError


def _write(root: Path, *names: str) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")


def _paths(root: Path, data_dir: Path | None) -> list[str]:
    return [source.path for source in enumerate_corpus(root, data_dir=data_dir)]


@pytest.mark.parametrize("name", [".docgrep", "idx-store", "nested/deeper/store"])
def test_data_dir_inside_corpus_is_not_enumerated(tmp_path: Path, name: str) -> None:
    _write(tmp_path, "a.txt", "sub/b.md")
    _write(tmp_path / name, "manifest.sqlite3", "qdrant/meta.json", "notes.txt")

    assert _paths(tmp_path, tmp_path / name) == ["a.txt", "sub/b.md"]


def test_without_a_data_dir_every_file_is_enumerated(tmp_path: Path) -> None:
    _write(tmp_path, "a.txt", ".docgrep/manifest.sqlite3")

    assert _paths(tmp_path, None) == [".docgrep/manifest.sqlite3", "a.txt"]


def test_absolute_data_dir_is_matched_by_resolved_path(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    _write(corpus, "a.txt", "store/data.txt")
    link = tmp_path / "link"
    link.symlink_to(corpus, target_is_directory=True)

    assert _paths(link, corpus / "store") == ["a.txt"]
    assert _paths(corpus, link / "store") == ["a.txt"]


def test_relative_data_dir_follows_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "a.txt", ".docgrep/manifest.sqlite3")
    monkeypatch.chdir(tmp_path)

    assert _paths(Path("."), Path(".docgrep")) == ["a.txt"]


def test_data_dir_outside_the_corpus_changes_nothing(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    _write(corpus, "a.txt", "store/data.txt")

    assert _paths(corpus, tmp_path / "elsewhere") == ["a.txt", "store/data.txt"]
    assert _paths(corpus, tmp_path / "corpus-store") == ["a.txt", "store/data.txt"]


def test_git_directories_are_skipped_but_other_dot_directories_are_not(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "a.txt", ".git/config", "sub/.git/HEAD", ".hidden/h.txt")

    assert _paths(tmp_path, None) == [".hidden/h.txt", "a.txt"]


@pytest.mark.parametrize("corpus_name", ["", "corpus", "corpus/deeper"])
def test_corpus_root_inside_data_dir_is_a_configuration_error(
    tmp_path: Path, corpus_name: str
) -> None:
    data_dir = tmp_path / "data"
    root = data_dir / corpus_name if corpus_name else data_dir
    _write(root, "a.txt")

    with pytest.raises(ValidationError) as raised:
        enumerate_corpus(root, data_dir=data_dir)

    assert raised.value.category == "validation_error"
    assert raised.value.details == {
        "corpus_root": str(root),
        "data_dir": str(data_dir),
    }
