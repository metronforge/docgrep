"""The stat-only corpus scan and its agreement with the build enumeration."""

import os
from pathlib import Path

import pytest

from docgrep.adapters.corpus import (
    NOT_MATCHED_BY_INCLUDES,
    enumerate_corpus,
    scan_corpus,
)
from docgrep.domain.errors import ValidationError
from docgrep.domain.models import FileType

FILES = {
    "a.txt": "alpha",
    "b.md": "# beta",
    "doc.pdf": "%PDF",
    "book.epub": "epub",
    "data.bin": "binary",
    "noext": "x",
    "UPPER.TXT": "upper",
    "notes/public.md": "public",
    "notes/private/secret.txt": "secret",
    "notes/private/deep/more.txt": "more",
    "drafts/one.md": "draft",
    "x/y/z/scratch.tmp": "tmp",
    ".hidden/h.txt": "hidden",
    ".git/config": "[core]",
    ".docgrep/manifest.sqlite3": "db",
}


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    for name, text in FILES.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


def test_scan_reports_size_extension_type_and_no_exclusion(corpus: Path) -> None:
    scanned = {
        item.path: item for item in scan_corpus(corpus, data_dir=corpus / ".docgrep")
    }

    assert ".git/config" not in scanned and ".docgrep/manifest.sqlite3" not in scanned
    assert scanned["a.txt"].size == 5 and scanned["a.txt"].extension == ".txt"
    assert scanned["a.txt"].file_type is FileType.TXT
    assert scanned["UPPER.TXT"].extension == ".txt"
    assert scanned["UPPER.TXT"].file_type is FileType.TXT
    assert scanned["book.epub"].file_type is FileType.EPUB
    assert (
        scanned["data.bin"].file_type is None
        and scanned["data.bin"].extension == ".bin"
    )
    assert scanned["noext"].extension == "" and scanned["noext"].file_type is None
    assert all(item.excluded_by is None for item in scanned.values())
    assert list(scanned) == sorted(scanned)


def test_scan_attributes_each_exclusion_to_its_pattern(corpus: Path) -> None:
    scanned = {
        item.path: item
        for item in scan_corpus(
            corpus,
            include_globs=("**/*.txt", "**/*.md", "**/*.pdf"),
            exclude_globs=("notes/private/", "*.tmp", "drafts/**"),
            data_dir=corpus / ".docgrep",
        )
    }

    assert scanned["notes/private/secret.txt"].excluded_by == "notes/private/"
    assert scanned["notes/private/deep/more.txt"].excluded_by == "notes/private/"
    assert scanned["drafts/one.md"].excluded_by == "drafts/**"
    assert scanned["x/y/z/scratch.tmp"].excluded_by == NOT_MATCHED_BY_INCLUDES
    assert scanned["data.bin"].excluded_by == NOT_MATCHED_BY_INCLUDES
    assert scanned["book.epub"].excluded_by == NOT_MATCHED_BY_INCLUDES
    assert scanned["a.txt"].excluded_by is None
    assert scanned["notes/public.md"].excluded_by is None


def test_the_first_matching_exclude_pattern_is_reported(corpus: Path) -> None:
    scanned = {
        item.path: item
        for item in scan_corpus(corpus, exclude_globs=("notes/**", "**/private/**"))
    }

    assert scanned["notes/private/secret.txt"].excluded_by == "notes/**"


@pytest.mark.parametrize(
    ("include", "exclude"),
    [
        (("**/*",), ()),
        (("**/*.txt", "**/*.md"), ()),
        (("**/*",), ("notes/private/",)),
        (("**/*.txt", "**/*.md", "**/*.pdf"), ("notes/private/**", "*.tmp", "/a.txt")),
        (("notes/**",), ("**/deep/**",)),
        (("nothing/**",), ()),
    ],
)
def test_the_scan_predicts_exactly_what_a_build_enumerates(
    corpus: Path, include: tuple[str, ...], exclude: tuple[str, ...]
) -> None:
    data_dir = corpus / ".docgrep"

    enumerated = enumerate_corpus(corpus, include, exclude, data_dir)
    scanned = scan_corpus(corpus, include, exclude, data_dir)

    kept = [item for item in scanned if item.excluded_by is None]
    assert [item.path for item in kept] == [source.path for source in enumerated]
    assert [item.size for item in kept] == [source.size for source in enumerated]
    assert [item.file_type or FileType.UNSUPPORTED for item in kept] == [
        source.file_type for source in enumerated
    ]
    skipped_by_both = {item.path for item in scanned} - {s.path for s in enumerated}
    assert skipped_by_both == {i.path for i in scanned if i.excluded_by is not None}


@pytest.mark.skipif(os.geteuid() == 0, reason="the superuser can read any file")
def test_the_scan_never_reads_file_content(corpus: Path) -> None:
    secret = corpus / "a.txt"
    secret.chmod(0o000)
    try:
        scanned = {item.path: item for item in scan_corpus(corpus)}
        with pytest.raises(ValidationError):
            enumerate_corpus(corpus)  # hashing needs to read the file
    finally:
        secret.chmod(0o644)

    assert scanned["a.txt"].size == 5


def test_the_scan_rejects_the_same_misconfigurations_as_the_enumeration(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError) as missing:
        scan_corpus(tmp_path / "missing")
    assert missing.value.details["path"] == str(tmp_path / "missing")

    data_dir = tmp_path / "data"
    (data_dir / "corpus").mkdir(parents=True)
    with pytest.raises(ValidationError, match="inside the data directory"):
        scan_corpus(data_dir / "corpus", data_dir=data_dir)


def test_symlinks_are_not_followed_by_the_scan(corpus: Path) -> None:
    (corpus / "link.txt").symlink_to(corpus / "a.txt")
    (corpus / "loop").symlink_to(corpus, target_is_directory=True)

    paths = [item.path for item in scan_corpus(corpus)]

    assert "link.txt" not in paths and not any(p.startswith("loop/") for p in paths)
