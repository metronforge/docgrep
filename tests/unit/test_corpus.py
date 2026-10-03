import os
from pathlib import Path

import pytest

from docgrep.adapters.corpus import enumerate_corpus
from docgrep.domain.errors import ValidationError
from docgrep.domain.models import FileType


def test_enumerates_classifies_and_hashes_in_relative_path_order(
    tmp_path: Path,
) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "z.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "nested" / "a.md").write_bytes(b"")
    (tmp_path / "nested" / "b.markdown").write_text("# Title", encoding="utf-8")
    (tmp_path / "nested" / "c.pdf").write_bytes(b"pdf")
    (tmp_path / "nested" / "d.docx").write_bytes(b"docx")
    (tmp_path / "nested" / "e.epub").write_bytes(b"epub")
    (tmp_path / "nested" / "f.djvu").write_bytes(b"djvu")
    (tmp_path / "nested" / "g.bin").write_bytes(b"other")

    first = enumerate_corpus(tmp_path)
    second = enumerate_corpus(tmp_path)

    assert first == second
    assert [source.path for source in first] == [
        "nested/a.md",
        "nested/b.markdown",
        "nested/c.pdf",
        "nested/d.docx",
        "nested/e.epub",
        "nested/f.djvu",
        "nested/g.bin",
        "z.txt",
    ]
    assert [source.file_type for source in first] == [
        FileType.MARKDOWN,
        FileType.MARKDOWN,
        FileType.PDF,
        FileType.DOCX,
        FileType.EPUB,
        FileType.DJVU,
        FileType.UNSUPPORTED,
        FileType.TXT,
    ]
    assert (
        first[0].content_hash
        == "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
    )
    assert first[0].size == 0
    assert first[0].mtime == (tmp_path / "nested" / "a.md").stat().st_mtime
    assert [source.source_id for source in first] == [source.path for source in first]


def test_include_and_exclude_globs_apply_to_relative_paths(tmp_path: Path) -> None:
    for directory in ("notes", "notes/private", "papers"):
        (tmp_path / directory).mkdir(parents=True)
    for name in (
        "root.md",
        "notes/public.md",
        "notes/private/secret.md",
        "papers/research.pdf",
    ):
        (tmp_path / name).write_text(name, encoding="utf-8")

    sources = enumerate_corpus(
        tmp_path,
        include_globs=("**/*.md",),
        exclude_globs=("notes/private/**",),
    )

    assert [source.path for source in sources] == ["notes/public.md", "root.md"]


def test_symlink_loop_and_file_symlink_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "real.txt").write_text("text", encoding="utf-8")
    (tmp_path / "sub" / "loop").symlink_to(tmp_path, target_is_directory=True)
    (tmp_path / "linked.txt").symlink_to(tmp_path / "sub" / "real.txt")

    assert [source.path for source in enumerate_corpus(tmp_path)] == ["sub/real.txt"]


@pytest.mark.parametrize("root_kind", ["missing", "file", "unreadable"])
def test_invalid_corpus_root_raises_structured_error(
    tmp_path: Path, root_kind: str
) -> None:
    if root_kind == "unreadable" and os.geteuid() == 0:
        pytest.skip("permission bits do not restrict the superuser")
    root = tmp_path / "corpus"
    if root_kind == "file":
        root.write_text("not a directory", encoding="utf-8")
    elif root_kind == "unreadable":
        root.mkdir()
        root.chmod(0o000)

    try:
        with pytest.raises(ValidationError) as raised:
            enumerate_corpus(root)
    finally:
        if root_kind == "unreadable":
            root.chmod(0o755)

    assert raised.value.category == "validation_error"
    assert raised.value.details["path"] == str(root)
