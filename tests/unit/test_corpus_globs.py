from pathlib import Path

import pytest

from docgrep.adapters.corpus import enumerate_corpus

FILES = (
    "root.md",
    "draft.md",
    "a.tmp",
    "notes/public.md",
    "notes/draft.md",
    "notes/private/a.txt",
    "notes/private/deep/b.txt",
    "notes/private/deep/er/c.md",
    "other/private/d.txt",
    "x/y/z/scratch.tmp",
    "x/y/keep.txt",
    "v1/file.txt",
    "v2/file.txt",
    "v10/file.txt",
)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    for name in FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    return tmp_path


def _paths(
    root: Path, include: tuple[str, ...] = ("**/*",), exclude: tuple[str, ...] = ()
) -> list[str]:
    return [s.path for s in enumerate_corpus(root, include, exclude)]


def test_trailing_double_star_excludes_everything_beneath_at_any_depth(
    corpus: Path,
) -> None:
    result = _paths(corpus, exclude=("notes/private/**",))

    assert not [path for path in result if path.startswith("notes/private/")]
    assert "notes/private/deep/b.txt" not in result
    assert "notes/private/deep/er/c.md" not in result
    assert "notes/public.md" in result
    assert "other/private/d.txt" in result


def test_leading_double_star_matches_any_depth_including_root(corpus: Path) -> None:
    result = _paths(corpus, exclude=("**/draft.md",))

    assert "draft.md" not in result
    assert "notes/draft.md" not in result
    assert "root.md" in result


def test_pattern_without_slash_matches_basename_at_depth_three(corpus: Path) -> None:
    result = _paths(corpus, exclude=("*.tmp",))

    assert "x/y/z/scratch.tmp" not in result
    assert "a.tmp" not in result
    assert "x/y/keep.txt" in result


def test_pattern_with_slash_is_anchored_at_the_root(corpus: Path) -> None:
    anchored = _paths(corpus, exclude=("private/**",))
    nested = _paths(corpus, exclude=("**/private/**",))

    assert "other/private/d.txt" in anchored
    assert "notes/private/a.txt" in anchored
    assert "other/private/d.txt" not in nested
    assert "notes/private/deep/b.txt" not in nested


def test_wildcards_do_not_cross_directory_separators(corpus: Path) -> None:
    result = _paths(corpus, exclude=("notes/*",))

    assert "notes/public.md" not in result
    assert "notes/private/deep/b.txt" in result


def test_character_class_and_question_mark_match_single_segment_characters(
    corpus: Path,
) -> None:
    result = _paths(corpus, exclude=("v[12]/*", "v?/nothing"))

    assert "v1/file.txt" not in result
    assert "v2/file.txt" not in result
    assert "v10/file.txt" in result


def test_include_globs_select_files_at_depth(corpus: Path) -> None:
    assert _paths(corpus, include=("notes/private/**",)) == [
        "notes/private/a.txt",
        "notes/private/deep/b.txt",
        "notes/private/deep/er/c.md",
    ]
    assert _paths(corpus, include=("**/*.md",)) == [
        "draft.md",
        "notes/draft.md",
        "notes/private/deep/er/c.md",
        "notes/public.md",
        "root.md",
    ]
    assert _paths(corpus, include=("*.tmp",)) == ["a.tmp", "x/y/z/scratch.tmp"]


def test_exclude_takes_precedence_over_include_at_depth(corpus: Path) -> None:
    result = _paths(corpus, include=("notes/**",), exclude=("**/deep/**",))

    assert result == [
        "notes/draft.md",
        "notes/private/a.txt",
        "notes/public.md",
    ]
