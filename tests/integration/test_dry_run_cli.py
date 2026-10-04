"""The dry-run command: report shape, scoping, sampling and read-only behaviour."""

import io
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import pytest

from docgrep.api.cli import main
from docgrep.api.container import create_dry_run
from docgrep.settings import Settings
from epub_support import epub_bytes, write_epub, xhtml

SECRET = "zebra-secret-phrase"
README = Path(__file__).resolve().parents[2] / "README.md"


def _corpus(root: Path) -> None:
    for name, content in {
        "a.txt": f"{SECRET} alpha beta gamma delta " * 20,
        "notes/b.md": f"# Title\n\n{SECRET} some markdown body text here " * 10,
        "notes/private/c.txt": f"{SECRET} private words",
        "books/old.docx": "docx bytes",
        "books/data.bin": "binary",
        "README": "plain",
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _env(monkeypatch: pytest.MonkeyPatch, corpus: Path, data: Path) -> None:
    monkeypatch.setenv("DOCGREP_CORPUS_ROOT", str(corpus))
    monkeypatch.setenv("DOCGREP_DATA_DIR", str(data))
    monkeypatch.setenv("DOCGREP_BACKEND", "fake")
    monkeypatch.setenv("DOCGREP_CHUNK_SIZE", "10")
    monkeypatch.setenv("DOCGREP_CHUNK_OVERLAP", "2")
    monkeypatch.setenv("DOCGREP_EXCLUDE_GLOBS", '["notes/private/"]')


def _run(*args: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(["dry-run", *args])
    return code, out.getvalue(), err.getvalue()


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
    }


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus"
    _corpus(corpus)
    _env(monkeypatch, corpus, tmp_path / "data")
    return corpus, tmp_path / "data"


def test_the_json_report_has_the_documented_keys(setup: tuple[Path, Path]) -> None:
    code, out, _ = _run("--json", "--sample", "2")

    assert code == 0
    report = json.loads(out)
    assert set(report) == {
        "schema_version",
        "scope",
        "totals",
        "types",
        "unsupported",
        "would_be_indexed",
        "recognised_without_parser",
        "excluded",
        "files",
        "sample",
    }
    assert report["schema_version"] == 1
    assert set(report["totals"]) == {"files", "size_bytes", "included", "excluded"}
    assert set(report["types"][0]) == {
        "type",
        "status",
        "files",
        "size_bytes",
        "average_size_bytes",
        "largest_size_bytes",
    }
    assert set(report["unsupported"]) == {
        "files",
        "size_bytes",
        "average_size_bytes",
        "largest_size_bytes",
        "extensions",
    }
    assert set(report["unsupported"]["extensions"][0]) == {"extension", "files"}
    assert set(report["excluded"]) == {"files", "size_bytes", "by_pattern"}
    assert set(report["files"]) == {"offset", "limit", "total", "items"}
    assert set(report["files"]["items"][0]) == {
        "path",
        "size_bytes",
        "extension",
        "type",
        "status",
        "excluded_by",
    }
    assert set(report["sample"]) == {"requested", "seed", "note", "types"}
    entry = report["sample"]["types"][0]
    assert set(entry) == {
        "type",
        "sampled_files",
        "parsed",
        "failed",
        "requires_ocr_fraction",
        "passages_per_file",
        "passages_per_mb",
        "parse_chunk_seconds_per_file",
        "estimate",
    }
    assert set(entry["estimate"]) == {
        "note",
        "corpus_files",
        "corpus_size_bytes",
        "by_file_count",
        "by_size",
    }
    assert set(entry["estimate"]["by_size"]) == {"passages", "parse_chunk_seconds"}


def _keys(value: object, found: set[str] | None = None) -> set[str]:
    found = set() if found is None else found
    if isinstance(value, dict):
        for key, inner in value.items():
            found.add(key)
            _keys(inner, found)
    elif isinstance(value, list):
        for inner in value:
            _keys(inner, found)
    return found


def test_every_json_key_is_documented_in_the_readme(setup: tuple[Path, Path]) -> None:
    _, out, _ = _run("--json", "--sample", "2")
    readme = README.read_text(encoding="utf-8")

    missing = {key for key in _keys(json.loads(out)) if f"`{key}`" not in readme}

    assert missing == set()


def test_the_report_matches_the_tree(setup: tuple[Path, Path]) -> None:
    _, out, _ = _run("--json")
    report = json.loads(out)

    assert report["totals"]["files"] == 6
    assert report["totals"]["excluded"] == 1
    assert report["excluded"]["by_pattern"] == [
        {"pattern": "notes/private/", "files": 1}
    ]
    types = {t["type"]: t for t in report["types"]}
    assert set(types) == {"txt", "markdown", "docx"}
    assert types["docx"]["status"] == "recognised_without_parser"
    assert report["unsupported"]["extensions"] == [
        {"extension": "(none)", "files": 1},
        {"extension": ".bin", "files": 1},
    ]
    assert report["would_be_indexed"]["files"] == 2


def test_the_text_report_is_readable_and_has_no_file_list_or_content(
    setup: tuple[Path, Path],
) -> None:
    code, out, err = _run("--sample", "2")

    assert code == 0 and err == ""
    assert "Corpus dry run (read-only" in out
    assert "recognised_without_parser" in out and "notes/private/: 1" in out
    assert "estimate" in out and "extrapolated" in out
    assert SECRET not in out
    assert "a.txt" not in out and "books/old.docx" not in out


def test_the_json_report_never_contains_file_content(setup: tuple[Path, Path]) -> None:
    _, out, _ = _run("--json", "--sample", "2")

    assert SECRET not in out and "alpha beta" not in out


def test_pagination_and_path_prefix_through_the_command(
    setup: tuple[Path, Path],
) -> None:
    _, out, _ = _run("--json", "--limit", "2", "--offset", "1")
    page = json.loads(out)["files"]
    _, everything, _ = _run("--json", "--limit", "500")

    assert [i["path"] for i in page["items"]] == [
        i["path"] for i in json.loads(everything)["files"]["items"]
    ][1:3]
    assert page["total"] == 6 and page["limit"] == 2 and page["offset"] == 1

    _, scoped, _ = _run("--json", "--path-prefix", "notes/")
    report = json.loads(scoped)
    assert report["scope"] == {"path_prefix": "notes/"}
    assert [i["path"] for i in report["files"]["items"]] == [
        "notes/b.md",
        "notes/private/c.txt",
    ]


@pytest.mark.parametrize(
    "args",
    [
        ("--limit", "0"),
        ("--limit", "501"),
        ("--offset", "-1"),
        ("--sample", "0"),
        ("--sample", "51"),
        ("--path-prefix", "../outside"),
        ("--path-prefix", "/abs"),
    ],
)
def test_invalid_values_are_structured_errors(
    setup: tuple[Path, Path], args: tuple[str, ...]
) -> None:
    code, out, err = _run(*args)

    assert code == 1 and out == ""
    assert err.startswith("validation_error: ")


def test_an_inaccessible_corpus_is_a_structured_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _env(monkeypatch, tmp_path / "missing", tmp_path / "data")

    code, _, err = _run()

    assert code == 1 and err.startswith("validation_error: corpus is not accessible")
    assert not (tmp_path / "data").exists()


def test_sampling_is_repeatable_through_the_command(setup: tuple[Path, Path]) -> None:
    def sample(*extra: str) -> list[Any]:
        _, out, _ = _run("--json", "--sample", "1", *extra)
        return [
            (t["type"], t["sampled_files"], t["passages_per_file"])
            for t in json.loads(out)["sample"]["types"]
        ]

    assert sample("--seed", "3") == sample("--seed", "3")
    assert sample() == sample("--seed", "0")


def test_a_dry_run_changes_nothing_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    _corpus(corpus)
    data = tmp_path / "data"
    (data / "qdrant").mkdir(parents=True)
    (data / "manifest.sqlite3").write_bytes(b"not a real database")
    (data / "qdrant" / "meta.json").write_text("{}", encoding="utf-8")
    _env(monkeypatch, corpus, data)
    before = (_snapshot(data), _snapshot(corpus))

    for args in (
        (),
        ("--json",),
        ("--sample", "5"),
        ("--json", "--sample", "50", "--seed", "9"),
    ):
        assert _run(*args)[0] == 0

    assert (_snapshot(data), _snapshot(corpus)) == before
    assert sorted(p.name for p in data.iterdir()) == ["manifest.sqlite3", "qdrant"]


def test_a_dry_run_creates_no_data_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    _corpus(corpus)
    data = tmp_path / "does-not-exist-yet"
    _env(monkeypatch, corpus, data)

    assert _run("--sample", "3")[0] == 0

    assert not data.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["corpus"]


def test_a_dry_run_takes_no_lock_and_works_while_a_build_holds_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fcntl

    corpus = tmp_path / "corpus"
    _corpus(corpus)
    data = tmp_path / "data"
    data.mkdir()
    _env(monkeypatch, corpus, data)
    with (data / "index.lock").open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert _run("--sample", "2")[0] == 0


def test_the_embedding_model_is_never_touched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    _corpus(corpus)
    monkeypatch.setenv("DOCGREP_CORPUS_ROOT", str(corpus))
    monkeypatch.setenv("DOCGREP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv(
        "DOCGREP_BACKEND", "real"
    )  # would need weights if it loaded a model
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    # Without sampling nothing is tokenized, so no model files are needed at all.
    assert _run("--json")[0] == 0
    assert not (tmp_path / "data").exists()


def test_the_command_wires_only_a_scanner_parsers_and_a_chunker(
    tmp_path: Path,
) -> None:
    dry_run = create_dry_run(
        Settings(corpus_root=tmp_path, data_dir=tmp_path / "d", backend="fake")
    )

    assert sorted(vars(dry_run)) == ["chunker", "parsers", "scanner"]
    assert not (tmp_path / "d").exists()
    assert os.listdir(tmp_path) == []


def test_epub_is_listed_as_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    write_epub(
        corpus / "books/novel.epub",
        epub_bytes([("one.xhtml", xhtml("<h1>Start</h1><p>Some book words.</p>"))]),
    )
    _env(monkeypatch, corpus, tmp_path / "data")

    code, out, _ = _run("--json", "--sample", "1")

    assert code == 0
    report = json.loads(out)
    types = {t["type"]: t for t in report["types"]}
    assert types["epub"]["status"] == "supported"
    assert report["would_be_indexed"]["files"] == 1
    assert report["recognised_without_parser"]["files"] == 0
    assert [f["status"] for f in report["files"]["items"]] == ["indexed"]
    (sampled,) = report["sample"]["types"]
    assert sampled["type"] == "epub"
    assert (sampled["parsed"], sampled["failed"]) == (1, 0)
