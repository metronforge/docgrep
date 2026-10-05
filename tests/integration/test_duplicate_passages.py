"""Duplicate collapsing over real SQLite, local-mode Qdrant and the transports."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from docgrep.api.cli import main
from docgrep.api.http import create_app
from docgrep.settings import Settings
from pdf_support import pdf_bytes, prose, write_pdf

_MTIME = 1_700_000_000
# Same text as a.txt with another line wrapping; the lowest passage id of the
# three copies of the sentence, so it represents them.
_FILES = {
    "a.txt": "A rare blue comet crosses the sky.",
    "b.txt": "A rare blue comet\ncrosses  the sky.\n",
    "c.txt": "A rare blue comet crosses the sea.",
    "d.txt": "a rare blue comet crosses the sky.",
}
_QUERY = {"query": "blue comet sky", "top_n": 3}

_ID_A = "4ce5b267e4de9962352ac53f6bb5d896032fb281269d3631eb11031629d601f1"
_ID_B = "12d658945ec07eb2ade8f21088751362c67fb4cb83a1c25545c7f59356f3142a"
_ID_D = "ce34de35f4c0520d76005901369d4e19ccf9dd8661b9bad68fe78dfa5ad61294"
_HASH_A = "ff83404252c47b35db518e7751e3623c6e4e270e35ad9a7566a2ebf27119808a"
_HASH_B = "c2e74ffdd4127999b4d1e823b65d88017f20710d4f3db8cd69c41383eace8d42"
_HASH_D = "e48818cfce7e220f616dc6efa4cb869a40d45c71d851c9d9c0de3fe11833e865"


def _result(
    passage_id: str, path: str, text: str, size: int, content_hash: str
) -> dict[str, Any]:
    return {
        "dense_score": 0.745355992500,
        "passage": {
            "chunk_offset": 0,
            "heading_path": [],
            "page_number": None,
            "passage_id": passage_id,
            "source_id": path,
            "text": text,
            "token_count": 7,
        },
        "reranker_score": 1.0,
        "score": 0.7310585786300049,
        "source": {
            "author": None,
            "content_hash": content_hash,
            "file_type": "txt",
            "indexed_at": None,
            "mtime": float(_MTIME),
            "parser_version": "1",
            "path": path,
            "size": size,
            "source_id": path,
            "title": None,
            "year": None,
        },
    }


# The response of release 1.10.0 to _QUERY over _FILES, recorded with that
# release; took_ms is removed because it varies per run.
_PREVIOUS_RELEASE: dict[str, Any] = {
    "applied_config": {
        "embedding_model": "deterministic-hash-v1",
        "max_query_len": 1000,
        "reranker_model": "lexical-overlap-v1",
        "top_k": 100,
        "top_n": 3,
    },
    "hint": None,
    "rerank_skip_reason": None,
    "rerank_skipped": False,
    "results": [
        _result(_ID_B, "b.txt", "A rare blue comet\ncrosses  the sky.", 36, _HASH_B),
        _result(_ID_A, "a.txt", "A rare blue comet crosses the sky.", 34, _HASH_A),
        _result(_ID_D, "d.txt", "a rare blue comet crosses the sky.", 34, _HASH_D),
    ],
    "total_hits": 4,
}


def _corpus(root: Path, files: dict[str, str] = _FILES) -> Path:
    corpus = root / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (corpus / name).write_text(text, encoding="utf-8")
        os.utime(corpus / name, (_MTIME, _MTIME))
    return corpus


def _settings(root: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "corpus_root": root / "corpus",
        "data_dir": root / "data",
        "backend": "fake",
        "chunk_size": 10,
        "chunk_overlap": 2,
    }
    return Settings(**(values | overrides))


def _normalized(response: dict[str, Any]) -> dict[str, Any]:
    """Drop the timing and round the dense score, whose last bits vary in local
    mode between reads of the same index."""
    response = dict(response)
    assert response.pop("took_ms") >= 0
    for item in response["results"]:
        item["dense_score"] = round(item["dense_score"], 12)
    return response


def _search(settings: Settings, *bodies: dict[str, Any]) -> list[dict[str, Any]]:
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        if not settings.data_dir.exists():
            assert client.post("/index/build").status_code == 200
        responses = [client.post("/search", json=body) for body in bodies]
    assert all(response.status_code == 200 for response in responses)
    return [_normalized(response.json()) for response in responses]


def test_collapsing_off_returns_the_response_of_the_previous_release(
    tmp_path: Path,
) -> None:
    _corpus(tmp_path)
    (by_request,) = _search(
        _settings(tmp_path), {**_QUERY, "collapse_duplicates": False}
    )
    (by_setting,) = _search(_settings(tmp_path, collapse_duplicates=False), _QUERY)

    assert by_request == _PREVIOUS_RELEASE
    assert by_setting == _PREVIOUS_RELEASE
    # The new keys are absent, not merely empty.
    assert "collapsed_duplicates" not in by_request
    assert all("duplicates" not in item for item in by_request["results"])


def test_http_collapses_identical_text_by_default(tmp_path: Path) -> None:
    _corpus(tmp_path)
    default, explicit = _search(
        _settings(tmp_path), _QUERY, {**_QUERY, "collapse_duplicates": None}
    )

    assert default == explicit
    assert [item["source"]["path"] for item in default["results"]] == [
        "b.txt",
        "d.txt",
        "c.txt",
    ]
    representative = default["results"][0]
    # The representative's own stored text, with its line break and spaces.
    assert representative["passage"]["text"] == _FILES["b.txt"].rstrip("\n")
    assert representative["duplicates"] == [
        {
            "passage_id": _ID_A,
            "source_id": "a.txt",
            "path": "a.txt",
            "file_type": "txt",
            "page_number": None,
            "heading_path": [],
        }
    ]
    assert {key: representative[key] for key in ("score", "dense_score")} == {
        key: _PREVIOUS_RELEASE["results"][0][key] for key in ("score", "dense_score")
    }
    assert all("duplicates" not in item for item in default["results"][1:])
    assert default["total_hits"] == 4
    assert default["collapsed_duplicates"] == 1
    assert set(default) == set(_PREVIOUS_RELEASE) | {"collapsed_duplicates"}


def test_the_request_field_overrides_the_setting_both_ways(tmp_path: Path) -> None:
    _corpus(tmp_path)
    (on,) = _search(
        _settings(tmp_path, collapse_duplicates=False),
        {**_QUERY, "collapse_duplicates": True},
    )
    (off,) = _search(_settings(tmp_path), {**_QUERY, "collapse_duplicates": False})

    assert on["collapsed_duplicates"] == 1
    assert len(on["results"][0]["duplicates"]) == 1
    assert off == _PREVIOUS_RELEASE


def test_collapse_setting_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert Settings().collapse_duplicates is True
    monkeypatch.setenv("DOCGREP_COLLAPSE_DUPLICATES", "false")
    assert Settings().collapse_duplicates is False


def test_identical_pdf_pages_report_the_page_of_each_copy(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    shared = prose(2)
    write_pdf(corpus / "first.pdf", pdf_bytes([prose(1), shared]))
    write_pdf(corpus / "second.pdf", pdf_bytes([prose(5), prose(6), shared]))
    settings = _settings(tmp_path, chunk_size=200, chunk_overlap=0)

    on, off = _search(
        settings,
        {"query": " ".join(shared), "top_n": 2},
        {"query": " ".join(shared), "top_n": 2, "collapse_duplicates": False},
    )

    copies = {
        item["source"]["path"]: item["passage"]["page_number"]
        for item in off["results"]
    }
    assert copies == {"first.pdf": 2, "second.pdf": 3}
    representative = on["results"][0]
    duplicate = representative["duplicates"][0]
    assert {
        representative["source"]["path"]: representative["passage"]["page_number"],
        duplicate["path"]: duplicate["page_number"],
    } == copies
    assert duplicate["file_type"] == "pdf"
    assert representative["passage"]["text"] == off["results"][0]["passage"]["text"]
    assert on["collapsed_duplicates"] == 1


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix != ".lock" and path.name != ".lock"
    }


def _cli(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    root: Path,
    *args: str,
) -> str:
    monkeypatch.setenv("DOCGREP_CORPUS_ROOT", str(root / "corpus"))
    monkeypatch.setenv("DOCGREP_DATA_DIR", str(root / "data"))
    monkeypatch.setenv("DOCGREP_BACKEND", "fake")
    monkeypatch.setenv("DOCGREP_CHUNK_SIZE", "10")
    monkeypatch.setenv("DOCGREP_CHUNK_OVERLAP", "2")
    capsys.readouterr()
    assert main(list(args)) == 0
    return capsys.readouterr().out


def test_cli_text_output_lists_the_other_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _corpus(tmp_path)
    _cli(monkeypatch, capsys, tmp_path, "index")
    query = ("search", "blue comet sky", "--top-n", "3")

    on = _cli(monkeypatch, capsys, tmp_path, *query)
    off = _cli(monkeypatch, capsys, tmp_path, *query, "--no-collapse-duplicates")

    assert on == (
        "b.txt (score 0.731)\nA rare blue comet\ncrosses  the sky.\n"
        "also in: a.txt\n\n"
        "d.txt (score 0.731)\na rare blue comet crosses the sky.\n\n"
        "c.txt (score 0.661)\nA rare blue comet crosses the sea.\n\n"
    )
    assert off == (
        "b.txt (score 0.731)\nA rare blue comet\ncrosses  the sky.\n\n"
        "a.txt (score 0.731)\nA rare blue comet crosses the sky.\n\n"
        "d.txt (score 0.731)\na rare blue comet crosses the sky.\n\n"
    )
    # Byte-identical on a repeated run, in both modes.
    assert _cli(monkeypatch, capsys, tmp_path, *query) == on
    assert (
        _cli(monkeypatch, capsys, tmp_path, *query, "--no-collapse-duplicates") == off
    )


def test_cli_text_output_names_the_page_or_heading_of_a_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    shared = prose(2)
    corpus = tmp_path / "corpus"
    write_pdf(corpus / "book.pdf", pdf_bytes([prose(1), shared]))
    write_pdf(corpus / "copy.pdf", pdf_bytes([shared]))
    _corpus(
        tmp_path,
        {
            "notes.md": "# Survey\n\n## Rivers\n\n" + "\n".join(shared) + "\n",
        },
    )
    monkeypatch.setenv("DOCGREP_CHUNK_SIZE", "200")
    monkeypatch.setenv("DOCGREP_CHUNK_OVERLAP", "0")
    monkeypatch.setenv("DOCGREP_CORPUS_ROOT", str(corpus))
    monkeypatch.setenv("DOCGREP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DOCGREP_BACKEND", "fake")
    assert main(["index"]) == 0
    capsys.readouterr()
    assert main(["search", " ".join(shared), "--top-n", "1"]) == 0
    out = capsys.readouterr().out

    also = [line for line in out.splitlines() if line.startswith("also in: ")]
    assert len(also) == 2
    expected = {
        "also in: book.pdf (page 2)",
        "also in: copy.pdf (page 1)",
        "also in: notes.md (Survey > Rivers)",
    }
    assert set(also) < expected
    assert out.count("(score ") == 1


def test_cli_json_carries_the_new_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _corpus(tmp_path)
    _cli(monkeypatch, capsys, tmp_path, "index")
    query = ("search", "blue comet sky", "--top-n", "3", "--json")

    on = _normalized(json.loads(_cli(monkeypatch, capsys, tmp_path, *query)))
    off = _normalized(
        json.loads(
            _cli(monkeypatch, capsys, tmp_path, *query, "--no-collapse-duplicates")
        )
    )

    assert off == _PREVIOUS_RELEASE
    assert on["collapsed_duplicates"] == 1
    assert on["results"][0]["duplicates"][0]["path"] == "a.txt"
    assert on["total_hits"] == 4


def test_searching_writes_nothing_to_the_index_or_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _corpus(tmp_path)
    _cli(monkeypatch, capsys, tmp_path, "index")
    before = _tree_digest(tmp_path / "data")

    _cli(monkeypatch, capsys, tmp_path, "search", "blue comet sky")
    _cli(monkeypatch, capsys, tmp_path, "search", "blue comet sky", "--json")
    _cli(
        monkeypatch,
        capsys,
        tmp_path,
        "search",
        "blue comet sky",
        "--no-collapse-duplicates",
    )

    assert _tree_digest(tmp_path / "data") == before


def test_the_representative_does_not_depend_on_which_copy_came_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    outputs = []
    for first, second in (("a.txt", "b.txt"), ("b.txt", "a.txt")):
        root = tmp_path / first
        _corpus(root, {first: _FILES[first]})
        _cli(monkeypatch, capsys, root, "index")
        _corpus(root, {second: _FILES[second]})
        _cli(monkeypatch, capsys, root, "update")
        outputs.append(_cli(monkeypatch, capsys, root, "search", "blue comet sky"))

    assert (
        outputs[0]
        == outputs[1]
        == (
            "b.txt (score 0.731)\nA rare blue comet\ncrosses  the sky.\n"
            "also in: a.txt\n\n"
        )
    )
