"""EPUB books through the composition root: build, search and manifest."""

from pathlib import Path

from fastapi.testclient import TestClient

from docgrep.adapters.manifest import SQLiteManifest
from docgrep.api.http import create_app
from docgrep.settings import Settings
from epub_support import AES, encryption_xml, epub_bytes, write_epub, xhtml


def test_epub_books_are_indexed_searched_and_rejected(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    write_epub(
        corpus / "novel.epub",
        epub_bytes(
            [
                ("one.xhtml", xhtml("<h1>Voyage</h1><p>The harbour was calm.</p>")),
                (
                    "two.xhtml",
                    xhtml(
                        "<h2>Storm</h2>"
                        "<p>The violet kraken lantern rose from the deep sea.</p>"
                    ),
                ),
            ],
            title="Sea Story",
            creators=("Jane Writer",),
            date="1901",
        ),
    )
    write_epub(
        corpus / "locked.epub",
        epub_bytes(
            [("a.xhtml", xhtml("<p>secret</p>"))], encryption=encryption_xml(AES)
        ),
    )
    write_epub(corpus / "blank.epub", epub_bytes([("a.xhtml", xhtml("<p> </p>"))]))
    settings = Settings(
        corpus_root=corpus,
        data_dir=tmp_path / "data",
        backend="fake",
        chunk_size=64,
        chunk_overlap=8,
    )

    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        built = client.post("/index/build")
        assert built.status_code == 200, built.text
        assert {k: built.json()[k] for k in ("indexed", "skipped", "failed")} == {
            "indexed": 1,
            "skipped": 1,
            "failed": 1,
        }
        response = client.post(
            "/search", json={"query": "violet kraken lantern", "top_n": 1}
        )
        assert response.status_code == 200, response.text
        top = response.json()["results"][0]

    assert top["source"]["path"] == "novel.epub"
    assert top["source"]["file_type"] == "epub"
    assert top["source"]["title"] == "Sea Story"
    assert top["source"]["author"] == "Jane Writer"
    assert top["source"]["year"] == 1901
    assert top["passage"]["text"] == (
        "The violet kraken lantern rose from the deep sea."
    )
    assert top["passage"]["heading_path"] == ["Voyage", "Storm"]
    assert top["passage"]["page_number"] is None
    manifest = SQLiteManifest(settings.data_dir)
    locked = manifest.get_source("locked.epub")
    blank = manifest.get_source("blank.epub")
    assert locked is not None and locked.status == "failed:parse_error"
    assert blank is not None and blank.status == "skipped:requires_ocr"
