"""End-to-end indexing and retrieval over a generated mixed corpus."""

from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from docgrep.adapters.manifest import SQLiteManifest
from docgrep.api.http import create_app
from docgrep.settings import Settings


def _write_corpus(root: Path) -> None:
    root.mkdir()
    (root / "background.txt").write_text(
        "The archive catalog describes a quiet reading room.", encoding="utf-8"
    )
    (root / "field-notes.md").write_text(
        "# Field notes\n\n## Garden\n\n"
        "The orbit lantern garden record describes moss and rain.\n",
        encoding="utf-8",
    )
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "An unrelated first page.")
        pdf.new_page().insert_text(
            (72, 72), "The orbit lantern zenith record describes stars and dust."
        )
        pdf.save(root / "observations.pdf")

    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 2, 2), 0)
    pixmap.clear_with(255)
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_image(page.rect, pixmap=pixmap)
        pdf.save(root / "scan.pdf")
    (root / "unsupported.bin").write_bytes(b"not a supported document")


@pytest.mark.parametrize(
    "backend",
    ["fake", pytest.param("real", marks=pytest.mark.gpu)],
)
def test_mixed_corpus_http_build_search_and_rebuild(
    tmp_path: Path, backend: str
) -> None:
    corpus = tmp_path / "corpus"
    _write_corpus(corpus)
    settings = Settings(
        corpus_root=corpus,
        data_dir=tmp_path / "data",
        backend=backend,
        chunk_size=64,
        chunk_overlap=8,
    )
    searches = (
        (
            "orbit lantern zenith record stars dust",
            "observations.pdf",
            "The orbit lantern zenith record describes stars and dust.",
            2,
            [],
        ),
        (
            "orbit lantern garden record moss rain",
            "field-notes.md",
            "The orbit lantern garden record describes moss and rain.",
            None,
            ["Field notes", "Garden"],
        ),
    )

    with TestClient(create_app(settings)) as client:
        snapshots = []
        for _ in range(2):
            built = client.post("/index/build")
            assert built.status_code == 200, built.text
            counts = {
                key: built.json()[key] for key in ("indexed", "skipped", "failed")
            }
            assert counts == {
                "indexed": 3,
                "skipped": 2,
                "failed": 0,
            }

            results = []
            for query, path, text, page, heading in searches:
                response = client.post("/search", json={"query": query, "top_n": 5})
                assert response.status_code == 200, response.text
                payload = response.json()
                assert payload["total_hits"] >= 2
                assert payload["rerank_skipped"] is False
                top = payload["results"][0]
                assert top["source"]["path"] == path
                assert top["passage"]["text"] == text
                assert top["passage"]["page_number"] == page
                assert top["passage"]["heading_path"] == heading
                results.append((payload["results"], payload["total_hits"]))
            snapshots.append(results)

    assert snapshots[0] == snapshots[1]
    manifest = SQLiteManifest(settings.data_dir)
    scan = manifest.get_source("scan.pdf")
    unsupported = manifest.get_source("unsupported.bin")
    assert scan is not None and scan.status == "skipped:requires_ocr"
    assert unsupported is not None and unsupported.status == "skipped:unsupported"
