"""POST /passages/context over the real stack (UC-12): the window read from
local Qdrant against the parsed documents, per-id statuses, request
validation and a missing index."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from context_support import MISSING, source_passages
from docgrep.adapters.manifest import SQLiteManifest
from docgrep.adapters.parsers.docx import DocxParser
from docgrep.adapters.parsers.pdf import PdfParser
from docgrep.api.container import create_container
from docgrep.api.http import create_app
from docgrep.domain.context import SECTION_BREAK
from docgrep.domain.models import ParsedSection
from docgrep.settings import Settings
from mcp_support import built_library, settings_for, snapshot
from resume_support import close


def _post(settings: Settings, body: Any) -> Any:
    services = create_container(settings)
    try:
        with TestClient(
            create_app(settings, services=services), base_url="http://127.0.0.1"
        ) as client:
            return client.post("/passages/context", json=body)
    finally:
        close(services)


def _whole(entry: dict[str, Any]) -> str:
    return str(
        entry["context_before"] + entry["passage"]["text"] + entry["context_after"]
    )


def _sections(settings: Settings, path: str) -> tuple[ParsedSection, ...]:
    manifest = SQLiteManifest(settings.data_dir, read_only=True)
    record = manifest.get_source(path)
    assert record is not None
    parser = (PdfParser if path.endswith(".pdf") else DocxParser)(settings.corpus_root)
    return tuple(
        section
        for section in parser.parse(record.source).sections
        if section.text.strip()
    )


@pytest.mark.parametrize("path", ["paper.pdf", "memo.docx"])
def test_a_wide_window_rebuilds_the_parsed_document(tmp_path: Path, path: str) -> None:
    settings = built_library(tmp_path)
    passages = source_passages(settings, path)
    sections = _sections(settings, path)
    # Every section of these documents has its own page or heading path, so
    # each break is a blank line; inside a section the overlap is cut.
    expected = SECTION_BREAK.join(section.text.strip() for section in sections)

    response = _post(
        settings,
        {"passage_ids": [item.passage_id for item in passages], "window_tokens": 5000},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["window_tokens"] == 5000
    assert len(body["results"]) == len(passages) > 0
    for entry, passage in zip(body["results"], passages, strict=True):
        assert entry["status"] == "ok"
        assert entry["passage"]["text"] == passage.text
        assert _whole(entry) == expected
        assert entry["at_source_start"] and entry["at_source_end"]
        assert entry["source"]["path"] == path
    if path == "memo.docx":
        locations = body["results"][0]["after_locations"]
        assert all(item["page_number"] is None for item in locations)


def test_an_ok_entry_has_the_documented_shape(tmp_path: Path) -> None:
    settings = built_library(tmp_path)
    first, second, *_ = source_passages(settings, "paper.pdf")

    body = _post(settings, {"passage_ids": [first.passage_id]}).json()

    (entry,) = body["results"]
    assert body["window_tokens"] == 500
    assert set(entry) == {
        "passage_id",
        "status",
        "passage",
        "source",
        "context_before",
        "context_after",
        "before_locations",
        "after_locations",
        "at_source_start",
        "at_source_end",
        "window_tokens_used",
    }
    assert entry["passage"] == {
        "passage_id": first.passage_id,
        "text": first.text,
        "page_number": 1,
        "heading_path": [],
    }
    assert entry["context_before"] == "" and entry["at_source_start"] is True
    assert entry["after_locations"][0] == {"page_number": 1, "heading_path": []}
    assert entry["window_tokens_used"]["before"] == 0
    assert entry["window_tokens_used"]["after"] >= second.token_count


def test_mixed_statuses_answer_200_in_request_order(tmp_path: Path) -> None:
    settings = built_library(tmp_path)
    paper = source_passages(settings, "paper.pdf")[0]
    memo = source_passages(settings, "memo.docx")[0]
    note = source_passages(settings, "notes/a.txt")[0]
    manifest = SQLiteManifest(settings.data_dir)
    record = manifest.get_source("memo.docx")
    assert record is not None
    manifest.record_source(replace(record.source, content_hash="0" * 64), "indexed")
    manifest.remove_source("notes/a.txt")

    response = _post(
        settings,
        {
            "passage_ids": [
                paper.passage_id,
                MISSING,
                memo.passage_id,
                note.passage_id,
                paper.passage_id,
            ],
            "window_tokens": 0,
        },
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert [(item["passage_id"], item["status"]) for item in results] == [
        (paper.passage_id, "ok"),
        (MISSING, "not_found"),
        (memo.passage_id, "source_modified"),
        (note.passage_id, "source_removed"),
    ]
    assert set(results[1]) == {"passage_id", "status", "hint"}
    assert results[2]["path"] == "memo.docx"
    assert results[2]["indexed_content_hash"] == record.source.content_hash
    assert results[2]["current_content_hash"] == "0" * 64
    assert "Search again" in results[2]["hint"]
    assert results[3]["path"] == "notes/a.txt"


@pytest.mark.parametrize(
    ("body", "message", "details"),
    [
        (
            {"passage_ids": []},
            "passage_ids must hold between 1 and 3 ids",
            {"minimum": 1, "maximum": 3},
        ),
        (
            {"passage_ids": [MISSING] * 4},
            "passage_ids must hold between 1 and 3 ids",
            {"minimum": 1, "maximum": 3},
        ),
        (
            {"passage_ids": [MISSING, "abc"]},
            "passage_ids must be passage ids from search or browse results",
            {"malformed": ["abc"]},
        ),
        (
            {"passage_ids": [MISSING], "window_tokens": -1},
            "window_tokens must be between 0 and 900",
            {"minimum": 0, "maximum": 900},
        ),
        (
            {"passage_ids": [MISSING], "window_tokens": 901},
            "window_tokens must be between 0 and 900",
            {"minimum": 0, "maximum": 900},
        ),
    ],
)
def test_an_invalid_request_is_a_validation_error(
    tmp_path: Path, body: dict[str, Any], message: str, details: dict[str, Any]
) -> None:
    settings = built_library(tmp_path).model_copy(
        update={"context_max_passages": 3, "context_max_window_tokens": 900}
    )

    response = _post(settings, body)

    assert response.status_code == 400
    assert response.json() == {
        "category": "validation_error",
        "message": message,
        "details": details,
    }


@pytest.mark.parametrize(
    ("body", "location"),
    [
        ({"passage_ids": [MISSING], "window": 5}, ["body", "window"]),
        ({"passage_ids": MISSING}, ["body", "passage_ids"]),
        (
            {"passage_ids": [MISSING], "window_tokens": "wide"},
            ["body", "window_tokens"],
        ),
        ({}, ["body", "passage_ids"]),
    ],
)
def test_a_malformed_body_is_a_validation_error(
    tmp_path: Path, body: dict[str, Any], location: list[str]
) -> None:
    settings = built_library(tmp_path)

    response = _post(settings, body)

    assert response.status_code == 400
    payload = response.json()
    assert payload["category"] == "validation_error"
    assert [item["location"] for item in payload["details"]["errors"]] == [location]


def test_without_an_index_every_id_is_not_found_and_nothing_is_created(
    tmp_path: Path,
) -> None:
    (tmp_path / "corpus").mkdir()
    settings = settings_for(tmp_path)

    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        found = client.post("/passages/context", json={"passage_ids": [MISSING]})
        invalid = client.post("/passages/context", json={"passage_ids": ["x"]})

    assert found.status_code == 200
    assert [item["status"] for item in found.json()["results"]] == ["not_found"]
    assert invalid.status_code == 400
    assert not (tmp_path / "data").exists()


def test_expanding_writes_nothing(tmp_path: Path) -> None:
    settings = built_library(tmp_path)
    ids = [item.passage_id for item in source_passages(settings, "paper.pdf")]
    services = create_container(settings)
    try:
        before = snapshot(tmp_path / "corpus", tmp_path / "data")
        with TestClient(
            create_app(settings, services=services), base_url="http://127.0.0.1"
        ) as client:
            response = client.post("/passages/context", json={"passage_ids": ids})
        after = snapshot(tmp_path / "corpus", tmp_path / "data")
    finally:
        close(services)

    assert response.status_code == 200
    # No lock file is created or touched, and nothing else changes.
    assert after == before


def test_the_openapi_document_declares_the_models(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)

    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        document = client.get("/openapi.json").json()

    operation = document["paths"]["/passages/context"]["post"]
    schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert schema == {"$ref": "#/components/schemas/PassageContextResponse"}
    components = document["components"]["schemas"]
    for name in (
        "PassageContextRequest",
        "PassageContextOk",
        "PassageNotFound",
        "PassageSourceRemoved",
        "PassageSourceModified",
    ):
        assert name in components
    entries = components["PassageContextResponse"]["properties"]["results"]["items"]
    assert entries["discriminator"]["propertyName"] == "status"
    assert components["PassageContextRequest"]["additionalProperties"] is False
