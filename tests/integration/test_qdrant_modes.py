"""Store selection, URL validation and the local-mode size warning."""

import logging
from pathlib import Path

import pytest

from docgrep.adapters.qdrant import (
    LOCAL_POINT_LIMIT,
    QdrantPassageStore,
    local_passage_count,
    warn_if_local_mode_is_large,
)
from docgrep.domain.errors import StoreUnavailable, ValidationError
from docgrep.domain.models import FileType, Passage, Source, make_passage_id
from docgrep.settings import Settings

ADVICE = (
    "local mode is intended for tests and small corpora; "
    "set DOCGREP_QDRANT_URL for larger corpora"
)


def _fill(store: QdrantPassageStore, count: int) -> None:
    source = Source("a.txt", "a.txt", FileType.TXT)
    store.recreate_collection(2)
    for offset in range(count):
        passage = Passage(make_passage_id("a.txt", offset), "a.txt", "t", offset, 1)
        store.upsert(passage, source, (1.0, 0.0), {1: 1.0})
    store.write_completion_metadata({"embedding_model": "m"})


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "ftp://host:6333",
        "localhost:6333",
        "http://",
        "not a url",
        "http://host:notaport",
        "http://host:99999",
    ],
)
def test_invalid_server_urls_are_structured_configuration_errors(url: str) -> None:
    with pytest.raises(ValidationError) as raised:
        QdrantPassageStore(url=url)

    assert raised.value.category == "validation_error"
    assert raised.value.details == {"setting": "qdrant_url"}


def test_unreachable_server_is_a_store_error_without_credentials() -> None:
    with pytest.raises(StoreUnavailable) as raised:
        QdrantPassageStore(url="http://user:secret@127.0.0.1:1")

    assert raised.value.category == "store_unavailable"
    assert raised.value.details["url"] == "http://127.0.0.1:1"
    assert "secret" not in str(raised.value) + str(raised.value.details)


def test_exactly_one_of_path_and_url_is_required(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        QdrantPassageStore()
    with pytest.raises(ValueError, match="exactly one"):
        QdrantPassageStore(tmp_path / "q", url="http://127.0.0.1:6333")


def test_qdrant_url_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DOCGREP_QDRANT_URL", raising=False)
    assert Settings().qdrant_url is None

    monkeypatch.setenv("DOCGREP_QDRANT_URL", "http://qdrant:6333")
    assert Settings().qdrant_url == "http://qdrant:6333"


def test_warning_helper_uses_a_strict_limit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="docgrep"):
        warn_if_local_mode_is_large(LOCAL_POINT_LIMIT)
        assert caplog.records == []
        warn_if_local_mode_is_large(LOCAL_POINT_LIMIT + 1)

    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert record.getMessage() == ADVICE
    assert record.__dict__["passages"] == LOCAL_POINT_LIMIT + 1
    assert record.__dict__["limit"] == LOCAL_POINT_LIMIT == 20_000


def test_build_end_warns_only_above_the_local_limit(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="docgrep"):
        _fill(QdrantPassageStore(tmp_path / "ok", local_point_limit=3), 3)
        assert [r.getMessage() for r in caplog.records] == []
        _fill(QdrantPassageStore(tmp_path / "big", local_point_limit=2), 3)

    [record] = [r for r in caplog.records if r.getMessage() == ADVICE]
    assert record.__dict__["passages"] == 3 and record.__dict__["limit"] == 2


def test_local_passage_count_does_not_keep_the_storage_lock(tmp_path: Path) -> None:
    path = tmp_path / "qdrant"
    assert local_passage_count(path) is None
    store = QdrantPassageStore(path)
    _fill(store, 3)
    store.close()

    assert local_passage_count(path) == 3
    reopened = QdrantPassageStore(path)
    assert reopened.count_passages() == 3
    reopened.close()


def test_local_passage_count_is_none_while_another_client_holds_the_store(
    tmp_path: Path,
) -> None:
    store = QdrantPassageStore(tmp_path / "qdrant")
    _fill(store, 2)

    assert local_passage_count(tmp_path / "qdrant") is None
