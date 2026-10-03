"""Store selection, URL validation and the local-mode size warning."""

import logging
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

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


@pytest.mark.parametrize(
    "alias", ["", " ", "a b", "a/b", "a.b", "alias!", "é", "a\n", "-x y"]
)
def test_invalid_aliases_are_structured_configuration_errors(
    tmp_path: Path, alias: str
) -> None:
    with pytest.raises(ValidationError) as raised:
        QdrantPassageStore(tmp_path / "q", alias=alias)

    assert raised.value.category == "validation_error"
    assert raised.value.details == {"setting": "qdrant_alias"}


def test_qdrant_alias_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DOCGREP_QDRANT_ALIAS", raising=False)
    assert Settings().qdrant_alias == "docgrep"

    monkeypatch.setenv("DOCGREP_QDRANT_ALIAS", "library-2")
    assert Settings().qdrant_alias == "library-2"


def _names(path: Path) -> tuple[dict[str, str], list[str]]:
    client = QdrantClient(path=str(path))
    try:
        aliases = {
            a.alias_name: a.collection_name for a in client.get_aliases().aliases
        }
        return aliases, [c.name for c in client.get_collections().collections]
    finally:
        client.close()


@pytest.mark.parametrize(
    ("alias", "prefix"), [("docgrep", "docgrep_"), ("a-b_9", "a-b_9_")]
)
def test_alias_names_the_read_alias_and_the_collections(
    tmp_path: Path, alias: str, prefix: str
) -> None:
    path = tmp_path / "qdrant"
    store = QdrantPassageStore(path, alias=alias)
    _fill(store, 2)
    assert store.count_passages() == 2
    _fill(store, 3)
    store.close()

    aliases, collections = _names(path)
    assert list(aliases) == [alias]
    assert collections == [aliases[alias]]
    assert collections[0].startswith(prefix)
    assert local_passage_count(path, alias) == 3
    assert local_passage_count(path, "other") == 0


def test_the_default_alias_is_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "qdrant"
    store = QdrantPassageStore(path)
    _fill(store, 1)
    store.close()

    aliases, collections = _names(path)
    assert list(aliases) == ["docgrep"] and collections[0].startswith("docgrep_")
    assert local_passage_count(path) == 1
