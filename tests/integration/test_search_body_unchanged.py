"""The public search body is unchanged by the diagnostic seam (ADR-0024):
over HTTP and from the use case, byte for byte against bodies recorded on
``main`` before ``FindPassages.diagnose`` existed."""

from pathlib import Path

from search_regression_support import (
    http_bodies,
    http_client,
    load_fixture,
    use_case_bodies,
)


def test_search_bodies_over_http_are_byte_identical(tmp_path: Path) -> None:
    with http_client(tmp_path) as api:
        bodies = http_bodies(api)

    assert bodies == load_fixture()["http"]


def test_search_bodies_of_the_use_case_are_byte_identical() -> None:
    assert use_case_bodies() == load_fixture()["use_case"]
