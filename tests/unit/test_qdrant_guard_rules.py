"""Ownership rule of the guard in front of the qdrant_server tests."""

from qdrant_server_guard import (
    SUITE_ALIASES,
    ForeignContent,
    foreign_content,
    refusal_message,
)

HEX = "0123456789abcdef0123456789abcdef"


def test_an_empty_server_has_no_foreign_content() -> None:
    assert not foreign_content([], {})


def test_suite_leftovers_are_owned() -> None:
    collections = [f"{alias}_{HEX}" for alias in sorted(SUITE_ALIASES)]
    aliases = {alias: f"{alias}_{HEX}" for alias in SUITE_ALIASES}

    assert not foreign_content(collections, aliases)


def test_a_real_index_under_the_default_alias_is_foreign() -> None:
    found = foreign_content([f"docgrep_{HEX}"], {"docgrep": f"docgrep_{HEX}"})

    assert found == ForeignContent((f"docgrep_{HEX}",), ("docgrep",))


def test_unrelated_names_are_foreign_and_reported_sorted() -> None:
    found = foreign_content(
        ["zoo", f"alpha_{HEX}", "library", f"beta_{HEX}"],
        {"books": "library", "alpha": f"alpha_{HEX}"},
    )

    assert found == ForeignContent(("library", "zoo"), ("books",))


def test_near_misses_of_the_suite_naming_are_foreign() -> None:
    names = [
        "alpha",
        f"alpha_{HEX[:-1]}",
        f"alpha_{HEX}0",
        f"alpha_{HEX.upper()}",
        f"alphax_{HEX}",
        f"xalpha_{HEX}",
    ]

    assert foreign_content(names, {}).collections == tuple(sorted(names))


def test_a_suite_alias_pointing_at_a_foreign_collection_is_foreign() -> None:
    # A rebuild through the alias would drop the collection it points at.
    found = foreign_content(
        [f"beta_{HEX}"], {"alpha": "library", "beta": f"alpha_{HEX}"}
    )

    assert found.aliases == ("alpha", "beta")


def test_the_refusal_names_everything_and_says_nothing_was_deleted() -> None:
    message = refusal_message(
        "http://127.0.0.1:6336", ForeignContent(("library",), ("books",))
    )

    assert "http://127.0.0.1:6336" in message
    assert "library" in message and "books" in message
    assert "delete" in message and "Nothing was deleted" in message
    assert "server of their own" in message
