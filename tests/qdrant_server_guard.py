"""Refuse to run the qdrant_server tests against a server that holds other data.

The server-mode tests create, swap and delete collections and aliases. Before
the first of them runs, the server is listed once; any collection or alias the
suite does not own stops the whole run, and nothing is deleted.

Ownership is decided by name. The suite owns the aliases in ``SUITE_ALIASES``
and the collections the store creates for them (the alias, an underscore and 32
hex digits); an owned alias must also point at an owned collection, because a
rebuild through it drops the collection it pointed at. The default alias
``docgrep`` is deliberately not owned: a real index uses it, so its leftovers
cannot be told apart from a library and also stop the run.
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

SUITE_ALIASES = frozenset(
    {
        "alpha",
        "beta",
        "upclean",
        "upupdated",
        "rsclean",
        "rsint",
        "bm25shape",
        "reempty",
    }
)

_SUITE_COLLECTION = re.compile(
    "(?:" + "|".join(sorted(SUITE_ALIASES)) + r")_[0-9a-f]{32}"
)


@dataclass(frozen=True)
class ForeignContent:
    """Names on the server that the suite does not own, sorted."""

    collections: tuple[str, ...]
    aliases: tuple[str, ...]

    def __bool__(self) -> bool:
        return bool(self.collections or self.aliases)


def suite_owns_collection(name: str) -> bool:
    return _SUITE_COLLECTION.fullmatch(name) is not None


def suite_owns_alias(name: str, collection: str) -> bool:
    return (
        name in SUITE_ALIASES
        and suite_owns_collection(collection)
        and collection.startswith(f"{name}_")
    )


def foreign_content(
    collections: Iterable[str], aliases: Mapping[str, str]
) -> ForeignContent:
    """Collections and aliases (alias name to collection) the suite does not own."""
    return ForeignContent(
        tuple(sorted(name for name in collections if not suite_owns_collection(name))),
        tuple(
            sorted(
                name
                for name, collection in aliases.items()
                if not suite_owns_alias(name, collection)
            )
        ),
    )


def refusal_message(url: str, foreign: ForeignContent) -> str:
    collections = ", ".join(foreign.collections) or "(none)"
    aliases = ", ".join(foreign.aliases) or "(none)"
    return (
        f"Refusing to run the qdrant_server tests against {url}: the server holds "
        f"collections or aliases that the test suite does not own. "
        f"Foreign collections: {collections}. Foreign aliases: {aliases}. "
        "The server-mode tests delete collections and aliases, so they need a "
        "Qdrant server of their own: start a throwaway container on another "
        "port and point DOCGREP_TEST_QDRANT_URL at it. Nothing was deleted."
    )


def check_server(url: str) -> str | None:
    """Return a refusal message if the server holds foreign content, else None.

    A server that cannot be reached is not refused here; the tests then fail to
    connect exactly as they would without this check.
    """
    client = QdrantClient(url=url, check_compatibility=False)
    try:
        collections = [c.name for c in client.get_collections().collections]
        aliases = {
            a.alias_name: a.collection_name for a in client.get_aliases().aliases
        }
    except ResponseHandlingException:
        return None
    finally:
        client.close()
    foreign = foreign_content(collections, aliases)
    return refusal_message(url, foreign) if foreign else None
