"""The shared source filter and the values of source browsing (UC-08).

One definition of filter semantics serves browsing now and search filters
later: AND across fields, OR within a list, and a source whose filtered field
is null never matches that field. Matching is pure and reads nothing.
"""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, cast, get_args
from unicodedata import normalize

from docgrep.domain.errors import ValidationError
from docgrep.domain.globs import matching_pattern
from docgrep.domain.models import FileType, Source

# The file types a source can be indexed as; OCR outputs are indexed as pdf.
INDEXABLE_FILE_TYPES: tuple[FileType, ...] = (
    FileType.TXT,
    FileType.MARKDOWN,
    FileType.PDF,
    FileType.EPUB,
    FileType.DOCX,
    FileType.DJVU,
)

HasMetadata = Literal["any", "with_year", "with_author", "with_both", "without_any"]
SortKey = Literal["path", "indexed_at", "passage_count", "year", "title"]
SortOrder = Literal["asc", "desc"]
OcrStatus = Literal["not_applicable", "processed_via_ocr"]

HAS_METADATA_VALUES: tuple[str, ...] = get_args(HasMetadata)
SORT_KEYS: tuple[str, ...] = get_args(SortKey)
SORT_ORDERS: tuple[str, ...] = get_args(SortOrder)

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 500

# The filter fields in the order they are reported.
FILTER_FIELDS = (
    "source_ids",
    "authors",
    "year_from",
    "year_to",
    "file_types",
    "path_prefixes",
    "path_globs",
)


def _fold(text: str) -> str:
    """The form compared by case-insensitive matching."""
    return normalize("NFC", text).casefold()


def normalize_path(path: str) -> str:
    """A stored path or a path filter value with forward slashes."""
    return path.replace("\\", "/")


def distinct_texts(name: str, values: Iterable[str]) -> tuple[str, ...]:
    """Distinct non-blank values of a filter field, sorted."""
    found = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(
                f"{name} must not contain empty values", details={"field": name}
            )
        found.add(value)
    return tuple(sorted(found))


def parse_file_types(values: Iterable[str | FileType]) -> tuple[FileType, ...]:
    """Indexable file types named case-insensitively, distinct and sorted."""
    allowed = {item.value: item for item in INDEXABLE_FILE_TYPES}
    found = set()
    for value in values:
        key = str(value).strip().lower()
        if key not in allowed:
            names = sorted(allowed)
            raise ValidationError(
                f"unknown file type {str(value)!r}; allowed: {', '.join(names)}",
                details={"field": "file_types", "allowed": names},
            )
        found.add(allowed[key])
    return tuple(sorted(found, key=lambda item: item.value))


@dataclass(frozen=True, slots=True)
class SourceFilter:
    """Which sources to keep; every field is optional and lists are OR'ed.

    Construction canonicalises the values (distinct, sorted, paths with
    forward slashes, file types parsed case-insensitively) and raises
    ``ValidationError`` for blank values, unknown file types and a
    ``year_from`` after ``year_to``.
    """

    source_ids: tuple[str, ...] = ()
    authors: tuple[str, ...] = ()
    year_from: int | None = None
    year_to: int | None = None
    file_types: tuple[FileType, ...] = ()
    path_prefixes: tuple[str, ...] = ()
    path_globs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "source_ids", distinct_texts("source_ids", self.source_ids))
        set_(self, "authors", distinct_texts("authors", self.authors))
        set_(self, "file_types", parse_file_types(self.file_types))
        for name in ("path_prefixes", "path_globs"):
            values = distinct_texts(name, getattr(self, name))
            set_(self, name, tuple(sorted({normalize_path(v) for v in values})))
        for name in ("year_from", "year_to"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise ValidationError(
                    f"{name} must be an integer", details={"field": name}
                )
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValidationError(
                "year_from must not be after year_to",
                details={"year_from": self.year_from, "year_to": self.year_to},
            )

    def applied(self) -> dict[str, list[str] | int]:
        """The canonical form: only the fields that are set."""
        applied: dict[str, list[str] | int] = {}
        for name in FILTER_FIELDS:
            value = getattr(self, name)
            if isinstance(value, tuple):
                if value:
                    applied[name] = [str(item) for item in value]
            elif value is not None:
                applied[name] = value
        return applied


def _within_prefix(path: str, prefixes: Sequence[str]) -> bool:
    # A leading "/" anchors at the corpus root, where stored paths start.
    return any(path.startswith(prefix.removeprefix("/")) for prefix in prefixes)


def matches(source_filter: SourceFilter, source: Source) -> bool:
    """Whether a source passes every field of the filter that is set."""
    if source_filter.source_ids and source.source_id not in source_filter.source_ids:
        return False
    if source_filter.authors:
        if source.author is None:
            return False
        author = _fold(source.author)
        if not any(_fold(item) in author for item in source_filter.authors):
            return False
    if source_filter.year_from is not None or source_filter.year_to is not None:
        if source.year is None:
            return False
        if (
            source_filter.year_from is not None
            and source.year < source_filter.year_from
        ):
            return False
        if source_filter.year_to is not None and source.year > source_filter.year_to:
            return False
    if source_filter.file_types and source.file_type not in source_filter.file_types:
        return False
    path = normalize_path(source.path)
    if source_filter.path_prefixes and not _within_prefix(
        path, source_filter.path_prefixes
    ):
        return False
    if source_filter.path_globs and (
        matching_pattern(path, source_filter.path_globs) is None
    ):
        return False
    return True


def _known(value: str | None) -> bool:
    return value is not None and bool(value.strip())


def has_metadata(mode: HasMetadata, source: Source) -> bool:
    """Whether a source has the bibliographic metadata asked for: a year, an
    author, both, or neither (``without_any``); ``any`` keeps every source."""
    year = source.year is not None
    author = _known(source.author)
    if mode == "with_year":
        return year
    if mode == "with_author":
        return author
    if mode == "with_both":
        return year and author
    if mode == "without_any":
        return not year and not author
    return True


@dataclass(frozen=True, slots=True)
class BrowseRequest:
    """A filter, a metadata presence rule, a page and a sort order."""

    source_filter: SourceFilter = field(default_factory=SourceFilter)
    has_metadata: HasMetadata = "any"
    page: int = 1
    page_size: int = DEFAULT_PAGE_SIZE
    sort_by: SortKey = "path"
    sort_order: SortOrder = "asc"

    def __post_init__(self) -> None:
        if self.has_metadata not in HAS_METADATA_VALUES:
            raise ValidationError(
                f"has_metadata must be one of: {', '.join(HAS_METADATA_VALUES)}",
                details={"field": "has_metadata", "allowed": list(HAS_METADATA_VALUES)},
            )
        if self.sort_by not in SORT_KEYS:
            raise ValidationError(
                f"sort_by must be one of: {', '.join(SORT_KEYS)}",
                details={"field": "sort_by", "allowed": list(SORT_KEYS)},
            )
        if self.sort_order not in SORT_ORDERS:
            raise ValidationError(
                f"sort_order must be one of: {', '.join(SORT_ORDERS)}",
                details={"field": "sort_order", "allowed": list(SORT_ORDERS)},
            )
        if self.page < 1:
            raise ValidationError(
                "page must be at least 1", details={"field": "page", "minimum": 1}
            )
        if not 1 <= self.page_size <= MAX_PAGE_SIZE:
            raise ValidationError(
                f"page_size must be between 1 and {MAX_PAGE_SIZE}",
                details={"field": "page_size", "minimum": 1, "maximum": MAX_PAGE_SIZE},
            )

    def selects(self, source: Source) -> bool:
        """Whether a source passes the filter and the metadata rule."""
        return matches(self.source_filter, source) and has_metadata(
            self.has_metadata, source
        )

    def applied_filters(self) -> dict[str, list[str] | int | str]:
        """The canonical filter, with ``has_metadata`` unless it is ``any``."""
        applied: dict[str, list[str] | int | str] = {**self.source_filter.applied()}
        if self.has_metadata != "any":
            applied["has_metadata"] = self.has_metadata
        return applied


@dataclass(frozen=True, slots=True)
class BrowsedSource:
    """An indexed source with its passage count and OCR origin.

    ``passage_count`` is None and ``degraded`` true when the count failed.
    """

    source: Source
    passage_count: int | None
    ocr_status: OcrStatus = "not_applicable"
    degraded: bool = False


@dataclass(frozen=True, slots=True)
class BrowseResult:
    """One page of matching sources and aggregates over every match."""

    sources: tuple[BrowsedSource, ...]
    total_hits: int
    page: int
    page_size: int
    by_file_type: tuple[tuple[str, int], ...]
    applied_filters: Mapping[str, list[str] | int | str]

    def __post_init__(self) -> None:
        if self.total_hits < 0 or self.total_hits != sum(
            count for _, count in self.by_file_type
        ):
            raise ValueError("by_file_type must add up to total_hits")
        if len(self.sources) > self.page_size:
            raise ValueError("a page holds at most page_size sources")

    @classmethod
    def empty(cls, request: BrowseRequest) -> "BrowseResult":
        return cls(
            (), 0, request.page, request.page_size, (), request.applied_filters()
        )


def file_type_counts(sources: Iterable[Source]) -> tuple[tuple[str, int], ...]:
    """How many sources there are of each file type, by type name."""
    counts = Counter(source.file_type.value for source in sources)
    return tuple(sorted(counts.items()))


_SortValue = str | int


def _sort_value(
    key: SortKey, counts: Mapping[str, int | None]
) -> Callable[[Source], _SortValue | None]:
    if key == "path":
        return lambda source: source.path
    if key == "title":
        return lambda source: None if source.title is None else _fold(source.title)
    if key == "year":
        return lambda source: source.year
    if key == "indexed_at":
        return lambda source: source.indexed_at
    return lambda source: counts.get(source.source_id)


def sort_sources(
    sources: Iterable[Source],
    key: SortKey,
    order: SortOrder,
    counts: Mapping[str, int | None] | None = None,
) -> list[Source]:
    """Sources in the requested order.

    Ties are broken by ``source_id`` ascending and null values sort last, in
    both directions. Titles compare case-insensitively. ``counts`` holds the
    passage counts by source id, needed only to sort by ``passage_count``.
    """
    value = _sort_value(key, counts or {})
    by_id = sorted(sources, key=lambda source: source.source_id)
    known = [source for source in by_id if value(source) is not None]
    unknown = [source for source in by_id if value(source) is None]
    # Python's sort is stable even in reverse, so ties keep the id order.
    known.sort(
        key=lambda source: cast(_SortValue, value(source)), reverse=order == "desc"
    )
    return known + unknown
