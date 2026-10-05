"""Pure rules of scan discovery (UC-5): which pages to sample and how a
file's text layer is classified."""

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256

# Pages sampled from every file: the first ones, the last one and a few
# from the middle, chosen by a stable hash of the path and the page count.
FIRST_PAGES = 3
LAST_PAGES = 1
MIDDLE_PAGES = 2
# Files beyond these sizes are flagged as large (OCR will be expensive).
LARGE_FILE_PAGES = 5000
LARGE_FILE_BYTES = 1024**3
# A file whose sampled pages are at least this share broken (rule R1) has a
# garbled text layer, which OCR must redo: it is classified as scanned.
GARBLED_SAMPLE_SHARE = 0.5

NATIVE = "native"
SCANNED = "scanned"
MIXED_OR_UNCERTAIN = "mixed_or_uncertain"
INSPECTION_FAILED = "inspection_failed"
NOT_PROCESSED = "not_processed"

LOW_TEXT_DENSITY = "low_text_density"
UNCERTAIN_TEXT_DENSITY = "uncertain_text_density"
GARBLED_TEXT_LAYER = "garbled_text_layer"
UNSAMPLABLE = "unsamplable"


@dataclass(frozen=True, slots=True)
class PageSample:
    """One sampled page: non-whitespace characters and whether rule R1 of
    the page policy would drop it. Never any text."""

    page_number: int
    non_whitespace: int
    dropped: bool


def _stable_index(seed: str, counter: int, size: int) -> int:
    digest = sha256(f"{seed}\x00{counter}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % size


def sample_pages(path: str, page_count: int) -> tuple[int, ...]:
    """The 1-based pages to sample, ascending and without duplicates.

    The first ``FIRST_PAGES`` and last ``LAST_PAGES`` pages, plus
    ``MIDDLE_PAGES`` pages from between them chosen by SHA-256 of the
    relative path and the page count, so the same file always gives the same
    sample. Capped at the page count.
    """
    if page_count <= 0:
        return ()
    chosen = set(range(1, min(FIRST_PAGES, page_count) + 1))
    chosen |= set(range(max(page_count - LAST_PAGES + 1, 1), page_count + 1))
    middle = [p for p in range(1, page_count + 1) if p not in chosen]
    seed = f"{path}\x00{page_count}"
    counter = 0
    while middle and len(chosen) < (
        min(FIRST_PAGES, page_count) + LAST_PAGES + MIDDLE_PAGES
    ):
        chosen.add(middle.pop(_stable_index(seed, counter, len(middle))))
        counter += 1
    return tuple(sorted(chosen))


def text_density(samples: Sequence[PageSample]) -> float:
    """Mean non-whitespace characters per sampled page; a page that rule R1
    would drop counts as zero."""
    if not samples:
        return 0.0
    return sum(0 if s.dropped else s.non_whitespace for s in samples) / len(samples)


def classify(
    samples: Sequence[PageSample], *, lower: float, upper: float
) -> tuple[str, str | None]:
    """Classification and reason for a file's sampled pages (not empty).

    Mostly broken pages: scanned with garbled_text_layer. Otherwise density
    below ``lower``: scanned; above ``upper``: native; in between (bounds
    included): mixed or uncertain.
    """
    dropped = sum(s.dropped for s in samples)
    if dropped >= GARBLED_SAMPLE_SHARE * len(samples):
        return SCANNED, GARBLED_TEXT_LAYER
    density = text_density(samples)
    if density < lower:
        return SCANNED, LOW_TEXT_DENSITY
    if density > upper:
        return NATIVE, None
    return MIXED_OR_UNCERTAIN, UNCERTAIN_TEXT_DENSITY
