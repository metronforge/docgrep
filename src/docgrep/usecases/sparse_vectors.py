"""Compute sparse vectors again from the passage text in the store (ADR-0025).

Shared by the full build, which ends with a re-encode of its staging
collection, and by ``ReencodeSparse``, which re-encodes the published one.
No model is involved: only the passage text and the sparse tokenizer.
"""

from collections.abc import Callable

from docgrep.domain.protocols import SparseEncoder, SparseEncoders, SparseVectorStore

# Passages read and written per store call.
DEFAULT_PAGE_SIZE = 512


def average_length(
    store: SparseVectorStore,
    encoders: SparseEncoders,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
    between_pages: Callable[[], None] | None = None,
) -> tuple[float, int]:
    """The average passage length in sparse tokens, over one scan of the
    store's target collection, and the number of passages scanned.

    A collection without passages, or whose passages have no token at all,
    has the average length 1.0, so that a weighting can still be recorded.
    """
    offset: str | None = None
    tokens = 0
    passages = 0
    while True:
        page, offset = store.passage_text_page(offset, page_size)
        for _, text in page:
            tokens += encoders.document_length(text)
        passages += len(page)
        if offset is None:
            break
        if between_pages is not None:
            between_pages()
    return (tokens / passages if passages and tokens else 1.0), passages


def encode_pages(
    store: SparseVectorStore,
    encoder: SparseEncoder,
    *,
    start: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    before_page: Callable[[], None] | None = None,
    after_page: Callable[[str | None, int], None] | None = None,
) -> int:
    """Write the sparse vector of every passage from ``start`` on, page by
    page; returns how many were written.

    ``before_page`` runs before each page is read (a cancel check);
    ``after_page`` gets the offset of the next page (None after the last)
    and the size of the page just written, once it is durable.
    """
    offset = start
    written = 0
    while True:
        if before_page is not None:
            before_page()
        page, following = store.passage_text_page(offset, page_size)
        store.write_sparse_vectors(
            [(passage_id, encoder.encode(text)) for passage_id, text in page]
        )
        written += len(page)
        if after_page is not None:
            after_page(following, len(page))
        if following is None:
            return written
        offset = following
