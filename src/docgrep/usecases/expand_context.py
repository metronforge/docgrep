"""Return passages with the text around them, read-only (UC-12)."""

import logging
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from docgrep.domain.context import (
    ContextResult,
    ContextWindow,
    OutlineEntry,
    PassageContext,
    select_window,
    stitch,
)
from docgrep.domain.errors import ValidationError
from docgrep.domain.models import ManifestSourceRecord, Passage, Source, is_passage_id
from docgrep.domain.protocols import (
    OcrOutputOwners,
    PassageContextReader,
    SourceRecordReader,
)
from docgrep.usecases.find_passages import RequestSanitizer

_LOG = logging.getLogger(__name__)

NOT_FOUND_HINT = (
    "No passage with this id is in the index; an update may have removed it. "
    "Search again for current passage ids."
)
SOURCE_REMOVED_HINT = (
    "The source of this passage is no longer indexed. "
    "Search again for current passage ids."
)
SOURCE_MODIFIED_HINT = (
    "The source changed after this passage was indexed. "
    "Search again for current passage ids."
)


def _outline_entry(passage: Passage) -> OutlineEntry:
    return OutlineEntry(passage.passage_id, passage.chunk_offset, passage.token_count)


def _contiguous(
    entries: Sequence[OutlineEntry], texts: dict[str, Passage]
) -> tuple[Passage, ...]:
    """The neighbours whose text was read, from the passage outward, up to
    the first one that was not: a window never has a hole."""
    kept: list[Passage] = []
    for entry in entries:
        passage = texts.get(entry.passage_id)
        if passage is None or passage.chunk_offset != entry.chunk_offset:
            break
        kept.append(passage)
    return tuple(kept)


class ExpandContext:
    """Return each requested passage with the passages around it.

    The window comes from the neighbouring indexed passages of the same
    source (ADR-0022): one outline listing per distinct source and one text
    fetch per request. The manifest says whether the source is still
    indexed and unchanged; the shown source metadata is sanitized like
    search results (ADR-0021), with the OCR registry read at most once.
    Nothing is written, no lock is taken and the corpus is not read.
    """

    def __init__(
        self,
        *,
        store: PassageContextReader,
        manifest: SourceRecordReader,
        ocr: OcrOutputOwners | None,
        max_passages: int,
        default_window_tokens: int,
        max_window_tokens: int,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if max_passages <= 0:
            raise ValueError("max_passages must be positive")
        if not 0 <= default_window_tokens <= max_window_tokens:
            raise ValueError("the default window must be within the maximum")
        self.store = store
        self.manifest = manifest
        self.ocr = ocr
        self.max_passages = max_passages
        self.default_window_tokens = default_window_tokens
        self.max_window_tokens = max_window_tokens
        self.now = now

    def _validated(
        self, passage_ids: Sequence[str], window_tokens: int | None
    ) -> tuple[list[str], int]:
        if not 1 <= len(passage_ids) <= self.max_passages:
            raise ValidationError(
                f"passage_ids must hold between 1 and {self.max_passages} ids",
                details={"minimum": 1, "maximum": self.max_passages},
            )
        malformed = [item for item in passage_ids if not is_passage_id(item)]
        if malformed:
            raise ValidationError(
                "passage_ids must be passage ids from search or browse results",
                details={"malformed": malformed},
            )
        window = self.default_window_tokens if window_tokens is None else window_tokens
        if not 0 <= window <= self.max_window_tokens:
            raise ValidationError(
                f"window_tokens must be between 0 and {self.max_window_tokens}",
                details={"minimum": 0, "maximum": self.max_window_tokens},
            )
        return list(dict.fromkeys(passage_ids)), window

    def run(
        self, passage_ids: Sequence[str], *, window_tokens: int | None = None
    ) -> ContextResult:
        """One entry per distinct id, in request order; a failed entry never
        aborts the others. Raises ``ValidationError`` before any read for an
        invalid request and ``StoreUnavailable`` when the store cannot be
        read."""
        unique, window_size = self._validated(passage_ids, window_tokens)
        found = dict(zip(unique, self.store.fetch_passages(unique), strict=True))

        records: dict[str, ManifestSourceRecord | None] = {}
        outlines: dict[str, tuple[OutlineEntry, ...]] = {}
        failures: dict[str, PassageContext] = {}
        windows: dict[str, tuple[Passage, Source, ContextWindow]] = {}
        for passage_id in unique:
            hit = found[passage_id]
            if hit is None:
                failures[passage_id] = PassageContext(
                    passage_id, "not_found", hint=NOT_FOUND_HINT
                )
                continue
            passage, source = hit
            if source.path not in records:
                records[source.path] = self.manifest.get_source(source.path)
            failure = self._stale(passage_id, source, records[source.path])
            if failure is not None:
                failures[passage_id] = failure
                continue
            if source.source_id not in outlines:
                outlines[source.source_id] = self.store.source_outline(source.source_id)
            windows[passage_id] = (
                passage,
                source,
                select_window(
                    outlines[source.source_id], _outline_entry(passage), window_size
                ),
            )

        held = {hit[0].passage_id: hit[0] for hit in found.values() if hit is not None}
        wanted = {
            entry.passage_id
            for _, _, window in windows.values()
            for entry in (*window.before, *window.after)
        } - set(held)
        texts = dict(held)
        if wanted:
            texts.update(self.store.passage_texts(wanted))

        sanitizer = RequestSanitizer(self.now().year, self.ocr, self.manifest)
        cleaned: dict[str, Source] = {}
        entries: list[PassageContext] = []
        for passage_id in unique:
            if passage_id in failures:
                entries.append(failures[passage_id])
                continue
            passage, source, window = windows[passage_id]
            before = _contiguous(window.before[::-1], texts)[::-1]
            after = _contiguous(window.after, texts)
            if len(before) < len(window.before) or len(after) < len(window.after):
                # The source changed while it was read; the window stops at
                # the first neighbour that is gone.
                _LOG.warning(
                    "context_neighbour_missing", extra={"passage_id": passage_id}
                )
            if source.source_id not in cleaned:
                cleaned[source.source_id] = sanitizer.clean(source)
            entries.append(
                PassageContext(
                    passage_id,
                    "ok",
                    passage=passage,
                    source=cleaned[source.source_id],
                    context=stitch(
                        passage,
                        before,
                        after,
                        at_source_start=window.at_source_start
                        and len(before) == len(window.before),
                        at_source_end=window.at_source_end
                        and len(after) == len(window.after),
                    ),
                )
            )
        statuses = Counter(entry.status for entry in entries)
        # Counts only: no passage id or text is logged here.
        _LOG.info(
            "context_expanded",
            extra={
                "requested": len(unique),
                "window_tokens": window_size,
                "sources": len(outlines),
                **{f"status_{name}": count for name, count in statuses.items()},
            },
        )
        return ContextResult(tuple(entries), window_size)

    @staticmethod
    def _stale(
        passage_id: str, source: Source, record: ManifestSourceRecord | None
    ) -> PassageContext | None:
        """The failure of a passage whose source is gone or changed."""
        if record is None or record.status != "indexed":
            return PassageContext(
                passage_id,
                "source_removed",
                hint=SOURCE_REMOVED_HINT,
                path=source.path,
            )
        if record.source.content_hash != source.content_hash:
            return PassageContext(
                passage_id,
                "source_modified",
                hint=SOURCE_MODIFIED_HINT,
                path=source.path,
                indexed_content_hash=source.content_hash,
                current_content_hash=record.source.content_hash,
            )
        return None
