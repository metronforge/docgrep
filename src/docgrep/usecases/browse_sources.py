"""List the indexed sources with their metadata, read-only (UC-08)."""

import logging
from collections.abc import Sequence

from docgrep.domain.errors import IndexIncomplete, StoreUnavailable
from docgrep.domain.models import FileType, Source
from docgrep.domain.protocols import (
    CommittedSourceReader,
    OcrOutputOwners,
    SourcePassageCounter,
)
from docgrep.domain.sources import (
    BrowsedSource,
    BrowseRequest,
    BrowseResult,
    OcrStatus,
    file_type_counts,
    sort_sources,
)

_LOG = logging.getLogger(__name__)


class _NotPublished(Exception):
    """No index is published, so nothing is browsable."""


class BrowseSources:
    """Filter, sort and page the committed indexed sources.

    The manifest gives the sources and their metadata; the store gives
    passage counts at query time (ADR-0008), for the returned page only,
    except when sorting by ``passage_count``, which counts every match; the
    OCR registry says whether a source is an OCR output. Nothing is written,
    no lock is taken and the corpus is not read.
    """

    def __init__(
        self,
        *,
        manifest: CommittedSourceReader,
        store: SourcePassageCounter,
        ocr: OcrOutputOwners,
    ) -> None:
        self.manifest = manifest
        self.store = store
        self.ocr = ocr

    def run(self, request: BrowseRequest) -> BrowseResult:
        matching = [
            record.source
            for record in self.manifest.committed_sources()
            if record.status == "indexed" and request.selects(record.source)
        ]
        if not matching:
            return BrowseResult.empty(request)
        start = (request.page - 1) * request.page_size
        try:
            counts: dict[str, int | None] = {}
            if request.sort_by == "passage_count":
                # The expensive case: the order depends on every count.
                counts = self._counts(matching)
            ordered = sort_sources(
                matching, request.sort_by, request.sort_order, counts
            )
            page = ordered[start : start + request.page_size]
            if request.sort_by != "passage_count":
                if page:
                    counts = self._counts(page)
                else:
                    # A page past the end still has to know whether an
                    # index is published at all.
                    self._counts(ordered[:1])
        except _NotPublished:
            return BrowseResult.empty(request)
        result = BrowseResult(
            tuple(
                BrowsedSource(
                    source,
                    counts[source.source_id],
                    self._ocr_status(source),
                    degraded=counts[source.source_id] is None,
                )
                for source in page
            ),
            len(matching),
            request.page,
            request.page_size,
            file_type_counts(matching),
            request.applied_filters(),
        )
        _LOG.info(
            "browse_completed",
            extra={
                "total_hits": result.total_hits,
                "returned": len(result.sources),
                "degraded": sum(item.degraded for item in result.sources),
            },
        )
        return result

    def _counts(self, sources: Sequence[Source]) -> dict[str, int | None]:
        """Passage counts by source id; None for a source whose count failed.

        Raises ``StoreUnavailable`` when every count failed and
        ``_NotPublished`` when no index is published.
        """
        counts: dict[str, int | None] = {}
        failure: StoreUnavailable | None = None
        for source in sources:
            try:
                counts[source.source_id] = self.store.count_source_passages(
                    source.source_id
                )
            except IndexIncomplete as error:
                raise _NotPublished from error
            except StoreUnavailable as error:
                failure = failure or error
                counts[source.source_id] = None
                _LOG.warning(
                    "browse_count_failed source_id=%s",
                    source.source_id,
                    extra={"source_id": source.source_id},
                )
        if failure is not None and all(value is None for value in counts.values()):
            raise failure
        return counts

    def _ocr_status(self, source: Source) -> OcrStatus:
        # OCR always writes a PDF, so no other type can be an OCR output.
        if source.file_type is FileType.PDF and self.ocr.output_owner(source.path):
            return "processed_via_ocr"
        return "not_applicable"
