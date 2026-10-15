# ADR-0013: Incremental update writes in place

**Status:** Accepted
**Date:** 2026-10-03
**Amends:** ADR-0012 (run state).

## Context

A full build writes into a staging collection and publishes it by alias swap. An incremental update typically touches a handful of sources in an index of hundreds of thousands of passages, so building a staged copy would be wasteful. The library is also expected to gain parsers over time (for example EPUB), and sources that were skipped as unsupported must be picked up later without rebuilding everything.

## Decision

1. `docgrep update` writes to the live collection in place, one source at a time. The `sources` row of a source is written after that source's points are durable.
2. Change detection compares the corpus with the `sources` table: a source is added if it is not recorded, removed if it is recorded but no longer enumerated, and modified if its size or modification time changed and its content hash differs, if the registered parser's version differs from the recorded one, or if it was previously skipped as unsupported and a parser is now registered. Sources recorded as failed are retried unless disabled. Content hash is authoritative; size and modification time are only a pre-filter.
3. A modified source is rewritten insert-first: its new points are written, then its points that are not in the new set are removed, so a failed update never makes a source disappear from search. If the failure happens before the prune, the old content stays searchable and the unchanged `sources` row makes the next update detect the source again.
4. A removed source has its points deleted and its `sources` row removed.
5. Updates are idempotent and are recorded as runs of kind update. They are not resumable in the sense of ADR-0012: running update again continues the work. Only full builds are resumable, so the resumable-run rule considers the latest run of kind build only.
6. An update is refused when the latest build is resumable, or when the runtime configuration differs from the collection metadata in a way that affects vectors or chunking; both cases point to a full rebuild.

## Consequences

During an update a query may see a mix of old and new sources, and briefly both versions of one source. An update cannot change the embedding model or chunking; that needs a full rebuild.

## Note (2026-10-15)

Point 3 applies to failures while writing points (`store_error`). A source that cannot be prepared, because parsing or embedding failed, is recorded as `failed:<reason>` with the error message, whatever its row said before: added, indexed, skipped (including as unsupported) or failed. Status then counts it as failed, and the next update retries it instead of planning it as modified again. The points of a previously indexed source are not removed, so its old content stays searchable, as point 3 promises, until an update succeeds or the file is removed.
