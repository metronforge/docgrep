# ADR-0020: Shared source filter and browse

**Status:** Accepted
**Date:** 2026-10-17

## Context

UC-08 lists the indexed sources with their metadata, filtered, sorted and paged. UC-02 will filter search results by the same source attributes. If the two had separate filter code, the same request could mean different things in each. The listing must reflect the published index, stay cheap for a library of a few thousand sources and never walk the corpus.

## Decision

**One filter, in the domain.** `SourceFilter` (in `docgrep.domain.sources`) has the fields `source_ids`, `authors`, `year_from`, `year_to`, `file_types`, `path_prefixes` and `path_globs`, all optional. `matches(filter, source)` is a pure function that answers yes or no for one `Source`. Search filters (UC-02) will call the same function to turn the matching manifest sources into a `source_id` allow-list for the store.

**Semantics.**
- Fields combine with AND; the values of one list field combine with OR.
- A source whose filtered field is null never matches that field: a source without a year never passes a year bound, one without an author never passes an author filter.
- `authors`: case-insensitive substring match (Unicode NFC, then case folding, on both sides). Blank values are rejected.
- `year_from` and `year_to`: inclusive bounds; `year_from` after `year_to` is rejected.
- `file_types`: matched case-insensitively against the file types a source can be indexed as: `pdf`, `markdown`, `txt`, `epub`, `docx` and `djvu`. Any other value is rejected with the allowed list.
- `source_ids`: exact, case-sensitive match.
- `path_prefixes` and `path_globs` match the path as stored in the manifest, which is the path shown in results: **relative to the corpus root, with `/` as the separator** (for example `Books/Math/arnold.pdf`). Backslashes in stored paths and in filter values are read as `/`. A prefix is a plain, case-sensitive string prefix (`Books/Ma` matches `Books/Math/...`); a single leading `/` is dropped, because stored paths start at the corpus root. Globs use the corpus pattern rules of `docgrep.domain.globs` (the same as `DOCGREP_INCLUDE_GLOBS`): `*`, `?` and `[...]` stay within a segment, `**` spans segments, and a pattern without `/` matches the file name at any depth.
- Unknown request fields are rejected by every front, never ignored.
- The canonical form (`applied_filters`) holds only the fields that are set, with list values de-duplicated and sorted and years as integers; `has_metadata` is added when it is not `any`. An empty filter is `{}`.

**Matching happens in Python against the manifest**, not as a Qdrant payload filter. This gives exact case-insensitive and glob semantics that do not depend on Qdrant text indexes or tokenizers, works the same in local and server mode, and keeps the filter testable without a store.

**Browsing.**
- Only sources whose committed manifest status is `indexed` are listed. Skipped and failed sources are reported by `docgrep status`.
- The request adds `has_metadata` (`any`, `with_year`, `with_author`, `with_both`, `without_any`; the last means neither a year nor an author), `page` (from 1), `page_size` (1 to 500, default 50), `sort_by` (`path`, `indexed_at`, `passage_count`, `year`, `title`) and `sort_order`. Ties are broken by `source_id` ascending and null values sort last in both directions. Titles sort case-insensitively.
- Passage counts are counted in the published collection at query time (ADR-0008), one exact count per source, and only for the sources on the returned page. Sorting by `passage_count` is the exception: the order depends on every count, so every matching source is counted once. That is the expensive case, one store call per match.
- A count that fails for one source gives that entry a null `passage_count` and `degraded: true`, with one warning in the log. When every count of the page fails, the store is down and the request fails with `store_unavailable`, as other operations do.
- No published index, or no committed sources, gives an empty success. A page past the end is empty with the correct `total_hits`.
- An indexed source whose passages are gone from the store is listed with `passage_count: 0`. The listing shows such drift and does not repair it.
- Aggregates (`total_sources` and `by_file_type`) cover every match, not only the page.
- `ocr_status` is `processed_via_ocr` when the source is the output of a live OCR registry entry (ADR-0017), otherwise `not_applicable`.
- Reads only the manifest, the store and the OCR registry; takes no lock and writes nothing. The registry's liveness rule checks that an output file exists, so a PDF on the page costs one file-existence check; no corpus file is opened or walked.

**Fronts.** `POST /sources/browse` (flat JSON body), the read-only MCP tool `browse_sources` (same arguments and body) and `docgrep sources` (table, or the body with `--json`). The MCP tool caps `page_size` at `DOCGREP_MCP_MAX_PAGE_SIZE` (default 100, at most 500), like the `top_n` cap. A tool result over the existing result budget leaves out the last sources of the page and says so in a `hint`, as `find_passages` does with passages. Over HTTP and MCP the listing shares the composition root of search, so a local store is never opened twice in one process.

**Deviations from the UC-08 draft.**
- No `ocr_pdf` file type: OCR outputs are PDF files and are indexed and filtered as `pdf`.
- No `delegated_to_sibling` OCR status: an original delegated to its OCR'd sibling is not indexed itself, so it is never listed; its output is listed as `processed_via_ocr`.
- The draft names the path filters in the singular; the request takes lists (`path_prefixes`, `path_globs`) like the other filter fields, OR within each list.

## Consequences

Browse and search will share one definition of each filter field. A filter costs one manifest scan in Python, which is linear in the number of committed sources and cheap for a few thousand. A default page costs at most one store count per listed source; `sort_by=passage_count` costs one per match. No new payload index and no new stored state are needed.

## Alternatives considered

- **Qdrant payload filters with keyword or text indexes:** case-insensitive substrings and segment-aware globs do not map cleanly onto them, and the results would depend on index configuration.
- **Pre-computed passage counts in the manifest:** rejected in ADR-0008; the `passages` column of the manifest is not used for browsing.
- **Cursor pagination:** offset paging is enough for libraries of this size.
