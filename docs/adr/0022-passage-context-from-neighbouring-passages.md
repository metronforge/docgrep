# ADR-0022: Passage context from neighbouring passages

**Status:** Accepted
**Date:** 2026-10-17

## Context

A client that holds a passage from search or browse results often needs the text around it to verify or quote it (UC-12, issue #24). ADR-0007 planned a structural cache, `cache/sources/<source_id>.txt`, for this; it was never built, and building it now needs a full reindex of the library (about five hours). The index already holds every passage of every source verbatim, with its `chunk_offset`, `token_count`, page and heading path, and consecutive passages of a section overlap by `chunk_overlap` tokens.

## Decision

**v1 assembles the window from the neighbouring indexed passages of the same source.** The structural cache of ADR-0007 stays unbuilt; nothing is read from the corpus on disk, nothing is re-parsed and nothing is reindexed. HTTP `POST /passages/context`, the MCP tool `expand_passage_context` and `docgrep context` run the same use case, read-only and without the index lock.

**Window unit: tokens, whole passages.** `window_tokens` applies to each side of every requested passage (default 500, at most 5000, both configurable). It is measured with the stored `token_count` of the neighbours: neighbours are taken outward from the passage while the running total stays within `window_tokens`, and the nearest neighbour on each side is always taken when `window_tokens > 0` and one exists. `window_tokens: 0` returns the passage alone. A neighbour is never cut except to remove its overlap. With the default chunking (512 tokens, overlap 64) the default window gives about one neighbour on each side. Paragraph and page units are out of scope.

**Exact stitching with the offsets alone.** `chunk_offset` is a character offset in a stream of the parsed sections (with one separator character between sections that belongs to no passage), and each passage text is an exact slice of its section. For an earlier passage A followed by B in reading order, A keeps `A.text[: B.chunk_offset - A.chunk_offset]` when that is shorter than `A.text`. For a later passage C, C keeps `C.text[end - C.chunk_offset :]`, where `end` is the furthest end (`chunk_offset + len(text)`) of the passages before it, when they overlap. These match the chunker: within a section the next passage starts at the first character of its first token, so the kept part of A ends exactly where B begins, whitespace included. No text is normalized.

**Joins.** Where the next piece starts exactly where the previous one ended, the pieces are concatenated. Where a gap remains inside one section (same `page_number` and `heading_path`), which happens only with `chunk_overlap = 0`, they are joined with one space. Where the pieces belong to different sections (the page or the heading path changes), they are joined with a blank line. No other text is added and no omission marker is inserted, because the window is contiguous. The join before the passage ends `context_before` and the join after it starts `context_after`, so `context_before + passage.text + context_after` is the whole window.

**Per-id statuses.** Each distinct requested id gets one entry, in request order, and a failed entry never aborts the others:

- `ok`: the passage, its source (sanitized as in search results, ADR-0021), the context, the page and heading path of each neighbour used (`before_locations`, `after_locations`), `at_source_start` / `at_source_end` (no earlier / later passage of the source was left out) and `window_tokens_used`;
- `not_found`: no passage with that id is in the published index, for example after an update removed it; with no index at all, every id is `not_found`;
- `source_removed`: the passage exists but the manifest has no indexed row for its path; the entry carries the last known path;
- `source_modified`: the passage payload's `content_hash` differs from the manifest's; the entry carries both hashes and a hint to search again.

An empty list, too many ids (default 20; the MCP tool accepts at most `DOCGREP_MCP_MAX_CONTEXT_PASSAGES`, default 5), a malformed id, a window out of range or an unknown field fails the whole request as a `validation_error` before any read. An unavailable store fails it as `store_unavailable`.

**Reads.** The requested passages are fetched in one call. Per distinct source the manifest row is read once and the source's outline (passage id, offset and token count, without text or vectors) is listed once through the existing `source_id` filter; a source has at most a few thousand passages, so no payload index or migration is added. The texts of all chosen neighbours are fetched in one call per request, and the OCR registry is read at most once for sanitizing.

## Known limits

- The whitespace between two sections, and inside a section when `chunk_overlap` is 0, is not stored; it is shown as a blank line or a space.
- Two consecutive sections with the same page and heading path (for example two DOCX parts under the same heading) are joined with a space, not a blank line: the offsets cannot tell them from one section.
- A source changed by an update while a request reads it can lose a neighbour between the outline and the text fetch; the window then stops at the first missing neighbour, so it stays contiguous, and the boundary flag on that side is false.
- The window stops at what was indexed: pages dropped as garbled and text outside any passage are not part of it.

## Consequences

Context works on the existing index without a rebuild. A request costs one passage fetch, one manifest read and one outline listing per source, and one text fetch. ADR-0007 is not revoked: a structural cache can still be added later, with a reindex, for windows that need the exact whitespace or units other than tokens.

## Alternatives considered

- **Build the structural cache of ADR-0007 now:** a full reindex for a feature the index can already serve, and a second copy of every text to keep consistent with the vector store.
- **Re-read and re-parse the source from disk:** slow for large PDFs and DjVu files, needs the corpus to be mounted and unchanged, and makes a read operation depend on the parsers.
- **Mark every join between passages:** there is nothing to mark inside a section with overlap, and a marker between sections would be text that is not in the document.
