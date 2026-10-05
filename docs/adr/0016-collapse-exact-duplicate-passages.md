# ADR-0016: Collapse exact duplicate passages at query time

**Status:** Accepted
**Date:** 2026-10-05

## Context

A library often holds the same book as several files with different bytes, for example two downloads of the same PDF. Each file is indexed as its own source, and its passages get their own ids, which derive from the source id and the chunk offset. A query then returns the same passage once per copy, with identical text, location and scores, and the copies take result slots that distinct passages should have: on a real library, two of the top five results were the same passage from two files.

## Decision

Exact duplicates are collapsed at query time. After the final ordering of the reranked candidates (or of the first-stage order when reranking is skipped) and before the cut to `top_n`, the candidates are grouped by a key built from their text: Unicode NFC, every run of whitespace replaced by one space, leading and trailing whitespace removed. Case and punctuation are not folded. The key is used for grouping only and never returned.

The first member of a group under the existing tie-break rule (reranker score, then dense score, descending, then passage id ascending) represents it. It is returned unchanged, with its own verbatim text and scores; the other members are listed on it as duplicates with their source and location, in ranking order, so that no source is hidden. The cut to `top_n` applies to the collapsed list, so the freed slots go to the next distinct passages. The response reports how many candidates were removed; `total_hits` keeps counting the candidates before collapsing.

Collapsing is on by default (`DOCGREP_COLLAPSE_DUPLICATES`) and can be switched per request (`collapse_duplicates` in `POST /search`, `--no-collapse-duplicates` on the command line). With it off, the response is that of the previous release.

Alternatives considered and rejected:

- Collapsing at index time, by storing one passage for all copies: search results would hide the other sources, removing or changing one copy would affect the others, and an existing index would need a rebuild.
- Deduplicating by file content hash: it only catches byte-identical files, while the copies that cause the problem typically differ in bytes (metadata, another download, another producer).
- Near-duplicate detection by shingling or by embedding similarity: it would also catch different line wrapping, editions or OCR noise, but needs thresholds that can merge passages that are not the same text, and costs more per query. It is deferred as a possible later step.

## Consequences

Result slots go to distinct text, and every copy remains visible as a location on the representative. The feature is query-time only: chunking, passage ids, payloads and stored data are unchanged, and no reindex is needed. Grouping costs one pass over at most `top_k` candidates.

Only exact matches under whitespace normalization are collapsed. A second edition, a different layout of the same book, or a copy with another text layer chunks differently or differs in a character, so its passages are not collapsed and still take their own slots.
