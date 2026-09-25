# ADR-0007: Structural cache as text files

**Status:** Accepted
**Date:** 2026-05-10

## Context

Parsed document text is reused for context expansion (UC-12) and reindexing. Storing it as SQLite blobs was considered and reversed after review: large blobs bloat the database and slow checkpoints.

## Decision

Store the parsed text per source as `cache/sources/<source_id>.txt` in the data directory.

## Consequences

Cheap reads by offset, easy inspection and deletion; the manifest holds only references.
