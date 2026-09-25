# ADR-0005: SQLite for manifest and OCR registry

**Status:** Accepted
**Date:** 2026-05-10

## Context

Indexing needs a durable, queryable record of sources, runs and checkpoints; OCR needs a registry and a way to mark originals delegated to an OCR'd sibling. JSON-Lines was the alternative.

## Decision

One SQLite database in the data directory (WAL mode) holds the manifest (runs, sources), the OCR registry, and a separate table for OCR sibling delegation.

## Consequences

ACID checkpoints enable UC-14 resume; single-file backup; no server.
