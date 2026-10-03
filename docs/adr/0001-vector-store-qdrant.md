# ADR-0001: Vector store: Qdrant local mode with alias swap

**Status:** Accepted (storage mode superseded in part by ADR-0010)
**Date:** 2026-05-10

## Context

The predecessor used FAISS, which lacks payload filtering, native incremental upsert/delete and hybrid search; filters had to be applied in application code.

## Decision

Use Qdrant in local (embedded, on-disk) mode. Each index lives in a physical collection accessed through an alias. A full rebuild (UC-3, UC-11) writes into a new collection and switches the alias on success; the old collection is dropped after the swap. Hybrid search uses a named dense vector plus a named sparse vector with the IDF modifier, fused by RRF.

## Consequences

UC-2 filters are pushed down as payload filters. Old index stays queryable until the swap. Alias atomicity in local mode must be verified by an integration test before UC-11 relies on it.
