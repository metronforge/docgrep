# ADR-0008: Browse aggregation computed at query time

**Status:** Accepted
**Date:** 2026-05-10

## Context

UC-8 needs passage counts per source. Pre-computing them in the manifest was considered and reversed: it duplicates state that can drift from the index.

## Decision

Compute counts at query time with Qdrant `count` using payload indexes.

## Consequences

No second source of truth; UC-8 latency depends on payload indexes on `source_id`.
