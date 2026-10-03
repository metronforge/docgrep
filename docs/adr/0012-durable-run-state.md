# ADR-0012: Durable run state for resumable builds

**Status:** Accepted
**Date:** 2026-10-03
**Amends:** ADR-0001 (staging collection lifecycle) and ADR-0005 (manifest layout).

## Context

A full build writes into a staging collection whose name exists only in process memory, and the manifest keeps one row per path that is overwritten by whichever run touched it last. A crash therefore orphans the staging collection, and a rebuild in progress overwrites the source rows that describe the live index, which queries read for source metadata. The target library holds over a thousand books, so a full build takes hours and an interrupted run must be resumable. Resuming and cancelling (UC-14, UC-15) and incremental updates (UC-4) all need the manifest to describe the index that queries actually read.

## Decision

1. The runs table records the staging collection name before the first upsert, the run this run resumed (if any) and the time a cancel was requested. These columns are added by an additive migration when the manifest is opened.
2. A per-run journal table records every source outcome together with the run id and the number of passages written. The `sources` table keeps its meaning: the committed state of the live index.
3. At the end of a successful full build the `sources` table is replaced from the run's journal in a single transaction, after the alias swap succeeds and not before. Until then queries see the unchanged state.
4. At the start of a build, staging collections that carry this store's staging prefix and are neither the alias target nor a resumable run's collection are removed.
5. Passage ids are deterministic, so redoing a source is idempotent. A changed or removed source is cleaned up by deleting its points by source id, a new writer operation.

## Consequences

Manifest writes roughly double during a build, which is small next to embedding cost. Between the alias swap and the `sources` replacement there is a short window in which new passages are read with the previous source metadata; the swap goes first because a stale metadata read is harmless, while the opposite order could describe a collection that does not exist yet. Resume, cancel and incremental update are built on this state and are specified separately.
