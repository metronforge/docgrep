# ADR-0009: Default GPU models

**Status:** Accepted
**Date:** 2026-05-10

## Context

The corpus is multilingual (Russian and English at minimum). Candidates were `bge-large-en-v1.5` (English-only) and `bge-m3`; for reranking `bge-reranker-large` and `bge-reranker-v2-m3`.

## Decision

GPU defaults: embeddings `BAAI/bge-m3`, reranker `BAAI/bge-reranker-v2-m3`. CPU fallback uses small-class models, configured in Settings.

## Consequences

Multilingual retrieval out of the box. Changing the embedding model requires a full reindex (UC-11); the collection records the model to detect drift.
