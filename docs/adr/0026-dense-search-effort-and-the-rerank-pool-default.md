# ADR-0026: Dense search effort and the rerank pool default

**Status:** Accepted
**Date:** 2026-10-17

## Context

After the sparse channel moved to BM25 (ADR-0025), the retrieval evaluation (ADR-0024) on the real library still lost two cases in the first stage of issue #20.

One of them was rank 6 of the dense channel when that channel was read to depth 2000, yet it was missing from the first stage. Reciprocal rank fusion cannot push a passage at dense rank 6 below the rerank pool, so the dense prefetch itself had not returned it. A direct comparison with exact search confirmed it: with Qdrant's default search parameters the dense query did not return the passage at all, while `hnsw_ef` 256 and exact search both put it at rank 6. For another query exact search placed the expected passage at rank 72; approximate search missed it even with `hnsw_ef` 512 and reported rank 177 at depth 2000.

Recall at 100 of the dense channel against exact search, as the mean over the fifteen evaluation queries, on the collection of about 870,000 passages (1024-dimensional cosine vectors, HNSW with `m` 16 and `ef_construct` 100, no quantization, optimizer idle):

| search effort | recall@100 | mean time per dense query |
|---|---|---|
| Qdrant default | 0.804 (worst query 0.46) | 21 ms |
| `hnsw_ef` 256 | 0.923 | 27 ms |
| `hnsw_ef` 512 | 0.971 | 34 ms |
| `hnsw_ef` 1024 | 0.987 | 42 ms |
| `hnsw_ef` 2048 | 0.995 | 62 ms |
| exact | 1.000 | about 150 ms |

With default parameters the first stage silently dropped about a fifth of its dense candidates on every query, not only on short ones. Rewriting every point during the re-encode rebuilt the segments and their graphs, which is why the deep dense ranks of earlier measurements shifted although no dense vector changed.

The rerank pool (ADR-0025) was measured on the same cases after the re-encode: a pool of 150 recovered a bare two-word phrase query whose passage was at fused rank 128 (final rank 23) and changed no other case; 200 added nothing. A search already spends far more time in the reranker than in either channel.

## Decision

**The dense channel searches with an explicit effort.** `DOCGREP_DENSE_HNSW_EF` is an integer `hnsw_ef` (1 to 16384, default 1024) or `exact`. The store sends it as the search parameters of the dense prefetch of the first stage and of the dense lists of the evaluation probe, raised to the limit of the request when that is larger, so a deep probe never searches with less effort than search does. `exact` sends exact search. The sparse channel has no graph and is unchanged. Qdrant's own default is no longer used for the dense channel.

**The rerank pool defaults to 150.** Unset, `DOCGREP_RERANK_POOL` is 150, or twice `top_k` when that is smaller (the union of the two channels cannot hold more). Setting it to `top_k` restores the reranking of earlier releases.

**Both are visible and measurable.** The applied configuration of a search carries `dense_hnsw_ef` whenever the store was given one, and `rerank_pool` whenever it differs from `top_k`; `GET /info` and the settings of `docgrep eval` always carry both. `docgrep eval --dense-hnsw-ef` overrides the effort for one run, and `--hnsw-recall` measures, per case, the recall at `top_k` of the dense channel against exact search, so the measurement above can be repeated after the library grows or its segments are rebuilt.

## Consequences

Search finds about 99% of the true dense candidates instead of 80%, for about 20 ms more per query, and reranks up to 150 candidates. Ranking changes by design. Nothing in the index changes and no reindex is needed. In local mode Qdrant searches exactly, so the effort has no effect there; the regression fixture of ADR-0024 therefore still matches byte for byte once the new configuration key is left out and the pool is set to `top_k`. A much larger corpus may need a larger effort or exact search; `--hnsw-recall` shows when.

## Alternatives considered

- **Exact search by default:** recall 1.0 and fully deterministic, but about 150 ms per query on this library and linear in its size. Kept as an option.
- **Rebuild the graph with a larger `m` and `ef_construct`:** better recall at a lower effort, but a rebuild of every segment, more memory, and a collection parameter that a later optimizer run can again affect. Not needed at this size; the option for a corpus many times larger.
- **A larger `top_k` instead:** grows both prefetches and the reranked set, and still cannot reach passages the dense search does not return.
