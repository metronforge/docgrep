# ADR-0025: BM25 sparse weights and a rerank pool

**Status:** Accepted
**Date:** 2026-10-17

## Context

The retrieval evaluation (ADR-0024) located the losses of issue #20 in the first stage, in two shapes.

The sparse channel stored the raw count of each term and let Qdrant's IDF modifier multiply it, so a passage scored the sum of idf × count over the query terms: TF-IDF without term-frequency saturation and without length normalization. On the real library, the ten best sparse hits of a two-word phrase query each repeated one of the two words 28 to 37 times (passages about an unrelated software package that uses one of the words as an identifier), and none contained both words. A short passage that contained the whole phrase once could not compete, for any of the queries measured, although its vectors were present and self-retrieval found it. The same weighting explains why back-of-book index pages ranked high for short queries.

The fusion also dropped passages that one channel ranked well. A passage at rank 71 of the dense channel, inside its 100 candidates, fell to 142 after reciprocal rank fusion with a sparse list that hardly had it, and was cut at `top_k` 100 before the reranker saw it. Raising `top_k` recovered these cases but grows both prefetches as well as the reranked set.

Re-embedding about 870,000 passages takes many GPU hours; the passage text needed to compute sparse vectors is already stored in every point's payload.

## Decision

**BM25 document weights on the client, IDF in Qdrant.** For a term with count `tf` in a passage of `|d|` tokens, the stored weight is `tf · (k1 + 1) / (tf + k1 · (1 − b + b · |d| / avgdl))`, with `k1 = 1.2` and `b = 0.75` by default and `avgdl` the average passage length of the collection. `tf` and `|d|` come from the sparse tokenizer itself (NFKC, casefold, runs of letters and digits), not from the embedder's token count. The collection keeps its `IDF` modifier, so it is not recreated. Query terms weigh 1.0 each, once per distinct term: the query side of BM25 under that modifier. Tokenization, normalization and the index hash are unchanged, so the indices of every existing vector stay valid and only the values change.

**The weighting is index metadata.** The completion marker records `sparse_encoder` (`tf-v1`, the counts of earlier releases, or `bm25-v2`) and, for BM25, `bm25_k1`, `bm25_b` and `bm25_avgdl`; an index without the key is `tf-v1`. Search encodes the query for the recorded weighting, and update (including the indexing of OCR outputs) encodes new passages with it, with the recorded `avgdl`. One collection therefore never mixes weightings, except while a re-encode runs. An update refuses to start while a re-encode is unfinished.

**Re-encode instead of re-embed.** `docgrep reencode-sparse` computes `avgdl` in one scan of the stored texts, then scans again and rewrites only the `sparse` vector of each point with `update_vectors`, in pages; no model is loaded. It holds the index lock and is a run of kind `reencode_sparse` in the run journal, with a checkpoint (the next scroll offset and the passages done) after every page. Running it again after a cancel, a crash or a kill continues from the checkpoint with the `avgdl` and parameters recorded at its start, as long as the index is the same one (its build time) and the parameters are the same. The weighting is merged into the metadata at the end; search keeps working on mixed vectors meanwhile. Without `--yes` it prints the plan and writes nothing. With `bm25-v2` and the requested parameters already recorded it does nothing.

**A full build writes `bm25-v2` with an exact average.** A build streams its sources and does not hold their passages, so a first pass over the texts would mean parsing the corpus twice. It writes BM25 weights under a provisional `avgdl` instead, then, before the alias swap, computes the true `avgdl` over its staging collection and re-encodes every passage in it, with the same code as the command. Passages carried over by a resumed build are re-encoded too, so a resumed build equals an uninterrupted one. A local-mode store is reloaded before such a rewrite: the local client holds the points written by its own process with a normalized dense vector and persists whatever it holds when a point is rewritten, so without the reload the dense vectors would change in their last digits depending on the history of the process. A server never touches the dense vector.

**A rerank pool, default unchanged.** `DOCGREP_RERANK_POOL` sets where the fused list is cut: each prefetch keeps `top_k`, the fusion keeps up to `rerank_pool` (at most `2 · top_k`, the size of the union) and all of them are reranked. Unset, it equals `top_k`; the retriever is then called as before and the applied configuration of a search lists `rerank_pool` only when it differs from `top_k`, so a default search returns the same body byte for byte. The default stays `top_k` until measurements on the real library, with per-query latency of reranking the larger pool, decide another.

## Consequences

A short passage that holds every query term now outranks long passages that repeat one of them, in the sparse channel and through it in the fusion. Existing indexes keep working as `tf-v1` until they are re-encoded, which takes minutes per 100,000 passages on a CPU. A full build pays one more scan and rewrite of its staging collection. The recorded `avgdl` is kept by updates, so it drifts slowly as the corpus changes; a rebuild or a re-encode with other parameters computes it anew. `docgrep eval` reports the weighting and the pool it ran with, so before-and-after measurements are comparable. Rankings of `bm25-v2` indexes differ from `tf-v1` ones by design.

## Alternatives considered

- **Re-embed with a model that produces sparse vectors:** many GPU hours for a change that only needs the stored text.
- **Keep counts and drop the IDF modifier for client-side IDF:** the IDF would have to be recomputed on the client for every change of the corpus; Qdrant maintains it already.
- **Two passes over the corpus in a build:** parsing PDF and EPUB sources twice costs more than one scan of the stored texts.
- **Raise `top_k` instead of adding a pool:** grows both prefetches and the reranked set together; the pool rescues fused candidates without changing the channels.
- **Stemming, stopwords or another tokenizer:** would change the indices of every vector and need a decision of its own; possible later work, as is a phrase channel for exact-phrase queries.
