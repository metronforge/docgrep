# ADR-0024: Retrieval evaluation through a diagnostic seam

**Status:** Accepted
**Date:** 2026-10-17

## Context

Decisions about `top_k`, the sparse signal, fusion, the reranker or the embedding model rested on single examples (issue #21). Short and exact-phrase queries lose their expected passage (issue #20): on the real library a passage that scores 0.68 when the search is restricted to its source is not among the first 1,000 fused candidates of the unrestricted query, and raising `top_k` does not help. What was missing is where the passage is lost: in the dense channel, in the sparse channel, in the fusion, or in the reranker. The public search response carries none of this, and must not change to carry it.

## Decision

**A read-only command, `docgrep eval CASES_FILE`, reads through a diagnostic seam.** It runs known-item cases (a JSON Lines file) through the real search pipeline and reports, for each case, the rank of the expected passage in every list, with aggregates. It takes no lock, writes nothing, needs no network beyond what the models already need, and runs no LLM. It has no HTTP or MCP endpoint.

**The seam is a method of the existing use case.** `FindPassages.diagnose` runs the same validation, index checks, filter scope, recall, rerank and duplicate collapse as `FindPassages.run`, and returns every candidate in the public order (not cut to `top_n`, sources not sanitized) with the first-stage position of each and its reranker score. `run` builds its result from the same internal step, so the two cannot drift. `diagnose` accepts a precomputed query vector, so all queries of a file are embedded in one call. The public search body is pinned by a test that compares `/search` bodies and use-case results with bodies recorded on `main` before the change, byte for byte (only the elapsed time and the indexing timestamp are masked).

**Per-channel ranks come from a narrow port.** `ChannelProbe.probe_channels` returns the dense list, the sparse list and their reciprocal rank fusion at a given depth (default 2000, at most 10000), with the same record-type condition and source allow-list as `retrieve`. The Qdrant store implements it with one `query_batch_points` call of up to three requests that read only the passage id and source id, and the text only when a case matches a fragment; no vectors are read. Fused ties break by dense similarity and passage id as in `retrieve`, so the fused list at depth `top_k` is the first stage itself; the deep fusion shows what a larger first stage would contain without changing it. A query without sparse terms, or an index without a sparse channel, reports a reason instead of a sparse rank, and the fusion then has the dense list alone.

**A control run per case.** For a case that names its source, the same search is repeated with a source filter restricted to the expected sources, through the filtered-search path (ADR-0020). It answers whether the passage is retrievable at all when competing sources are removed: a good control rank with no first-stage rank locates the loss in the first stage, as in issue #20.

**What a hit is.** A passage is a hit when it satisfies every expectation given: its stored path equals `source` or matches `source_glob` (the shared glob rules), and its text contains `fragment` after whitespace is collapsed in both, case-sensitively. The rank of a case in a list is the position of its best-placed hit; a source rank, the position of the first passage of a matching source, is reported beside it. A passage collapsed as a duplicate counts at the position of the passage that carries it, where search shows it.

**Deterministic output.** The same index and settings give identical numbers. `--json` has a fixed key order and no timings or timestamps.

**No ranking change is made with this decision.** Ranking, fusion, weights, `top_k` defaults, models, chunking and the index stay as they are. Changes to them, such as a phrase mode or different fusion weights, are decided after the first measurements, each with its measured effect.

## Consequences

Issue #20 can be answered with numbers per channel. A case costs one diagnosis of the pipeline, one probe (three store queries in one round trip) and, with the control run, one more diagnosis; the models are loaded once for the whole file. Real cases name private files, so they live in git-ignored `*.private.jsonl` files and only a synthetic example is in the repository.

## Alternatives considered

- **Add the diagnostics to the public search response:** changes a public contract for a developer tool, and costs every search the deeper reads.
- **A separate script outside the package:** would duplicate the pipeline and drift from it; the point is to measure the pipeline that search runs.
- **Recompute the channels inside the use case from `retrieve` with a larger `top_k`:** gives the fused list only, not the channels apart, and reranks thousands of candidates.
