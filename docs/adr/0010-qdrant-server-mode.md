# ADR-0010: Qdrant server mode for production use

**Status:** Accepted
**Date:** 2026-10-03
**Supersedes in part:** ADR-0001 (storage mode). **Amends:** ADR-0008.

## Context

ADR-0001 chose Qdrant in embedded local mode. Measurements on the v1.1 retrieval slice with deterministic fake models show search latency growing linearly with collection size: a median of 176 ms at 3,000 passages, 718 ms at 12,000 and about 1.7 s at 30,000. Qdrant itself advises against local mode beyond about 20,000 points. Payload indexes are not applied in local mode, which ADR-0008 relies on for browse counts. A local-mode store also holds a process-level lock, so a running `docgrep serve` blocks the CLI. The target corpus already exceeds 1,000 books, which is hundreds of thousands of passages.

## Decision

A Qdrant server (HTTP or gRPC) is the supported production mode, selected by `qdrant_url`. Local mode remains supported for tests and small corpora and logs a warning above about 20,000 points. Collection layout, alias swap, named dense and sparse vectors and RRF fusion from ADR-0001 are unchanged. The store stays behind the existing Protocol, so another backend can be added without touching domain or use cases.

## Consequences

A second process (container or binary) must be running; the README documents the setup. Payload indexes take effect, so ADR-0008 holds. The CLI and the server can run at the same time. Tests against a real server run in CI through a service container. The server image tag must match the client's major and minor version. Dense vectors need about 4 KB per passage (1,024 float32 dimensions); quantization and on-disk vectors become settings if the corpus grows beyond roughly one million passages. If a single-process deployment becomes a requirement again, evaluate an embedded engine behind the same Protocol.
