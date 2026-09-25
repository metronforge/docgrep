# ADR-0006: Content hashing: BLAKE3

**Status:** Accepted
**Date:** 2026-05-10

## Context

Change detection (UC-4) hashes every file; SHA-256 was the alternative.

## Decision

Use BLAKE3 for content hashes.

## Consequences

Several times faster than SHA-256 on CPU. The `blake3` wheel must be available for all target platforms in `uv.lock`.
