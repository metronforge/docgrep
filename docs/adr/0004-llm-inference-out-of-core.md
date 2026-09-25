# ADR-0004: LLM inference out of core; companion in a separate repository

**Status:** Accepted
**Date:** 2026-05-10

## Context

The predecessor mixed retrieval with an Ollama-based `/answer` endpoint, coupling core to LLM runtimes. A monorepo for the companion was considered.

## Decision

Core never performs LLM inference. `/answer` moves to `docgrep-llm-bridge` (llama-cpp-python), which consumes the public HTTP API and lives in its own repository.

## Consequences

Independent lifecycle and releases, `python-semantic-release` stays single-package, heavy GPU dependencies never enter core CI. A docker-compose file wires both services for users who want the full experience.
