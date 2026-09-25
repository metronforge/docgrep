# ADR-0002: Package manager: uv

**Status:** Accepted
**Date:** 2026-05-10

## Context

The predecessor used conda, which produced oversized images and dependency drift.

## Decision

Use `uv` with a committed `uv.lock`, a single `pyproject.toml`, `src/` layout, Python 3.12.

## Consequences

Reproducible installs, fast CI. Optional extras (`docgrep[ocr]`) are declared in `pyproject.toml`.
