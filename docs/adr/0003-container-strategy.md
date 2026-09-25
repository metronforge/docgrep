# ADR-0003: Container strategy

**Status:** Accepted
**Date:** 2026-05-10

## Context

The predecessor had several divergent Dockerfiles and large images.

## Decision

One Dockerfile with `--build-arg ACCEL=cpu|gpu`. GPU build is primary, CPU is fallback. Images are published to ghcr only on release tags, manually triggered; the GPU image is built on a self-hosted runner. Dev pushes run only lint, types and tests.

## Consequences

Size targets: GPU ~4 GB (ceiling 6 GB), CPU ~1 GB (ceiling 1.5 GB). OCR stays out of the default image.
