# ADR-0011: Torch backend selection through extras

**Status:** Accepted
**Date:** 2026-10-03
**Amends:** ADR-0003 and ADR-0009 (build matrix); refines the CUDA statement in Vision section 4.3.

## Context

The lockfile resolved the default PyPI torch, which on Linux is a CUDA 13.0 build pulling about 18 nvidia packages (roughly 3 GB). Every CI run downloads them, and the CPU image target of about 1 GB in ADR-0003 is unreachable. CUDA 13.0 wheels need a recent driver (R580 or newer), while the Vision states CUDA 12.x or newer.

## Decision

Torch is selected through mutually exclusive extras: `cpu` and `cu130`. Each maps to the matching PyTorch index through `tool.uv.sources` and explicit `tool.uv.index` entries, and the extras are declared conflicting in `tool.uv.conflicts`. CI and the CPU image use `cpu`. GPU users install one CUDA extra. The Docker build argument `ACCEL` maps to these extras. GPU tests stay deselected in CI.

A `cu128` extra, which would serve drivers limited to CUDA 12.x, was dropped: the PyTorch cu128 index offers torch only up to 2.11, while the project needs 2.14 or newer. It can be added when that changes.

The environment is chosen by an explicit `uv sync --extra <extra>`; commands then run without re-syncing, and `scripts/check.sh` runs with `UV_NO_SYNC=1`. A plain `uv run` re-resolves without extras and replaces torch with the PyPI build. This is documented, not prevented.

## Consequences

One lockfile covers every variant. CI is faster and smaller. Contributors must choose an extra explicitly; the README documents it. Supporting a new CUDA version means adding an extra. Small CPU-class models for the CPU fallback (ADR-0009) remain a separate decision.
