#!/usr/bin/env bash
set -euo pipefail

uv run ruff check .
uv run ruff format --check .
uv run mypy --strict src
uv run pytest -m "not gpu" -q
