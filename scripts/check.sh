#!/usr/bin/env bash
set -euo pipefail

# Run in the environment chosen by an explicit "uv sync --extra <x>"; a plain
# "uv run" would re-resolve without extras and replace torch.
export UV_NO_SYNC=1

if [ ! -x .venv/bin/python ]; then
  echo "No virtual environment: run 'uv sync --extra cpu' (or '--extra cu130') first." >&2
  exit 1
fi

uv run ruff check .
uv run ruff format --check .
uv run mypy --strict src
uv run pytest -m "not gpu" -q
