# docgrep

docgrep searches a local document corpus and returns ranked, verbatim passages with source locations. The current slice indexes TXT, Markdown, and text-bearing PDF files. Image-only PDFs are recorded as needing OCR; other file types are skipped.

## Install

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). Choose the PyTorch build explicitly when you sync:

```sh
uv sync --extra cpu     # CPU-only torch, no CUDA libraries
uv sync --extra cu130   # CUDA 13.0 build (needs a recent NVIDIA driver, R580 or newer)
```

Then run commands without re-syncing, for example after `export UV_NO_SYNC=1` (or with `uv run --no-sync ...`). A plain `uv run` or `uv sync` re-resolves without extras and replaces torch with the PyPI build, which is the CUDA build with about 3 GB of nvidia packages. `scripts/check.sh` sets `UV_NO_SYNC=1` itself and stops with a hint when no environment exists.

The default backend uses `BAAI/bge-m3` for embeddings and `BAAI/bge-reranker-v2-m3` for reranking. Model weights download on first use. The reranker runs in float16 on CUDA and in float32 on CPU; set `DOCGREP_RERANKER_DTYPE` to `auto` (default), `float16` or `float32` to override it. The embedding model follows the same rule through `DOCGREP_EMBEDDING_DTYPE`; vectors are always returned as float32, and the precision is recorded in the index metadata for information only, so an index built in float16 stays searchable in float32. For a small local trial without model downloads, set `DOCGREP_BACKEND=fake` as shown below.

## Configure and index

Set a corpus directory and a separate data directory for the SQLite manifest and, in local mode, the Qdrant index (see [Running with a Qdrant server](#running-with-a-qdrant-server) for larger corpora):

```sh
export DOCGREP_CORPUS_ROOT=/absolute/path/to/documents
export DOCGREP_DATA_DIR=/absolute/path/to/docgrep-data
export DOCGREP_BACKEND=fake
uv run docgrep index
```

Omit `DOCGREP_BACKEND=fake` to use the real models. Other settings, such as `DOCGREP_CHUNK_SIZE` and `DOCGREP_CHUNK_OVERLAP`, use the same `DOCGREP_` prefix. The data directory may be placed inside the corpus: it is skipped during enumeration, as are `.git` directories. The corpus root itself must not be inside the data directory; that configuration is rejected with a validation error.

## Include and exclude patterns

`DOCGREP_INCLUDE_GLOBS` and `DOCGREP_EXCLUDE_GLOBS` take JSON arrays of patterns that are matched against paths relative to the corpus root, using `/` as the separator. An excluded path is never indexed, even if it is included.

- `*`, `?` and `[...]` match within one path segment and never cross `/`.
- `**` as a whole segment matches zero or more segments; as the last segment it matches everything beneath a directory.
- A pattern without `/` (for example `*.tmp`) is compared with the file name at any depth.
- A pattern containing `/` (for example `notes/private/**`) is anchored at the corpus root; use a leading `**/` to match at any depth.
- A single leading `/` anchors the pattern at the corpus root even without another `/`: `/draft.md` matches only a `draft.md` in the root directory, and `/notes/private/**` is the same as `notes/private/**`.
- A trailing `/` is shorthand for a trailing `/**`: `notes/private/` matches everything beneath that directory.

## Exit codes

`docgrep` exits with 0 on success and 1 on failure. `docgrep index` also exits with 1 when the build completed but one or more sources failed; the summary and the failed sources are still printed on standard output. Errors are written to standard error as `category: message`, followed by `(reason: ...)` when the underlying cause is known.

## Logging

Diagnostics go to standard error as one line per record: a UTC timestamp, the level, the logger name, the message and any extra fields as `key=value` pairs. Command output on standard output is not affected. `DOCGREP_LOG_LEVEL` selects the level (`DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`; default `INFO`). The server logs a warning at start when `DOCGREP_HOST` is not a loopback address, because the API has no authentication.

## Search

```sh
uv run docgrep search "phrase to find" --top-n 5
uv run docgrep search "phrase to find" --json
```

Results include the exact stored passage text, source path, and a page number or Markdown heading path when available. The JSON form includes scores and query metadata.

## HTTP API

```sh
uv run docgrep serve
```

The server listens on `127.0.0.1:8000` by default. In another shell with the same configuration:

```sh
curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"phrase to find","top_n":5}'
```

`POST /index/build` triggers the same full rebuild as `docgrep index`. `GET /health` and `GET /info` provide basic service information.

When `rerank_skipped` is true, results are ordered by their stage-1 hybrid rank and `score` is derived from the dense similarity. It is not comparable with reranked scores and need not be monotonic with the returned order.

## Running with a Qdrant server

Without further configuration the index is stored in Qdrant's embedded local mode under the data directory. Local mode is intended for tests and small corpora: search time grows with the collection size, Qdrant advises against it beyond about 20,000 points, and only one process can open the store, so `docgrep index` and `docgrep search` fail while `docgrep serve` is running. docgrep logs a warning when a local collection holds more than 20,000 passages.

For larger corpora run a Qdrant server and point docgrep at it. The server image tag must match the major and minor version of `qdrant-client` (1.19):

```sh
docker run -d --name docgrep-qdrant -p 127.0.0.1:6333:6333 \
  -v docgrep-qdrant:/qdrant/storage qdrant/qdrant:v1.19.1
export DOCGREP_QDRANT_URL=http://localhost:6333
uv run docgrep index
```

Several corpora can share one server when each uses its own `DOCGREP_QDRANT_ALIAS` (default `docgrep`; letters, digits, `_` and `-`).

With `DOCGREP_QDRANT_URL` set, nothing is stored in local mode, payload indexes take effect, and the CLI and the server can run at the same time. The data directory still holds the SQLite manifest. Tests that need a server are marked `qdrant_server` and run when `DOCGREP_TEST_QDRANT_URL` is set:

```sh
DOCGREP_TEST_QDRANT_URL=http://localhost:6333 uv run pytest -m qdrant_server
```

## Tests

```sh
scripts/check.sh
```

This runs lint, formatting, strict type checking, and the CPU test suite, including the fake-backend end-to-end test. On a machine with the real models and a GPU available, run the separately marked tests with:

```sh
uv run pytest -m gpu -q
```

## License

License: Apache-2.0
