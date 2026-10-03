# docgrep

docgrep searches a local document corpus and returns ranked, verbatim passages with source locations. The current slice indexes TXT, Markdown, and text-bearing PDF files. Image-only PDFs are recorded as needing OCR; other file types are skipped.

## Install

Use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```sh
uv sync
```

The default backend uses `BAAI/bge-m3` for embeddings and `BAAI/bge-reranker-v2-m3` for reranking. Model weights download on first use. For a small local trial without model downloads, set `DOCGREP_BACKEND=fake` as shown below.

## Configure and index

Set a corpus directory and a separate directory for the local Qdrant index and SQLite manifest:

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
