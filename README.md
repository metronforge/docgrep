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

Omit `DOCGREP_BACKEND=fake` to use the real models. Other settings, such as `DOCGREP_CHUNK_SIZE` and `DOCGREP_CHUNK_OVERLAP`, use the same `DOCGREP_` prefix. Keep the data directory outside the corpus so index files are not enumerated as documents.

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
