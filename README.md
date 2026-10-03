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

`docgrep` exits with 0 on success and 1 on failure. `docgrep index` also exits with 1 when the build completed but one or more sources failed; the summary and the failed sources are still printed on standard output. It exits with 2 when the build was cancelled (see [Long builds](#long-builds)). Errors are written to standard error as `category: message`, followed by `(reason: ...)` when the underlying cause is known.

## Logging

Diagnostics go to standard error as one line per record: a UTC timestamp, the level, the logger name, the message and any extra fields as `key=value` pairs. Command output on standard output is not affected. `DOCGREP_LOG_LEVEL` selects the level (`DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`; default `INFO`). The server logs a warning at start when `DOCGREP_HOST` is not a loopback address, because the API has no authentication.

## Long builds

While `docgrep index` runs it logs an `index_progress` record at INFO level every 25 sources or every 60 seconds, whichever comes first, and once more when the loop over the sources ends. The fields are `run_id`, `sources_done`, `sources_total`, `passages_written`, `elapsed_seconds`, `passages_per_second`, `eta_seconds` and `failed`. `eta_seconds` is `null` until at least one source is done; afterwards it extrapolates the pace so far over the remaining sources. The records carry counters and timings only, never paths, titles or text.

### Cancelling a build

Cancelling is cooperative. The first Ctrl-C (or SIGTERM) during `docgrep index` asks the build to stop: it finishes the batch it is embedding, does not record the source it stopped in the middle of, marks the run `cancelled`, keeps the unpublished staging collection and the per-source journal, releases the index lock and exits with status 2. The live index and the committed sources are not touched, so search keeps working throughout. A second Ctrl-C stops the process immediately; the run then stays `running` and, because nothing holds the lock any more, is reported as interrupted.

`docgrep cancel` does the same from another shell or for a build started elsewhere, such as through `docgrep serve`. It takes no lock and writes only the cancel request on the latest run, so it also works while another process owns the local Qdrant store. It says that the operation is stopping and exits with 0; a second call reports the time of the first request. If the latest run is `running` but nothing holds the lock, it reports the run as interrupted (exit 0) and changes nothing. With nothing running it says so and exits with 1. There is no force option.

### Resuming a build

A build that was cancelled, aborted by an error before the index was published, or killed (its run is still `running` but nothing holds the lock) can be continued:

```sh
uv run docgrep index --resume                      # continue the latest run
uv run docgrep index --resume --no-retry-failed    # keep sources that failed instead of retrying them
uv run docgrep index --restart                     # discard it and build from scratch
```

A plain `docgrep index` refuses while the latest run is resumable (`resume_available`, exit status 1) and names both options, so unfinished work is never thrown away by accident; `--restart` removes the retained staging collection. With a completed or absent latest run it behaves as before, and a run that had published its index but not yet committed its sources is finished first (a `swapped` run, or a `running` or `aborted` one whose staging collection is already the published index and whose journal holds every planned source; the log gets a WARNING `run_recovered`, with `reason=alias_target_reached` in the second case). A run whose staging collection is the published index but whose journal is shorter than the plan is not resumable: `--resume` rejects it as `staging_inconsistent`. `--resume` and `--restart` exclude each other. `POST /index/build` is unchanged and always starts from scratch.

Only the latest run can be resumed, and resuming a resumed run works. The resume creates a new run that records the one it continues, adopts the retained staging collection, carries over the journal of everything still valid, and processes the rest: sources the old run did not finish, sources that failed (unless `--no-retry-failed`) and sources whose content changed. Passage ids are deterministic, so a source that was half written is simply written again. The summary reports the run it resumed, and how many sources were carried over, processed, retried and removed. Resuming a run that is already completed reports `nothing_to_resume` (exit 1); a run that had only the commit left is finished and reported.

A resume is refused, before anything is written, when it cannot be a coherent continuation:

- `config_drift`: the keys that decide what vectors and passages contain differ from the run's recorded values. These are `embedding_model`, `embedding_dimension`, `chunk_size`, `chunk_overlap`, `tokenizer` and `normalize_embeddings`. Operational settings (device, dtypes, batch sizes, log level, host, port) may differ.
- `staging_inconsistent`: the retained staging collection is missing, has another vector dimension, holds fewer passages than the run recorded as indexed, or is the published index itself. Use `--restart` (or check `docgrep status` first in the last case).
- `corpus_drift`: more than 5% of the sources the run had indexed changed or vanished. Use `--restart`. Below that threshold, changed and vanished sources have their staged passages removed, and changed ones are processed again.

### Status

```sh
uv run docgrep status          # plain text
uv run docgrep status --json   # the same report with stable key names
```

`docgrep status` is read-only: it takes no lock, loads no model, writes nothing (it does not even create a data directory or a local store that does not exist) and never fails because a component is degraded; it exits with 0 unless the command line or the settings are invalid. It reports what it can read and lists what it cannot as warnings. It works while a build runs in another process, and in local mode it reports `busy` instead of failing when another process holds the Qdrant store. A Qdrant server is pinged with a 2 second timeout.

The `index_state` is one of:

- `not_built`: no completed index exists.
- `ready`: a completed index exists and the latest run is not waiting for anything (a build running right now does not change this).
- `aborted_pending_resume`, `cancelled_pending_resume`, `interrupted_pending_resume`: the latest run is unfinished and can be resumed. An interrupted run is one that is recorded as `running` while nothing holds the index lock; the state is derived on every read and never stored.
- `swap_pending_commit`: a run published its index but died before committing its sources; the next build or `docgrep index --resume` finishes it. This includes a run that died right between the alias swap and the write of its `swapped` status: it is still recorded as `running` (or `aborted`), but its staging collection is already the published index and its journal covers the planned sources. `latest_run.state` shows `swapped` for it, and `status` only reports it.
- `manifest_unreadable`: the manifest cannot be read.

The JSON output has these keys (`schema_version` is 1; sections that do not apply are `null`):

- `schema_version`, `index_state`.
- `latest_run`: `run_id`, `status` (stored: `running`, `swapped`, `completed`, `aborted` or `cancelled`), `state` (the status, or `interrupted` for a `running` run without a lock holder), `started_at`, `finished_at`, `totals` (`indexed`, `skipped`, `failed`), `total_sources`, `resumed_from` and `carried_sources` (sources taken over from the resumed run).
- `active_operation`: present while the index lock is held. `kind` (`build` or `resume`), `run_id`, `started_at`, `sources_done` and `passages_written` (from the run's journal, so they include carried-over sources), `sources_total`, `elapsed_seconds`, `eta_seconds` (`null` until computable; the pace of this process, ignoring carried-over sources) and `cancel_requested_at`.
- `vector_store`: `mode` (`local` or `server`), `status` (`reachable`, `unreachable`, `busy` or `absent` for a local store that does not exist yet) and `reason`, a short text for the degraded cases.
- `collection`: when the index can be read. `passage_count`, `embedding_model`, `embedding_dimension`, `embedding_dtype`, `docgrep_version` and `completed_at`.
- `drift`: a list of `field`, `indexed` and `runtime` for the settings that decide what vectors contain (see the resume rules above) and differ from the index. Only keys the index recorded and that are known without loading a model are compared.
- `warnings`: a list of `code` and `message`, for example `store_unreachable`, `store_busy`, `manifest_unreadable`, `config_drift`, `resume_available`, `swap_pending_commit`, `index_incomplete`, `lock_held_without_run` and `lock_unreadable`.

## Dry run

`docgrep dry-run` previews what a build would do with the configured corpus. It reads directory entries only: it takes no index lock, writes no manifest, does not open Qdrant, loads no model and creates no file, and it uses the configured corpus root, include and exclude patterns, so it predicts what `docgrep index` would enumerate.

```sh
uv run docgrep dry-run                     # plain-text report
uv run docgrep dry-run --json              # machine-readable report with a file list
uv run docgrep dry-run --path-prefix books/ --sample 20 --seed 1
```

The report gives the total number of files, and for each file type the count and the total, average and largest size. Types that docgrep recognises but cannot parse yet (for example EPUB, DOCX and DjVu) are listed separately as recognised without a parser, files with other extensions as unsupported with the 20 most common extensions, and excluded files with the pattern that excluded them (`not_matched_by_includes` when no include pattern matched).

- `--path-prefix` restricts every figure to paths starting with the given relative prefix.
- `--limit` (1 to 500, default 100) and `--offset` page the file list, which is part of the JSON output only.
- `--sample N` (1 to 50, off by default) parses and chunks N randomly chosen files of the supported types, spread proportionally over the types, with the seed given by `--seed` (default 0), so the same seed picks the same files. Nothing is embedded or written. For each type it reports passages per file and per MB (1 MB is 1,000,000 bytes), parse plus chunk seconds per file and the fraction flagged as needing OCR, and then extrapolates the passage count and parse time of the whole corpus both by file count and by size. These are estimates from a sample, not measurements. Sampling reads whole files, which is why it is opt-in; with the real backend it loads the tokenizer files of the embedding model, never its weights.

The report never contains file content, only counts, sizes, timings, patterns and, in the JSON file list and sample, relative paths. Invalid values give a `validation_error` and exit status 1.

The JSON output has these keys (`schema_version` is 1):

- `schema_version`, `scope` (`path_prefix`), `totals` (`files`, `size_bytes`, `included`, `excluded`).
- `types`: a list sorted by type with `type`, `status` (`supported` or `recognised_without_parser`), `files`, `size_bytes`, `average_size_bytes` and `largest_size_bytes`.
- `unsupported`: `files`, `size_bytes`, `average_size_bytes`, `largest_size_bytes` and `extensions`, a list of `extension` and `files`.
- `would_be_indexed` (`files`, `size_bytes`) and `recognised_without_parser` (`files`, `size_bytes`, `types`).
- `excluded`: `files`, `size_bytes` and `by_pattern`, a list of `pattern` and `files`.
- `files`: `offset`, `limit`, `total` and `items`, a list of `path`, `size_bytes`, `extension`, `type` (null when unsupported), `status` (`indexed`, `recognised_without_parser`, `unsupported` or `excluded`) and `excluded_by`.
- `sample`: null without `--sample`, otherwise `requested`, `seed`, `note` and `types`, a list of `type`, `sampled_files`, `parsed`, `failed`, `requires_ocr_fraction`, `passages_per_file`, `passages_per_mb`, `parse_chunk_seconds_per_file` and `estimate` (`note`, `corpus_files`, `corpus_size_bytes`, and `by_file_count` and `by_size`, each with `passages` and `parse_chunk_seconds`).

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
