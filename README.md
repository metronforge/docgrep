# docgrep

docgrep searches a local document corpus and returns ranked, verbatim passages with source locations. The current slice indexes TXT, Markdown, EPUB, and text-bearing PDF and DjVu files. Image-only PDFs and DjVu files and EPUBs without text are recorded as needing OCR; other file types are skipped.

## Supported formats

| Format | Extensions | Location in results | Metadata | Notes |
|---|---|---|---|---|
| Plain text | `.txt` | none | none | UTF-8, else a confidently detected legacy encoding |
| Markdown | `.md`, `.markdown` | heading path | `title`, `author`, `year` from YAML frontmatter | wikilinks are recorded |
| PDF | `.pdf` | page number | title, author and creation year from the PDF metadata | pages without text and pages with a garbled text layer are skipped; a PDF without any text, or whose text layer is garbled as a whole, is recorded as needing OCR |
| EPUB (2 and 3) | `.epub` | heading path (`h1` to `h6`, continued across chapters) | title, creators and year from the package metadata | text in spine (reading) order, navigation document and NCX skipped; DRM-encrypted books fail with `parse_error` (font obfuscation is accepted); a book without text is recorded as needing OCR |
| DjVu | `.djvu`, `.djv` | page number | none | the existing hidden text layer, read with the djvulibre tools (`djvused`, `djvutxt`); same soft-hyphen handling and page policy as PDF; a file without any text is recorded as needing OCR with the reason `no_text_layer`; when the tools are missing the file is recorded as failed (`djvu_tools_missing`) and retried by `docgrep update` |

Passages are verbatim. The one exception is the PDF soft hyphen (U+00AD), which is removed from extracted PDF text; when it ends a line (optionally followed by spaces), the line break and those spaces are removed too, so the two halves of the word are joined. A visible hyphen `-` is never touched.

A PDF page is judged only when it has at least 200 non-whitespace characters, by three rules. R1, broken characters: replacement, private-use or control characters and `(cid:N)` placeholders make up more than 10% of them. R2, long tokens: tokens longer than 40 characters (other than URLs) make up more than 40%. R3, few letters and digits: letters and digits of any script make up less than 40%. R2 and R3 ignore dot leaders (four or more dots or ellipsis characters, with or without spaces between them), so a table of contents is measured by its titles and page numbers; the text itself keeps them. Only R1 marks a page as garbled: such pages never become passages, and a WARNING `pdf_pages_dropped` with `source_id`, `dropped` and `total` page counts is logged. R2 and R3 are weak signals, because tables of contents, code with long class names, keys and hashes, and regular expressions trigger them on readable pages; they never drop a page by themselves. The whole PDF is recorded as `skipped:requires_ocr` with the reason `garbled_text_layer` when R1 fires on half or more of the pages with text, or when any of the three rules fires on 80% or more of them.

DOCX files are recognised but not parsed yet; they are recorded as skipped (unsupported), and `docgrep update` indexes them once a parser for their type is registered.

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

`docgrep` exits with 0 on success and 1 on failure. `docgrep index` and `docgrep update` also exit with 1 when the run completed but one or more sources failed, and when an update is refused; the summary and the failed sources are still printed on standard output. They exit with 2 when the run was cancelled (see [Long builds](#long-builds)). Errors are written to standard error as `category: message`, followed by `(reason: ...)` when the underlying cause is known.

## Logging

Diagnostics go to standard error as one line per record: a UTC timestamp, the level, the logger name, the message and any extra fields as `key=value` pairs. Command output on standard output is not affected. `DOCGREP_LOG_LEVEL` selects the level (`DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`; default `INFO`). The server logs a warning at start when `DOCGREP_HOST` is not a loopback address, because the API has no authentication.

## Long builds

While `docgrep index` runs it logs an `index_progress` record at INFO level every 25 sources or every 60 seconds, whichever comes first, and once more when the loop over the sources ends. The fields are `run_id`, `sources_done`, `sources_total`, `passages_written`, `elapsed_seconds`, `passages_per_second`, `eta_seconds` and `failed`. `eta_seconds` is `null` until at least one source is done; afterwards it extrapolates the pace so far over the remaining sources. The records carry counters and timings only, never paths, titles or text.

### Updating the index

After the first full build, `docgrep update` brings the index up to date with the corpus without rebuilding it:

```sh
uv run docgrep update                     # apply added, modified and removed files
uv run docgrep update --no-retry-failed   # keep sources that failed instead of retrying them
```

It compares the corpus with the committed sources. A file is added when it is not recorded, removed when it is recorded but gone (or now excluded by the patterns), and modified when its content hash changed (size and modification time are only a cheap first check), when the parser of its type now declares a different version, or when it was skipped as unsupported and a parser for its type now exists. A file whose size or time changed but whose content did not only gets its recorded size and time refreshed. Sources that failed before are tried again unless `--no-retry-failed` is given. It prints the plan in two lines, then a summary line with what was processed, how many sources were skipped or failed, and how many passages were written and removed, followed by the failed sources:

```text
Update plan: 2 added, 4 modified, 2 removed, 0 retried, 5 unchanged.
Modified by reason: 2 newly supported, 2 content changed, 0 parser changed, 0 retried.
```

`modified` is the sum of three reasons: `newly supported` (skipped as unsupported before, a parser exists now), `content changed` (the content hash differs) and `parser changed` (the parser of the type declares another version than the one recorded, or no parser is registered any more). `retried` repeats the count of failed sources tried again, which are not part of `modified`. Without any change only `No changes detected (N sources unchanged).` is printed. The `index_update_plan` log record carries the same counts as `added`, `modified`, `removed`, `unchanged`, `retried`, `refreshed`, `content_changed`, `parser_changed` and `newly_supported`.

Every built-in parser declares a version (text, Markdown, PDF and EPUB all started at `1`; the PDF parser is at `3`), and builds and updates record it with each source. A parser's version is raised whenever what it extracts from the same file changes. When a parser's version changes, `docgrep update` extracts the sources of that type again, insert-first like any other modified source. Sources recorded before versions existed have no recorded version; a missing or empty version counts as `1`, so they are not processed again merely because nothing was recorded.

An update writes into the live index, one source at a time, so search keeps working throughout; a query may briefly see the old and the new version of a source. A changed source has its new passages written before its old ones are removed, and its record follows only after that, so a source that fails to update stays searchable with its old content and is picked up again by the next update. The run is recorded with kind `update`, logs `index_progress` like a build, and on completion records the time of the update in the index (`last_updated_at`); the settings of the index are never changed.

Updates are idempotent: a second `docgrep update` without corpus changes reports `No changes detected` and only records the check. An update is not resumable; one that was cancelled, aborted or killed keeps everything it applied, and running `docgrep update` again continues the work. Such an update never blocks `docgrep index`, `docgrep update` or `docgrep status`.

An update is refused (exit status 1) before anything is written:

- `resume_available`: the latest full build is unfinished. Continue it with `docgrep index --resume` or discard it with `docgrep index --restart`.
- `config_drift`: a key that decides vectors or chunking (the same keys as for resume, see below) differs from the index. An update cannot change them; run a full rebuild with `docgrep index`, or restore the settings.
- `index_incomplete`: no completed index exists yet; build one with `docgrep index`.
- `index_lock_held`: another build or update is running.

The HTTP API has no update endpoint yet.

### Cancelling a build

Cancelling is cooperative. The first Ctrl-C (or SIGTERM) during `docgrep index` asks the build to stop: it finishes the batch it is embedding, does not record the source it stopped in the middle of, marks the run `cancelled`, keeps the unpublished staging collection and the per-source journal, releases the index lock and exits with status 2. The live index and the committed sources are not touched, so search keeps working throughout. A second Ctrl-C stops the process immediately; the run then stays `running` and, because nothing holds the lock any more, is reported as interrupted.

`docgrep update` is cancelled the same way: it stops after the current batch, keeps the changes applied so far, marks the run `cancelled` and exits with 2; run it again to continue.

`docgrep cancel` does the same from another shell or for a build or update started elsewhere, such as through `docgrep serve`. It takes no lock and writes only the cancel request on the latest run, so it also works while another process owns the local Qdrant store. It says that the operation is stopping and exits with 0; a second call reports the time of the first request. If the latest run is `running` but nothing holds the lock, it reports the run as interrupted (exit 0) and changes nothing. With nothing running it says so and exits with 1. There is no force option.

### Resuming a build

A build that was cancelled, aborted by an error before the index was published, or killed (its run is still `running` but nothing holds the lock) can be continued:

```sh
uv run docgrep index --resume                      # continue the latest run
uv run docgrep index --resume --no-retry-failed    # keep sources that failed instead of retrying them
uv run docgrep index --restart                     # discard it and build from scratch
```

A plain `docgrep index` refuses while the latest run is resumable (`resume_available`, exit status 1) and names both options, so unfinished work is never thrown away by accident; `--restart` removes the retained staging collection. With a completed or absent latest run it behaves as before, and a run that had published its index but not yet committed its sources is finished first (a `swapped` run, or a `running` or `aborted` one whose staging collection is already the published index and whose journal holds every planned source; the log gets a WARNING `run_recovered`, with `reason=alias_target_reached` in the second case). A run whose staging collection is the published index but whose journal is shorter than the plan is not resumable: `--resume` rejects it as `staging_inconsistent`. `--resume` and `--restart` exclude each other. `POST /index/build` follows the same rule: while the latest run is resumable it answers 409 with category `resume_available` (the lock-held 409 keeps `index_lock_held`), and the optional request field `restart: true` (default `false`) starts from scratch and removes the retained staging collection. Resuming is not offered over HTTP.

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
- `ready`: a completed index exists and the latest run is not waiting for anything (a build running right now does not change this). Updates never change the index state: only full builds can be resumed, so these states consider the latest build.
- `aborted_pending_resume`, `cancelled_pending_resume`, `interrupted_pending_resume`: the latest run is unfinished and can be resumed. An interrupted run is one that is recorded as `running` while nothing holds the index lock; the state is derived on every read and never stored.
- `swap_pending_commit`: a run published its index but died before committing its sources; the next build or `docgrep index --resume` finishes it. This includes a run that died right between the alias swap and the write of its `swapped` status: it is still recorded as `running` (or `aborted`), but its staging collection is already the published index and its journal covers the planned sources. `latest_run.state` shows `swapped` for it, and `status` only reports it.
- `manifest_unreadable`: the manifest cannot be read.

The JSON output has these keys (`schema_version` is 1; sections that do not apply are `null`):

- `schema_version`, `index_state`.
- `latest_run`: the latest run of either kind. `run_id`, `kind` (`build`, including a resumed build, or `update`), `status` (stored: `running`, `swapped`, `completed`, `aborted` or `cancelled`), `state` (the status, or `interrupted` for a `running` run without a lock holder), `started_at`, `finished_at`, `totals` (`indexed`, `skipped`, `failed`; an update adds `added`, `modified`, `removed`, `retried` and `unchanged` source counts and `passages_written` and `passages_removed`), `total_sources`, `resumed_from` and `carried_sources` (sources taken over from the resumed run).
- `active_operation`: present while the index lock is held. `kind` (`build`, `resume` or `update`), `run_id`, `started_at`, `sources_done` and `passages_written` (from the run's journal, so they include carried-over sources), `sources_total`, `elapsed_seconds`, `eta_seconds` (`null` until computable; the pace of this process, ignoring carried-over sources) and `cancel_requested_at`.
- `vector_store`: `mode` (`local` or `server`), `status` (`reachable`, `unreachable`, `busy` or `absent` for a local store that does not exist yet) and `reason`, a short text for the degraded cases.
- `collection`: when the index can be read. `passage_count`, `embedding_model`, `embedding_dimension`, `embedding_dtype`, `docgrep_version`, `completed_at` and `last_updated_at` (when the latest `docgrep update` completed, `null` before the first).
- `drift`: a list of `field`, `indexed` and `runtime` for the settings that decide what vectors contain (see the resume rules above) and differ from the index. Only keys the index recorded and that are known without loading a model are compared.
- `sources`: what the committed sources table says, or `null` when the manifest is missing or cannot be read. `total`; `by_state` with `indexed`, `skipped` and `failed`; `skipped_by_reason` with `requires_ocr` (no text layer), `garbled_text_layer` (a PDF whose text layer is garbled), `unsupported` (no parser for the type) and `other`; `skipped_by_extension`, the 10 most common extensions among the skipped sources, a list of `extension` (`(none)` without one) and `sources`; `failed`, the first 20 failed sources by path, each with `path`, `category` (for example `parse_error`, `embedding_error` or `store_error`) and `reason` (the stored error message, shortened to 200 characters); and `parser_changed`, the number of sources that the next `docgrep update` would extract again because their parser declares another version (the same rule as update, missing versions counting as `1`). The plain-text report shows a `Sources:` line, the skipped counts per reason and the failed sources with their paths.
- `warnings`: a list of `code` and `message`, for example `store_unreachable`, `store_busy`, `manifest_unreadable`, `config_drift`, `resume_available`, `swap_pending_commit`, `update_unfinished` (the latest update was cancelled, aborted or interrupted; run `docgrep update` again), `index_incomplete`, `lock_held_without_run` and `lock_unreadable`.

## Dry run

`docgrep dry-run` previews what a build would do with the configured corpus. It reads directory entries only: it takes no index lock, writes no manifest, does not open Qdrant, loads no model and creates no file, and it uses the configured corpus root, include and exclude patterns, so it predicts what `docgrep index` would enumerate.

```sh
uv run docgrep dry-run                     # plain-text report
uv run docgrep dry-run --json              # machine-readable report with a file list
uv run docgrep dry-run --path-prefix books/ --sample 20 --seed 1
```

The report gives the total number of files, and for each file type the count and the total, average and largest size. Types that docgrep recognises but cannot parse yet (for example DOCX) are listed separately as recognised without a parser, files with other extensions as unsupported with the 20 most common extensions, and excluded files with the pattern that excluded them (`not_matched_by_includes` when no include pattern matched).

- `--path-prefix` restricts every figure to paths starting with the given relative prefix.
- `--limit` (1 to 500, default 100) and `--offset` page the file list, which is part of the JSON output only.
- `--sample N` (1 to 50, off by default) parses and chunks N randomly chosen files of the supported types, spread proportionally over the types, with the seed given by `--seed` (default 0), so the same seed picks the same files. Nothing is embedded or written. For each type it reports passages per file and per MB (1 MB is 1,000,000 bytes), parse plus chunk seconds per file and the fraction flagged as needing OCR, and then extrapolates the passage count and parse time of the whole corpus both by file count and by size. These are estimates from a sample, not measurements. For PDFs it also measures the text layer of the sampled files: pages sampled, pages classified as garbled (count and fraction), the files that a build would skip as `requires_ocr` because of a garbled text layer (count, fraction of the parsed sampled files and their paths) and the soft hyphens removed per file on average, per type and for all sampled files together; the corpus-wide number of such files is extrapolated as an estimate like the others. Sampling reads whole files, which is why it is opt-in; with the real backend it loads the tokenizer files of the embedding model, never its weights.

The report never contains file content, only counts, sizes, timings, patterns and, in the JSON file list and sample, relative paths; the plain-text report lists no files except the sampled files that would be skipped for a garbled text layer. Invalid values give a `validation_error` and exit status 1.

The JSON output has these keys (`schema_version` is 1):

- `schema_version`, `scope` (`path_prefix`), `totals` (`files`, `size_bytes`, `included`, `excluded`).
- `types`: a list sorted by type with `type`, `status` (`supported` or `recognised_without_parser`), `files`, `size_bytes`, `average_size_bytes` and `largest_size_bytes`.
- `unsupported`: `files`, `size_bytes`, `average_size_bytes`, `largest_size_bytes` and `extensions`, a list of `extension` and `files`.
- `would_be_indexed` (`files`, `size_bytes`) and `recognised_without_parser` (`files`, `size_bytes`, `types`).
- `excluded`: `files`, `size_bytes` and `by_pattern`, a list of `pattern` and `files`.
- `files`: `offset`, `limit`, `total` and `items`, a list of `path`, `size_bytes`, `extension`, `type` (null when unsupported), `status` (`indexed`, `recognised_without_parser`, `unsupported` or `excluded`) and `excluded_by`.
- `sample`: null without `--sample`, otherwise `requested`, `seed`, `note` and `types`, a list of `type`, `sampled_files`, `parsed`, `failed`, `requires_ocr_fraction`, `passages_per_file`, `passages_per_mb`, `parse_chunk_seconds_per_file`, `text_quality` and `estimate` (`note`, `corpus_files`, `corpus_size_bytes`, and `by_file_count` and `by_size`, each with `passages` and `parse_chunk_seconds`; `by_file_count` also has `garbled_text_layer_files`, the estimated number of files in the corpus that would be skipped for a garbled text layer). `sample.text_quality` totals the same figures over all sampled types.
- `text_quality` (measured on the sampled files, never text): `pages_sampled`, `pages_garbled`, `pages_garbled_fraction` (of the pages sampled), `garbled_text_layer_files` (paths of the sampled files that would be skipped as `requires_ocr` for a garbled text layer), `garbled_text_layer_fraction` (of the parsed sampled files), `soft_hyphens_removed_per_file` (average over the parsed sampled files), `pages_by_rule` (on how many sampled pages each rule fired, counted separately from the dropped pages so that the weak signals stay visible: `r1_broken_characters`, `r2_long_tokens` and `r3_low_alnum`) and `flagged_files`, a list of the sampled files on which any rule fired, each with `path`, `pages_dropped` (pages dropped for broken characters), `pages_by_rule` and `dropped_pages` (at most the first 20 dropped page numbers). The plain-text report shows the counts per rule but lists no flagged files; use `--json` for them. Fractions are null when there is nothing to divide by; types other than PDF report zero pages.

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

After pulling a release into a working copy with an editable install, refresh the installed package metadata so that it matches the code version:

```sh
export UV_NO_SYNC=1
uv pip install --no-deps -e .
```

Otherwise the package test that compares the installed version with the code version fails, although the code is fine. `--no-deps` leaves the installed dependencies, including the chosen torch build, untouched.

## License

docgrep is licensed under AGPL-3.0-or-later (see [LICENSE](LICENSE)).

Using docgrep locally for yourself or inside your organisation imposes no obligations beyond keeping the licence notices. If you distribute modified versions, or let other people use a modified version over a network, the AGPL requires offering them the corresponding source of that version. The PDF backend PyMuPDF is itself licensed under AGPL-3.0 (or a commercial licence).
