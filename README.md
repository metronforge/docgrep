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

DjVu text layers are read with the djvulibre command-line tools (`djvused` and `djvutxt`), an optional system dependency (see [ADR-0015](docs/adr/0015-djvu-via-djvulibre-subprocess.md)). Without them everything else works; DjVu files are recorded as failed with `djvu_tools_missing`, scan discovery reports them as `inspection_failed`, and the next `docgrep update` processes them once the tools are installed. On Debian-family systems:

```sh
sudo apt install djvulibre-bin
```

## How-to examples

Short recipes to copy and paste. Paths such as `~/library` and `~/docgrep-data` are examples: replace them with your own. Every command reads its settings from the environment, so each shell needs the same settings exported; the simplest way is to keep them in one file and `source` it. The reference sections further down describe every command in full.

### First run

Keep the settings in one file, for example `~/docgrep-trial.env`, with these lines:

```sh
export UV_NO_SYNC=1                                # keep the torch build chosen at install
export DOCGREP_CORPUS_ROOT="$HOME/library"          # the documents to search
export DOCGREP_DATA_DIR="$HOME/docgrep-data"       # manifest (and index in local mode)
export DOCGREP_QDRANT_URL=http://localhost:6333    # omit for local mode
export DOCGREP_BACKEND=real                        # fake: no models, for a dry trial
export DOCGREP_DEVICE=cuda                         # cpu, cuda or auto
```

`DOCGREP_QDRANT_URL` needs a running Qdrant server (see [Running with a Qdrant server](#running-with-a-qdrant-server)); without it the index lives in local mode under the data directory. Then, in every shell you use:

```sh
source ~/docgrep-trial.env
uv run docgrep index                               # first full build
uv run docgrep status                              # should report Index: ready
uv run docgrep search "a sentence you remember from a book"
```

`docgrep index` builds the whole index from scratch; once it exists, use `docgrep update` to apply changes (see [Keeping the index current](#keeping-the-index-current)).

### Searching from the command line

```sh
uv run docgrep search "phrase to find"                            # 10 results as text
uv run docgrep search "phrase to find" --top-n 5
uv run docgrep search "phrase to find" --json                     # scores and metadata
uv run docgrep search "phrase to find" --no-collapse-duplicates   # list every copy separately
```

The text output gives the location, the score and the verbatim passage; copies of the same passage in other files are listed below it (see [Duplicate passages](#duplicate-passages)):

```text
books/example.pdf:42 (score 0.874)
Placeholder passage text, exactly as stored in the index.
also in: downloads/example (1).pdf (page 42)
```

To read the JSON output compactly without extra tools, define a small filter once per shell. It prints rank, score, path, page or heading path and the first 200 characters of each passage:

```sh
compact() { python3 -c '
import json, sys
for n, r in enumerate(json.load(sys.stdin)["results"], 1):
    p = r["passage"]
    at = p["page_number"] or " > ".join(p["heading_path"]) or "-"
    print(n, round(r["score"], 3), r["source"]["path"], at)
    print("   ", p["text"][:200].replace("\n", " "))
    for d in r.get("duplicates", []):
        print("    also in:", d["path"], d["page_number"] or "-")
'; }
uv run docgrep search "phrase to find" --json | compact
```

```text
1 0.874 books/example.pdf 42
    Placeholder passage text, exactly as stored in the index.
    also in: downloads/example (1).pdf 42
2 0.731 notes/example.md Chapter > Section
    Another placeholder passage.
```

### Searching over HTTP

Start the server in one terminal, after sourcing the settings:

```sh
source ~/docgrep-trial.env
uv run docgrep serve
```

In another terminal, use the same `compact` filter on the response:

```sh
curl -s -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "phrase to find", "top_n": 5}' | compact
curl -s -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "phrase to find", "collapse_duplicates": false}' | compact
```

The first request after the server starts loads the models and takes longer (about 15 s on the reference GPU); later requests are fast (about 2 s). Each `docgrep search` invocation is a new process and pays the load time every time, so for many queries the server is quicker. In local mode only one process can open the index, so use a Qdrant server to run `docgrep search` while `docgrep serve` is running (see [HTTP API](#http-api)).

### Different kinds of queries

Search combines a sparse keyword match with a dense meaning match and then reranks the candidates, so very different queries work:

```sh
uv run docgrep search "the exact sentence as it is printed in the book"   # exact text: found best by the sparse part
uv run docgrep search "the same idea said in other words"                 # paraphrase: found by the dense part
uv run docgrep search "river survey water levels"                         # a few keywords: sparse part
uv run docgrep search "why were the water levels measured every spring?"  # a question: passages, not an answer
uv run docgrep search 'max_connections = 100'                             # code or configuration fragment
```

A query in another language works too, in any language the embedding model supports, and can find passages written in a different language, mostly through the dense part. Quote the query with single quotes when it contains `$`, `!` or backticks. Queries are limited to 1000 characters (`DOCGREP_MAX_QUERY_LEN`).

### Keeping the index current

```sh
uv run docgrep dry-run --sample 50   # before a first build: what would be indexed, with estimates
uv run docgrep update                # after adding, changing or removing files
uv run docgrep status                # index state, latest run and the sources block
uv run docgrep cancel                # from another shell: stop a running build or update
```

The sources block of `docgrep status` counts the recorded sources and lists the failed ones:

```text
Sources: 120 total, 112 indexed, 6 skipped, 2 failed
Skipped by reason: requires_ocr 4, garbled_text_layer 1, unsupported 1, other 0
Failed sources:
  books/example.djvu: parse_error: DjVu cannot be read (djvu_tools_missing): ...
```

See [Dry run](#dry-run), [Updating the index](#updating-the-index), [Status](#status) and [Cancelling a build](#cancelling-a-build) for the details.

### Finding files that need OCR

```sh
uv run docgrep discover-scans                          # list and counts as text
uv run docgrep discover-scans --file-type pdf          # PDFs only (djvu for DjVu only)
uv run docgrep discover-scans --file-type djvu --json  # machine-readable report
```

Files classified `scanned` have no usable text layer and need OCR (reason `low_text_density`, or `garbled_text_layer` when the layer is broken); `mixed_or_uncertain` files are worth a look by hand; `inspection_failed` files could not be read, and their reason says why (for example `djvu_tools_missing`). Native files are only counted. OCR itself is not part of this release. See [Scan discovery](#scan-discovery) for the thresholds and the JSON keys.

### Troubleshooting

- `docgrep status` reports `not_built` although you built an index: the settings are not exported in this shell; `source` the settings file again.
- The vector store is `unreachable`: the Qdrant server is not running, or `DOCGREP_QDRANT_URL` points to the wrong address.
- `docgrep search` fails with `store_unavailable` while `docgrep serve` runs: in local mode only one process can open the index; stop the server, search over HTTP, or use a Qdrant server.
- The first query is slow: the models are loaded on first use; later requests to a running server are fast.
- DjVu files are skipped or fail with `djvu_tools_missing`: install `djvulibre-bin`, then run `uv run docgrep update`.

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

## Scan discovery

`docgrep discover-scans` lists the PDF and DjVu files whose text layer is missing or unusable, so that you can decide which ones need OCR. It is read-only like the dry run: it takes no index lock, opens neither the manifest nor Qdrant, loads no model, writes nothing and can run before any build. It uses the build's corpus root and include and exclude patterns.

```sh
uv run docgrep discover-scans                          # plain-text list and counts
uv run docgrep discover-scans --json                   # machine-readable report
uv run docgrep discover-scans --path-glob 'books/**' --file-type djvu
```

- `--path-glob` keeps the files whose relative path matches the pattern (the same glob rules as the include and exclude patterns).
- `--file-type` (`pdf` or `djvu`, repeatable) restricts the types; the default is both.

For each file a few pages are sampled: the first 3, the last one and 2 from the middle, capped at the page count and without duplicates. The middle pages are chosen from SHA-256 of the relative path and the page count, so the same file is always sampled the same way and two runs on an unchanged corpus give the same report. The text density is the mean number of non-whitespace characters per sampled page, where a page that the page policy would drop for broken characters (R1) counts as zero. Below `DOCGREP_SCAN_DENSITY_LOWER` (default 100) a file is `scanned`, above `DOCGREP_SCAN_DENSITY_UPPER` (default 500) it is `native`, in between (bounds included) `mixed_or_uncertain`. An ordinary book page carries well over 1,000 characters, so a book stays native even with blank pages or short chapter openings in its sample, while a scan whose text layer is empty, or holds only a page number or a title page, stays far below 100. A file whose sampled pages are at least half broken (R1) is `scanned` with the reason `garbled_text_layer`, because OCR has to redo such a layer. Native files are only counted; the others are listed. A file that cannot be inspected is listed as `inspection_failed` and the scan continues.

Exit status 0 on success and 2 for usage or configuration errors (an unknown option or file type, invalid thresholds, an inaccessible corpus root).

The JSON report has these keys (`schema_version` is 1):

- `schema_version`, `filters` (`path_glob`, `file_types`) and `thresholds` (`density_lower`, `density_upper`).
- `files`: the scanned, uncertain and failed files, sorted by file type and path, each with `path` (relative to the corpus root), `file_type` (`pdf` or `djvu`), `size_bytes`, `page_count` (null when unknown), `classification` (`scanned`, `mixed_or_uncertain` or `inspection_failed`), `reason` (`low_text_density`, `garbled_text_layer` or `uncertain_text_density`; for failures `encrypted`, `corrupt_header`, `djvu_invalid`, `djvu_tools_missing`, `djvu_timeout`, `djvu_output_too_large` or `unsamplable`), `hint` (how to install the DjVu tools when they are missing, else null), `ocr_status` (always `not_processed` until an OCR registry exists), `large_file` (more than 5000 pages or 1 GiB), `text_density` (null for failures) and `sampled_pages`.
- `totals` and `by_file_type` (per requested type): `inspected`, `native`, `scanned`, `uncertain` and `inspection_failed`.

## Search

```sh
uv run docgrep search "phrase to find" --top-n 5
uv run docgrep search "phrase to find" --json
```

Results include the exact stored passage text, source path, and a page number or Markdown heading path when available. The JSON form includes scores and query metadata.

### Duplicate passages

The same book often exists as several files with different bytes, for example two downloads of the same PDF. Their passages have different ids, but the same text would take several result slots. By default docgrep therefore collapses passages whose text is identical into one result:

- The grouping key is the passage text after Unicode NFC normalization, with every run of whitespace replaced by one space and leading and trailing whitespace removed. Case and punctuation are not folded. Only exact matches under this rule are collapsed: near duplicates, such as a second edition, an OCR'd copy or a layout that chunks differently, stay separate results.
- Grouping happens after the final ordering of all reranked candidates (or of the first-stage order when reranking is skipped) and before the cut to `top_n`. The first member of a group under the tie-break rule (reranker score, then dense score, descending, then passage id ascending) represents it; the slots of the others are filled by the next distinct passages.
- The representative is returned unchanged, with its own verbatim text and its own scores. The other members are listed on it in `duplicates`, in ranking order, each with `passage_id`, `source_id`, `path`, `file_type`, `page_number` and `heading_path`. A result without duplicates has no `duplicates` key.
- The response gains `collapsed_duplicates`, the number of candidates removed from the ranking as duplicates of another one (over the whole candidate pool, including groups below the cut). `total_hits` still counts the candidates before collapsing.

The plain-text output lists the other copies below a result, one line each:

```text
books/comet.pdf:12 (score 0.912)
...passage text...
also in: downloads/comet (1).pdf (page 12)
```

Collapsing is controlled by `DOCGREP_COLLAPSE_DUPLICATES` (default `true`), by the request field `collapse_duplicates` of `POST /search` (`true`, `false`, or `null` for the setting) and by `docgrep search --no-collapse-duplicates`. With collapsing off, results and response are exactly those of release 1.10.0: no `duplicates` and no `collapsed_duplicates` keys. Collapsing works at query time on the stored passages; it changes nothing in the index and needs no rebuild.

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

The body of `POST /search` takes `query`, an optional `top_n` and an optional `collapse_duplicates` (see [Duplicate passages](#duplicate-passages)).

`POST /index/build` triggers the same full rebuild as `docgrep index`. `POST /scans/discover` runs the read-only scan discovery with an optional JSON body `{"path_glob": ..., "file_types": [...]}` and returns the same report as `docgrep discover-scans --json` (see [Scan discovery](#scan-discovery)). `GET /health` and `GET /info` provide basic service information.

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

With `DOCGREP_QDRANT_URL` set, nothing is stored in local mode, payload indexes take effect, and the CLI and the server can run at the same time. The data directory still holds the SQLite manifest. Tests that need a server are marked `qdrant_server` and run when `DOCGREP_TEST_QDRANT_URL` is set.

**Warning: the server-mode tests create, replace and delete collections and aliases. Never point `DOCGREP_TEST_QDRANT_URL` at a server that holds a real index; give the tests a throwaway server of their own.** Before the first server-mode test runs, the suite lists the server and stops without deleting anything if it finds a collection or alias it does not own (this includes the default `docgrep` alias). Run them against a temporary container on another port and remove it afterwards:

```sh
docker run -d --rm --name docgrep-test-qdrant-$$ -p 127.0.0.1:6335:6333 qdrant/qdrant:v1.19.1
DOCGREP_TEST_QDRANT_URL=http://localhost:6335 uv run pytest -m qdrant_server
docker stop docgrep-test-qdrant-$$
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
