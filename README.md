# docgrep

docgrep searches a local document corpus and returns ranked, verbatim passages with source locations. The current slice indexes TXT, Markdown, EPUB, DOCX, and text-bearing PDF and DjVu files. Image-only PDFs and DjVu files, and EPUB and DOCX files without text, are recorded as needing OCR; other file types are skipped.

## Supported formats

| Format | Extensions | Location in results | Metadata | Notes |
|---|---|---|---|---|
| Plain text | `.txt` | none | none | UTF-8, else a confidently detected legacy encoding |
| Markdown | `.md`, `.markdown` | heading path | `title`, `author`, `year` from YAML frontmatter | wikilinks are recorded |
| PDF | `.pdf` | page number | title, author and creation year from the PDF metadata | pages without text and pages with a garbled text layer are skipped; a PDF without any text, or whose text layer is garbled as a whole, is recorded as needing OCR |
| EPUB (2 and 3) | `.epub` | heading path (`h1` to `h6`, continued across chapters) | title, creators and year from the package metadata | text in spine (reading) order, navigation document and NCX skipped; DRM-encrypted books fail with `parse_error` (font obfuscation is accepted); a book without text is recorded as needing OCR |
| DOCX | `.docx` | heading path (outline levels of `Title` and `Heading 1` to `Heading 9`, localized heading styles included) | title and author from the core properties | body paragraphs and tables in document order, one line per table row with cells joined by ` \| `; merged cells read once, nested tables flattened; headers, footers, footnotes, endnotes, comments, text boxes, tracked deletions and images are not read; password-protected files fail with `parse_error` (`encrypted_content`); a document without text is recorded as needing OCR; when python-docx is missing the file is recorded as failed (`docx_library_missing`) and retried by `docgrep update` |
| DjVu | `.djvu`, `.djv` | page number | none | the existing hidden text layer, read with the djvulibre tools (`djvused`, `djvutxt`); same soft-hyphen handling and page policy as PDF; a file without any text is recorded as needing OCR with the reason `no_text_layer`; when the tools are missing the file is recorded as failed (`djvu_tools_missing`) and retried by `docgrep update` |

Passages are verbatim. The one exception is the PDF soft hyphen (U+00AD), which is removed from extracted PDF text; when it ends a line (optionally followed by spaces), the line break and those spaces are removed too, so the two halves of the word are joined. A visible hyphen `-` is never touched.

A PDF page is judged only when it has at least 200 non-whitespace characters, by three rules. R1, broken characters: replacement, private-use or control characters and `(cid:N)` placeholders make up more than 10% of them. R2, long tokens: tokens longer than 40 characters (other than URLs) make up more than 40%. R3, few letters and digits: letters and digits of any script make up less than 40%. R2 and R3 ignore dot leaders (four or more dots or ellipsis characters, with or without spaces between them), so a table of contents is measured by its titles and page numbers; the text itself keeps them. Only R1 marks a page as garbled: such pages never become passages, and a WARNING `pdf_pages_dropped` with `source_id`, `dropped` and `total` page counts is logged. R2 and R3 are weak signals, because tables of contents, code with long class names, keys and hashes, and regular expressions trigger them on readable pages; they never drop a page by themselves. The whole PDF is recorded as `skipped:requires_ocr` with the reason `garbled_text_layer` when R1 fires on half or more of the pages with text, or when any of the three rules fires on 80% or more of them.

DOCX has no reliable page numbers, so DOCX results carry a section locator (the heading path) and their `page_number` is null, as for EPUB. Other Word-family files (`.doc`, `.docm`, `.dotx`, `.dotm`) as well as `.rtf` and `.odt` stay unsupported and are recorded as skipped (unsupported). DOCX files recorded as skipped (unsupported) by an earlier version are indexed by the next `docgrep update`. The design is recorded in [ADR-0018](docs/adr/0018-docx-via-python-docx.md).

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
export DOCGREP_DATA_DIR="$HOME/docgrep-data"       # roots file, manifest (and index in local mode)
export DOCGREP_QDRANT_URL=http://localhost:6333    # omit for local mode
export DOCGREP_BACKEND=real                        # fake: no models, for a dry trial
export DOCGREP_DEVICE=cuda                         # cpu, cuda or auto
```

`DOCGREP_QDRANT_URL` needs a running Qdrant server (see [Running with a Qdrant server](#running-with-a-qdrant-server)); without it the index lives in local mode under the data directory. Then, in every shell you use:

```sh
source ~/docgrep-trial.env
uv run docgrep roots add library "$HOME/library"   # the documents to search (once)
uv run docgrep index                               # first full build
uv run docgrep status                              # should report Index: ready
uv run docgrep search "a sentence you remember from a book"
```

`docgrep index` builds the whole index from scratch; once it exists, use `docgrep update` to apply changes (see [Keeping the index current](#keeping-the-index-current)). A corpus can have several folders; see [Corpus roots](#corpus-roots).

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
uv run docgrep update --plan         # what an update would change; writes nothing
uv run docgrep update                # after adding, changing or removing files
uv run docgrep diff /backup/library --root books   # a folder against the index
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

A file that was moved, renamed or copied, or a backup folder added as a root, is indexed by copying the passages already stored for its content, without parsing or embedding (see [Moved, renamed and copied files](#moved-renamed-and-copied-files)).

See [Dry run](#dry-run), [Updating the index](#updating-the-index), [Seeing the plan first](#seeing-the-plan-first), [Comparing a folder with the index](#comparing-a-folder-with-the-index), [Status](#status) and [Cancelling a build](#cancelling-a-build) for the details.

### Finding files that need OCR

```sh
uv run docgrep discover-scans                          # list and counts as text
uv run docgrep discover-scans --file-type pdf          # PDFs only (djvu for DjVu only)
uv run docgrep discover-scans --file-type djvu --json  # machine-readable report
```

Files classified `scanned` have no usable text layer and need OCR (reason `low_text_density`, or `garbled_text_layer` when the layer is broken); `mixed_or_uncertain` files are worth a look by hand; `inspection_failed` files could not be read, and their reason says why (for example `djvu_tools_missing`). Native files are only counted. See [Scan discovery](#scan-discovery) for the thresholds and the JSON keys, and [OCR](#ocr) to make these files searchable:

```sh
uv run docgrep ocr --all-discovered --plan   # what an OCR batch would do
uv run docgrep ocr --all-discovered          # run it, then: uv run docgrep update
```

### Troubleshooting

- `docgrep status` reports `not_built` although you built an index: the settings are not exported in this shell; `source` the settings file again.
- The vector store is `unreachable`: the Qdrant server is not running, or `DOCGREP_QDRANT_URL` points to the wrong address.
- `docgrep search` fails with `store_unavailable` while `docgrep serve` runs: in local mode only one process can open the index; stop the server, search over HTTP, or use a Qdrant server.
- The first query is slow: the models are loaded on first use; later requests to a running server are fast.
- DjVu files are skipped or fail with `djvu_tools_missing`: install `djvulibre-bin`, then run `uv run docgrep update`.

## Configure and index

Set a data directory for the roots file, the SQLite manifest and, in local mode, the Qdrant index (see [Running with a Qdrant server](#running-with-a-qdrant-server) for larger corpora), save one or more corpus folders as roots, and build:

```sh
export DOCGREP_DATA_DIR=/absolute/path/to/docgrep-data
export DOCGREP_BACKEND=fake
uv run docgrep roots add documents /absolute/path/to/documents
uv run docgrep index
```

Omit `DOCGREP_BACKEND=fake` to use the real models. Other settings, such as `DOCGREP_CHUNK_SIZE` and `DOCGREP_CHUNK_OVERLAP`, use the same `DOCGREP_` prefix. The data directory may be placed inside a root: it is skipped during enumeration, as are `.git` directories. A root must not be inside the data directory; that configuration is rejected with a validation error.

## Corpus roots

The corpus is one or more folders, each saved as a named root. The roots are stored in `roots.json` in the data directory and managed with `docgrep roots`; there is no environment variable for the list. The design is recorded in [ADR-0023](docs/adr/0023-corpus-roots-and-stored-identity.md).

```sh
uv run docgrep roots add books /srv/library          # save a root
uv run docgrep roots add notes ~/notes               # and another one
uv run docgrep roots list                            # name, path, flags, indexed sources
uv run docgrep roots list --json
uv run docgrep roots remove notes                    # only edits roots.json
```

The file has the shape `{"version": 1, "roots": [{"name": "books", "path": "/srv/library", "legacy": true}, {"name": "notes", "path": "/home/me/notes"}]}`; paths are stored absolute and resolved, and the file is written atomically (a temporary file, then a rename), so a failed write leaves the old list intact. `docgrep roots list --json` prints `origin` (`file`, `bootstrap` for the deprecated variable below, or `none`), `roots` (each with `name`, `path`, `legacy`, `readable` and `indexed_sources`) and `warnings`.

**Names and stored paths.** Every source is stored, and identified by its `source_id`, as its path relative to its root behind the root's prefix: `<name>/` for an ordinary root (`notes/2024/ideas.md`), nothing for the legacy root. The legacy root is the folder of an index built before roots had names: its sources keep the stored paths, source ids and passage ids they always had, so an existing index needs no rebuild. At most one root is legacy (`docgrep roots add NAME PATH --legacy`); a new installation needs none. Two roots can therefore hold the same relative path: `a.txt` of the legacy root and `notes/a.txt` are two sources. The root of a source is derived from its stored path, and search, browse and status show it as `root`.

The rules, checked by `roots add` and again whenever the file is read:

- A name starts with a letter or a digit and continues with letters, digits, `.`, `_` and `-`; names are unique without regard to case.
- The path must exist and be a directory when the root is added.
- No root may be inside another root, or contain one: a sub-folder of an existing root is a path prefix filter (`--path-prefix`), not a new root.
- Only one root can be the legacy root.
- An ordinary root may not be named like a top-level entry of the legacy root (a root `notes` next to a legacy folder `notes/`), because its source ids would collide with that entry's.

**One-off roots.** `index`, `update`, `dry-run`, `discover-scans` and `ocr` accept `--root NAME=PATH`, repeatable (a path alone is a root named after its folder). With any `--root`, only the given roots are used for that run and the saved list is ignored for it; a name that is saved must point to the same folder (or it is an error) and keeps its legacy flag. The run writes nothing to `roots.json`. What it indexed stays in the index, but later updates do not visit an unsaved root. So after a successful `index` or `update` with roots that are not saved, docgrep asks on a terminal:

```text
Add root notes (/home/me/notes) to the saved list so future updates include it? [y/N]
```

Without a terminal it prints the `docgrep roots add NAME PATH` command to run instead. `--save-roots` saves without asking and `--no-save-roots` neither asks nor prints the command. A saved one-off root is never the legacy root. The JSON reports of `dry-run`, `discover-scans` and `ocr` run with `--root` carry `unsaved_roots`, a list of `name` and `path`.

Roots cannot be added or chosen through MCP or the HTTP API, so an AI client never decides which folders are indexed.

**Missing and removed roots.** A root that is missing, not a directory or not readable makes `index`, `update` and `ocr` fail before any work, naming the root; it is never taken for an empty folder. Search, browse, context and status read only the index and keep working. `update` removes a source only when its root is part of the run, readable, and the file is gone. The sources of a root that is not part of the run (removed from the list, or not passed with `--root`) are kept: the update plan says how many it kept per root, and `docgrep status` lists them under their root with `configured: false`. A full `docgrep index` replaces the whole index with the sources of its roots, so it refuses while the index holds sources of another root. To remove such sources, use:

```sh
uv run docgrep forget-root notes          # shows how many sources and passages it would remove
uv run docgrep forget-root notes --yes    # removes them from the index; the files stay
```

`forget-root` removes the sources through the same path as an update that finds their files gone. It refuses the legacy root, whose sources have no prefix to tell them apart. Without `--yes` it changes nothing and exits with 1.

**Upgrading from `DOCGREP_CORPUS_ROOT`.** The old variable is deprecated. Without a roots file it still works as a single legacy root named after its folder (`default` when the folder name is not a valid name), and every command prints one line saying so, with the exact command that saves it, for example `warning: DOCGREP_CORPUS_ROOT is deprecated; save it as a root with: docgrep roots add library /srv/library --legacy`. Nothing is written for you. When a roots file exists, the variable is ignored, with a warning. With neither, commands that index fail with `no corpus roots configured`, while search, browse, context and status still work. To move an existing installation to several roots, without a rebuild:

```sh
uv run docgrep roots add books /srv/library --legacy   # the folder DOCGREP_CORPUS_ROOT named
uv run docgrep roots add notes ~/notes                 # a new root
unset DOCGREP_CORPUS_ROOT                              # and remove it from your settings file
uv run docgrep dry-run                                 # totals and per-root counts
uv run docgrep update                                  # indexes the new root only
```

If a roots file has no legacy root but the index holds unprefixed sources, they are reported as the unconfigured root `_legacy` and nothing is removed.

**Restricting to roots.** Search and browse take a `roots` filter, a list of root names combined with OR (`--in-root NAME` on the command line, repeatable, because `--root` names folders to index). An unknown name is a `validation_error` that lists the known roots, configured or found in the index. The existing path filters see stored paths, so `--path-prefix notes/` selects the same sources as `--in-root notes`.

## Include and exclude patterns

`DOCGREP_INCLUDE_GLOBS` and `DOCGREP_EXCLUDE_GLOBS` take JSON arrays of patterns that are matched against paths relative to each corpus root (without the root's prefix), using `/` as the separator. An excluded path is never indexed, even if it is included.

- `*`, `?` and `[...]` match within one path segment and never cross `/`.
- `**` as a whole segment matches zero or more segments; as the last segment it matches everything beneath a directory.
- A pattern without `/` (for example `*.tmp`) is compared with the file name at any depth.
- A pattern containing `/` (for example `notes/private/**`) is anchored at the corpus root; use a leading `**/` to match at any depth.
- A single leading `/` anchors the pattern at the corpus root even without another `/`: `/draft.md` matches only a `draft.md` in the root directory, and `/notes/private/**` is the same as `notes/private/**`.
- A trailing `/` is shorthand for a trailing `/**`: `notes/private/` matches everything beneath that directory.

## Exit codes

`docgrep` exits with 0 on success and 1 on failure. `docgrep index` and `docgrep update` also exit with 1 when the run completed but one or more sources failed, and when an update is refused (the removal guard included); the summary and the failed sources are still printed on standard output. They exit with 2 when the run was cancelled (see [Long builds](#long-builds)). `docgrep ocr` differs: 0 when no file failed, 1 when a file failed or the batch was cancelled or aborted, 2 for usage errors (see [OCR](#ocr)). `docgrep sources` exits with 0 also when nothing matches and with 2 for invalid arguments or settings. `docgrep context` exits with 0 also when some or all ids have a failure status (the statuses are printed), with 1 on a runtime failure and with 2 for invalid arguments or settings. `docgrep update --plan` and `docgrep diff` exit with 0, with 1 without an index and with 2 for usage errors (for `diff` also an unknown or ambiguous root). `docgrep eval` exits with 0 when the run completed, also when cases are not found, with 1 on a runtime failure (no index, store or model unavailable) and with 2 for invalid arguments or settings and for a malformed cases file. Errors are written to standard error as `category: message`, followed by `(reason: ...)` when the underlying cause is known.

## Logging

Diagnostics go to standard error as one line per record: a UTC timestamp, the level, the logger name, the message and any extra fields as `key=value` pairs. Command output on standard output is not affected. `DOCGREP_LOG_LEVEL` selects the level (`DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`; default `INFO`). The server logs a warning at start when `DOCGREP_HOST` is not a loopback address, because the API has no authentication.

## Long builds

While `docgrep index` runs it logs an `index_progress` record at INFO level every 25 sources or every 60 seconds, whichever comes first, and once more when the loop over the sources ends. The fields are `run_id`, `sources_done`, `sources_total`, `passages_written`, `elapsed_seconds`, `passages_per_second`, `eta_seconds` and `failed`. `eta_seconds` is `null` until at least one source is done; afterwards it extrapolates the pace so far over the remaining sources. The records carry counters and timings only, never paths, titles or text.

### The progress line

In a terminal, `docgrep index`, `docgrep update`, `docgrep ocr` and `docgrep reencode-sparse` draw one status line on standard error and redraw it in place, at most ten times a second:

```text
Indexing  37%  612/1,650 sources  210,400 passages  95 p/s  ETA 1h 12m  Rust.Atomics.and.Locks.pdf
```

It shows the percentage, the sources done of the total, the passages written, the rate in passages per second, the estimated time left and the name of the file being worked on (the last component of its path, shortened in the middle to fit the terminal width). `docgrep ocr` counts pages when every file has a known page count, and files otherwise; `docgrep reencode-sparse` shows its scan for the average length (`Scanning`) and its writing (`Re-encoding`) one after the other, and the ETA of the writing does not include the scan. When a phase ends its last line stays on screen. A log record or the message of a cancel request first clears the line.

`--progress=auto|bar|log|off` chooses what is shown (default `auto`):

- `auto` draws the line only when standard error is a terminal, and never with `--json`;
- `bar` always draws it;
- `log` never draws it and leaves the `index_progress` log records as they are;
- `off` never draws it and lowers the `index_progress` records to DEBUG.

Scripts and logs are unchanged: without a terminal (a pipe, `tee`, cron, CI) the output of every command is exactly what it was, and the log records and the JSON output are the same with or without the line. The line goes to standard error only, so a redirected standard output is unaffected; `docgrep update 2>&1 | cat` prints the usual output.

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

`modified` is the sum of three reasons: `newly supported` (skipped as unsupported before, a parser exists now), `content changed` (the content hash differs) and `parser changed` (the parser of the type declares another version than the one recorded, or no parser is registered any more). `retried` repeats the count of failed sources tried again, which are not part of `modified`. Two more reasons appear in that line only when they occur: `delegated to OCR` (an original now has a live OCR output; its passages are removed and it is recorded as skipped with `delegated_to_ocr`, while the output is picked up as an added source) and `OCR delegation ended` (the output was deleted or the original changed; the original is indexed again). The log record carries them as `delegated_to_ocr` and `delegation_ended`. Without any change only `No changes detected (N sources unchanged).` is printed. Files indexed by copying the stored passages of the same content end the first line with `N moved` and `N copied`, and the summary line names them too; they are not counted in `added`, `removed` or `modified` (see [Moved, renamed and copied files](#moved-renamed-and-copied-files)). The `index_update_plan` log record carries the same counts as `added`, `modified`, `removed`, `unchanged`, `retried`, `refreshed`, `moved`, `copied`, `content_changed`, `parser_changed` and `newly_supported`.

Every built-in parser declares a version (text, Markdown, PDF, EPUB and DOCX all started at `1`; the PDF parser is at `3`), and builds and updates record it with each source. A parser's version is raised whenever what it extracts from the same file changes. When a parser's version changes, `docgrep update` extracts the sources of that type again, insert-first like any other modified source. Sources recorded before versions existed have no recorded version; a missing or empty version counts as `1`, so they are not processed again merely because nothing was recorded.

An update writes into the live index, one source at a time, so search keeps working throughout; a query may briefly see the old and the new version of a source. A changed source has its new passages written before its old ones are removed, and its record follows only after that, so a source that fails to update stays searchable with its old content and is picked up again by the next update. The run is recorded with kind `update`, logs `index_progress` like a build, and on completion records the time of the update in the index (`last_updated_at`); the settings of the index are never changed.

Updates are idempotent: a second `docgrep update` without corpus changes reports `No changes detected` and only records the check. An update is not resumable; one that was cancelled, aborted or killed keeps everything it applied, and running `docgrep update` again continues the work. Such an update never blocks `docgrep index`, `docgrep update` or `docgrep status`.

An update is refused (exit status 1) before anything is written:

- `resume_available`: the latest full build is unfinished. Continue it with `docgrep index --resume` or discard it with `docgrep index --restart`.
- `config_drift`: a key that decides vectors or chunking (the same keys as for resume, see below) differs from the index. An update cannot change them; run a full rebuild with `docgrep index`, or restore the settings.
- `index_incomplete`: no completed index exists yet; build one with `docgrep index`.
- `index_lock_held`: another build or update is running.
- `removal_guard`: the update would remove too much of a root (see [The removal guard](#the-removal-guard)).

The HTTP API has no update endpoint yet.

### Seeing the plan first

`docgrep update --plan` finds the changes exactly as `docgrep update` would, prints them per root and exits with 0. It takes no lock, starts no run and writes nothing (manifest, index, run journal and OCR registry stay as they are), so it also works while another build or update runs; the corpus or the index can of course change before the update itself runs. It honours `--root` and `--no-retry-failed` like `update`. Files whose size and modification time are unchanged are not read; the others, and every added file, are hashed.

```sh
uv run docgrep update --plan                  # counts per root
uv run docgrep update --plan --paths          # and the stored paths of each group
uv run docgrep update --plan --json --paths   # one JSON object
```

```text
Update plan. Nothing was changed. The corpus or the index can change before `docgrep update` runs.
Root books: 1 added, 1 modified, 1 removed, 0 retried, 1 refreshed, 29 unchanged, 1 moved, 1 copied.
  Modified by reason: 1 content changed.
  added:
    vol30.txt
  modified:
    vol01.txt (content changed)
  removed:
    vol04.txt
  refreshed:
    vol02.txt
  moved:
    vol03.txt -> archive/vol03.txt
  copied:
    vol05.txt -> backup/vol05.txt
Root notes: 0 added, 0 modified, 2 removed, 0 retried, 0 refreshed, 0 unchanged, 0 moved.
  removed:
    notes/kite.txt
    notes/river.txt
Total: 1 added, 1 modified, 3 removed, 0 retried, 1 refreshed, 29 unchanged, 1 moved, 1 copied.
  Modified by reason: 1 content changed.
Removal guard: root books would lose 2 of 32 sources (at most 1 allowed); `docgrep update` would stop without changing anything unless run with --yes.
```

The groups are those of `update`: `added`, `modified` (with its reasons), `removed`, `retried`, `refreshed` (same content, new size or time: only the recorded values are updated, nothing is parsed or embedded; such a source also counts as `unchanged`), `unchanged` and the roots outside the run, and `moved` and `copied` (see [Moved, renamed and copied files](#moved-renamed-and-copied-files)). A move is listed under the root of its removed source, a copy under the root of the copy, each as `old -> new`; `copied` is printed only when there is one. Without `--paths` only the counts are printed; `unchanged` sources are never listed.

`--json` prints one object with the keys `note`; `roots`, a list with one object per root of the run with `root`, `recorded` (the sources the index records for it), `guard_threshold`, `guarded`, the counts `added`, `modified`, `removed`, `retried`, `refreshed`, `unchanged`, `moved` and `copied`, `modified_by_reason` (`content_changed`, `parser_changed`, `newly_supported`, `delegated_to_ocr`, `delegation_ended`) and `paths` (`null` without `--paths`; otherwise `added`, `modified` as `path` and `reason`, `removed`, `retried`, `refreshed`, `moved` as `removed` and `added`, and `copied` as `original` and `copy`); `total`, the counts of the whole run; `outside_run`, a list of `root` and `sources`; and `guarded_roots`, a list of `root`, `removals`, `recorded`, `threshold` and `reason` (`empties_root` when every recorded source of the root would be removed, otherwise `threshold`).

### Moved, renamed and copied files

A source is identified by its stored path, so a file that is moved, renamed or copied (inside a root, to another root, restored from a backup, or a backup folder added as a root) is a new source. When its content is already indexed, `docgrep update` does not parse or embed it again: it copies the stored passages of that content, their text, location, metadata and both vectors, under the new path. A large book moves in seconds instead of minutes of GPU time.

- **Moved or renamed**: a removed source and an added file with the same content hash and size form a move. The passages are copied to the new path first, then the old source is removed as any removed source is. Each removed source pairs with at most one added file, the first in stored path order (removed sources are taken in stored path order too), so two new copies of one moved file make one move and one copy.
- **Copied**: an added file, or a file whose content changed, with the content of a source that stays (any root, the first in stored path order) gets a copy of its passages; both are indexed. Search collapses the identical passages into one result that lists the other location (see [Duplicate passages](#duplicate-passages)).
- **Not reused**: the stored source must be indexed, with the same content hash, size and file type, the parser version the file would be parsed with now and the embedding model of the index. An original delegated to its OCR output, or an OCR output, is never reused (see [OCR](#ocr)). A failed source that is retried goes through the normal pipeline. Any of these is processed as an addition or a modification.
- **A full build** reuses only within its own run: of two files with the same content, the first is indexed and the next ones are copies of it (the build summary says `N indexed (M copied from a file with the same content)`). It never reuses the index it replaces.

A move counts against the [removal guard](#the-removal-guard) like any removal of its root.

If a copy fails in the store, what was written of the new source is deleted, it is recorded as `failed:store_error` and the old source of a move stays; the next update retries it. A move whose copy is stored but whose old source could not be removed (or whose process stopped in between) leaves both indexed, and the next update removes the old one, whose file is gone.

The run journal of the update or build records the source each copy was made from (`reused_from`); the committed source record is the same as for a source indexed normally. The run totals, the `index_update_finished` log record and the `Latest run:` line of `docgrep status` carry `moved` and `copied` (a build records `copied` when it made a copy), and each copy logs a `source_reused` record.

Passage ids follow the path (ADR-0023), so the passages of a moved or renamed file get new ids: in [Passage context](#passage-context) a passage id saved before the move answers `not_found` once the old source is removed (`source_removed`, with the old path, while a passage of it is still stored but its record is gone), exactly like the ids of any removed source. Search again for the current ids. The ids of the source a copy was made from do not change.

`DOCGREP_REUSE_BY_HASH=false` turns reuse off: every move is a removal and an addition again, and every file is parsed and embedded; the plan then counts no moves or copies. It is meant for comparison and as an escape hatch. The design is recorded in [ADR-0028](docs/adr/0028-reuse-indexed-passages-by-content-hash.md).

### The removal guard

Before applying anything, `docgrep update` checks the planned removals of each root of the run, sources that moved or that the include and exclude patterns no longer match included. When they exceed `min(DOCGREP_UPDATE_GUARD_RATIO × the sources the index records for the root, DOCGREP_UPDATE_GUARD_COUNT)` (defaults `0.05` and `50`; a root with fewer than 20 recorded sources is checked against the count alone), or when they would remove every source the index records for a root, whatever its size and the thresholds (an existing but empty folder, such as a mount point without its disk, can still be read), the update writes nothing at all, not even refreshed times, prints the plan, names the roots and exits with 1:

```text
removal_guard: Update stopped before any change: root 'books' would lose 2 of 32 sources (at most 1 allowed). Check that the root folder is mounted, complete and the right one; run `docgrep update --yes` to apply the removals anyway
```

A root that would be emptied is reported as `root 'notes' would lose all its 2 sources`, in the error and in `docgrep update --plan`.

An unmounted disk, a sync client that briefly emptied a folder or a root saved with the wrong path would otherwise remove a large part of the index, and rebuilding it takes hours. Check the folder, or `docgrep update --plan --paths`, then run:

```sh
uv run docgrep update --yes   # apply the plan even when the guard would stop it
```

`DOCGREP_UPDATE_GUARD_RATIO` takes a number from 0 to 1 and `DOCGREP_UPDATE_GUARD_COUNT` a non-negative integer; `0` for the count makes every removal need `--yes`. `docgrep forget-root` is not guarded: it has its own `--yes`.

### Comparing a folder with the index

`docgrep diff PATH` compares the files under a folder, for example a backup copy before restoring it, with the sources the index records for one root, by path relative to the root. It is read-only, takes no lock and exits with 0. The folder is listed as a root would be, with the include and exclude patterns. `--root NAME` picks the root; without it the only configured root is used (or, with no configured root, the legacy root of the index), and several configured roots are a usage error (exit 2) that names them. Without `--hash` a file with the recorded size and modification time counts as the same, as in an update, and only the others are hashed; `--hash` hashes every file present on both sides.

```sh
uv run docgrep diff /backup/library --root books --paths
```

```text
Folder /backup/library against root books, compared by size and modification time, hashing only the files whose stat differs (2 files hashed):
  same: 28
  different: 1
    vol01.txt
  only in the index: 3
    archive/vol03.txt
    vol05.txt
    vol30.txt
  only in the folder: 3
    extra.txt
    vol03.txt
    vol04.txt
```

`--paths` lists the paths of every group but `same`. `--json` prints one object with `folder`, `root`, `compared_by` (`stat` or `hash`), `hashed`, the counts `same`, `different`, `only_in_index` and `only_in_folder`, and `paths` (`null` without `--paths`; otherwise `different`, `only_in_index` and `only_in_folder`). A file whose content changed while its size and time stayed the same is found only with `--hash`.

The design of the plan, the guard and `diff` is recorded in [ADR-0027](docs/adr/0027-update-plan-removal-guard-and-diff.md).

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
- `latest_run`: the latest run of any kind. `run_id`, `kind` (`build`, including a resumed build, `update`, `ocr` or `reencode_sparse`), `status` (stored: `running`, `swapped`, `completed`, `aborted` or `cancelled`), `state` (the status, or `interrupted` for a `running` run without a lock holder), `started_at`, `finished_at`, `totals` (`indexed`, `skipped`, `failed`, and `copied` when a build copied passages; an update adds `added`, `modified`, `removed`, `retried`, `moved`, `copied` and `unchanged` source counts and `passages_written` and `passages_removed`; an OCR run records `processed`, `skipped`, `failed`, `cancelled` and `not_started` instead, a sparse re-encode `passages`), `total_sources` (passages for a sparse re-encode), `resumed_from` and `carried_sources` (sources taken over from the resumed run).
- `active_operation`: present while the index lock is held. `kind` (`build`, `resume`, `update`, `ocr` or `reencode_sparse`; for `ocr` the sources are the files to process and `passages_written` is 0; for `reencode_sparse` `sources_done`, `sources_total` and `passages_written` count passages), `run_id`, `started_at`, `sources_done` and `passages_written` (from the run's journal, so they include carried-over sources), `sources_total`, `elapsed_seconds`, `eta_seconds` (`null` until computable; the pace of this process, ignoring carried-over sources) and `cancel_requested_at`.
- `vector_store`: `mode` (`local` or `server`), `status` (`reachable`, `unreachable`, `busy` or `absent` for a local store that does not exist yet) and `reason`, a short text for the degraded cases.
- `collection`: when the index can be read. `passage_count`, `embedding_model`, `embedding_dimension`, `embedding_dtype`, `docgrep_version`, `completed_at` and `last_updated_at` (when the latest `docgrep update` completed, `null` before the first).
- `drift`: a list of `field`, `indexed` and `runtime` for the settings that decide what vectors contain (see the resume rules above) and differ from the index. Only keys the index recorded and that are known without loading a model are compared.
- `sources`: what the committed sources table says, or `null` when the manifest is missing or cannot be read. `total`; `by_state` with `indexed`, `skipped` and `failed`; `skipped_by_reason` with `requires_ocr` (no text layer), `garbled_text_layer` (a PDF whose text layer is garbled), `unsupported` (no parser for the type) and `other`, plus `delegated_to_ocr` (an original whose text is indexed from its OCR output) when any source has that reason; `skipped_by_extension`, the 10 most common extensions among the skipped sources, a list of `extension` (`(none)` without one) and `sources`; `failed`, the first 20 failed sources by path, each with `path`, `category` (for example `parse_error`, `embedding_error` or `store_error`) and `reason` (the stored error message, shortened to 200 characters); and `parser_changed`, the number of sources that the next `docgrep update` would extract again because their parser declares another version (the same rule as update, missing versions counting as `1`). The plain-text report shows a `Sources:` line, the skipped counts per reason and the failed sources with their paths.
- `warnings`: a list of `code` and `message`, for example `store_unreachable`, `store_busy`, `manifest_unreadable`, `config_drift`, `resume_available`, `swap_pending_commit`, `update_unfinished` (the latest update was cancelled, aborted or interrupted; run `docgrep update` again), `reencode_unfinished` (the latest sparse re-encode stopped early; run `docgrep reencode-sparse --yes` again), `index_incomplete`, `lock_held_without_run`, `lock_unreadable`, `root_not_configured` (the index holds sources of roots that are not configured) and `root_unreadable` (a configured root cannot be read).
- `roots`: the configured [corpus roots](#corpus-roots) in order, then the roots found only in the index, each with `name`, `legacy`, `configured`, `readable` (`null` for a root that is not configured) and the numbers of its committed sources that are `indexed`, `skipped` and `failed`. No path is shown. The plain-text report lists the roots when there is more than one or one needs attention.
- `sparse`: the [sparse weighting](#sparse-weights-bm25) of the index, `null` without a completed index. `encoder` (`tf-v1` when the index records none, or `bm25-v2`), `bm25_k1`, `bm25_b` and `bm25_avgdl` (`null` for `tf-v1`), `reencode` (the latest sparse re-encode while it runs or after it stopped early, else `null`: `run_id`, `state`, `passages_done` and `passages_total`) and `hint` (what to run next, for a `tf-v1` index or an unfinished re-encode, else `null`). The plain-text report shows a `Sparse encoder:` line, `reencoding (n of N)` while a re-encode runs.

## Dry run

`docgrep dry-run` previews what a build would do with the configured corpus. It reads directory entries only: it takes no index lock, writes no manifest, does not open Qdrant, loads no model and creates no file, and it uses the configured corpus roots (or the `--root` roots), include and exclude patterns, so it predicts what `docgrep index` would enumerate.

```sh
uv run docgrep dry-run                     # plain-text report
uv run docgrep dry-run --json              # machine-readable report with a file list
uv run docgrep dry-run --path-prefix books/ --sample 20 --seed 1
```

The report gives the total number of files, and for each file type the count and the total, average and largest size. Types that docgrep recognises but has no registered parser for are listed separately as recognised without a parser, files with other extensions as unsupported with the 20 most common extensions, and excluded files with the pattern that excluded them (`not_matched_by_includes` when no include pattern matched).

- `--path-prefix` restricts every figure to stored paths starting with the given prefix (a prefixed root's paths start with its name and `/`).
- `--limit` (1 to 500, default 100) and `--offset` page the file list, which is part of the JSON output only.
- `--sample N` (1 to 50, off by default) parses and chunks N randomly chosen files of the supported types, spread proportionally over the types, with the seed given by `--seed` (default 0), so the same seed picks the same files. Nothing is embedded or written. For each type it reports passages per file and per MB (1 MB is 1,000,000 bytes), parse plus chunk seconds per file and the fraction flagged as needing OCR, and then extrapolates the passage count and parse time of the whole corpus both by file count and by size. These are estimates from a sample, not measurements. For PDFs it also measures the text layer of the sampled files: pages sampled, pages classified as garbled (count and fraction), the files that a build would skip as `requires_ocr` because of a garbled text layer (count, fraction of the parsed sampled files and their paths) and the soft hyphens removed per file on average, per type and for all sampled files together; the corpus-wide number of such files is extrapolated as an estimate like the others. Sampling reads whole files, which is why it is opt-in; with the real backend it loads the tokenizer files of the embedding model, never its weights.

The report never contains file content, only counts, sizes, timings, patterns and, in the JSON file list and sample, relative paths; the plain-text report lists no files except the sampled files that would be skipped for a garbled text layer. Invalid values give a `validation_error` and exit status 1.

The JSON output has these keys (`schema_version` is 1):

- `schema_version`, `scope` (`path_prefix`), `totals` (`files`, `size_bytes`, `included`, `excluded`).
- `types`: a list sorted by type with `type`, `status` (`supported` or `recognised_without_parser`), `files`, `size_bytes`, `average_size_bytes` and `largest_size_bytes`.
- `unsupported`: `files`, `size_bytes`, `average_size_bytes`, `largest_size_bytes` and `extensions`, a list of `extension` and `files`.
- `would_be_indexed` (`files`, `size_bytes`) and `recognised_without_parser` (`files`, `size_bytes`, `types`).
- `excluded`: `files`, `size_bytes` and `by_pattern`, a list of `pattern` and `files`.
- `by_root`: one entry per [corpus root](#corpus-roots) of the run, in order, with `root`, `files`, `size_bytes`, `included`, `excluded` and `would_be_indexed`; the plain-text report shows them when there is more than one root.
- `files`: `offset`, `limit`, `total` and `items`, a list of `path`, `size_bytes`, `extension`, `type` (null when unsupported), `status` (`indexed`, `recognised_without_parser`, `unsupported` or `excluded`) and `excluded_by`.
- `sample`: null without `--sample`, otherwise `requested`, `seed`, `note` and `types`, a list of `type`, `sampled_files`, `parsed`, `failed`, `requires_ocr_fraction`, `passages_per_file`, `passages_per_mb`, `parse_chunk_seconds_per_file`, `text_quality` and `estimate` (`note`, `corpus_files`, `corpus_size_bytes`, and `by_file_count` and `by_size`, each with `passages` and `parse_chunk_seconds`; `by_file_count` also has `garbled_text_layer_files`, the estimated number of files in the corpus that would be skipped for a garbled text layer). `sample.text_quality` totals the same figures over all sampled types.
- `text_quality` (measured on the sampled files, never text): `pages_sampled`, `pages_garbled`, `pages_garbled_fraction` (of the pages sampled), `garbled_text_layer_files` (paths of the sampled files that would be skipped as `requires_ocr` for a garbled text layer), `garbled_text_layer_fraction` (of the parsed sampled files), `soft_hyphens_removed_per_file` (average over the parsed sampled files), `pages_by_rule` (on how many sampled pages each rule fired, counted separately from the dropped pages so that the weak signals stay visible: `r1_broken_characters`, `r2_long_tokens` and `r3_low_alnum`) and `flagged_files`, a list of the sampled files on which any rule fired, each with `path`, `pages_dropped` (pages dropped for broken characters), `pages_by_rule` and `dropped_pages` (at most the first 20 dropped page numbers). The plain-text report shows the counts per rule but lists no flagged files; use `--json` for them. Fractions are null when there is nothing to divide by; types other than PDF report zero pages.

## Scan discovery

`docgrep discover-scans` lists the PDF and DjVu files whose text layer is missing or unusable, so that you can decide which ones need OCR. It is read-only like the dry run: it takes no index lock, opens neither the manifest nor Qdrant, loads no model, writes nothing and can run before any build. It uses the build's corpus roots (or the `--root` roots) and include and exclude patterns.

```sh
uv run docgrep discover-scans                          # plain-text list and counts
uv run docgrep discover-scans --json                   # machine-readable report
uv run docgrep discover-scans --path-glob 'books/**' --file-type djvu
```

- `--path-glob` keeps the files whose relative path matches the pattern (the same glob rules as the include and exclude patterns).
- `--file-type` (`pdf` or `djvu`, repeatable) restricts the types; the default is both.
- `--include-processed` also lists files already processed through OCR (`docgrep ocr`); by default they are left out.

For each file a few pages are sampled: the first 3, the last one and 2 from the middle, capped at the page count and without duplicates. The middle pages are chosen from SHA-256 of the relative path and the page count, so the same file is always sampled the same way and two runs on an unchanged corpus give the same report. The text density is the mean number of non-whitespace characters per sampled page, where a page that the page policy would drop for broken characters (R1) counts as zero. Below `DOCGREP_SCAN_DENSITY_LOWER` (default 100) a file is `scanned`, above `DOCGREP_SCAN_DENSITY_UPPER` (default 500) it is `native`, in between (bounds included) `mixed_or_uncertain`. An ordinary book page carries well over 1,000 characters, so a book stays native even with blank pages or short chapter openings in its sample, while a scan whose text layer is empty, or holds only a page number or a title page, stays far below 100. A file whose sampled pages are at least half broken (R1) is `scanned` with the reason `garbled_text_layer`, because OCR has to redo such a layer. Native files are only counted; the others are listed. A file that cannot be inspected is listed as `inspection_failed` and the scan continues.

Exit status 0 on success and 2 for usage or configuration errors (an unknown option or file type, invalid thresholds, an inaccessible corpus root).

The JSON report has these keys (`schema_version` is 1):

- `schema_version`, `filters` (`path_glob`, `file_types`) and `thresholds` (`density_lower`, `density_upper`).
- `files`: the scanned, uncertain and failed files, sorted by file type and path, each with `path` (the stored path: relative to its root, behind the root's prefix), `file_type` (`pdf` or `djvu`), `size_bytes`, `page_count` (null when unknown), `classification` (`scanned`, `mixed_or_uncertain` or `inspection_failed`), `reason` (`low_text_density`, `garbled_text_layer` or `uncertain_text_density`; for failures `encrypted`, `corrupt_header`, `djvu_invalid`, `djvu_tools_missing`, `djvu_timeout`, `djvu_output_too_large` or `unsamplable`), `hint` (how to install the DjVu tools when they are missing, else null), `ocr_status` (`not_processed`; `processed_via_ocr_pdf` for an original whose text is indexed from its OCR output; `processed` for an OCR output itself; the last two are listed only with `--include-processed`), `large_file` (more than 5000 pages or 1 GiB), `text_density` (null for failures) and `sampled_pages`.
- `totals` and `by_file_type` (per requested type): `inspected`, `native`, `scanned`, `uncertain` and `inspection_failed`.
- `by_root`: the same counts for each [corpus root](#corpus-roots) of the run, a list of entries with `root` and the counts; the plain-text report shows them when there is more than one root.
- `include_already_processed` (whether `--include-processed`, or the HTTP field of the same name, was given) and `already_processed`: the number of originals left out because a live OCR output stands in for them. Without the option, OCR outputs and such originals are neither listed nor counted in `totals`.

## OCR

`docgrep ocr` makes the files that scan discovery lists searchable: scanned PDFs and DjVu files without a text layer, and PDFs whose text layer is garbage. It runs the system `ocrmypdf` command (with Tesseract) on each file and writes a new, searchable PDF next to the original; the original is never changed. Nothing is indexed by the command itself: run `docgrep update` afterwards. The design is recorded in [ADR-0017](docs/adr/0017-ocr-via-ocrmypdf-with-registry-and-delegation.md).

Install the tools from the distribution (Debian; `djvulibre-bin` as well for DjVu files):

```sh
sudo apt install ocrmypdf tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus tesseract-ocr-ukr
```

Without them the command stops before any work with `ocr_tools_missing` and that install command; a requested language that Tesseract lacks is reported the same way, naming the language.

### Workflow

```sh
uv run docgrep discover-scans                          # which files need OCR
uv run docgrep ocr --all-discovered --plan             # the work plan, without OCR and without the lock
uv run docgrep ocr 'books/one scan.pdf' books/old.djvu # a trial on one or two files
uv run docgrep ocr --all-discovered                    # the full batch
uv run docgrep update                                  # index the outputs
```

Name the files with exactly one of:

- explicit paths, as stored (relative to their root behind the root's prefix, as `discover-scans` lists them) or absolute. Every path must be an existing `.pdf`, `.djvu` or `.djv` file inside a corpus root; a path outside them, or a file of another type, rejects the whole request (exit status 2);
- `--path-glob PATTERN` (the glob rules of the include patterns); matching files of other types are left out and counted as `filtered`;
- `--all-discovered`: every file scan discovery classifies as `scanned`; add `--include-uncertain` for the `mixed_or_uncertain` ones too.

Options: `--languages rus+eng` (default `DOCGREP_OCR_LANGUAGES`), `--force` to process files that are already processed again (for example with other languages), `--plan` to print the plan and exit (status 0; it checks the tools but takes no lock and writes nothing), `--override-native` (explicit paths only) to process a file classified as native, for a text layer that is wrong although it looks fine, and `--json` for a machine-readable report at the end instead of the text output.

### What happens to each file

The plan checks every file, in this order: a file whose content is already registered with an existing output, and every OCR output itself, is `skipped:already_processed` (unless `--force`; outputs are never processed again); a file classified as native is `skipped:does_not_need_ocr` (unless `--override-native`); an encrypted file is `failed:encrypted`; a file that cannot be read is `failed:unreadable` (or `failed:djvu_tools_missing` for DjVu without the djvulibre tools). The rest is processed, sorted by file type and path:

1. A DjVu file is converted into a temporary PDF of page images with `ddjvu -format=pdf`.
2. `ocrmypdf` runs with the languages, `--output-type pdf --rotate-pages --deskew --optimize 1`, `--jobs N` when `DOCGREP_OCR_JOBS` is set, and one of two modes: `--skip-text` (keep pages that have text, recognise the others) for files without text or with too little, and `--force-ocr` (discard every page's text and recognise it again from the page image) for a garbled text layer and for `--override-native`.
3. The result, written to a hidden `.<name>.docgrep-ocr-<random>.part` file in the same directory, must open with the input's page count; it is then renamed to `<stem>.ocr.pdf` next to the original (`books/scan.pdf` gives `books/scan.ocr.pdf`, `books/book.djvu` gives `books/book.ocr.pdf`). An existing file of that name that is not this original's earlier output is never overwritten. The output is always in the original's [corpus root](#corpus-roots), so an original and its output never belong to different roots.
4. The output is registered in the manifest database together with a delegation: the original is from now on represented by its output. `docgrep update` then indexes the output as an added PDF, records the original as skipped with the reason `delegated_to_ocr` and removes the passages it had.

Failures stay per file: `failed:djvu_conversion`, `failed:ocr_engine_error` (with a short message), `failed:timeout`, `failed:output_invalid`, `failed:output_collision` and `failed:registry_write` (the output is left on disk but not registered, so the next run processes the file again). Before each file at least three times its size must be free in its directory; otherwise, or when the engine reports a full disk, the file is `failed:disk_full` and the rest of the batch is aborted. Files completed before stay valid.

To revert a file, delete its `.ocr.pdf`: the delegation ends, and the next `docgrep update` removes the output's passages and indexes the original again. Changing the original ends the delegation too.

### Output, progress and time

The plan is printed first (counts to process, to skip and to fail by reason and file type, total input size and pages), then one line per finished file and a summary:

```text
[1/3] processed book.djvu, 1 pages, 2s, ETA 6m 30s
[2/3] processed garbled.pdf, 1 pages, 8s, ETA 10s
[3/3] processed latin.pdf, 1 pages, 2s, ETA 0s
OCR run 1 completed in 13s: 3 processed, 0 skipped, 0 failed, 0 cancelled, 0 not started.
Run `docgrep update` so that the OCR outputs are indexed.
```

This is a run on three generated one-page files. The ETA extrapolates the pages processed so far when every file has a known page count (files otherwise), so it is rough while few files are done. OCR takes several seconds per page on one core, so a batch of books can take hours; `DOCGREP_OCR_JOBS` lets ocrmypdf use more cores. The log gets INFO records `ocr_plan`, `ocr_file_started` (path, size, pages, mode) and `ocr_file_finished` (status, elapsed), and the tools' own output at DEBUG.

### Cancel and resume

Ctrl-C, SIGTERM or `docgrep cancel` from another shell stop the batch. With `DOCGREP_OCR_CANCEL_MODE=terminate` (the default) the running OCR is stopped (SIGTERM to its process group, SIGKILL after 10 seconds), its temporary files are removed and the file is reported `cancelled`; with `finish` the current file completes first. Completed files stay registered. There is no separate resume: run the same command again, and every completed file is skipped as `already_processed`. While a batch runs, `docgrep status` shows it as an active operation of kind `ocr`, and it holds the index lock, so builds and updates wait for it.

### Settings

- `DOCGREP_OCR_LANGUAGES`: Tesseract languages joined with `+`, default `rus+eng+ukr`.
- `DOCGREP_OCR_TIMEOUT_SECONDS`: time limit per file (DjVu conversion and OCR together), default 0 for no limit; a file over the limit is `failed:timeout`.
- `DOCGREP_OCR_JOBS`: passed to `ocrmypdf --jobs`; unset leaves ocrmypdf's default (all cores).
- `DOCGREP_OCR_CANCEL_MODE`: `terminate` (default) or `finish`.

### JSON

`docgrep ocr --plan --json` prints the plan (`schema_version` 1): `languages`, `force`, `filtered`, `to_process`, `to_skip` and `to_fail` (counts by reason), `by_file_type` (per `pdf` and `djvu`: `to_process`, `to_skip`, `to_fail`), `input_bytes`, `pages` and `pages_unknown` (files to process whose page count is not known), and `files`, each with `path`, `file_type`, `size_bytes`, `page_count`, `classification` and `reason` (from scan discovery), `mode` (`skip-text` or `force-ocr`, null when not processed) and `status` (`to_process`, `skipped:<reason>` or `failed:<reason>`).

`docgrep ocr --json` prints the final report (`schema_version` 1): `run_id`, `status` (`completed`, `cancelled` or `aborted`), `abort_reason` (`disk_full` or null), `plan` (the plan above without `files`), `totals` (`processed`, `skipped`, `failed`, `cancelled`, `not_started`), `skipped_by_reason`, `failed_by_reason`, `elapsed_seconds`, `files` (the plan's file keys with the final `status`, plus `output_path`, `message` and `elapsed_seconds`) and `hint`, the reminder to run `docgrep update`.

## Search

```sh
uv run docgrep search "phrase to find" --top-n 5
uv run docgrep search "phrase to find" --json
```

Results include the exact stored passage text, source path, and a page number or Markdown heading path when available. The JSON form includes scores and query metadata.

### Filtered search

Search can be restricted to the indexed sources that match the same filters as [Browse sources](#browse-sources): `source_ids`, `authors`, `year_from`, `year_to`, `file_types`, `path_prefixes`, `path_globs` and `roots` (on the command line `--source-id`, `--author`, `--year-from`, `--year-to`, `--type`, `--path-prefix`, `--glob`, `--in-root`, each list option repeatable). Their semantics are those of the browse section:

- Different filters combine with AND; several values of one filter combine with OR.
- A source without the filtered metadata (no author, no year) never matches that filter.
- Authors match as case-insensitive substrings; file types and source ids as listed there.
- Paths are matched as stored, with `/` separators, as a case-sensitive prefix or a glob; `roots` names [corpus roots](#corpus-roots).

```sh
uv run docgrep search "comet orbit" --author arnold --year-to 1980 --type pdf
```

```sh
curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"comet orbit","authors":["arnold"],"year_to":1980,"file_types":["pdf"]}'
```

The MCP tool `find_passages` takes the same fields next to `query`, for example `{"query": "comet orbit", "authors": ["arnold"], "path_prefixes": ["Books/Math/"]}`.

The filter is applied before ranking: docgrep turns the matching sources from the manifest into a list of `source_id` values, and the vector store searches only their passages, so the `top_k` first-stage candidates all come from the filtered sources. Reranking, duplicate collapsing and `top_k` work as without filters, and `total_hits` still counts the first-stage candidates (at most `top_k`), now taken from the filtered sources. The response adds:

- `applied_filters`: the canonical filter, `{}` for an unfiltered search (whose body is otherwise unchanged).
- `root` in each result's `source` and in each entry of `duplicates`: the name of the source's [corpus root](#corpus-roots).
- `sources_excluded_for_missing_metadata`: only when `authors`, `year_from` or `year_to` is set; the number of sources, not passages, that pass every other filter but are left out only because the filtered author or year is missing.

Filters that match no indexed source give an empty success with `total_hits: 0` and a `hint`, without loading a model. Invalid values (an unknown file type, `year_from` after `year_to`, a blank author) and unknown fields are `validation_error`s. The plain-text output starts with a `Filters:` line when filters are set. The design is recorded in [ADR-0020](docs/adr/0020-shared-source-filter-and-browse.md).

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

Collapsing is controlled by `DOCGREP_COLLAPSE_DUPLICATES` (default `true`), by the request field `collapse_duplicates` of `POST /search` (`true`, `false`, or `null` for the setting) and by `docgrep search --no-collapse-duplicates`. With collapsing off, results and response are those of release 1.10.0, plus `applied_filters: {}` (see [Filtered search](#filtered-search)): no `duplicates` and no `collapsed_duplicates` keys. Collapsing works at query time on the stored passages; it changes nothing in the index and needs no rebuild.

## Browse sources

List what is indexed, with each source's path, file type, title, author, year, passage count, index time, content hash and OCR origin. Browsing is read-only: it reads the manifest, the vector store and the OCR registry, never the corpus, and takes no lock. Only indexed sources are listed; skipped and failed ones are in `docgrep status`.

```sh
uv run docgrep sources --author arnold --year-from 1960 --type pdf --sort year --desc
uv run docgrep sources --glob 'Books/**/*.djvu' --has-metadata without_any --json
```

```sh
curl -X POST http://127.0.0.1:8000/sources/browse \
  -H 'Content-Type: application/json' \
  -d '{"authors":["arnold"],"file_types":["pdf"],"page":1,"page_size":50}'
```

The MCP tool `browse_sources` takes the same fields as the JSON body, for example `{"path_prefixes": ["Books/Math/"], "sort_by": "passage_count", "sort_order": "desc"}`.

Filters (`source_ids`, `authors`, `year_from`, `year_to`, `file_types`, `path_prefixes`, `path_globs`, `roots`; on the command line `--source-id`, `--author`, `--year-from`, `--year-to`, `--type`, `--path-prefix`, `--glob`, `--in-root`, each list option repeatable):

- Different filters combine with AND; several values of one filter combine with OR. A source without the filtered metadata (no author, no year) never matches that filter.
- Authors match as case-insensitive substrings; file types are `pdf`, `markdown`, `txt`, `epub`, `docx` and `djvu`, in any case; source ids match exactly.
- Paths are matched as stored and shown in results: relative to their root behind the root's prefix (see [Corpus roots](#corpus-roots)), with `/` separators. A prefix is a case-sensitive string prefix (`Books/Math/`); a glob follows the [include and exclude pattern](#include-and-exclude-patterns) rules.
- `roots` keeps the sources of the named roots; an unknown name is a `validation_error` listing the known roots.
- `has_metadata` keeps sources with a year (`with_year`), an author (`with_author`), both (`with_both`) or neither (`without_any`); the default `any` keeps all.

Titles, authors and years are shown and matched after junk values are dropped: impossible years (such as `101`, or `1601`, the Windows epoch), placeholder authors and titles (`Unknown`, `python-docx`, `book title`), bare numbers and titles that only repeat the file name. A dropped value is shown as null; nothing is guessed. An OCR output takes a missing or dropped value from its original. The stored values do not change. To judge the rules on your library, each entry of the JSON body carries a `metadata_dropped` object when something was dropped or inherited, with the raw value and a reason, and `--dropped-metadata` (`"metadata_dropped": true`) lists only those sources:

```sh
uv run docgrep sources --dropped-metadata --json
```

The rules are recorded in [ADR-0021](docs/adr/0021-sanitize-source-metadata-on-read.md).

Each entry carries `root`, the name of its [corpus root](#corpus-roots). The response holds the page (`sources`), `total_hits`, `page`, `page_size`, `aggregates` (`total_sources` and `by_file_type`, over every match) and the canonical `applied_filters`. Pages start at 1 and hold 1 to 500 sources (default 50); `sort_by` is `path` (default), `indexed_at`, `passage_count`, `year` or `title`, with ties broken by source id and missing values last. Passage counts are counted in the index at query time for the page only; sorting by `passage_count` counts every match and is the slow case. A count that fails gives that source `passage_count: null` and `degraded: true`. Unknown fields are rejected. The design is recorded in [ADR-0020](docs/adr/0020-shared-source-filter-and-browse.md).

## Passage context

Fetch the text around passages from search or browse results, to verify or quote them. The context is taken from the neighbouring indexed passages of the same source, verbatim; the corpus is not read, nothing is reindexed and no lock is taken.

```sh
uv run docgrep search "phrase to find" --json | jq -r '.results[0].passage.passage_id'
uv run docgrep context <passage-id> --window 1000
uv run docgrep context <passage-id> <another-id> --json
```

```sh
curl -X POST http://127.0.0.1:8000/passages/context \
  -H 'Content-Type: application/json' \
  -d '{"passage_ids":["<passage-id>"],"window_tokens":500}'
```

The MCP tool `expand_passage_context` takes the same fields, for example `{"passage_ids": ["<passage-id>"], "window_tokens": 200}`.

The window rule:

- `window_tokens` applies to each side of every passage (default `DOCGREP_CONTEXT_WINDOW_TOKENS`, 500; at most `DOCGREP_CONTEXT_MAX_WINDOW_TOKENS`, 5000). Whole neighbouring passages are added outward while their stored token counts fit, and the nearest one on each side always is; `0` returns the passage alone. With the default chunking this is about one passage on each side.
- The overlap between consecutive passages is removed using their offsets, so no text is repeated. Where two passages do not touch, they are joined with a space inside a page or section and with a blank line across pages or sections; nothing else is added. `context_before + passage.text + context_after` is the whole window.

Each requested id (at most `DOCGREP_CONTEXT_MAX_PASSAGES`, 20; duplicates are answered once) gets an entry with a `status`: `ok`, `not_found` (no such passage in the index, for example after an update), `source_removed` (with the last known `path`) or `source_modified` (with `indexed_content_hash` and `current_content_hash`); for the last three, search again for current ids. An `ok` entry carries the `passage`, its `source` (with the cleaned metadata of search results), `context_before`, `context_after`, the page and heading path of each neighbour used (`before_locations`, `after_locations`), `at_source_start`, `at_source_end` and `window_tokens_used`. The request answers 200 even when every entry failed; an empty list, too many ids, a malformed id or a window out of range is a validation error. The design is recorded in [ADR-0022](docs/adr/0022-passage-context-from-neighbouring-passages.md).

## Retrieval evaluation

Measure where known passages rank, to decide changes to retrieval with numbers. `docgrep eval` runs each case of a cases file through the same search pipeline and reports, for the expected passage, its rank at every stage. It is read-only (no lock, no writes) and changes nothing about search.

```sh
uv run docgrep eval my-cases.private.jsonl
uv run docgrep eval my-cases.private.jsonl --json > eval.json
uv run docgrep eval my-cases.private.jsonl --top-k 300 --depth 5000 --no-control
uv run docgrep eval my-cases.private.jsonl --rerank-pool 150
uv run docgrep eval my-cases.private.jsonl --dense-hnsw-ef exact --hnsw-recall
```

The cases file is JSON Lines: one JSON object per line; blank lines and lines starting with `#` are skipped. Fields:

- `query` (required);
- `source`: the stored path of the expected source (the `source_id` of search results; paths of a prefixed root start with its name and `/`);
- `source_glob`: a path glob matched against the stored path, with the rules of `DOCGREP_INCLUDE_GLOBS`;
- `fragment`: text of the expected passage; it matches when the passage text contains it after runs of whitespace are collapsed in both, case-sensitively;
- `filters`: the filter fields of `POST /search` (see [Filtered search](#filtered-search)), applied as in search;
- `id` (default: the line number) and `note`, which are only reported.

At least one of `source`, `source_glob` and `fragment` is required; a passage is a hit when it satisfies all that are given. Unknown fields, a malformed line or an invalid value stop the command before anything runs, naming the line. For example:

```json
{"id": "arnold-comet", "query": "comet orbit", "source": "books/arnold.md", "fragment": "mention a comet once", "note": "decoys outrank it"}
```

[`examples/eval-cases.example.jsonl`](examples/eval-cases.example.jsonl) is a synthetic file for the generated library of the tests. Real cases name private files: keep them in files named `*.private.jsonl`, which git ignores, and never commit them.

Each case reports a rank (1-based; `-` or `null` when not found) in five lists, and a source rank, the position of the first passage of a matching source, when the case names a source:

- `dense` and `sparse`: each first-stage channel alone, read to `--depth` candidates (default 2000, at most 10000). `sparse` has no rank and a reason when the query has no sparse terms (`no_sparse_terms`) or the index has no sparse channel (`sparse_unavailable`); the fusion then has the dense list alone and carries the same reason;
- `fused_deep`: the fusion of both channels at `--depth`, the first stage as search computes it, only deeper;
- `stage1`: the position among the first-stage candidates the search reranks: the [rerank pool](#rerank-pool) (`--top-k` replaces `DOCGREP_TOP_K`, `--rerank-pool` replaces `DOCGREP_RERANK_POOL`, `--dense-hnsw-ef` replaces `DOCGREP_DENSE_HNSW_EF`);
- `final`: the position in the search order after rerank and duplicate collapse, over all `top_k` candidates, not only the first `top_n`. A passage collapsed into a duplicate counts at the position of the passage that carries it.

`within_top_k` says whether a hit is among the `top_k` candidates; `rerank_score` is the `score` search returns for the best-placed hit (empty when it is not within `top_k` or rerank was skipped). Unless `--no-control` is given, a case with `source` or `source_glob` also runs a control search restricted to those sources, through filtered search: `control_rank` and `control_score` show whether the passage can be retrieved at all when the other sources are out of the way.

Reading the result: a hit in a channel or in `fused_deep` that is not within `top_k` is lost in the first stage; a hit within `top_k` with a poor `final` rank is demoted by the reranker; a good control rank with no `stage1` rank means the competing sources crowd it out of the first stage. The aggregates give, over all cases, how many are found in each list, recall@1, 3, 5, 10 and 25 and MRR on `final` (a case not found counts 0), recall within `top_k` and within the depth, and the ids of the cases lost in the first stage, lost in rerank (within `top_k`, final rank above 10) and not found in any list.

`--hnsw-recall` also measures, for every case, the recall of the dense channel at `top_k` against exact search: the share of the exact `top_k` nearest passages that the configured [search effort](#dense-search-effort) returns. It is reported per case and as a mean and a minimum (`hnsw_recall` in `--json`, after the aggregates); a value well below 1 means the dense channel loses passages before the fusion.

The text output starts with the index summary (passage count, models, `top_k`, depth, rerank pool, dense search effort and the [sparse weighting](#sparse-weights-bm25) of the index) and cuts long queries and notes with `…`. `--json` prints one object with `settings` (among them `top_k`, `rerank_pool`, `dense_hnsw_ef`, `sparse_encoder`, `bm25_k1`, `bm25_b` and `bm25_avgdl`), `cases` (in file order, never cut), `aggregates` and, with `--hnsw-recall`, `hnsw_recall`, in a fixed key order and without timings, so two runs over the same index and settings give identical output. All queries are embedded in one call. The design is recorded in [ADR-0024](docs/adr/0024-retrieval-evaluation-through-a-diagnostic-seam.md).

## Sparse weights (BM25)

The first stage of search fuses two channels: dense vectors and sparse term vectors. Since 1.21.0 a new index weighs the sparse vectors of its passages with BM25: a term repeated many times in one passage saturates (`k1`, default 1.2) and the same count weighs more in a shorter passage (`b`, default 0.75, relative to the average passage length of the corpus). Qdrant applies the inverse document frequency. Queries weigh each distinct term once. Earlier releases stored raw term counts (`tf-v1`), so a passage that repeated one query term 30 times outranked a passage that held the whole query once. The weighting is recorded in the index (`sparse_encoder` `tf-v1` or `bm25-v2`, with `bm25_k1`, `bm25_b` and `bm25_avgdl`; an index without the key is `tf-v1`), and search, `docgrep update` and the indexing of OCR outputs always use the weighting the index records. `docgrep status` shows it and suggests a re-encode for a `tf-v1` index; search works on both. The design is recorded in [ADR-0025](docs/adr/0025-bm25-sparse-weights-and-a-rerank-pool.md).

An index built before 1.21.0 does not need to be rebuilt: the sparse vectors are computed again from the passage text stored in the index, without embedding anything.

```sh
uv run docgrep reencode-sparse                 # the plan: passages, current and new weighting, average length; writes nothing
uv run docgrep reencode-sparse --yes           # re-encode
```

`--k1` and `--b` choose other BM25 parameters, for example `docgrep reencode-sparse --yes --k1 1.5 --b 0.5`.

The plan reads the text of every passage once to compute the average length. With `--yes` the command takes the index lock (it refuses while a build, update or OCR run holds it, and while a build is unfinished), reads the passages again and rewrites only the sparse vector of each point, in pages of 512; dense vectors and payloads are not touched and no model is loaded, so it needs no GPU. On the reference server it took about 2 minutes 20 seconds per 100,000 passages against a Qdrant server (15 seconds to compute the average length, the rest to rewrite), so a library of 870,000 passages takes 20 to 25 minutes. It logs `index_progress` records like a build, counting `passages_done` of `passages_total`, and is recorded as a run of kind `reencode_sparse`. Search keeps working throughout; until the run ends some passages carry the new weights and some the old, and the index still records the old weighting. Ctrl-C or `docgrep cancel` stops it after the current page; running `docgrep reencode-sparse --yes` again continues from the last page written, with the average length and parameters the run started with (other `--k1` or `--b` values start over). While a re-encode is unfinished `docgrep update` refuses to run, so new passages never get the weighting being replaced. At the end the weighting is recorded in the index. Running it again with the same parameters does nothing; other parameters re-encode every passage again.

A full `docgrep index` writes `bm25-v2` itself: it writes BM25 weights under a provisional average length while it streams the sources, then computes the true average over everything it wrote and re-encodes every passage of the new collection before publishing it, so a resumed build equals an uninterrupted one. `docgrep update` keeps the average length the index records, so after many updates it may drift from the true average; a re-encode with the same parameters does nothing, while a rebuild or a re-encode with other parameters computes it anew.

### Rerank pool

Each channel contributes its `top_k` best passages (`DOCGREP_TOP_K`, default 100), and their fusion is cut before the reranker. `DOCGREP_RERANK_POOL` sets how many fused candidates the reranker scores: a positive number, at most twice `top_k` (the size of the union of the two channels). Unset, it is 150, or twice `top_k` when that is smaller (since 1.22.0; earlier releases reranked `top_k`). A larger pool rescues a passage that one channel ranks well but the fusion pushes below `top_k`, at the cost of reranking more candidates per query. On the reference library a pool of 150 recovered a case lost at fused rank 128 and changed no other case, and 200 added nothing. The pool appears in `GET /info` and, when it differs from `top_k`, in the `applied_config` of a search. Set `DOCGREP_RERANK_POOL` to `top_k` for the reranking of earlier releases.

### Dense search effort

Qdrant searches the dense vectors approximately (HNSW). With its default search parameters the dense channel of the reference library (870,000 passages, 1024 dimensions) returned on average only 80% of its true 100 nearest passages, and as little as 46% for some queries, so passages were lost before the fusion. `DOCGREP_DENSE_HNSW_EF` sets the search effort of the dense channel: an `hnsw_ef` value (default 1024; a request for more hits than that searches with its own limit) or `exact` for exact search. Measured on the same library, recall at 100 and mean time per dense query:

| `DOCGREP_DENSE_HNSW_EF` | recall@100 | time |
|---|---|---|
| Qdrant default | 0.80 | 21 ms |
| 512 | 0.97 | 34 ms |
| 1024 | 0.99 | 42 ms |
| 2048 | 0.995 | 62 ms |
| `exact` | 1.00 | about 150 ms |

The reranker takes far longer per query than any of these, so the default favours recall. The effort applies to search and to `docgrep eval` (whose `--hnsw-recall` measures it on your own cases), appears in `GET /info` and in the `applied_config` of a search, and needs no reindex. In local mode Qdrant always searches exactly, so the setting has no effect there. The design is recorded in [ADR-0026](docs/adr/0026-dense-search-effort-and-the-rerank-pool-default.md).

### Measuring the effect

Use [retrieval evaluation](#retrieval-evaluation) with the same cases before and after each change:

```sh
uv run docgrep eval my-cases.private.jsonl --json > before.json
uv run docgrep reencode-sparse --yes
uv run docgrep eval my-cases.private.jsonl --json > after.json
uv run docgrep eval my-cases.private.jsonl --rerank-pool 150 --json > pool-150.json
uv run docgrep eval my-cases.private.jsonl --rerank-pool 200 --json > pool-200.json
```

The `settings` of each report name the sparse weighting, the rerank pool and the dense search effort it ran with. Back up the data directory and the Qdrant storage before a re-encode of a large library.

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

The body of `POST /search` takes `query`, an optional `top_n`, an optional `collapse_duplicates` (see [Duplicate passages](#duplicate-passages)) and the optional filter fields (see [Filtered search](#filtered-search)); unknown fields are rejected.

`POST /sources/browse` lists the indexed sources (see [Browse sources](#browse-sources)). `POST /passages/context` returns passages with the text around them (see [Passage context](#passage-context)). `POST /index/build` triggers the same full rebuild as `docgrep index`. `POST /scans/discover` runs the read-only scan discovery with an optional JSON body `{"path_glob": ..., "file_types": [...]}` and returns the same report as `docgrep discover-scans --json` (see [Scan discovery](#scan-discovery)). `POST /scans/process` runs an OCR batch synchronously, like `POST /index/build`, with a JSON body of `paths` (a list), `path_glob` or `all_discovered: true`, plus optional `languages` and `force`, and returns the same report as `docgrep ocr --json` (see [OCR](#ocr)); `docgrep cancel` stops it, missing OCR tools give 503 with category `ocr_tools_missing` and a held lock gives 409. `POST /scans/discover` also accepts `include_already_processed`. `GET /health` and `GET /info` provide basic service information: `GET /info` returns `version`, `backend` and `roots`, the configured [corpus roots](#corpus-roots) in order, each with `name`, `path` (absolute), `legacy` and `readable` (the folder exists and can be listed; nothing is walked). It does not read the index, so it gives no source counts (see `docgrep status`), and lists no root found only in the index. No endpoint adds or changes roots, and `POST /index/build` and `POST /scans/process` take no folder: they use the saved roots, and OCR `paths` must lie inside one of them.

When `rerank_skipped` is true, results are ordered by their stage-1 hybrid rank and `score` is derived from the dense similarity. It is not comparable with reranked scores and need not be monotonic with the returned order.

## MCP server

docgrep offers its search to MCP clients, such as Claude Code, as read-only tools over the same use cases as the command line and the HTTP API. Results are verbatim passages from your own documents, never summaries; clients are told to cite the path, the heading path and, when the passage has one, the page. Passage text is data from your documents, not instructions for the client. The design is recorded in [ADR-0019](docs/adr/0019-mcp-server-as-api-front-end.md).

- `find_passages`: ranked verbatim passages for `query`, with an optional `top_n` (default 10, at most `DOCGREP_MCP_MAX_TOP_N`), `collapse_duplicates` and the filters of [Filtered search](#filtered-search), `roots` included; the structured result is the body of `POST /search`, and DOCX and EPUB passages have a null page number. Filters that match nothing give an empty result, not an error.
- `get_index_status`: the report of `docgrep status --json`, including the failed sources and their errors and the `roots` (names, flags and counts; never a path).
- `browse_sources`: one page of indexed sources with their metadata and passage counts, filtered and sorted as described in [Browse sources](#browse-sources) (`page_size` default 50, at most `DOCGREP_MCP_MAX_PAGE_SIZE`); the structured result is the body of `POST /sources/browse`. A page too large for one tool result keeps its first sources and says so in `hint`.
- `expand_passage_context`: the passages with the given `passage_ids` (at most `DOCGREP_MCP_MAX_CONTEXT_PASSAGES`) and the text around them, as described in [Passage context](#passage-context); the structured result is the body of `POST /passages/context`. A result too large for one tool result keeps its first entries and says so in `hint`.

No tool indexes, runs OCR, changes the configuration (the corpus roots included) or cancels anything. A missing or empty index, an invalid argument and every other failure come back as a tool error with a category, a message and details; only `browse_sources` answers a missing index with an empty listing and `expand_passage_context` with `not_found` entries, as their HTTP endpoints do.

There are two transports. `docgrep serve` serves the tools over streamable HTTP at `/mcp`, next to the REST API; this is the one to use day to day, because one process holds the models for every client. `docgrep mcp` serves them on standard input and output for a client that starts the server itself; each such process loads its own models on the first search.

With a local-mode index (no `DOCGREP_QDRANT_URL`), `get_index_status` can report the vector store as busy right after a search in the same server process; a Qdrant server (`DOCGREP_QDRANT_URL`) avoids it.

Settings:

- `DOCGREP_MCP_HTTP_ENABLED`: serve `/mcp` from `docgrep serve` (default `true`).
- `DOCGREP_MCP_MAX_TOP_N`: the largest `top_n` a tool accepts (default 25, never more than `DOCGREP_TOP_N_MAX`); a larger value is an error, not cut down.
- `DOCGREP_MCP_MAX_PAGE_SIZE`: the largest `page_size` `browse_sources` accepts (default 100, at most 500); a larger value is an error, not cut down.
- `DOCGREP_MCP_MAX_CONTEXT_PASSAGES`: the most passage ids `expand_passage_context` accepts in one call (default 5, never more than `DOCGREP_CONTEXT_MAX_PASSAGES`); more is an error, not cut down.

The `/mcp` endpoint accepts the same Host and Origin values as the REST API: 127.0.0.1, localhost, ::1, the bind host and `DOCGREP_ALLOWED_HOSTS`.

To connect Claude Code to a running `docgrep serve` (the address is `DOCGREP_HOST` and `DOCGREP_PORT`, by default 127.0.0.1 and 8000):

```shell
claude mcp add --transport http docgrep http://127.0.0.1:8000/mcp
```

A stdio server does not inherit your shell profile, so start it through a small script that loads your settings first, for example `~/bin/docgrep-mcp`:

```sh
#!/bin/sh
set -a
. "$HOME/.config/docgrep/env"   # DOCGREP_DATA_DIR, ...
set +a
PATH="/path/to/docgrep/.venv/bin:$PATH"
exec docgrep mcp
```

Make it executable and register it; everything after `--` is the server command:

```shell
claude mcp add --transport stdio docgrep -- "$HOME/bin/docgrep-mcp"
```

`claude mcp add` stores the server in the `local` scope by default (this project, only for you); `--scope project` shares it with the repository through `.mcp.json`, and `--scope user` makes it available in all your projects.

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
