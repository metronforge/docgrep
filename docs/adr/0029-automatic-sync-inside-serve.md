# ADR-0029: Automatic sync inside `docgrep serve`

**Status:** Accepted
**Date:** 2026-10-23
**Amends:** ADR-0027 (a removal that pairs with an added file does not count against the removal guard).

## Context

The index followed the corpus only when someone ran `docgrep update`. A library that changes every day (notes, downloads, a synced folder) is out of date until then, and an update that is forgotten is a search that misses the newest files. Issue #45 asks for the index to follow the configured roots by itself (UC-21). The maintainer decided that it runs inside `docgrep serve`.

Running unattended changes what an update must be. Files are seen while they are still being written or downloaded. A root can disappear for a moment (a disk that is not mounted, a sync client that empties a folder). A failed file must not fail again every few minutes. Embedding shares the GPU with search. And the pieces to do it safely already exist: a read-only plan, the removal guard as a pure function (ADR-0027), reuse of stored passages for moves, renames and copies (ADR-0028), and a progress observer on long operations.

## Decision

**Where it runs.** With `DOCGREP_SYNC=true` (default `false`) the lifespan of `docgrep serve` starts one background task and stops it on shutdown: no new cycle, and an applied run finishes the source it is on. The task takes the container that search uses, so the models are loaded once, and asks for it on a worker thread, so the server answers while they load. There is no separate sync process and no sync command, and neither HTTP nor MCP can enable, trigger or configure it. Its state is in memory and is rebuilt by the first cycle, so a restart loses nothing.

**Detection by periodic scan.** Every `DOCGREP_SYNC_INTERVAL_SECONDS` (default 120) the task scans the readable roots with the enumeration and patterns of the update plan, from directory entries alone: a file differs from the index when its size or modification time differs from the record, and it is hashed only later, by the update. No file-system notification library is used, which keeps the dependencies and works on network mounts.

**Settling.** An added or modified file is applied when its size and modification time were identical in two consecutive scans and its modification time is older than `DOCGREP_SYNC_SETTLE_SECONDS` (default 60). A removal is applied when the file was absent in two consecutive scans of a readable root. A built-in ignore list applies on top of the root patterns: `*.part`, `*.crdownload`, `*.tmp`, `~$*`, `.~lock.*#`, `.#*`, `*.swp` and the folders `.obsidian/`, `.git/` and `.trash/`; sources already recorded under such a name are neither changed nor removed. A file that has not settled after 30 minutes is logged once and stays watched.

**Applying.** A cycle with settled changes runs the update use case scoped to their paths (`UpdateScope`): only these paths are compared and changed, every other source counts as unchanged, no other file is added, and the corpus is asked for those files alone (one stat each) instead of being walked, so a root that is gone cannot make the rest look deleted. The run is recorded with the kind `sync`. Moves reuse stored passages (ADR-0028). The cycle only tries the index lock; when another operation holds it the cycle ends and the changes stay pending. Sources are processed one at a time with a short pause after each, so a waiting search gets the models.

**Reconciliation.** When the task starts, every `DOCGREP_SYNC_RECONCILE_HOURS` (default 24) and for a root that is readable again, the readable roots are compared with the index with the update plan itself, which also finds what a scan of directory entries cannot: sources to retry, parsers whose version changed and OCR delegations. What it finds settles at once, except removals (second scan). The target of a move waits for the removal of its old path, so that both are applied in one run and pair up.

**Safety holds.**

- A root that is missing, unmounted or unreadable is paused: nothing in it is scanned, added or removed, and it is compared in full when it is readable again.
- Removals that trip the removal guard (ADR-0027, per root and cycle, as a pure function) are held back: the update leaves them out and names them, while additions, modifications and moves still apply. Once a root has held removals, its later removals are judged together with them. Only a manual `docgrep update --yes` applies held removals; they are dropped when the files reappear.
- A file that failed is retried only when its size or modification time changes, at the daily comparison and at start-up.
- A newly configured root is not synced until its first manual `docgrep update` has completed; sync reports it as awaiting a first update.
- A configuration that drifted from the index, an unfinished build or re-encode, or a missing index keep sync idle, and it says why.
- OCR is never started by sync; the OCR outputs that `docgrep ocr` writes are picked up as added files.

**Moves do not count against the removal guard (amends ADR-0027).** A removed source that pairs with an added file of the same content hash and size (the one-to-one pairing of ADR-0027) loses no content. It is no longer counted against the guard of its old root, whether or not reuse (ADR-0028) applies to it. Removals without a pair count as before, including `empties_root`: a root whose sources all moved elsewhere is not emptied. Without this, sync would hold the removals of a large folder that was only moved, and `docgrep update` would ask for `--yes` for it. The plan and the guard message show the paired and the unpaired counts separately. ADR-0027 itself is unchanged.

**Status.** `docgrep status`, `GET /info` and the MCP `get_index_status` body gain a `sync` section: enabled, running, interval, last scan, last applied run, pending counts by kind, changes still settling, held removals per root, paused roots, roots awaiting a first update, the idle reason, the last error and the progress of an applied run (through the progress observer). `docgrep status` is another process, so the server writes the section to `sync-status.json` in the data directory after every cycle and `docgrep status` reads it (with the process id, to tell a dead server from a live one); `docgrep status` adds a few stored paths of each root with held removals, and MCP never shows a path. One `sync_cycle` log record per cycle carries counts only.

## Consequences

An index of a changing library stays current within a few minutes of the files settling, at the cost of a scan of directory entries every interval (about a quarter of a second for 3,000 files). Changes made while the server is stopped are found at the start.

A deletion that is real but large (a whole collection removed on purpose) is held until someone runs `docgrep update --yes`; that is the intended trade for running unattended. A file that is rewritten more often than the settle time never gets indexed until it rests, and an editor that writes a temporary file and renames it is seen after the second scan.

Moving a large folder is applied as moves (a copy of stored passages each) without the guard asking for confirmation, because nothing is lost. A file that is moved and changed at the same time is an unpaired removal and an addition, and counts against the guard.

The state of sync is in memory; a crash loses the pending list only, and the start-up comparison rebuilds it. The running `serve` holds the local-mode store, so a manual update in local mode needs the server stopped, as before; the calls of a local-mode store are now serialized, because a search and a sync write run in different threads of the server and the local client does not synchronize its arrays.

## Alternatives considered

- **File-system notifications:** miss events on network mounts and in containers, need a dependency per platform, and still need a periodic comparison to be trusted.
- **A separate `docgrep sync` process:** it cannot open the single-process local store and would load a second copy of the models.
- **Applying changes as soon as they are seen:** half-written and half-downloaded files would be indexed and then indexed again.
- **A guard that counts moves too, as in ADR-0027:** a moved folder would be held and need `--yes` in sync and in update, although it loses nothing.
- **Holding every removal until a person confirms:** unattended sync would never delete anything; the guard holds only the removals that look like an accident.
- **Keeping the sync status only in memory:** `docgrep status` could not show it.
