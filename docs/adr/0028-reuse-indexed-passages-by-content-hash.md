# ADR-0028: Reuse indexed passages by content hash

**Status:** Accepted
**Date:** 2026-10-18
**Amends:** ADR-0027 (moves are no longer information only).

## Context

A source is identified by its stored path (ADR-0023): its source id is the path with the root prefix, its passage ids derive from the source id and the chunk offset, and its point ids from the passage ids. A file that is moved, renamed or copied, inside a root or to another one, restored from a backup, or a whole backup folder added as a root, is therefore a new source. It went through parsing and embedding again: minutes of GPU time for a large book, and a second full embedding of the library for a backup root.

The index already holds everything such a file needs. A passage's text, location and metadata come from the content, and so does its dense vector; its sparse weights depend on its text and on the average passage length recorded for the index (ADR-0025). ADR-0027 found move pairs, a removed source and an added file with the same content hash and size, and reported them without using them.

Issue #44.

## Decision

**Reuse the stored passages of the same content.** A file whose content is already indexed is indexed by copying the stored passages of that source: the payload, with the source fields of the new file (source id, path, modification time, size, indexing time) and what the parser extracted from the content they share (title, author, year, file type, content hash, parser version) taken from the existing source; the passage id and the point id computed from the new source id and the same chunk offset; the dense and the sparse vector unchanged. Nothing is parsed and no model is called. The store reads the passages of a source with their vectors in pages (one scroll with payload and vectors per page, the size of an upsert batch) and the copies are written with the existing batched upsert.

**When reuse applies.** The existing source is committed with the status `indexed`; its content hash, size and file type equal the file's; its recorded parser version is the version the file would be parsed with now (missing versions counting as `1`, as for an update); its recorded embedding model is the model of the index. Neither of them is an original delegated to its OCR output or a live OCR output (ADR-0017); those are processed normally. The rule is a pure function in the domain, `can_reuse`, with the OCR check beside it in the use cases (`ReuseCheck`).

**What an update reuses.** An added file, or a file whose content changed, can be reused; a retried failed source and a file whose parser changed are processed normally.

- **Move:** a removed source paired with an added file (the pairing of ADR-0027, one to one and deterministic) is a move when it can be reused. The copy is written first, then the old source goes through the normal removal path, as `update` writes a modification before pruning it. A move is one change, counted as `moved` and not in `added` or `removed`, listed under the root of its removed source, and counted against the removal guard of that root like any removal.
- **Copy:** any other added file or content change is a copy of the first source by stored path, of any root, that can be reused and that the update leaves as it is. A source that the same run changes or removes is never copied from: its passages are rewritten or deleted before the copy would be made. Both sources stay indexed, and query-time duplicate collapsing (ADR-0016) shows one result with the other location. Counted as `copied`.

The plan of `docgrep update --plan` and of the update are the same detection, so both show the moves and copies; a pair that cannot be reused is a removal and an addition, and no longer appears as a move.

**A full build** reuses only within its own run: of files with the same content the first is indexed, the next ones are copies of it from the collection being built (a resumed build also from the sources it took over). It never reuses the collection it replaces: a full build is a rebuild. The provisional sparse vectors of copies are re-encoded with the others before the build is published (ADR-0025).

**Failures.** A store error while copying in an update deletes what was written of a new source and records it as `failed:store_error` (a file that was recorded keeps its record, as for a failed modification); the old source of a move is removed only after the copy succeeded. A move whose old source cannot be removed, or whose process stops between the copy and the removal, leaves both indexed; the next update removes the old one, whose file is gone. In a build a failed copy aborts the run, as a failed upsert does, after deleting what it wrote; resume cleans what is left. When the stored passages of the existing source cannot be found, the file is processed normally.

**Record.** The run journal row of a copy names the source its passages were copied from (`reused_from`, a column of the journal only; the committed source record is unchanged). Update totals carry `moved` and `copied`; a build records `copied` when it made a copy.

**Setting.** `DOCGREP_REUSE_BY_HASH` (default `true`); `false` processes every file normally, for comparison and as an escape hatch. The use cases take it as a parameter that is off unless the composition root turns it on.

**Identity stays path based.** Source ids, passage ids and point ids still derive from the path. A saved passage id of a moved source answers `not_found` or `source_removed`, as for any removed source.

## Consequences

Moving, renaming or copying a file costs a scroll and an upsert of its passages instead of parsing and embedding it; on a test server a 10,000-passage source moves in about three seconds. Adding a backup folder as a root copies its files instead of embedding them again, and doubles the passages of the index; collapsing keeps results readable.

A moved file is counted once, as a move, so the counts of the plan change for moves compared with ADR-0027: `added` and `removed` no longer include them. With reuse off no move is shown.

Reuse trusts the index: the stored passages of a source are taken to be what parsing and embedding its content would give with the recorded parser version and the model of the index. That is the same trust an update already places in unchanged sources.

## Alternatives considered

- **Content-based identity (passage ids from the content hash):** moves would need no copy, but every saved passage id, every point id and the per-source operations change, two copies of one file share points, and the index needs a rebuild. Out of scope of #44.
- **Recompute the sparse vectors of copies:** within one index the weights depend on the text and the recorded average length only, so they would come out the same.
- **Reuse across indexes or from the collection a build replaces:** a rebuild exists to recompute everything; an index built with other settings may differ in ways the recorded values do not show.
- **Copy points inside Qdrant without reading them:** Qdrant has no server-side copy of points under new ids; reading and writing in pages keeps memory bounded and uses the write path every other source uses.
- **Reuse retried sources too:** a retry is rare and repeats an earlier failure; keeping it on the normal pipeline keeps the rule to new content.
