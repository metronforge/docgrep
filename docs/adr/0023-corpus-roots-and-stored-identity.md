# ADR-0023: Corpus roots and the stored identity of their sources

**Status:** Accepted
**Date:** 2026-10-17

## Context

The corpus had one folder, `DOCGREP_CORPUS_ROOT`, and every stored path and `source_id` was the path relative to it (issue #37). Real use needs several folders (a book library and some note folders) indexed and searched together, and one-off folders passed on the command line. Two folders can hold the same relative path, so the relative path alone no longer identifies a source.

The `source_id` cannot simply be renamed for the existing index. `make_passage_id(source_id, chunk_offset)` hashes it, and the vector point id is a uuid5 of the passage id: renaming every `source_id` would change every passage id and point id (about 867,000 passages in the reference library), force a full reindex and invalidate every passage id a client has saved.

## Decision

**Every root has a name; storage identity is separate from it.** A root is `{name, path, legacy}`. Its stored prefix is empty for the **legacy root** and `<name>/` for every other root, and a source's stored path and `source_id` are the prefix followed by its path relative to the root. The legacy root is the one that owns an index built before roots had names, so its sources keep their stored paths, `source_id`, passage ids and point ids unchanged: no reindex and no rewrite of stored data. At most one root is legacy; a fresh installation may have none, and then every root is prefixed. Which root is legacy is configured explicitly and validated, never guessed.

**The root of a source is derived, not stored.** A `source_id` whose first segment names a known prefixed root belongs to that root; anything else belongs to the legacy root (or, when none is configured, to an unconfigured legacy root reported as `_legacy`, a name no configured root can take). The known prefixed roots are the configured ones plus those the index records: every build and update stores the roots it visited and the prefixed root names the index may hold afterwards in its run configuration (`corpus_roots`, `root_prefixes`), so a root that is no longer configured is still recognised. No manifest column is added.

**One helper maps paths.** `CorpusRoots` maps a stored path to the file on disk and back. Enumeration, the parsers, the scan probe, OCR (engine, converter, file operations, registry) and the content hash checks all go through it. A stored path whose root is not configured is "root not available": read-only operations treat it as such and do not fail.

**Configuration is a roots file.** The roots live in `<data_dir>/roots.json` (`{"version": 1, "roots": [{"name", "path", "legacy"}]}`), written atomically by `docgrep roots add | remove` with absolute, resolved paths. There is no environment variable for the list. Names match `[A-Za-z0-9][A-Za-z0-9._-]*` and are unique case-insensitively; no root may be inside another (a sub-folder of a root is a path prefix filter, not a root); and a prefixed root may not be named like a top-level entry of the legacy root, whose `source_id` values it would collide with. These rules are checked when a root is added and again when the file is read. `DOCGREP_CORPUS_ROOT` remains only as a deprecated bootstrap: without a roots file it is one legacy root, with a warning; with a roots file it is ignored, with a warning.

**Parameter roots are one-off.** `--root NAME=PATH` on `index`, `update`, `dry-run`, `discover-scans` and `ocr` replaces the saved list for that run. A name the roots file holds must point to the same folder and keeps its legacy flag; a new name is always prefixed. Nothing is written to the roots file unless the user accepts the offer to save it (or passes `--save-roots`). What a run indexed stays in the index; only later updates do not visit an unsaved root. Neither MCP nor HTTP can add or choose roots, so an AI client never picks the folders that get indexed.

**An unreadable root is an error, never an empty root.** Indexing, update and OCR check every root of the run before any work and fail naming the root that is missing, not a directory or not readable. Search, browse, context and status read only the manifest and the store and keep working.

**Sources of absent roots are kept and reported.** An update compares only the roots of the run with the disk: a source is removed only when its root is part of the run, readable, and the file is gone. Sources of a root that is not configured, or not passed to the run, stay untouched; the update plan and `docgrep status` report them per root. A full `index` replaces the whole index with the sources of its roots, so it refuses while the index holds sources of another root. `docgrep forget-root NAME --yes` removes the sources of a prefixed root through the removal path of an update; the legacy root cannot be forgotten this way, because its unprefixed sources cannot be told apart by path.

**OCR stays inside a root.** An OCR output is written next to its original, so it has the same root and prefix. Delegation across roots therefore cannot arise and is not supported.

## Consequences

Existing indexes keep working unchanged with the old variable or with the old folder saved as the legacy root, and new roots are added with `docgrep update` without a reindex. Search and browse gain a `roots` filter and report each source's root. The legacy root keeps unprefixed paths forever, so its name never appears in its `source_id` values. Renaming a prefixed root changes its sources' identity: they are then an unconfigured root until forgotten, and the renamed root is indexed again.

## Alternatives considered

- **Prefix every root, including the existing one:** re-keys every passage and point id, needs a full reindex and breaks saved passage ids.
- **Store a root column in the manifest:** a schema migration for a fact the prefix already carries; it would also not change the identity problem.
- **A JSON list in an environment variable:** hard to edit and to validate, and invisible to the commands that should manage it.
- **Treat an unmounted root as empty:** an update would delete its sources from the index.
