# ADR-0027: The update plan, the removal guard and diff

**Status:** Accepted
**Date:** 2026-10-18

## Context

`docgrep update` (ADR-0013) found its changes and applied them in one go, without a way to see them first. Its planning also wrote: a file whose size or modification time had changed while its content hash had not got its recorded size and time refreshed, in the store and the manifest, while the changes were still being found. So there was no read-only way to ask what an update would do.

Nothing limited what an update removed either. A root folder that is not mounted, a folder emptied for a moment by a sync client, a root saved with the wrong path, or new exclude patterns, all look like files that were deleted, and the update removed their sources. Rebuilding a large library takes hours. A root that cannot be read at all already stops the update (ADR-0023), but an empty or wrong folder can be read.

Finally, there was no way to compare another folder, such as a backup copy, with what the index holds; it was done by hand with `rsync -n`.

Issue #43. Automatic sync (#45) will run updates unattended and needs both a plan it can inspect and a limit on removals.

## Decision

**Planning writes nothing.** The detection of an update is a use case of its own (`PlanUpdate`). It reads the committed sources, the corpus (a stat-only scan, plus a content hash for every added file and every file whose size or time differs), the parser versions and the OCR delegations, and returns the ordered changes, the plan and the pending refreshes. `UpdateIndex` runs it under the index lock and then applies what it found: the refreshes first, exactly as before, then the changes in order. The end state of an update is unchanged.

**`docgrep update --plan`** runs the same detection without the lock, without starting a run and without opening the store, the manifest and the OCR registry being opened read-only, and prints the plan per root, with the stored paths of each group on request and as JSON with fixed keys. It may run while another operation holds the lock, so it says that the plan can change before an update runs.

**Moves are information.** A removed source and an added file with the same content hash and size form a move pair. Removed sources are taken in stored path order, and each gets the first unpaired added file in stored path order, so pairs are one to one and deterministic. Moves are reported, and still counted and processed as a removal and an addition; reusing their passages is #44.

**The removal guard.** Before anything is written, the planned removals of each root of the run (moves and removals caused by the patterns included) are compared with `min(ratio × the sources the index records for the root, count)`, rounded down, from `DOCGREP_UPDATE_GUARD_RATIO` (default 0.05) and `DOCGREP_UPDATE_GUARD_COUNT` (default 50); a root that records fewer than 20 sources is compared with the count alone. Whatever its size and the thresholds, an update that would remove every source a root records is refused too: an existing but empty folder (a mount point without its disk, a folder a sync client emptied) passes the check that a root can be read. Each guarded root carries its reason, `empties_root` or `threshold`. When refused, the update writes nothing, not even the refreshes or a run record, reports the plan and the guarded roots and fails with the category `removal_guard`. `docgrep update --yes` applies the plan anyway. The decision is a pure function in the domain, so that sync can apply the same rule. `forget-root`, which removes a root by design and has its own confirmation, is not guarded.

**`docgrep diff PATH`** compares any folder with the sources the index records for one root, by path relative to the root: only in the index, only in the folder, different, same. The folder is listed with the include and exclude patterns, as a root would be. By default equal size and modification time count as the same, as in the update pre-filter, and only the other files are hashed; `--hash` hashes every file. It reads only and takes no lock. Without `--root` the only configured root is used, or without any configured root the legacy root of the index; several configured roots are a usage error.

## Consequences

An update can be inspected before it runs, and an update that would remove a large part of a root stops instead. A legitimate mass removal, such as deleting a whole collection, now needs `--yes`. The guard counts removals only: a root replaced by another folder with as many files still passes it, but its plan shows the additions. A small root (under 20 sources) can lose most of its sources without `--yes` while the count allows it, but never all of them.

Planning and applying are now separate steps under the same lock, so the plan an update prints is the plan it applies. A detection error is raised before a run is recorded; previously such an update was recorded as aborted.

`--plan` and `diff` hash the same files an update would, so on a large library with many touched files they take as long as an update's detection.

## Alternatives considered

- **Keep refreshing during planning and skip it only for `--plan`:** two code paths for one detection, and an update stopped by the guard would still have written.
- **A guard on the total of all roots:** a small root emptied next to a large one would pass.
- **A ratio alone:** a large library could lose thousands of sources at 5%; the count caps it.
- **Pairing moves by file name as well:** a renamed file is a move too, and content plus size is what #44 needs to reuse passages.
- **`diff` through the update plan with a one-off root:** it would hash added files and mix in parser and OCR rules that have nothing to do with a copy of the files.
