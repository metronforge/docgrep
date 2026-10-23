"""Detect what an update would change, without writing anything (ADR-0027),
and which files it can index by reusing stored passages (ADR-0028)."""

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Literal

from docgrep.domain.models import (
    DELEGATED_TO_OCR,
    FileType,
    ManifestSourceRecord,
    ScannedFile,
    Source,
)
from docgrep.domain.protocols import (
    CorpusAvailability,
    CorpusReader,
    DocumentParser,
    ManifestReader,
    OcrDelegations,
    OcrOutputOwners,
    PathScanner,
    RootCatalog,
    parser_reprocess_reason,
    parser_version,
)
from docgrep.domain.roots import CorpusRoots
from docgrep.domain.run_state import latest_root_prefixes
from docgrep.domain.update_plan import (
    Copy,
    GuardedRoot,
    Move,
    RemovalGuard,
    UpdateScope,
    can_reuse,
    guarded_roots,
    pair_moves,
)

ChangeKind = Literal["removed", "modified", "retried", "moved", "copied", "added"]
# Why a recorded source is processed again (the retried ones aside).
ModifiedReason = Literal[
    "content_changed",
    "parser_changed",
    "newly_supported",
    "delegated_to_ocr",
    "delegation_ended",
]
CHANGE_ORDER: tuple[ChangeKind, ...] = (
    "removed",
    "modified",
    "retried",
    "moved",
    "copied",
    "added",
)
MODIFIED_REASONS: tuple[ModifiedReason, ...] = (
    "content_changed",
    "parser_changed",
    "newly_supported",
    "delegated_to_ocr",
    "delegation_ended",
)


@dataclass(frozen=True, slots=True)
class UpdatePlan:
    """What an update found to do, before doing any of it.

    ``retried`` counts sources recorded as failed whose content did not change;
    a failed source whose content changed counts as modified. ``refreshed``
    counts unchanged sources whose size or modification time changed while
    their content hash did not; only their recorded size and time are updated.

    ``modified`` is the sum of the reasons: ``content_changed`` (the content
    hash differs), ``parser_changed`` (the registered parser's version differs
    from the recorded one, or the parser is gone) and ``newly_supported``
    (skipped as unsupported before, a parser exists now), and for OCR
    ``delegated_to_ocr`` (an original now delegated to its OCR'd sibling,
    whose passages are removed) and ``delegation_ended`` (a delegated
    original whose output was removed: it is indexed again).

    ``moved`` counts removed sources whose passages are copied to an added
    file with the same content before they are removed, and ``copied`` the
    added files, and files whose content changed, indexed by copying the
    passages of a source that stays (ADR-0028). Neither is counted in
    ``added``, ``removed`` or ``modified``.
    """

    added: int = 0
    modified: int = 0
    removed: int = 0
    unchanged: int = 0
    retried: int = 0
    refreshed: int = 0
    content_changed: int = 0
    parser_changed: int = 0
    newly_supported: int = 0
    delegated_to_ocr: int = 0
    delegation_ended: int = 0
    # Sources kept because their root is not part of the run, by root name
    # (ADR-0023): they are neither checked nor removed.
    outside_run: tuple[tuple[str, int], ...] = ()
    moved: int = 0
    copied: int = 0

    @property
    def changes(self) -> int:
        return (
            self.added
            + self.modified
            + self.removed
            + self.retried
            + self.moved
            + self.copied
        )


@dataclass(frozen=True, slots=True)
class PlannedChange:
    """One change of an update, in the root it belongs to.

    A move belongs to the root of its removed source.
    """

    kind: ChangeKind
    # The current file for added, modified, retried, moved and copied; the
    # record for removed.
    source: Source
    # The record of the file's path, None when it is not recorded.
    record: ManifestSourceRecord | None
    root: str
    reason: ModifiedReason | None = None
    # Moved and copied: the source whose stored passages are copied; a moved
    # one is removed afterwards.
    reuse: ManifestSourceRecord | None = None


@dataclass(frozen=True, slots=True)
class PendingRefresh:
    """An unchanged source whose recorded size and time follow the file."""

    record: ManifestSourceRecord
    current: Source
    root: str

    @property
    def refreshed(self) -> Source:
        return replace(
            self.record.source, size=self.current.size, mtime=self.current.mtime
        )


@dataclass(frozen=True, slots=True)
class DetectedChanges:
    """Everything an update would do, found without writing anything."""

    # In the order an update applies them.
    changes: tuple[PlannedChange, ...]
    refreshes: tuple[PendingRefresh, ...]
    # The roots of the run, by name, in configured order.
    roots: tuple[str, ...]
    # Per root of the run: the sources the index records, and how many of
    # them stay as they are.
    recorded: Mapping[str, int]
    unchanged: Mapping[str, int]
    outside_run: tuple[tuple[str, int], ...] = ()
    # The root of each recorded source of the run, by stored path.
    source_roots: Mapping[str, str] = field(default_factory=dict)
    # The stored paths of the removed sources that pair with an added file of
    # the same content and size (``pair_moves``), whether or not their
    # passages are reused: they lose no content.
    paired: frozenset[str] = frozenset()

    @property
    def moves(self) -> tuple[Move, ...]:
        """The moves the update applies, in order."""
        return tuple(
            Move(change.reuse.source.path, change.source.path)
            for change in self.changes
            if change.kind == "moved" and change.reuse is not None
        )

    def removals(self) -> dict[str, int]:
        """The planned removals of each root of the run that lose content:
        the unpaired ones. A removed source that pairs with an added file of
        the same content and size (a move) loses nothing and is counted by
        ``paired_removals`` instead (ADR-0029)."""
        counts = dict.fromkeys(self.roots, 0)
        for change in self.changes:
            if change.kind == "removed" and change.source.path not in self.paired:
                counts[change.root] = counts.get(change.root, 0) + 1
        return counts

    def paired_removals(self) -> dict[str, int]:
        """The planned removals of each root of the run that pair with an
        added file, moves included."""
        counts = dict.fromkeys(self.roots, 0)
        for change in self.changes:
            if change.kind == "moved" or (
                change.kind == "removed" and change.source.path in self.paired
            ):
                counts[change.root] = counts.get(change.root, 0) + 1
        return counts

    def guarded(self, guard: RemovalGuard) -> tuple[GuardedRoot, ...]:
        return guarded_roots(
            self.removals(), self.recorded, guard, self.paired_removals()
        )

    def plan(self, root: str | None = None) -> UpdatePlan:
        """The counts of one root, or of the whole run."""
        changes = [c for c in self.changes if root is None or c.root == root]
        kinds = Counter(change.kind for change in changes)
        reasons = Counter(change.reason for change in changes if change.reason)
        unchanged = (
            sum(self.unchanged.values())
            if root is None
            else self.unchanged.get(root, 0)
        )
        return UpdatePlan(
            added=kinds["added"],
            modified=kinds["modified"],
            removed=kinds["removed"],
            unchanged=unchanged,
            retried=kinds["retried"],
            refreshed=sum(1 for r in self.refreshes if root in (None, r.root)),
            **{reason: reasons[reason] for reason in MODIFIED_REASONS},
            outside_run=self.outside_run if root is None else (),
            moved=kinds["moved"],
            copied=kinds["copied"],
        )


@dataclass(frozen=True, slots=True)
class RootPreview:
    """The plan of one root of the run, with the stored paths of each group.

    ``unchanged`` sources are counted only; a move is listed under the root
    of its removed source, a copy under the root of the copy.
    """

    name: str
    plan: UpdatePlan
    recorded: int
    threshold: int
    added: tuple[str, ...]
    modified: tuple[tuple[str, ModifiedReason], ...]
    removed: tuple[str, ...]
    retried: tuple[str, ...]
    refreshed: tuple[str, ...]
    moved: tuple[Move, ...]
    copied: tuple[Copy, ...] = ()
    # The removals the guard counts, and those it does not because they pair
    # with an added file of the same content (moves included).
    unpaired_removals: int = 0
    paired_removals: int = 0


@dataclass(frozen=True, slots=True)
class UpdatePreview:
    """What ``docgrep update`` would do now; nothing was written to get it.

    ``guarded`` lists the roots for which the update would stop at the
    removal guard.
    """

    plan: UpdatePlan
    roots: tuple[RootPreview, ...]
    guarded: tuple[GuardedRoot, ...]

    @classmethod
    def of(cls, detected: DetectedChanges, guard: RemovalGuard) -> "UpdatePreview":
        def paths(kind: ChangeKind, root: str) -> tuple[str, ...]:
            return tuple(
                c.source.path
                for c in detected.changes
                if c.kind == kind and c.root == root
            )

        def reused(kind: ChangeKind, root: str) -> tuple[tuple[str, str], ...]:
            return tuple(
                (c.reuse.source.path, c.source.path)
                for c in detected.changes
                if c.kind == kind and c.root == root and c.reuse is not None
            )

        unpaired, paired = detected.removals(), detected.paired_removals()
        roots = tuple(
            RootPreview(
                name=name,
                plan=detected.plan(name),
                recorded=detected.recorded.get(name, 0),
                threshold=guard.threshold(detected.recorded.get(name, 0)),
                added=paths("added", name),
                modified=tuple(
                    (c.source.path, c.reason)
                    for c in detected.changes
                    if c.kind == "modified" and c.root == name and c.reason
                ),
                removed=paths("removed", name),
                retried=paths("retried", name),
                refreshed=tuple(
                    r.record.source.path for r in detected.refreshes if r.root == name
                ),
                moved=tuple(Move(*pair) for pair in reused("moved", name)),
                copied=tuple(Copy(*pair) for pair in reused("copied", name)),
                unpaired_removals=unpaired.get(name, 0),
                paired_removals=paired.get(name, 0),
            )
            for name in detected.roots
        )
        return cls(detected.plan(), roots, detected.guarded(guard))


def is_delegated(delegations: OcrDelegations | None, source: Source) -> bool:
    """Whether a PDF or DjVu source has a live delegation to its OCR'd
    sibling, for its current content."""
    if delegations is None or source.file_type not in (FileType.PDF, FileType.DJVU):
        return False
    found = delegations.live_delegation(source.path, source.content_hash)
    return found is not None


class ReuseCheck:
    """Whether the stored passages of a recorded source can be copied for a
    file (ADR-0028): ``can_reuse`` with the registered parsers and the model
    of the index, and neither of them is an original delegated to its OCR'd
    sibling or a live OCR output (ADR-0017), which are processed normally.

    The live OCR outputs are read once, when first needed.
    """

    def __init__(
        self,
        *,
        parsers: Mapping[FileType, DocumentParser],
        embedding_model: str,
        delegations: OcrDelegations | None = None,
        ocr_outputs: OcrOutputOwners | None = None,
    ) -> None:
        self.parsers = parsers
        self.embedding_model = embedding_model
        self.delegations = delegations
        self.ocr_outputs = ocr_outputs
        self._outputs: frozenset[str] | None = None

    def __call__(self, record: ManifestSourceRecord, file: Source) -> bool:
        return (
            can_reuse(
                record,
                file,
                parser_version=parser_version(self.parsers.get(file.file_type)),
                embedding_model=self.embedding_model,
            )
            and not self._involved_in_ocr(record.source.path)
            and not self._involved_in_ocr(file.path)
        )

    def _involved_in_ocr(self, path: str) -> bool:
        if self._outputs is None:
            live = {} if self.ocr_outputs is None else self.ocr_outputs.live_outputs()
            self._outputs = frozenset(live)
        if path in self._outputs:
            return True
        return (
            self.delegations is not None
            and self.delegations.live_delegation(path) is not None
        )


class PlanUpdate:
    """Find the changes an update would apply, reading only.

    It takes no lock, starts no run and writes nothing: neither the manifest
    nor the store nor the OCR registry. Run on its own it shows what
    ``docgrep update`` would do; ``UpdateIndex`` applies what it finds.
    """

    def __init__(
        self,
        *,
        corpus: CorpusReader,
        manifest: ManifestReader,
        parsers: Mapping[FileType, DocumentParser],
        delegations: OcrDelegations | None = None,
        roots: RootCatalog | None = None,
        availability: CorpusAvailability | None = None,
        guard: RemovalGuard | None = None,
        ocr_outputs: OcrOutputOwners | None = None,
        reuse_by_hash: bool = False,
        embedding_model: str | None = None,
    ) -> None:
        """With ``reuse_by_hash`` (``DOCGREP_REUSE_BY_HASH``, on by default
        in the application) an added file, or a file whose content changed,
        is planned as a move or a copy when an indexed source has its
        content (ADR-0028). ``embedding_model`` is the model of the
        index; None: the model of the latest build. ``ocr_outputs`` tells
        OCR outputs apart, which are never reused."""
        self.ocr_outputs = ocr_outputs
        self.reuse_by_hash = reuse_by_hash
        self.embedding_model = embedding_model
        self.corpus = corpus
        self.manifest = manifest
        self.parsers = parsers
        self.delegations = delegations
        self.roots = roots
        self.availability = availability
        self.guard = guard or RemovalGuard()

    def run(self, *, retry_failed: bool = True) -> UpdatePreview:
        """The plan of an update with the same roots, per root.

        Raises ``ValidationError`` when a root cannot be read, as an update
        would. The corpus or the index can change before an update runs.
        """
        if self.availability is not None:
            self.availability.require_available()
        roots = (
            None
            if self.roots is None
            else self.roots.roots().with_recorded(latest_root_prefixes(self.manifest))
        )
        return UpdatePreview.of(self.detect(retry_failed, roots), self.guard)

    def detect(
        self,
        retry_failed: bool,
        roots: CorpusRoots | None = None,
        scope: UpdateScope | None = None,
    ) -> DetectedChanges:
        """Compare the roots of the run with the committed sources.

        Only files whose size or modification time differ from the record,
        and added files, are read and hashed. With a ``scope`` only its paths
        are compared and can be added, changed or removed; the other
        recorded sources count as unchanged, and a corpus that can stat
        single files (``PathScanner``) does not walk its roots.
        """
        default = CorpusRoots().legacy_name
        run_roots = (default,) if roots is None else roots.names
        recorded: dict[str, ManifestSourceRecord] = {}
        root_of: dict[str, str] = {}
        outside: dict[str, int] = {}
        committed = self.manifest.committed_sources()
        for entry in committed:
            # Only the roots of the run are compared with the disk; the
            # sources of any other root stay as they are.
            name = default if roots is None else roots.root_of(entry.source.path)
            if roots is None or roots.get(name) is not None:
                recorded[entry.source.path] = entry
                root_of[entry.source.path] = name
            else:
                outside[name] = outside.get(name, 0) + 1
        found: dict[ChangeKind, list[PlannedChange]] = {k: [] for k in CHANGE_ORDER}
        refreshes: list[PendingRefresh] = []
        unchanged: Counter[str] = Counter()
        seen: set[str] = set()

        def change(
            kind: ChangeKind,
            source: Source,
            record: ManifestSourceRecord | None,
            name: str,
            reason: ModifiedReason | None = None,
        ) -> None:
            found[kind].append(PlannedChange(kind, source, record, name, reason))

        for item in self._items(scope):
            if item.excluded_by is not None:
                continue
            seen.add(item.path)
            record = recorded.get(item.path)
            if record is None:
                name = default if roots is None else roots.root_of(item.path)
                change("added", self.corpus.describe(item.path), None, name)
                continue
            name = root_of[item.path]
            current: Source | None = None
            old = record.source
            if item.size != old.size or item.mtime != old.mtime:
                current = self.corpus.describe(item.path)
                if current.content_hash != old.content_hash:
                    change("modified", current, record, name, "content_changed")
                    continue
                # Same content: only the recorded size and time follow the
                # file, when the update applies its plan.
                refreshes.append(PendingRefresh(record, current, name))
            delegation = self._delegation_reason(record, current or old)
            if delegation is not None:
                current = current or self.corpus.describe(item.path)
                change("modified", current, record, name, delegation)
                continue
            if record.status == f"skipped:{DELEGATED_TO_OCR}":
                unchanged[name] += 1
                continue
            reason = self._parser_reason(record)
            if reason is not None:
                current = current or self.corpus.describe(item.path)
                change("modified", current, record, name, reason)
            elif record.status.startswith("failed:") and retry_failed:
                current = current or self.corpus.describe(item.path)
                change("retried", current, record, name)
            else:
                unchanged[name] += 1
        gone = set(recorded) - seen
        if scope is not None:
            for path in set(recorded) - scope.paths:
                unchanged[root_of[path]] += 1
            gone &= scope.paths
        for path in sorted(gone):
            change("removed", recorded[path].source, recorded[path], root_of[path])
        paired = frozenset(
            move.removed
            for move in pair_moves(
                (c.source for c in found["removed"]),
                (c.source for c in found["added"]),
            )
        )
        if self.reuse_by_hash:
            self._plan_reuse(found, committed)
        changes = tuple(
            planned
            for kind in CHANGE_ORDER
            for planned in sorted(found[kind], key=lambda c: c.source.path)
        )
        return DetectedChanges(
            changes=changes,
            refreshes=tuple(refreshes),
            roots=run_roots,
            recorded=Counter(root_of.values()),
            unchanged=unchanged,
            outside_run=tuple(sorted(outside.items())),
            source_roots=root_of,
            paired=paired,
        )

    def _items(self, scope: UpdateScope | None) -> Iterable[ScannedFile]:
        if scope is None:
            return self.corpus.scan()
        if isinstance(self.corpus, PathScanner):
            return self.corpus.scan_paths(scope.paths)
        return [item for item in self.corpus.scan() if item.path in scope.paths]

    def _index_model(self) -> str | None:
        if self.embedding_model is not None:
            return self.embedding_model
        build = self.manifest.latest_run_of_kind("build")
        model = None if build is None else dict(build.config).get("embedding_model")
        return model if isinstance(model, str) else None

    def _plan_reuse(
        self,
        found: dict[ChangeKind, list[PlannedChange]],
        committed: tuple[ManifestSourceRecord, ...],
    ) -> None:
        """Turn the additions and content changes that can reuse stored
        passages into moves and copies, in place (ADR-0028).

        A removed source paired with an added file (``pair_moves``) is a
        move when it can be reused. Any other addition or content change is
        a copy of the first source by stored path that can be reused and
        that the update leaves as it is, whatever its root: the passages of
        a source that is changed or removed in the same run are rewritten
        or deleted before a copy would be made.
        """
        model = self._index_model()
        if model is None:
            return
        reusable = ReuseCheck(
            parsers=self.parsers,
            embedding_model=model,
            delegations=self.delegations,
            ocr_outputs=self.ocr_outputs,
        )

        removed = {c.source.path: c for c in found["removed"]}
        added = {c.source.path: c for c in found["added"]}
        for move in pair_moves(
            (c.source for c in removed.values()), (c.source for c in added.values())
        ):
            old, new = removed[move.removed], added[move.added]
            if old.record is not None and reusable(old.record, new.source):
                found["moved"].append(
                    PlannedChange("moved", new.source, None, old.root, reuse=old.record)
                )
                del removed[move.removed], added[move.added]

        changed = {c.source.path for kind in CHANGE_ORDER for c in found[kind]}
        donors: dict[tuple[str, int], list[ManifestSourceRecord]] = {}
        for record in committed:
            kept = record.source
            if kept.path in changed or kept.content_hash is None or kept.size is None:
                continue
            donors.setdefault((kept.content_hash, kept.size), []).append(record)

        def donor(file: Source) -> ManifestSourceRecord | None:
            if file.content_hash is None or file.size is None:
                return None
            candidates = donors.get((file.content_hash, file.size), ())
            return next((r for r in candidates if reusable(r, file)), None)

        copies = [
            c
            for c in (*added.values(), *found["modified"])
            if c.kind == "added" or c.reason == "content_changed"
        ]
        copied: set[str] = set()
        for planned in copies:
            original = donor(planned.source)
            if original is not None:
                found["copied"].append(
                    PlannedChange(
                        "copied",
                        planned.source,
                        planned.record,
                        planned.root,
                        reuse=original,
                    )
                )
                copied.add(planned.source.path)
        found["removed"] = list(removed.values())
        found["added"] = [c for c in added.values() if c.source.path not in copied]
        found["modified"] = [
            c for c in found["modified"] if c.source.path not in copied
        ]

    def _delegation_reason(
        self, record: ManifestSourceRecord, source: Source
    ) -> ModifiedReason | None:
        """Whether a recorded source with unchanged content became delegated
        to its OCR'd sibling, or stopped being so."""
        delegated = record.status == f"skipped:{DELEGATED_TO_OCR}"
        live = is_delegated(self.delegations, source)
        if live and not delegated:
            return "delegated_to_ocr"
        if delegated and not live:
            return "delegation_ended"
        return None

    def _parser_reason(self, record: ManifestSourceRecord) -> ModifiedReason | None:
        """Why the registered parser would treat the source differently, if so.

        A source skipped as unsupported is newly supported once a parser is
        registered; an indexed or OCR-pending source has a changed parser when
        the parser's version differs from the recorded one or the parser is
        gone. A missing or empty version, recorded or declared, counts as
        version "1" (``effective_parser_version``), so sources recorded before
        versions existed are not reprocessed for that alone. Failed sources are
        left to the retry rule.
        """
        parser = self.parsers.get(record.source.file_type)
        return parser_reprocess_reason(
            record.status,
            record.source.parser_version,
            registered=parser is not None,
            declared_version=parser_version(parser),
        )
