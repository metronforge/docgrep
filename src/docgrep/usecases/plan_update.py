"""Detect what an update would change, without writing anything (ADR-0027)."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Literal

from docgrep.domain.models import (
    DELEGATED_TO_OCR,
    FileType,
    ManifestSourceRecord,
    Source,
)
from docgrep.domain.protocols import (
    CorpusAvailability,
    CorpusReader,
    DocumentParser,
    ManifestReader,
    OcrDelegations,
    RootCatalog,
    parser_reprocess_reason,
    parser_version,
)
from docgrep.domain.roots import CorpusRoots
from docgrep.domain.run_state import latest_root_prefixes
from docgrep.domain.update_plan import (
    GuardedRoot,
    Move,
    RemovalGuard,
    guarded_roots,
    pair_moves,
)

ChangeKind = Literal["removed", "modified", "retried", "added"]
# Why a recorded source is processed again (the retried ones aside).
ModifiedReason = Literal[
    "content_changed",
    "parser_changed",
    "newly_supported",
    "delegated_to_ocr",
    "delegation_ended",
]
CHANGE_ORDER: tuple[ChangeKind, ...] = ("removed", "modified", "retried", "added")
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

    ``moved`` counts removed sources paired with an added file of the same
    content; they are counted in ``removed`` and ``added`` as well, and
    processed as such.
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

    @property
    def changes(self) -> int:
        return self.added + self.modified + self.removed + self.retried


@dataclass(frozen=True, slots=True)
class PlannedChange:
    """One change of an update, in the root it belongs to."""

    kind: ChangeKind
    # The current file for added, modified and retried; the record for removed.
    source: Source
    record: ManifestSourceRecord | None
    root: str
    reason: ModifiedReason | None = None


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
    moves: tuple[Move, ...] = ()
    outside_run: tuple[tuple[str, int], ...] = ()
    # The root of each recorded source of the run, by stored path.
    source_roots: Mapping[str, str] = field(default_factory=dict)

    def removals(self) -> dict[str, int]:
        """The planned removals of each root of the run."""
        counts = dict.fromkeys(self.roots, 0)
        for change in self.changes:
            if change.kind == "removed":
                counts[change.root] = counts.get(change.root, 0) + 1
        return counts

    def guarded(self, guard: RemovalGuard) -> tuple[GuardedRoot, ...]:
        return guarded_roots(self.removals(), self.recorded, guard)

    def plan(self, root: str | None = None) -> UpdatePlan:
        """The counts of one root, or of the whole run."""
        changes = [c for c in self.changes if root is None or c.root == root]
        kinds = Counter(change.kind for change in changes)
        reasons = Counter(change.reason for change in changes if change.reason)
        moves = [m for m in self.moves if root is None or self.root_of(m) == root]
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
            moved=len(moves),
        )

    def root_of(self, move: Move) -> str:
        """A move belongs to the root of its removed source."""
        return self.source_roots[move.removed]


@dataclass(frozen=True, slots=True)
class RootPreview:
    """The plan of one root of the run, with the stored paths of each group.

    ``unchanged`` sources are counted only; a move is listed under the root
    of its removed source.
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
                moved=tuple(m for m in detected.moves if detected.root_of(m) == name),
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
    ) -> None:
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
        self, retry_failed: bool, roots: CorpusRoots | None = None
    ) -> DetectedChanges:
        """Compare the roots of the run with the committed sources.

        Only files whose size or modification time differ from the record,
        and added files, are read and hashed.
        """
        default = CorpusRoots().legacy_name
        run_roots = (default,) if roots is None else roots.names
        recorded: dict[str, ManifestSourceRecord] = {}
        root_of: dict[str, str] = {}
        outside: dict[str, int] = {}
        for entry in self.manifest.committed_sources():
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

        for item in self.corpus.scan():
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
        for path in sorted(set(recorded) - seen):
            change("removed", recorded[path].source, recorded[path], root_of[path])
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
            moves=pair_moves(
                (c.source for c in found["removed"]),
                (c.source for c in found["added"]),
            ),
            outside_run=tuple(sorted(outside.items())),
            source_roots=root_of,
        )

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
