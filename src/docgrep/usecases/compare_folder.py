"""Compare a folder with the indexed sources of a root (ADR-0027)."""

from dataclasses import dataclass

from docgrep.domain.errors import ValidationError
from docgrep.domain.protocols import FolderReader, ManifestReader, RootCatalog
from docgrep.domain.roots import CorpusRoots, known_root_names
from docgrep.domain.run_state import latest_root_prefixes


@dataclass(frozen=True, slots=True)
class FolderDiff:
    """A folder against a root, by path relative to the root, sorted.

    ``same``: the indexed source and the file have the same content;
    ``different``: they do not; ``only_in_index`` and ``only_in_folder``:
    the path is on one side only. ``hashed`` counts the files read to tell.
    """

    root: str
    by_hash: bool
    same: tuple[str, ...]
    different: tuple[str, ...]
    only_in_index: tuple[str, ...]
    only_in_folder: tuple[str, ...]
    hashed: int


class CompareFolder:
    """Compare any folder with what the index records for one root.

    Reads only: no lock, no run, nothing written.
    """

    def __init__(
        self, *, manifest: ManifestReader, roots: RootCatalog | None = None
    ) -> None:
        self.manifest = manifest
        self.roots = roots

    def run(
        self, folder: FolderReader, *, root: str | None = None, by_hash: bool = False
    ) -> FolderDiff:
        """Without ``by_hash`` a file whose size and modification time equal
        the record's is the same, as for an update, and only the others are
        hashed; with it every file present on both sides is hashed.

        Without ``root`` the only configured root is compared, or with no
        configured root the legacy root of the index. Raises
        ``ValidationError`` for an unknown root, and without ``root`` when
        several roots are configured.
        """
        catalog = CorpusRoots() if self.roots is None else self.roots.roots()
        roots = catalog.with_recorded(latest_root_prefixes(self.manifest))
        records = self.manifest.committed_sources()
        known = known_root_names(roots, (r.source.source_id for r in records))
        name = _chosen_root(root, roots, known)
        prefix = "" if name == roots.legacy_name else f"{name}/"
        indexed = {
            r.source.path.removeprefix(prefix): r.source
            for r in records
            if roots.root_of(r.source.path) == name
        }
        same: list[str] = []
        different: list[str] = []
        only_in_folder: list[str] = []
        hashed = 0
        seen: set[str] = set()
        for item in folder.scan():
            if item.excluded_by is not None:
                continue
            seen.add(item.path)
            source = indexed.get(item.path)
            if source is None:
                only_in_folder.append(item.path)
                continue
            if not by_hash and (item.size, item.mtime) == (source.size, source.mtime):
                same.append(item.path)
                continue
            hashed += 1
            if folder.content_hash(item.path) == source.content_hash:
                same.append(item.path)
            else:
                different.append(item.path)
        return FolderDiff(
            root=name,
            by_hash=by_hash,
            same=tuple(sorted(same)),
            different=tuple(sorted(different)),
            only_in_index=tuple(sorted(set(indexed) - seen)),
            only_in_folder=tuple(sorted(only_in_folder)),
            hashed=hashed,
        )


def _chosen_root(name: str | None, roots: CorpusRoots, known: tuple[str, ...]) -> str:
    """The root to compare with: the one named; or else the only configured
    root, or without any configured root the legacy root of the index."""
    if name is not None:
        if name not in known:
            raise ValidationError(
                f"unknown root {name!r}; known roots: {', '.join(known) or 'none'}",
                details={"field": "root", "name": name, "known": list(known)},
            )
        return name
    if len(roots.names) == 1:
        return roots.names[0]
    if not roots.names:
        if roots.legacy_name in known:
            return roots.legacy_name
        raise ValidationError(
            "no corpus roots configured or indexed",
            details={"field": "root", "reason": "no_roots"},
        )
    raise ValidationError(
        f"several roots are configured ({', '.join(roots.names)}); choose one "
        "with --root NAME",
        details={
            "field": "root",
            "reason": "ambiguous_root",
            "configured": list(roots.names),
        },
    )
