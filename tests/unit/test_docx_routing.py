"""Only ``.docx`` among the Word-family extensions reaches the DOCX parser."""

from pathlib import Path
from typing import Any

from docgrep.adapters.corpus import enumerate_corpus, scan_corpus
from docgrep.adapters.parsers import DocxParser
from docgrep.api.container import create_dry_run
from docgrep.domain.models import FileType
from docgrep.settings import Settings

UNSUPPORTED = (".doc", ".docm", ".dotx", ".dotm", ".rtf", ".odt")


def _parsers(root: Path) -> Any:
    """The parsers the composition root registers; building them reads nothing."""
    dry_run: Any = create_dry_run(
        Settings(corpus_root=root, data_dir=root / "data", backend="fake")
    )
    return dry_run.parsers


def test_docx_is_routed_to_the_docx_parser(tmp_path: Path) -> None:
    for name in ("a.docx", "B.DOCX", *(f"x{ext}" for ext in UNSUPPORTED)):
        (tmp_path / name).write_bytes(b"content")

    types = {s.path: s.file_type for s in enumerate_corpus(tmp_path)}
    scanned = {s.path: s.file_type for s in scan_corpus(tmp_path)}
    parsers = _parsers(tmp_path)

    assert types["a.docx"] is FileType.DOCX and types["B.DOCX"] is FileType.DOCX
    assert isinstance(parsers[FileType.DOCX], DocxParser)
    for ext in UNSUPPORTED:
        assert types[f"x{ext}"] is FileType.UNSUPPORTED, ext
        assert scanned[f"x{ext}"] is None, ext


def test_legacy_doc_stays_unsupported(tmp_path: Path) -> None:
    (tmp_path / "letter.doc").write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")

    (source,) = enumerate_corpus(tmp_path)

    assert source.file_type is FileType.UNSUPPORTED
    assert source.file_type not in _parsers(tmp_path)
