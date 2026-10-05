"""Scan discovery through fakes at the scanner and probe boundaries (UC-5)."""

from collections.abc import Sequence
from typing import Any

import pytest

from docgrep.domain.errors import InspectionFailed, ValidationError
from docgrep.domain.models import FileType, ScannedFile
from docgrep.domain.scans import PageSample
from docgrep.usecases.discover_scans import DiscoverScans

_TYPES = {".pdf": FileType.PDF, ".djvu": FileType.DJVU, ".djv": FileType.DJVU}


def _file(path: str, size: int = 1000, excluded_by: str | None = None) -> ScannedFile:
    extension = path[path.rfind(".") :] if "." in path else ""
    return ScannedFile(path, size, extension, _TYPES.get(extension, None), excluded_by)


class Scanner:
    def __init__(self, *files: ScannedFile) -> None:
        self.files = files

    def scan(self) -> tuple[ScannedFile, ...]:
        return self.files


class Probe:
    """Pages per file as character counts (negative: dropped by R1; None: the
    page fails), or an exception for the whole file."""

    def __init__(self, files: dict[str, Any]) -> None:
        self.files = files
        self.calls: list[tuple[str, tuple[int, ...]]] = []

    def page_count(self, path: str, file_type: FileType) -> int:
        answer = self.files[path]
        if isinstance(answer, Exception):
            raise answer
        return len(answer)

    def sample(
        self, path: str, file_type: FileType, pages: Sequence[int]
    ) -> tuple[PageSample | None, ...]:
        self.calls.append((path, tuple(pages)))
        counts = self.files[path]
        return tuple(
            None
            if counts[page - 1] is None
            else PageSample(page, abs(counts[page - 1]), counts[page - 1] < 0)
            for page in pages
        )


def _run(
    files: dict[str, Any], *scanned: ScannedFile, **filters: Any
) -> dict[str, Any]:
    scanner = Scanner(*(scanned or tuple(_file(path) for path in files)))
    use_case = DiscoverScans(
        scanner=scanner, probe=Probe(files), density_lower=100, density_upper=500
    )
    return use_case.run(**filters).as_dict()


def _entry(report: dict[str, Any], path: str) -> dict[str, Any]:
    (entry,) = [e for e in report["files"] if e["path"] == path]
    return entry


NATIVE = [2000] * 10
SCAN = [0] * 10


def test_scans_are_listed_and_native_files_only_counted() -> None:
    report = _run({"native.pdf": NATIVE, "scan.pdf": SCAN, "mixed.djvu": [300] * 10})

    assert [e["path"] for e in report["files"]] == ["mixed.djvu", "scan.pdf"]
    scan = _entry(report, "scan.pdf")
    assert scan == {
        "path": "scan.pdf",
        "file_type": "pdf",
        "size_bytes": 1000,
        "page_count": 10,
        "classification": "scanned",
        "reason": "low_text_density",
        "hint": None,
        "ocr_status": "not_processed",
        "large_file": False,
        "text_density": 0.0,
        "sampled_pages": list(scan["sampled_pages"]),
    }
    assert len(scan["sampled_pages"]) == 6
    assert _entry(report, "mixed.djvu")["classification"] == "mixed_or_uncertain"
    assert report["totals"] == {
        "inspected": 3,
        "native": 1,
        "scanned": 1,
        "uncertain": 1,
        "inspection_failed": 0,
    }
    assert report["by_file_type"]["pdf"]["native"] == 1
    assert report["by_file_type"]["djvu"]["uncertain"] == 1


def test_broken_pages_count_as_zero_and_make_a_garbled_scan() -> None:
    report = _run({"broken.pdf": [-3000] * 10})

    entry = _entry(report, "broken.pdf")
    assert (entry["classification"], entry["reason"]) == (
        "scanned",
        "garbled_text_layer",
    )
    assert entry["text_density"] == 0.0


def test_blank_pages_do_not_tip_a_text_heavy_file() -> None:
    pages = [1800, 0, 1700, 0, 1900, 0, 1600, 0, 0, 1500]

    report = _run({"book.pdf": pages})

    assert report["files"] == [] and report["totals"]["native"] == 1


@pytest.mark.parametrize(
    ("error", "reason", "hint"),
    [
        (InspectionFailed("x", details={"reason": "encrypted"}), "encrypted", None),
        (
            InspectionFailed("x", details={"reason": "corrupt_header"}),
            "corrupt_header",
            None,
        ),
        (
            InspectionFailed("x", details={"reason": "djvu_invalid"}),
            "djvu_invalid",
            None,
        ),
        (
            InspectionFailed(
                "x", details={"reason": "djvu_tools_missing", "hint": "install it"}
            ),
            "djvu_tools_missing",
            "install it",
        ),
        (RuntimeError("unexpected"), "corrupt_header", None),
    ],
)
def test_failures_are_reported_and_the_scan_continues(
    error: Exception, reason: str, hint: str | None
) -> None:
    report = _run({"bad.pdf": error, "scan.pdf": SCAN})

    bad = _entry(report, "bad.pdf")
    assert (bad["classification"], bad["reason"], bad["hint"]) == (
        "inspection_failed",
        reason,
        hint,
    )
    assert bad["page_count"] is None
    assert _entry(report, "scan.pdf")["classification"] == "scanned"
    assert report["totals"]["inspection_failed"] == 1


def test_failing_pages_are_skipped_and_all_failing_is_unsamplable() -> None:
    some = [2000, None, 2000, 2000, None, 2000, 2000, 2000, 2000, 2000]
    report = _run({"some.pdf": some, "none.pdf": [None] * 10, "empty.pdf": []})

    assert report["totals"]["native"] == 1
    for path, count in (("none.pdf", 10), ("empty.pdf", 0)):
        entry = _entry(report, path)
        assert (entry["classification"], entry["reason"]) == (
            "inspection_failed",
            "unsamplable",
        )
        assert entry["page_count"] == count


def test_only_pdf_and_djvu_files_of_the_build_are_inspected() -> None:
    files = {"a.pdf": SCAN, "b.djv": SCAN, "c.djvu": SCAN}
    report = _run(
        files,
        _file("a.pdf"),
        _file("b.djv"),
        _file("c.djvu"),
        _file("notes.txt"),
        _file("book.epub"),
        _file("drafts/x.pdf", excluded_by="drafts/**"),
    )

    assert [e["path"] for e in report["files"]] == ["b.djv", "c.djvu", "a.pdf"]


def test_filters_by_file_type_and_path_glob() -> None:
    files = {"books/a.pdf": SCAN, "books/b.djvu": SCAN, "papers/c.pdf": SCAN}

    pdf_only = _run(files, file_types=["pdf"])
    books = _run(files, path_glob="books/**")
    both = _run(files, path_glob="*.pdf", file_types=["pdf", "djvu"])

    assert [e["path"] for e in pdf_only["files"]] == ["books/a.pdf", "papers/c.pdf"]
    assert list(pdf_only["by_file_type"]) == ["pdf"]
    assert pdf_only["filters"] == {"path_glob": None, "file_types": ["pdf"]}
    assert [e["path"] for e in books["files"]] == ["books/b.djvu", "books/a.pdf"]
    assert [e["path"] for e in both["files"]] == ["books/a.pdf", "papers/c.pdf"]


def test_invalid_filters_and_thresholds_are_refused() -> None:
    with pytest.raises(ValidationError):
        _run({}, file_types=["epub"])
    with pytest.raises(ValidationError):
        _run({}, path_glob=" ")
    with pytest.raises(ValidationError):
        DiscoverScans(
            scanner=Scanner(), probe=Probe({}), density_lower=500, density_upper=100
        )


def test_large_files_are_flagged() -> None:
    files = {"long.pdf": [0] * 5001, "big.pdf": SCAN, "small.pdf": SCAN}

    report = _run(
        files,
        _file("long.pdf"),
        _file("big.pdf", size=1024**3 + 1),
        _file("small.pdf"),
    )

    flags = {e["path"]: e["large_file"] for e in report["files"]}
    assert flags == {"big.pdf": True, "long.pdf": True, "small.pdf": False}


def test_two_runs_give_the_same_report_and_samples() -> None:
    files = {f"f{i}.pdf": [i * 40] * 50 for i in range(10)}

    assert _run(files) == _run(files)
    scanner = Scanner(*(_file(path) for path in files))
    probe = Probe(files)
    use_case = DiscoverScans(
        scanner=scanner, probe=probe, density_lower=100, density_upper=500
    )
    use_case.run()
    first = list(probe.calls)
    use_case.run()
    assert probe.calls[len(first) :] == first


def test_an_empty_corpus_gives_zero_counts() -> None:
    report = _run({})

    assert report["files"] == []
    assert set(report["totals"].values()) == {0}
