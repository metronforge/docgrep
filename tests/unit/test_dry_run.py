"""The dry-run use case over fake ports: aggregation, scoping, sampling."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from docgrep.domain.errors import ValidationError
from docgrep.domain.models import (
    FileType,
    ParsedDocument,
    ParsedSection,
    Passage,
    ScannedFile,
    Source,
    make_passage_id,
)
from docgrep.usecases.dry_run import DryRun


def _file(path: str, size: int, excluded_by: str | None = None) -> ScannedFile:
    extension = Path(path).suffix.lower()
    types = {
        ".txt": FileType.TXT,
        ".md": FileType.MARKDOWN,
        ".pdf": FileType.PDF,
        ".epub": FileType.EPUB,
        ".docx": FileType.DOCX,
        ".djvu": FileType.DJVU,
    }
    return ScannedFile(path, size, extension, types.get(extension), excluded_by)


class Scanner:
    def __init__(self, *files: ScannedFile) -> None:
        self.files = tuple(sorted(files, key=lambda f: f.path))
        self.calls = 0

    def scan(self) -> tuple[ScannedFile, ...]:
        self.calls += 1
        return self.files


class Parser:
    """Parses by name: scan-* needs OCR, broken-* fails, others give one section."""

    def parse(self, source: Source) -> ParsedDocument:
        name = Path(source.path).name
        if name.startswith("broken"):
            raise ValueError("unreadable")
        if name.startswith("scan"):
            return ParsedDocument(source, (), True)
        return ParsedDocument(source, (ParsedSection("body"),))


class FixedChunker:
    """Every parsed document yields a number of passages taken from its size."""

    def __init__(self, per_kb: int = 1) -> None:
        self.per_kb = per_kb

    def chunk(self, document: ParsedDocument) -> tuple[Passage, ...]:
        count = max(1, (document.source.size or 0) // 1000 * self.per_kb)
        return tuple(
            Passage(
                make_passage_id(document.source.source_id, i),
                document.source.source_id,
                "t",
                i,
                1,
            )
            for i in range(count)
        )


PARSERS: Mapping[FileType, Parser] = {
    FileType.TXT: Parser(),
    FileType.MARKDOWN: Parser(),
    FileType.PDF: Parser(),
}


def _dry_run(*files: ScannedFile) -> tuple[DryRun, Scanner]:
    scanner = Scanner(*files)
    return DryRun(scanner=scanner, parsers=PARSERS, chunker=FixedChunker()), scanner


MIXED = (
    _file("a.txt", 1000),
    _file("b.txt", 3000),
    _file("notes/c.md", 500),
    _file("books/one.pdf", 10_000),
    _file("books/two.pdf", 30_000),
    _file("books/three.epub", 700),
    _file("docs/x.docx", 900),
    _file("old/y.djvu", 1200),
    _file("img/p1.png", 100),
    _file("img/p2.png", 300),
    _file("data.bin", 50),
    _file("README", 20),
    _file("drafts/d1.md", 40, "drafts/**"),
    _file("drafts/d2.txt", 60, "drafts/**"),
    _file("scratch.tmp", 5, "not_matched_by_includes"),
)


def test_totals_and_per_type_statistics() -> None:
    dry_run, _ = _dry_run(*MIXED)

    report = dry_run.run().as_dict()

    assert report["totals"] == {
        "files": 15,
        "size_bytes": sum(f.size for f in MIXED),
        "included": 12,
        "excluded": 3,
    }
    types = {entry["type"]: entry for entry in report["types"]}
    assert types["txt"] == {
        "type": "txt",
        "status": "supported",
        "files": 2,
        "size_bytes": 4000,
        "average_size_bytes": 2000,
        "largest_size_bytes": 3000,
    }
    assert types["pdf"]["files"] == 2 and types["pdf"]["largest_size_bytes"] == 30_000
    assert types["pdf"]["average_size_bytes"] == 20_000
    assert [entry["type"] for entry in report["types"]] == sorted(types)
    assert report["would_be_indexed"] == {"files": 5, "size_bytes": 44_500}


def test_recognised_types_without_a_parser_are_reported_separately() -> None:
    dry_run, _ = _dry_run(*MIXED)

    report = dry_run.run().as_dict()

    types = {entry["type"]: entry for entry in report["types"]}
    for name in ("epub", "docx", "djvu"):
        assert types[name]["status"] == "recognised_without_parser"
    assert report["recognised_without_parser"] == {
        "files": 3,
        "size_bytes": 700 + 900 + 1200,
        "types": ["djvu", "docx", "epub"],
    }
    assert report["unsupported"]["files"] == 4
    assert report["would_be_indexed"]["files"] == 5


def test_unsupported_files_get_an_extension_histogram_capped_at_twenty() -> None:
    files = [_file(f"misc/f{i}.ext{i % 30}", 10) for i in range(120)]
    files += [_file("noext", 1), _file("a.PNG", 1), _file("b.png", 1)]
    dry_run, _ = _dry_run(*files)

    unsupported = dry_run.run().as_dict()["unsupported"]

    histogram = unsupported["extensions"]
    assert len(histogram) == 20
    counts = [entry["files"] for entry in histogram]
    assert counts == sorted(counts, reverse=True)
    assert unsupported["files"] == 123
    names = [entry["extension"] for entry in histogram]
    assert names == sorted(
        names,
        key=lambda n: (-dict((e["extension"], e["files"]) for e in histogram)[n], n),
    )


def test_extension_histogram_counts_case_insensitively_and_names_no_extension() -> None:
    dry_run, _ = _dry_run(_file("a.PNG", 1), _file("b.png", 1), _file("README", 1))

    histogram = dry_run.run().as_dict()["unsupported"]["extensions"]

    assert histogram == [
        {"extension": ".png", "files": 2},
        {"extension": "(none)", "files": 1},
    ]


def test_excluded_files_are_counted_per_matching_pattern() -> None:
    dry_run, _ = _dry_run(*MIXED)

    excluded = dry_run.run().as_dict()["excluded"]

    assert excluded["files"] == 3 and excluded["size_bytes"] == 105
    assert excluded["by_pattern"] == [
        {"pattern": "drafts/**", "files": 2},
        {"pattern": "not_matched_by_includes", "files": 1},
    ]


def test_an_empty_corpus_gives_a_report_of_zeros() -> None:
    dry_run, _ = _dry_run()

    report = dry_run.run().as_dict()

    assert report["totals"] == {
        "files": 0,
        "size_bytes": 0,
        "included": 0,
        "excluded": 0,
    }
    assert report["types"] == [] and report["excluded"]["by_pattern"] == []
    assert report["files"]["items"] == [] and report["files"]["total"] == 0
    assert dry_run.run(sample=5).as_dict()["sample"]["types"] == []


def test_the_path_prefix_restricts_every_figure() -> None:
    dry_run, _ = _dry_run(*MIXED)

    report = dry_run.run(path_prefix="books/").as_dict()

    assert report["scope"] == {"path_prefix": "books/"}
    assert report["totals"]["files"] == 3
    assert {e["type"] for e in report["types"]} == {"pdf", "epub"}
    assert [i["path"] for i in report["files"]["items"]] == [
        "books/one.pdf",
        "books/three.epub",
        "books/two.pdf",
    ]


def test_the_file_list_is_paginated_with_a_status_per_file() -> None:
    dry_run, _ = _dry_run(*MIXED)
    everything = dry_run.run(limit=500).as_dict()["files"]["items"]

    page = dry_run.run(limit=4, offset=2).as_dict()["files"]

    assert page["total"] == 15 and page["offset"] == 2 and page["limit"] == 4
    assert page["items"] == everything[2:6]
    statuses = {item["path"]: item["status"] for item in everything}
    assert statuses["a.txt"] == "indexed"
    assert statuses["books/three.epub"] == "recognised_without_parser"
    assert statuses["data.bin"] == "unsupported"
    assert statuses["drafts/d1.md"] == "excluded"
    item = next(i for i in everything if i["path"] == "drafts/d1.md")
    assert item == {
        "path": "drafts/d1.md",
        "size_bytes": 40,
        "extension": ".md",
        "type": "markdown",
        "status": "excluded",
        "excluded_by": "drafts/**",
    }
    assert dry_run.run(limit=4, offset=99).as_dict()["files"]["items"] == []
    assert "files" not in dry_run.run().as_dict(with_files=False)


@pytest.mark.parametrize(
    ("arguments", "bound"),
    [
        ({"limit": 0}, "limit"),
        ({"limit": 501}, "limit"),
        ({"limit": -1}, "limit"),
        ({"offset": -1}, "offset"),
        ({"sample": 0}, "sample"),
        ({"sample": 51}, "sample"),
        ({"path_prefix": ""}, "path_prefix"),
        ({"path_prefix": "/etc"}, "path_prefix"),
        ({"path_prefix": "a/../b"}, "path_prefix"),
    ],
)
def test_invalid_parameters_raise_structured_validation_errors(
    arguments: dict[str, Any], bound: str
) -> None:
    dry_run, scanner = _dry_run(*MIXED)

    with pytest.raises(ValidationError) as raised:
        dry_run.run(**arguments)

    assert raised.value.category == "validation_error"
    assert bound in str(raised.value) or bound in raised.value.details
    assert scanner.calls == 0


def test_the_documented_bounds_are_accepted() -> None:
    dry_run, _ = _dry_run(*MIXED)

    for limit in (1, 500):
        assert dry_run.run(limit=limit).limit == limit
    for sample in (1, 50):
        assert dry_run.run(sample=sample).sample_requested == sample


def _many() -> list[ScannedFile]:
    files = [_file(f"t/f{i:02}.txt", 2000) for i in range(30)]
    files += [_file(f"m/f{i:02}.md", 1000) for i in range(10)]
    files += [_file(f"p/f{i:02}.pdf", 5000) for i in range(10)]
    files += [_file("e/book.epub", 100), _file("drafts/x.txt", 5, "drafts/**")]
    return files


def _sampled(report: dict[str, object]) -> dict[str, list[str]]:
    sample = report["sample"]
    assert isinstance(sample, dict)
    return {t["type"]: t["sampled_files"] for t in sample["types"]}


def test_sampling_is_proportional_over_the_supported_types() -> None:
    dry_run, _ = _dry_run(*_many())

    sampled = _sampled(dry_run.run(sample=10).as_dict())

    assert {k: len(v) for k, v in sampled.items()} == {
        "markdown": 2,
        "pdf": 2,
        "txt": 6,
    }
    assert all(p.startswith("t/") for p in sampled["txt"])
    flat = [p for paths in sampled.values() for p in paths]
    assert "e/book.epub" not in flat and "drafts/x.txt" not in flat


def test_sampling_is_repeatable_for_a_seed_and_differs_between_seeds() -> None:
    dry_run, _ = _dry_run(*_many())

    first = _sampled(dry_run.run(sample=8, seed=7).as_dict())
    again = _sampled(dry_run.run(sample=8, seed=7).as_dict())
    other = _sampled(dry_run.run(sample=8, seed=8).as_dict())
    default = _sampled(dry_run.run(sample=8).as_dict())

    assert first == again
    assert first != other
    assert default == _sampled(dry_run.run(sample=8, seed=0).as_dict())
    for paths in first.values():
        assert paths == sorted(paths)


def test_the_sample_is_capped_by_the_available_files() -> None:
    dry_run, _ = _dry_run(_file("a.txt", 1000), _file("b.md", 1000))

    sampled = _sampled(dry_run.run(sample=50).as_dict())

    assert sampled == {"markdown": ["b.md"], "txt": ["a.txt"]}


def test_the_sample_reports_rates_and_extrapolates_by_count_and_by_size() -> None:
    files = [_file(f"t/f{i}.txt", 2000) for i in range(10)]
    dry_run, _ = _dry_run(*files)

    sample = dry_run.run(sample=4).as_dict()["sample"]

    [entry] = sample["types"]
    assert (entry["parsed"], entry["failed"]) == (4, 0)
    assert entry["passages_per_file"] == 2.0
    assert entry["passages_per_mb"] == 1000.0
    assert entry["requires_ocr_fraction"] == 0.0
    assert entry["parse_chunk_seconds_per_file"] >= 0
    estimate = entry["estimate"]
    assert estimate["corpus_files"] == 10 and estimate["corpus_size_bytes"] == 20_000
    assert estimate["by_file_count"]["passages"] == 20.0
    assert estimate["by_size"]["passages"] == 20.0
    assert "extrapolated" in estimate["note"] and "estimates" in sample["note"]
    assert sample["requested"] == 4 and sample["seed"] == 0


def test_the_two_estimates_differ_when_sampled_files_are_not_typical() -> None:
    files = [_file("t/small.txt", 1000)] + [
        _file(f"t/big{i}.txt", 9000) for i in range(3)
    ]
    dry_run, _ = _dry_run(*files)

    sample = dry_run.run(sample=1, seed=0).as_dict()["sample"]

    [entry] = sample["types"]
    assert entry["sampled_files"] and len(entry["sampled_files"]) == 1
    by_count = entry["estimate"]["by_file_count"]["passages"]
    by_size = entry["estimate"]["by_size"]["passages"]
    assert by_count is not None and by_size is not None
    sampled_size = {f.path: f.size for f in files}[entry["sampled_files"][0]]
    expected_by_count = 4 * (sampled_size // 1000)
    assert by_count == expected_by_count
    assert by_size == pytest.approx(
        sum(f.size for f in files) / sampled_size * (sampled_size // 1000)
    )


def test_failures_and_ocr_candidates_are_counted_without_skewing_the_averages() -> None:
    files = [
        _file("p/a.pdf", 4000),
        _file("p/scan-1.pdf", 4000),
        _file("p/broken-1.pdf", 4000),
        _file("p/b.pdf", 4000),
    ]
    dry_run, _ = _dry_run(*files)

    [entry] = dry_run.run(sample=4).as_dict()["sample"]["types"]

    assert (entry["parsed"], entry["failed"]) == (3, 1)
    assert entry["requires_ocr_fraction"] == pytest.approx(1 / 3, abs=1e-4)
    assert entry["passages_per_file"] == pytest.approx(8 / 3, abs=0.01)


def test_a_sample_of_only_failures_has_no_rates() -> None:
    dry_run, _ = _dry_run(_file("broken-1.txt", 100), _file("broken-2.txt", 100))

    [entry] = dry_run.run(sample=2).as_dict()["sample"]["types"]

    assert (entry["parsed"], entry["failed"]) == (0, 2)
    assert entry["passages_per_file"] is None
    assert entry["estimate"]["by_size"] == {
        "passages": None,
        "parse_chunk_seconds": None,
    }


def test_without_sampling_no_file_is_parsed() -> None:
    class Exploding:
        def parse(self, source: Source) -> ParsedDocument:
            raise AssertionError("a file was parsed")

    scanner = Scanner(*MIXED)
    dry_run = DryRun(
        scanner=scanner,
        parsers={
            t: Exploding() for t in (FileType.TXT, FileType.MARKDOWN, FileType.PDF)
        },
        chunker=FixedChunker(),
    )

    assert dry_run.run().as_dict()["sample"] is None
    assert scanner.calls == 1


def test_one_off_chunker_costs_are_not_counted_as_parse_time() -> None:
    import time

    class SlowFirstChunker(FixedChunker):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def chunk(self, document: ParsedDocument) -> tuple[Passage, ...]:
            self.calls += 1
            if self.calls == 1:
                time.sleep(0.3)  # a tokenizer loading on first use
            return super().chunk(document)

    chunker = SlowFirstChunker()
    dry_run = DryRun(
        scanner=Scanner(_file("a.txt", 1000), _file("b.txt", 1000)),
        parsers=PARSERS,
        chunker=chunker,
    )

    [entry] = dry_run.run(sample=2).as_dict()["sample"]["types"]

    assert entry["parse_chunk_seconds_per_file"] < 0.1
    assert chunker.calls == 3  # the warm-up plus the two sampled files
