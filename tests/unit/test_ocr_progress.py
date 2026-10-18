"""OCR progress by pages: the events of the observer and the ETA of a file."""

from collections.abc import Iterator
from contextlib import contextmanager

from docgrep.domain.progress import ProgressEvent
from docgrep.usecases.discover_scans import DiscoverScans
from docgrep.usecases.process_scans import OcrFileResult, OcrRequest, ProcessScans
from fakes import InMemoryManifest, InMemoryOcrRegistry
from ocr_support import NATIVE_PAGE, FakeConverter, FakeCorpus, FakeEngine

SECONDS_PER_PAGE = 10.0


class Lock:
    def is_held(self) -> bool:
        return False

    @contextmanager
    def hold(self, operation: str = "build") -> Iterator[None]:
        yield


class Clock:
    def __init__(self) -> None:
        self.now = 500.0

    def __call__(self) -> float:
        return self.now


def _run(
    files: dict[str, int],
    *,
    sizes: dict[str, int] | None = None,
    page: str | None = None,
) -> tuple[list[ProgressEvent], list[OcrFileResult]]:
    corpus = FakeCorpus()
    for path, pages in files.items():
        data = b"x" * (sizes or {}).get(path, 10)
        corpus.add(path, [page] * pages, data=data)
    clock = Clock()
    engine = FakeEngine(corpus)
    engine.on_run = lambda source: setattr(
        clock, "now", clock.now + SECONDS_PER_PAGE * files[source]
    )
    use_case = ProcessScans(
        lock=Lock(),
        manifest=InMemoryManifest(),
        scanner=corpus,
        discovery=DiscoverScans(
            scanner=corpus, probe=corpus, density_lower=100, density_upper=500
        ),
        registry=InMemoryOcrRegistry(
            exists=corpus.exists, content_hash=corpus.content_hash
        ),
        engine=engine,
        converter=FakeConverter(corpus),
        files=corpus,
        languages="eng",
        poll_seconds=0,
        clock=clock,
    )
    events: list[ProgressEvent] = []
    results: list[OcrFileResult] = []
    use_case.run(
        OcrRequest(paths=tuple(files)),
        on_file=results.append,
        on_progress=events.append,
    )
    return events, results


def test_the_bar_counts_pages_and_the_eta_follows_them() -> None:
    # The short file is the big one in bytes; bytes would predict almost no
    # time left after it, pages predict three times what has passed.
    events, results = _run(
        {"a.pdf": 10, "b.pdf": 30}, sizes={"a.pdf": 1_000_000, "b.pdf": 10}
    )

    assert all(e.operation == "ocr" and e.unit == "pages" for e in events)
    assert all(e.total == 40 for e in events)
    assert results[0].eta_seconds == 300.0
    assert results[-1].eta_seconds == 0.0
    after_first = [e for e in events if e.done == 10 and e.label == "a.pdf"]
    assert after_first and after_first[-1].eta_seconds == 300.0


def test_events_name_the_file_by_its_last_path_component_and_end_the_phase() -> None:
    events, _ = _run({"books/a.pdf": 4, "books/b.pdf": 6})

    assert events[0].done == 0 and events[0].label is None
    assert {e.label for e in events} == {None, "a.pdf", "b.pdf"}
    assert [e.done for e in events][-1] == 10
    assert events[-1].finished and events[-1].label is None
    assert not any(e.finished for e in events[:-1])
    assert events[-1].passages is None and events[-1].rate is None


def test_no_event_without_work() -> None:
    events, _ = _run({"native.pdf": 3}, page=NATIVE_PAGE)

    assert events == []
