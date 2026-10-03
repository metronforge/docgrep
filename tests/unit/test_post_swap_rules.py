"""Recovery of a run that died after the swap, on the in-memory fakes."""

from pathlib import Path
from typing import Any

import pytest
from test_resume_rules import Crash, Harness

from docgrep.domain.errors import BuildFailed, NothingToResume, StagingInconsistent
from docgrep.domain.models import ManifestRunRecord
from docgrep.domain.run_state import swap_already_happened
from fakes import InMemoryManifest


def _run(status: str, total: int | None, **extra: Any) -> ManifestRunRecord:
    return ManifestRunRecord(
        "1",
        status,
        (),
        total_sources=total,
        **extra,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("status", "total", "rows", "published", "expected"),
    [
        ("running", 5, 5, True, True),
        ("aborted", 5, 7, True, True),
        ("running", 5, 4, True, False),
        ("running", 5, 5, False, False),
        ("completed", 5, 5, True, False),
        ("cancelled", 5, 5, True, False),
        ("swapped", 5, 5, True, False),  # handled as swapped, not by this rule
        ("running", None, 3, True, True),
        ("running", None, 0, True, False),
        ("running", 0, 0, True, True),
    ],
)
def test_the_decision_table(
    status: str, total: int | None, rows: int, published: bool, expected: bool
) -> None:
    assert (
        swap_already_happened(
            _run(status, total), journal_rows=rows, staging_published=published
        )
        is expected
    )


class DiesBeforeSwappedMark:
    """A manifest that fails exactly at the swapped status write."""

    def __init__(self, inner: InMemoryManifest, error: BaseException) -> None:
        self.inner = inner
        self.error = error

    def mark_swapped(self, run_id: str) -> None:
        raise self.error

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def _died(harness: Harness, *, crash: bool) -> None:
    real = harness.manifest
    harness.manifest = DiesBeforeSwappedMark(  # type: ignore[assignment]  # test double
        real, Crash() if crash else OSError("disk")
    )
    with pytest.raises(Crash if crash else BuildFailed):
        harness.build().run()
    harness.manifest = real


@pytest.mark.parametrize(("status", "crash"), [("running", True), ("aborted", False)])
def test_resume_finishes_the_run_and_a_second_resume_has_nothing_to_do(
    tmp_path: Path, status: str, crash: bool
) -> None:
    harness = Harness(tmp_path)
    _died(harness, crash=crash)
    run = harness.manifest.latest_run()
    assert run is not None and run.status == status
    assert harness.manifest.get_source("doc000.txt") is None

    summary = harness.build().resume()

    assert summary.recovered and summary.indexed == 5
    done = harness.manifest.latest_run()
    assert done is not None and done.status == "completed"
    assert {r.source.path: r.status for r in harness.manifest.run_sources("1")} == {
        f"doc{i:03d}.txt": "indexed" for i in range(5)
    }
    assert harness.manifest.get_source("doc004.txt") is not None
    assert harness.store.count_passages() == 5
    with pytest.raises(NothingToResume):
        harness.build().resume()


def test_plain_index_recovers_first_on_the_fakes(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    _died(harness, crash=True)

    summary = harness.build().run(refuse_resumable=True)

    assert summary.run_id == "2" and not summary.cancelled
    first = harness.manifest._runs[0]
    assert first.status == "completed"


def test_a_short_journal_is_not_recovered(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    _died(harness, crash=True)
    del harness.manifest._journal["1"]["doc004.txt"]

    with pytest.raises(StagingInconsistent):
        harness.build().resume()
    assert harness.manifest._runs[0].status == "running"
