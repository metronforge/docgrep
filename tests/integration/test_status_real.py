"""Status against real SQLite, local Qdrant and a lock held by another process."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from docgrep.adapters.manifest import SQLiteManifest
from docgrep.adapters.qdrant import QdrantPassageStore
from docgrep.api.cli import main
from docgrep.api.container import create_status
from docgrep.domain.errors import StoreUnavailable
from docgrep.domain.status import INDEX_STATES
from resume_support import (
    cancel_at_embed_call,
    close,
    fail_upsert_at,
    new_services,
    settings_for,
    write_corpus,
)


class Crash(BaseException):
    """A process death."""


class CrashOnCommit:
    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def commit_run(self, run_id: str) -> None:
        raise Crash

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def _listing(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def _report(settings: Any) -> dict[str, Any]:
    before = _listing(settings.data_dir.parent)
    report = create_status(settings).run().as_dict()
    assert _listing(settings.data_dir.parent) == before  # nothing was written
    return report


@pytest.fixture
def setup(tmp_path: Path) -> Any:
    corpus = tmp_path / "corpus"
    write_corpus(corpus, 5)
    return settings_for(corpus, tmp_path / "data")


def test_an_unused_data_directory_is_not_built_and_stays_absent(
    tmp_path: Path, setup: Any
) -> None:
    report = _report(setup)

    assert report["index_state"] == "not_built"
    assert report["latest_run"] is None and report["collection"] is None
    assert report["vector_store"]["status"] == "absent"
    assert not setup.data_dir.exists()


def test_a_completed_build_is_ready(setup: Any) -> None:
    services = new_services(setup)
    services.build.run()
    close(services)

    report = _report(setup)

    assert report["index_state"] == "ready"
    run = report["latest_run"]
    assert (run["status"], run["state"], run["total_sources"]) == (
        "completed",
        "completed",
        5,
    )
    assert run["totals"] == {"indexed": 5, "skipped": 0, "failed": 0}
    collection = report["collection"]
    assert collection["passage_count"] > 5
    assert collection["embedding_model"] == "deterministic-hash-v1"
    assert collection["embedding_dimension"] == 64
    assert collection["embedding_dtype"] == "float32"
    assert collection["docgrep_version"] and collection["completed_at"]
    assert report["vector_store"] == {
        "mode": "local",
        "status": "reachable",
        "reason": None,
    }
    assert report["drift"] == [] and report["warnings"] == []
    assert report["active_operation"] is None


def test_a_vector_setting_that_changed_is_reported_as_drift(setup: Any) -> None:
    services = new_services(setup)
    services.build.run()
    close(services)

    report = _report(setup.model_copy(update={"chunk_size": 12}))

    assert report["drift"] == [{"field": "chunk_size", "indexed": 10, "runtime": 12}]
    assert [w["code"] for w in report["warnings"]] == ["config_drift"]
    assert report["index_state"] == "ready"


def test_a_cancelled_build_is_pending_resume(setup: Any) -> None:
    services = new_services(setup)
    cancel_at_embed_call(services, 2)
    services.build.run()
    close(services)

    report = _report(setup)

    assert report["index_state"] == "cancelled_pending_resume"
    assert report["latest_run"]["state"] == "cancelled"
    assert report["latest_run"]["total_sources"] == 5
    assert report["active_operation"] is None
    assert "resume_available" in [w["code"] for w in report["warnings"]]


def test_an_aborted_build_is_pending_resume(setup: Any) -> None:
    services = new_services(setup)
    fail_upsert_at(services, 2)
    with pytest.raises(StoreUnavailable):
        services.build.run()
    close(services)

    assert _report(setup)["index_state"] == "aborted_pending_resume"


def test_a_killed_build_is_interrupted_pending_resume(setup: Any) -> None:
    manifest = SQLiteManifest(setup.data_dir)
    manifest.start_run({"chunk_size": 10})  # its process never came back

    report = _report(setup)

    assert report["index_state"] == "interrupted_pending_resume"
    assert report["latest_run"]["status"] == "running"
    assert report["latest_run"]["state"] == "interrupted"
    assert report["active_operation"] is None


def test_a_crash_between_swap_and_commit_is_swap_pending_commit(setup: Any) -> None:
    services = new_services(setup)
    build: Any = services.build
    build.manifest = CrashOnCommit(build.manifest)
    with pytest.raises(Crash):
        build.run()
    close(services)

    report = _report(setup)

    assert report["index_state"] == "swap_pending_commit"
    assert report["latest_run"]["state"] == "swapped"
    assert report["collection"]["passage_count"] > 0  # the new index is live
    # Reporting it changes nothing: the run is still swapped afterwards.
    latest = SQLiteManifest(setup.data_dir).latest_run()
    assert latest is not None and latest.status == "swapped"


def test_an_unreadable_manifest_degrades_the_report(setup: Any) -> None:
    services = new_services(setup)
    services.build.run()
    close(services)
    (setup.data_dir / "manifest.sqlite3").write_bytes(b"this is not a database" * 50)

    report = _report(setup)

    assert report["index_state"] == "manifest_unreadable"
    assert report["latest_run"] is None
    assert report["collection"]["passage_count"] > 5  # the store is still read
    assert "manifest_unreadable" in [w["code"] for w in report["warnings"]]


def test_a_local_store_held_by_another_process_is_busy(setup: Any) -> None:
    services = new_services(setup)
    services.build.run()
    close(services)
    holder = QdrantPassageStore(setup.data_dir / "qdrant")
    try:
        report = _report(setup)
    finally:
        holder.close()

    assert report["vector_store"]["status"] == "busy"
    assert report["collection"] is None
    assert report["index_state"] == "ready"  # the manifest still says so
    assert [w["code"] for w in report["warnings"]] == ["store_busy"]


def test_an_unreachable_server_is_reported_quickly(setup: Any) -> None:
    started = time.monotonic()

    report = _report(setup.model_copy(update={"qdrant_url": "http://127.0.0.1:9"}))

    assert time.monotonic() - started < 10
    assert report["vector_store"]["mode"] == "server"
    assert report["vector_store"]["status"] == "unreachable"
    assert report["vector_store"]["reason"]
    assert report["collection"] is None


HOLDER = """
import sys, time
from pathlib import Path
from docgrep.adapters.filelock import FileIndexLock
from docgrep.adapters.manifest import SQLiteManifest
from docgrep.domain.models import FileType, Source

data = Path(sys.argv[1])
manifest = SQLiteManifest(data)
run = manifest.start_run({"chunk_size": 10})
manifest.set_total_sources(run, 8)
for i in range(3):
    manifest.record_source(
        Source(f"d{i}", f"d{i}", FileType.TXT), "indexed", run_id=run, passages=4
    )
manifest.request_cancel(run)
with FileIndexLock(data / "index.lock").hold():
    print("ready", flush=True)
    sys.stdin.read()
"""


def test_the_active_operation_of_a_build_in_another_process(
    tmp_path: Path, setup: Any
) -> None:
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(setup.data_dir)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "ready"
        report = _report(setup)
    finally:
        assert holder.stdin is not None
        holder.stdin.close()
        holder.wait(10)

    active = report["active_operation"]
    assert (active["kind"], active["run_id"]) == ("build", "1")
    assert (active["sources_done"], active["sources_total"]) == (3, 8)
    assert active["passages_written"] == 12
    assert active["elapsed_seconds"] >= 0 and active["eta_seconds"] is not None
    assert active["cancel_requested_at"] is not None
    assert report["latest_run"]["state"] == "running"
    assert report["index_state"] == "not_built"


def _cli(setup: Any, *args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        DOCGREP_CORPUS_ROOT=str(setup.corpus_root),
        DOCGREP_DATA_DIR=str(setup.data_dir),
        DOCGREP_BACKEND="fake",
        DOCGREP_CHUNK_SIZE="10",
        DOCGREP_CHUNK_OVERLAP="2",
    )
    return subprocess.run(
        [sys.executable, "-m", "docgrep.api.cli", *args],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
        timeout=60,
    )


def test_the_console_prints_text_and_json_and_exits_0_even_when_degraded(
    setup: Any,
) -> None:
    services = new_services(setup)
    cancel_at_embed_call(services, 2)
    services.build.run()
    close(services)

    text = _cli(setup, "status")
    as_json = _cli(setup, "status", "--json")

    assert text.returncode == 0 and as_json.returncode == 0, text.stderr
    assert "Index: cancelled_pending_resume" in text.stdout
    assert "Vector store: local, reachable" in text.stdout
    assert "resume_available" in text.stdout
    data = json.loads(as_json.stdout)
    assert data["schema_version"] == 1
    assert data["index_state"] == "cancelled_pending_resume"

    (setup.data_dir / "manifest.sqlite3").write_bytes(b"garbage" * 100)
    degraded = _cli(setup, "status")
    assert degraded.returncode == 0
    assert "Index: manifest_unreadable" in degraded.stdout


def test_status_of_an_empty_directory_through_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DOCGREP_DATA_DIR", str(tmp_path / "data"))

    assert main(["status"]) == 0
    assert "Index: not_built" in capsys.readouterr().out
    assert not (tmp_path / "data").exists()


def _keys(value: object, found: set[str] | None = None) -> set[str]:
    found = set() if found is None else found
    if isinstance(value, dict):
        for key, inner in value.items():
            found.add(key)
            _keys(inner, found)
    elif isinstance(value, list):
        for inner in value:
            _keys(inner, found)
    return found


def test_every_json_key_and_state_is_documented_in_the_readme(setup: Any) -> None:
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text("utf-8")
    services = new_services(setup)
    services.build.run()
    close(services)
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(setup.data_dir)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "ready"
        # An index, a latest run, an active operation and drift: every section.
        report = _report(setup.model_copy(update={"chunk_size": 12}))
    finally:
        assert holder.stdin is not None
        holder.stdin.close()
        holder.wait(10)

    assert report["active_operation"] and report["collection"] and report["drift"]
    assert report["warnings"] and report["latest_run"]
    missing = {key for key in _keys(report) if f"`{key}`" not in readme}
    assert missing == set()
    assert {s for s in INDEX_STATES if f"`{s}`" not in readme} == set()
