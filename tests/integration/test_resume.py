"""Resume against real SQLite and local-mode Qdrant."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient

from docgrep.adapters.manifest import SQLiteManifest
from docgrep.adapters.qdrant import QdrantPassageStore
from docgrep.domain.errors import (
    ConfigDrift,
    CorpusDrift,
    NothingToResume,
    StagingInconsistent,
    StoreUnavailable,
)
from resume_support import (
    cancel_at_embed_call,
    close,
    dump_points,
    fail_upsert_at,
    new_services,
    search_snapshot,
    settings_for,
    write_corpus,
)

SOURCES = 5


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("corpus")
    write_corpus(root, SOURCES)
    return root


@pytest.fixture(scope="module")
def reference(
    corpus: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[list[Any], list[Any]]:
    """The points and the search results of an uninterrupted build."""
    settings = settings_for(corpus, tmp_path_factory.mktemp("clean"))
    services = new_services(settings)
    services.build.run()
    close(services)
    searched = _searched_after_reopening(settings)
    points = dump_points(settings)
    assert len(points) > SOURCES * 2 and any(searched)
    return points, searched


def _searched_after_reopening(settings: Any) -> list[Any]:
    """Search in a fresh container: local mode reloads (and re-normalises) vectors
    from disk, so scores are only comparable between stores opened the same way."""
    services = new_services(settings)
    try:
        return search_snapshot(services)
    finally:
        close(services)


def _finish_and_compare(settings: Any, reference: tuple[list[Any], list[Any]]) -> Any:
    services = new_services(settings)
    summary = services.build.resume()
    assert not summary.cancelled
    close(services)
    assert dump_points(settings) == reference[0]
    assert _searched_after_reopening(settings) == reference[1]
    return summary


@pytest.mark.parametrize("after", range(SOURCES + 1))
def test_a_build_cancelled_after_k_sources_resumes_to_the_uninterrupted_index(
    corpus: Path, tmp_path: Path, reference: tuple[list[Any], list[Any]], after: int
) -> None:
    settings = settings_for(corpus, tmp_path / "data")
    services = new_services(settings)
    cancel_at_embed_call(services, after)
    cancelled = services.build.run()
    assert cancelled.cancelled and cancelled.indexed == after
    close(services)

    summary = _finish_and_compare(settings, reference)

    assert summary.resumed_from == cancelled.run_id
    assert (summary.carried_over, summary.processed) == (after, SOURCES - after)
    assert (summary.retried, summary.removed) == (0, 0)
    manifest = SQLiteManifest(settings.data_dir)
    run = manifest.latest_run()
    assert run is not None and run.status == "completed"
    assert run.resumed_from == cancelled.run_id
    assert run.total_sources == SOURCES and run.carried_sources == after
    assert len(manifest.run_sources(run.run_id)) == SOURCES
    committed = [manifest.get_source(f"doc{i:03d}.txt") for i in range(SOURCES)]
    assert all(r is not None and r.status == "indexed" for r in committed)


@pytest.mark.parametrize("failing_call", [1, 2, 3, 4, 7, 10, 14])
def test_a_failure_in_the_middle_of_a_source_resumes_to_the_same_index(
    corpus: Path,
    tmp_path: Path,
    reference: tuple[list[Any], list[Any]],
    failing_call: int,
) -> None:
    # One passage per upsert call, so the failure leaves a half-written source.
    settings = settings_for(corpus, tmp_path / "data", upsert_batch_size=1)
    services = new_services(settings)
    fail_upsert_at(services, failing_call)
    with pytest.raises(StoreUnavailable):
        services.build.run()
    manifest = SQLiteManifest(settings.data_dir)
    run = manifest.latest_run()
    assert run is not None and run.status == "aborted"
    assert run.staging_collection is not None
    close(services)

    # The upsert batch size is an operational setting and may differ.
    resumed = settings_for(corpus, tmp_path / "data")
    summary = _finish_and_compare(resumed, reference)

    assert summary.resumed_from == run.run_id


def test_a_resume_of_a_resume_chains(
    corpus: Path, tmp_path: Path, reference: tuple[list[Any], list[Any]]
) -> None:
    settings = settings_for(corpus, tmp_path / "data")
    first = new_services(settings)
    cancel_at_embed_call(first, 1)
    one = first.build.run()
    close(first)
    second = new_services(settings)
    cancel_at_embed_call(second, 2)
    two = second.build.resume()
    assert two.cancelled and two.resumed_from == one.run_id
    assert (two.carried_over, two.processed) == (1, SOURCES - 1)
    close(second)

    summary = _finish_and_compare(settings, reference)

    assert summary.resumed_from == two.run_id
    assert (summary.carried_over, summary.processed) == (3, SOURCES - 3)
    manifest = SQLiteManifest(settings.data_dir)
    run = manifest.latest_run()
    assert run is not None and run.resumed_from == two.run_id


def test_resume_after_a_successful_resume_has_nothing_to_resume(
    corpus: Path, tmp_path: Path, reference: tuple[list[Any], list[Any]]
) -> None:
    settings = settings_for(corpus, tmp_path / "data")
    services = new_services(settings)
    cancel_at_embed_call(services, 2)
    services.build.run()
    close(services)
    _finish_and_compare(settings, reference)

    services = new_services(settings)
    with pytest.raises(NothingToResume) as raised:
        services.build.resume()
    close(services)

    assert raised.value.category == "nothing_to_resume"


def test_resume_without_any_run_has_nothing_to_resume(
    corpus: Path, tmp_path: Path
) -> None:
    services = new_services(settings_for(corpus, tmp_path / "data"))
    with pytest.raises(NothingToResume):
        services.build.resume()
    close(services)


def _interrupted(corpus: Path, data: Path, after: int = 2) -> Any:
    settings = settings_for(corpus, data)
    services = new_services(settings)
    cancel_at_embed_call(services, after)
    services.build.run()
    close(services)
    return settings


def test_a_changed_vector_setting_is_rejected_and_an_operational_one_is_not(
    corpus: Path, tmp_path: Path
) -> None:
    settings = _interrupted(corpus, tmp_path / "data")

    drifted = new_services(
        settings_for(corpus, tmp_path / "data").model_copy(update={"chunk_size": 12})
    )
    with pytest.raises(ConfigDrift) as raised:
        drifted.build.resume()
    close(drifted)

    assert raised.value.category == "config_drift"
    assert raised.value.details["fields"] == "chunk_size"
    manifest = SQLiteManifest(settings.data_dir)
    run = manifest.latest_run()
    assert run is not None and run.status == "cancelled"  # nothing was started

    operational = new_services(
        settings.model_copy(
            update={"upsert_batch_size": 7, "device": "cpu", "log_level": "WARNING"}
        )
    )
    assert not operational.build.resume().cancelled
    close(operational)


def _client(settings: Any) -> QdrantClient:
    return QdrantClient(path=str(settings.data_dir / "qdrant"))


def test_a_missing_staging_collection_is_rejected(corpus: Path, tmp_path: Path) -> None:
    settings = _interrupted(corpus, tmp_path / "data")
    client = _client(settings)
    run = SQLiteManifest(settings.data_dir).latest_run()
    assert run is not None and run.staging_collection is not None
    client.delete_collection(run.staging_collection)
    client.close()

    services = new_services(settings)
    with pytest.raises(StagingInconsistent) as raised:
        services.build.resume()
    close(services)

    assert raised.value.category == "staging_inconsistent"
    assert "--restart" in str(raised.value)


def test_a_staging_collection_with_fewer_points_than_journaled_is_rejected(
    corpus: Path, tmp_path: Path
) -> None:
    settings = _interrupted(corpus, tmp_path / "data")
    run = SQLiteManifest(settings.data_dir).latest_run()
    assert run is not None and run.staging_collection is not None
    client = _client(settings)
    points, _ = client.scroll(run.staging_collection, limit=1)
    client.delete(run.staging_collection, points_selector=[points[0].id])
    client.close()

    services = new_services(settings)
    with pytest.raises(StagingInconsistent) as raised:
        services.build.resume()
    close(services)

    assert raised.value.details["found"] < raised.value.details["journaled"]


def test_a_staging_collection_of_another_dimension_is_rejected(
    corpus: Path, tmp_path: Path
) -> None:
    settings = _interrupted(corpus, tmp_path / "data")
    run = SQLiteManifest(settings.data_dir).latest_run()
    assert run is not None and run.staging_collection is not None
    client = _client(settings)
    client.delete_collection(run.staging_collection)
    client.close()
    store = QdrantPassageStore(settings.data_dir / "qdrant")
    store.recreate_collection(2)
    wrong = store.staging_name()
    assert wrong is not None
    store.close()
    SQLiteManifest(settings.data_dir).set_staging_collection(run.run_id, wrong)

    services = new_services(settings)
    with pytest.raises(StagingInconsistent) as raised:
        services.build.resume()
    close(services)

    assert "dimension" in str(raised.value)


def test_a_recorded_staging_collection_that_is_the_alias_target_is_rejected(
    corpus: Path, tmp_path: Path
) -> None:
    settings = settings_for(corpus, tmp_path / "data")
    services = new_services(settings)
    services.build.run()
    manifest = SQLiteManifest(settings.data_dir)
    config = dict(services.build.run_config)  # type: ignore[attr-defined]  # BuildIndex
    store = services.build.passage_writer  # type: ignore[attr-defined]
    published = store._active_collection()
    close(services)
    stale = manifest.start_run(config)
    manifest.set_staging_collection(stale, published)
    manifest.finish_run(stale, "aborted", {})

    services = new_services(settings)
    with pytest.raises(StagingInconsistent) as raised:
        services.build.resume()
    close(services)

    message = str(raised.value)
    assert "published index" in message and "status" in message
    assert "--restart" in message


def test_too_much_corpus_drift_is_rejected_and_nothing_is_started(
    corpus: Path, tmp_path: Path
) -> None:
    drifting = tmp_path / "corpus"
    write_corpus(drifting, SOURCES)
    settings = _interrupted(drifting, tmp_path / "data", after=3)
    (drifting / "doc000.txt").write_text("rewritten completely", encoding="utf-8")
    runs_before = SQLiteManifest(settings.data_dir).latest_run()

    services = new_services(settings)
    with pytest.raises(CorpusDrift) as raised:
        services.build.resume()
    close(services)

    assert raised.value.category == "corpus_drift"
    assert raised.value.details["changed_or_missing"] == 1
    assert SQLiteManifest(settings.data_dir).latest_run() == runs_before


def test_a_small_drift_reprocesses_changed_sources_and_drops_vanished_ones(
    tmp_path: Path,
) -> None:
    many = tmp_path / "corpus"
    write_corpus(many, 100, words=6)
    settings = settings_for(many, tmp_path / "data")
    services = new_services(settings)
    cancel_at_embed_call(services, 40)
    cancelled = services.build.run()
    assert cancelled.cancelled and cancelled.indexed == 40
    close(services)
    (many / "doc003.txt").write_text("changed after the interruption", "utf-8")
    (many / "doc004.txt").unlink()  # 2 of 40 is exactly the 5% that is tolerated
    (many / "doc999.txt").write_text("brand new late arrival", "utf-8")

    resumed = new_services(settings)
    summary = resumed.build.resume()
    close(resumed)

    assert (summary.carried_over, summary.removed) == (38, 1)
    assert summary.processed == 100 - 38  # 59 still to do, doc003 and doc999
    clean = settings_for(many, tmp_path / "clean")
    reference = new_services(clean)
    reference.build.run()
    close(reference)
    queries = ("changed after", "marker4", "late arrival")
    searched = []
    for each in (settings, clean):
        reopened = new_services(each)
        searched.append(search_snapshot(reopened, queries))
        close(reopened)
    assert dump_points(settings) == dump_points(clean)
    assert searched[0] == searched[1]
    assert all(searched[0][0]) and searched[0][2]


def _cli(tmp_path: Path, corpus: Path, *args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        DOCGREP_CORPUS_ROOT=str(corpus),
        DOCGREP_DATA_DIR=str(tmp_path / "data"),
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


def test_plain_index_refuses_a_resumable_run_and_restart_discards_it(
    corpus: Path, tmp_path: Path, reference: tuple[list[Any], list[Any]]
) -> None:
    settings = _interrupted(corpus, tmp_path / "data")
    staging = SQLiteManifest(settings.data_dir).latest_run()
    assert staging is not None and staging.staging_collection is not None

    refused = _cli(tmp_path, corpus, "index")

    assert refused.returncode == 1 and refused.stdout == ""
    assert refused.stderr.startswith("resume_available:")
    assert "--resume" in refused.stderr and "--restart" in refused.stderr
    run = SQLiteManifest(settings.data_dir).latest_run()
    assert run == staging  # nothing changed

    restarted = _cli(tmp_path, corpus, "index", "--restart")

    assert restarted.returncode == 0, restarted.stderr
    assert "Build" in restarted.stdout
    assert dump_points(settings) == reference[0]
    client = _client(settings)
    names = {c.name for c in client.get_collections().collections}
    client.close()
    assert staging.staging_collection not in names and len(names) == 1


def test_the_console_resumes_and_then_reports_nothing_to_resume(
    corpus: Path, tmp_path: Path, reference: tuple[list[Any], list[Any]]
) -> None:
    settings = _interrupted(corpus, tmp_path / "data")

    resumed = _cli(tmp_path, corpus, "index", "--resume")

    assert resumed.returncode == 0, resumed.stderr
    assert "Resumed run 1 as run 2: 2 sources carried over, 3 processed" in (
        resumed.stdout
    )
    assert dump_points(settings) == reference[0]
    again = _cli(tmp_path, corpus, "index", "--resume")
    assert again.returncode == 1
    assert again.stderr.startswith("nothing_to_resume:")
    plain = _cli(tmp_path, corpus, "index")  # a completed latest run: as before
    assert plain.returncode == 0, plain.stderr


def test_resume_and_restart_exclude_each_other_and_no_retry_needs_resume(
    corpus: Path, tmp_path: Path
) -> None:
    both = _cli(tmp_path, corpus, "index", "--resume", "--restart")
    assert both.returncode == 2 and "not allowed with" in both.stderr
    alone = _cli(tmp_path, corpus, "index", "--no-retry-failed")
    assert alone.returncode == 1 and "requires --resume" in alone.stderr
