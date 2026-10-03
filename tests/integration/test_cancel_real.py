"""Cancel across processes and against the real stores."""

import os
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from docgrep.adapters.filelock import FileIndexLock
from docgrep.adapters.manifest import SQLiteManifest
from docgrep.api.container import Services, create_container
from docgrep.settings import Settings


def _cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        DOCGREP_CORPUS_ROOT=str(tmp_path / "corpus"),
        DOCGREP_DATA_DIR=str(tmp_path / "data"),
        DOCGREP_BACKEND="fake",
    )
    return subprocess.run(
        [sys.executable, "-m", "docgrep.api.cli", *args],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
        timeout=30,
    )


def _services(tmp_path: Path) -> tuple[Services, Path]:
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    settings = Settings(
        corpus_root=corpus,
        data_dir=tmp_path / "data",
        backend="fake",
        chunk_size=10,
        chunk_overlap=2,
    )
    return create_container(settings), corpus


def test_a_cancel_command_in_another_process_stops_a_running_build(
    tmp_path: Path,
) -> None:
    services, corpus = _services(tmp_path)
    for name in ("a", "b", "c"):
        (corpus / f"{name}.txt").write_text(f"{name} alpha beta", encoding="utf-8")
    services.build.run()
    (corpus / "d.txt").write_text("d delta epsilon", encoding="utf-8")
    seen: dict[str, str] = {}
    original = services.build.embedder

    class Probe:
        model_name = original.model_name
        dimension = original.dimension
        dtype = original.dtype
        calls = 0

        def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
            Probe.calls += 1
            if Probe.calls == 2:
                first = _cli(tmp_path, "cancel")
                second = _cli(tmp_path, "cancel")
                seen["first"], seen["second"] = first.stdout, second.stdout
                seen["codes"] = f"{first.returncode}{second.returncode}"
            return original.embed(texts)

    services.build.embedder = Probe()  # type: ignore[assignment]  # test double

    summary = services.build.run()

    assert summary.cancelled and summary.indexed == 2  # b.txt finishes its batch
    assert seen["codes"] == "00"
    assert "Cancel requested for run 2" in seen["first"]
    assert "stopping" in seen["first"]
    assert "already requested" in seen["second"]
    requested = re.search(r"at (\S+):", seen["second"])
    manifest = SQLiteManifest(tmp_path / "data")
    run = manifest.latest_run()
    assert run is not None and run.status == "cancelled"
    assert requested is not None and requested.group(1) == run.cancel_requested_at
    # The live index and the committed sources are the ones of the first build.
    assert manifest.get_source("d.txt") is None
    assert manifest.get_source("a.txt") is not None
    assert [h.source.path for h in services.search.run("alpha", top_n=5).results]
    assert not FileIndexLock(tmp_path / "data" / "index.lock").is_held()


def test_the_cancel_command_reports_an_interrupted_run_and_an_idle_directory(
    tmp_path: Path,
) -> None:
    idle = _cli(tmp_path, "cancel")
    assert idle.returncode == 1 and "Nothing is running" in idle.stdout
    assert not (tmp_path / "data").exists()

    manifest = SQLiteManifest(tmp_path / "data")
    manifest.start_run({"chunk_size": 1})  # a process that never came back

    interrupted = _cli(tmp_path, "cancel")

    assert interrupted.returncode == 0 and "interrupted" in interrupted.stdout
    assert manifest.cancel_requested_at("1") is None
