"""The update command: plan and summary output, exit codes, refusals, cancel
and status, over the real stores and the deterministic backend (UC-4, UC-7)."""

import io
import json
import signal
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import pytest

from docgrep.adapters.manifest import SQLiteManifest
from docgrep.api.cli import main
from docgrep.api.container import Services, create_container
from docgrep.domain.models import FileType, ParsedDocument, Source
from docgrep.settings import Settings
from resume_support import cancel_at_embed_call, close
from update_support import (
    FINAL,
    INITIAL,
    HookedManifest,
    change_corpus,
    write_files,
)

README = Path(__file__).resolve().parents[2] / "README.md"


class Signals:
    """A stand-in for signal.signal that only remembers the handlers."""

    def __init__(self) -> None:
        self.handlers: dict[int, Any] = {}

    def __call__(self, number: int, handler: Any) -> Any:
        previous = self.handlers.get(number, signal.SIG_DFL)
        self.handlers[number] = handler
        return previous

    def deliver(self, number: int = signal.SIGINT) -> None:
        self.handlers[number](number, None)


Setup = Callable[[Services], None]


def _run(*args: str, setup: Setup | None = None, signals: Signals | None = None) -> Any:
    created: list[Services] = []

    def factory(settings: Settings) -> Services:
        services = create_container(settings)
        update: Any = services.update
        update.sleep = lambda seconds: None
        if setup is not None:
            setup(services)
        created.append(services)
        return services

    out, err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            code = main(
                list(args),
                container_factory=factory,
                register_signal=signals or Signals(),
            )
    finally:
        for services in created:
            close(services)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "corpus"
    write_files(root, INITIAL)
    monkeypatch.setenv("DOCGREP_CORPUS_ROOT", str(root))
    monkeypatch.setenv("DOCGREP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DOCGREP_BACKEND", "fake")
    monkeypatch.setenv("DOCGREP_CHUNK_SIZE", "10")
    monkeypatch.setenv("DOCGREP_CHUNK_OVERLAP", "2")
    return root


@pytest.fixture
def built(corpus: Path) -> Path:
    code, _, err = _run("index")
    assert code == 0, err
    change_corpus(corpus)
    return corpus


def _status(*args: str) -> str:
    code, out, _ = _run("status", *args)
    assert code == 0
    return str(out)


def test_update_prints_the_plan_and_a_summary(built: Path) -> None:
    code, out, err = _run("update")

    assert code == 0, err
    lines = out.splitlines()
    assert lines[0] == (
        "Update plan: 2 added, 2 modified, 2 removed, 0 retried, 7 unchanged."
    )
    assert lines[1].startswith(
        "Update 2: 2 added, 2 modified, 2 removed, 0 retried, 7 unchanged, "
        "0 skipped, 0 failed; "
    )
    assert "passages written" in lines[1] and len(lines) == 2


def test_a_second_update_reports_no_changes(built: Path) -> None:
    assert _run("update")[0] == 0

    code, out, _ = _run("update")

    assert code == 0
    assert (
        out.splitlines()[0] == f"No changes detected ({len(FINAL)} sources unchanged)."
    )


class _FailsFor:
    def __init__(self, inner: Any, path: str) -> None:
        self.inner = inner
        self.path = path

    def parse(self, source: Source) -> ParsedDocument:
        if source.path == self.path:
            raise ValueError("injected parse failure")
        document: ParsedDocument = self.inner.parse(source)
        return document


def _fail_doc10(services: Services) -> None:
    update: Any = services.update
    text = update.parsers[FileType.TXT]
    update.parsers = {**update.parsers, FileType.TXT: _FailsFor(text, "doc10.txt")}


def test_failed_sources_exit_1_and_are_listed(built: Path) -> None:
    code, out, _ = _run("update", setup=_fail_doc10)

    assert code == 1
    assert ", 1 failed; " in out
    assert "  doc10.txt: parse_error: injected parse failure" in out

    code, out, _ = _run("update", "--no-retry-failed")
    assert code == 0 and out.startswith("No changes detected")

    code, out, _ = _run("update")
    assert code == 0
    assert out.startswith("Update plan: 0 added, 0 modified, 0 removed, 1 retried")


def test_refused_without_an_index(corpus: Path) -> None:
    code, out, err = _run("update")

    assert code == 1 and out == ""
    assert err.startswith("index_incomplete: ")
    assert "docgrep index" in err


def test_refused_on_config_drift(built: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCGREP_CHUNK_SIZE", "12")

    code, out, err = _run("update")

    assert code == 1 and out == ""
    assert err.startswith("config_drift: ")
    assert "full rebuild" in err and "docgrep index" in err


def test_refused_while_a_build_can_be_resumed(corpus: Path) -> None:
    code, _, _ = _run("index", setup=lambda s: cancel_at_embed_call(s, 2))
    assert code == 2

    code, out, err = _run("update")

    assert code == 1 and out == ""
    assert err.startswith("resume_available: ")
    assert "docgrep index --resume" in err and "docgrep index --restart" in err


def test_ctrl_c_cancels_an_update_with_exit_2_and_update_continues(
    built: Path,
) -> None:
    signals = Signals()

    def interrupt_after_first_source(services: Services) -> None:
        update: Any = services.update
        update.manifest = HookedManifest(update.manifest, 1, signals.deliver)

    code, out, err = _run("update", setup=interrupt_after_first_source, signals=signals)

    assert code == 2
    assert "Cancel requested: the update stops after the current batch." in err
    assert "Update 2 cancelled: 0 added, 0 modified, 1 removed" in out
    assert "run `docgrep update` again to continue" in out
    status = _status()
    assert "Latest run: 2 (update) cancelled" in status
    assert "update_unfinished" in status
    assert "resume_available" not in status and "Index: ready" in status

    code, out, _ = _run("update")
    assert code == 0
    assert out.startswith("Update plan: 2 added, 2 modified, 1 removed")


def test_docgrep_cancel_reports_an_interrupted_update(built: Path) -> None:
    manifest = SQLiteManifest(Settings().data_dir)
    run_id = manifest.start_run({}, kind="update")

    code, out, _ = _run("cancel")

    assert code == 0
    assert out.startswith(f"Update run {run_id} was interrupted")
    assert "`docgrep update` again" in out


def test_status_after_an_update(built: Path) -> None:
    assert _run("update")[0] == 0

    text = _status()
    report = json.loads(_status("--json"))

    assert "Index: ready" in text
    assert "Latest run: 2 (update) completed" in text
    assert "2 added, 2 modified, 2 removed, 0 retried, 0 failed, 7 unchanged" in text
    assert ", last update " in text
    run = report["latest_run"]
    assert (run["kind"], run["status"], run["total_sources"]) == (
        "update",
        "completed",
        6,
    )
    assert run["totals"]["added"] == 2 and run["totals"]["unchanged"] == 7
    assert run["totals"]["passages_written"] > 0
    assert report["collection"]["last_updated_at"] is not None
    assert report["index_state"] == "ready"


def test_status_shows_the_kind_of_a_build(corpus: Path) -> None:
    assert _run("index")[0] == 0

    report = json.loads(_status("--json"))

    assert report["latest_run"]["kind"] == "build"
    assert report["collection"]["last_updated_at"] is None
    assert "Latest run: 1 (build) completed" in _status()


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


def test_every_status_key_and_warning_after_an_update_is_documented(
    built: Path,
) -> None:
    signals = Signals()

    def interrupt(services: Services) -> None:
        update: Any = services.update
        update.manifest = HookedManifest(update.manifest, 1, signals.deliver)

    assert _run("update")[0] == 0
    extra = built / "doc99.txt"
    extra.write_text("marker99 extra words", encoding="utf-8")
    assert _run("update", setup=interrupt, signals=signals)[0] == 2
    readme = README.read_text("utf-8")

    report = json.loads(_status("--json"))

    assert report["latest_run"]["kind"] == "update"
    assert report["collection"]["last_updated_at"]
    codes = {w["code"] for w in report["warnings"]}
    assert codes == {"update_unfinished"}
    missing = {key for key in _keys(report) if f"`{key}`" not in readme}
    assert missing == set()
    assert {c for c in codes if f"`{c}`" not in readme} == set()
