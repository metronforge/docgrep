"""Automatic sync in the lifespan of `docgrep serve`: off by default, started
and stopped with the app, sharing its container, and finishing the source it
is on when the server stops (ADR-0029)."""

import asyncio
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from docgrep.api.container import Services, create_container
from docgrep.api.http import create_app
from docgrep.api.sync_task import SyncTask
from docgrep.usecases.sync_index import SyncIndex
from resume_support import close
from roots_support import settings
from sync_support import Rig, write


class StubSync:
    """Stands for SyncIndex: counts cycles and stop requests."""

    def __init__(self) -> None:
        self.cycles = 0
        self.stopped = threading.Event()
        self.first_cycle = threading.Event()
        self.published: list[bool] = []

    def run_cycle(self) -> None:
        self.cycles += 1
        self.first_cycle.set()

    def request_stop(self) -> None:
        self.stopped.set()

    def publish(self, *, running: bool = True) -> None:
        self.published.append(running)


def _services(sync: Any) -> Services:
    class Anything:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"unexpected use of {name}")

    return Services(search=Anything(), build=Anything(), sync=sync)  # type: ignore[arg-type]  # test double


def test_sync_is_off_by_default(tmp_path: Path) -> None:
    stub = StubSync()
    app = create_app(settings(tmp_path / "data"), services=_services(stub))

    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/health").status_code == 200

    assert stub.cycles == 0 and not stub.stopped.is_set()


def test_sync_starts_and_stops_with_the_app(tmp_path: Path) -> None:
    stub = StubSync()
    app = create_app(
        settings(tmp_path / "data", sync=True, sync_interval_seconds=3600),
        services=_services(stub),
    )

    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert stub.first_cycle.wait(timeout=10)
        assert client.get("/health").status_code == 200
        assert not stub.stopped.is_set()

    assert stub.stopped.is_set()
    assert stub.cycles == 1
    # The last state written says it is no longer running.
    assert stub.published == [False]


def test_a_cycle_that_fails_does_not_stop_the_task(tmp_path: Path) -> None:
    class Failing(StubSync):
        def run_cycle(self) -> None:
            super().run_cycle()
            raise RuntimeError("injected")

    stub = Failing()
    app = create_app(
        settings(tmp_path / "data", sync=True, sync_interval_seconds=3600),
        services=_services(stub),
    )

    with TestClient(app, base_url="http://127.0.0.1"):
        assert stub.first_cycle.wait(timeout=10)

    assert stub.stopped.is_set()


def test_the_task_shares_the_container_of_search(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    services = rig.open()
    created: list[Services] = []

    def operations() -> Services:
        created.append(services)
        return services

    async def run() -> None:
        task = SyncTask(operations, interval=3600)
        task.start()
        # The first cycle has run once the task has a status to publish.
        while rig.sync.status().last_scan_at is None:
            await asyncio.sleep(0)
            await asyncio.to_thread(lambda: None)
        await task.stop()
        assert task.sync is services.sync

    asyncio.run(run())
    rig.close()

    assert created == [services]


def test_shutdown_during_an_applied_run_finishes_the_current_source(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    services = rig.open()
    sync = services.sync
    assert isinstance(sync, SyncIndex)
    rig.cycle()
    write(rig.books, {f"new{i}.txt": f"Brand new file number {i}." for i in range(5)})
    rig.cycle()  # seen once; settles at the next scan
    rig.clock.advance(120)

    in_run, release = threading.Event(), threading.Event()
    calls: list[float] = []

    def at_first_source(seconds: float) -> None:
        # The first pause is before any source; the second is at the start
        # of the first one, which is then in progress when the stop comes.
        calls.append(seconds)
        if len(calls) == 2:
            in_run.set()
            assert release.wait(timeout=10)

    sync.sleep = at_first_source

    async def run() -> Any:
        task = SyncTask(lambda: replace(services, sync=sync), interval=3600)
        # One cycle on a worker thread, as the task would run it.
        cycle = asyncio.create_task(asyncio.to_thread(sync.run_cycle))
        task.sync = sync
        task._stop = asyncio.Event()
        task._task = asyncio.create_task(asyncio.sleep(0))
        await asyncio.to_thread(in_run.wait, 10)
        stopping = asyncio.create_task(task.stop())
        await asyncio.sleep(0)
        release.set()
        await stopping
        return await cycle

    result = asyncio.run(run())
    sources = rig.sources()
    rig.close()

    assert result.applied is not None and result.applied.cancelled
    # The source being processed was finished, no other was started.
    assert result.applied.added == 1
    assert sum(1 for path in sources if path.startswith("new")) == 1


@pytest.mark.parametrize("enabled", [False, True])
def test_info_and_health_work_with_sync_on_or_off(
    tmp_path: Path, enabled: bool
) -> None:
    app = create_app(
        settings(tmp_path / "data", sync=enabled, sync_interval_seconds=3600),
        services=_services(StubSync()),
    )

    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/info").status_code == 200


def test_the_container_builds_sync_only_when_enabled(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    off = create_container(rig.settings())
    try:
        assert off.sync is None
    finally:
        close(off)
    on = create_container(rig.settings(sync=True))
    try:
        assert isinstance(on.sync, SyncIndex)
    finally:
        close(on)
