"""The background task of `docgrep serve` that keeps the index following the
corpus (ADR-0029)."""

import asyncio
import logging
from collections.abc import Callable

from docgrep.api.container import Services
from docgrep.usecases.sync_index import SyncIndex

_LOG = logging.getLogger(__name__)


class SyncTask:
    """Run sync cycles on a worker thread, one after the other, until stopped.

    The task takes the container the server shares with search, so the models
    are loaded once; it asks for it on a worker thread, so the server answers
    while they load. Stopping starts no new cycle and lets an applied run
    finish the source it is on.
    """

    def __init__(self, operations: Callable[[], Services], interval: float) -> None:
        self._operations = operations
        self._interval = interval
        self._stop: asyncio.Event | None = None
        self._task: asyncio.Task[None] | None = None
        self.sync: SyncIndex | None = None

    def start(self) -> None:
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="docgrep-sync")

    async def stop(self) -> None:
        """Stop after the source being processed and wait for the task."""
        if self._task is None or self._stop is None:
            return
        self._stop.set()
        if self.sync is not None:
            self.sync.request_stop()
        await self._task
        self._task = None
        if self.sync is not None:
            self.sync.publish(running=False)

    async def _run(self) -> None:
        assert self._stop is not None
        try:
            services = await asyncio.to_thread(self._operations)
            self.sync = services.sync
            if self.sync is None:
                _LOG.warning("sync_unavailable", extra={"reason": "not configured"})
                return
            if self._stop.is_set():
                self.sync.request_stop()
            _LOG.info("sync_started")
            while not self._stop.is_set():
                try:
                    await asyncio.to_thread(self.sync.run_cycle)
                except Exception:
                    # A cycle that fails is tried again at the next one.
                    _LOG.exception("sync_cycle_failed")
                try:
                    await asyncio.wait_for(self._stop.wait(), self._interval)
                except TimeoutError:
                    pass
        except Exception:
            _LOG.exception("sync_stopped_unexpectedly")
        finally:
            _LOG.info("sync_stopped")
