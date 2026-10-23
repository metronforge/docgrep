"""A local-mode store used from several threads, as `docgrep serve` does
while sync writes: searches never fail while points are written."""

import threading
from pathlib import Path

from docgrep.api.container import create_container
from resume_support import close
from roots_support import save_roots, settings
from sync_support import BOOKS, write


def test_searches_succeed_while_sources_are_written(tmp_path: Path) -> None:
    books, data = tmp_path / "books", tmp_path / "data"
    write(books, BOOKS)
    save_roots(data, ("books", books, True))
    services = create_container(settings(data))
    services.build.run()
    failures: list[BaseException] = []
    stop = threading.Event()

    def search() -> None:
        while not stop.is_set():
            try:
                services.search.run("harbours lamps tale")
            except BaseException as error:  # noqa: BLE001  # recorded and asserted below
                failures.append(error)
                return

    readers = [threading.Thread(target=search) for _ in range(3)]
    for reader in readers:
        reader.start()
    try:
        for index in range(40):
            write(books, {f"new{index}.txt": f"New file {index} about glaciers."})
            assert services.update is not None
            services.update.run()
    finally:
        stop.set()
        for reader in readers:
            reader.join(30)
        close(services)

    assert failures == []
