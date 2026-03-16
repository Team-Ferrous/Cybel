from __future__ import annotations

import multiprocessing

import pytest

from saguaro.errors import SaguaroBusyError
from saguaro.storage.locks import RepoLockManager


def _hold_index_lock(saguaro_dir: str, ready: multiprocessing.Event, release: multiprocessing.Event) -> None:
    manager = RepoLockManager(saguaro_dir)
    lease = manager.acquire_task("index", operation="holder", timeout_seconds=5.0)
    try:
        ready.set()
        release.wait(5.0)
    finally:
        lease.release()


def test_index_lock_detects_multiprocess_contention(tmp_path) -> None:
    saguaro_dir = str(tmp_path / ".saguaro")
    manager = RepoLockManager(saguaro_dir)
    ready = multiprocessing.Event()
    release = multiprocessing.Event()

    proc = multiprocessing.Process(
        target=_hold_index_lock,
        args=(saguaro_dir, ready, release),
    )
    proc.start()
    assert ready.wait(5.0)

    try:
        with pytest.raises(SaguaroBusyError):
            manager.acquire_task("index", operation="contender", timeout_seconds=0.05)
    finally:
        release.set()
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2.0)
    assert proc.exitcode == 0
