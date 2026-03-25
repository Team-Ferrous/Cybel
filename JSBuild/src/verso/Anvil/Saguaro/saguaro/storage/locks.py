"""Repo-scoped file locks for many-reader / one-writer Saguaro operations."""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass

from saguaro.errors import SaguaroBusyError
from saguaro.storage.atomic_fs import atomic_write_json

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]


@dataclass
class RepoLock:
    """Held repo-scoped lock handle."""

    name: str
    mode: str
    path: str
    meta_path: str
    operation: str
    file_handle: object
    wait_ms: float = 0.0

    def release(self) -> None:
        if fcntl is not None:
            fcntl.flock(self.file_handle.fileno(), fcntl.LOCK_UN)
        try:
            self.file_handle.close()
        finally:
            if self.mode == "exclusive":
                try:
                    os.remove(self.meta_path)
                except OSError:
                    pass

    def __enter__(self) -> "RepoLock":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class RepoLockManager:
    """Coordinate advisory repo-scoped locks backed by flock."""

    _STATUS_LOCKS = (
        "index",
        "state",
        "compare_read.primary",
        "target_refresh.primary",
    )

    def __init__(self, saguaro_dir: str) -> None:
        self.saguaro_dir = os.path.abspath(saguaro_dir)
        self.locks_dir = os.path.join(self.saguaro_dir, "locks")
        os.makedirs(self.locks_dir, exist_ok=True)

    def acquire(
        self,
        name: str,
        *,
        mode: str,
        operation: str,
        timeout_seconds: float = 30.0,
    ) -> RepoLock:
        path = os.path.join(self.locks_dir, f"{name}.lock")
        meta_path = os.path.join(self.locks_dir, f"{name}.meta.json")
        os.makedirs(self.locks_dir, exist_ok=True)
        handle = open(path, "a+", encoding="utf-8")
        if fcntl is None:  # pragma: no cover - non-POSIX fallback
            return RepoLock(name, mode, path, meta_path, operation, handle, 0.0)

        desired = fcntl.LOCK_SH if mode == "shared" else fcntl.LOCK_EX
        started_at = time.perf_counter()
        deadline = time.time() + max(timeout_seconds, 0.0)
        while True:
            try:
                fcntl.flock(handle.fileno(), desired | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.time() >= deadline:
                    handle.close()
                    holder = self.describe_lock(name)
                    raise SaguaroBusyError(
                        f"Timed out acquiring {mode} lock '{name}' for {operation}: {holder}"
                    ) from exc
                time.sleep(0.1)

        if mode == "exclusive":
            atomic_write_json(
                meta_path,
                {
                    "name": name,
                    "mode": mode,
                    "operation": operation,
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "started_at": time.time(),
                },
            )
        return RepoLock(
            name,
            mode,
            path,
            meta_path,
            operation,
            handle,
            round((time.perf_counter() - started_at) * 1000.0, 3),
        )

    def compare_read(
        self,
        *,
        operation: str,
        target_id: str = "primary",
        timeout_seconds: float = 30.0,
    ) -> RepoLock:
        return self.acquire(
            f"compare_read.{target_id}",
            mode="shared",
            operation=operation,
            timeout_seconds=timeout_seconds,
        )

    def target_refresh(
        self,
        *,
        operation: str,
        target_id: str = "primary",
        timeout_seconds: float = 30.0,
    ) -> RepoLock:
        return self.acquire(
            f"target_refresh.{target_id}",
            mode="exclusive",
            operation=operation,
            timeout_seconds=timeout_seconds,
        )

    def external_corpus_build(
        self,
        *,
        operation: str,
        corpus_id: str,
        timeout_seconds: float = 30.0,
    ) -> RepoLock:
        return self.acquire(
            f"external_corpus_build.{corpus_id}",
            mode="exclusive",
            operation=operation,
            timeout_seconds=timeout_seconds,
        )

    def describe_lock(self, name: str) -> dict[str, object]:
        path = os.path.join(self.locks_dir, f"{name}.lock")
        meta_path = os.path.join(self.locks_dir, f"{name}.meta.json")
        metadata: dict[str, object] = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as handle:
                    payload = json.load(handle) or {}
                if isinstance(payload, dict):
                    metadata = payload
            except Exception:
                metadata = {"status": "unreadable_meta"}

        locked = False
        if fcntl is not None and os.path.exists(path):
            handle = open(path, "a+", encoding="utf-8")
            try:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except BlockingIOError:
                    locked = True
            finally:
                handle.close()

        return {
            "name": name,
            "locked": locked,
            "metadata": metadata,
        }

    def status(self) -> dict[str, object]:
        return {name: self.describe_lock(name) for name in self._STATUS_LOCKS}
