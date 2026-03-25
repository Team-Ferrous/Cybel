"""Atomic filesystem helpers for Saguaro state persistence."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from typing import Any, Iterator

import yaml

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]


def fsync_dir(path: str) -> None:
    """Best-effort fsync of a directory entry."""
    directory = path if os.path.isdir(path) else os.path.dirname(path) or "."
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


@contextmanager
def _locked_file(path: str, mode: str) -> Iterator[Any]:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, mode, encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield handle
        finally:
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_write_bytes(path: str, payload: bytes) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        fsync_dir(directory)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def atomic_write_text(path: str, text: str) -> None:
    """Atomically replace a text file."""
    _atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(
    path: str,
    payload: Any,
    *,
    indent: int = 2,
    sort_keys: bool = True,
) -> None:
    """Atomically replace a JSON file."""
    text = json.dumps(payload, indent=indent, sort_keys=sort_keys)
    atomic_write_text(path, text + "\n")


def atomic_write_yaml(path: str, payload: Any) -> None:
    """Atomically replace a YAML file."""
    text = yaml.safe_dump(payload, sort_keys=True)
    atomic_write_text(path, text)


def atomic_append_jsonl(path: str, payload: Any) -> None:
    """Append a JSON line with an exclusive file lock and durability flush."""
    line = json.dumps(payload, sort_keys=True) + "\n"
    with _locked_file(path, "a") as handle:
        handle.write(line)
