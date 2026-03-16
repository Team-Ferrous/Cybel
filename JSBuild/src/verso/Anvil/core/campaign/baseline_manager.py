"""Baseline capture for shared experiment lanes."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import time
from typing import Any, Dict, Iterable


class BaselineManager:
    """Captures deterministic baseline metadata before experiment execution."""

    def __init__(self, cwd: str) -> None:
        self.cwd = os.path.abspath(cwd)

    def capture(
        self,
        *,
        editable_scope: Iterable[str] | None = None,
        imported_baseline: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        editable = [str(path) for path in editable_scope or []]
        return {
            "captured_at": time.time(),
            "cwd": self.cwd,
            "git": self._git_snapshot(),
            "hardware_fingerprint": self._hardware_fingerprint(),
            "editable_scope": editable,
            "file_hashes": self._hash_scope(editable),
            "imported_baseline": dict(imported_baseline or {}),
        }

    def _git_snapshot(self) -> Dict[str, Any]:
        revision = ""
        branch = ""
        dirty = False
        try:
            revision = self._git(["rev-parse", "HEAD"])
            branch = self._git(["rev-parse", "--abbrev-ref", "HEAD"])
            dirty = bool(self._git(["status", "--porcelain"]))
        except Exception:
            pass
        return {
            "revision": revision,
            "branch": branch,
            "dirty": dirty,
        }

    def _git(self, argv: list[str]) -> str:
        completed = subprocess.run(
            ["git", *argv],
            cwd=self.cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()

    def _hardware_fingerprint(self) -> Dict[str, Any]:
        return {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
        }

    def _hash_scope(self, editable_scope: Iterable[str]) -> Dict[str, str]:
        hashes: Dict[str, str] = {}
        for path in editable_scope:
            resolved = path
            if not os.path.isabs(resolved):
                resolved = os.path.join(self.cwd, resolved)
            if not os.path.isfile(resolved):
                continue
            digest = hashlib.sha256()
            with open(resolved, "rb") as handle:
                digest.update(handle.read())
            hashes[path] = digest.hexdigest()
        return hashes
