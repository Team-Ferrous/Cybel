"""Immutable cache for external and analysis repositories."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from typing import Dict, Optional

from core.campaign.workspace import CampaignWorkspace


class RepoCache:
    """Stores de-duplicated repo snapshots outside the target workspace."""

    def __init__(self, workspace: CampaignWorkspace):
        self.workspace = workspace
        self.workspace.ensure_layout()

    @staticmethod
    def fingerprint(origin: str, revision: str) -> str:
        return hashlib.sha1(f"{origin}@{revision}".encode("utf-8")).hexdigest()

    def cache_dir(self, origin: str, revision: str) -> str:
        path = os.path.join(self.workspace.repo_cache_dir, self.fingerprint(origin, revision))
        os.makedirs(path, exist_ok=True)
        return path

    def snapshot_local_repo(
        self,
        source_path: str,
        origin: Optional[str] = None,
        revision: Optional[str] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        abs_source = os.path.abspath(source_path)
        revision = revision or self._git_revision(abs_source) or "working-tree"
        origin = origin or abs_source
        cache_dir = self.cache_dir(origin, revision)
        target_source = os.path.join(cache_dir, "source")
        if not os.path.exists(target_source):
            shutil.copytree(
                abs_source,
                target_source,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"),
            )
        result = {
            "origin": origin,
            "revision": revision,
            "source_path": target_source,
            "metadata_path": os.path.join(cache_dir, "metadata.json"),
            "analysis_pack_dir": os.path.join(cache_dir, "analysis_pack"),
            "fetched_at": time.time(),
            "metadata": dict(metadata or {}),
        }
        os.makedirs(result["analysis_pack_dir"], exist_ok=True)
        with open(result["metadata_path"], "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, default=str)
        return result

    def clone_remote_repo(
        self,
        origin_url: str,
        revision: str = "HEAD",
        timeout: int = 120,
    ) -> Dict[str, object]:
        cache_dir = self.cache_dir(origin_url, revision)
        target_source = os.path.join(cache_dir, "source")
        if not os.path.exists(os.path.join(target_source, ".git")):
            subprocess.run(
                ["git", "clone", "--depth", "1", origin_url, target_source],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if revision != "HEAD":
                subprocess.run(
                    ["git", "-C", target_source, "checkout", revision],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
        return self.snapshot_local_repo(target_source, origin=origin_url, revision=revision)

    @staticmethod
    def _git_revision(path: str) -> Optional[str]:
        try:
            return (
                subprocess.check_output(
                    ["git", "-C", path, "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=2,
                )
                .strip()
            )
        except Exception:
            return None


CampaignRepoCache = RepoCache


CampaignRepoCache = RepoCache
