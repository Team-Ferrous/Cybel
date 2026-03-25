"""Lightweight isolated worktree management for shared lanes."""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from typing import Any, Dict, Iterable


class CampaignWorktreeManager:
    """Creates deterministic lane workspaces without mutating the source root."""

    def __init__(self, cwd: str) -> None:
        self.cwd = os.path.abspath(cwd)
        self.base_dir = os.path.join(self.cwd, ".anvil_lane_runtime")
        os.makedirs(self.base_dir, exist_ok=True)

    def prepare(self, lane_id: str, editable_scope: Iterable[str] | None = None) -> Dict[str, Any]:
        lane_dir = os.path.join(self.base_dir, lane_id)
        if os.path.isdir(lane_dir):
            shutil.rmtree(lane_dir)
        os.makedirs(lane_dir, exist_ok=True)
        snapshot_dir = os.path.join(lane_dir, "baseline_scope")
        workspace_dir = os.path.join(lane_dir, "workspace")
        os.makedirs(snapshot_dir, exist_ok=True)
        os.makedirs(workspace_dir, exist_ok=True)
        normalized_scope = sorted(
            {
                self._relative_path(path)
                for path in (editable_scope or [])
                if self._relative_path(path)
            }
        )
        self._seed_overlay_workspace(workspace_dir)
        copied: list[str] = []
        baseline_hashes: dict[str, str] = {}
        for relative in normalized_scope:
            resolved = os.path.join(self.cwd, relative)
            if not os.path.isfile(resolved):
                continue
            target = os.path.join(snapshot_dir, relative)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(resolved, target)
            self._materialize_editable_file(workspace_dir, relative)
            copied.append(relative)
            baseline_hashes[relative] = self._hash_file(target)
        return {
            "lane_id": lane_id,
            "path": lane_dir,
            "snapshot_dir": snapshot_dir,
            "workspace_dir": workspace_dir,
            "copied_scope": copied,
            "isolation_mode": "directory_overlay",
            "baseline_hashes": baseline_hashes,
            "prepared_at": time.time(),
        }

    def finalize(self, lane_id: str, *, keep: bool) -> Dict[str, Any]:
        lane_dir = os.path.join(self.base_dir, lane_id)
        workspace_dir = os.path.join(lane_dir, "workspace")
        snapshot_dir = os.path.join(lane_dir, "baseline_scope")
        changed_files = self.changed_files(lane_id)
        return {
            "lane_id": lane_id,
            "path": lane_dir,
            "workspace_dir": workspace_dir,
            "snapshot_dir": snapshot_dir,
            "changed_files": changed_files,
            "keep": bool(keep),
            "finalized_at": time.time(),
        }

    def changed_files(self, lane_id: str) -> list[str]:
        lane_dir = os.path.join(self.base_dir, lane_id)
        snapshot_dir = os.path.join(lane_dir, "baseline_scope")
        workspace_dir = os.path.join(lane_dir, "workspace")
        if not os.path.isdir(snapshot_dir) or not os.path.isdir(workspace_dir):
            return []
        changed: list[str] = []
        for root, _, files in os.walk(snapshot_dir):
            for filename in files:
                baseline = os.path.join(root, filename)
                relative = os.path.relpath(baseline, snapshot_dir)
                candidate = os.path.join(workspace_dir, relative)
                if not os.path.exists(candidate):
                    changed.append(relative)
                    continue
                if self._hash_file(baseline) != self._hash_file(candidate):
                    changed.append(relative)
        return sorted(set(changed))

    def promote(self, lane_id: str, *, files: Iterable[str] | None = None) -> Dict[str, Any]:
        lane_dir = os.path.join(self.base_dir, lane_id)
        workspace_dir = os.path.join(lane_dir, "workspace")
        if not os.path.isdir(workspace_dir):
            raise FileNotFoundError(f"Lane workspace is missing for {lane_id}")
        selected = sorted(set(files or self.changed_files(lane_id)))
        promoted: list[str] = []
        for relative in selected:
            source = os.path.join(workspace_dir, relative)
            target = os.path.join(self.cwd, relative)
            if not os.path.isfile(source):
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(source, target)
            promoted.append(relative)
        return {
            "lane_id": lane_id,
            "promoted_files": promoted,
            "promoted_at": time.time(),
        }

    def virtual_union_snapshot(self, lane_ids: Iterable[str]) -> Dict[str, Any]:
        lane_list = list(lane_ids)
        union: Dict[str, Dict[str, Any]] = {}
        for lane_id in lane_list:
            changed = self.changed_files(lane_id)
            for relative in changed:
                source = os.path.join(self.base_dir, lane_id, "workspace", relative)
                if not os.path.isfile(source):
                    continue
                union[relative] = {
                    "lane_id": lane_id,
                    "path": source,
                    "content_hash": self._hash_file(source),
                }
        return {
            "lane_count": len(lane_list),
            "file_count": len(union),
            "files": union,
        }

    def _seed_overlay_workspace(self, workspace_dir: str) -> None:
        for entry in sorted(os.listdir(self.cwd)):
            if entry in {os.path.basename(self.base_dir), ".", ".."}:
                continue
            source = os.path.join(self.cwd, entry)
            target = os.path.join(workspace_dir, entry)
            os.symlink(source, target, target_is_directory=os.path.isdir(source))

    def _materialize_editable_file(self, workspace_dir: str, relative: str) -> None:
        source = os.path.join(self.cwd, relative)
        if not os.path.isfile(source):
            return
        parent_relative = os.path.dirname(relative)
        self._ensure_real_directory(workspace_dir, parent_relative)
        target = os.path.join(workspace_dir, relative)
        if os.path.lexists(target):
            if os.path.isdir(target) and not os.path.islink(target):
                shutil.rmtree(target)
            else:
                os.unlink(target)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(source, target)

    def _ensure_real_directory(self, workspace_dir: str, relative_dir: str) -> None:
        if not relative_dir:
            return
        current = workspace_dir
        source_current = self.cwd
        for part in [segment for segment in relative_dir.split(os.sep) if segment]:
            current = os.path.join(current, part)
            source_current = os.path.join(source_current, part)
            if os.path.islink(current):
                source_target = os.readlink(current)
                os.unlink(current)
                os.makedirs(current, exist_ok=True)
                for child in sorted(os.listdir(source_current)):
                    os.symlink(
                        os.path.join(source_target, child)
                        if os.path.isabs(source_target)
                        else os.path.join(source_current, child),
                        os.path.join(current, child),
                        target_is_directory=os.path.isdir(os.path.join(source_current, child)),
                    )
            elif not os.path.isdir(current):
                os.makedirs(current, exist_ok=True)

    def _relative_path(self, path: str) -> str:
        if not path:
            return ""
        resolved = os.path.abspath(path) if os.path.isabs(path) else os.path.abspath(os.path.join(self.cwd, path))
        try:
            relative = os.path.relpath(resolved, self.cwd)
        except ValueError:
            return ""
        if relative.startswith(".."):
            return ""
        return relative

    @staticmethod
    def _hash_file(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
