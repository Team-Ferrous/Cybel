"""Single-write-path coordinator for incremental indexing workflows."""

from __future__ import annotations

import os
from typing import Any

from saguaro.api import SaguaroAPI
from saguaro.indexing.tracker import IndexTracker
from saguaro.utils.file_utils import get_code_files


class IndexCoordinator:
    """Coordinate deterministic change discovery and index ingestion."""

    def __init__(self, repo_path: str = ".", api: SaguaroAPI | None = None) -> None:
        self.repo_path = os.path.abspath(repo_path)
        self.saguaro_dir = os.path.join(self.repo_path, ".saguaro")
        self.api = api or SaguaroAPI(self.repo_path)
        self.tracker = IndexTracker(os.path.join(self.saguaro_dir, "tracking.json"))

    def discover_changes(
        self,
        *,
        path: str = ".",
        prune_deleted: bool = True,
    ) -> dict[str, list[str]]:
        """Discover changed/deleted files against ledger-backed repo state."""
        tracker_watermark = self.tracker.load_watermark()
        if tracker_watermark:
            try:
                changeset = self.api.changeset_since(tracker_watermark)
            except Exception:
                changeset = {}
            changed_files = sorted(
                self._normalize_rel_paths(
                    list(changeset.get("changed_files", []) or [])
                )
            )
            deleted_files = sorted(
                self._normalize_rel_paths(
                    list(changeset.get("deleted_files", []) or [])
                )
            )
            if changed_files or deleted_files:
                return {
                    "changed_files": changed_files,
                    "deleted_files": deleted_files,
                }
        target_path = (
            path
            if os.path.isabs(path)
            else os.path.abspath(os.path.join(self.repo_path, path))
        )
        files = get_code_files(target_path)
        filesystem = self.api._state_ledger.compare_with_filesystem(  # noqa: SLF001
            [self._to_repo_rel(item) for item in files]
        )
        changed_rel = sorted(
            self._normalize_rel_paths(list(filesystem.get("changed_files", []) or []))
        )
        deleted_rel = (
            sorted(
                self._normalize_rel_paths(
                    list(filesystem.get("deleted_files", []) or [])
                )
            )
            if prune_deleted
            else []
        )
        return {
            "changed_files": changed_rel,
            "deleted_files": deleted_rel,
        }

    def sync(
        self,
        *,
        path: str = ".",
        changed_files: list[str] | None = None,
        deleted_files: list[str] | None = None,
        full: bool = False,
        reason: str = "coordinator",
        prune_deleted: bool = True,
        events_path: str | None = None,
    ) -> dict[str, Any]:
        """Route all ingestion through API.sync/API.index with deterministic semantics."""
        if full:
            result = self.api.sync(action="index", full=True, reason=reason)
            try:
                self.tracker.save_watermark(
                    dict(result.get("watermark") or self.api.watermark())
                )
            except Exception:
                pass
            return result

        if changed_files is None and deleted_files is None:
            discovered = self.discover_changes(path=path, prune_deleted=prune_deleted)
            changed_files = discovered.get("changed_files", [])
            deleted_files = discovered.get("deleted_files", [])
            if not changed_files and not deleted_files and not events_path:
                return {
                    "status": "ok",
                    "action": "index",
                    "index": {
                        "status": "noop",
                        "indexed_files": 0,
                        "indexed_entities": 0,
                        "removed_files": 0,
                        "updated_files": 0,
                    },
                    "events": {
                        "workspace_id": self.api.current_workspace_id(),
                        "events_written": 0,
                    },
                    "watermark": self.api.watermark(),
                }
        changed_files = self._normalize_rel_paths(changed_files or [])
        deleted_files = self._normalize_rel_paths(deleted_files or [])

        if events_path:
            before_clock = int(self.api.watermark().get("logical_clock", 0) or 0)
            index_payload = self.api.index(
                path=path,
                force=False,
                incremental=True,
                changed_files=changed_files,
                events_path=events_path,
                prune_deleted=prune_deleted,
            )
            after_watermark = self.api.watermark()
            after_clock = int(after_watermark.get("logical_clock", 0) or 0)
            try:
                self.tracker.save_watermark(dict(after_watermark))
            except Exception:
                pass
            return {
                "status": "ok",
                "action": "index",
                "index": index_payload,
                "events": {
                    "workspace_id": self.api.current_workspace_id(),
                    "events_written": max(after_clock - before_clock, 0),
                },
                "watermark": after_watermark,
            }

        result = self.api.sync(
            action="index",
            changed_files=changed_files,
            deleted_files=deleted_files,
            full=False,
            reason=reason,
        )
        try:
            self.tracker.save_watermark(
                dict(result.get("watermark") or self.api.watermark())
            )
        except Exception:
            pass
        return result

    def _to_repo_rel(self, value: str) -> str:
        abs_path = (
            os.path.abspath(value)
            if os.path.isabs(value)
            else os.path.abspath(os.path.join(self.repo_path, value))
        )
        rel = os.path.relpath(abs_path, self.repo_path)
        return rel.replace("\\", "/")

    def _normalize_rel_paths(self, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            rel = self._to_repo_rel(value)
            if rel.startswith("Saguaro/"):
                rel = rel[len("Saguaro/") :]
            while rel.startswith("saguaro/saguaro/"):
                rel = rel[len("saguaro/") :]
            if rel and rel not in normalized:
                normalized.append(rel)
        return sorted(normalized)
