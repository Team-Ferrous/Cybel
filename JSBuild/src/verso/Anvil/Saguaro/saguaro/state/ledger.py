"""Append-only state ledger for workspace and sync operations."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import socket
import time
import uuid
from typing import Any

from saguaro.storage.atomic_fs import atomic_append_jsonl, atomic_write_json
from saguaro.storage.locks import RepoLockManager


def _now() -> float:
    return time.time()


def _load_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            value = json.load(f)
        return value
    except Exception:
        return default


def _write_json(path: str, payload: Any) -> None:
    atomic_write_json(path, payload, indent=2, sort_keys=True)


class StateLedger:
    """Persist deterministic workspace and sync state in `.saguaro/state`."""

    def __init__(self, repo_path: str, *, saguaro_dir: str | None = None) -> None:
        self.repo_path = os.path.abspath(repo_path)
        self.saguaro_dir = os.path.abspath(
            saguaro_dir or os.path.join(self.repo_path, ".saguaro")
        )
        self.state_dir = os.path.join(self.saguaro_dir, "state")
        self._lock_manager = RepoLockManager(self.saguaro_dir)
        self.events_path = os.path.join(self.state_dir, "events.jsonl")
        self.meta_path = os.path.join(self.state_dir, "meta.json")
        self.workspaces_path = os.path.join(self.state_dir, "workspaces.json")
        self.corpus_sessions_path = os.path.join(self.state_dir, "corpus_sessions.json")
        self.peers_path = os.path.join(self.state_dir, "peers.json")
        self.seen_events_path = os.path.join(self.state_dir, "seen_events.json")
        self.snapshots_path = os.path.join(self.state_dir, "snapshots.jsonl")
        self.outbox_dir = os.path.join(self.state_dir, "outbox")
        self._ensure_bootstrap()

    # -------------------------
    # Workspace lifecycle
    # -------------------------

    def create_workspace(
        self,
        name: str,
        description: str = "",
        *,
        switch: bool = False,
    ) -> dict[str, Any]:
        clean = self._sanitize_workspace_id(name)
        if not clean:
            raise ValueError("Workspace name cannot be empty.")
        payload = self._load_workspaces()
        if clean in payload["workspaces"]:
            return {
                "status": "exists",
                "workspace": payload["workspaces"][clean],
                "active": payload.get("active"),
            }
        now = _now()
        payload["workspaces"][clean] = {
            "workspace_id": clean,
            "name": name,
            "description": description or "",
            "created_at": now,
            "updated_at": now,
            "logical_clock": 0,
            "last_snapshot": None,
            "files": {},
        }
        if switch:
            payload["active"] = clean
        self._save_workspaces(payload)
        return {
            "status": "ok",
            "workspace": payload["workspaces"][clean],
            "active": payload["active"],
        }

    def switch_workspace(self, workspace_id: str) -> dict[str, Any]:
        clean = self._sanitize_workspace_id(workspace_id)
        payload = self._load_workspaces()
        ws = payload["workspaces"].get(clean)
        if ws is None:
            return {
                "status": "error",
                "message": f"Workspace '{workspace_id}' does not exist.",
            }
        payload["active"] = clean
        ws["updated_at"] = _now()
        self._save_workspaces(payload)
        return {"status": "ok", "active": clean, "workspace": ws}

    def workspace_status(
        self, workspace_id: str | None = None, limit: int = 200
    ) -> dict[str, Any]:
        payload = self._load_workspaces()
        active = self._resolve_workspace_id(payload, workspace_id)
        ws = payload["workspaces"][active]
        files = ws.get("files", {})
        alive = sorted(
            path for path, meta in files.items() if not bool(meta.get("deleted", False))
        )
        deleted = sorted(
            path for path, meta in files.items() if bool(meta.get("deleted", False))
        )
        return {
            "status": "ok",
            "instance_id": self.instance_id,
            "workspace_id": active,
            "active": payload.get("active"),
            "workspace_count": len(payload.get("workspaces", {})),
            "tracked_files": len(alive),
            "deleted_files": len(deleted),
            "tracked_sample": alive[:limit],
            "deleted_sample": deleted[:limit],
            "latest_event": self.watermark(),
        }

    def list_workspaces(self) -> dict[str, Any]:
        payload = self._load_workspaces()
        workspaces = sorted(
            payload.get("workspaces", {}).values(),
            key=lambda item: str(item.get("workspace_id", "")),
        )
        return {
            "status": "ok",
            "active": payload.get("active"),
            "count": len(workspaces),
            "workspaces": workspaces,
        }

    def workspace_file_set(self, workspace_id: str | None = None) -> set[str]:
        payload = self._load_workspaces()
        active = self._resolve_workspace_id(payload, workspace_id)
        ws = payload["workspaces"].get(active, {})
        files = ws.get("files", {})
        alive = {
            str(path).replace("\\", "/")
            for path, meta in files.items()
            if not bool((meta or {}).get("deleted", False))
        }
        return alive

    def workspace_history(
        self, workspace_id: str | None = None, limit: int = 200
    ) -> dict[str, Any]:
        active = self.current_workspace_id(workspace_id=workspace_id)
        events = self.list_events(workspace_id=active, limit=limit)
        return {
            "status": "ok",
            "workspace_id": active,
            "count": len(events),
            "events": events,
        }

    def workspace_diff(
        self,
        *,
        workspace_id: str | None = None,
        against: str = "main",
        limit: int = 200,
    ) -> dict[str, Any]:
        payload = self._load_workspaces()
        left_id = self._resolve_workspace_id(payload, workspace_id)
        right_id = self._resolve_workspace_id(payload, against)
        left_files = payload["workspaces"][left_id].get("files", {})
        right_files = payload["workspaces"][right_id].get("files", {})

        left_alive = {
            path: str(meta.get("hash") or "")
            for path, meta in left_files.items()
            if not bool(meta.get("deleted", False))
        }
        right_alive = {
            path: str(meta.get("hash") or "")
            for path, meta in right_files.items()
            if not bool(meta.get("deleted", False))
        }
        left_paths = set(left_alive)
        right_paths = set(right_alive)
        added = sorted(left_paths - right_paths)
        removed = sorted(right_paths - left_paths)
        changed = sorted(
            path
            for path in (left_paths & right_paths)
            if left_alive[path] != right_alive[path]
        )
        return {
            "status": "ok",
            "workspace_id": left_id,
            "against": right_id,
            "added_count": len(added),
            "removed_count": len(removed),
            "changed_count": len(changed),
            "added": added[:limit],
            "removed": removed[:limit],
            "changed": changed[:limit],
        }

    # -------------------------
    # Corpus sessions
    # -------------------------

    def create_corpus_session(self, session: dict[str, Any]) -> dict[str, Any]:
        payload = self._load_corpus_sessions()
        corpus_id = self._sanitize_workspace_id(str(session.get("corpus_id") or ""))
        if not corpus_id:
            raise ValueError("corpus_id is required.")
        now = _now()
        entry = dict(session)
        entry["corpus_id"] = corpus_id
        entry["created_at"] = float(entry.get("created_at") or now)
        entry["updated_at"] = now
        entry["last_accessed_at"] = float(entry.get("last_accessed_at") or now)
        entry["expires_at"] = float(
            entry.get("expires_at")
            or (entry["created_at"] + float(entry.get("ttl_seconds", 86400) or 86400))
        )
        payload["sessions"][corpus_id] = entry
        self._save_corpus_sessions(payload)
        return {"status": "ok", "session": entry}

    def get_corpus_session(
        self,
        corpus_id: str,
        *,
        touch: bool = False,
    ) -> dict[str, Any] | None:
        clean = self._sanitize_workspace_id(corpus_id)
        payload = self._load_corpus_sessions()
        session = payload["sessions"].get(clean)
        if session is None:
            return None
        if touch:
            session["last_accessed_at"] = _now()
            payload["sessions"][clean] = session
            self._save_corpus_sessions(payload)
        return session

    def list_corpus_sessions(self, *, include_expired: bool = False) -> dict[str, Any]:
        now = _now()
        payload = self._load_corpus_sessions()
        sessions = []
        for session in payload["sessions"].values():
            expires_at = float(session.get("expires_at", 0.0) or 0.0)
            expired = expires_at > 0.0 and expires_at <= now
            session["expired"] = expired
            if expired and not include_expired:
                continue
            sessions.append(session)
        sessions.sort(key=lambda item: (item.get("expired", False), str(item.get("corpus_id", ""))))
        return {
            "status": "ok",
            "count": len(sessions),
            "sessions": sessions,
        }

    def gc_corpus_sessions(self) -> dict[str, Any]:
        now = _now()
        payload = self._load_corpus_sessions()
        removed: list[str] = []
        survivors: dict[str, Any] = {}
        for corpus_id, session in payload["sessions"].items():
            expires_at = float(session.get("expires_at", 0.0) or 0.0)
            if expires_at > 0.0 and expires_at <= now:
                removed.append(corpus_id)
                index_root = str(session.get("index_root") or "").strip()
                if index_root and os.path.exists(index_root):
                    shutil.rmtree(index_root, ignore_errors=True)
                continue
            survivors[corpus_id] = session
        payload["sessions"] = survivors
        self._save_corpus_sessions(payload)
        return {
            "status": "ok",
            "removed_count": len(removed),
            "removed": removed,
            "corpus_session_gc_ms": 0.0,
        }

    def snapshot(
        self, label: str = "manual", workspace_id: str | None = None
    ) -> dict[str, Any]:
        ws_id = self.current_workspace_id(workspace_id=workspace_id)
        payload = self._load_workspaces()
        ws = payload["workspaces"][ws_id]
        alive = {
            path: str(meta.get("hash") or "")
            for path, meta in ws.get("files", {}).items()
            if not bool(meta.get("deleted", False))
        }
        digest_input = "\n".join(f"{path}:{alive[path]}" for path in sorted(alive))
        state_hash = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
        record = {
            "snapshot_id": hashlib.sha256(
                f"{self.instance_id}:{ws_id}:{label}:{_now()}".encode("utf-8")
            ).hexdigest()[:16],
            "workspace_id": ws_id,
            "label": label,
            "created_at": _now(),
            "state_hash": state_hash,
            "tracked_files": len(alive),
            "logical_clock": int(ws.get("logical_clock", 0)),
        }
        self._append_jsonl(self.snapshots_path, record)
        ws["last_snapshot"] = record["snapshot_id"]
        ws["updated_at"] = _now()
        payload["workspaces"][ws_id] = ws
        self._save_workspaces(payload)
        return {"status": "ok", "snapshot": record}

    # -------------------------
    # Event stream
    # -------------------------

    @property
    def instance_id(self) -> str:
        return str(self._load_meta().get("instance_id"))

    def current_workspace_id(self, workspace_id: str | None = None) -> str:
        payload = self._load_workspaces()
        return self._resolve_workspace_id(payload, workspace_id)

    def watermark(self) -> dict[str, Any]:
        return self.delta_watermark()

    def delta_watermark(self, workspace_id: str | None = None) -> dict[str, Any]:
        events = self.list_events(workspace_id=workspace_id, limit=1)
        if not events:
            git_head, git_status = self._git_snapshot()
            return {
                "event_id": None,
                "delta_id": None,
                "parent_delta_id": None,
                "logical_clock": 0,
                "created_at": None,
                "workspace_id": self.current_workspace_id(workspace_id=workspace_id),
                "git_head": git_head,
                "git_status": git_status,
                "changed_paths": [],
            }
        latest = events[-1]
        return {
            "event_id": latest.get("event_id"),
            "delta_id": latest.get("delta_id"),
            "parent_delta_id": latest.get("parent_delta_id"),
            "logical_clock": int(latest.get("logical_clock", 0) or 0),
            "created_at": latest.get("created_at"),
            "workspace_id": latest.get("workspace_id"),
            "git_head": latest.get("git_head"),
            "git_status": latest.get("git_status"),
            "changed_paths": (
                [str(latest.get("path") or "")] if latest.get("path") else []
            ),
        }

    def delta_stream(
        self,
        *,
        workspace_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        return self.list_events(workspace_id=workspace_id, limit=limit)

    def delta_since(
        self,
        watermark: dict[str, Any] | None,
        *,
        workspace_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        baseline = int((watermark or {}).get("logical_clock", 0) or 0)
        deltas = [
            event
            for event in self.list_events(workspace_id=workspace_id, limit=0)
            if int(event.get("logical_clock", 0) or 0) > baseline
        ]
        if limit > 0:
            return deltas[:limit]
        return deltas

    def changeset_since(
        self,
        watermark: dict[str, Any] | None,
        *,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        deltas = self.delta_since(watermark, workspace_id=workspace_id, limit=0)
        changed = sorted(
            {
                str(event.get("path") or "")
                for event in deltas
                if str(event.get("op") or "upsert") != "delete"
                and str(event.get("path") or "")
            }
        )
        deleted = sorted(
            {
                str(event.get("path") or "")
                for event in deltas
                if str(event.get("op") or "") == "delete"
                and str(event.get("path") or "")
            }
        )
        return {
            "workspace_id": self.current_workspace_id(workspace_id=workspace_id),
            "delta_count": len(deltas),
            "changed_files": changed,
            "deleted_files": deleted,
            "watermark": self.delta_watermark(workspace_id=workspace_id),
        }

    def workspace_file_records(
        self,
        *,
        workspace_id: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        payload = self._load_workspaces()
        active = self._resolve_workspace_id(payload, workspace_id)
        workspace = payload["workspaces"].get(active, {})
        records = workspace.get("files", {})
        normalized: dict[str, dict[str, Any]] = {}
        for path, meta in records.items():
            rel = self._normalize_rel_path(path)
            if not rel:
                continue
            entry = dict(meta or {})
            normalized[rel] = {
                "hash": str(entry.get("hash") or ""),
                "mtime_ns": int(entry.get("mtime_ns", 0) or 0),
                "logical_clock": int(entry.get("logical_clock", 0) or 0),
                "deleted": bool(entry.get("deleted", False)),
            }
        return normalized

    def compare_with_filesystem(
        self,
        candidate_files: list[str],
        *,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        records = self.workspace_file_records(workspace_id=workspace_id)
        normalized_candidates = {
            self._normalize_rel_path(path)
            for path in (candidate_files or [])
            if self._normalize_rel_path(path)
        }
        changed: list[str] = []
        unchanged: list[str] = []
        for rel in sorted(normalized_candidates):
            full = os.path.join(self.repo_path, rel)
            record = records.get(rel, {})
            if not os.path.exists(full):
                continue
            current_hash = self._hash_file(full)
            current_mtime_ns = self._safe_mtime_ns(full)
            if (
                bool(record.get("deleted", False))
                or str(record.get("hash") or "") != current_hash
                or int(record.get("mtime_ns", 0) or 0) != current_mtime_ns
            ):
                changed.append(rel)
            else:
                unchanged.append(rel)

        tracked_alive = {
            path
            for path, meta in records.items()
            if not bool((meta or {}).get("deleted", False))
        }
        deleted = sorted(path for path in tracked_alive if path not in normalized_candidates)
        return {
            "workspace_id": self.current_workspace_id(workspace_id=workspace_id),
            "changed_files": changed,
            "deleted_files": deleted,
            "unchanged_files": sorted(unchanged),
            "watermark": self.delta_watermark(workspace_id=workspace_id),
        }

    def state_projection_lines(
        self,
        *,
        workspace_id: str | None = None,
        include_deleted: bool = False,
    ) -> list[str]:
        records = self.workspace_file_records(workspace_id=workspace_id)
        lines: list[str] = []
        for path, meta in sorted(records.items()):
            if not include_deleted and bool(meta.get("deleted", False)):
                continue
            hash_value = str(meta.get("hash") or "")
            deleted = "1" if bool(meta.get("deleted", False)) else "0"
            logical_clock = int(meta.get("logical_clock", 0) or 0)
            lines.append(f"{path}:{hash_value}:{deleted}:{logical_clock}")
        return lines

    def state_projection_blob(
        self,
        *,
        workspace_id: str | None = None,
        include_deleted: bool = False,
    ) -> bytes:
        lines = self.state_projection_lines(
            workspace_id=workspace_id,
            include_deleted=include_deleted,
        )
        if not lines:
            return b""
        return ("\n".join(lines) + "\n").encode("utf-8")

    def list_events(
        self, workspace_id: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        target = (
            self.current_workspace_id(workspace_id=workspace_id)
            if workspace_id
            else None
        )
        events: list[dict[str, Any]] = []
        if not os.path.exists(self.events_path):
            return events
        try:
            with open(self.events_path, encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        event = json.loads(raw)
                    except Exception:
                        continue
                    if target and str(event.get("workspace_id")) != target:
                        continue
                    events.append(event)
        except Exception:
            return []
        if limit <= 0:
            return events
        return events[-limit:]

    def record_changes(
        self,
        changed_files: list[str] | None = None,
        deleted_files: list[str] | None = None,
        *,
        workspace_id: str | None = None,
        reason: str = "sync",
        source_instance_id: str | None = None,
    ) -> dict[str, Any]:
        ws_id = self.current_workspace_id(workspace_id=workspace_id)
        payload = self._load_workspaces()
        ws = payload["workspaces"][ws_id]
        files = ws.setdefault("files", {})

        changed = [self._normalize_rel_path(path) for path in (changed_files or [])]
        deleted = [self._normalize_rel_path(path) for path in (deleted_files or [])]
        changed = sorted(
            {path for path in changed if path and path not in set(deleted)}
        )
        deleted = sorted({path for path in deleted if path})

        events_written = 0
        for rel_path in changed:
            full = os.path.join(self.repo_path, rel_path)
            if not os.path.exists(full):
                continue
            before = str((files.get(rel_path) or {}).get("hash") or "")
            after = self._hash_file(full)
            if after and before == after:
                continue
            event = self._build_event(
                workspace_id=ws_id,
                rel_path=rel_path,
                op="upsert",
                content_hash_before=before or None,
                content_hash_after=after or None,
                mtime_ns=self._safe_mtime_ns(full),
                reason=reason,
                source_instance_id=source_instance_id,
            )
            if self._append_event(event):
                files[rel_path] = {
                    "hash": after,
                    "mtime_ns": event["mtime_ns"],
                    "logical_clock": event["logical_clock"],
                    "deleted": False,
                }
                events_written += 1

        for rel_path in deleted:
            before = str((files.get(rel_path) or {}).get("hash") or "")
            event = self._build_event(
                workspace_id=ws_id,
                rel_path=rel_path,
                op="delete",
                content_hash_before=before or None,
                content_hash_after=None,
                mtime_ns=0,
                reason=reason,
                source_instance_id=source_instance_id,
            )
            if self._append_event(event):
                files[rel_path] = {
                    "hash": None,
                    "mtime_ns": 0,
                    "logical_clock": event["logical_clock"],
                    "deleted": True,
                }
                events_written += 1

        ws["updated_at"] = _now()
        ws["logical_clock"] = int(
            max((v.get("logical_clock", 0) for v in files.values()), default=0)
        )
        payload["workspaces"][ws_id] = ws
        self._save_workspaces(payload)
        return {"workspace_id": ws_id, "events_written": events_written}

    def apply_peer_events(
        self,
        events: list[dict[str, Any]],
        *,
        workspace_id: str | None = None,
        peer_id: str | None = None,
    ) -> dict[str, Any]:
        ws_id = self.current_workspace_id(workspace_id=workspace_id)
        payload = self._load_workspaces()
        ws = payload["workspaces"][ws_id]
        files = ws.setdefault("files", {})
        applied = 0
        for item in events:
            if not isinstance(item, dict):
                continue
            rel_path = self._normalize_rel_path(str(item.get("path") or ""))
            if not rel_path:
                continue
            op = str(item.get("op") or "upsert")
            source_instance = str(item.get("instance_id") or "")
            event = dict(item)
            event["workspace_id"] = ws_id
            event["path"] = rel_path
            event.setdefault("reason", "peer_pull")
            if peer_id and not event.get("peer_id"):
                event["peer_id"] = peer_id
            if not event.get("event_id"):
                event = self._build_event(
                    workspace_id=ws_id,
                    rel_path=rel_path,
                    op=op,
                    content_hash_before=event.get("content_hash_before"),
                    content_hash_after=event.get("content_hash_after"),
                    mtime_ns=int(event.get("mtime_ns", 0) or 0),
                    reason=str(event.get("reason", "peer_pull")),
                    source_instance_id=source_instance or None,
                )
            if not self._append_event(event):
                continue
            if op == "delete":
                files[rel_path] = {
                    "hash": None,
                    "mtime_ns": 0,
                    "logical_clock": int(event.get("logical_clock", 0) or 0),
                    "deleted": True,
                }
            else:
                files[rel_path] = {
                    "hash": event.get("content_hash_after"),
                    "mtime_ns": int(event.get("mtime_ns", 0) or 0),
                    "logical_clock": int(event.get("logical_clock", 0) or 0),
                    "deleted": False,
                }
            applied += 1
        ws["updated_at"] = _now()
        ws["logical_clock"] = int(
            max((v.get("logical_clock", 0) for v in files.values()), default=0)
        )
        payload["workspaces"][ws_id] = ws
        self._save_workspaces(payload)
        return {"workspace_id": ws_id, "applied": applied}

    # -------------------------
    # Peer metadata + local sync bundles
    # -------------------------

    def peer_add(
        self, name: str, url: str, auth_token: str | None = None
    ) -> dict[str, Any]:
        peer_name = str(name or "").strip()
        peer_url = str(url or "").strip()
        if not peer_name or not peer_url:
            raise ValueError("Both peer name and URL are required.")
        peers = self._load_peers()
        peer_id = hashlib.sha256(f"{peer_name}:{peer_url}".encode("utf-8")).hexdigest()[
            :12
        ]
        peers[peer_id] = {
            "peer_id": peer_id,
            "name": peer_name,
            "url": peer_url,
            "auth_token": auth_token or "",
            "created_at": _now(),
            "updated_at": _now(),
            "subscribed": bool((peers.get(peer_id) or {}).get("subscribed", False)),
        }
        self._save_peers(peers)
        return {"status": "ok", "peer": peers[peer_id]}

    def peer_remove(self, peer_id: str) -> dict[str, Any]:
        peers = self._load_peers()
        clean = str(peer_id or "").strip()
        if clean not in peers:
            return {"status": "error", "message": f"Peer '{peer_id}' not found."}
        removed = peers.pop(clean)
        self._save_peers(peers)
        return {"status": "ok", "peer": removed}

    def peer_list(self) -> dict[str, Any]:
        peers = self._load_peers()
        ordered = sorted(
            peers.values(), key=lambda item: str(item.get("name", "")).lower()
        )
        return {"status": "ok", "count": len(ordered), "peers": ordered}

    def sync_serve(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "instance_id": self.instance_id,
            "workspace_id": self.current_workspace_id(),
            "events_path": self.events_path,
            "peers": self.peer_list().get("count", 0),
            "watermark": self.watermark(),
        }

    def sync_push(
        self, peer_id: str, limit: int = 1000, workspace_id: str | None = None
    ) -> dict[str, Any]:
        peer = self._load_peers().get(str(peer_id or "").strip())
        if peer is None:
            return {"status": "error", "message": f"Peer '{peer_id}' not found."}
        events = self.list_events(workspace_id=workspace_id, limit=limit)
        bundle = {
            "instance_id": self.instance_id,
            "workspace_id": self.current_workspace_id(workspace_id=workspace_id),
            "generated_at": _now(),
            "events": events,
        }
        os.makedirs(self.outbox_dir, exist_ok=True)
        file_name = f"{peer['peer_id']}_{int(_now())}.json"
        out_path = os.path.join(self.outbox_dir, file_name)
        _write_json(out_path, bundle)
        peer["updated_at"] = _now()
        peers = self._load_peers()
        peers[peer["peer_id"]] = peer
        self._save_peers(peers)
        return {
            "status": "ok",
            "peer_id": peer["peer_id"],
            "bundle_path": out_path,
            "events": len(events),
        }

    def sync_pull(
        self,
        peer_id: str,
        bundle_path: str,
        *,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        peer = self._load_peers().get(str(peer_id or "").strip())
        if peer is None:
            return {"status": "error", "message": f"Peer '{peer_id}' not found."}
        target = os.path.abspath(bundle_path) if bundle_path else ""
        if not target or not os.path.exists(target):
            return {
                "status": "error",
                "message": f"Bundle path not found: {bundle_path}",
            }
        payload = _load_json(target, {})
        events = payload.get("events", []) if isinstance(payload, dict) else []
        applied = self.apply_peer_events(
            [event for event in events if isinstance(event, dict)],
            workspace_id=workspace_id,
            peer_id=peer["peer_id"],
        )
        peer["updated_at"] = _now()
        peers = self._load_peers()
        peers[peer["peer_id"]] = peer
        self._save_peers(peers)
        return {
            "status": "ok",
            "peer_id": peer["peer_id"],
            "bundle_path": target,
            "applied_events": int(applied.get("applied", 0) or 0),
            "workspace_id": applied.get("workspace_id"),
        }

    def sync_subscribe(self, peer_id: str, enabled: bool = True) -> dict[str, Any]:
        peers = self._load_peers()
        clean = str(peer_id or "").strip()
        peer = peers.get(clean)
        if peer is None:
            return {"status": "error", "message": f"Peer '{peer_id}' not found."}
        peer["subscribed"] = bool(enabled)
        peer["updated_at"] = _now()
        peers[clean] = peer
        self._save_peers(peers)
        return {"status": "ok", "peer": peer}

    # -------------------------
    # Internal helpers
    # -------------------------

    def _ensure_bootstrap(self) -> None:
        os.makedirs(self.state_dir, exist_ok=True)
        os.makedirs(self.outbox_dir, exist_ok=True)

        if not os.path.exists(self.meta_path):
            _write_json(
                self.meta_path,
                {
                    "instance_id": str(uuid.uuid4()),
                    "hostname": socket.gethostname(),
                    "created_at": _now(),
                    "logical_clock": 0,
                },
            )

        if not os.path.exists(self.workspaces_path):
            now = _now()
            _write_json(
                self.workspaces_path,
                {
                    "active": "main",
                    "workspaces": {
                        "main": {
                            "workspace_id": "main",
                            "name": "main",
                            "description": "default workspace",
                            "created_at": now,
                            "updated_at": now,
                            "logical_clock": 0,
                            "last_snapshot": None,
                            "files": {},
                        }
                    },
                },
            )

        if not os.path.exists(self.corpus_sessions_path):
            _write_json(self.corpus_sessions_path, {"sessions": {}})
        if not os.path.exists(self.peers_path):
            _write_json(self.peers_path, {})
        if not os.path.exists(self.seen_events_path):
            _write_json(self.seen_events_path, {})

        for path in (self.events_path, self.snapshots_path):
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8"):
                    pass

    def _append_jsonl(self, path: str, payload: dict[str, Any]) -> None:
        with self._lock_manager.acquire(
            "state",
            mode="exclusive",
            operation=f"append:{os.path.basename(path)}",
        ):
            atomic_append_jsonl(path, payload)

    def _load_meta(self) -> dict[str, Any]:
        return _load_json(
            self.meta_path,
            {
                "instance_id": str(uuid.uuid4()),
                "hostname": socket.gethostname(),
                "created_at": _now(),
                "logical_clock": 0,
            },
        )

    def _save_meta(self, payload: dict[str, Any]) -> None:
        with self._lock_manager.acquire(
            "state", mode="exclusive", operation="meta_save"
        ):
            _write_json(self.meta_path, payload)

    def _next_clock(self) -> int:
        meta = self._load_meta()
        next_value = int(meta.get("logical_clock", 0) or 0) + 1
        meta["logical_clock"] = next_value
        self._save_meta(meta)
        return next_value

    def _load_workspaces(self) -> dict[str, Any]:
        payload = _load_json(self.workspaces_path, {})
        workspaces = payload.get("workspaces")
        if not isinstance(workspaces, dict) or not workspaces:
            now = _now()
            payload = {
                "active": "main",
                "workspaces": {
                    "main": {
                        "workspace_id": "main",
                        "name": "main",
                        "description": "default workspace",
                        "created_at": now,
                        "updated_at": now,
                        "logical_clock": 0,
                        "last_snapshot": None,
                        "files": {},
                    }
                },
            }
        if not payload.get("active") or payload["active"] not in payload["workspaces"]:
            payload["active"] = sorted(payload["workspaces"].keys())[0]
        return payload

    def _save_workspaces(self, payload: dict[str, Any]) -> None:
        with self._lock_manager.acquire(
            "state",
            mode="exclusive",
            operation="workspaces_save",
        ):
            _write_json(self.workspaces_path, payload)

    def _load_corpus_sessions(self) -> dict[str, Any]:
        payload = _load_json(self.corpus_sessions_path, {})
        sessions = payload.get("sessions")
        if not isinstance(sessions, dict):
            payload = {"sessions": {}}
        return payload

    def _save_corpus_sessions(self, payload: dict[str, Any]) -> None:
        with self._lock_manager.acquire(
            "state",
            mode="exclusive",
            operation="corpus_sessions_save",
        ):
            _write_json(self.corpus_sessions_path, payload)

    def _load_peers(self) -> dict[str, Any]:
        payload = _load_json(self.peers_path, {})
        if isinstance(payload, dict):
            return payload
        return {}

    def _save_peers(self, payload: dict[str, Any]) -> None:
        with self._lock_manager.acquire(
            "state", mode="exclusive", operation="peers_save"
        ):
            _write_json(self.peers_path, payload)

    def _load_seen_events(self) -> dict[str, float]:
        payload = _load_json(self.seen_events_path, {})
        if isinstance(payload, dict):
            return {str(k): float(v) for k, v in payload.items()}
        return {}

    def _save_seen_events(self, payload: dict[str, float]) -> None:
        # Keep seen-event set bounded.
        if len(payload) > 50000:
            ordered = sorted(payload.items(), key=lambda item: item[1], reverse=True)[
                :40000
            ]
            payload = dict(ordered)
        with self._lock_manager.acquire(
            "state",
            mode="exclusive",
            operation="seen_events_save",
        ):
            _write_json(self.seen_events_path, payload)

    def _append_event(self, event: dict[str, Any]) -> bool:
        event_id = str(event.get("event_id") or "").strip()
        if not event_id:
            return False
        seen = self._load_seen_events()
        if event_id in seen:
            return False
        self._append_jsonl(self.events_path, event)
        seen[event_id] = _now()
        self._save_seen_events(seen)
        return True

    def _build_event(
        self,
        *,
        workspace_id: str,
        rel_path: str,
        op: str,
        content_hash_before: str | None,
        content_hash_after: str | None,
        mtime_ns: int,
        reason: str,
        source_instance_id: str | None = None,
    ) -> dict[str, Any]:
        logical_clock = self._next_clock()
        instance_id = source_instance_id or self.instance_id
        previous = self.delta_watermark(workspace_id=workspace_id)
        git_head, git_status = self._git_snapshot()
        event_seed = (
            f"{instance_id}:{workspace_id}:{rel_path}:{op}:{logical_clock}:"
            f"{content_hash_before or ''}:{content_hash_after or ''}:{mtime_ns}"
        )
        event_id = hashlib.sha256(event_seed.encode("utf-8")).hexdigest()
        return {
            "event_id": event_id,
            "delta_id": event_id,
            "parent_delta_id": previous.get("delta_id"),
            "instance_id": instance_id,
            "workspace_id": workspace_id,
            "path": rel_path,
            "op": op,
            "actor": instance_id,
            "change_kind": op,
            "content_hash_before": content_hash_before,
            "content_hash_after": content_hash_after,
            "mtime_ns": int(mtime_ns or 0),
            "logical_clock": int(logical_clock),
            "created_at": _now(),
            "reason": reason,
            "timestamp": _now(),
            "git_head": git_head,
            "git_status": git_status,
            "graph_dirty": True,
            "chronicle_dirty": True,
            "affected_symbols": [],
            "semantic_impact_hint": "",
        }

    def _git_snapshot(self) -> tuple[str, str]:
        try:
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_path,
                check=False,
                capture_output=True,
                text=True,
            )
            status = subprocess.run(
                ["git", "status", "--short"],
                cwd=self.repo_path,
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception:
            return "", ""
        return (
            head.stdout.strip() if head.returncode == 0 else "",
            status.stdout.strip() if status.returncode == 0 else "",
        )

    def _resolve_workspace_id(
        self, payload: dict[str, Any], requested: str | None
    ) -> str:
        clean = self._sanitize_workspace_id(requested or "")
        if clean and clean in payload["workspaces"]:
            return clean
        return str(payload.get("active") or "main")

    def _sanitize_workspace_id(self, value: str) -> str:
        clean = "".join(
            ch if ch.isalnum() or ch in {"-", "_"} else "-"
            for ch in str(value).strip().lower()
        )
        clean = clean.strip("-_")
        return clean

    def _normalize_rel_path(self, path: str) -> str:
        raw = str(path or "").strip()
        if not raw:
            return ""
        absolute = raw if os.path.isabs(raw) else os.path.join(self.repo_path, raw)
        absolute = os.path.abspath(absolute)
        if absolute.startswith(self.repo_path):
            rel = os.path.relpath(absolute, self.repo_path)
        else:
            rel = raw
        return rel.replace("\\", "/")

    def _hash_file(self, path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _safe_mtime_ns(self, path: str) -> int:
        try:
            return int(os.stat(path).st_mtime_ns)
        except Exception:
            return 0
