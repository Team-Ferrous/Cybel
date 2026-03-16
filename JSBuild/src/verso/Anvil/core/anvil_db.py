from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Optional[Dict[str, Any]]) -> str:
    return json.dumps(value or {}, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True)
class TimelineEvent:
    event_type: str
    wall_clock: str
    monotonic_elapsed_ms: Optional[int]
    payload: Dict[str, Any]


class AnvilDB:
    """SQLite-backed persistence and audit store for Anvil."""

    def __init__(self, db_path: str = ".anvil/anvil.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterable[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        objective TEXT,
                        metadata TEXT NOT NULL DEFAULT '{}'
                    );

                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        monotonic_elapsed_ms INTEGER,
                        metadata TEXT NOT NULL DEFAULT '{}',
                        summary_message_id INTEGER,
                        FOREIGN KEY(session_id) REFERENCES sessions(id)
                    );
                    CREATE INDEX IF NOT EXISTS idx_messages_session_ts
                        ON messages(session_id, timestamp);

                    CREATE TABLE IF NOT EXISTS file_operations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        path TEXT NOT NULL,
                        operation TEXT NOT NULL,
                        status TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        monotonic_elapsed_ms INTEGER,
                        freshness_token TEXT,
                        details TEXT NOT NULL DEFAULT '{}',
                        trace_id TEXT
                    );
                    CREATE INDEX IF NOT EXISTS idx_file_ops_session_ts
                        ON file_operations(session_id, timestamp);

                    CREATE TABLE IF NOT EXISTS file_versions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        path TEXT NOT NULL,
                        operation_id INTEGER,
                        hash_before TEXT,
                        hash_after TEXT,
                        backup_path TEXT,
                        git_head TEXT,
                        timestamp TEXT NOT NULL,
                        trace_id TEXT,
                        details TEXT NOT NULL DEFAULT '{}',
                        FOREIGN KEY(operation_id) REFERENCES file_operations(id)
                    );
                    CREATE INDEX IF NOT EXISTS idx_file_versions_session_ts
                        ON file_versions(session_id, timestamp);

                    CREATE TABLE IF NOT EXISTS permission_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        tool_name TEXT NOT NULL,
                        decision TEXT NOT NULL,
                        reason TEXT,
                        policy_profile TEXT,
                        timestamp TEXT NOT NULL,
                        signature TEXT,
                        metadata TEXT NOT NULL DEFAULT '{}'
                    );
                    CREATE INDEX IF NOT EXISTS idx_permission_events_session_ts
                        ON permission_events(session_id, timestamp);

                    CREATE TABLE IF NOT EXISTS task_state (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        task_id TEXT,
                        state TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        metadata TEXT NOT NULL DEFAULT '{}'
                    );
                    CREATE INDEX IF NOT EXISTS idx_task_state_session_ts
                        ON task_state(session_id, timestamp);

                    CREATE TABLE IF NOT EXISTS agent_diagnostics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        source TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        message TEXT NOT NULL,
                        file_path TEXT,
                        line INTEGER,
                        col INTEGER,
                        timestamp TEXT NOT NULL,
                        metadata TEXT NOT NULL DEFAULT '{}'
                    );
                    CREATE INDEX IF NOT EXISTS idx_agent_diag_session_ts
                        ON agent_diagnostics(session_id, timestamp);

                    CREATE TABLE IF NOT EXISTS compaction_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        event_type TEXT NOT NULL,
                        compression_ratio REAL,
                        pre_hash TEXT,
                        post_hash TEXT,
                        confidence REAL,
                        timestamp TEXT NOT NULL,
                        metadata TEXT NOT NULL DEFAULT '{}'
                    );
                    CREATE INDEX IF NOT EXISTS idx_compaction_session_ts
                        ON compaction_events(session_id, timestamp);

                    CREATE TABLE IF NOT EXISTS policy_evaluations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        tool_name TEXT NOT NULL,
                        allowed INTEGER NOT NULL,
                        reason TEXT,
                        profile TEXT,
                        timestamp TEXT NOT NULL,
                        metadata TEXT NOT NULL DEFAULT '{}'
                    );
                    CREATE INDEX IF NOT EXISTS idx_policy_eval_session_ts
                        ON policy_evaluations(session_id, timestamp);

                    CREATE TABLE IF NOT EXISTS timeline_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        event_type TEXT NOT NULL,
                        wall_clock TEXT NOT NULL,
                        monotonic_elapsed_ms INTEGER,
                        payload TEXT NOT NULL DEFAULT '{}'
                    );
                    CREATE INDEX IF NOT EXISTS idx_timeline_session_id
                        ON timeline_events(session_id, id);
                    """
                )

    def ensure_session(
        self,
        session_id: str,
        objective: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = _utc_now()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO sessions (id, created_at, updated_at, objective, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        updated_at=excluded.updated_at,
                        objective=COALESCE(excluded.objective, sessions.objective),
                        metadata=excluded.metadata
                    """,
                    (session_id, now, now, objective, _json_dumps(metadata)),
                )

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        timestamp: Optional[str] = None,
        monotonic_elapsed_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        summary_message_id: Optional[int] = None,
    ) -> int:
        ts = timestamp or _utc_now()
        self.ensure_session(session_id)
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO messages (
                        session_id, role, content, timestamp,
                        monotonic_elapsed_ms, metadata, summary_message_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        role,
                        content,
                        ts,
                        monotonic_elapsed_ms,
                        _json_dumps(metadata),
                        summary_message_id,
                    ),
                )
                return int(cur.lastrowid)

    def load_messages(self, session_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT role, content, timestamp, monotonic_elapsed_ms, metadata
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY id ASC
                    """,
                    (session_id,),
                ).fetchall()

        output: List[Dict[str, Any]] = []
        for row in rows:
            item: Dict[str, Any] = {
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["timestamp"],
            }
            if row["monotonic_elapsed_ms"] is not None:
                item["monotonic_elapsed_ms"] = row["monotonic_elapsed_ms"]
            try:
                meta = json.loads(row["metadata"] or "{}")
            except json.JSONDecodeError:
                meta = {}
            item.update(meta)
            output.append(item)
        return output

    def log_timeline_event(
        self,
        session_id: str,
        event_type: str,
        wall_clock: Optional[str] = None,
        monotonic_elapsed_ms: Optional[int] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.ensure_session(session_id)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO timeline_events (session_id, event_type, wall_clock, monotonic_elapsed_ms, payload)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        event_type,
                        wall_clock or _utc_now(),
                        monotonic_elapsed_ms,
                        _json_dumps(payload),
                    ),
                )

    def get_timeline(self, session_id: str, limit: int = 200) -> List[TimelineEvent]:
        capped_limit = max(1, min(limit, 5000))
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT event_type, wall_clock, monotonic_elapsed_ms, payload
                    FROM timeline_events
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (session_id, capped_limit),
                ).fetchall()

        events: List[TimelineEvent] = []
        for row in reversed(rows):
            try:
                payload = json.loads(row["payload"] or "{}")
            except json.JSONDecodeError:
                payload = {}
            events.append(
                TimelineEvent(
                    event_type=row["event_type"],
                    wall_clock=row["wall_clock"],
                    monotonic_elapsed_ms=row["monotonic_elapsed_ms"],
                    payload=payload,
                )
            )
        return events

    def log_file_operation(
        self,
        *,
        session_id: Optional[str],
        path: str,
        operation: str,
        status: str,
        monotonic_elapsed_ms: Optional[int] = None,
        freshness_token: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
    ) -> int:
        ts = _utc_now()
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO file_operations (
                        session_id, path, operation, status, timestamp, monotonic_elapsed_ms,
                        freshness_token, details, trace_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        path,
                        operation,
                        status,
                        ts,
                        monotonic_elapsed_ms,
                        freshness_token,
                        _json_dumps(details),
                        trace_id,
                    ),
                )
                return int(cur.lastrowid)

    def log_file_version(
        self,
        *,
        session_id: Optional[str],
        path: str,
        operation_id: Optional[int],
        hash_before: Optional[str],
        hash_after: Optional[str],
        backup_path: Optional[str],
        git_head: Optional[str] = None,
        trace_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> int:
        ts = _utc_now()
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO file_versions (
                        session_id, path, operation_id, hash_before, hash_after,
                        backup_path, git_head, timestamp, trace_id, details
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        path,
                        operation_id,
                        hash_before,
                        hash_after,
                        backup_path,
                        git_head,
                        ts,
                        trace_id,
                        _json_dumps(details),
                    ),
                )
                return int(cur.lastrowid)

    def log_permission_event(
        self,
        *,
        session_id: Optional[str],
        tool_name: str,
        decision: str,
        reason: Optional[str],
        policy_profile: Optional[str],
        signature: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO permission_events (
                        session_id, tool_name, decision, reason, policy_profile,
                        timestamp, signature, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        tool_name,
                        decision,
                        reason,
                        policy_profile,
                        _utc_now(),
                        signature,
                        _json_dumps(metadata),
                    ),
                )

    def log_policy_evaluation(
        self,
        *,
        session_id: Optional[str],
        tool_name: str,
        allowed: bool,
        reason: Optional[str],
        profile: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO policy_evaluations (
                        session_id, tool_name, allowed, reason, profile, timestamp, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        tool_name,
                        1 if allowed else 0,
                        reason,
                        profile,
                        _utc_now(),
                        _json_dumps(metadata),
                    ),
                )

    def log_task_state(
        self,
        *,
        session_id: Optional[str],
        task_id: Optional[str],
        state: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO task_state (session_id, task_id, state, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        task_id,
                        state,
                        _utc_now(),
                        _json_dumps(metadata),
                    ),
                )

    def log_diagnostic(
        self,
        *,
        session_id: Optional[str],
        source: str,
        severity: str,
        message: str,
        file_path: Optional[str] = None,
        line: Optional[int] = None,
        col: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO agent_diagnostics (
                        session_id, source, severity, message, file_path, line, col, timestamp, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        source,
                        severity,
                        message,
                        file_path,
                        line,
                        col,
                        _utc_now(),
                        _json_dumps(metadata),
                    ),
                )

    def log_compaction_event(
        self,
        *,
        session_id: Optional[str],
        event_type: str,
        compression_ratio: Optional[float],
        pre_hash: Optional[str],
        post_hash: Optional[str],
        confidence: Optional[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO compaction_events (
                        session_id, event_type, compression_ratio, pre_hash, post_hash,
                        confidence, timestamp, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        event_type,
                        compression_ratio,
                        pre_hash,
                        post_hash,
                        confidence,
                        _utc_now(),
                        _json_dumps(metadata),
                    ),
                )

    def export_audit(self, session_id: str, output_path: str) -> str:
        tables = [
            "sessions",
            "messages",
            "file_operations",
            "file_versions",
            "permission_events",
            "task_state",
            "agent_diagnostics",
            "compaction_events",
            "policy_evaluations",
            "timeline_events",
        ]

        bundle: Dict[str, Any] = {
            "schema_version": "1.0",
            "generated_at": _utc_now(),
            "session_id": session_id,
            "tables": {},
        }

        with self._lock:
            with self._connect() as conn:
                for table in tables:
                    if table == "sessions":
                        rows = conn.execute(
                            f"SELECT * FROM {table} WHERE id = ?", (session_id,)
                        ).fetchall()
                    elif "session_id" in {
                        col[1]
                        for col in conn.execute(f"PRAGMA table_info({table})").fetchall()
                    }:
                        rows = conn.execute(
                            f"SELECT * FROM {table} WHERE session_id = ? ORDER BY id ASC",
                            (session_id,),
                        ).fetchall()
                    else:
                        rows = conn.execute(
                            f"SELECT * FROM {table} ORDER BY id ASC"
                        ).fetchall()

                    bundle["tables"][table] = [dict(row) for row in rows]

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        return str(output)


_GLOBAL_DB: Optional[AnvilDB] = None


def get_anvil_db(db_path: str = ".anvil/anvil.db") -> AnvilDB:
    global _GLOBAL_DB
    if _GLOBAL_DB is None or str(_GLOBAL_DB.db_path) != db_path:
        _GLOBAL_DB = AnvilDB(db_path=db_path)
    return _GLOBAL_DB
