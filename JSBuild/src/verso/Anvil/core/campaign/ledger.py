"""Persistent cross-phase campaign memory."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional


class TheLedger:
    """SQLite-backed store for campaign metrics, artifacts, and gate outcomes."""

    def __init__(
        self,
        campaign_name: str,
        campaign_id: str,
        db_path: str = ".anvil/campaigns/campaign_ledger.db",
    ):
        self.campaign_name = campaign_name
        self.campaign_id = campaign_id
        self.db_path = db_path
        self._lock = threading.RLock()

        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    campaign_id TEXT NOT NULL,
                    metric_key TEXT NOT NULL,
                    metric_value TEXT NOT NULL,
                    recorded_at REAL NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    campaign_id TEXT NOT NULL,
                    artifact_name TEXT NOT NULL,
                    artifact_content TEXT NOT NULL,
                    recorded_at REAL NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS phase_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    campaign_id TEXT NOT NULL,
                    phase_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    recorded_at REAL NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS gate_verdicts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    campaign_id TEXT NOT NULL,
                    phase_id TEXT NOT NULL,
                    passed INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    recorded_at REAL NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS resources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    campaign_id TEXT NOT NULL,
                    resource_name TEXT NOT NULL,
                    resource_path TEXT NOT NULL,
                    role TEXT NOT NULL,
                    read_only INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    recorded_at REAL NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    campaign_id TEXT NOT NULL,
                    evidence_name TEXT NOT NULL,
                    evidence_type TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    source_phase TEXT,
                    recorded_at REAL NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_metrics_campaign_key
                ON metrics (campaign_id, metric_key, recorded_at DESC)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_phase_results_campaign_phase
                ON phase_results (campaign_id, phase_id, recorded_at DESC)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_resources_campaign_role
                ON resources (campaign_id, role, recorded_at DESC)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_evidence_campaign_type
                ON evidence (campaign_id, evidence_type, recorded_at DESC)
                """
            )
            self._conn.commit()

    def record_metric(self, key: str, value: Any) -> None:
        payload = json.dumps(value, default=str)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO metrics (campaign_id, metric_key, metric_value, recorded_at)
                VALUES (?, ?, ?, ?)
                """,
                (self.campaign_id, key, payload, time.time()),
            )
            self._conn.commit()

    def record_artifact(self, name: str, content: Any) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO artifacts (campaign_id, artifact_name, artifact_content, recorded_at)
                VALUES (?, ?, ?, ?)
                """,
                (self.campaign_id, name, str(content), time.time()),
            )
            self._conn.commit()

    def record_phase_result(self, phase_id: str, status: str, data: Dict[str, Any]) -> None:
        payload = json.dumps(data, default=str)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO phase_results (campaign_id, phase_id, status, data_json, recorded_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (self.campaign_id, phase_id, status, payload, time.time()),
            )
            self._conn.commit()

    def record_gate_verdict(self, phase_id: str, passed: bool, reason: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO gate_verdicts (campaign_id, phase_id, passed, reason, recorded_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (self.campaign_id, phase_id, int(bool(passed)), reason, time.time()),
            )
            self._conn.commit()

    def record_resource(
        self,
        name: str,
        path: str,
        role: str,
        read_only: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = json.dumps(metadata or {}, default=str)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO resources (
                    campaign_id, resource_name, resource_path, role, read_only, metadata_json, recorded_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.campaign_id,
                    name,
                    path,
                    role,
                    int(bool(read_only)),
                    payload,
                    time.time(),
                ),
            )
            self._conn.commit()

    def record_evidence(
        self,
        name: str,
        summary: str,
        evidence_type: str = "finding",
        confidence: str = "medium",
        payload: Optional[Dict[str, Any]] = None,
        source_phase: Optional[str] = None,
    ) -> None:
        serialized = json.dumps(payload or {}, default=str)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO evidence (
                    campaign_id, evidence_name, evidence_type, summary, confidence, payload_json, source_phase, recorded_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.campaign_id,
                    name,
                    evidence_type,
                    summary,
                    confidence,
                    serialized,
                    source_phase,
                    time.time(),
                ),
            )
            self._conn.commit()

    def get_metric(self, key: str) -> Any:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT metric_value
                FROM metrics
                WHERE campaign_id = ? AND metric_key = ?
                ORDER BY recorded_at DESC, id DESC
                LIMIT 1
                """,
                (self.campaign_id, key),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["metric_value"])

    def get_all_metrics(self) -> Dict[str, Any]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT metric_key, metric_value
                FROM metrics
                WHERE campaign_id = ?
                ORDER BY recorded_at ASC, id ASC
                """,
                (self.campaign_id,),
            ).fetchall()

        output: Dict[str, Any] = {}
        for row in rows:
            output[row["metric_key"]] = json.loads(row["metric_value"])
        return output

    def get_all_artifacts(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT artifact_name, artifact_content, recorded_at
                FROM artifacts
                WHERE campaign_id = ?
                ORDER BY recorded_at ASC, id ASC
                """,
                (self.campaign_id,),
            ).fetchall()

        return [
            {
                "name": row["artifact_name"],
                "content": row["artifact_content"],
                "recorded_at": row["recorded_at"],
            }
            for row in rows
        ]

    def get_phase_results(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT phase_id, status, data_json, recorded_at
                FROM phase_results
                WHERE campaign_id = ?
                ORDER BY recorded_at ASC, id ASC
                """,
                (self.campaign_id,),
            ).fetchall()

        return [
            {
                "phase_id": row["phase_id"],
                "status": row["status"],
                "data": json.loads(row["data_json"]),
                "recorded_at": row["recorded_at"],
            }
            for row in rows
        ]

    def get_all_gate_verdicts(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT phase_id, passed, reason, recorded_at
                FROM gate_verdicts
                WHERE campaign_id = ?
                ORDER BY recorded_at ASC, id ASC
                """,
                (self.campaign_id,),
            ).fetchall()

        return [
            {
                "phase_id": row["phase_id"],
                "passed": bool(row["passed"]),
                "reason": row["reason"],
                "recorded_at": row["recorded_at"],
            }
            for row in rows
        ]

    def get_resources(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT resource_name, resource_path, role, read_only, metadata_json, recorded_at
                FROM resources
                WHERE campaign_id = ?
                ORDER BY recorded_at ASC, id ASC
                """,
                (self.campaign_id,),
            ).fetchall()
        return [
            {
                "name": row["resource_name"],
                "path": row["resource_path"],
                "role": row["role"],
                "read_only": bool(row["read_only"]),
                "metadata": json.loads(row["metadata_json"]),
                "recorded_at": row["recorded_at"],
            }
            for row in rows
        ]

    def get_evidence(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT evidence_name, evidence_type, summary, confidence, payload_json, source_phase, recorded_at
                FROM evidence
                WHERE campaign_id = ?
                ORDER BY recorded_at ASC, id ASC
                """,
                (self.campaign_id,),
            ).fetchall()
        return [
            {
                "name": row["evidence_name"],
                "type": row["evidence_type"],
                "summary": row["summary"],
                "confidence": row["confidence"],
                "payload": json.loads(row["payload_json"]),
                "source_phase": row["source_phase"],
                "recorded_at": row["recorded_at"],
            }
            for row in rows
        ]

    def get_context_summary(self, budget_tokens: int = 10000) -> str:
        payload = {
            "campaign": {
                "name": self.campaign_name,
                "id": self.campaign_id,
            },
            "metrics": self.get_all_metrics(),
            "gate_verdicts": self.get_all_gate_verdicts(),
            "phase_results": self.get_phase_results(),
            "artifacts": self.get_all_artifacts()[-50:],
            "resources": self.get_resources()[-25:],
            "evidence": self.get_evidence()[-25:],
        }
        text = json.dumps(payload, indent=2, default=str)
        return self._truncate_to_token_budget(text, budget_tokens)

    @staticmethod
    def _truncate_to_token_budget(text: str, budget_tokens: int) -> str:
        if budget_tokens <= 0:
            return ""
        max_words = max(1, int(budget_tokens * 0.75))
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]) + "\n... [ledger summary truncated]"

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
