import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class EventStore:
    """
    Append-only SQLite event log for system-wide auditing and recovery.
    """

    def __init__(self, db_path: str = ".anvil/events.db"):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    run_id TEXT,
                    event_type TEXT NOT NULL,
                    source TEXT,
                    payload TEXT,
                    metadata TEXT
                )
            """)
            existing_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(events)").fetchall()
            }
            if "run_id" not in existing_columns:
                conn.execute("ALTER TABLE events ADD COLUMN run_id TEXT")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS event_links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    run_id TEXT,
                    link_type TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    target_ref TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY(event_id) REFERENCES events(id)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_type ON events(event_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_links_run_id ON event_links(run_id)"
            )
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mission_checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    status TEXT NOT NULL,
                    checkpoint_type TEXT NOT NULL,
                    metadata TEXT,
                    artifacts TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mission_checkpoints_run_id ON mission_checkpoints(run_id)"
            )

    def emit(
        self,
        event_type: str,
        payload: Dict[str, Any],
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        run_id: Optional[str] = None,
        links: Optional[list[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Log an event to the store."""
        timestamp = datetime.now(timezone.utc).isoformat()
        payload_json = json.dumps(payload, sort_keys=True)
        metadata_json = json.dumps(metadata or {}, sort_keys=True)
        normalized_run_id = run_id or self._coerce_run_id(metadata)
        normalized_links = self._normalize_links(
            payload=payload,
            metadata=metadata or {},
            links=links or [],
        )

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO events (timestamp, run_id, event_type, source, payload, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (timestamp, normalized_run_id, event_type, source, payload_json, metadata_json),
            )
            event_id = int(cursor.lastrowid)
            if normalized_links:
                conn.executemany(
                    """
                    INSERT INTO event_links (
                        event_id, run_id, link_type, target_type, target_ref, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            event_id,
                            normalized_run_id,
                            link["link_type"],
                            link["target_type"],
                            link["target_ref"],
                            json.dumps(link.get("metadata") or {}, sort_keys=True),
                        )
                        for link in normalized_links
                    ],
                )
        return {
            "event_id": event_id,
            "timestamp": timestamp,
            "run_id": normalized_run_id,
            "link_count": len(normalized_links),
        }

    def query(
        self,
        event_type: Optional[str] = None,
        limit: int = 100,
        *,
        run_id: Optional[str] = None,
    ):
        """Query events from the store."""
        query = "SELECT * FROM events"
        params = []
        filters = []
        if event_type:
            filters.append("event_type = ?")
            params.append(event_type)
        if run_id:
            filters.append("run_id = ?")
            params.append(run_id)
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()

    def events(self, *, run_id: Optional[str] = None, limit: int = 500) -> list[Dict[str, Any]]:
        """Return structured events for one run or the latest timeline."""
        query = (
            "SELECT id, timestamp, run_id, event_type, source, payload, metadata "
            "FROM events"
        )
        params: list[Any] = []
        if run_id:
            query += " WHERE run_id = ?"
            params.append(run_id)
        query += " ORDER BY id ASC"
        if limit > 0:
            query += " LIMIT ?"
            params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_event(row) for row in rows]

    def export_run(
        self,
        run_id: str,
        *,
        output_path: Optional[str] = None,
        limit: int = 2000,
    ) -> Dict[str, Any]:
        """Export one run with typed edges and replay metadata."""
        events = self.events(run_id=run_id, limit=limit)
        event_ids = [int(item["id"]) for item in events]
        links = self._links_for_events(event_ids)
        checkpoints = self.checkpoints(run_id)
        digest = hashlib.sha256()
        digest.update(str(run_id).encode("utf-8"))
        for event in events:
            digest.update(json.dumps(event, sort_keys=True).encode("utf-8"))
        for link in links:
            digest.update(json.dumps(link, sort_keys=True).encode("utf-8"))
        for checkpoint in checkpoints:
            digest.update(json.dumps(checkpoint, sort_keys=True).encode("utf-8"))
        replay_metadata = {
            "format_version": 1,
            "deterministic_hash": digest.hexdigest(),
            "inspectable_without_model": True,
            "event_count": len(events),
            "link_count": len(links),
            "checkpoint_count": len(checkpoints),
            "ordered_event_ids": [event["id"] for event in events],
        }
        mission_capsule = self._mission_capsule(
            run_id=run_id,
            events=events,
            checkpoints=checkpoints,
            links=links,
            replay_hash=replay_metadata["deterministic_hash"],
        )
        safety_case = self._safety_case(
            run_id=run_id,
            events=events,
            checkpoints=checkpoints,
            links=links,
            mission_capsule=mission_capsule,
        )
        payload = {
            "status": "ok",
            "run_id": run_id,
            "events": events,
            "links": links,
            "checkpoints": checkpoints,
            "replay": replay_metadata,
            "mission_capsule": mission_capsule,
            "safety_case": safety_case,
            "closure_summary": {
                "replay_hash": replay_metadata["deterministic_hash"],
                "capsule_id": mission_capsule["capsule_id"],
                "safety_case_node_count": len(safety_case["nodes"]),
                "unresolved_risk_count": int(safety_case["unresolved_risk_count"]),
            },
        }
        if output_path:
            path = Path(output_path)
        else:
            path = self.db_path.parent / "flight_recorder" / f"{run_id}_events.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        payload["path"] = str(path)
        return payload

    def record_qsg_replay_event(
        self,
        *,
        request_id: str,
        stage: str,
        payload: Dict[str, Any],
        event_type: str = "qsg.replay",
        source: str = "native_qsg",
        metadata: Optional[Dict[str, Any]] = None,
        links: Optional[list[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        replay_run_id = f"qsg-replay:{request_id}"
        normalized_metadata = dict(metadata or {})
        normalized_metadata.setdefault("request_id", request_id)
        normalized_metadata.setdefault("stage", stage)
        normalized_links = list(links or [])
        normalized_links.append(
            {
                "link_type": "request",
                "target_type": "qsg_request",
                "target_ref": request_id,
                "metadata": {"stage": stage},
            }
        )
        return self.emit(
            event_type=event_type,
            payload={"request_id": request_id, "stage": stage, **dict(payload or {})},
            source=source,
            metadata=normalized_metadata,
            run_id=replay_run_id,
            links=normalized_links,
        )

    def export_replay_tape(
        self,
        request_id: str,
        *,
        output_path: Optional[str] = None,
        limit: int = 2000,
    ) -> Dict[str, Any]:
        return self.export_run(
            f"qsg-replay:{request_id}",
            output_path=output_path,
            limit=limit,
        )

    def record_synthesis_replay_event(
        self,
        *,
        synthesis_id: str,
        stage: str,
        payload: Dict[str, Any],
        event_type: str = "synthesis.replay",
        source: str = "deterministic_synthesis",
        metadata: Optional[Dict[str, Any]] = None,
        links: Optional[list[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        replay_run_id = f"synthesis:{synthesis_id}"
        normalized_metadata = dict(metadata or {})
        normalized_metadata.setdefault("synthesis_id", synthesis_id)
        normalized_metadata.setdefault("stage", stage)
        normalized_links = list(links or [])
        normalized_links.append(
            {
                "link_type": "request",
                "target_type": "synthesis_spec",
                "target_ref": synthesis_id,
                "metadata": {"stage": stage},
            }
        )
        return self.emit(
            event_type=event_type,
            payload={"synthesis_id": synthesis_id, "stage": stage, **dict(payload or {})},
            source=source,
            metadata=normalized_metadata,
            run_id=replay_run_id,
            links=normalized_links,
        )

    def export_synthesis_replay_tape(
        self,
        synthesis_id: str,
        *,
        output_path: Optional[str] = None,
        limit: int = 2000,
    ) -> Dict[str, Any]:
        return self.export_run(
            f"synthesis:{synthesis_id}",
            output_path=output_path,
            limit=limit,
        )

    def record_checkpoint(
        self,
        run_id: str,
        phase: str,
        status: str,
        *,
        checkpoint_type: str = "phase",
        metadata: Optional[Dict[str, Any]] = None,
        artifacts: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        metadata_json = json.dumps(metadata or {}, sort_keys=True)
        artifacts_json = json.dumps(list(artifacts or []), sort_keys=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO mission_checkpoints (
                    timestamp, run_id, phase, status, checkpoint_type, metadata, artifacts
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    str(run_id),
                    str(phase),
                    str(status),
                    str(checkpoint_type),
                    metadata_json,
                    artifacts_json,
                ),
            )
            checkpoint_id = int(cursor.lastrowid)
        return {
            "checkpoint_id": checkpoint_id,
            "timestamp": timestamp,
            "run_id": str(run_id),
            "phase": str(phase),
            "status": str(status),
            "checkpoint_type": str(checkpoint_type),
        }

    def checkpoints(self, run_id: str) -> list[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, timestamp, run_id, phase, status, checkpoint_type, metadata, artifacts
                FROM mission_checkpoints
                WHERE run_id = ?
                ORDER BY id ASC
                """,
                (str(run_id),),
            ).fetchall()
        return [
            {
                "id": int(row[0]),
                "timestamp": str(row[1]),
                "run_id": str(row[2]),
                "phase": str(row[3]),
                "status": str(row[4]),
                "checkpoint_type": str(row[5]),
                "metadata": json.loads(row[6] or "{}"),
                "artifacts": json.loads(row[7] or "[]"),
            }
            for row in rows
        ]

    def latest_checkpoint(self, run_id: str) -> Optional[Dict[str, Any]]:
        checkpoints = self.checkpoints(run_id)
        if not checkpoints:
            return None
        return checkpoints[-1]

    def build_resume_payload(self, run_id: str) -> Dict[str, Any]:
        latest = self.latest_checkpoint(run_id)
        return {
            "status": "ok" if latest else "missing",
            "run_id": str(run_id),
            "checkpoint_count": len(self.checkpoints(run_id)),
            "latest_checkpoint": latest,
            "artifacts": list((latest or {}).get("artifacts") or []),
            "metadata": dict((latest or {}).get("metadata") or {}),
        }

    @staticmethod
    def _coerce_run_id(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        if not metadata:
            return None
        run_id = metadata.get("run_id") or metadata.get("trace_id")
        if run_id is None:
            return None
        return str(run_id)

    def _normalize_links(
        self,
        *,
        payload: Dict[str, Any],
        metadata: Dict[str, Any],
        links: list[Dict[str, Any]],
    ) -> list[Dict[str, Any]]:
        normalized: list[Dict[str, Any]] = []
        for link in links:
            target_ref = str(link.get("target_ref") or "").strip()
            if not target_ref:
                continue
            normalized.append(
                {
                    "link_type": str(link.get("link_type") or "related_to"),
                    "target_type": str(link.get("target_type") or "artifact"),
                    "target_ref": target_ref,
                    "metadata": dict(link.get("metadata") or {}),
                }
            )
        for file_path in list(payload.get("files") or metadata.get("files") or []):
            normalized.append(
                {
                    "link_type": "touches",
                    "target_type": "file",
                    "target_ref": str(file_path),
                    "metadata": {},
                }
            )
        for symbol in list(payload.get("symbols") or metadata.get("symbols") or []):
            normalized.append(
                {
                    "link_type": "touches",
                    "target_type": "symbol",
                    "target_ref": str(symbol),
                    "metadata": {},
                }
            )
        for test_ref in list(payload.get("tests") or metadata.get("tests") or []):
            normalized.append(
                {
                    "link_type": "verifies",
                    "target_type": "test",
                    "target_ref": str(test_ref),
                    "metadata": {},
                }
            )
        for tool_name in list(payload.get("tool_calls") or metadata.get("tool_calls") or []):
            normalized.append(
                {
                    "link_type": "invokes",
                    "target_type": "tool",
                    "target_ref": str(tool_name),
                    "metadata": {},
                }
            )
        for artifact in list(payload.get("artifacts") or metadata.get("artifacts") or []):
            normalized.append(
                {
                    "link_type": "emits",
                    "target_type": "artifact",
                    "target_ref": str(artifact),
                    "metadata": {},
                }
            )
        for proof_capsule in list(
            payload.get("proof_capsules") or metadata.get("proof_capsules") or []
        ):
            normalized.append(
                {
                    "link_type": "proves",
                    "target_type": "proof_capsule",
                    "target_ref": str(proof_capsule),
                    "metadata": {},
                }
            )
        for spec_ref in list(
            payload.get("synthesis_specs") or metadata.get("synthesis_specs") or []
        ):
            normalized.append(
                {
                    "link_type": "driven_by",
                    "target_type": "synthesis_spec",
                    "target_ref": str(spec_ref),
                    "metadata": {},
                }
            )
        deduped: dict[tuple[str, str, str], Dict[str, Any]] = {}
        for link in normalized:
            key = (link["link_type"], link["target_type"], link["target_ref"])
            deduped[key] = link
        return list(deduped.values())

    @staticmethod
    def _row_to_event(row: tuple[Any, ...]) -> Dict[str, Any]:
        payload = json.loads(row[5] or "{}")
        metadata = json.loads(row[6] or "{}")
        return {
            "id": int(row[0]),
            "timestamp": str(row[1]),
            "run_id": row[2],
            "event_type": str(row[3]),
            "source": row[4],
            "payload": payload,
            "metadata": metadata,
            "files": list(payload.get("files") or metadata.get("files") or []),
            "symbols": list(payload.get("symbols") or metadata.get("symbols") or []),
            "tests": list(payload.get("tests") or metadata.get("tests") or []),
            "tool_calls": list(
                payload.get("tool_calls") or metadata.get("tool_calls") or []
            ),
        }

    def _links_for_events(self, event_ids: list[int]) -> list[Dict[str, Any]]:
        if not event_ids:
            return []
        placeholders = ",".join("?" for _ in event_ids)
        query = (
            "SELECT event_id, run_id, link_type, target_type, target_ref, metadata "
            f"FROM event_links WHERE event_id IN ({placeholders}) ORDER BY id ASC"
        )
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, event_ids).fetchall()
        return [
            {
                "event_id": int(row[0]),
                "run_id": row[1],
                "link_type": str(row[2]),
                "target_type": str(row[3]),
                "target_ref": str(row[4]),
                "metadata": json.loads(row[5] or "{}"),
            }
            for row in rows
        ]

    @staticmethod
    def _mission_capsule(
        *,
        run_id: str,
        events: list[Dict[str, Any]],
        checkpoints: list[Dict[str, Any]],
        links: list[Dict[str, Any]],
        replay_hash: str,
    ) -> Dict[str, Any]:
        phases = sorted(
            {
                str(checkpoint.get("phase") or "")
                for checkpoint in checkpoints
                if str(checkpoint.get("phase") or "").strip()
            }
            | {
                str(event.get("payload", {}).get("phase") or "")
                for event in events
                if str(event.get("payload", {}).get("phase") or "").strip()
            }
        )
        promoted_lanes = sorted(
            {
                str(event.get("payload", {}).get("branch_lane_id") or "")
                for event in events
                if str(event.get("payload", {}).get("branch_lane_id") or "").strip()
            }
        )
        verification_summaries = [
            {
                "event_type": event["event_type"],
                "all_passed": bool(event.get("payload", {}).get("all_passed", False)),
                "counterexamples": list(event.get("payload", {}).get("counterexamples") or []),
            }
            for event in events
            if "verification" in str(event.get("event_type") or "")
        ]
        artifact_refs = sorted(
            {
                *(
                    str(artifact)
                    for checkpoint in checkpoints
                    for artifact in list(checkpoint.get("artifacts") or [])
                    if str(artifact).strip()
                ),
                *(
                    str(link.get("target_ref") or "")
                    for link in links
                    if str(link.get("target_type") or "") == "artifact"
                    and str(link.get("target_ref") or "").strip()
                ),
            }
        )
        proof_capsule_refs = sorted(
            str(link.get("target_ref") or "")
            for link in links
            if str(link.get("target_type") or "") == "proof_capsule"
            and str(link.get("target_ref") or "").strip()
        )
        return {
            "capsule_id": f"capsule_{replay_hash[:12]}",
            "run_id": str(run_id),
            "phase_count": len(phases),
            "phases": phases,
            "promoted_lanes": promoted_lanes,
            "verification_summaries": verification_summaries,
            "artifact_refs": artifact_refs,
            "proof_capsule_refs": proof_capsule_refs,
            "key_checkpoints": [
                {
                    "phase": checkpoint["phase"],
                    "status": checkpoint["status"],
                    "artifacts": list(checkpoint.get("artifacts") or []),
                }
                for checkpoint in checkpoints[-5:]
            ],
        }

    @staticmethod
    def _safety_case(
        *,
        run_id: str,
        events: list[Dict[str, Any]],
        checkpoints: list[Dict[str, Any]],
        links: list[Dict[str, Any]],
        mission_capsule: Dict[str, Any],
    ) -> Dict[str, Any]:
        nodes: list[Dict[str, Any]] = [
            {"node_id": f"campaign:{run_id}", "kind": "campaign", "label": str(run_id)},
            {
                "node_id": f"capsule:{mission_capsule['capsule_id']}",
                "kind": "mission_capsule",
                "label": mission_capsule["capsule_id"],
            },
        ]
        edges: list[Dict[str, Any]] = [
            {
                "source": f"campaign:{run_id}",
                "target": f"capsule:{mission_capsule['capsule_id']}",
                "relation": "summarized_by",
            }
        ]
        unresolved_risk_count = 0
        for checkpoint in checkpoints:
            node_id = f"checkpoint:{checkpoint['id']}"
            nodes.append(
                {
                    "node_id": node_id,
                    "kind": "checkpoint",
                    "label": f"{checkpoint['phase']}:{checkpoint['status']}",
                }
            )
            edges.append(
                {
                    "source": f"campaign:{run_id}",
                    "target": node_id,
                    "relation": "checkpointed_by",
                }
            )
        for event in events:
            node_id = f"event:{event['id']}"
            nodes.append(
                {
                    "node_id": node_id,
                    "kind": "event",
                    "label": str(event.get("event_type") or "event"),
                }
            )
            edges.append(
                {
                    "source": f"campaign:{run_id}",
                    "target": node_id,
                    "relation": "evidenced_by",
                }
            )
            counterexamples = list(event.get("payload", {}).get("counterexamples") or [])
            if counterexamples or event.get("payload", {}).get("promotion_blocked"):
                unresolved_risk_count += len(counterexamples) or 1
        for link in links:
            target = f"{link['target_type']}:{link['target_ref']}"
            nodes.append(
                {
                    "node_id": target,
                    "kind": str(link["target_type"]),
                    "label": str(link["target_ref"]),
                }
            )
            edges.append(
                {
                    "source": f"event:{link['event_id']}",
                    "target": target,
                    "relation": str(link["link_type"]),
                }
            )
        deduped_nodes = {node["node_id"]: node for node in nodes}
        return {
            "nodes": list(deduped_nodes.values()),
            "edges": edges,
            "unresolved_risk_count": unresolved_risk_count,
        }


# Global Event Store instance (Singleton pattern)
_event_store = None


def get_event_store() -> EventStore:
    global _event_store
    if _event_store is None:
        _event_store = EventStore()
    return _event_store
