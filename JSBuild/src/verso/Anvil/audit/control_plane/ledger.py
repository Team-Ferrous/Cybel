from __future__ import annotations

from typing import Any

import jsonschema

from audit.control_plane.mission import MissionContext
from audit.control_plane.reducers import file_digest, now_iso, question_for_artifact
from audit.store.schema_validation import load_schema
from audit.store.suite_layout import required_suite_artifacts
from audit.store.writer import read_ndjson

RUN_LEDGER_SCHEMA_VERSION = "native_qsg_suite.run_ledger.v1"


def reduce_runtime_state(event_rows: list[dict[str, Any]]) -> dict[str, Any]:
    lifecycle: list[dict[str, Any]] = []
    completed_lanes: list[str] = []
    completed_attempt_ids: list[str] = []
    node_receipts: list[dict[str, Any]] = []
    terminal_state = ""
    run_exit_reason = ""
    last_successful_lane = ""

    seen_lifecycle: set[tuple[str, str]] = set()
    for row in event_rows:
        if not isinstance(row, dict):
            continue
        event_type = str(row.get("event_type") or "").strip()
        payload = dict(row.get("payload") or {})
        phase = str(row.get("phase") or "").strip()
        lane = str(row.get("lane") or "").strip()
        attempt_id = str(row.get("attempt_id") or "").strip()
        timestamp = str(row.get("timestamp") or "")

        if event_type == "suite_state":
            state = str(payload.get("state") or phase or "").strip()
            artifact = str(payload.get("artifact") or "suite_status.json")
            key = (state, artifact)
            if state and key not in seen_lifecycle:
                lifecycle.append(
                    {
                        "state": state,
                        "artifact": artifact,
                        "timestamp": timestamp,
                        "ok": bool(payload.get("ok", True)),
                    }
                )
                seen_lifecycle.add(key)
            terminal_state = str(payload.get("terminal_state") or terminal_state or "")
            run_exit_reason = str(payload.get("run_exit_reason") or run_exit_reason or "")
        elif event_type == "suite_checkpoint":
            completed_lanes = [
                str(item)
                for item in list(payload.get("completed_lanes") or completed_lanes)
                if str(item).strip()
            ]
            completed_attempt_ids = [
                str(item)
                for item in list(
                    payload.get("completed_attempt_ids") or completed_attempt_ids
                )
                if str(item).strip()
            ]
            run_exit_reason = str(payload.get("run_exit_reason") or run_exit_reason or "")
            last_successful_lane = str(
                payload.get("last_successful_lane") or last_successful_lane or ""
            )
        elif event_type in {"attempt_complete", "attempt_finish"} and attempt_id:
            if attempt_id not in completed_attempt_ids:
                completed_attempt_ids.append(attempt_id)
        elif event_type == "lane_complete" and lane and lane not in completed_lanes:
            completed_lanes.append(lane)
            last_successful_lane = lane
        elif event_type == "mission_node_receipt":
            receipt = {
                "node_id": str(payload.get("node_id") or ""),
                "phase": str(payload.get("phase") or phase or ""),
                "kind": str(payload.get("kind") or ""),
                "status": str(payload.get("status") or ""),
                "blocking": bool(payload.get("blocking", False)),
                "attempt_id": attempt_id,
                "lane": lane,
                "model": str(row.get("model") or ""),
                "timestamp": timestamp,
                "details": dict(payload.get("details") or {}),
            }
            if receipt["node_id"]:
                node_receipts.append(receipt)
                if (
                    receipt["kind"] == "lane"
                    and receipt["status"] == "completed"
                    and lane
                    and lane not in completed_lanes
                ):
                    completed_lanes.append(lane)
                    last_successful_lane = lane

    return {
        "completed_lanes": completed_lanes,
        "completed_attempt_ids": completed_attempt_ids,
        "last_successful_lane": last_successful_lane or None,
        "run_exit_reason": run_exit_reason,
        "terminal_state": terminal_state,
        "lifecycle": lifecycle,
        "node_receipts": node_receipts,
    }


def build_run_ledger(mission: MissionContext) -> dict[str, Any]:
    event_rows = read_ndjson(mission.layout.events_ndjson)
    event_schema = load_schema("benchmark_event.schema.json")
    normalized_events: list[dict[str, Any]] = []
    for row in event_rows:
        if not isinstance(row, dict):
            continue
        jsonschema.validate(instance=row, schema=event_schema)
        normalized_events.append(row)
    reducer_state = reduce_runtime_state(normalized_events)
    lifecycle = list(reducer_state.get("lifecycle") or [])
    if not lifecycle:
        lifecycle = [
            {
                "state": "initialized",
                "artifact": mission.layout.manifest_json.relative_to(
                    mission.layout.root
                ).as_posix(),
            },
            {
                "state": "preflight",
                "artifact": mission.layout.preflight_json.relative_to(
                    mission.layout.root
                ).as_posix(),
            },
            {
                "state": "finalized",
                "artifact": mission.layout.summary_json.relative_to(
                    mission.layout.root
                ).as_posix(),
            },
        ]
    artifact_states = [
        {
            **file_digest(path, root=mission.layout.root),
            "question": question_for_artifact(
                path.relative_to(mission.layout.root).as_posix()
            ),
            "required": True,
        }
        for path in required_suite_artifacts(mission.layout)
    ]
    return {
        "schema_version": RUN_LEDGER_SCHEMA_VERSION,
        "run_id": mission.run_id,
        "generated_at": now_iso(),
        "lifecycle": lifecycle,
        "events": normalized_events,
        "artifact_states": artifact_states,
        "reducer_state": reducer_state,
        "node_receipts": list(reducer_state.get("node_receipts") or []),
    }
