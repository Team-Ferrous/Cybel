"""Runtime event logging and execution reality graph persistence."""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

from saguaro.omnigraph.model import OmniNode, OmniRelation


class RealityGraphStore:
    """Persist replayable runtime events and derive a reality graph."""

    def __init__(self, repo_path: str) -> None:
        self.repo_path = os.path.abspath(repo_path)
        self.base_dir = os.path.join(self.repo_path, ".saguaro", "reality")
        self.events_path = os.path.join(self.base_dir, "events.jsonl")
        self.graph_path = os.path.join(self.base_dir, "graph.json")
        self.twin_path = os.path.join(self.base_dir, "twin.json")

    def record_event(
        self,
        event_type: str,
        *,
        run_id: str,
        task_id: str | None = None,
        phase: str | None = None,
        status: str | None = None,
        files: list[str] | None = None,
        symbols: list[str] | None = None,
        tests: list[str] | None = None,
        tool_calls: list[str] | None = None,
        counterexamples: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        artifacts: dict[str, str] | None = None,
        source: str | None = None,
        segment_id: str | None = None,
    ) -> dict[str, Any]:
        """Append a runtime event to the event stream."""
        os.makedirs(self.base_dir, exist_ok=True)
        observed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        metadata_payload = dict(metadata or {})
        payload = {
            "event_id": self._event_id(
                event_type=event_type,
                run_id=run_id,
                task_id=task_id,
                phase=phase,
                status=status,
                files=files or [],
                metadata=metadata_payload,
            ),
            "event_type": event_type,
            "run_id": run_id,
            "task_id": task_id or run_id,
            "phase": phase,
            "status": status,
            "files": sorted({str(item) for item in (files or []) if str(item)}),
            "symbols": self._string_list(symbols or metadata_payload.pop("symbols", [])),
            "tests": self._string_list(tests or metadata_payload.pop("tests", [])),
            "tool_calls": self._string_list(
                tool_calls or metadata_payload.pop("tool_calls", [])
            ),
            "counterexamples": self._normalize_counterexamples(
                counterexamples or metadata_payload.pop("counterexamples", [])
            ),
            "metadata": metadata_payload,
            "artifacts": dict(artifacts or {}),
            "source": source or "runtime",
            "segment_id": segment_id,
            "observed_at": observed_at,
            "ts": time.time(),
        }
        with open(self.events_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return payload

    def build_graph(
        self,
        *,
        run_id: str | None = None,
        limit: int = 2000,
    ) -> dict[str, Any]:
        """Build a typed reality graph from persisted events."""
        events = self._load_events(run_id=run_id, limit=limit)
        nodes: dict[str, OmniNode] = {}
        relations: dict[str, OmniRelation] = {}

        for event in events:
            runtime_id = f"run::{event['run_id']}"
            nodes.setdefault(
                runtime_id,
                OmniNode(
                    id=runtime_id,
                    type="runtime_run",
                    label=event["run_id"],
                    metadata={},
                ),
            )

            event_node_id = f"event::{event['event_id']}"
            nodes[event_node_id] = OmniNode(
                id=event_node_id,
                type="runtime_event",
                label=event["event_type"],
                metadata={
                    "phase": event.get("phase"),
                    "status": event.get("status"),
                    "observed_at": event.get("observed_at"),
                    "metadata": event.get("metadata", {}),
                },
            )
            relations[f"{runtime_id}->{event_node_id}::emits"] = OmniRelation(
                id=f"{runtime_id}->{event_node_id}::emits",
                src_type="runtime_run",
                src_id=runtime_id,
                dst_type="runtime_event",
                dst_id=event_node_id,
                relation_type="emits",
                evidence_types=["runtime"],
                confidence=1.0,
                verified=True,
                drift_state="fresh",
                generation_id=run_id or event["run_id"],
            )

            phase = str(event.get("phase") or "").strip()
            if phase:
                phase_id = f"phase::{phase.lower()}"
                nodes.setdefault(
                    phase_id,
                    OmniNode(
                        id=phase_id,
                        type="runtime_phase",
                        label=phase,
                        metadata={},
                    ),
                )
                relations[f"{event_node_id}->{phase_id}::in_phase"] = OmniRelation(
                    id=f"{event_node_id}->{phase_id}::in_phase",
                    src_type="runtime_event",
                    src_id=event_node_id,
                    dst_type="runtime_phase",
                    dst_id=phase_id,
                    relation_type="in_phase",
                    evidence_types=["runtime"],
                    confidence=1.0,
                    verified=True,
                    drift_state="fresh",
                    generation_id=run_id or event["run_id"],
                )

            for file_path in list(event.get("files") or []):
                file_id = f"file::{file_path}"
                nodes.setdefault(
                    file_id,
                    OmniNode(
                        id=file_id,
                        type="artifact",
                        label=file_path,
                        file=file_path,
                        metadata={"kind": "file"},
                    ),
                )
                relations[f"{event_node_id}->{file_id}::touches"] = OmniRelation(
                    id=f"{event_node_id}->{file_id}::touches",
                    src_type="runtime_event",
                    src_id=event_node_id,
                    dst_type="artifact",
                    dst_id=file_id,
                    relation_type="touches",
                    evidence_types=["runtime"],
                    confidence=0.9,
                    verified=True,
                    drift_state="fresh",
                    generation_id=run_id or event["run_id"],
                )

            for name, path in sorted((event.get("artifacts") or {}).items()):
                artifact_id = f"artifact::{path}"
                nodes.setdefault(
                    artifact_id,
                    OmniNode(
                        id=artifact_id,
                        type="artifact_bundle",
                        label=name,
                        file=path,
                        metadata={"path": path},
                    ),
                )
                relations[f"{event_node_id}->{artifact_id}::emits_artifact"] = OmniRelation(
                    id=f"{event_node_id}->{artifact_id}::emits_artifact",
                    src_type="runtime_event",
                    src_id=event_node_id,
                    dst_type="artifact_bundle",
                    dst_id=artifact_id,
                    relation_type="emits_artifact",
                    evidence_types=["runtime", "artifact"],
                    confidence=0.95,
                    verified=True,
                    drift_state="fresh",
                    generation_id=run_id or event["run_id"],
                )

            for symbol in list(event.get("symbols") or []):
                symbol_id = f"symbol::{symbol}"
                nodes.setdefault(
                    symbol_id,
                    OmniNode(
                        id=symbol_id,
                        type="runtime_symbol",
                        label=symbol,
                        metadata={"symbol": symbol},
                    ),
                )
                relations[f"{event_node_id}->{symbol_id}::references_symbol"] = OmniRelation(
                    id=f"{event_node_id}->{symbol_id}::references_symbol",
                    src_type="runtime_event",
                    src_id=event_node_id,
                    dst_type="runtime_symbol",
                    dst_id=symbol_id,
                    relation_type="references_symbol",
                    evidence_types=["runtime"],
                    confidence=0.88,
                    verified=True,
                    drift_state="fresh",
                    generation_id=run_id or event["run_id"],
                )

            for test_path in list(event.get("tests") or []):
                test_id = f"test::{test_path}"
                nodes.setdefault(
                    test_id,
                    OmniNode(
                        id=test_id,
                        type="test_case",
                        label=test_path,
                        file=test_path,
                        metadata={"kind": "test"},
                    ),
                )
                relations[f"{event_node_id}->{test_id}::witnessed_by"] = OmniRelation(
                    id=f"{event_node_id}->{test_id}::witnessed_by",
                    src_type="runtime_event",
                    src_id=event_node_id,
                    dst_type="test_case",
                    dst_id=test_id,
                    relation_type="witnessed_by",
                    evidence_types=["runtime", "test"],
                    confidence=0.9,
                    verified=True,
                    drift_state="fresh",
                    generation_id=run_id or event["run_id"],
                )

            for tool_name in list(event.get("tool_calls") or []):
                tool_id = f"tool::{tool_name}"
                nodes.setdefault(
                    tool_id,
                    OmniNode(
                        id=tool_id,
                        type="tool_call",
                        label=tool_name,
                        metadata={"tool_name": tool_name},
                    ),
                )
                relations[f"{event_node_id}->{tool_id}::invokes_tool"] = OmniRelation(
                    id=f"{event_node_id}->{tool_id}::invokes_tool",
                    src_type="runtime_event",
                    src_id=event_node_id,
                    dst_type="tool_call",
                    dst_id=tool_id,
                    relation_type="invokes_tool",
                    evidence_types=["runtime"],
                    confidence=0.96,
                    verified=True,
                    drift_state="fresh",
                    generation_id=run_id or event["run_id"],
                )

            for counterexample in list(event.get("counterexamples") or []):
                counterexample_id = str(
                    counterexample.get("id")
                    or f"counterexample::{event['event_id']}::{len(nodes)}"
                )
                nodes.setdefault(
                    counterexample_id,
                    OmniNode(
                        id=counterexample_id,
                        type="counterexample",
                        label=str(
                            counterexample.get("counterexample_type") or "counterexample"
                        ),
                        metadata=dict(counterexample),
                    ),
                )
                relations[
                    f"{event_node_id}->{counterexample_id}::exposes_counterexample"
                ] = OmniRelation(
                    id=f"{event_node_id}->{counterexample_id}::exposes_counterexample",
                    src_type="runtime_event",
                    src_id=event_node_id,
                    dst_type="counterexample",
                    dst_id=counterexample_id,
                    relation_type="exposes_counterexample",
                    evidence_types=["runtime", "verification"],
                    confidence=0.9,
                    verified=True,
                    drift_state="fresh",
                    generation_id=run_id or event["run_id"],
                )

        for previous, current in zip(events, events[1:]):
            prev_id = f"event::{previous['event_id']}"
            curr_id = f"event::{current['event_id']}"
            relations[f"{prev_id}->{curr_id}::precedes"] = OmniRelation(
                id=f"{prev_id}->{curr_id}::precedes",
                src_type="runtime_event",
                src_id=prev_id,
                dst_type="runtime_event",
                dst_id=curr_id,
                relation_type="precedes",
                evidence_types=["runtime"],
                confidence=1.0,
                verified=True,
                drift_state="fresh",
                generation_id=run_id or current["run_id"],
            )

        payload = {
            "status": "ok",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id": run_id,
            "nodes": {key: value.to_dict() for key, value in nodes.items()},
            "relations": {key: value.to_dict() for key, value in relations.items()},
            "summary": {
                "event_count": len(events),
                "node_count": len(nodes),
                "relation_count": len(relations),
                "run_count": len({event["run_id"] for event in events}),
                "artifact_count": sum(
                    1 for item in nodes.values() if item.type in {"artifact", "artifact_bundle"}
                ),
                "symbol_count": sum(
                    1 for item in nodes.values() if item.type == "runtime_symbol"
                ),
                "test_count": sum(
                    1 for item in nodes.values() if item.type == "test_case"
                ),
                "tool_call_count": sum(
                    1 for item in nodes.values() if item.type == "tool_call"
                ),
                "counterexample_count": sum(
                    1 for item in nodes.values() if item.type == "counterexample"
                ),
            },
        }
        os.makedirs(self.base_dir, exist_ok=True)
        with open(self.graph_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        return payload

    def twin_state(self, *, run_id: str | None = None, limit: int = 500) -> dict[str, Any]:
        """Summarize latest runtime state for planning and audit."""
        events = self._load_events(run_id=run_id, limit=limit)
        latest = events[-1] if events else {}
        phases: dict[str, dict[str, Any]] = {}
        touched_files: set[str] = set()
        artifacts: dict[str, str] = {}
        symbols: set[str] = set()
        tests: set[str] = set()
        tool_calls: set[str] = set()
        counterexamples: list[dict[str, Any]] = []
        for event in events:
            phase = str(event.get("phase") or "").strip()
            if phase:
                phases[phase] = {
                    "status": event.get("status"),
                    "observed_at": event.get("observed_at"),
                }
            touched_files.update(str(item) for item in list(event.get("files") or []))
            symbols.update(str(item) for item in list(event.get("symbols") or []))
            tests.update(str(item) for item in list(event.get("tests") or []))
            tool_calls.update(str(item) for item in list(event.get("tool_calls") or []))
            counterexamples.extend(list(event.get("counterexamples") or []))
            artifacts.update({str(key): str(value) for key, value in dict(event.get("artifacts") or {}).items()})
        payload = {
            "status": "ok",
            "run_id": run_id or latest.get("run_id"),
            "latest_event": latest,
            "phase_state": phases,
            "files_touched": sorted(touched_files),
            "runtime_symbols": sorted(symbols),
            "tests": sorted(tests),
            "tool_calls": sorted(tool_calls),
            "counterexamples": counterexamples,
            "artifacts": artifacts,
            "risk_budget": dict((latest.get("metadata") or {}).get("risk_budget") or {}),
            "observability": {
                "observed": bool(events),
                "inferred": bool((latest.get("metadata") or {}).get("inferred") or False),
                "unobserved_count": int((latest.get("metadata") or {}).get("unobserved_count", 0) or 0),
            },
            "summary": {
                "event_count": len(events),
                "phase_count": len(phases),
                "file_count": len(touched_files),
                "artifact_count": len(artifacts),
                "symbol_count": len(symbols),
                "test_count": len(tests),
                "tool_call_count": len(tool_calls),
                "counterexample_count": len(counterexamples),
            },
        }
        os.makedirs(self.base_dir, exist_ok=True)
        with open(self.twin_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        return payload

    def events(self, *, run_id: str | None = None, limit: int = 2000) -> dict[str, Any]:
        """Return ordered runtime events for a run."""
        items = self._load_events(run_id=run_id, limit=limit)
        return {
            "status": "ok",
            "run_id": run_id,
            "count": len(items),
            "events": items,
        }

    def export_run(
        self,
        *,
        run_id: str,
        output_path: str | None = None,
        limit: int = 2000,
    ) -> dict[str, Any]:
        """Export one run with raw events, graph, and twin state."""
        events_payload = self.events(run_id=run_id, limit=limit)
        graph_payload = self.build_graph(run_id=run_id, limit=limit)
        twin_payload = self.twin_state(run_id=run_id, limit=limit)
        payload = {
            "status": "ok",
            "run_id": run_id,
            "events": events_payload["events"],
            "graph_summary": dict(graph_payload.get("summary") or {}),
            "twin_summary": dict(twin_payload.get("summary") or {}),
        }
        os.makedirs(self.base_dir, exist_ok=True)
        path = output_path or os.path.join(self.base_dir, f"{run_id}_export.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        payload["path"] = path
        payload["artifacts"] = {
            "reality_graph": self.graph_path,
            "reality_twin": self.twin_path,
        }
        return payload

    def _load_events(self, *, run_id: str | None = None, limit: int = 2000) -> list[dict[str, Any]]:
        if not os.path.exists(self.events_path):
            return []
        rows: list[dict[str, Any]] = []
        with open(self.events_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if run_id and str(item.get("run_id") or "") != str(run_id):
                    continue
                rows.append(item)
        rows.sort(key=lambda item: float(item.get("ts", 0.0) or 0.0))
        if limit > 0:
            rows = rows[-limit:]
        return rows

    @staticmethod
    def _event_id(
        *,
        event_type: str,
        run_id: str,
        task_id: str | None,
        phase: str | None,
        status: str | None,
        files: list[str],
        metadata: dict[str, Any],
    ) -> str:
        payload = json.dumps(
            {
                "event_type": event_type,
                "run_id": run_id,
                "task_id": task_id,
                "phase": phase,
                "status": status,
                "files": files,
                "metadata": metadata,
                "now": time.time_ns(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _string_list(values: Any) -> list[str]:
        if values is None:
            return []
        if isinstance(values, str):
            values = [values]
        return sorted({str(item) for item in values if str(item).strip()})

    @staticmethod
    def _normalize_counterexamples(values: Any) -> list[dict[str, Any]]:
        items = values or []
        if isinstance(items, dict):
            items = [items]
        normalized: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict):
                normalized.append(dict(item))
            elif str(item).strip():
                normalized.append(
                    {
                        "id": f"counterexample::{hashlib.sha1(str(item).encode('utf-8')).hexdigest()[:12]}",
                        "counterexample_type": "runtime_observation",
                        "observed_failure": str(item),
                    }
                )
        return normalized
