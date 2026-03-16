"""Run-level black box recorder for Anvil control-plane execution."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable

from shared_kernel.event_store import EventStore
from saguaro.reality.store import RealityGraphStore


class BlackBoxRecorder:
    """Persist replayable run traces, message segments, and runtime graph artifacts."""

    def __init__(
        self,
        repo_root: str,
        *,
        event_store: EventStore | None = None,
        reality_graph: RealityGraphStore | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.event_store = event_store or EventStore(str(self.repo_root / ".anvil" / "events.db"))
        self.reality_graph = reality_graph or RealityGraphStore(str(self.repo_root))
        self.base_dir = self.repo_root / ".anvil" / "flight_recorder"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.run_id: str | None = None
        self.task_id: str | None = None
        self.run_metadata: dict[str, Any] = {}
        self.timeline: list[dict[str, Any]] = []
        self.message_segments: list[dict[str, Any]] = []
        self.artifacts: dict[str, str] = {}

    def start_run(
        self,
        *,
        run_id: str,
        task_id: str | None,
        task: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Reset recorder state for a new run."""
        self.run_id = str(run_id)
        self.task_id = str(task_id or run_id)
        self.timeline = []
        self.message_segments = []
        self.artifacts = {}
        self.run_metadata = {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "task_preview": task[:280],
            "task_length": len(task),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **dict(metadata or {}),
        }
        self.record_event(
            "run_started",
            phase="initialize",
            status="started",
            metadata={"task_length": len(task), **dict(metadata or {})},
        )

    def bind_message_bus(self, message_bus: Any) -> None:
        """Attach a message bus trace sink if the bus supports it."""
        attach = getattr(message_bus, "attach_trace_sink", None)
        if callable(attach):
            attach(self.capture_message_segment)

    def record_event(
        self,
        event_type: str,
        *,
        phase: str | None = None,
        status: str | None = None,
        files: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        artifacts: dict[str, str] | None = None,
        source: str = "anvil.black_box",
    ) -> dict[str, Any]:
        """Record one run event into the event store and runtime graph."""
        if not self.run_id:
            return {}
        normalized_metadata = dict(metadata or {})
        normalized_files = [str(item) for item in list(files or []) if str(item)]
        artifact_values = [str(item) for item in dict(artifacts or {}).values() if str(item)]
        payload = {
            "event_type": event_type,
            "phase": phase,
            "status": status,
            "files": normalized_files,
            "artifacts": artifact_values,
        }
        links = [
            {"link_type": "touches", "target_type": "file", "target_ref": item}
            for item in normalized_files
        ]
        links.extend(
            {
                "link_type": "emits",
                "target_type": "artifact",
                "target_ref": item,
            }
            for item in artifact_values
        )
        receipt = self.event_store.emit(
            event_type=event_type,
            payload=payload,
            source=source,
            metadata={
                "run_id": self.run_id,
                "task_id": self.task_id,
                **normalized_metadata,
            },
            run_id=self.run_id,
            links=links,
        )
        reality_payload = self.reality_graph.record_event(
            event_type,
            run_id=self.run_id,
            task_id=self.task_id,
            phase=phase,
            status=status,
            files=normalized_files,
            metadata=normalized_metadata,
            artifacts=artifacts or {},
        )
        event_record = {
            "event_type": event_type,
            "phase": phase,
            "status": status,
            "files": normalized_files,
            "metadata": normalized_metadata,
            "artifacts": dict(artifacts or {}),
            "event_store": receipt,
            "reality": reality_payload,
        }
        self.timeline.append(event_record)
        if artifacts:
            self.artifacts.update({key: str(value) for key, value in artifacts.items()})
        return event_record

    def record_tool_plan(self, tool_calls: list[dict[str, Any]]) -> None:
        """Record planned tool execution wave."""
        for ordinal, tool_call in enumerate(tool_calls, start=1):
            args = dict(tool_call.get("args") or {})
            touched = [
                str(item)
                for item in (
                    args.get("file_path"),
                    args.get("path"),
                    args.get("workspace_id"),
                )
                if item
            ]
            self.record_event(
                "tool_planned",
                phase="execute",
                status="planned",
                files=touched,
                metadata={
                    "ordinal": ordinal,
                    "tool": tool_call.get("tool"),
                    "args": args,
                },
            )

    def record_tool_results(self, tool_results: list[dict[str, Any]]) -> None:
        """Record tool outcomes with success/failure detail."""
        for result in tool_results:
            args = dict(result.get("args") or {})
            touched = [
                str(item)
                for item in (
                    args.get("file_path"),
                    args.get("path"),
                )
                if item
            ]
            self.record_event(
                "tool_result",
                phase="execute",
                status="ok" if result.get("success") else "error",
                files=touched,
                metadata={
                    "tool": result.get("tool"),
                    "args": args,
                    "success": bool(result.get("success")),
                    "error": result.get("error"),
                    "result_preview": str(result.get("result", ""))[:240],
                },
            )

    def record_verification(
        self,
        *,
        modified_files: list[str],
        passed: bool,
        issues: list[str],
        runtime_symbols: list[dict[str, Any]] | None = None,
        counterexamples: list[dict[str, Any]] | None = None,
    ) -> None:
        """Record post-change verification results."""
        self.record_event(
            "verification_result",
            phase="observe",
            status="passed" if passed else "failed",
            files=list(modified_files),
            metadata={
                "issues": list(issues or []),
                "runtime_symbols": list(runtime_symbols or []),
                "counterexamples": list(counterexamples or []),
            },
        )

    def record_performance_snapshot(self, snapshot: Any) -> None:
        """Record a performance snapshot in normalized JSON form."""
        if snapshot is None:
            return
        if is_dataclass(snapshot):
            payload = asdict(snapshot)
        else:
            payload = dict(getattr(snapshot, "__dict__", {}) or {})
        self.record_event(
            "performance_snapshot",
            phase="observe",
            status="ok",
            metadata=payload,
        )

    def capture_message_segment(self, segment: dict[str, Any]) -> None:
        """Accept a normalized message-bus trace segment."""
        normalized = dict(segment or {})
        normalized.setdefault("run_id", self.run_id)
        normalized.setdefault("task_id", self.task_id)
        self.message_segments.append(normalized)
        self.record_event(
            "message_segment",
            phase=str(normalized.get("phase") or "coordination"),
            status="observed",
            metadata=normalized,
            source="anvil.message_bus",
        )

    def _collect_qsg_replay_context(self) -> tuple[list[str], list[dict[str, Any]]]:
        request_ids: set[str] = set()
        descriptors: list[dict[str, Any]] = []

        def _walk(value: Any) -> None:
            if isinstance(value, dict):
                request_id = str(value.get("request_id") or "").strip()
                if request_id:
                    request_ids.add(request_id)
                descriptor = value.get("mission_replay_descriptor")
                if isinstance(descriptor, dict):
                    descriptors.append(dict(descriptor))
                    descriptor_request_id = str(
                        descriptor.get("request_id") or ""
                    ).strip()
                    if descriptor_request_id:
                        request_ids.add(descriptor_request_id)
                for nested in value.values():
                    _walk(nested)
                return
            if isinstance(value, list):
                for item in value:
                    _walk(item)

        _walk(self.timeline)
        return sorted(request_ids), descriptors

    def finalize(
        self,
        *,
        stop_reason: str,
        success: bool,
        message_bus: Any | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write final run artifacts and return the manifest."""
        if not self.run_id:
            return {"status": "missing_run"}

        self.record_event(
            "run_stopped",
            phase="run",
            status=stop_reason,
            metadata={"success": success, **dict(extra_metadata or {})},
        )

        run_dir = self.base_dir / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        if message_bus is not None:
            export_message_log = getattr(message_bus, "export_message_log", None)
            if callable(export_message_log):
                message_log_path = run_dir / "message_log.json"
                export_message_log(str(message_log_path))
                self.artifacts["message_log"] = str(message_log_path)
            export_trace_segments = getattr(message_bus, "export_trace_segments", None)
            if callable(export_trace_segments):
                trace_path = run_dir / "message_trace.json"
                export_trace_segments(str(trace_path))
                self.artifacts["message_trace"] = str(trace_path)

        event_export = self.event_store.export_run(
            self.run_id,
            output_path=str(run_dir / "events.json"),
        )
        self.artifacts["events"] = str(event_export["path"])

        qsg_runtime_status = dict((extra_metadata or {}).get("qsg_runtime_status") or {})
        if qsg_runtime_status:
            qsg_runtime_path = run_dir / "qsg_runtime_status.json"
            qsg_runtime_path.write_text(
                json.dumps(qsg_runtime_status, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            self.artifacts["qsg_runtime_status"] = str(qsg_runtime_path)

        reality_export = self.reality_graph.export_run(
            run_id=self.run_id,
            output_path=str(run_dir / "reality.json"),
        )
        self.artifacts["reality"] = str(reality_export["path"])
        self.artifacts.update(reality_export.get("artifacts", {}))

        replay_request_ids, replay_descriptors = self._collect_qsg_replay_context()
        exported_replay_tapes: dict[str, str] = {}
        for request_id in replay_request_ids:
            try:
                replay_tape = self.event_store.export_replay_tape(
                    request_id,
                    output_path=str(run_dir / f"qsg_replay_{request_id}.json"),
                )
            except Exception:
                continue
            path = str(replay_tape.get("path") or "")
            if path:
                exported_replay_tapes[request_id] = path
                self.artifacts[f"qsg_replay_tape_{request_id}"] = path

        if replay_descriptors:
            descriptor_path = run_dir / "qsg_mission_replays.json"
            descriptor_path.write_text(
                json.dumps(
                    {"count": len(replay_descriptors), "items": replay_descriptors},
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            self.artifacts["qsg_mission_replays"] = str(descriptor_path)

        manifest = {
            "status": "ok",
            "run_id": self.run_id,
            "task_id": self.task_id,
            "metadata": {
                **self.run_metadata,
                "stopped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "stop_reason": stop_reason,
                "success": success,
                **dict(extra_metadata or {}),
            },
            "timeline_count": len(self.timeline),
            "message_segment_count": len(self.message_segments),
            "artifacts": dict(self.artifacts),
            "replay": dict(event_export.get("replay") or {}),
            "qsg_replay": {
                "request_ids": replay_request_ids,
                "replay_tapes": exported_replay_tapes,
                "descriptor_count": len(replay_descriptors),
            },
        }
        manifest_path = run_dir / "run_trace.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        manifest["path"] = str(manifest_path)
        return manifest
