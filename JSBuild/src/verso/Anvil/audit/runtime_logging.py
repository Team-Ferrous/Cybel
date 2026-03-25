from __future__ import annotations

import os
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import resource
except Exception:  # pragma: no cover - non-Unix fallback
    resource = None  # type: ignore[assignment]

from audit.evidence_capsule import build_evidence_capsule
from audit.evidence_capsule import extract_compiler_diagnostics
from audit.evidence_capsule import extract_failed_tests
from audit.evidence_capsule import next_sequence_id
from audit.evidence_capsule import write_evidence_capsule
from audit.store.writer import append_ndjson
from shared_kernel.event_store import EventStore

try:  # pragma: no cover - exercised in integration
    from rich.console import Group
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
except Exception:  # pragma: no cover - plain-text fallback
    Console = None  # type: ignore[assignment]
    Group = None  # type: ignore[assignment]
    Live = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Text = None  # type: ignore[assignment]


_LEVEL_ORDER = {
    "trace": 10,
    "debug": 20,
    "info": 30,
    "warn": 40,
    "warning": 40,
    "error": 50,
}
_ACTIVE_LOGGER: "SuiteEventLogger | None" = None


def env_ui_mode(default: str = "glassbox") -> str:
    return str(os.getenv("ANVIL_SUITE_UI_MODE", default) or default).strip().lower()


def env_log_level(default: str = "trace") -> str:
    raw = str(os.getenv("ANVIL_SUITE_LOG_LEVEL", default) or default).strip().lower()
    return raw if raw in _LEVEL_ORDER else default


def current_log_level() -> int:
    return int(_LEVEL_ORDER.get(env_log_level(), _LEVEL_ORDER["trace"]))


def set_active_logger(logger: "SuiteEventLogger | None") -> None:
    global _ACTIVE_LOGGER
    _ACTIVE_LOGGER = logger


def get_active_logger() -> "SuiteEventLogger | None":
    return _ACTIVE_LOGGER


def _append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip("\n"))
        handle.write("\n")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None


def _child_rusage_snapshot() -> tuple[float | None, float | None]:
    if resource is None:
        return None, None
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(usage.ru_utime), float(usage.ru_stime)


def _delta_ms(after: float | None, before: float | None) -> float | None:
    if after is None or before is None:
        return None
    return max(0.0, (float(after) - float(before)) * 1000.0)


def _poll_peak_rss_mb(pid: int, stop_event: threading.Event, state: dict[str, float | None]) -> None:
    status_path = Path(f"/proc/{pid}/status")
    peak_kb = 0
    while not stop_event.is_set():
        try:
            raw = status_path.read_text(encoding="utf-8", errors="ignore")
        except FileNotFoundError:
            break
        for line in raw.splitlines():
            if line.startswith("VmHWM:") or line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        peak_kb = max(peak_kb, int(parts[1]))
                    except Exception:
                        pass
        stop_event.wait(0.01)
    state["max_rss_mb"] = round(peak_kb / 1024.0, 3) if peak_kb else None


def _default_evidence_path(
    *,
    cwd: Path,
    logger: "SuiteEventLogger | None",
    stdout_path: Path | None,
    stderr_path: Path | None,
    tool_run_id: str,
) -> Path:
    for candidate in (stdout_path, stderr_path):
        if candidate is not None:
            return candidate.parent / "evidence_capsule.json"
    if logger is not None:
        return logger.run_root / "artifacts" / "evidence" / f"{tool_run_id}.json"
    return cwd / ".anvil" / "evidence" / f"{tool_run_id}.json"


def _tool_run_id(source: str, attempt_id: str | None, sequence_id: int) -> str:
    stem = str(attempt_id or source or "tool").strip().replace(" ", "_")
    return f"{stem}:{sequence_id}"
    try:
        return float(value)
    except Exception:
        return None


class SuiteEventLogger:
    def __init__(
        self,
        *,
        run_id: str,
        run_root: Path,
        events_path: Path,
        transcript_path: Path,
        console_log_path: Path | None = None,
        ui_mode: str | None = None,
        log_level: str | None = None,
    ) -> None:
        self.run_id = str(run_id)
        self.run_root = run_root
        self.events_path = events_path
        self.transcript_path = transcript_path
        self.console_log_path = console_log_path
        self.ui_mode = str(ui_mode or env_ui_mode()).strip().lower()
        self.log_level = str(log_level or env_log_level()).strip().lower()
        self.transcript_from_stderr = str(os.getenv("ANVIL_SUITE_TRANSCRIPT_FROM_STDERR", "0")).strip() == "1"
        self.event_store = EventStore(str(self.run_root / "telemetry" / "event_store.db"))
        self.event_store_export_path = self.run_root / "telemetry" / "event_store_export.json"
        self._lock = threading.RLock()
        self._console = Console(file=sys.stderr, force_terminal=False) if Console else None
        self._live: Live | None = None
        self._started = False
        self._state: dict[str, Any] = {
            "phase": "",
            "lane": "",
            "attempt_id": "",
            "model": "",
            "thread_tuple": "",
            "completed_attempts": 0,
            "planned_attempts": 0,
            "completed_lanes": 0,
            "total_lanes": 0,
            "latest_metrics": "",
            "latest_warning": "",
        }

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
            self.events_path.touch(exist_ok=True)
            self.transcript_path.touch(exist_ok=True)
            if self.console_log_path is not None:
                self.console_log_path.parent.mkdir(parents=True, exist_ok=True)
                self.console_log_path.touch(exist_ok=True)
            if self._should_use_live():
                self._live = Live(
                    self._render_dashboard(),
                    console=self._console,
                    refresh_per_second=4,
                    auto_refresh=False,
                    transient=False,
                )
                self._live.start()
            self._started = True

    def close(self) -> None:
        with self._lock:
            try:
                self.event_store.export_run(
                    self.run_id,
                    output_path=str(self.event_store_export_path),
                    limit=20_000,
                )
            except Exception:
                pass
            if self._live is not None:
                self._live.stop()
                self._live = None
            self._started = False

    def __enter__(self) -> "SuiteEventLogger":
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def set_state(self, **updates: Any) -> None:
        with self._lock:
            self._update_state(updates)
            self._refresh_live()

    def emit(
        self,
        *,
        level: str,
        source: str,
        event_type: str,
        message: str,
        phase: str | None = None,
        lane: str | None = None,
        attempt_id: str | None = None,
        model: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        normalized = str(level or "info").strip().lower()
        if normalized not in _LEVEL_ORDER:
            normalized = "info"
        event = {
            "schema_version": "native_qsg_suite.event.v1",
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": normalized,
            "source": str(source or ""),
            "event_type": str(event_type or ""),
            "phase": str(phase or ""),
            "lane": str(lane or ""),
            "attempt_id": str(attempt_id or ""),
            "model": str(model or ""),
            "message": str(message or ""),
        }
        if payload:
            event["payload"] = payload
        line = self._format_event(event)
        with self._lock:
            append_ndjson(self.events_path, event)
            self._emit_event_store_record(event)
            if not self.transcript_from_stderr:
                _append_line(self.transcript_path, line)
            if self.console_log_path is not None:
                _append_line(self.console_log_path, line)
            state_updates = {
                "phase": event["phase"] or None,
                "lane": event["lane"] or None,
                "attempt_id": event["attempt_id"] or None,
                "model": event["model"] or None,
            }
            self._update_state({key: value for key, value in state_updates.items() if value is not None})
            if payload:
                self._update_state(payload)
                self._maybe_update_metric_snapshot(payload)
            if normalized in {"warn", "warning", "error"}:
                self._state["latest_warning"] = line
            if self._should_print(normalized, event["event_type"]):
                if self._live is not None:
                    self._live.console.print(line)
                else:
                    print(line, file=sys.stderr, flush=True)
            self._refresh_live()

    def _emit_event_store_record(self, event: dict[str, Any]) -> None:
        payload = dict(event.get("payload") or {})
        artifact_path = str(payload.get("artifact_path") or "").strip()
        artifact_refs = [artifact_path] if artifact_path else []
        store_payload = {
            "schema_version": str(event.get("schema_version") or ""),
            "level": str(event.get("level") or ""),
            "phase": str(event.get("phase") or ""),
            "lane": str(event.get("lane") or ""),
            "attempt_id": str(event.get("attempt_id") or ""),
            "model": str(event.get("model") or ""),
            "message": str(event.get("message") or ""),
            **payload,
        }
        if artifact_refs:
            store_payload["artifacts"] = artifact_refs
        metadata = {
            "run_id": self.run_id,
            "run_root": str(self.run_root),
            "events_path": str(self.events_path),
            "transcript_path": str(self.transcript_path),
            "console_log_path": (
                str(self.console_log_path) if self.console_log_path is not None else ""
            ),
            "ui_mode": self.ui_mode,
            "log_level": self.log_level,
        }
        self.event_store.emit(
            event_type=str(event.get("event_type") or ""),
            payload=store_payload,
            source=str(event.get("source") or ""),
            metadata=metadata,
            run_id=self.run_id,
        )

    def emit_artifact(
        self,
        *,
        source: str,
        kind: str,
        path: Path,
        summary: str,
        phase: str | None = None,
        lane: str | None = None,
        attempt_id: str | None = None,
        model: str | None = None,
        payload: dict[str, Any] | None = None,
        level: str = "debug",
    ) -> None:
        merged_payload = dict(payload or {})
        merged_payload["artifact_kind"] = str(kind)
        merged_payload["artifact_path"] = str(path)
        self.emit(
            level=level,
            source=source,
            event_type="artifact_write",
            message=f"{summary} -> {path}",
            phase=phase,
            lane=lane,
            attempt_id=attempt_id,
            model=model,
            payload=merged_payload,
        )

    def _should_use_live(self) -> bool:
        if self.ui_mode not in {"glassbox", "dashboard"}:
            return False
        if self._console is None or Live is None:
            return False
        if not getattr(self._console, "is_terminal", False):
            return False
        return True

    def _should_print(self, level: str, event_type: str) -> bool:
        raw_enabled = self.ui_mode in {"glassbox", "raw", "plain"}
        if raw_enabled:
            return int(_LEVEL_ORDER[level]) >= int(_LEVEL_ORDER.get(self.log_level, _LEVEL_ORDER["trace"]))
        if self.ui_mode == "dashboard":
            return level in {"warn", "warning", "error"}
        return True

    def _format_event(self, event: dict[str, Any]) -> str:
        scope = []
        for key in ("phase", "lane", "attempt_id", "model"):
            value = str(event.get(key) or "").strip()
            if value:
                scope.append(f"{key}={value}")
        scope_text = f" [{' '.join(scope)}]" if scope else ""
        return (
            f"[{event['timestamp']}] {str(event['level']).upper():5s} "
            f"{event['source']}:{event['event_type']}{scope_text} {event['message']}"
        )

    def _refresh_live(self) -> None:
        if self._live is not None:
            self._live.update(self._render_dashboard(), refresh=True)

    def _render_dashboard(self) -> Any:
        lines = [
            self._dashboard_line("Phase", self._state.get("phase") or "-"),
            self._dashboard_line("Lane", self._state.get("lane") or "-"),
            self._dashboard_line("Attempt", self._state.get("attempt_id") or "-"),
            self._dashboard_line("Model", self._state.get("model") or "-"),
            self._dashboard_line("Threads", self._state.get("thread_tuple") or "-"),
            self._dashboard_line(
                "Attempts",
                f"{int(self._state.get('completed_attempts', 0) or 0)}/"
                f"{int(self._state.get('planned_attempts', 0) or 0)}",
            ),
            self._dashboard_line(
                "Lanes",
                f"{int(self._state.get('completed_lanes', 0) or 0)}/"
                f"{int(self._state.get('total_lanes', 0) or 0)}",
            ),
            self._dashboard_line("Metrics", self._state.get("latest_metrics") or "-"),
            self._dashboard_line("Latest", self._state.get("latest_warning") or "-"),
            self._dashboard_line("Run", str(self.run_root)),
            self._dashboard_line("Transcript", str(self.transcript_path)),
        ]
        if Panel is None or Group is None or Text is None:
            return "\n".join(lines)
        return Panel(
            Group(*[Text(line) for line in lines]),
            title="Native QSG Glass Box",
            border_style="cyan",
        )

    def _dashboard_line(self, label: str, value: str) -> str:
        return f"{label:10s} {value}"

    def _update_state(self, updates: dict[str, Any]) -> None:
        mapping = {
            "phase": "phase",
            "lane": "lane",
            "attempt_id": "attempt_id",
            "model": "model",
            "thread_tuple": "thread_tuple",
            "completed_attempts": "completed_attempts",
            "planned_attempts": "planned_attempts",
            "completed_lanes": "completed_lanes",
            "total_lanes": "total_lanes",
            "latest_metrics": "latest_metrics",
        }
        for key, state_key in mapping.items():
            if key in updates and updates[key] is not None and updates[key] != "":
                self._state[state_key] = updates[key]

    def _maybe_update_metric_snapshot(self, payload: dict[str, Any]) -> None:
        decode_tps = _safe_float(payload.get("decode_tps"))
        ttft_ms = _safe_float(payload.get("ttft_ms"))
        e2e_tps = _safe_float(payload.get("e2e_tps"))
        parts: list[str] = []
        if decode_tps is not None:
            parts.append(f"decode_tps={decode_tps:.2f}")
        if e2e_tps is not None:
            parts.append(f"e2e_tps={e2e_tps:.2f}")
        if ttft_ms is not None:
            parts.append(f"ttft_ms={ttft_ms:.2f}")
        if parts:
            self._state["latest_metrics"] = ", ".join(parts)


def run_logged_subprocess(
    *,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    source: str,
    phase: str | None = None,
    lane: str | None = None,
    attempt_id: str | None = None,
    model: str | None = None,
    timeout: int | float | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    logger = get_active_logger()
    sequence_id = next_sequence_id()
    tool_run_id = _tool_run_id(source, attempt_id, sequence_id)
    if logger is not None:
        logger.emit(
            level="debug",
            source=source,
            event_type="subprocess_start",
            message=f"starting command: {shlex.join(cmd)}",
            phase=phase,
            lane=lane,
            attempt_id=attempt_id,
            model=model,
            payload={"cwd": str(cwd), "cmd": list(cmd)},
        )
    usage_before = _child_rusage_snapshot()
    wall_started = time.perf_counter()
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
    if stderr_path is not None:
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_handle = stdout_path.open("a", encoding="utf-8") if stdout_path is not None else None
    stderr_handle = stderr_path.open("a", encoding="utf-8") if stderr_path is not None else None
    rss_stop = threading.Event()
    rss_state: dict[str, float | None] = {"max_rss_mb": None}
    rss_thread = threading.Thread(
        target=_poll_peak_rss_mb,
        args=(int(process.pid), rss_stop, rss_state),
        daemon=True,
    )
    rss_thread.start()

    def _drain(pipe: Any, chunks: list[str], stream_name: str, handle: Any) -> None:
        try:
            for raw in iter(pipe.readline, ""):
                chunks.append(raw)
                if handle is not None:
                    handle.write(raw)
                    handle.flush()
                line = raw.rstrip("\n")
                if line and logger is not None:
                    logger.emit(
                        level="trace",
                        source=source,
                        event_type="subprocess_line",
                        message=f"[{stream_name}] {line}",
                        phase=phase,
                        lane=lane,
                        attempt_id=attempt_id,
                        model=model,
                        payload={"stream": stream_name},
                    )
        finally:
            try:
                pipe.close()
            except Exception:
                pass
            if handle is not None:
                handle.close()

    stdout_thread = threading.Thread(
        target=_drain,
        args=(process.stdout, stdout_chunks, "stdout", stdout_handle),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_drain,
        args=(process.stderr, stderr_chunks, "stderr", stderr_handle),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    timed_out = False
    timeout_note = ""
    try:
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        return_code = 124
        timeout_note = f"process timed out after {timeout}s"
        try:
            process.kill()
        except Exception:
            pass
        try:
            process.wait(timeout=5.0)
        except Exception:
            pass
    finally:
        stdout_thread.join(timeout=5.0)
        stderr_thread.join(timeout=5.0)
        rss_stop.set()
        rss_thread.join(timeout=0.2)
    if timed_out:
        stderr_chunks.append(f"{timeout_note}\n")
    wall_time_ms = max(0.0, (time.perf_counter() - wall_started) * 1000.0)
    usage_after = _child_rusage_snapshot()
    completed = subprocess.CompletedProcess(
        args=cmd,
        returncode=return_code,
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
    )
    user_time_ms = _delta_ms(
        usage_after[0] if usage_after is not None else None,
        usage_before[0] if usage_before is not None else None,
    )
    sys_time_ms = _delta_ms(
        usage_after[1] if usage_after is not None else None,
        usage_before[1] if usage_before is not None else None,
    )
    artifact_paths: dict[str, str] = {}
    if stdout_path is not None:
        artifact_paths["stdout_log"] = str(stdout_path)
    if stderr_path is not None:
        artifact_paths["stderr_log"] = str(stderr_path)
    if logger is not None:
        artifact_paths["flight_recorder_timeline"] = str(logger.events_path)
        artifact_paths["terminal_transcript"] = str(logger.transcript_path)
        if logger.console_log_path is not None:
            artifact_paths["console_log"] = str(logger.console_log_path)
    evidence_path = _default_evidence_path(
        cwd=cwd,
        logger=logger,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        tool_run_id=tool_run_id,
    )
    artifact_paths["evidence_capsule"] = str(evidence_path)
    evidence_capsule = build_evidence_capsule(
        sequence_id=sequence_id,
        tool_run_id=tool_run_id,
        source=source,
        command=[str(part) for part in cmd],
        cwd=str(cwd),
        exit_code=int(return_code),
        wall_time_ms=wall_time_ms,
        user_time_ms=user_time_ms,
        sys_time_ms=sys_time_ms,
        max_rss_mb=rss_state.get("max_rss_mb"),
        stdout_path=str(stdout_path) if stdout_path is not None else None,
        stderr_path=str(stderr_path) if stderr_path is not None else None,
        artifact_paths=artifact_paths,
        failing_tests=extract_failed_tests(completed.stdout, completed.stderr),
        compiler_diagnostics=extract_compiler_diagnostics(completed.stderr),
        benchmark_metrics={},
        summary=f"{source} exited with return_code={return_code}",
        replay={
            "checkpoint_metadata_path": artifact_paths.get("checkpoint_metadata"),
            "flight_recorder_timeline_path": artifact_paths.get("flight_recorder_timeline"),
            "terminal_transcript_path": artifact_paths.get("terminal_transcript"),
            "inspectable_without_model": True,
        },
        stdout_text=completed.stdout,
        stderr_text=completed.stderr,
    )
    write_evidence_capsule(evidence_path, evidence_capsule)
    setattr(
        completed,
        "anvil_subprocess_metrics",
        {
            "sequence_id": sequence_id,
            "tool_run_id": tool_run_id,
            "wall_time_ms": wall_time_ms,
            "user_time_ms": user_time_ms,
            "sys_time_ms": sys_time_ms,
            "max_rss_mb": rss_state.get("max_rss_mb"),
        },
    )
    setattr(completed, "anvil_evidence_capsule", evidence_capsule)
    setattr(completed, "anvil_evidence_path", str(evidence_path))
    if logger is not None:
        logger.emit_artifact(
            source=source,
            kind="evidence_capsule",
            path=evidence_path,
            summary=f"wrote evidence capsule for {tool_run_id}",
            phase=phase,
            lane=lane,
            attempt_id=attempt_id,
            model=model,
            level="debug" if return_code == 0 else "warn",
        )
    if logger is not None:
        if timed_out:
            logger.emit(
                level="warn",
                source=source,
                event_type="subprocess_timeout",
                message=timeout_note,
                phase=phase,
                lane=lane,
                attempt_id=attempt_id,
                model=model,
                payload={"timeout_seconds": timeout},
            )
        logger.emit(
            level="debug" if return_code == 0 else "warn",
            source=source,
            event_type="subprocess_exit",
            message=f"command exited with return_code={return_code}",
            phase=phase,
            lane=lane,
            attempt_id=attempt_id,
            model=model,
            payload={"return_code": return_code, "cmd": list(cmd)},
        )
    return completed
