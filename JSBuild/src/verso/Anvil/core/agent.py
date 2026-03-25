import os
import json
import re
import time
import logging
import hashlib
import sqlite3
from datetime import datetime
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table  # Added import for Table
from rich.text import Text
from core.utils.smart_truncator import SmartTruncator
from core.utils.logger import get_logger, emit_structured_event

logger = get_logger(__name__)

from config.settings import (
    MASTER_MODEL,
    AGENTIC_THINKING,
    PERFORMANCE_CONFIG,
    AES_PROMPT_CONTRACT_REQUIRED,
    GENERATION_PARAMS,
)
from core.ollama_client import DeterministicOllama
from cli.history import ConversationHistory
from tools.registry import ToolRegistry
from core.serialization import (
    SerializableMixin,
    redact_secret_material,
    serialize_tool_provenance,
)

from core.approval import ApprovalManager, ApprovalMode
from core.anvil_db import get_anvil_db
from core.config_manager import ConfigManager
from core.project_context import ProjectContextManager
from core.checkpoint import CheckpointManager
from core.session_manager import SessionManager
from core.corrections import CorrectionManager
from core.proactive_context import ProactiveContextManager
from core.response_utils import ResponseStreamParser, clean_response
from core.memory.latent_memory import LatentMemory
from core.analysis.semantic import SemanticEngine
from core.pipelines import PipelineManager
from core.prompts import PromptManager
from core.context import ContextManager
from core.context_compression import (
    apply_context_updates,
    auto_compress_dead_end_reads,
    ensure_context_updates_arg,
    extract_context_updates,
    find_low_relevance_tc_ids,
    infer_next_tc_id,
    label_tool_result,
)
from core.task_memory import ContextCompressionMemory
from core.agent_tool_helpers import (
    extract_tool_calls as extract_tool_calls_helper,
    execute_tool as execute_tool_helper,
)
from core.agent_loop_helpers import (
    run_loop as run_loop_helper,
    simple_chat as simple_chat_helper,
)
from saguaro.reality.store import RealityGraphStore
from shared_kernel.event_store import get_event_store

# Legacy core.loops.LoopOrchestrator removed - REPL uses core.orchestrator.loop_orchestrator directly

# Performance optimizations - check availability without importing
# (actual imports done lazily in _init_managers to avoid circular imports)
try:
    import importlib.util

    ADAPTIVE_CONTEXT_AVAILABLE = (
        importlib.util.find_spec("core.adaptive_context") is not None
    )
except (ImportError, AttributeError, ValueError):
    ADAPTIVE_CONTEXT_AVAILABLE = False

try:
    PERF_MONITORING_AVAILABLE = (
        importlib.util.find_spec("core.performance_monitor") is not None
    )
except (ImportError, AttributeError, ValueError):
    PERF_MONITORING_AVAILABLE = False

try:
    ERROR_RECOVERY_AVAILABLE = (
        importlib.util.find_spec("core.error_recovery") is not None
    )
except (ImportError, AttributeError, ValueError):
    ERROR_RECOVERY_AVAILABLE = False


class BaseAgent(SerializableMixin):  # Inherit from SerializableMixin
    def __init__(
        self,
        name: str = "Anvil",
        system_prompt_prefix: str = None,
        max_steps: int = 15,
        output_format: str = "text",
        brain: Optional[DeterministicOllama] = None,
        env_info: Optional[Dict[str, str]] = None,
        history_file: Optional[str] = "history.json",
        console: Optional[Console] = None,
        initial_state: Optional[Dict[str, Any]] = None,
    ):  # Add initial_state for deserialization
        self.name = name
        self.output_format = output_format
        self.coconut_enabled = AGENTIC_THINKING.get("coconut_enabled", False)

        # Smart Truncation for Tool Outputs (Claude Code style)
        self.truncator = SmartTruncator(char_threshold=250000)
        self.env_info = env_info or {}  # Virtual environment info
        # Suppress console handling if outputting JSON
        quiet = self.output_format == "json"
        self.console = console or Console(quiet=quiet)

        if initial_state:
            # Reconstruct from initial_state
            self.history = ConversationHistory.from_dict(
                initial_state.get("history", {})
            )
            self.config = ConfigManager.from_dict(initial_state.get("config", {}))
            self.name = initial_state.get("name", name)
            self.output_format = initial_state.get("output_format", output_format)
            self.max_autonomous_steps = initial_state.get(
                "max_autonomous_steps", max_steps
            )
        else:
            self.history = ConversationHistory(history_file=history_file)
            self.config = ConfigManager()

        self.session_id = getattr(self.history, "session_id", None)
        self.audit_db = None
        db_path = getattr(self.history, "db_path", ".anvil/anvil.db")
        if not isinstance(db_path, str):
            db_path = ".anvil/anvil.db"
        history_ready = callable(getattr(self.history, "get_messages", None))
        if db_path and history_ready:
            try:
                self.audit_db = get_anvil_db(db_path)
                if self.session_id:
                    self.audit_db.ensure_session(
                        self.session_id,
                        metadata={"agent": self.name, "output_format": self.output_format},
                    )
            except (OSError, sqlite3.Error, TypeError, ValueError):
                self.audit_db = None

        # Initialize prompt manager
        self.prompt_manager = PromptManager()

        # Initialize Managers and Components
        self._init_managers(brain)
        self.event_store = get_event_store()
        self.reality_graph = RealityGraphStore(os.getcwd())
        self._runtime_event_counter = 0
        self.context_token_manager = ContextManager(
            max_tokens=GENERATION_PARAMS.get("num_ctx", 400000),
            system_prompt_tokens=2000,
        )
        self.context_compression_memory = ContextCompressionMemory()
        self.context_compression_session_id = self._get_context_compression_session_id()
        self.tool_call_counter = infer_next_tc_id(self.history.get_messages())
        self._active_tool_execution = False
        self._last_tool_execution_meta: Dict[str, Any] = {}
        self.soft_compact_threshold = 85.0
        self.hard_compact_threshold = 92.0
        self._soft_compact_noted = False
        self._hard_compact_noted = False
        self._rehydrate_compressed_history()

        # Lazy-initialized UnifiedChatLoop (REPL uses its own LoopOrchestrator with dashboard)
        self._unified_loop = None

    @property
    def unified_loop(self):
        """Lazy-load UnifiedChatLoop for BaseAgent.run()."""
        if self._unified_loop is None:
            from core.unified_chat_loop import UnifiedChatLoop

            self._unified_loop = UnifiedChatLoop(
                self,
                enhanced_mode=getattr(self, "enhanced_loop_enabled", True),
            )
        return self._unified_loop

    def run(self, user_input: str) -> str:
        """Top-level entry point using UnifiedChatLoop directly."""
        return self.unified_loop.run(user_input)

    def _ensure_runtime_compliance_context(self, task: str = "") -> Dict[str, Any]:
        seed = task or self.name
        trace_id = getattr(self, "current_mission_id", None)
        if not trace_id:
            trace_hash = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:10]
            trace_id = f"trace::{int(time.time())}::{trace_hash}"

        evidence_bundle_id = getattr(self, "current_evidence_bundle_id", None)
        if not evidence_bundle_id:
            evidence_bundle_id = f"evidence::{trace_id}"

        waiver_id = getattr(self, "current_waiver_id", None)
        waiver_ids = getattr(self, "current_waiver_ids", None)
        if waiver_ids is None:
            waiver_ids = [waiver_id] if waiver_id else []
        waiver_ids = [str(item).strip() for item in waiver_ids if str(item).strip()]
        red_team_required = bool(getattr(self, "current_red_team_required", False))

        self.current_mission_id = trace_id
        self.current_evidence_bundle_id = evidence_bundle_id
        self.current_waiver_id = waiver_id
        self.current_waiver_ids = waiver_ids
        self.current_red_team_required = red_team_required
        return {
            "trace_id": trace_id,
            "evidence_bundle_id": evidence_bundle_id,
            "waiver_id": waiver_id,
            "waiver_ids": waiver_ids,
            "red_team_required": red_team_required,
        }

    def _format_runtime_compliance_context(self, task: str = "") -> str:
        context = self._ensure_runtime_compliance_context(task)
        waiver_ids = context.get("waiver_ids") or []
        waiver_text = ", ".join(str(item) for item in waiver_ids) if waiver_ids else "none"
        return (
            "## ACTIVE ASSURANCE CONTEXT\n"
            f"Trace ID: {context['trace_id']}\n"
            f"Evidence Bundle ID: {context['evidence_bundle_id']}\n"
            f"Waiver IDs: {waiver_text}\n"
            f"Red Team Required: {bool(context.get('red_team_required'))}"
        )

    def _record_runtime_event(
        self,
        event_type: str,
        *,
        phase: Optional[str] = None,
        status: Optional[str] = None,
        files: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
        tests: Optional[List[str]] = None,
        tool_calls: Optional[List[str]] = None,
        counterexamples: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, str]] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        context = self._ensure_runtime_compliance_context(event_type)
        self._runtime_event_counter += 1
        runtime_metadata = dict(metadata or {})
        runtime_metadata.setdefault(
            "signal_id",
            f"{context['trace_id']}::sig::{self._runtime_event_counter:04d}",
        )
        runtime_metadata.setdefault("evidence_bundle_id", context["evidence_bundle_id"])
        runtime_metadata.setdefault("waiver_ids", list(context.get("waiver_ids") or []))
        runtime_metadata.setdefault(
            "red_team_required", bool(context.get("red_team_required"))
        )
        normalized_files = sorted({str(item) for item in (files or []) if str(item).strip()})
        normalized_symbols = sorted(
            {str(item) for item in (symbols or []) if str(item).strip()}
        )
        normalized_tests = sorted({str(item) for item in (tests or []) if str(item).strip()})
        normalized_tools = sorted(
            {str(item) for item in (tool_calls or []) if str(item).strip()}
        )
        normalized_counterexamples = [dict(item) for item in (counterexamples or [])]

        event_payload = {
            "run_id": context["trace_id"],
            "phase": phase,
            "status": status,
            "files": normalized_files,
            "symbols": normalized_symbols,
            "tests": normalized_tests,
            "tool_calls": normalized_tools,
            "counterexamples": normalized_counterexamples,
            "artifacts": dict(artifacts or {}),
            "metadata": runtime_metadata,
        }
        event_result = {
            "run_id": context["trace_id"],
            "signal_id": runtime_metadata["signal_id"],
        }
        try:
            event_result.update(
                self.event_store.emit(
                    event_type=event_type,
                    source=source or self.name,
                    payload=event_payload,
                    metadata={
                        "run_id": context["trace_id"],
                        "phase": phase,
                        "status": status,
                        "files": normalized_files,
                        "symbols": normalized_symbols,
                        "tests": normalized_tests,
                        "tool_calls": normalized_tools,
                        "counterexamples": normalized_counterexamples,
                        "artifacts": sorted((artifacts or {}).values()),
                        **runtime_metadata,
                    },
                    run_id=context["trace_id"],
                )
            )
        except Exception:
            pass
        try:
            self.reality_graph.record_event(
                event_type,
                run_id=context["trace_id"],
                task_id=context["trace_id"],
                phase=phase,
                status=status,
                files=normalized_files,
                symbols=normalized_symbols,
                tests=normalized_tests,
                tool_calls=normalized_tools,
                counterexamples=normalized_counterexamples,
                metadata=runtime_metadata,
                artifacts=artifacts or {},
                source=source or self.name,
                segment_id=runtime_metadata["signal_id"],
            )
        except Exception:
            pass
        return event_result

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the object's state into a dictionary."""
        return {
            "name": self.name,
            "output_format": self.output_format,
            "max_autonomous_steps": self.max_autonomous_steps,
            "history": self.history.to_dict(),
            "config": self.config.to_dict(),
            # Other complex objects will need their own to_dict implementations
            # For now, we only serialize basic types and already serializable objects.
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], console: Optional[Console] = None):
        """Deserializes an object from a dictionary."""
        # Reconstruct BaseAgent using the initial_state argument
        agent = cls(
            name=data.get("name"),
            output_format=data.get("output_format"),
            max_steps=data.get("max_autonomous_steps"),
            console=console,
            initial_state=data,  # Pass the whole dict as initial_state
        )
        return agent

    def _init_managers(self, brain: Optional[DeterministicOllama] = None):
        """Initialize all agent managers and core components."""
        # New Managers (Must be init before Brain/Registry if they depend on config)
        mode = ApprovalMode(self.config.get("approval_mode", "full-auto"))
        policy_profile = self.config.get("policy_profile")
        self.approval_manager = ApprovalManager(
            mode=mode,
            policy_profile=policy_profile,
            audit_callback=self._on_policy_audit,
            session_id=self.session_id,
        )
        self.project_context = ProjectContextManager()
        self.checkpoint_manager = CheckpointManager()
        self.session_manager = SessionManager()
        self.latent_memory = LatentMemory()
        self.correction_manager = CorrectionManager()

        # Proactive scanning (Legacy - will be superseded by SemanticEngine in REPL)
        self.proactive_context = ProactiveContextManager(root_dir=".")

        # Core Components
        self.brain = brain or DeterministicOllama(
            self.config.get("model", MASTER_MODEL)
        )
        self.pipeline_manager = PipelineManager(self.brain)
        self.semantic_engine = SemanticEngine(root_dir=".", brain=self.brain)

        from core.orchestrator import MissionDecomposer

        self.mission_decomposer = MissionDecomposer(brain=self.brain)

        # Hook Registry
        from core.hooks.registry import HookRegistry
        from core.hooks.aes_pre_verify import AESPreVerifyHook
        from core.hooks.builtin import (
            AALClassifyHook,
            ChronicleHook,
            ToolAuditHook,
            PrivacySafetyHook,
            TimingHook,
            SaguaroSyncHook,
        )

        self.hook_registry = HookRegistry()

        # Register default hooks
        self.hook_registry.register("pre_tool_use", ToolAuditHook())
        self.hook_registry.register("pre_tool_use", PrivacySafetyHook())
        self.hook_registry.register("post_tool_use", TimingHook())
        self.hook_registry.register("post_tool_use", SaguaroSyncHook())
        self.hook_registry.register("post_write_verify", AESPreVerifyHook(repo_path="."))
        self.hook_registry.register("pre_dispatch", AALClassifyHook())
        self.hook_registry.register("post_dispatch", ChronicleHook())

        # Tool Registry
        self.registry = ToolRegistry(
            console=self.console,
            brain=self.brain,
            semantic_engine=self.semantic_engine,
            agent=self,
        )

        self.max_autonomous_steps = self.config.get("max_steps", 15)
        self.tool_schemas = self.registry.get_schemas().get("tools", [])

        # Lazy-load complex subsystems
        self.adaptive_context_manager = None
        if (
            PERFORMANCE_CONFIG.get("adaptive_context", False)
            and ADAPTIVE_CONTEXT_AVAILABLE
        ):
            from core.adaptive_context import AdaptiveContextManager, ContextCompressor

            self.adaptive_context_manager = AdaptiveContextManager()
            self.context_compressor = ContextCompressor()

        self.perf_monitor = None
        if (
            PERFORMANCE_CONFIG.get("enable_perf_monitoring", False)
            and PERF_MONITORING_AVAILABLE
        ):
            from core.performance_monitor import PerformanceMonitor

            self.perf_monitor = PerformanceMonitor()

        if ERROR_RECOVERY_AVAILABLE:
            from core.error_recovery import ErrorRecoveryManager

            self.error_recovery = ErrorRecoveryManager()

    def _get_cwd_context(self) -> str:
        """Returns a string describing the current working directory state."""
        try:
            cwd = os.getcwd()
            files = os.listdir(cwd)
            visible_files = [f for f in files if not f.startswith(".")]
            file_list_str = ", ".join(visible_files[:50])
            if len(visible_files) > 50:
                file_list_str += f" ... (+{len(visible_files)-50} more)"

            return f"\nCURRENT WORKING DIRECTORY: {cwd}\nFILES IN CONTEXT: {file_list_str}\n"
        except OSError as e:
            return f"Error reading CWD: {e}"

    def _get_context_compression_session_id(self) -> str:
        if self.session_id:
            return f"session:{self.session_id}"
        history_file = getattr(self.history, "history_file", "")
        if history_file:
            return f"history:{history_file}"
        return f"agent:{self.name}"

    def _current_wall_clock(self) -> str:
        return datetime.now().astimezone().isoformat()

    def _monotonic_elapsed_ms(self) -> int:
        return int(time.monotonic() * 1000)

    def _log_timeline_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self.audit_db is None or not self.session_id:
            return
        try:
            self.audit_db.log_timeline_event(
                session_id=self.session_id,
                event_type=event_type,
                wall_clock=self._current_wall_clock(),
                monotonic_elapsed_ms=self._monotonic_elapsed_ms(),
                payload=payload,
            )
        except (OSError, sqlite3.Error, TypeError, ValueError):
            return

    def _on_policy_audit(self, event: Dict[str, Any]) -> None:
        if self.audit_db is None:
            return
        try:
            self.audit_db.log_permission_event(
                session_id=self.session_id,
                tool_name=event.get("tool_name", "unknown"),
                decision=event.get("decision", "unknown"),
                reason=event.get("reason"),
                policy_profile=event.get("policy_profile"),
                signature=event.get("signature"),
                metadata=event.get("metadata"),
            )
            decision = str(event.get("decision", "")).lower()
            allowed = decision in {"auto-approved", "approved"}
            self.audit_db.log_policy_evaluation(
                session_id=self.session_id,
                tool_name=event.get("tool_name", "unknown"),
                allowed=allowed,
                reason=event.get("reason"),
                profile=event.get("policy_profile"),
                metadata={
                    "mode": event.get("mode"),
                    "decision": event.get("decision"),
                    "signature": event.get("signature"),
                },
            )
            self._log_timeline_event(
                f"policy:{event.get('decision', 'unknown')}",
                {
                    "tool_name": event.get("tool_name"),
                    "profile": event.get("policy_profile"),
                    "mode": event.get("mode"),
                    "reason": event.get("reason"),
                },
            )
        except (OSError, sqlite3.Error, TypeError, ValueError):
            return

    def _get_temporal_context(self, limit: int = 8) -> str:
        wall_clock = self._current_wall_clock()
        timeline = self.history.get_timeline(limit=limit)
        if not timeline:
            return f"Current wall clock: {wall_clock}\nNo prior timeline events."

        lines = [f"Current wall clock: {wall_clock}", "Recent timeline events:"]
        for event in timeline[-limit:]:
            event_type = event.get("event_type", "unknown")
            event_clock = event.get("wall_clock", "unknown")
            elapsed = event.get("monotonic_elapsed_ms")
            payload = event.get("payload") or {}
            lines.append(
                f"- {event_clock} | {event_type} | elapsed_ms={elapsed} | payload_keys={sorted(payload.keys())}"
            )
        return "\n".join(lines)

    def _on_tool_message_compressed(
        self, tc_id: int, message: Dict[str, Any], summary: str
    ) -> None:
        self.context_compression_memory.remember_summary(
            session_id=self.context_compression_session_id,
            tc_id=tc_id,
            summary=summary,
            tool_name=message.get("tool_name"),
            tool_args=message.get("tool_args", {}),
        )

    def _rehydrate_compressed_history(self) -> None:
        updates = self.context_compression_memory.get_updates_payload(
            self.context_compression_session_id
        )
        if not updates:
            return
        messages = self.history.get_messages()
        outcome = apply_context_updates(
            messages, updates, on_compressed=self._on_tool_message_compressed
        )
        if outcome.get("applied"):
            self.history.save()

    def _auto_compress_dead_context(self) -> None:
        messages = self.history.get_messages()
        pre_hash = self._history_digest(messages)
        pre_count = len(messages)
        compressed = auto_compress_dead_end_reads(
            messages,
            min_age_messages=6,
            on_compressed=self._on_tool_message_compressed,
        )
        if compressed:
            post_hash = self._history_digest(messages)
            self._record_compaction_event(
                event_type="auto_compress_dead_context",
                pre_hash=pre_hash,
                post_hash=post_hash,
                before_messages=pre_count,
                after_messages=len(messages),
                compressed_count=len(compressed),
                metadata={
                    "compressed_tc_ids": list(compressed.keys())[:20],
                    "summary_schema_version": "1.0",
                },
            )
            self.history.save()

    def _history_digest(self, messages: List[Dict[str, Any]]) -> str:
        material = []
        for message in messages:
            material.append(
                {
                    "role": message.get("role"),
                    "content": message.get("content"),
                    "tc_id": message.get("tc_id"),
                    "compressed_from_tc": message.get("compressed_from_tc"),
                    "is_compressed": bool(message.get("is_compressed", False)),
                }
            )
        blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _record_compaction_event(
        self,
        *,
        event_type: str,
        pre_hash: str,
        post_hash: str,
        before_messages: int,
        after_messages: int,
        compressed_count: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.audit_db is None:
            return
        try:
            compression_ratio = 1.0
            if after_messages > 0:
                compression_ratio = round(before_messages / max(1, after_messages), 4)
            payload = {
                "before_messages": before_messages,
                "after_messages": after_messages,
                "compressed_count": compressed_count,
            }
            if metadata:
                payload.update(metadata)

            self.audit_db.log_compaction_event(
                session_id=self.session_id,
                event_type=event_type,
                compression_ratio=compression_ratio,
                pre_hash=pre_hash,
                post_hash=post_hash,
                confidence=1.0 if compressed_count > 0 else 0.0,
                metadata=payload,
            )
        except (OSError, sqlite3.Error, TypeError, ValueError):
            return

    def _run_compaction_replay_check(self) -> Dict[str, Any]:
        messages = self.history.get_messages()
        pinned = {
            "session_id": self.session_id,
            "policy_profile": getattr(
                getattr(self, "approval_manager", None), "policy_profile", None
            ),
            "tool_call_counter": self.tool_call_counter,
        }
        return {
            "ok": bool(pinned["session_id"]) and bool(messages),
            "pinned_state": {
                "session_id": pinned["session_id"],
                "policy_profile": (
                    pinned["policy_profile"].value
                    if hasattr(pinned["policy_profile"], "value")
                    else str(pinned["policy_profile"])
                ),
                "tool_call_counter": pinned["tool_call_counter"],
            },
            "message_count": len(messages),
        }

    def _context_pressure_guidance(self, task: str = "") -> str:
        self._auto_compress_dead_context()
        stats = self.context_token_manager.get_fill_percentage(self.history.get_messages())
        fill = float(stats["fill_percentage"])
        if fill >= self.soft_compact_threshold and not self._soft_compact_noted:
            self._soft_compact_noted = True
            self._log_timeline_event(
                "context:soft_compact",
                {
                    "fill_percentage": round(fill, 2),
                    "threshold": self.soft_compact_threshold,
                },
            )
        if fill < self.soft_compact_threshold:
            self._soft_compact_noted = False

        if fill >= self.hard_compact_threshold and not self._hard_compact_noted:
            self._hard_compact_noted = True
            before_messages = self.history.get_messages()
            before_count = len(before_messages)
            before = self._history_digest(before_messages)
            self._auto_compress_dead_context()
            after_messages = self.history.get_messages()
            after = self._history_digest(after_messages)
            replay = self._run_compaction_replay_check()
            self._record_compaction_event(
                event_type="hard_compact_trigger",
                pre_hash=before,
                post_hash=after,
                before_messages=before_count,
                after_messages=len(after_messages),
                compressed_count=0,
                metadata={
                    "fill_percentage": round(fill, 2),
                    "threshold": self.hard_compact_threshold,
                    "replay_ok": replay.get("ok"),
                    "replay_pinned_state": replay.get("pinned_state"),
                },
            )
        if fill < self.hard_compact_threshold:
            self._hard_compact_noted = False

        guidance = (
            "CONTEXT: "
            f"{stats['used_tokens']} / {stats['max_tokens']} tokens "
            f"({stats['fill_percentage']:.1f}% used). "
            "Compress old tool results via _context_updates on every tool call. "
            f"After {self.soft_compact_threshold:.0f}%, compress aggressively. "
            f"Hard compaction triggers at {self.hard_compact_threshold:.0f}%."
        )

        low_relevance = find_low_relevance_tc_ids(
            task=task or "",
            messages=self.history.get_messages(),
            semantic_engine=self.semantic_engine,
            threshold=0.10,
        )
        if low_relevance:
            guidance += (
                "\nLOW_RELEVANCE_HINT: Prefer compressing these stale results first: "
                f"{', '.join(low_relevance[:8])}."
            )
        return guidance

    def _record_tool_result_message(
        self, tool_name: str, tool_args: Dict[str, Any], result: str
    ) -> None:
        redacted_result = self._redact_sensitive_output(str(result))
        provenance = self._build_tool_provenance(tool_name, tool_args)
        execution_meta = self._last_tool_execution_meta if isinstance(
            self._last_tool_execution_meta, dict
        ) else {}

        tc_id = self.tool_call_counter
        self.tool_call_counter += 1
        content = label_tool_result(
            f"Tool '{tool_name}' Result: {redacted_result}",
            tc_id=tc_id,
        )
        self.history.add_message(
            "tool",
            content,
            tc_id=tc_id,
            tool_name=tool_name,
            tool_args=tool_args or {},
            tool_provenance=provenance,
            hook_receipts=execution_meta.get("hook_receipts", []),
            post_write_verification=execution_meta.get("post_write_verification"),
            is_compressed=False,
        )
        self._last_tool_execution_meta = {}

    def _redact_sensitive_output(self, text: str) -> str:
        return redact_secret_material(text)

    def _build_tool_provenance(
        self, tool_name: str, tool_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        return serialize_tool_provenance(
            tool_name=tool_name,
            tool_args=tool_args,
            trace_id=getattr(self, "current_mission_id", None),
        )

    def _build_system_prompt(self, task: str = "") -> str:
        tracked_files = sorted(
            set(getattr(self, "files_read", set()) or set())
            | set(getattr(self, "files_edited", set()) or set())
        )
        aes_runtime, prompt_contract = self.prompt_manager.aes_builder.build_master_prompt(
            task_text=task,
            task_files=tracked_files or None,
        )
        missing_contract_keys = self.prompt_manager.validate_prompt_contract(prompt_contract)
        if AES_PROMPT_CONTRACT_REQUIRED and missing_contract_keys:
            missing = ", ".join(missing_contract_keys)
            raise RuntimeError(f"AES prompt contract missing required keys: {missing}")
        prompt_contract_block = self.prompt_manager.format_prompt_contract(prompt_contract)
        aal = str(prompt_contract.get("AAL_CLASSIFICATION", "AAL-3"))
        high_aal_task = aal in {"AAL-0", "AAL-1"}
        self.current_red_team_required = high_aal_task

        runtime_compliance = self._format_runtime_compliance_context(task)
        # Tool schema lazy loading: only include names, not full schemas
        if PERFORMANCE_CONFIG.get("tool_lazy_loading", False):
            tool_names = [schema["name"] for schema in self.tool_schemas]
            joined_names = ", ".join(tool_names)
            tools_section = (
                f"Available Tools: {joined_names}\n"
                "(Invoke tools by name. Full schemas provided upon execution.)"
            )
        else:
            # Full schema (legacy mode)
            import json

            tools_json = json.dumps(self.tool_schemas, indent=2)
            tools_section = f"Available Tools:\n{tools_json}"

        variables = {
            "agent_name": self.name,
            "project_context": self.project_context.get_context(),
            "mode_context": self._get_mode_context(),
            "thinking_protocol": self.prompt_manager.get_template("thinking_protocol"),
            "tools_section": tools_section,
            "cwd_context": self._get_cwd_context(),
            "env_context": (
                f"\nVIRTUAL ENVIRONMENT: {self.env_info.get('venv_dir')}\n"
                f"PYTHON EXECUTABLE: {self.env_info.get('python')}\n"
                if self.env_info
                else ""
            ),
            "proactive_context": self.proactive_context.get_context_prompt(),
            "latent_context": self.latent_memory.get_context_prompt(),
        }

        model_name = getattr(self.brain, "model_name", "")
        model_compression_guidance = (
            self.prompt_manager.get_model_family_compression_guidance(model_name)
        )
        context_pressure = self._context_pressure_guidance(task)
        temporal_context = self._get_temporal_context()
        high_aal_instruction = (
            "HIGH-AAL CLOSURE RULE: This output is incomplete unless it includes "
            "traceability and evidence artifacts matching REQUIRED_ARTIFACTS."
            if high_aal_task
            else "Standard closure rule: include verification evidence for all claims."
        )

        return self.prompt_manager.format_prompt("base_agent_core", variables) + f"""
### AES PROMPT CONTRACT
{prompt_contract_block}

### AES RUNTIME CONTRACT
{aes_runtime}

{runtime_compliance}

### STRICT TOOL CALL FORMAT AND NATURAL LANGUAGE MANDATE
You MUST use the following XML-based syntax for ALL tool calls. No exceptions.
Example:
<tool_call>
{"name": "tool_name", "arguments": {"arg1": "value1", "arg2": "value2"}}
</tool_call>

Crucially, you MUST provide at least one sentence of natural language content either BEFORE or AFTER your tool calls. NEVER output ONLY tool calls or ONLY thinking blocks.

Your final response MUST include a brief section (e.g., "Verification" or "Evidence") summarizing which files or tool results were used to confirm your answer, IF a tool was used.

### HIGH-ASSURANCE OUTPUT REQUIREMENT
{high_aal_instruction}

### CONTEXT COMPRESSION (MANDATORY)
_context_updates is REQUIRED on every tool call:
<tool_call>
{"name": "read_file", "arguments": {"path": "core/agent.py", "_context_updates": []}}
</tool_call>

If old `[tcN]` tool results are no longer needed, summarize them:
<tool_call>
{"name": "read_file", "arguments": {"path": "core/context.py", "_context_updates": [{"tc4": "Old grep output had no relevant matches."}]}}
</tool_call>

Results without `[tcN]` are already compressed. Never re-compress them.
""" + f"""

{context_pressure}
{model_compression_guidance}

### ANTI-LOOP PROTOCOL
To prevent repetitive outputs and "semantic loops" when analyzing complex or low-level components (especially native code, C++, or performance optimizations):
1. **Focus on High-Level Summaries**: Do NOT attempt to explain every line of low-level code. Instead, synthesize its *purpose*, *input/output*, and *integration points* within the broader system.
2. **Avoid Repetition**: If you find your thoughts or generated text becoming repetitive (e.g., using the same phrases or cycling through similar ideas), immediately break the pattern.
3. **Explicitly State Limitations**: If you cannot synthesize a non-repetitive, meaningful explanation of a specific low-level detail, state "I lack sufficient high-level context or specialized knowledge to fully explain this low-level component without additional guidance."
4. **Ask for Clarification**: If stuck, ask the user a specific, guiding question to move forward.
5. **Prioritize Actions**: If internal reasoning becomes circular, prioritize using tools to gather new information or attempt a different approach.
""" + f"""

### TEMPORAL CONTEXT
{temporal_context}
"""

    def _get_mode_context(self) -> str:
        """Helper to get mode-specific instructions."""
        try:
            from core.task_state import TaskStateManager
            from core.agent_mode import AgentMode

            manager = TaskStateManager()
            state = manager.get_state()

            if not state:
                return ""

            if state.mode == AgentMode.PLANNING:
                return """
### CURRENT MODE: PLANNING
**Goal:** Research and design a comprehensive implementation plan.
**Objectives:**
1. Use semantic research tools (`query`, `skeleton`, `slice`) to map out the codebase.
2. Identify cross-component dependencies.
3. Produce a structured `implementation_plan.md` artifact.
**Constraint:** Do NOT modify source files. Only read and create planning artifacts.
"""
            elif state.mode == AgentMode.EXECUTION:
                return """
### CURRENT MODE: EXECUTION
**Goal:** Implement the approved changes with precision.
**Objectives:**
1. Follow the `implementation_plan.md` strictly.
2. Update `task.md` as you complete each sub-task.
3. Use `<thinking type="reasoning">` before complex edits.
**Tooling:** Use `write_file` or `edit_file` for modifications.
"""
            elif state.mode == AgentMode.VERIFICATION:
                return """
### CURRENT MODE: VERIFICATION
**Goal:** Prove the implementation is correct and robust.
**Objectives:**
1. Run all relevant tests using `run_tests`.
2. Check syntax and types using `verify_all`.
3. Create a final `walkthrough.md` documenting your success.
**Constraint:** Source files are read-only; you may only modify test files.
"""
        except (ImportError, AttributeError, RuntimeError, ValueError, OSError):
            return ""
        return ""

    @staticmethod
    def _extract_tool_calls_with_helper(text: str) -> List[Dict[str, Any]]:
        return extract_tool_calls_helper(text)

    @staticmethod
    def _execute_tool_with_helper(
        agent: "BaseAgent", tool_call: Dict[str, Any], retries: int, delay: int
    ) -> str:
        return execute_tool_helper(agent, tool_call, retries=retries, delay=delay)

    def _extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        return self._extract_tool_calls_with_helper(text)

    def _execute_tool(
        self, tool_call: Dict[str, Any], retries: int = 3, delay: int = 2
    ) -> str:
        return self._execute_tool_with_helper(
            self, tool_call, retries=retries, delay=delay
        )

    @staticmethod
    def _extract_write_targets(tool_name: str, tool_args: Dict[str, Any]) -> List[str]:
        if tool_name in {"write_file", "edit_file", "apply_patch", "delete_file", "rollback_file"}:
            path = tool_args.get("path") or tool_args.get("file_path")
            return [path] if isinstance(path, str) and path else []
        if tool_name == "write_files":
            files = tool_args.get("files")
            if isinstance(files, dict):
                return [str(path) for path in files.keys()]
            return []
        if tool_name == "move_file":
            targets = []
            for key in ("src", "dst"):
                value = tool_args.get(key)
                if isinstance(value, str) and value:
                    targets.append(value)
            return targets
        return []

    @staticmethod
    def _stream_json_response(
        brain: Any,
        messages: List[Dict[str, Any]],
        assistant_prefix: str,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        try:
            return assistant_prefix + "".join(
                brain.stream_chat(
                    messages,
                    assistant_prefix=assistant_prefix,
                    **(generation_kwargs or {}),
                )
            )
        except (AttributeError, TypeError, ValueError, RuntimeError, OSError):
            return assistant_prefix

    @staticmethod
    def _record_stream_chunk_metrics(metrics: Dict[str, Any], chunk_text: str) -> None:
        now = time.perf_counter()
        delta_ms = int((now - metrics["last_chunk_ts"]) * 1000)
        metrics["last_chunk_ts"] = now
        metrics["chunk_count"] += 1
        metrics["total_chunk_chars"] += len(chunk_text or "")
        should_trace = metrics["trace_stream"] and (
            metrics["chunk_count"] <= 5 or metrics["chunk_count"] % 50 == 0
        )
        if should_trace:
            logger.debug(
                json.dumps(
                    {
                        "component": "stream",
                        "event": "stream.chunk",
                        "mode": metrics["stream_mode"],
                        "mission_id": metrics["mission_id"],
                        "index": metrics["chunk_count"],
                        "chars": len(chunk_text or ""),
                        "delta_ms": delta_ms,
                    },
                    sort_keys=True,
                    ensure_ascii=True,
                )
            )

    @staticmethod
    def _stream_loop_detected(
        new_text: str, stream_history: List[str], metrics: Dict[str, Any]
    ) -> bool:
        clean_text = new_text.strip()
        if not clean_text or len(clean_text) < 2:
            return False

        stream_history.append(clean_text)
        if len(stream_history) > 30:
            stream_history.pop(0)
        if len(stream_history) < 12:
            return False

        last_word = stream_history[-1]
        if stream_history[-6:].count(last_word) >= 5:
            metrics["loop_detector"] = "single_word_repeat"
            metrics["loop_window_len"] = 6
            metrics["loop_history_snapshot"] = stream_history[-10:]
            return True

        for length in range(2, 5):
            if len(stream_history) < length * 2:
                continue
            if stream_history[-length:] == stream_history[-2 * length : -length]:
                metrics["loop_detector"] = "sequence_repeat"
                metrics["loop_window_len"] = length
                metrics["loop_history_snapshot"] = stream_history[-10:]
                return True
        return False

    @staticmethod
    def _recover_non_loop_prefix(text: str, window: int = 200) -> str:
        if len(text) < window * 2:
            return text
        tail = text[-window:]
        prior = text[:-window]
        first_occurrence = prior.find(tail)
        if first_occurrence >= 0:
            return text[: first_occurrence + window]
        return text

    @staticmethod
    def _sanitize_recovered_response(text: str) -> str:
        sanitized = clean_response(text)
        sanitized = re.sub(
            r"<tool_call>.*?</tool_call>",
            "",
            sanitized,
            flags=re.DOTALL,
        )
        sanitized = re.sub(
            r'^\s*\{\s*"name"\s*:\s*"[a-zA-Z0-9_]+"\s*,\s*"arguments"\s*:\s*\{.*\}\s*\}\s*$',
            "",
            sanitized,
            flags=re.DOTALL,
        )
        return sanitized.strip()

    @staticmethod
    def _recover_from_stream_loop(
        agent: Any,
        full_response: str,
        metrics: Dict[str, Any],
        warn_console: bool = False,
    ) -> str:
        if warn_console and getattr(agent, "console", None):
            agent.console.print(
                "\n[bold red]⚠ Detected infinite streaming loop. Recovering valid prefix and breaking stream.[/bold red]"
            )
        metrics["loop_break"] = True
        before_len = len(full_response)
        full_response = BaseAgent._sanitize_recovered_response(
            BaseAgent._recover_non_loop_prefix(full_response)
        )
        metrics["recovered_prefix_len"] = len(full_response)
        emit_structured_event(
            logger,
            component="stream",
            event="stream.loop.detected",
            mission_id=metrics["mission_id"],
            model=metrics["model_name"],
            phase=metrics["stream_mode"],
            metrics={
                "detector": metrics["loop_detector"],
                "window_length": metrics["loop_window_len"],
                "history": metrics["loop_history_snapshot"][-10:],
                "full_response_chars_before_recovery": before_len,
                "recovered_prefix_chars": metrics["recovered_prefix_len"],
            },
            level=logging.WARNING,
        )
        return full_response

    @staticmethod
    def _emit_stream_start(
        metrics: Dict[str, Any],
        messages: List[Dict[str, Any]],
        assistant_prefix: str,
        stop_on_tool: bool,
    ) -> None:
        emit_structured_event(
            logger,
            component="stream",
            event="stream.start",
            mission_id=metrics["mission_id"],
            model=metrics["model_name"],
            phase=metrics["stream_mode"],
            metrics={
                "messages": len(messages),
                "assistant_prefix_chars": len(assistant_prefix or ""),
                "stop_on_tool": bool(stop_on_tool),
                "trace_enabled": metrics["trace_stream"],
            },
            level=logging.INFO,
        )

    @staticmethod
    def _emit_stream_complete(
        metrics: Dict[str, Any], full_response: str, include_recovery_metrics: bool
    ) -> None:
        payload = {
            "chunks": metrics["chunk_count"],
            "chunk_chars": metrics["total_chunk_chars"],
            "response_chars": len(full_response),
            "loop_break": metrics["loop_break"],
            "loop_detector": metrics["loop_detector"],
        }
        if include_recovery_metrics:
            payload.update(
                {
                    "loop_window_length": metrics["loop_window_len"],
                    "recovered_prefix_chars": metrics["recovered_prefix_len"],
                    "response_fallback_applied": metrics["response_fallback_applied"],
                }
            )
        emit_structured_event(
            logger,
            component="stream",
            event="stream.complete",
            mission_id=metrics["mission_id"],
            model=metrics["model_name"],
            phase=metrics["stream_mode"],
            duration_ms=int((time.perf_counter() - metrics["stream_start"]) * 1000),
            metrics=payload,
            level=logging.INFO,
        )

    @staticmethod
    def _process_callback_stream_chunk(
        parser: ResponseStreamParser,
        chunk: str,
        full_response: str,
        callback: Any,
        stop_on_tool: bool,
    ) -> Tuple[str, bool]:
        tool_called = False
        for event in parser.process_chunk(chunk):
            full_response += event.content
            logger.debug(
                f"Stream event: {event.type} | Content length: {len(event.content)}"
            )
            callback(event)
            if stop_on_tool and event.type == "tool_start":
                logger.info("Tool call detected in stream, stopping generation.")
                tool_called = True
                break
        return full_response, tool_called

    @staticmethod
    def _stream_response_streaming_ui(
        agent: Any,
        parser: ResponseStreamParser,
        messages: List[Dict[str, Any]],
        assistant_prefix: str,
        callback: Any,
        stop_on_tool: bool,
        metrics: Dict[str, Any],
        stream_history: List[str],
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        full_response = ""
        if assistant_prefix:
            for event in parser.process_chunk(assistant_prefix):
                full_response += event.content
                callback(event)

        tool_called = False
        generator = agent.brain.stream_chat(
            messages,
            assistant_prefix=assistant_prefix,
            **(generation_kwargs or {}),
        )
        for chunk in generator:
            BaseAgent._record_stream_chunk_metrics(metrics, chunk)
            if BaseAgent._stream_loop_detected(chunk, stream_history, metrics):
                full_response = BaseAgent._recover_from_stream_loop(
                    agent, full_response, metrics, warn_console=False
                )
                if os.getenv("ANVIL_STREAM_TRACE", "0") != "1":
                    full_response += "\n[SYSTEM: Streaming loop terminated.]"
                break
            full_response, tool_called = BaseAgent._process_callback_stream_chunk(
                parser, chunk, full_response, callback, stop_on_tool
            )
            if tool_called:
                break

        for event in parser.finalize():
            full_response += event.content
            callback(event)
        return full_response

    @staticmethod
    def _process_nested_prefix_event(agent: Any, event: Any, full_response: str) -> str:
        if event.type == "content":
            agent.console.print(event.content, end="")
            return full_response + event.content
        if event.type == "thinking_start":
            t_type = event.metadata.get("type", "reasoning")
            agent.console.print(f"\n[bold dim]╭─ Thinking ({t_type})[/bold dim]")
            return full_response
        if event.type == "thinking_chunk":
            agent.console.print(f"[dim italic]{event.content}[/dim italic]", end="")
            return full_response + event.content
        return full_response

    @staticmethod
    def _process_nested_stream_event(agent: Any, event: Any, full_response: str) -> str:
        if event.type == "content":
            agent.console.print(event.content, end="")
            return full_response + event.content
        if event.type == "thinking_start":
            t_type = event.metadata.get("type", "reasoning")
            agent.console.print(f"\n[bold dim]╭─ Thinking ({t_type})[/bold dim]")
            return full_response + f'<thinking type="{t_type}">'
        if event.type == "thinking_chunk":
            agent.console.print(f"[dim italic]{event.content}[/dim italic]", end="")
            return full_response + event.content
        if event.type == "thinking_end":
            agent.console.print("[bold dim]╰────────────────[/bold dim]\n")
            return full_response + "</thinking>"
        if event.type == "tool_start":
            agent.console.print("\n[bold cyan]→ Tool Call detected...[/bold cyan]")
            return full_response
        if event.type == "tool_chunk":
            return full_response + event.content
        return full_response

    @staticmethod
    def _stream_response_nested_live(
        agent: Any,
        parser: ResponseStreamParser,
        messages: List[Dict[str, Any]],
        assistant_prefix: str,
        stop_on_tool: bool,
        metrics: Dict[str, Any],
        stream_history: List[str],
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        full_response = ""
        if assistant_prefix:
            for event in parser.process_chunk(assistant_prefix):
                full_response = BaseAgent._process_nested_prefix_event(
                    agent, event, full_response
                )

        generator = agent.brain.stream_chat(
            messages,
            assistant_prefix=assistant_prefix,
            **(generation_kwargs or {}),
        )
        for chunk in generator:
            BaseAgent._record_stream_chunk_metrics(metrics, chunk)
            if BaseAgent._stream_loop_detected(chunk, stream_history, metrics):
                full_response = BaseAgent._recover_from_stream_loop(
                    agent, full_response, metrics, warn_console=True
                )
                break
            logger.debug(f"Raw chunk (nested): {repr(chunk)}")
            for event in parser.process_chunk(chunk):
                full_response = BaseAgent._process_nested_stream_event(
                    agent, event, full_response
                )
            if stop_on_tool and parser.in_tool_call:
                break
        return full_response

    @staticmethod
    def _get_live_renderable(state: Dict[str, Any]) -> Group:
        items = list(state["turn_history"])
        if state["active_panel"]:
            items.append(state["active_panel"])
        else:
            items.append(state["current_content"])
        return Group(*items)

    @staticmethod
    def _process_live_prefix_event(
        event: Any, state: Dict[str, Any], full_response: str
    ) -> str:
        if event.type == "thinking_start":
            t_type = event.metadata.get("type", "reasoning")
            state["active_panel_text"] = Text(style="dim italic")
            state["active_panel"] = Panel(
                state["active_panel_text"],
                title=f"Thinking ({t_type})",
                border_style="dim",
            )
            return full_response
        if event.type == "thinking_chunk":
            state["active_panel_text"].append(event.content)
            return full_response + event.content
        if event.type == "content":
            state["current_content"].append(event.content)
            return full_response + event.content
        return full_response

    @staticmethod
    def _live_handle_content_event(
        event: Any, state: Dict[str, Any], full_response: str
    ) -> str:
        if state["active_panel"]:
            state["active_panel"].border_style = "green"
            state["turn_history"].append(state["active_panel"])
            state["active_panel"] = None
            state["current_content"] = Text()
        state["current_content"].append(event.content)
        return full_response + event.content

    @staticmethod
    def _live_handle_thinking_start_event(
        event: Any, state: Dict[str, Any], full_response: str
    ) -> str:
        if not state["active_panel"] and state["current_content"]:
            state["turn_history"].append(state["current_content"])
            state["current_content"] = Text()
        state["active_panel_text"] = Text(style="dim italic")
        t_type = event.metadata.get("type", "reasoning")
        state["active_panel"] = Panel(
            state["active_panel_text"],
            title=f"Thinking ({t_type})",
            border_style="dim",
        )
        return full_response + f'<thinking type="{t_type}">'

    @staticmethod
    def _live_handle_thinking_chunk_event(
        event: Any, state: Dict[str, Any], full_response: str
    ) -> str:
        state["active_panel_text"].append(event.content)
        return full_response + event.content

    @staticmethod
    def _live_handle_thinking_end_event(
        agent: Any,
        parser: ResponseStreamParser,
        state: Dict[str, Any],
        full_response: str,
    ) -> Tuple[str, bool]:
        content_str = str(state["active_panel_text"]).strip()
        full_response += "</thinking>"
        break_events = False
        active_panel = state["active_panel"]
        if active_panel:
            if content_str and len(content_str) > 5:
                latent_memory = getattr(agent, "latent_memory", None)
                add_thought = getattr(latent_memory, "add_thought", None)
                if callable(add_thought):
                    add_thought(parser.thinking_type, content_str)
                active_panel.border_style = "green"
                active_panel.title = f"Thought complete ({parser.thinking_type})"
                state["turn_history"].append(active_panel)
                state["consecutive_empty_thoughts"] = 0
            else:
                state["consecutive_empty_thoughts"] += 1
                if getattr(agent, "console", None):
                    empty_count = state["consecutive_empty_thoughts"]
                    if empty_count < 3:
                        agent.console.print(
                            f"  [dim yellow]⚠ Empty {parser.thinking_type} thinking block detected and skipped[/dim yellow]"
                        )
                    else:
                        agent.console.print(
                            f"  [bold red]⚠ Detected infinite empty thinking loop ({empty_count}). Breaking stream.[/bold red]"
                        )
                        full_response += "\n[SYSTEM: Thinking loop terminated.]"
                        break_events = True
            state["active_panel"] = None
        state["current_content"] = Text()
        return full_response, break_events

    @staticmethod
    def _live_handle_tool_start_event(
        state: Dict[str, Any], full_response: str
    ) -> str:
        if not state["active_panel"] and state["current_content"]:
            state["turn_history"].append(state["current_content"])
            state["current_content"] = Text()
        state["active_panel_text"] = Text(style="dim")
        state["active_panel"] = Panel(
            state["active_panel_text"],
            title="Action / Tool Call",
            border_style="cyan",
        )
        return full_response + "<tool_call>"

    @staticmethod
    def _live_handle_tool_chunk_event(
        event: Any, state: Dict[str, Any], full_response: str
    ) -> str:
        state["active_panel_text"].append(event.content)
        return full_response + event.content

    @staticmethod
    def _live_handle_tool_end_event(
        state: Dict[str, Any], full_response: str, stop_on_tool: bool
    ) -> Tuple[str, bool]:
        full_response += "</tool_call>"
        if state["active_panel"]:
            state["active_panel"].border_style = "cyan"
            state["turn_history"].append(state["active_panel"])
            state["active_panel"] = None
        state["current_content"] = Text()
        return full_response, bool(stop_on_tool)

    @staticmethod
    def _process_live_stream_event(
        agent: Any,
        event: Any,
        parser: ResponseStreamParser,
        state: Dict[str, Any],
        full_response: str,
        stop_on_tool: bool,
    ) -> Tuple[str, bool, bool]:
        if event.type == "content":
            updated = BaseAgent._live_handle_content_event(event, state, full_response)
            return updated, False, False
        if event.type == "thinking_start":
            updated = BaseAgent._live_handle_thinking_start_event(
                event, state, full_response
            )
            return updated, False, False
        if event.type == "thinking_chunk":
            updated = BaseAgent._live_handle_thinking_chunk_event(
                event, state, full_response
            )
            return updated, False, False
        if event.type == "thinking_end":
            updated, break_events = BaseAgent._live_handle_thinking_end_event(
                agent, parser, state, full_response
            )
            return updated, break_events, False
        if event.type == "tool_start":
            updated = BaseAgent._live_handle_tool_start_event(state, full_response)
            return updated, False, False
        if event.type == "tool_chunk":
            updated = BaseAgent._live_handle_tool_chunk_event(
                event, state, full_response
            )
            return updated, False, False
        if event.type == "tool_end":
            updated, tool_called = BaseAgent._live_handle_tool_end_event(
                state, full_response, stop_on_tool
            )
            return updated, False, tool_called
        return full_response, False, False

    @staticmethod
    def _finalize_live_parser(
        parser: ResponseStreamParser, state: Dict[str, Any], full_response: str
    ) -> str:
        for event in parser.finalize():
            if event.type == "content":
                state["current_content"].append(event.content)
                full_response += event.content
            elif event.type in {"thinking_chunk", "tool_chunk"}:
                state["active_panel_text"].append(event.content)
                full_response += event.content
        if state["active_panel"]:
            state["turn_history"].append(state["active_panel"])
        if state["current_content"]:
            state["turn_history"].append(state["current_content"])
        return full_response

    @staticmethod
    def _stream_response_live(
        agent: Any,
        parser: ResponseStreamParser,
        messages: List[Dict[str, Any]],
        assistant_prefix: str,
        stop_on_tool: bool,
        metrics: Dict[str, Any],
        stream_history: List[str],
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        full_response = ""
        state: Dict[str, Any] = {
            "turn_history": [],
            "current_content": Text(),
            "active_panel": None,
            "active_panel_text": Text(),
            "consecutive_empty_thoughts": 0,
        }
        tool_called = False
        with Live(
            BaseAgent._get_live_renderable(state),
            console=agent.console,
            refresh_per_second=10,
        ) as live:
            if assistant_prefix:
                for event in parser.process_chunk(assistant_prefix):
                    full_response = BaseAgent._process_live_prefix_event(
                        event, state, full_response
                    )

            logger.info("Starting response stream")
            generator = agent.brain.stream_chat(
                messages,
                assistant_prefix=assistant_prefix,
                **(generation_kwargs or {}),
            )
            for chunk in generator:
                BaseAgent._record_stream_chunk_metrics(metrics, chunk)
                logger.debug(f"Raw chunk: {repr(chunk)}")
                if BaseAgent._stream_loop_detected(chunk, stream_history, metrics):
                    full_response = BaseAgent._recover_from_stream_loop(
                        agent, full_response, metrics, warn_console=True
                    )
                    break
                for event in parser.process_chunk(chunk):
                    full_response, break_events, tool_called = (
                        BaseAgent._process_live_stream_event(
                            agent,
                            event,
                            parser,
                            state,
                            full_response,
                            stop_on_tool,
                        )
                    )
                    live.update(BaseAgent._get_live_renderable(state))
                    if break_events or tool_called:
                        break
                if tool_called:
                    break

            full_response = BaseAgent._finalize_live_parser(parser, state, full_response)
            live.update(BaseAgent._get_live_renderable(state))
        return full_response

    @staticmethod
    def _apply_response_fallback(
        full_response: str, parser: ResponseStreamParser
    ) -> Tuple[str, bool]:
        artifact_only = bool(
            re.fullmatch(
                r'\s*\{\s*"name"\s*:\s*"[a-zA-Z0-9_]+"\s*,\s*"arguments"\s*:\s*\{.*\}\s*\}\s*',
                full_response,
                flags=re.DOTALL,
            )
        )
        has_only_thinking = (
            "<thinking" in full_response
            and "</thinking>" in full_response
            and not re.sub(
                r"<thinking.*?>.*?</thinking>", "", full_response, flags=re.DOTALL
            ).strip()
        )
        needs_fallback = (not full_response.strip()) or artifact_only or has_only_thinking
        if not needs_fallback:
            return full_response, False

        blocks = parser.THINK_START.findall(full_response)
        if blocks:
            full_response += (
                "\n\nI have processed the request through internal reasoning. "
                "Please see the thinking blocks above for details."
            )
        else:
            full_response = (
                "I have analyzed your request. Please let me know if you would like me "
                "to investigate further using any specific tools."
            )
        return full_response, True

    def _stream_response(
        self,
        messages: List[Dict[str, Any]],
        assistant_prefix: str = "",
        whitelist: Optional[List[str]] = None,
        stop_on_tool: bool = False,
        callback: Optional[Any] = None,
        is_streaming_ui: bool = False,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Shared streaming logic for run_loop and simple_chat."""
        parser = ResponseStreamParser()
        stream_start = time.perf_counter()
        metrics: Dict[str, Any] = {
            "stream_start": stream_start,
            "last_chunk_ts": stream_start,
            "chunk_count": 0,
            "total_chunk_chars": 0,
            "stream_mode": "streaming_ui" if (is_streaming_ui and callback) else "unknown",
            "trace_stream": (
                os.getenv("ANVIL_STREAM_TRACE", "0") == "1"
                and logger.isEnabledFor(logging.DEBUG)
            ),
            "mission_id": getattr(self, "current_mission_id", None),
            "model_name": getattr(getattr(self, "brain", None), "model_name", None),
            "loop_break": False,
            "loop_detector": None,
            "loop_window_len": 0,
            "recovered_prefix_len": 0,
            "loop_history_snapshot": [],
            "response_fallback_applied": False,
        }
        stream_history: List[str] = []
        if self.output_format == "json":
            return BaseAgent._stream_json_response(
                self.brain,
                messages,
                assistant_prefix,
                generation_kwargs=generation_kwargs,
            )

        if is_streaming_ui and callback:
            BaseAgent._emit_stream_start(metrics, messages, assistant_prefix, stop_on_tool)
            full_response = BaseAgent._stream_response_streaming_ui(
                self,
                parser,
                messages,
                assistant_prefix,
                callback,
                stop_on_tool,
                metrics,
                stream_history,
                generation_kwargs,
            )
            BaseAgent._emit_stream_complete(
                metrics, full_response, include_recovery_metrics=False
            )
            return full_response

        is_nested_live = (
            hasattr(self.console, "_live_display")
            and self.console._live_display is not None
        )
        if metrics["stream_mode"] == "unknown":
            metrics["stream_mode"] = "nested_live" if is_nested_live else "live"
        BaseAgent._emit_stream_start(metrics, messages, assistant_prefix, stop_on_tool)
        if is_nested_live:
            full_response = BaseAgent._stream_response_nested_live(
                self,
                parser,
                messages,
                assistant_prefix,
                stop_on_tool,
                metrics,
                stream_history,
                generation_kwargs,
            )
        else:
            full_response = BaseAgent._stream_response_live(
                self,
                parser,
                messages,
                assistant_prefix,
                stop_on_tool,
                metrics,
                stream_history,
                generation_kwargs,
            )
        full_response, metrics["response_fallback_applied"] = (
            BaseAgent._apply_response_fallback(full_response, parser)
        )
        BaseAgent._emit_stream_complete(
            metrics, full_response, include_recovery_metrics=True
        )
        return full_response

    def run_loop(self, user_input: str) -> Dict[str, Any]:
        """The core thinking/action loop."""
        return run_loop_helper(self, user_input)

    def simple_chat(self, user_input: str) -> str:
        """
        Conversational chat with deep multi-turn thinking and tool support.
        Optimized for evidence-based repository exploration and complex reasoning.
        """
        return simple_chat_helper(self, user_input)

    def simple_chat_enterprise(self, user_input: str) -> str:
        """
        Enterprise-grade chat loop for evidence-based Q&A.
        Optimized for smaller models with structured execution.

        Uses the EnterpriseChatLoop for phase-based execution:
        1. Classification: Determine if code exploration needed
        2. Evidence Gathering: Execute tools systematically
        3. Synthesis: Generate evidence-based answer
        """
        from core.chat_loop_enterprise import EnterpriseChatLoop

        loop = EnterpriseChatLoop(self)
        return loop.run(user_input)
