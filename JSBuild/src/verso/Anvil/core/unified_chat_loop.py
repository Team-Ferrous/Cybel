"""
Unified Chat Loop - Merging Enterprise Chat with Enhanced Tool Calling

Combines the best of both worlds:
- Enterprise Chat: Phase-based execution (classify → gather → action → synthesize)
- Enhanced Tools: Parallel execution, task memory, progressive context, auto-verification

This is the unified intelligent conversation system for Anvil.
"""

import ast
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from rich import box
from rich.panel import Panel

from core.utils.logger import get_logger

logger = get_logger(__name__)

from cli.progress_dashboard import LiveProgressDashboard
from config.settings import AGENTIC_THINKING, DYNAMIC_COCONUT_CONFIG
from core.aes import (
    AALClassifier,
    ActionEscalationEngine,
    AESRuleRegistry,
    ComplianceContext,
    DomainDetector,
    GovernanceEngine,
    ObligationEngine,
    RedTeamProtocol,
    ReviewGate,
    RuntimeGateRunner,
)
from core.agents.specialists import (
    SpecialistRegistry,
    build_specialist_subagent,
    route_specialist,
)
from core.agents.subagent_quality_gate import SubagentQualityGate
from core.cache.response_cache import ResponseCache
from core.context import ContextBudgetAllocator, ContextManager
from core.context_compression import (
    apply_context_updates,
    auto_compress_dead_end_reads,
    ensure_context_updates_arg,
    find_low_relevance_tc_ids,
    infer_next_tc_id,
    label_tool_result,
)
from core.campaign.roadmap_validator import RoadmapValidator as CampaignRoadmapValidator
from core.evidence_builder import EvidenceBuilder
from core.hallucination_gate import HallucinationGate
from core.loops.phases import (
    EvidencePhase,
    ExecutionPhase,
    SynthesisPhase,
    UnderstandingPhase,
)
from core.multi_file_refactor import MultiFileRefactorer

# Import enhancement modules
from core.parallel_executor import ParallelToolExecutor, SaguaroParallelSearch
from core.performance_metrics import get_performance_monitor
from core.progressive_context import ProgressiveContextLoader, SmartContextExpander
from core.prompts import PromptManager
from core.reasoning.complexity_scorer import ComplexityProfile, ComplexityScorer
from core.response_utils import clean_response, finalize_synthesis
from core.smart_context_manager import ContextOptimizer, SmartContextManager
from core.smart_editor import SmartFileEditor
from core.subagent_communication import MessageBus, MessageType, Priority
from core.task_memory import (
    AdaptiveLearner,
    ContextCompressionMemory,
    TaskMemory,
    TaskMemoryManager,
)
from core.telemetry.black_box import BlackBoxRecorder
from core.thinking import EnhancedThinkingSystem, ThinkingType
from core.runtime_control_policy import RuntimeControlPolicy
from core.artifacts.compiled_mission_plan import CompiledMissionPlan
from domains.verification.auto_verifier import AutoVerifier, VerificationLoop
from saguaro.reality.store import RealityGraphStore
from saguaro.sentinel.policy import PolicyManager
from saguaro.sentinel.verifier import SentinelVerifier

INTENT_PROTOTYPES = {
    "question": "explain architecture how something works where implementation lives what is this",
    "creation": "create new file implement feature scaffold build component from scratch",
    "modification": "fix bug refactor improve optimize change existing behavior",
    "deletion": "delete remove drop cleanup old file or obsolete logic",
    "investigation": "investigate trace find usage locate references debug root cause",
    "mission": "implement complete plan end-to-end full implementation execute roadmap",
    "conversational": "general conversation casual response greeting acknowledgement",
}


class UnifiedChatLoop:
    """
    Main orchestration loop for the Anvil.
    """

    def __init__(self, agent, enhanced_mode: bool = True):
        self.response_cache = ResponseCache()
        """
        Initialize unified chat loop.
        
        Args:
            agent: The parent agent instance
            enhanced_mode: If True, all enhancements are active (default).
                          If False, falls back to basic enterprise chat behavior.
        """
        self.agent = agent
        self.console = agent.console
        self.brain = agent.brain
        self.history = agent.history
        self.registry = agent.registry
        self.semantic_engine = agent.semantic_engine
        self.approval_manager = agent.approval_manager
        self.pipeline_manager = getattr(agent, "pipeline_manager", None)

        # Enhancement mode toggle
        self.enhanced_mode = enhanced_mode

        # Initialize Saguaro integration
        from domains.code_intelligence.saguaro_substrate import SaguaroSubstrate
        from tools.saguaro_tools import SaguaroTools

        self.saguaro = SaguaroSubstrate()
        self.saguaro_tools = SaguaroTools(self.saguaro)
        self.reality_graph = RealityGraphStore(self.saguaro.root_dir)
        self.evidence_builder = EvidenceBuilder(
            self.saguaro_tools,
            self.registry,
            self.console,
            repo_root=self.saguaro.root_dir,
        )
        try:
            from core.multi_agent_gatherer import MultiAgentEvidenceGatherer

            self.multi_agent_gatherer = MultiAgentEvidenceGatherer(
                brain=self.brain, console=self.console, registry=self.registry
            )
        except Exception as exc:
            logger.warning("Multi-agent gatherer unavailable: %s", exc)
            self.multi_agent_gatherer = None
        self.hallucination_gate = HallucinationGate()
        self.runtime_control_policy = RuntimeControlPolicy()

        # Initialize prompt manager
        self.prompt_manager = PromptManager()

        # Initialize prompt manager
        self.prompt_manager = PromptManager()

        # Initialize enhancement modules (Eagerly loaded as requested)
        self._init_enhancements()
        self.black_box = BlackBoxRecorder(
            self.saguaro.root_dir,
            reality_graph=self.reality_graph,
        )

        # Session tracking
        self.files_read = set()
        self.files_edited = set()
        self.current_task_id = None
        self.current_complexity_profile: Optional[ComplexityProfile] = None
        self.current_adaptive_complexity: Dict[str, Any] = {}
        self._last_classification_meta: Dict[str, Any] = {}
        self.current_compliance_context: Dict[str, Any] = {
            "run_id": None,
            "aal": "AAL-3",
            "domains": [],
            "changed_files": [],
            "hot_paths": [],
            "public_api_changes": [],
            "dependency_changes": [],
            "required_rule_ids": [],
            "required_runtime_gates": [],
            "trace_id": None,
            "evidence_bundle_id": None,
            "red_team_required": False,
            "waiver_ids": [],
            "waiver_id": None,
            "runtime_posture": None,
            "runtime_control": {},
            "compiled_plan_path": None,
            "mission_thread_id": None,
            "workset_id": None,
            "sync_receipt_id": None,
        }
        self.current_runtime_control: Dict[str, Any] = {}
        self.session_start = time.time()
        self.context_budget_allocator = ContextBudgetAllocator(total_budget=400000)
        self.context_token_manager = ContextManager(
            max_tokens=self.context_budget_allocator.get_budget("master")
            + self.context_budget_allocator.get_budget("system"),
            system_prompt_tokens=2000,
        )
        self.compression_memory = ContextCompressionMemory()
        self.compression_session_id = self._get_context_compression_session_id()
        self.tool_call_counter = infer_next_tc_id(self.history.get_messages())
        self._rehydrate_compressed_history()

        self.complexity_scorer = ComplexityScorer()
        try:
            from core.reasoning.complexity_analyzer import TaskComplexityAnalyzer

            self.task_complexity_analyzer = TaskComplexityAnalyzer()
        except Exception:
            self.task_complexity_analyzer = None

        # Enhanced thinking system with COCONUT
        self.thinking_system = EnhancedThinkingSystem(
            thinking_budget=AGENTIC_THINKING.get("thinking_budget", 300000),
            coconut_enabled=enhanced_mode
            and AGENTIC_THINKING.get("coconut_enabled", True),
            brain=self.brain,
            model_name=getattr(self.brain, "model_name", None),
        )
        self.subagent_quality_gate = SubagentQualityGate(
            repo_root=getattr(self.saguaro, "root_dir", "."),
            brain=self.brain,
            thinking_system=self.thinking_system,
        )
        self.specialist_registry = SpecialistRegistry()
        self.aal_classifier = AALClassifier()
        self.domain_detector = DomainDetector()
        self.aes_rule_registry = AESRuleRegistry()
        rules_path = os.path.join(
            getattr(self.saguaro, "root_dir", "."),
            "standards",
            "AES_RULES.json",
        )
        self.aes_rule_registry.load(rules_path)
        self.governance_engine = GovernanceEngine()
        self.review_gate = ReviewGate()
        self.red_team_protocol = RedTeamProtocol()
        self.action_escalation_engine = ActionEscalationEngine()
        self.policy_manager = PolicyManager(getattr(self.saguaro, "root_dir", "."))
        self.obligation_engine = ObligationEngine(
            os.path.join(
                getattr(self.saguaro, "root_dir", "."),
                "standards",
                "AES_OBLIGATIONS.json",
            )
        )
        self.runtime_gate_runner = RuntimeGateRunner(
            getattr(self.saguaro, "root_dir", ".")
        )
        self._sentinel_verifier: Optional[SentinelVerifier] = None
        self.runtime_aal = "AAL-3"
        self.message_bus = MessageBus(console=self.console)
        self.master_agent_id = f"{self.agent.name}:master"
        try:
            self.message_bus.register_agent(
                self.master_agent_id,
                subscriptions=["progress", "coordination", "context.guidance"],
                metadata={"role": "master"},
            )
        except (AttributeError, RuntimeError, ValueError):
            pass
        self.black_box.bind_message_bus(self.message_bus)

    def _refresh_runtime_bindings(self) -> None:
        """Rebind mutable agent runtime references (e.g., after /model switch)."""
        self.brain = self.agent.brain
        self.history = self.agent.history
        self.registry = self.agent.registry
        self.semantic_engine = self.agent.semantic_engine
        self.approval_manager = self.agent.approval_manager
        self.pipeline_manager = getattr(self.agent, "pipeline_manager", None)
        if hasattr(self.thinking_system, "brain"):
            self.thinking_system.brain = self.brain
        if self.multi_agent_gatherer is not None and hasattr(
            self.multi_agent_gatherer, "brain"
        ):
            self.multi_agent_gatherer.brain = self.brain
        if hasattr(self.subagent_quality_gate, "brain"):
            self.subagent_quality_gate.brain = self.brain

    def _init_enhancements(self):
        """Initialize enhancement modules eagerly."""
        # Parallel Execution
        self.parallel_executor = ParallelToolExecutor(
            self.registry,
            self.semantic_engine,
            self.console,
            self.approval_manager,
            tool_executor=self._execute_tool,
        )
        self.parallel_search = SaguaroParallelSearch(self.saguaro_tools, self.console)

        # Context Management
        self.context_loader = ProgressiveContextLoader(
            self.registry, self.semantic_engine, self.saguaro_tools, self.console
        )
        self.context_expander = SmartContextExpander(self.context_loader)
        self.context_manager = SmartContextManager(self.saguaro, self.console)
        self.context_optimizer = ContextOptimizer(self.saguaro, self.console)

        # Memory & Learning
        self.memory_manager = TaskMemoryManager(
            self.saguaro_tools, self.semantic_engine, self.console
        )
        self.adaptive_learner = AdaptiveLearner(self.memory_manager, self.console)

        # Editor & Verification
        self.smart_editor = SmartFileEditor(
            self.registry,
            self.console,
            tool_executor=self._execute_tool,
        )
        self.verifier = AutoVerifier(self.registry, self.console)
        self.verification_loop = VerificationLoop(self.verifier, self.smart_editor)
        self.multi_file_refactor = MultiFileRefactorer(
            self.registry,
            self.saguaro_tools,
            self.smart_editor,
            self.console,
            tool_executor=self._execute_tool,
        )

        # Phases
        self.understanding_phase = UnderstandingPhase(
            self, self.prompt_manager, self.console
        )
        self.evidence_phase = EvidencePhase(self, self.prompt_manager, self.console)
        self.execution_phase = ExecutionPhase(self, self.prompt_manager, self.console)
        self.synthesis_phase = SynthesisPhase(self, self.prompt_manager, self.console)

    def _get_context_compression_session_id(self) -> str:
        history_file = getattr(self.history, "history_file", None)
        if history_file:
            return f"history:{history_file}"
        return f"agent:{self.agent.name}"

    def _build_risk_budget(
        self, user_input: str, complexity_profile: ComplexityProfile
    ) -> Dict[str, Any]:
        aal = str(self.runtime_aal or "AAL-3").upper()
        high_assurance = aal in {"AAL-0", "AAL-1"}
        budget = {
            "aal": aal,
            "max_tool_calls": 8 if high_assurance else 16,
            "max_branch_candidates": 1 if high_assurance else 3,
            "max_subagents": (
                1
                if high_assurance
                else self._complexity_subagent_slots(complexity_profile)
            ),
            "verification_depth": "strict" if high_assurance else "standard",
            "requires_structural_triggers": high_assurance,
            "task_length_hint": len(user_input.split()),
        }
        return budget

    def _record_reality_event(
        self,
        event_type: str,
        *,
        phase: Optional[str] = None,
        status: Optional[str] = None,
        files: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, str]] = None,
    ) -> None:
        run_id = (
            self.current_compliance_context.get("trace_id")
            or self.current_task_id
            or f"task_{int(time.time())}"
        )
        try:
            self.message_bus.set_trace_context(
                run_id=run_id,
                task_id=self.current_task_id or run_id,
                phase=phase or self.message_bus.trace_context.get("phase"),
            )
            self.black_box.record_event(
                event_type,
                phase=phase,
                status=status,
                files=files or [],
                metadata=metadata or {},
                artifacts=artifacts or {},
                source="core.unified_chat_loop",
            )
        except Exception as exc:
            logger.warning("Reality event recording failed: %s", exc)

    def _record_phase_transition(
        self,
        phase: str,
        status: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message_bus.set_trace_context(
            run_id=self.current_compliance_context.get("trace_id")
            or self.current_task_id,
            task_id=self.current_task_id,
            phase=phase,
        )
        self._record_reality_event(
            "phase_transition",
            phase=phase,
            status=status,
            metadata=metadata or {},
        )
        try:
            run_id = (
                self.current_compliance_context.get("trace_id") or self.current_task_id
            )
            if run_id:
                self.black_box.event_store.record_checkpoint(
                    run_id=str(run_id),
                    phase=phase,
                    status=status,
                    metadata=metadata or {},
                )
        except Exception as exc:
            logger.warning("Phase checkpoint recording failed: %s", exc)

    def _current_prompt_contract_context(
        self,
        tracked_files: List[str],
    ) -> Dict[str, Any]:
        graph_snapshot_id = "graph::current"
        graph_path = (
            Path(self.saguaro.root_dir) / ".saguaro" / "graph" / "code_graph.json"
        )
        if graph_path.exists():
            try:
                payload = json.loads(graph_path.read_text(encoding="utf-8")) or {}
                graph_snapshot_id = str(
                    payload.get("generated_at")
                    or payload.get("generated_fmt")
                    or graph_path.stat().st_mtime
                )
            except Exception:
                graph_snapshot_id = f"graph::{int(graph_path.stat().st_mtime)}"
        policy_decision = self.policy_manager.runtime_decision(
            [],
            aal=self.current_compliance_context.get("aal") or self.runtime_aal,
        )
        connectivity_context = {}
        repo_presence = getattr(self.agent, "repo_presence", None)
        if repo_presence is not None:
            try:
                connectivity_context = dict(repo_presence.build_prompt_context() or {})
            except Exception:
                connectivity_context = {}
        return {
            "trace_id": self.current_compliance_context.get("trace_id"),
            "graph_snapshot_id": graph_snapshot_id,
            "policy_posture": policy_decision.get("decision", "continue"),
            "changed_files": tracked_files,
            "runtime_posture": (
                self.current_runtime_control.get("posture")
                or self.current_compliance_context.get("runtime_posture")
                or policy_decision.get("reason", "no_runtime_posture")
            ),
            "toolchain_state_vector": self.current_compliance_context.get(
                "toolchain_state_vector", []
            ),
            "connectivity_context": connectivity_context,
        }

    def _safe_runtime_status(self) -> Dict[str, Any]:
        getter = getattr(self.brain, "get_runtime_status", None)
        if not callable(getter):
            return {}
        try:
            payload = getter()
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            logger.debug("Runtime status unavailable: %s", exc)
            return {}

    def _current_mission_thread_context(self) -> Dict[str, Any]:
        trace_id = (
            self.current_compliance_context.get("trace_id") or self.current_task_id
        )
        sync_receipt: Dict[str, Any] = {}
        sync_receipt_path = ""
        loader = getattr(self.saguaro, "_load_sync_receipt", None)
        path_getter = getattr(self.saguaro, "_sync_receipt_path", None)
        if callable(loader):
            try:
                payload = loader()
                if isinstance(payload, dict):
                    sync_receipt = payload
            except Exception:
                sync_receipt = {}
        if callable(path_getter):
            try:
                sync_receipt_path = str(path_getter() or "")
            except Exception:
                sync_receipt_path = ""
        workset_id = str(getattr(self.saguaro, "active_mission_id", "") or "")
        sync_receipt_id = str(
            sync_receipt.get("receipt_id")
            or sync_receipt.get("sync_id")
            or sync_receipt.get("id")
            or ""
        )
        thread_context = {
            "thread_id": str(trace_id or "trace_unknown"),
            "workset_id": workset_id,
            "sync_receipt_id": sync_receipt_id,
            "sync_receipt_path": sync_receipt_path,
        }
        self.current_compliance_context["mission_thread_id"] = thread_context[
            "thread_id"
        ]
        self.current_compliance_context["workset_id"] = workset_id
        self.current_compliance_context["sync_receipt_id"] = sync_receipt_id
        return thread_context

    def _refresh_runtime_control(self, task: str) -> Dict[str, Any]:
        status = self._safe_runtime_status()
        decision = self.runtime_control_policy.decide(status).to_dict()
        self.current_runtime_control = decision
        self.current_compliance_context["runtime_posture"] = decision.get("posture")
        self.current_compliance_context["runtime_control"] = dict(decision)
        if decision.get("degraded"):
            self._record_reality_event(
                "runtime_control_policy",
                phase="plan",
                status="degraded",
                metadata={
                    "task": task,
                    "posture": decision.get("posture"),
                    "reasons": list(decision.get("reasons") or []),
                    "verification_breadth": decision.get("verification_breadth"),
                    "planning_depth": decision.get("planning_depth"),
                },
            )
        return decision

    def _runtime_control_prompt_directive(self, control: Dict[str, Any]) -> str:
        posture = str(control.get("posture") or "unknown")
        planning_depth = str(control.get("planning_depth") or "standard")
        verification_breadth = str(control.get("verification_breadth") or "standard")
        reasons = ", ".join(str(item) for item in control.get("reasons") or [])
        return (
            f"- Runtime posture: {posture}\n"
            f"- Planning depth: {planning_depth}\n"
            f"- Verification breadth: {verification_breadth}\n"
            f"- Runtime reasons: {reasons or 'none'}"
        )

    def _mission_artifact_dir(self, trace_id: str) -> str:
        safe_trace = str(trace_id or "trace_unknown").replace("/", "_")
        return os.path.join(self.saguaro.root_dir, ".anvil", "missions", safe_trace)

    def _persist_compiled_mission_plan(
        self,
        *,
        action_plan: str,
        tool_calls: List[Dict[str, Any]],
        task: str,
    ) -> Dict[str, Any]:
        trace_id = self._resolve_execution_trace_id()
        artifact_dir = self._mission_artifact_dir(trace_id)
        path = os.path.join(artifact_dir, "compiled_plan.json")
        thread_context = self._current_mission_thread_context()
        compiled_plan = CompiledMissionPlan.from_action_plan(
            run_id=trace_id,
            trace_id=trace_id,
            task=task,
            action_plan=action_plan,
            tool_calls=tool_calls,
            thread_context=thread_context,
            runtime_control=self.current_runtime_control,
        )
        saved_path = compiled_plan.save(path)
        payload = compiled_plan.to_dict()
        payload["path"] = saved_path
        self.current_compliance_context["compiled_plan_path"] = saved_path
        try:
            self.black_box.event_store.record_checkpoint(
                run_id=str(trace_id),
                phase="plan",
                status="compiled",
                checkpoint_type="compiled_plan",
                metadata={
                    "step_count": int(payload.get("step_count", 0) or 0),
                    "planning_depth": self.current_runtime_control.get(
                        "planning_depth"
                    ),
                    "workset_id": thread_context.get("workset_id"),
                    "sync_receipt_id": thread_context.get("sync_receipt_id"),
                },
                artifacts=[saved_path],
            )
        except Exception as exc:
            logger.warning("Compiled mission plan checkpoint failed: %s", exc)
        self._record_reality_event(
            "compiled_mission_plan",
            phase="plan",
            status="ok",
            metadata={
                "step_count": int(payload.get("step_count", 0) or 0),
                "planning_depth": self.current_runtime_control.get("planning_depth"),
                "workset_id": thread_context.get("workset_id"),
                "sync_receipt_id": thread_context.get("sync_receipt_id"),
            },
            artifacts={"compiled_plan": saved_path},
        )
        return payload

    def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Route loop-initiated tools through BaseAgent for hooks and tracing."""
        return self.agent._execute_tool({"name": name, "arguments": arguments})

    def _derive_evidence_bundle_id(
        self,
        query: str = "",
        evidence: Optional[Dict[str, Any]] = None,
    ) -> str:
        trace_id = (
            self.current_compliance_context.get("trace_id")
            or getattr(self.agent, "current_mission_id", None)
            or self.current_task_id
            or f"task_{int(time.time())}"
        )
        fingerprint = hashlib.sha256()
        fingerprint.update(trace_id.encode("utf-8"))
        fingerprint.update((query or "").encode("utf-8"))

        if evidence:
            file_refs = sorted(
                set(evidence.get("codebase_files", []))
                | set(evidence.get("file_contents", {}).keys())
            )
            for file_ref in file_refs:
                fingerprint.update(file_ref.encode("utf-8"))

            for key in ("question_type", "request_type", "subagent_type"):
                value = evidence.get(key)
                if value:
                    fingerprint.update(str(value).encode("utf-8"))

            if evidence.get("subagent_analysis"):
                fingerprint.update(
                    str(len(str(evidence.get("subagent_analysis", "")))).encode("utf-8")
                )

            fingerprint.update(
                str(len(evidence.get("search_results", []))).encode("utf-8")
            )
            fingerprint.update(
                str(len(evidence.get("web_results", []))).encode("utf-8")
            )
            fingerprint.update(str(len(evidence.get("errors", []))).encode("utf-8"))

        return f"evidence::{trace_id}::{fingerprint.hexdigest()[:12]}"

    def _build_compliance_context(self) -> Dict[str, Any]:
        trace_id = (
            getattr(self.agent, "current_mission_id", None)
            or self.current_task_id
            or f"task_{int(time.time())}"
        )
        waiver_ids = self.current_compliance_context.get("waiver_ids") or []
        context = ComplianceContext.from_mapping(self.current_compliance_context)
        context.run_id = trace_id
        context.trace_id = trace_id
        context.evidence_bundle_id = self._derive_evidence_bundle_id()
        context.waiver_ids = waiver_ids
        context.red_team_required = bool(
            self.current_compliance_context.get("red_team_required")
        )
        return context.to_dict()

    def _infer_domains_from_files(self, changed_files: List[str]) -> List[str]:
        existing: List[str] = []
        for candidate in changed_files:
            if not candidate:
                continue
            path = (
                candidate
                if os.path.isabs(candidate)
                else os.path.join(self.saguaro.root_dir, candidate)
            )
            if os.path.exists(path):
                existing.append(path)
        domains = self.domain_detector.detect_domains(existing) if existing else set()
        return sorted(domains or {"universal"})

    def _infer_hot_paths(self, changed_files: List[str]) -> List[str]:
        hot_paths: List[str] = []
        for candidate in changed_files:
            lowered = candidate.lower()
            if any(
                token in lowered
                for token in ("benchmark", "kernel", "simd", "omp", "critical", "perf")
            ):
                hot_paths.append(candidate)
        return list(dict.fromkeys(hot_paths))

    def _infer_public_api_changes(self, changed_files: List[str]) -> List[str]:
        public_api: List[str] = []
        for candidate in changed_files:
            lowered = candidate.lower()
            if any(token in lowered for token in ("api", "routes", "service", "cli")):
                public_api.append(candidate)
        return list(dict.fromkeys(public_api))

    def _infer_dependency_changes(self, changed_files: List[str]) -> List[str]:
        dependency_files: List[str] = []
        for candidate in changed_files:
            if Path(candidate).name in {
                "requirements.txt",
                "requirements-dev.txt",
                "pyproject.toml",
                "poetry.lock",
                "uv.lock",
            }:
                dependency_files.append(candidate)
        return list(dict.fromkeys(dependency_files))

    def _build_change_manifest(
        self,
        compliance: Dict[str, Any],
        evidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        evidence = evidence or {}
        changed_files = list(
            dict.fromkeys(
                list(evidence.get("codebase_files", []) or [])
                + list(compliance.get("changed_files", []) or [])
            )
        )
        return {
            "run_id": compliance.get("run_id") or compliance.get("trace_id"),
            "changed_files": changed_files,
            "aal": compliance.get("aal", self.runtime_aal),
            "domains": compliance.get("domains", []),
            "hot_paths": compliance.get("hot_paths", []),
            "public_api_changes": compliance.get("public_api_changes", []),
            "dependency_changes": compliance.get("dependency_changes", []),
            "required_rule_ids": compliance.get("required_rule_ids", []),
            "required_runtime_gates": compliance.get("required_runtime_gates", []),
            "trace_id": compliance.get("trace_id"),
            "evidence_bundle_id": compliance.get("evidence_bundle_id"),
            "runtime_posture": compliance.get("runtime_posture"),
            "compiled_plan_path": compliance.get("compiled_plan_path"),
            "mission_thread_id": compliance.get("mission_thread_id"),
            "workset_id": compliance.get("workset_id"),
            "sync_receipt_id": compliance.get("sync_receipt_id"),
        }

    def _compliance_artifact_dir(self, compliance: Dict[str, Any]) -> str:
        run_id = compliance.get("run_id") or compliance.get("trace_id") or "run"
        return os.path.join(self.saguaro.root_dir, ".anvil", "compliance", run_id)

    def _persist_compliance_artifacts(
        self,
        compliance: Dict[str, Any],
        evidence: Dict[str, Any],
        runtime_gate_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        artifact_dir = self._compliance_artifact_dir(compliance)
        os.makedirs(artifact_dir, exist_ok=True)
        manifest = self._build_change_manifest(compliance, evidence)
        paths = {
            "change_manifest": os.path.join(artifact_dir, "change_manifest.json"),
            "traceability": os.path.join(artifact_dir, "traceability.json"),
            "evidence_bundle": os.path.join(artifact_dir, "evidence_bundle.json"),
            "runtime_gates": os.path.join(artifact_dir, "runtime_gates.json"),
        }
        evidence_artifacts = evidence.get("artifacts", {}) or {}
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        default_test_refs = list(evidence.get("test_refs", []) or [])
        if not default_test_refs:
            default_test_refs = list(evidence.get("test_artifacts", []) or [])
        if not default_test_refs:
            default_test_refs = ["tests/unspecified"]
        default_verification_refs = list(evidence.get("verification_refs", []) or [])
        if not default_verification_refs:
            default_verification_refs = [
                "saguaro verify . --engines native,ruff,semantic,aes"
            ]
        traceability_record = {
            "trace_id": compliance.get("trace_id"),
            "run_id": compliance.get("run_id"),
            "requirement_id": (
                (compliance.get("required_rule_ids", []) or [None])[0]
                or f"AES-RUN::{manifest['run_id']}"
            ),
            "design_ref": str(evidence.get("design_ref") or "AES_plan.md"),
            "code_refs": manifest["changed_files"]
            or list(evidence.get("code_refs", []) or []),
            "test_refs": default_test_refs,
            "verification_refs": default_verification_refs,
            "aal": manifest["aal"],
            "owner": str(getattr(self.agent, "name", None) or "anvil"),
            "timestamp": timestamp,
            "changed_files": manifest["changed_files"],
            "required_rule_ids": compliance.get("required_rule_ids", []),
            "evidence_bundle_id": compliance.get("evidence_bundle_id"),
        }
        custom_traceability = evidence_artifacts.get("traceability")
        if isinstance(custom_traceability, dict):
            traceability_record.update(custom_traceability)

        evidence_bundle = {
            "bundle_id": (
                compliance.get("evidence_bundle_id") or f"bundle::{manifest['run_id']}"
            ),
            "change_id": manifest["run_id"],
            "trace_id": compliance.get("trace_id"),
            "changed_files": manifest["changed_files"],
            "aal": manifest["aal"],
            "chronicle_snapshot": str(
                evidence_artifacts.get("chronicle_snapshot")
                or evidence_artifacts.get("chronicle", {}).get("snapshot_id")
                or "not-required"
            ),
            "chronicle_diff": str(
                evidence_artifacts.get("chronicle_diff")
                or evidence_artifacts.get("chronicle", {}).get("diff_id")
                or "not-required"
            ),
            "verification_report_path": str(
                evidence_artifacts.get("verification_report_path")
                or "runtime://verification"
            ),
            "red_team_report_path": str(
                evidence_artifacts.get("red_team_report_path") or "runtime://red-team"
            ),
            "review_signoffs": list(evidence.get("review_signoffs", []) or []),
            "waivers": list(compliance.get("waiver_ids", []) or []),
            "author": str(getattr(self.agent, "name", None) or "anvil"),
            "compiled_plan_path": str(compliance.get("compiled_plan_path") or ""),
            "mission_thread_id": str(compliance.get("mission_thread_id") or ""),
            "workset_id": str(compliance.get("workset_id") or ""),
            "sync_receipt_id": str(compliance.get("sync_receipt_id") or ""),
        }
        custom_bundle = evidence_artifacts.get("evidence_bundle")
        if isinstance(custom_bundle, dict):
            evidence_bundle.update(custom_bundle)

        runtime_gate_payload = {
            "run_id": manifest["run_id"],
            "trace_id": compliance.get("trace_id"),
            "required_runtime_gates": [],
            "missing_artifacts": [],
            "results": [],
        }
        if isinstance(runtime_gate_summary, dict):
            runtime_gate_payload.update(runtime_gate_summary)
        Path(paths["change_manifest"]).write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        Path(paths["traceability"]).write_text(
            json.dumps(traceability_record, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        Path(paths["evidence_bundle"]).write_text(
            json.dumps(evidence_bundle, indent=2, sort_keys=True), encoding="utf-8"
        )
        Path(paths["runtime_gates"]).write_text(
            json.dumps(runtime_gate_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self._record_reality_event(
            "compliance_artifacts_persisted",
            phase="observe",
            status="ok",
            files=manifest["changed_files"],
            metadata={
                "aal": manifest["aal"],
                "required_rule_ids": compliance.get("required_rule_ids", []),
            },
            artifacts=paths,
        )
        return paths

    def _refresh_compliance_context(
        self,
        query: str = "",
        evidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        trace_id = (
            self.current_compliance_context.get("trace_id")
            or getattr(self.agent, "current_mission_id", None)
            or self.current_task_id
            or f"task_{int(time.time())}"
        )
        waiver_ids = self.current_compliance_context.get("waiver_ids") or []
        if not waiver_ids and self.current_compliance_context.get("waiver_id"):
            waiver_ids = [str(self.current_compliance_context.get("waiver_id")).strip()]
            waiver_ids = [item for item in waiver_ids if item]
        changed_files = list(
            dict.fromkeys(
                list(self.current_compliance_context.get("changed_files", []) or [])
                + list((evidence or {}).get("codebase_files", []) or [])
            )
        )
        domains = self._infer_domains_from_files(changed_files)
        hot_paths = self._infer_hot_paths(changed_files)
        public_api_changes = self._infer_public_api_changes(changed_files)
        dependency_changes = self._infer_dependency_changes(changed_files)
        updated = {
            "run_id": trace_id,
            "aal": self.runtime_aal,
            "domains": domains,
            "changed_files": changed_files,
            "hot_paths": hot_paths,
            "public_api_changes": public_api_changes,
            "dependency_changes": dependency_changes,
            "required_rule_ids": list(
                self.current_compliance_context.get("required_rule_ids", []) or []
            ),
            "required_runtime_gates": list(
                self.current_compliance_context.get("required_runtime_gates", []) or []
            ),
            "trace_id": trace_id,
            "evidence_bundle_id": self._derive_evidence_bundle_id(query, evidence),
            "red_team_required": bool(
                self.current_compliance_context.get("red_team_required")
            ),
            "waiver_ids": waiver_ids,
            "waiver_id": self.current_compliance_context.get("waiver_id")
            or (waiver_ids[0] if waiver_ids else None),
        }
        self.current_compliance_context = updated
        self.agent.current_mission_id = updated["trace_id"]
        self.agent.current_evidence_bundle_id = updated["evidence_bundle_id"]
        self.agent.current_waiver_id = updated["waiver_id"]
        self.agent.current_waiver_ids = updated["waiver_ids"]
        self.agent.current_red_team_required = updated["red_team_required"]
        self.thinking_system.set_compliance_context(**updated)
        return dict(updated)

    def _get_sentinel_verifier(self) -> SentinelVerifier:
        if self._sentinel_verifier is None:
            self._sentinel_verifier = SentinelVerifier(
                repo_path=getattr(self.saguaro, "root_dir", "."),
                engines=["native", "ruff", "semantic", "aes"],
            )
        return self._sentinel_verifier

    def _classify_runtime_aal(self, query: str, evidence: Dict[str, Any]) -> str:
        files = []
        for candidate in evidence.get("codebase_files", []) or []:
            if not candidate:
                continue
            path = (
                candidate
                if os.path.isabs(candidate)
                else os.path.join(self.saguaro.root_dir, candidate)
            )
            if os.path.exists(path):
                files.append(path)
        if files:
            return self.aal_classifier.classify_changeset(files)
        return self.aal_classifier.classify_from_description(query or "")

    def _build_deterministic_synthesis_bundle(
        self,
        evidence: Dict[str, Any],
        execution_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        artifacts = dict(evidence.get("artifacts") or {})
        synthesis = execution_result.get("deterministic_synthesis")
        if isinstance(synthesis, dict):
            artifacts.update(
                {
                    "spec_path": synthesis.get("spec_path") or artifacts.get("spec_path"),
                    "replay_tape_path": synthesis.get("replay_tape_path")
                    or artifacts.get("replay_tape_path")
                    or synthesis.get("replay_path"),
                    "proof_capsule_path": synthesis.get("proof_capsule_path")
                    or artifacts.get("proof_capsule_path")
                    or synthesis.get("proof_path"),
                    "benchmark_summary": synthesis.get("benchmark_summary")
                    or artifacts.get("benchmark_summary"),
                    "roadmap_validation": synthesis.get("roadmap_validation")
                    or artifacts.get("roadmap_validation"),
                }
            )
        return {
            "spec_path": artifacts.get("spec_path"),
            "replay_tape_path": artifacts.get("replay_tape_path")
            or artifacts.get("replay_path"),
            "proof_capsule_path": artifacts.get("proof_capsule_path")
            or artifacts.get("proof_path"),
            "verification_passed": bool(
                (execution_result.get("verification") or {}).get("passed") is True
            ),
            "benchmark_summary": artifacts.get("benchmark_summary")
            or execution_result.get("benchmark_summary"),
            "roadmap_validation": artifacts.get("roadmap_validation")
            or execution_result.get("roadmap_validation"),
        }

    def _evaluate_deterministic_synthesis_promotion(
        self,
        evidence: Dict[str, Any],
        execution_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not execution_result.get("deterministic_synthesis"):
            return {"allowed": True, "errors": [], "bundle": {}}
        validator = CampaignRoadmapValidator()
        bundle = self._build_deterministic_synthesis_bundle(evidence, execution_result)
        return validator.summarize_synthesis_promotion(bundle)

    def _post_evidence_governance_checkpoint(
        self, user_input: str, evidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        aal = self._classify_runtime_aal(user_input, evidence)
        self.runtime_aal = aal
        self.current_compliance_context["red_team_required"] = aal in {"AAL-0", "AAL-1"}
        compliance = self._refresh_compliance_context(
            query=user_input, evidence=evidence
        )
        obligations = self.obligation_engine.evaluate(
            ComplianceContext.from_mapping(compliance)
        )
        compliance["required_rule_ids"] = obligations.required_rule_ids
        compliance["required_runtime_gates"] = obligations.required_runtime_gates
        self.current_compliance_context = dict(compliance)
        evidence["aal"] = aal
        evidence["compliance"] = compliance

        quality = evidence.get("subagent_quality") or {}
        if quality and not quality.get("accepted", True):
            result = {
                "allowed": False,
                "reason": "subagent_quality_gate_failed",
                "aal": aal,
            }
            self._record_reality_event(
                "governance_checkpoint",
                phase="evidence",
                status="blocked",
                metadata=result,
            )
            return result

        if aal in {"AAL-0", "AAL-1"} and not compliance.get("evidence_bundle_id"):
            result = {
                "allowed": False,
                "reason": "missing_evidence_bundle_id",
                "aal": aal,
            }
            self._record_reality_event(
                "governance_checkpoint",
                phase="evidence",
                status="blocked",
                metadata=result,
            )
            return result

        result = {"allowed": True, "aal": aal}
        self._record_reality_event(
            "governance_checkpoint",
            phase="evidence",
            status="passed",
            metadata=result,
        )
        return result

    def _pre_action_tool_checkpoint(
        self, tool_calls: List[Dict[str, Any]], task: str
    ) -> Dict[str, Any]:
        aal = self.runtime_aal
        waiver_ids = self.current_compliance_context.get("waiver_ids") or []
        sentinel_gate = self._pre_action_aes_rule_gate(tool_calls, aal)
        if not sentinel_gate.get("allowed", True):
            sentinel_gate["task"] = task
            return sentinel_gate

        for tool_call in tool_calls:
            tool_name = str(tool_call.get("tool") or "")
            args = tool_call.get("args", {}) or {}
            action_repr = f"{tool_name} {json.dumps(args, default=str)}"
            runtime_status = {}
            brain = getattr(self, "brain", None)
            runtime_fn = getattr(brain, "runtime_status", None)
            if callable(runtime_fn):
                try:
                    runtime_status = dict(runtime_fn())
                except Exception:
                    runtime_status = {}
            governance = self.governance_engine.check_action(
                action=action_repr,
                aal=aal,
                action_type=(
                    "code_modification"
                    if tool_name
                    in {
                        "write_file",
                        "edit_file",
                        "write_files",
                        "delete_file",
                        "move_file",
                        "run_command",
                    }
                    else "analysis"
                ),
                waiver_ids=waiver_ids,
                qsg_runtime_status=runtime_status,
            )
            if not governance.allowed:
                return {
                    "allowed": False,
                    "reason": governance.reason,
                    "tool": tool_name,
                    "aal": aal,
                }

            escalation = self.action_escalation_engine.evaluate(
                tool_name=tool_name,
                arguments=args,
                aal=aal,
                review_signoff_token=self.current_compliance_context.get(
                    "review_signoff_token"
                ),
                rollback_plan_artifact=self.current_compliance_context.get(
                    "rollback_plan_artifact"
                ),
                waiver_ids=waiver_ids,
            )
            if not escalation.allowed:
                return {
                    "allowed": False,
                    "reason": escalation.reason,
                    "tool": tool_name,
                    "aal": aal,
                }
        return {"allowed": True, "aal": aal, "task": task}

    def _collect_checkpoint_target_files(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[str]:
        candidates: List[str] = []
        for tool_call in tool_calls:
            args = tool_call.get("args") or {}
            for key in ("path", "file_path", "dst"):
                value = args.get(key)
                if isinstance(value, str) and value.strip():
                    candidates.append(value.strip())
            files_payload = args.get("files")
            if isinstance(files_payload, dict):
                for path in files_payload.keys():
                    if isinstance(path, str) and path.strip():
                        candidates.append(path.strip())
        return list(dict.fromkeys(candidates))

    def _run_sentinel_for_targets(
        self,
        target_files: List[str],
        aal: str,
        require_trace: bool,
        require_evidence: bool,
        require_valid_waivers: bool,
        change_manifest_path: Optional[str] = None,
        compliance_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        verifier = self._get_sentinel_verifier()
        targets = [
            path for path in target_files if isinstance(path, str) and path.strip()
        ]
        if not targets:
            targets = ["."]

        aggregated: List[Dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()

        for path_arg in dict.fromkeys(targets):
            scoped = verifier.verify_all(
                path_arg=path_arg,
                aal=aal,
                require_trace=require_trace,
                require_evidence=require_evidence,
                require_valid_waivers=require_valid_waivers,
                change_manifest_path=change_manifest_path,
                compliance_context=compliance_context,
            )
            for violation in scoped:
                if not isinstance(violation, dict):
                    violation = {
                        "rule_id": "SENTINEL-MALFORMED-VIOLATION",
                        "file": path_arg,
                        "line": 0,
                        "severity": "P0",
                        "closure_level": "blocking",
                        "message": f"Malformed Sentinel violation payload: {violation!r}",
                    }
                key = (
                    str(violation.get("rule_id", "")),
                    str(violation.get("file", "")),
                    str(violation.get("line", "")),
                    str(violation.get("message", "")),
                )
                if key in seen:
                    continue
                seen.add(key)
                aggregated.append(violation)

        return aggregated

    def _pre_action_aes_rule_gate(
        self, tool_calls: List[Dict[str, Any]], aal: str
    ) -> Dict[str, Any]:
        registry = getattr(self, "aes_rule_registry", None)
        rules = getattr(registry, "rules", []) if registry is not None else []
        if not rules:
            return {
                "allowed": False,
                "reason": "aes_rules_registry_empty",
                "aal": aal,
            }
        target_files = self._collect_checkpoint_target_files(tool_calls)
        strict = str(aal or "").upper() in {"AAL-0", "AAL-1"}
        try:
            violations = self._run_sentinel_for_targets(
                target_files=target_files,
                aal=aal,
                require_trace=strict,
                require_evidence=strict,
                require_valid_waivers=strict,
                compliance_context=self.current_compliance_context,
            )
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            return {
                "allowed": False,
                "reason": f"pre_action_sentinel_error: {exc}",
                "aal": aal,
            }

        policy_decision = self.policy_manager.runtime_decision(violations, aal=aal)
        if policy_decision.get("should_fail"):
            return {
                "allowed": False,
                "reason": "pre_action_sentinel_policy_blocking_outcome",
                "aal": aal,
                "policy": policy_decision,
                "violations": violations[:20],
            }
        return {"allowed": True, "aal": aal}

    def _pre_finalize_governance_checkpoint(
        self,
        user_input: str,
        evidence: Dict[str, Any],
        execution_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        synthesis_gate = self._evaluate_deterministic_synthesis_promotion(
            evidence,
            execution_result,
        )
        if not synthesis_gate.get("allowed", True):
            result = {
                "allowed": False,
                "reason": "deterministic_synthesis_promotion_gate_failed",
                "errors": list(synthesis_gate.get("errors") or []),
                "bundle": dict(synthesis_gate.get("bundle") or {}),
            }
            self._record_reality_event(
                "governance_checkpoint",
                phase="synthesize",
                status="blocked",
                metadata=result,
            )
            return result

        aal = self.runtime_aal or self._classify_runtime_aal(user_input, evidence)
        if aal not in {"AAL-0", "AAL-1"}:
            result = {"allowed": True, "aal": aal}
            self._record_reality_event(
                "governance_checkpoint",
                phase="synthesize",
                status="passed",
                metadata=result,
            )
            return result

        verification = execution_result.get("verification") or {}
        if verification.get("passed") is not True:
            result = {
                "allowed": False,
                "reason": "verification_not_passed",
                "aal": aal,
                "verification": verification,
            }
            self._record_reality_event(
                "governance_checkpoint",
                phase="synthesize",
                status="blocked",
                metadata=result,
            )
            return result

        compliance = self.current_compliance_context or {}
        if not compliance.get("trace_id") or not compliance.get("evidence_bundle_id"):
            result = {
                "allowed": False,
                "reason": "missing_trace_or_evidence_bundle",
                "aal": aal,
            }
            self._record_reality_event(
                "governance_checkpoint",
                phase="synthesize",
                status="blocked",
                metadata=result,
            )
            return result

        artifacts = evidence.get("artifacts") or {}
        red_team_required = bool(compliance.get("red_team_required", False))
        red_team = self.red_team_protocol.validate(artifacts, aal, red_team_required)
        if red_team.required and not red_team.passed:
            result = {
                "allowed": False,
                "reason": "red_team_artifacts_incomplete",
                "aal": aal,
                "missing_artifacts": red_team.missing_artifacts,
                "unresolved_critical": red_team.unresolved_critical_findings,
            }
            self._record_reality_event(
                "governance_checkpoint",
                phase="synthesize",
                status="blocked",
                metadata=result,
            )
            return result

        review_gate = self.review_gate.evaluate_from_evidence(
            aal=aal,
            evidence=evidence,
            author=getattr(self.agent, "name", None),
            irreversible_action=bool(
                (execution_result.get("files_written") or [])
                or (execution_result.get("files_edited") or [])
                or (execution_result.get("commands_run") or [])
            ),
        )
        if not review_gate.passed:
            result = {
                "allowed": False,
                "reason": "review_gate_failed",
                "aal": aal,
                "review_gate_reasons": review_gate.reasons,
            }
            self._record_reality_event(
                "governance_checkpoint",
                phase="synthesize",
                status="blocked",
                metadata=result,
            )
            return result

        target_files = (execution_result.get("files_written") or []) + (
            execution_result.get("files_edited") or []
        )
        artifact_paths = self._persist_compliance_artifacts(
            compliance,
            evidence,
            runtime_gate_summary={
                "required_runtime_gates": compliance.get("required_runtime_gates", []),
                "results": [],
            },
        )
        try:
            violations = self._run_sentinel_for_targets(
                target_files=target_files,
                aal=aal,
                require_trace=True,
                require_evidence=True,
                require_valid_waivers=True,
                change_manifest_path=artifact_paths["change_manifest"],
                compliance_context=compliance,
            )
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            result = {
                "allowed": False,
                "reason": f"sentinel_verify_error: {exc}",
                "aal": aal,
            }
            self._record_reality_event(
                "governance_checkpoint",
                phase="synthesize",
                status="blocked",
                metadata=result,
            )
            return result

        policy_decision = self.policy_manager.runtime_decision(violations, aal=aal)
        if policy_decision.get("should_fail"):
            result = {
                "allowed": False,
                "reason": "sentinel_policy_blocking_outcome",
                "aal": aal,
                "policy": policy_decision,
                "violations": violations[:20],
            }
            self._record_reality_event(
                "governance_checkpoint",
                phase="synthesize",
                status="blocked",
                metadata=result,
            )
            return result

        obligations = self.obligation_engine.evaluate(
            ComplianceContext.from_mapping(compliance)
        )
        runtime_summary = self.runtime_gate_runner.evaluate(
            ComplianceContext.from_mapping(compliance),
            obligations.required_runtime_gates,
            thresholds=obligations.thresholds,
        )
        artifact_paths = self._persist_compliance_artifacts(
            compliance,
            evidence,
            runtime_gate_summary={
                "required_runtime_gates": obligations.required_runtime_gates,
                "results": [
                    {
                        "gate_id": result.gate_id,
                        "passed": result.passed,
                        "required_artifacts": result.required_artifacts,
                        "missing_artifacts": result.missing_artifacts,
                        "message": result.message,
                    }
                    for result in runtime_summary.results
                ],
                "missing_artifacts": runtime_summary.missing_artifacts,
            },
        )
        if not runtime_summary.passed:
            result = {
                "allowed": False,
                "reason": "runtime_gate_incomplete",
                "aal": aal,
                "missing_artifacts": runtime_summary.missing_artifacts,
                "runtime_gate_artifacts": artifact_paths,
            }
            self._record_reality_event(
                "governance_checkpoint",
                phase="synthesize",
                status="blocked",
                metadata=result,
                artifacts=artifact_paths,
            )
            return result

        result = {"allowed": True, "aal": aal}
        self._record_reality_event(
            "governance_checkpoint",
            phase="synthesize",
            status="passed",
            metadata=result,
            artifacts=artifact_paths,
        )
        return result

    def _on_tool_message_compressed(
        self, tc_id: int, message: Dict[str, Any], summary: str
    ) -> None:
        self.compression_memory.remember_summary(
            session_id=self.compression_session_id,
            tc_id=tc_id,
            summary=summary,
            tool_name=message.get("tool_name"),
            tool_args=message.get("tool_args", {}),
        )

    def _rehydrate_compressed_history(self) -> None:
        updates = self.compression_memory.get_updates_payload(
            self.compression_session_id
        )
        if not updates:
            return
        messages = self.history.get_messages()
        result = apply_context_updates(
            messages, updates, on_compressed=self._on_tool_message_compressed
        )
        if result.get("applied"):
            self.history.save()

    def _apply_context_updates_from_results(
        self, tool_results: List[Dict[str, Any]]
    ) -> None:
        updates: List[Dict[str, str]] = []
        for result in tool_results:
            result_updates = result.get("context_updates") or []
            if result_updates:
                updates.extend(result_updates)

        if not updates:
            return

        messages = self.history.get_messages()
        outcome = apply_context_updates(
            messages, updates, on_compressed=self._on_tool_message_compressed
        )
        if outcome.get("applied"):
            self.history.save()

    def _record_tool_result(
        self, tool_name: str, tool_args: Dict[str, Any], result_text: str
    ) -> None:
        tc_id = self.tool_call_counter
        self.tool_call_counter += 1
        message = label_tool_result(
            f"Tool '{tool_name}' Result: {result_text}",
            tc_id=tc_id,
        )
        self.history.add_message(
            "tool",
            message,
            tc_id=tc_id,
            tool_name=tool_name,
            tool_args=tool_args or {},
            is_compressed=False,
        )

    def _auto_compress_dead_context(self) -> None:
        messages = self.history.get_messages()
        compressed = auto_compress_dead_end_reads(
            messages,
            min_age_messages=6,
            on_compressed=self._on_tool_message_compressed,
        )
        if compressed:
            self.history.save()

    def _context_pressure_guidance(self, task: str = "") -> str:
        self._auto_compress_dead_context()
        stats = self.context_token_manager.get_fill_percentage(
            self.history.get_messages()
        )
        guidance = (
            "CONTEXT: "
            f"{stats['used_tokens']} / {stats['max_tokens']} tokens "
            f"({stats['fill_percentage']:.1f}% used). "
            "Compress old tool results via _context_updates on every tool call. "
            "After 70%, compress aggressively."
        )

        low_relevance = find_low_relevance_tc_ids(
            task=task or "",
            messages=self.history.get_messages(),
            semantic_engine=self.semantic_engine,
            threshold=0.10,
        )
        if low_relevance:
            guidance += (
                "\nLOW_RELEVANCE_HINT: These tool results are likely stale and should "
                f"be compressed first: {', '.join(low_relevance[:8])}."
            )
        return guidance

    def _initialize_dashboard(
        self, dashboard: Optional[LiveProgressDashboard]
    ) -> LiveProgressDashboard:
        renderer = getattr(self.agent, "renderer", None)
        if dashboard is None:
            dashboard = LiveProgressDashboard(console=self.console, renderer=renderer)
            dashboard.add_phase("Understand", status="in_progress")
            dashboard.add_phase("Plan")
            dashboard.add_phase("Execute")
            dashboard.add_phase("Observe")
            dashboard.add_phase("Synthesize")
            return dashboard

        dashboard.update_phase("Understand", status="in_progress")
        return dashboard

    def _build_adaptive_complexity_snapshot(
        self,
        user_input: str,
        complexity_profile: Optional[ComplexityProfile],
        referenced_files: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        default_depth = int(
            max(1, getattr(complexity_profile, "coconut_steps", 2) or 2)
        )
        default_slots = self._complexity_subagent_slots(complexity_profile)
        snapshot: Dict[str, Any] = {
            "coconut_depth": default_depth,
            "subagent_slots": default_slots,
            "max_steps_per_agent": max(4, default_depth * 3),
            "subagent_coconut": bool(
                getattr(complexity_profile, "subagent_coconut", default_depth >= 4)
            ),
        }
        analyzer = getattr(self, "task_complexity_analyzer", None)
        if analyzer is None:
            return snapshot
        try:
            adaptive_profile = analyzer.analyze(
                user_input,
                candidate_files=referenced_files,
            )
            snapshot["complexity_score"] = float(
                getattr(adaptive_profile, "complexity_score", 0.0)
            )
            snapshot["coconut_depth"] = int(
                max(
                    1,
                    getattr(
                        adaptive_profile, "coconut_depth", snapshot["coconut_depth"]
                    ),
                )
            )
            snapshot["subagent_slots"] = int(
                max(
                    1,
                    min(
                        self._max_configured_subagent_slots(),
                        getattr(
                            adaptive_profile,
                            "subagent_count",
                            snapshot["subagent_slots"],
                        ),
                    ),
                )
            )
            snapshot["max_steps_per_agent"] = int(
                max(
                    1,
                    getattr(
                        adaptive_profile,
                        "max_steps_per_agent",
                        snapshot["max_steps_per_agent"],
                    ),
                )
            )
            snapshot["subagent_coconut"] = bool(
                getattr(
                    adaptive_profile, "subagent_coconut", snapshot["subagent_coconut"]
                )
            )
        except Exception:
            logger.debug("Failed to apply adaptive profile snapshot.", exc_info=True)
        return snapshot

    def _resolve_coconut_depth(
        self,
        complexity_profile: Optional[Any],
        adaptive_complexity: Optional[Dict[str, Any]] = None,
    ) -> int:
        adaptive = adaptive_complexity or {}
        try:
            adaptive_depth = int(adaptive.get("coconut_depth", 0) or 0)
        except Exception:
            adaptive_depth = 0
        if adaptive_depth > 0:
            return max(1, adaptive_depth)
        try:
            profile_depth = int(getattr(complexity_profile, "coconut_depth", 0) or 0)
        except Exception:
            profile_depth = 0
        if profile_depth > 0:
            return max(1, profile_depth)
        try:
            steps = int(getattr(complexity_profile, "coconut_steps", 0) or 0)
        except Exception:
            steps = 0
        if steps > 0:
            return max(1, steps)
        return 2

    def _resolve_subagent_slot_count(self, complexity_profile: Optional[Any]) -> int:
        adaptive = self.current_adaptive_complexity or {}
        try:
            adaptive_slots = int(adaptive.get("subagent_slots", 0) or 0)
        except Exception:
            adaptive_slots = 0
        if adaptive_slots > 0:
            return max(1, min(self._max_configured_subagent_slots(), adaptive_slots))
        return self._complexity_subagent_slots(complexity_profile)

    def _initialize_run_phase_context(
        self, user_input: str
    ) -> tuple[ComplexityProfile, Dict[str, Any]]:
        complexity_profile = self.complexity_scorer.score_request(user_input)
        self.current_complexity_profile = complexity_profile
        adaptive_complexity = self._build_adaptive_complexity_snapshot(
            user_input=user_input,
            complexity_profile=complexity_profile,
        )
        self.current_adaptive_complexity = adaptive_complexity
        self.context_budget_allocator.recommend_total_budget(
            complexity_profile.recommended_context_budget
        )
        context_max = self.context_budget_allocator.get_budget(
            "master"
        ) + self.context_budget_allocator.get_budget("system")
        self.context_token_manager.max_tokens = context_max
        self.context_token_manager.available_tokens = max(1000, context_max - 2000)
        self._last_classification_meta = {}
        risk_budget = self._build_risk_budget(user_input, complexity_profile)
        phase_context = {
            "enhanced_mode": self.enhanced_mode,
            "current_task_id": self.current_task_id,
            "files_read": self.files_read,
            "files_edited": self.files_edited,
            "complexity_profile": complexity_profile,
            "adaptive_complexity": adaptive_complexity,
            "risk_budget": risk_budget,
        }
        self._record_reality_event(
            "run_initialized",
            phase="initialize",
            status="ok",
            metadata={
                "complexity_score": complexity_profile.score,
                "request_type": "pending",
                "risk_budget": risk_budget,
                "observed": True,
                "inferred": False,
                "unobserved_count": 0,
            },
        )
        return complexity_profile, phase_context

    def _run_mission_request(
        self,
        request_type: str,
        user_input: str,
        dashboard: LiveProgressDashboard,
    ) -> Optional[str]:
        if request_type != "mission":
            return None

        logger.info("Mission-level request detected - routing to AgentOrchestrator")
        dashboard.update_phase(
            "Plan", status="in_progress", message="Mission planning..."
        )
        try:
            from core.orchestrator.scheduler import AgentOrchestrator

            mission_orchestrator = AgentOrchestrator(
                root_dir=".",
                brain=self.brain,
                semantic_engine=self.semantic_engine,
                console=self.console,
            )
            dashboard.add_agent(
                "Plan",
                "PlannerAgent",
                status="running",
                message="Decomposing objective...",
            )
            dashboard.add_agent("Execute", "WorkerAgent", status="pending")
            dashboard.add_agent("Observe", "VerifierAgent", status="pending")

            result = mission_orchestrator.run(user_input)

            dashboard.update_agent("PlannerAgent", status="completed", progress=1.0)
            dashboard.update_agent("WorkerAgent", status="completed", progress=1.0)
            dashboard.update_agent("VerifierAgent", status="completed", progress=1.0)
            dashboard.update_phase("Execute", status="completed", progress=1.0)
            dashboard.update_phase("Observe", status="completed", progress=1.0)
            dashboard.update_phase("Synthesize", status="completed", progress=1.0)
            return result
        except (
            ImportError,
            AttributeError,
            RuntimeError,
            ValueError,
            TypeError,
            OSError,
        ) as exc:
            logger.error(f"AgentOrchestrator failed: {exc}")
            dashboard.update_phase("Plan", status="failed", message=str(exc))
            return None

    def _run_evidence_phase_with_checkpoint(
        self,
        user_input: str,
        phase_context: Dict[str, Any],
        request_type: str,
        complexity_profile: ComplexityProfile,
        dashboard: LiveProgressDashboard,
    ) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        evidence_result = self.evidence_phase.execute(
            user_input, phase_context, dashboard
        )
        evidence_result["request_type"] = request_type
        evidence_result["complexity_profile"] = complexity_profile
        adaptive_complexity = phase_context.get("adaptive_complexity") or {}
        evidence_result["adaptive_complexity"] = adaptive_complexity
        evidence_result.setdefault("complexity_score", complexity_profile.score)
        evidence_result.setdefault("coconut_paths", complexity_profile.coconut_paths)
        evidence_result.setdefault("coconut_steps", complexity_profile.coconut_steps)
        evidence_result.setdefault(
            "coconut_depth",
            self._resolve_coconut_depth(complexity_profile, adaptive_complexity),
        )
        evidence_result.setdefault(
            "subagent_slots",
            int(
                max(
                    1,
                    int(adaptive_complexity.get("subagent_slots", 1) or 1),
                )
            ),
        )
        evidence_checkpoint = self._post_evidence_governance_checkpoint(
            user_input, evidence_result
        )
        if not evidence_checkpoint.get("allowed", True):
            reason = str(evidence_checkpoint.get("reason", "governance_block"))
            logger.warning("Post-evidence governance checkpoint blocked: %s", reason)
            dashboard.update_phase("Plan", status="failed", message=reason)
            blocked = (
                f"AES GOVERNANCE BLOCK ({evidence_checkpoint.get('aal', 'AAL-3')}): "
                f"{reason}"
            )
            return None, blocked

        phase_context.update({"evidence": evidence_result})
        return evidence_result, None

    def _run_coconut_phase(
        self,
        user_input: str,
        evidence_result: Dict[str, Any],
        phase_context: Dict[str, Any],
        dashboard: LiveProgressDashboard,
    ) -> None:
        complexity_profile = phase_context.get("complexity_profile")
        should_run_phase_coconut = (
            self.thinking_system
            and self.thinking_system.coconut_enabled
            and complexity_profile is not None
            and complexity_profile.coconut_frequency in {"per_phase", "per_step"}
        )
        if not should_run_phase_coconut:
            return

        logger.info("Starting Latent Reasoning Phase (COCONUT)")
        dashboard.add_agent("Synthesize", "COCONUT", status="running")
        try:
            coconut = self.thinking_system.coconut
            if coconut is not None:
                coconut.config["num_paths"] = complexity_profile.coconut_paths
                resolved_depth = self._resolve_coconut_depth(
                    complexity_profile,
                    phase_context.get("adaptive_complexity"),
                )
                coconut.config["steps"] = resolved_depth
                evidence_result["coconut_depth"] = resolved_depth
            context_str = (
                user_input + "\n" + str(evidence_result.get("agent_summaries", ""))
            )
            context_emb = self._get_embedding_vector(context_str)
            if context_emb is not None:
                latent_vectors = self._collect_reinjected_latent_vectors(
                    evidence_result,
                    target_dim=int(context_emb.size),
                )
                if latent_vectors:
                    prior = np.mean(
                        np.asarray(latent_vectors, dtype=np.float32), axis=0
                    )
                    prior_norm = float(np.linalg.norm(prior))
                    if prior_norm > 1e-8:
                        prior = prior / prior_norm
                    context_emb = (
                        0.8 * np.asarray(context_emb, dtype=np.float32) + 0.2 * prior
                    )
                refined_emb = self.thinking_system.deep_think(np.array(context_emb))
                if (
                    refined_emb is not None
                    and self.thinking_system.coconut.amplitudes is not None
                ):
                    evidence_result["coconut_refined"] = np.asarray(
                        refined_emb, dtype=np.float32
                    )
                    evidence_result["coconut_amplitudes"] = [
                        float(x)
                        for x in np.asarray(
                            self.thinking_system.coconut.amplitudes,
                            dtype=np.float32,
                        ).flatten()
                    ]
                    reranked = self._rerank_evidence_with_coconut(
                        evidence_result,
                        context_embedding=np.asarray(refined_emb).reshape(-1),
                    )
                    if reranked:
                        evidence_result["coconut_reranked_files"] = reranked

                    from rich.panel import Panel
                    from rich.table import Table

                    amplitudes = self.thinking_system.coconut.amplitudes
                    table = Table(
                        title="COCONUT Multipath Amplitudes",
                        show_header=True,
                        header_style="bold magenta",
                    )
                    table.add_column("Path", style="cyan")
                    table.add_column("Amplitude", style="green")
                    for i, amp in enumerate(amplitudes):
                        table.add_row(f"Path {i+1}", f"{amp:.4f}")
                    self.console.print(
                        Panel(
                            table,
                            title="[bold yellow]Unified Loop COCONUT[/bold yellow]",
                            border_style="yellow",
                        )
                    )
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.warning(f"Unified loop COCONUT failed: {exc}")
        dashboard.update_agent("COCONUT", status="completed", progress=1.0)

    def _run_synthesis_phase_with_checkpoint(
        self,
        user_input: str,
        evidence_result: Dict[str, Any],
        execution_result: Dict[str, Any],
        phase_context: Dict[str, Any],
        dashboard: LiveProgressDashboard,
    ) -> tuple[Optional[str], Optional[str]]:
        logger.info("Starting Synthesis Phase")
        finalize_checkpoint = self._pre_finalize_governance_checkpoint(
            user_input,
            evidence_result,
            execution_result,
        )
        if not finalize_checkpoint.get("allowed", True):
            reason = str(finalize_checkpoint.get("reason", "finalization_blocked"))
            logger.warning("Pre-finalize governance checkpoint blocked: %s", reason)
            dashboard.update_phase("Synthesize", status="failed", message=reason)
            blocked = (
                f"AES FINALIZATION BLOCK ({finalize_checkpoint.get('aal', 'AAL-3')}): "
                f"{reason}"
            )
            return None, blocked

        finalize_automation_error = self._run_high_assurance_finalize_automation(
            user_input=user_input,
            execution_result=execution_result,
        )
        if finalize_automation_error:
            logger.warning(
                "Finalize high-assurance automation blocked completion: %s",
                finalize_automation_error,
            )
            dashboard.update_phase(
                "Synthesize",
                status="failed",
                message=finalize_automation_error,
            )
            blocked = (
                f"AES FINALIZATION BLOCK ({finalize_checkpoint.get('aal', 'AAL-3')}): "
                f"{finalize_automation_error}"
            )
            return None, blocked

        response = self.synthesis_phase.execute(user_input, phase_context, dashboard)
        return response, None

    def _finalize_run_performance(
        self,
        perf_monitor: Any,
        request_type: str,
        tools_used: List[str],
        *,
        success: bool,
    ) -> Any:
        try:
            return perf_monitor.end_tracking(
                success=success,
                metadata={
                    "request_type": request_type,
                    "tools_used": tools_used,
                    "files_read": len(self.files_read),
                    "files_edited": len(self.files_edited),
                },
            )
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            self.console.print(
                f"[dim yellow]⚠ Performance tracking failed: {exc}[/dim yellow]"
            )
            return None

    def run(
        self, user_input: str, dashboard: Optional[LiveProgressDashboard] = None
    ) -> str:
        """
        Main entry point for unified chat loop.

        Flow:
        0. FAST-PATH: Simple questions bypass heavy machinery
        1. Classify request type
        2. Check memory for similar past tasks (if enhanced)
        3. Gather context with enhancements
        4. Execute appropriate phase
        5. Record task memory (if enhanced)
        6. Track performance metrics (Phase 4.1)

        Args:
            user_input: The user's query or request
            dashboard: Optional external dashboard for progress updates. If not provided,
                      creates an internal one.
        """
        self._refresh_runtime_bindings()

        # FAST-PATH: Simple questions bypass the full pipeline
        # (Disabled by user request - too slow)
        # if self._is_simple_question(user_input):
        #     return self._handle_simple_question(user_input)

        self.current_task_id = f"task_{int(time.time())}"
        self.current_compliance_context = self._build_compliance_context()
        self._refresh_compliance_context(query=user_input)
        logger.info(f"Starting UnifiedChatLoop for task: {self.current_task_id}")
        logger.debug(f"User Input: {user_input}")
        start_time = time.time()
        tools_used = []
        request_type = "unknown"
        response = ""
        stop_reason = "interrupted"
        run_success = False

        # Phase 4.1: Start performance tracking
        perf_monitor = get_performance_monitor()
        perf_monitor.start_tracking(
            component_name="unified_chat_loop", component_type="loop"
        )

        run_id = self.current_compliance_context.get("trace_id") or self.current_task_id
        self.black_box.start_run(
            run_id=run_id,
            task_id=self.current_task_id,
            task=user_input,
            metadata={
                "agent_name": getattr(self.agent, "name", "anvil"),
                "enhanced_mode": self.enhanced_mode,
            },
        )
        self.message_bus.set_trace_context(run_id=run_id, task_id=self.current_task_id)

        self.history.add_message("user", user_input)
        dashboard = self._initialize_dashboard(dashboard)
        perf_snapshot = None
        try:
            with dashboard.live():
                complexity_profile, phase_context = self._initialize_run_phase_context(
                    user_input
                )

                self._record_phase_transition("understand", "started")
                understanding_result = self.understanding_phase.execute(
                    user_input, phase_context, dashboard
                )
                request_type = understanding_result["request_type"]
                phase_context.update(understanding_result)
                self._record_phase_transition(
                    "understand",
                    "completed",
                    metadata={
                        "request_type": request_type,
                        **self._last_classification_meta,
                    },
                )

                mission_result = self._run_mission_request(
                    request_type, user_input, dashboard
                )
                if mission_result is not None:
                    stop_reason = "mission_handoff"
                    response = mission_result
                    return mission_result

                self._record_phase_transition("evidence", "started")
                evidence_result, evidence_block = (
                    self._run_evidence_phase_with_checkpoint(
                        user_input,
                        phase_context,
                        request_type,
                        complexity_profile,
                        dashboard,
                    )
                )
                if evidence_block:
                    stop_reason = "evidence_blocked"
                    self._record_phase_transition(
                        "evidence",
                        "blocked",
                        metadata={"reason": evidence_block},
                    )
                    response = evidence_block
                    return evidence_block
                self._record_phase_transition("evidence", "completed")

                self._record_phase_transition("execute", "started")
                execution_result = self.execution_phase.execute(
                    user_input, phase_context, dashboard
                )
                phase_context.update({"execution_result": execution_result})
                self._record_phase_transition(
                    "execute",
                    "completed",
                    metadata={
                        "files_written": len(
                            execution_result.get("files_written") or []
                        ),
                        "files_edited": len(execution_result.get("files_edited") or []),
                        "commands_run": len(execution_result.get("commands_run") or []),
                    },
                )

                self._record_phase_transition("observe", "started")
                self._run_coconut_phase(
                    user_input, evidence_result or {}, phase_context, dashboard
                )
                self._record_phase_transition("observe", "completed")

                self._record_phase_transition("synthesize", "started")
                response, synthesis_block = self._run_synthesis_phase_with_checkpoint(
                    user_input,
                    evidence_result,
                    execution_result,
                    phase_context,
                    dashboard,
                )
                if synthesis_block:
                    stop_reason = "synthesis_blocked"
                    self._record_phase_transition(
                        "synthesize",
                        "blocked",
                        metadata={"reason": synthesis_block},
                    )
                    response = synthesis_block
                    return synthesis_block

                logger.info("UnifiedChatLoop execution completed")
                dashboard.update_phase("Synthesize", status="completed", progress=1.0)
                self._record_phase_transition("synthesize", "completed")

                if self.enhanced_mode and request_type != "conversational":
                    self._record_task_memory(
                        user_input, request_type, start_time, tools_used
                    )

                dashboard.update_display()
                stop_reason = "completed"
                run_success = True
        finally:
            perf_snapshot = self._finalize_run_performance(
                perf_monitor, request_type, tools_used, success=run_success
            )
            self.black_box.record_performance_snapshot(perf_snapshot)
            qsg_runtime_status = {}
            runtime_fn = getattr(getattr(self, "brain", None), "runtime_status", None)
            if callable(runtime_fn):
                try:
                    qsg_runtime_status = dict(runtime_fn())
                except Exception:
                    qsg_runtime_status = {}
            self.black_box.finalize(
                stop_reason=stop_reason,
                success=run_success,
                message_bus=self.message_bus,
                extra_metadata={
                    "request_type": request_type,
                    "response_length": len(response or ""),
                    "qsg_runtime_status": qsg_runtime_status,
                },
            )

        return response or ""

    def _classify_request(self, user_input: str) -> str:
        """
        Hybrid intent classifier:
        1) fast keyword heuristics
        2) semantic prototype similarity on ambiguous inputs
        3) constrained LLM tie-break for low-confidence cases
        """
        intent, meta = self._classify_request_hybrid(user_input)
        self._last_classification_meta = meta
        return intent

    def _classify_request_hybrid(self, user_input: str) -> tuple[str, Dict[str, Any]]:
        heuristic_scores = self._heuristic_intent_scores(user_input)
        best_heuristic_intent = max(heuristic_scores, key=heuristic_scores.get)
        best_heuristic_score = float(heuristic_scores[best_heuristic_intent])
        total = float(sum(heuristic_scores.values())) or 1.0
        heuristic_confidence = best_heuristic_score / total

        if best_heuristic_intent == "mission" and best_heuristic_score >= 1.0:
            return "mission", {
                "source": "heuristic",
                "confidence": 0.98,
                "scores": heuristic_scores,
            }

        if best_heuristic_score >= 1.0 and heuristic_confidence >= 0.66:
            return best_heuristic_intent, {
                "source": "heuristic",
                "confidence": heuristic_confidence,
                "scores": heuristic_scores,
            }

        semantic_intent, semantic_confidence = self._semantic_intent_similarity(
            user_input
        )
        if semantic_intent and semantic_confidence >= 0.36:
            if (
                best_heuristic_score < 1.0
                or semantic_confidence >= 0.70
                or semantic_confidence >= max(0.48, heuristic_confidence + 0.05)
            ):
                return semantic_intent, {
                    "source": "semantic",
                    "confidence": semantic_confidence,
                    "scores": heuristic_scores,
                    "semantic_intent": semantic_intent,
                }

        if max(heuristic_confidence, semantic_confidence) < 0.52:
            llm_intent = self._llm_classify_intent(user_input)
            if llm_intent:
                return llm_intent, {
                    "source": "llm",
                    "confidence": 0.60,
                    "scores": heuristic_scores,
                    "semantic_intent": semantic_intent,
                }

        return best_heuristic_intent, {
            "source": "heuristic_fallback",
            "confidence": heuristic_confidence,
            "scores": heuristic_scores,
            "semantic_intent": semantic_intent,
            "semantic_confidence": semantic_confidence,
        }

    def _heuristic_intent_scores(self, user_input: str) -> Dict[str, float]:
        text = (user_input or "").lower()
        scores: Dict[str, float] = {
            "question": 0.0,
            "creation": 0.0,
            "modification": 0.0,
            "deletion": 0.0,
            "investigation": 0.0,
            "mission": 0.0,
            "conversational": 0.1,
        }
        keyword_map = {
            "question": [
                "how does",
                "how do",
                "explain",
                "describe",
                "what is",
                "where is",
                "why",
                "when",
                "who",
                "which",
            ],
            "creation": [
                "create",
                "add",
                "write",
                "implement",
                "build",
                "generate",
                "new file",
                "scaffold",
            ],
            "modification": [
                "edit",
                "modify",
                "change",
                "update",
                "fix",
                "refactor",
                "rename",
                "move",
                "improve",
                "optimize",
            ],
            "deletion": ["delete", "remove", "drop", "clear", "clean up"],
            "investigation": [
                "search for",
                "find",
                "investigate",
                "analyze",
                "explore",
                "look for",
                "locate",
                "trace",
                "debug",
            ],
            "mission": [
                "execute the roadmap",
                "implement the plan",
                "complete all phases",
                "run the mission",
                "build the entire",
                "deploy the",
                "migrate the",
                "refactor the entire",
                "redesign the",
                "overhaul",
                "modernize the",
                "end-to-end",
                "full implementation",
            ],
        }
        for intent, keywords in keyword_map.items():
            for kw in keywords:
                if kw in text:
                    scores[intent] += 1.0

        if "@" in user_input and scores["modification"] > 0:
            scores["investigation"] += 1.2
        if len(user_input.split()) <= 4 and scores["question"] == 0:
            scores["conversational"] += 0.4
        return scores

    def _semantic_intent_similarity(
        self, user_input: str
    ) -> tuple[Optional[str], float]:
        query_vec = self._get_embedding_vector(user_input)
        if query_vec is None:
            return None, 0.0

        if not hasattr(self, "_intent_prototype_embeddings"):
            self._intent_prototype_embeddings = {}

        best_intent = None
        best_score = -1.0
        for intent, proto_text in INTENT_PROTOTYPES.items():
            cached = self._intent_prototype_embeddings.get(intent)
            if cached is None:
                cached = self._get_embedding_vector(proto_text)
                self._intent_prototype_embeddings[intent] = cached
            if cached is None:
                continue
            similarity = self._cosine_similarity(query_vec, cached)
            if similarity > best_score:
                best_intent = intent
                best_score = similarity
        return best_intent, max(0.0, float(best_score))

    def _llm_classify_intent(self, user_input: str) -> Optional[str]:
        allowed = ", ".join(INTENT_PROTOTYPES.keys())
        messages = [
            {
                "role": "system",
                "content": (
                    "Classify the user request intent. "
                    f"Respond with exactly one word from: {allowed}."
                ),
            },
            {"role": "user", "content": user_input},
        ]
        try:
            output = self.brain.chat(messages, max_tokens=8, temperature=0.0)
            label = re.findall(r"[a-z_]+", str(output).lower())
            if not label:
                return None
            candidate = label[0]
            if candidate in INTENT_PROTOTYPES:
                return candidate
        except (AttributeError, RuntimeError, ValueError, TypeError):
            return None
        return None

    def _classify_question_type(self, query: str) -> str:
        """
        Classify question type to determine which subagent should handle it.

        Returns:
            - "research": Web + codebase research needed
            - "architecture": Understanding how components work together
            - "investigation": Finding specific implementations
            - "simple": Basic factual questions
        """
        query_lower = query.lower()

        # Research indicators (external knowledge needed)
        research_keywords = [
            "best practice",
            "best way to",
            "how to",
            "tutorial",
            "documentation",
            "latest",
            "current",
            "recommended approach",
            "what are the options",
            "pros and cons",
        ]
        if any(kw in query_lower for kw in research_keywords):
            return "research"

        # Architecture indicators (understanding system design)
        architecture_keywords = [
            "how does",
            "how do",
            "explain",
            "architecture",
            "structure",
            "design",
            "flow",
            "works",
            "bridge",
            "interface",
            "system",
        ]
        if any(kw in query_lower for kw in architecture_keywords):
            return "architecture"

        # Investigation indicators (finding code)
        investigation_keywords = [
            "find",
            "locate",
            "where is",
            "which file",
            "search for",
            "show me",
            "implementations of",
        ]
        if any(kw in query_lower for kw in investigation_keywords):
            return "investigation"

        # Default to simple for short factual questions
        return "simple"

    def _check_memory(self, task: str):
        """Check task memory for similar past tasks."""
        self.console.print("[dim]Checking task memory...[/dim]")

        similar = self.memory_manager.recall_similar(task, limit=3)

        if similar:
            self.console.print(f"[cyan]Found {len(similar)} similar past tasks[/cyan]")

            # Get best suggestion
            best = similar[0]
            if best.success:
                self.console.print(
                    f"  [green]✓[/green] Similar task succeeded with {best.iterations} iterations"
                )

    def _start_question_phase_ui(
        self, dashboard: Optional[LiveProgressDashboard]
    ) -> None:
        if dashboard:
            dashboard.update_phase(
                "Plan",
                status="in_progress",
                message="Analyzing question...",
            )
            return

        self.console.print("\n")
        self.console.print(
            Panel(
                "[bold yellow]Phase 1: Evidence Gathering[/bold yellow]\n"
                "[dim]Analyzing question, delegating to specialists...[/dim]",
                border_style="yellow",
                box=box.HEAVY,
                padding=(0, 2),
            )
        )
        self.console.print("")

    def _collect_semantic_question_evidence(
        self,
        user_input: str,
        evidence: Dict[str, Any],
        dashboard: Optional[LiveProgressDashboard],
        add_dashboard_agent: bool = False,
    ) -> None:
        if add_dashboard_agent and dashboard:
            dashboard.add_agent("Plan", "SemanticSearch", status="running")

        if not self.enhanced_mode:
            raise RuntimeError(
                "SAGUARO_STRICT_MODE_REQUIRES_ENHANCED_LOOP: basic evidence mode is disabled."
            )
        self._gather_evidence_enhanced(user_input, evidence, dashboard=dashboard)

        if add_dashboard_agent and dashboard:
            dashboard.update_agent("SemanticSearch", status="completed", progress=1.0)

    def _gather_question_evidence(
        self,
        question_type: str,
        user_input: str,
        evidence: Dict[str, Any],
        dashboard: Optional[LiveProgressDashboard],
    ) -> None:
        if question_type in {"research", "architecture", "investigation"}:
            delegated = self._delegate_to_question_specialist(
                query=user_input,
                question_type=question_type,
                dashboard=dashboard,
            )
            evidence.update(delegated)
            return

        self.console.print("[cyan]Phase 1: Gathering evidence (inline)...[/cyan]")
        self._gather_evidence_enhanced(user_input, evidence, dashboard=dashboard)

    def _route_question_specialist(
        self, query: str, question_type: str
    ) -> tuple[Any, list[str], str]:
        domains = sorted(self.domain_detector.detect_from_description(query))
        aal = self.aal_classifier.classify_from_description(query)
        role_hint = ""
        if question_type == "research":
            role_hint = "ResearchLibrarianSubagent"
        elif question_type == "architecture":
            role_hint = "SoftwareArchitectureSubagent"
        elif question_type == "investigation":
            role_hint = "CodebaseCartographerSubagent"

        routing = route_specialist(
            registry=self.specialist_registry,
            objective=query,
            requested_role=role_hint,
            aal=aal,
            domains=domains,
            question_type=question_type,
            repo_roles=["analysis_local"],
        )
        return routing, domains, aal

    def _delegate_to_question_specialist(
        self,
        *,
        query: str,
        question_type: str,
        dashboard: Optional[LiveProgressDashboard],
    ) -> Dict[str, Any]:
        routing, domains, aal = self._route_question_specialist(query, question_type)
        role = str(routing.primary_role or "")
        agent_label = role or "SpecialistSubagent"

        if dashboard:
            dashboard.add_agent("Plan", agent_label, status="running")
        self.console.print(
            f"  [cyan]→ Routed specialist:[/cyan] [bold]{agent_label}[/bold]"
        )

        # Reuse mature specialist loops where available.
        if role in {
            "ResearchSubagent",
            "ResearchLibrarianSubagent",
            "ResearchCrawlerSubagent",
        }:
            delegated = self._delegate_to_research_subagent(query)
        elif role in {"RepoAnalysisSubagent", "RepoCampaignAnalysisSubagent"}:
            delegated = self._delegate_to_repo_analysis_subagent(query)
        else:
            delegated = self._delegate_to_routed_specialist_subagent(
                query=query,
                question_type=question_type,
                routed_role=role,
                routed_domains=domains,
                aal=aal,
                routing_reasons=list(routing.reasons),
            )

        delegated.setdefault("routing_reasons", list(routing.reasons))
        delegated.setdefault("subagent_role", role)
        delegated.setdefault("domains", domains)
        delegated.setdefault("aal", aal)

        if dashboard:
            dashboard.update_agent(agent_label, status="completed", progress=1.0)
        return delegated

    def _delegate_to_routed_specialist_subagent(
        self,
        *,
        query: str,
        question_type: str,
        routed_role: str,
        routed_domains: List[str],
        aal: str,
        routing_reasons: List[str],
    ) -> Dict[str, Any]:
        try:
            complexity_profile = (
                self.current_complexity_profile
                or self.complexity_scorer.score_request(query)
            )
            if not self.current_adaptive_complexity:
                self.current_adaptive_complexity = (
                    self._build_adaptive_complexity_snapshot(
                        user_input=query,
                        complexity_profile=complexity_profile,
                    )
                )
            subagent_slots = self._resolve_subagent_slot_count(complexity_profile)
            coconut_depth = self._resolve_coconut_depth(
                complexity_profile,
                self.current_adaptive_complexity,
            )
            context_seed = self._get_embedding_vector(query)
            prompt_key = self.specialist_registry.prompt_key_for_role(routed_role)
            specialist = build_specialist_subagent(
                role=routed_role,
                task=query,
                parent_name=self.agent.name,
                brain=self.brain,
                console=self.console,
                parent_agent=self.agent,
                message_bus=self.message_bus,
                ownership_registry=getattr(self.agent, "ownership_registry", None),
                complexity_profile=complexity_profile,
                context_budget=self.context_budget_allocator.get_subagent_budget(
                    subagent_slots
                ),
                coconut_context_vector=context_seed,
                coconut_depth=coconut_depth,
                prompt_profile="sovereign_build",
                specialist_prompt_key=prompt_key,
                sovereign_build_policy_enabled=True,
                prompt_injection=(
                    f"Routed from UnifiedChatLoop question handling.\n"
                    f"Question type: {question_type}\n"
                    f"Routing reasons: {', '.join(routing_reasons) if routing_reasons else 'none'}"
                ),
            )
            specialist.tool_strategy = self._select_tool_strategy(
                None,
                (
                    question_type
                    if question_type in {"research", "architecture", "investigation"}
                    else "investigation"
                ),
                {"request_type": "question", "direct_file": False},
            )["name"]
            self._seed_subagent_guidance(specialist.name, query)
            result = specialist.run(
                query,
                prompt_profile="sovereign_build",
                specialist_prompt_key=prompt_key,
            )
            if isinstance(result, dict):
                response_text = result.get("summary", result.get("full_response", ""))
            else:
                response_text = str(result)
            response_text = clean_response(response_text)

            files_read = (
                result.get("files_read", []) if isinstance(result, dict) else []
            )
            latent_payload = self._extract_subagent_latent_payload(
                result=result,
                source=routed_role or "specialist",
            )
            evidence = {
                "subagent_type": routed_role or "specialist",
                "subagent_analysis": response_text,
                "codebase_files": files_read,
                "file_contents": {},
                "web_results": [],
                "errors": [],
                "domains": list(routed_domains),
                "aal": aal,
                "routing_reasons": list(routing_reasons),
                "compliance": dict(self.current_compliance_context),
                "complexity_profile": complexity_profile,
                "adaptive_complexity": dict(self.current_adaptive_complexity),
            }
            self._append_subagent_latent_payload(evidence, latent_payload)
            return self._apply_subagent_quality_gate(evidence, query)
        except (AttributeError, RuntimeError, ValueError, TypeError, OSError) as e:
            self.console.print(f"[red]{routed_role} error: {e}[/red]")
            return {
                "codebase_files": [],
                "file_contents": {},
                "web_results": [],
                "search_results": [],
                "errors": [f"{routed_role} failed: {e}"],
                "domains": list(routed_domains),
                "aal": aal,
                "routing_reasons": list(routing_reasons),
                "subagent_role": routed_role,
            }

    def _start_question_synthesis_ui(
        self, dashboard: Optional[LiveProgressDashboard]
    ) -> None:
        if dashboard:
            dashboard.update_phase("Plan", status="completed")
            dashboard.update_phase(
                "Synthesize",
                status="in_progress",
                message="Synthesizing final answer...",
            )
            return

        self.console.print("[cyan]Phase 2: Synthesizing answer...[/cyan]")

    def _handle_question(
        self, user_input: str, dashboard: Optional[LiveProgressDashboard] = None
    ) -> str:
        """
        Handle question/investigation requests with subagent delegation.

        Enhanced flow:
        1. Classify question type
        2. Delegate to appropriate subagent based on type
        3. Subagent explores and returns structured findings
        4. Master synthesizes with COCONUT thinking
        """
        self._start_question_phase_ui(dashboard)

        self._ensure_saguaro_ready()

        # Classify to determine delegation strategy
        question_type = self._classify_question_type(user_input)
        logger.info(f"Question type: {question_type}")

        self.console.print(
            f"  [cyan]→ Question type:[/cyan] [bold]{question_type}[/bold]"
        )

        evidence = {
            "codebase_files": [],
            "file_contents": {},
            "web_results": [],
            "search_results": [],
            "errors": [],
            "question_type": question_type,
        }

        self._gather_question_evidence(question_type, user_input, evidence, dashboard)

        if evidence.get("subagent_type"):
            evidence = self._apply_subagent_quality_gate(evidence, user_input)

        evidence["compliance"] = self._refresh_compliance_context(
            query=user_input,
            evidence=evidence,
        )

        self._start_question_synthesis_ui(dashboard)

        response = self._synthesize_answer(user_input, evidence, dashboard=dashboard)

        return response

    def _delegate_to_research_subagent(self, query: str) -> Dict[str, Any]:
        """Delegate to ResearchSubagent for comprehensive research."""
        from core.agents.researcher import ResearchSubagent

        try:
            complexity_profile = (
                self.current_complexity_profile
                or self.complexity_scorer.score_request(query)
            )
            if not self.current_adaptive_complexity:
                self.current_adaptive_complexity = (
                    self._build_adaptive_complexity_snapshot(
                        user_input=query,
                        complexity_profile=complexity_profile,
                    )
                )
            subagent_slots = self._resolve_subagent_slot_count(complexity_profile)
            coconut_depth = self._resolve_coconut_depth(
                complexity_profile,
                self.current_adaptive_complexity,
            )
            context_seed = self._get_embedding_vector(query)
            researcher = ResearchSubagent(
                task=query,
                parent_name=self.agent.name,
                brain=self.brain,
                console=self.console,
                message_bus=self.message_bus,
                ownership_registry=getattr(self.agent, "ownership_registry", None),
                complexity_profile=complexity_profile,
                context_budget=self.context_budget_allocator.get_subagent_budget(
                    subagent_slots
                ),
                coconut_context_vector=context_seed,
                coconut_depth=coconut_depth,
            )
            researcher.tool_strategy = self._select_tool_strategy(
                None,
                "investigation",
                {"request_type": "question", "direct_file": False},
            )["name"]
            self._seed_subagent_guidance(researcher.name, query)

            # Run research using web tools plus strict Saguaro code discovery.
            result = researcher.run(query)

            # Extract findings - use summary (cleaned) instead of raw response
            if isinstance(result, dict):
                # Prefer summary (has thinking/tools removed), fallback to full_response
                response_text = result.get("summary", result.get("full_response", ""))
            else:
                response_text = str(result)

            # Additional cleaning to remove any model artifacts
            response_text = clean_response(response_text)

            # Validate the response is useful (not just garbage or empty)
            if not response_text or len(response_text.strip()) < 50:
                self.console.print(
                    f"[yellow]⚠ ResearchSubagent returned insufficient analysis ({len(response_text)} chars)[/yellow]"
                )
                # Fall through to return the result anyway, but mark as potentially incomplete

            files_read = (
                result.get("files_read", []) if isinstance(result, dict) else []
            )
            latent_payload = self._extract_subagent_latent_payload(
                result=result,
                source="research",
            )
            evidence = {
                "subagent_type": "research",
                "subagent_analysis": response_text,
                "codebase_files": files_read,
                "file_contents": {},  # Subagent summary contains findings
                "web_results": [],
                "errors": [],
                "compliance": dict(self.current_compliance_context),
                "complexity_profile": complexity_profile,
                "adaptive_complexity": dict(self.current_adaptive_complexity),
            }
            self._append_subagent_latent_payload(evidence, latent_payload)
            return self._apply_subagent_quality_gate(evidence, query)
        except (AttributeError, RuntimeError, ValueError, TypeError, OSError) as e:
            self.console.print(f"[red]ResearchSubagent error: {e}[/red]")
            return {
                "codebase_files": [],
                "file_contents": {},
                "web_results": [],
                "search_results": [],
                "errors": [f"ResearchSubagent failed: {e}"],
            }

    def _delegate_to_repo_analysis_subagent(self, query: str) -> Dict[str, Any]:
        """Delegate to RepoAnalysisSubagent for architecture understanding."""
        from core.agents.repo_analyzer import RepoAnalysisSubagent

        try:
            direct_file_path = self._extract_direct_file_path(query)
            complexity_profile = (
                self.current_complexity_profile
                or self.complexity_scorer.score_request(
                    query,
                    referenced_files=[direct_file_path] if direct_file_path else None,
                )
            )
            if not self.current_adaptive_complexity:
                self.current_adaptive_complexity = (
                    self._build_adaptive_complexity_snapshot(
                        user_input=query,
                        complexity_profile=complexity_profile,
                        referenced_files=(
                            [direct_file_path] if direct_file_path else None
                        ),
                    )
                )
            subagent_slots = self._resolve_subagent_slot_count(complexity_profile)
            coconut_depth = self._resolve_coconut_depth(
                complexity_profile,
                self.current_adaptive_complexity,
            )
            context_seed = self._get_embedding_vector(query)
            analyzer = RepoAnalysisSubagent(
                task=query,
                parent_name=self.agent.name,
                brain=self.brain,
                console=self.console,
                message_bus=self.message_bus,
                ownership_registry=getattr(self.agent, "ownership_registry", None),
                complexity_profile=complexity_profile,
                context_budget=self.context_budget_allocator.get_subagent_budget(
                    subagent_slots
                ),
                coconut_context_vector=context_seed,
                coconut_depth=coconut_depth,
            )
            analyzer.tool_strategy = self._select_tool_strategy(
                None,
                "architecture",
                {"request_type": "question", "direct_file": bool(direct_file_path)},
            )["name"]
            self._seed_subagent_guidance(analyzer.name, query)

            # Run analysis using strict Saguaro discovery and inspection tools.
            result = analyzer.run()

            # Extract findings - use summary (cleaned) instead of raw response
            if isinstance(result, dict):
                # Prefer summary (has thinking/tools removed), fallback to full_response
                response_text = result.get("summary", result.get("full_response", ""))
            else:
                response_text = str(result)

            # Additional cleaning to remove any model artifacts
            response_text = clean_response(response_text)

            # Validate the response is useful (not just garbage or empty)
            if not response_text or len(response_text.strip()) < 50:
                self.console.print(
                    f"[yellow]⚠ RepoAnalysisSubagent returned insufficient analysis ({len(response_text)} chars)[/yellow]"
                )
                # Fall through to return the result anyway, but mark as potentially incomplete

            unique_files = self._collect_repo_analysis_candidate_files(
                direct_file_path, result, response_text
            )
            response_text, unique_files, file_analyst = (
                self._augment_repo_analysis_with_file_subagent(
                    query=query,
                    unique_files=unique_files,
                    response_text=response_text,
                )
            )
            file_contents, read_errors = self._load_repo_analysis_file_contents(
                unique_files
            )
            latent_payload = self._extract_subagent_latent_payload(
                result=result,
                source="architecture",
            )

            evidence = {
                "subagent_type": "architecture",
                "subagent_analysis": response_text,
                "primary_file": direct_file_path
                or (unique_files[0] if unique_files else None),
                "codebase_files": unique_files,
                "file_contents": file_contents,
                "web_results": [],
                "errors": read_errors,
                "compliance": dict(self.current_compliance_context),
                "complexity_profile": complexity_profile,
                "adaptive_complexity": dict(self.current_adaptive_complexity),
            }
            if file_analyst:
                evidence["file_analyst"] = file_analyst
            self._append_subagent_latent_payload(evidence, latent_payload)
            return self._apply_subagent_quality_gate(evidence, query)
        except (AttributeError, RuntimeError, ValueError, TypeError, OSError) as e:
            self.console.print(f"[red]RepoAnalysisSubagent error: {e}[/red]")
            return {
                "codebase_files": [],
                "file_contents": {},
                "web_results": [],
                "search_results": [],
                "errors": [f"RepoAnalysisSubagent failed: {e}"],
            }

    def _collect_repo_analysis_candidate_files(
        self,
        direct_file_path: Optional[str],
        result: Any,
        response_text: str,
    ) -> List[str]:
        candidate_files: List[str] = []
        if direct_file_path:
            candidate_files.append(direct_file_path)

        if isinstance(result, dict):
            for file_path in result.get("files_read", []) or []:
                if isinstance(file_path, str) and file_path:
                    candidate_files.append(file_path)

        for file_path in re.findall(
            r"[\w./-]+\.(?:py|cc|cpp|h|js|ts|md)", response_text
        ):
            candidate_files.append(file_path)

        return list(dict.fromkeys(candidate_files))

    def _load_repo_analysis_file_contents(
        self, unique_files: List[str]
    ) -> tuple[Dict[str, str], List[str]]:
        file_contents: Dict[str, str] = {}
        read_errors: List[str] = []
        for file_path in unique_files[:8]:
            content = self._execute_tool("read_file", {"path": file_path})
            if content and not str(content).startswith("Error"):
                file_contents[file_path] = str(content)
            else:
                read_errors.append(f"{file_path}: {content}")
        return file_contents, read_errors

    def _augment_repo_analysis_with_file_subagent(
        self, *, query: str, unique_files: List[str], response_text: str
    ) -> tuple[str, List[str], Dict[str, Any]]:
        """Use FileAnalysisSubagent when repo analysis returns a large file set."""
        if len(unique_files) <= 10:
            return response_text, unique_files, {}

        try:
            from core.subagents.file_analyst import FileAnalysisSubagent

            analyst = FileAnalysisSubagent(
                parent_agent=self.agent,
                files=unique_files[:30],
                query=query,
                quiet=True,
            )
            analysis = analyst.analyze() or {}
            analyst_summary = clean_response(str(analysis.get("summary") or "").strip())
            key_files = [
                str(path).strip()
                for path in (analysis.get("key_files") or [])
                if str(path).strip()
            ]
            merged_files = list(dict.fromkeys(key_files + unique_files))

            enriched_response = response_text
            if analyst_summary:
                enriched_response = (
                    f"{response_text}\n\n## File Analyst Synthesis\n{analyst_summary}"
                )

            payload = {
                "summary": analyst_summary,
                "key_files": key_files,
                "token_usage": int(analysis.get("token_usage") or 0),
                "entity_count": len(analysis.get("entities_found") or {}),
            }
            return enriched_response, merged_files, payload
        except (
            ImportError,
            AttributeError,
            RuntimeError,
            ValueError,
            TypeError,
        ) as exc:
            logger.debug(
                "FileAnalysisSubagent augmentation failed; continuing without it.",
                exc_info=True,
            )
            return response_text, unique_files, {"error": str(exc)}

    def _coerce_latent_vector(
        self, value: Any, target_dim: Optional[int] = None
    ) -> Optional[np.ndarray]:
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
        except Exception:
            logger.debug("Failed to coerce latent vector payload.", exc_info=True)
            return None
        if arr.size == 0 or not np.isfinite(arr).all():
            return None
        if target_dim is not None and target_dim > 0:
            if arr.size > target_dim:
                arr = arr[:target_dim]
            elif arr.size < target_dim:
                arr = np.pad(arr, (0, target_dim - arr.size), mode="constant")
        norm = float(np.linalg.norm(arr))
        if norm > 1e-8:
            arr = arr / norm
        return arr

    def _extract_subagent_latent_payload(
        self, result: Any, source: str
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(result, dict):
            return None
        latent_block = (
            result.get("latent", {}) if isinstance(result.get("latent"), dict) else {}
        )
        raw_state = latent_block.get("state", result.get("latent_state"))
        state_arr = self._coerce_latent_vector(raw_state)
        if state_arr is None:
            return None

        tool_signals = latent_block.get(
            "tool_signals", result.get("latent_tool_signals", [])
        )
        if not isinstance(tool_signals, list):
            tool_signals = []
        try:
            reinjections = int(
                latent_block.get("reinjections", result.get("latent_reinjections", 0))
                or 0
            )
        except Exception:
            reinjections = 0
        try:
            depth_used = int(latent_block.get("depth_used", 0) or 0)
        except Exception:
            depth_used = 0

        return {
            "source": source,
            "state": [float(v) for v in state_arr],
            "state_dim": int(state_arr.size),
            "reinjections": max(0, reinjections),
            "tool_signals": tool_signals,
            "depth_used": max(0, depth_used),
            "seeded_from_master": bool(latent_block.get("seeded_from_master", False)),
        }

    def _append_subagent_latent_payload(
        self, evidence: Dict[str, Any], payload: Optional[Dict[str, Any]]
    ) -> None:
        if not payload:
            return
        signals = evidence.setdefault("subagent_latent_signals", [])
        signals.append(payload)

        vectors: List[np.ndarray] = []
        for signal in signals:
            vec = self._coerce_latent_vector(signal.get("state"))
            if vec is not None:
                vectors.append(vec)
        if not vectors:
            return

        target_dim = min(vec.shape[0] for vec in vectors)
        if target_dim <= 0:
            return
        aligned = [self._coerce_latent_vector(vec, target_dim) for vec in vectors]
        aligned = [vec for vec in aligned if vec is not None]
        if not aligned:
            return
        merged = np.mean(np.asarray(aligned, dtype=np.float32), axis=0)
        norm = float(np.linalg.norm(merged))
        if norm > 1e-8:
            merged = merged / norm
        merged_list = [float(v) for v in merged.reshape(-1)]
        evidence["subagent_latent_merged"] = merged_list
        evidence["subagent_latent_reinjections"] = int(
            sum(int(signal.get("reinjections", 0) or 0) for signal in signals)
        )
        evidence["subagent_latent_signal_count"] = int(len(signals))
        if hasattr(self.agent, "latent_memory"):
            try:
                self.agent.latent_memory.add_thought(
                    "subagent_latent",
                    f"{payload.get('source', 'subagent')} returned latent state (dim={payload.get('state_dim', 0)}).",
                    vector=merged_list,
                )
            except Exception:
                logger.debug(
                    "Failed to persist subagent latent payload.", exc_info=True
                )

    def _ensure_saguaro_ready(self) -> None:
        """Fail fast unless Saguaro is healthy and queryable."""
        from core.env_manager import EnvironmentManager

        env = EnvironmentManager()
        env.ensure_ready(self.console)
        probe = self.saguaro._api.query("repository architecture", k=1)
        if not probe.get("results"):
            raise RuntimeError(
                "Saguaro strict mode is enabled but the index returned no results."
            )

    def _extract_saguaro_files(self, result: Dict[str, Any]) -> List[str]:
        files = []
        seen = set()
        for item in result.get("results", []):
            file_path = item.get("file")
            if file_path and file_path not in seen:
                seen.add(file_path)
                files.append(file_path)
        return files

    def _seed_subagent_guidance(self, subagent_id: str, query: str) -> None:
        try:
            profile = (
                self.current_complexity_profile
                or self.complexity_scorer.score_request(query)
            )
            adaptive = self.current_adaptive_complexity or {}
            guidance = (
                f"Stay grounded. Use Saguaro first, then read concrete files directly as needed. "
                f"Complexity score={profile.score}, coconut_frequency={profile.coconut_frequency}, "
                f"coconut_depth={self._resolve_coconut_depth(profile, adaptive)}, "
                f"subagent_slots={self._resolve_subagent_slot_count(profile)}."
            )
            self.message_bus.send(
                sender=self.master_agent_id,
                recipient=subagent_id,
                message_type=MessageType.COORDINATION,
                payload={"guidance": guidance},
                priority=Priority.HIGH,
            )
            self.message_bus.set_shared_context(f"guidance.{subagent_id}", guidance)
        except (AttributeError, RuntimeError, ValueError, TypeError):
            pass

    def _apply_subagent_quality_gate(
        self, evidence: Dict[str, Any], query: str
    ) -> Dict[str, Any]:
        profile = (
            self.current_complexity_profile
            or self.complexity_scorer.score_request(
                query, referenced_files=evidence.get("codebase_files", [])
            )
        )
        gate = self.subagent_quality_gate.evaluate(
            dict(evidence),
            original_query=query,
            complexity_score=profile.score,
        )
        evidence["subagent_quality"] = gate
        evidence["compliance"] = dict(self.current_compliance_context)
        shared_notes = []
        try:
            for key, value in self.message_bus.shared_context.items():
                if key.startswith("scratchpad."):
                    shared_notes.append({"key": key, "value": value})
        except (AttributeError, RuntimeError, TypeError):
            shared_notes = []
        if shared_notes:
            evidence["shared_scratchpad"] = shared_notes[:5]
        if gate.get("should_retry"):
            reason = (
                f"Subagent quality gate requested remediation "
                f"(confidence={gate.get('confidence')}, alignment={gate.get('alignment', {}).get('score', 0):.2f})."
            )
            evidence.setdefault("errors", []).append(reason)
            # Retry signal handling: supplement with direct grounded evidence.
            self._gather_evidence_enhanced(query, evidence)
        return evidence

    def _get_embedding_vector(self, text: str) -> Optional[np.ndarray]:
        if not text:
            return None
        try:
            if hasattr(self.brain, "get_embeddings"):
                emb = self.brain.get_embeddings(text)
            else:
                emb = self.brain.embeddings(text)
            arr = np.asarray(emb, dtype=np.float32)
            if arr.ndim > 1:
                arr = np.mean(arr, axis=0)
            arr = arr.reshape(-1)
            if arr.size == 0 or not np.isfinite(arr).all():
                return None
            return arr
        except (AttributeError, RuntimeError, ValueError, TypeError):
            return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.0
        a_arr = np.asarray(a, dtype=np.float32).reshape(-1)
        b_arr = np.asarray(b, dtype=np.float32).reshape(-1)
        dim = min(a_arr.size, b_arr.size)
        if dim == 0:
            return 0.0
        a_arr = a_arr[:dim]
        b_arr = b_arr[:dim]
        denom = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
        if denom <= 1e-8:
            return 0.0
        return float(np.dot(a_arr, b_arr) / denom)

    def _rerank_evidence_with_coconut(
        self, evidence: Dict[str, Any], context_embedding: np.ndarray
    ) -> List[str]:
        file_contents = evidence.get("file_contents") or {}
        if not file_contents:
            return []

        files = list(file_contents.keys())
        chunks = [str(file_contents[path]) for path in files]
        ranks = self.thinking_system.rank_evidence(chunks, context_embedding)
        if not ranks:
            return []

        ordered_files = [files[idx] for idx, _ in ranks if 0 <= idx < len(files)]
        reordered = {path: file_contents[path] for path in ordered_files}
        for path, content in file_contents.items():
            if path not in reordered:
                reordered[path] = content
        evidence["file_contents"] = reordered
        return ordered_files

    def _build_coconut_reasoning_insight(
        self, evidence: Dict[str, Any], strategy: Dict[str, Any]
    ) -> str:
        amplitudes = (
            strategy.get("amplitudes") or evidence.get("coconut_amplitudes") or []
        )
        amp_arr = np.asarray(amplitudes, dtype=np.float32).flatten()
        amp_arr = amp_arr[np.isfinite(amp_arr)] if amp_arr.size else amp_arr
        dominant_idx = int(np.argmax(amp_arr)) if amp_arr.size else 0
        dominant_direction = {
            0: "architecture/structure",
            1: "implementation/control-flow",
            2: "integration/dependencies",
        }.get(dominant_idx, "mixed")

        confidence = 0.45
        if amp_arr.size > 1 and float(np.sum(amp_arr)) > 0:
            normalized = amp_arr / float(np.sum(amp_arr))
            entropy = float(
                -np.sum(normalized * np.log(np.clip(normalized, 1e-8, None)))
            )
            max_entropy = float(np.log(len(normalized)))
            confidence = max(
                0.0, min(1.0, 1.0 - (entropy / max_entropy if max_entropy else 1.0))
            )

        strongest_files = evidence.get("coconut_reranked_files") or list(
            (evidence.get("file_contents") or {}).keys()
        )
        strongest_files = strongest_files[:3]
        files_line = ", ".join(strongest_files) if strongest_files else "none"
        return (
            "[COCONUT REASONING INSIGHT]\n"
            f"- Dominant direction: {dominant_direction}\n"
            f"- Strongest evidence paths: {files_line}\n"
            f"- Confidence: {confidence:.2f}\n"
        )

    def _strategy_query_text(self, query: str, strategy_name: str) -> str:
        if strategy_name == "structure":
            return f"architecture modules boundaries {query}"
        if strategy_name == "implementation":
            return f"implementation logic control flow {query}"
        if strategy_name == "integration":
            return f"integration dependencies imports interfaces {query}"
        return query

    def _strategy_emphasis_text(self, strategy_name: str) -> str:
        if strategy_name == "structure":
            return (
                "Prioritize module boundaries, ownership, and high-level architecture."
            )
        if strategy_name == "integration":
            return "Prioritize dependencies, import chains, interfaces, and integration risks."
        return "Prioritize concrete implementation details, control flow, and execution behavior."

    def _select_tool_strategy(
        self,
        coconut_amplitudes: Optional[Any],
        question_type: str,
        evidence_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        evidence_state = evidence_state or {}
        amplitudes: List[float] = []
        entropy = None
        if coconut_amplitudes is not None:
            try:
                arr = np.asarray(coconut_amplitudes, dtype=np.float32).flatten()
                if arr.size > 0 and np.isfinite(arr).all():
                    arr = np.clip(arr, 1e-8, None)
                    arr = arr / float(np.sum(arr))
                    amplitudes = [float(x) for x in arr.tolist()]
                    entropy = float(-np.sum(arr * np.log(arr)))
            except (TypeError, ValueError):
                amplitudes = []
                entropy = None

        direct_file = bool(evidence_state.get("direct_file"))
        if amplitudes:
            dominant_idx = int(np.argmax(np.asarray(amplitudes)))
            if dominant_idx == 0:
                strategy = "structure"
            elif dominant_idx == 1:
                strategy = "implementation"
            else:
                strategy = "integration"
            reason = f"coconut_path_{dominant_idx + 1}_dominant"
        elif direct_file:
            strategy = "implementation"
            reason = "direct_file_focus"
        elif question_type == "architecture":
            strategy = "structure"
            reason = "question_type_architecture"
        elif question_type == "investigation":
            strategy = "integration"
            reason = "question_type_investigation"
        else:
            strategy = "implementation"
            reason = "default_implementation"

        return {
            "name": strategy,
            "reason": reason,
            "entropy": entropy,
            "amplitudes": amplitudes,
        }

    def _log_evidence_inventory(self, evidence: Dict[str, Any], stage: str) -> None:
        file_contents = evidence.get("file_contents", {}) or {}
        total_chars = sum(len(str(v)) for v in file_contents.values())
        payload = {
            "component": "evidence",
            "event": "inventory",
            "stage": stage,
            "metrics": {
                "codebase_files": len(evidence.get("codebase_files", []) or []),
                "file_contents": len(file_contents),
                "file_content_chars": total_chars,
                "skeletons": len(evidence.get("skeletons", {}) or {}),
                "entities": len(evidence.get("entities", {}) or {}),
                "tree_views": len(evidence.get("tree_views", {}) or {}),
                "subagent_analysis_chars": len(
                    evidence.get("subagent_analysis", "") or ""
                ),
                "workspace_top_dirs": len(
                    (evidence.get("workspace_map", {}) or {}).get("top_dirs", []) or []
                ),
                "tool_strategy": evidence.get("tool_strategy"),
            },
        }
        logger.info(json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str))

    def _merge_evidence_builder_output(
        self, evidence: Dict[str, Any], built_evidence: Dict[str, Any]
    ) -> None:
        evidence["primary_file"] = built_evidence.get("primary_file")
        evidence["codebase_files"] = built_evidence.get("codebase_files", [])
        evidence["file_contents"].update(built_evidence.get("file_contents", {}))
        evidence["skeletons"] = built_evidence.get("skeletons", {})
        evidence["workspace_map"] = built_evidence.get("workspace_map", {})
        evidence["imports"] = built_evidence.get("imports", {})
        evidence["entities"] = built_evidence.get("entities", {})
        evidence["integration_map"] = built_evidence.get("integration_map", {})
        evidence["dependency_graph"] = built_evidence.get("dependency_graph", {})
        evidence["tree_views"] = built_evidence.get("tree_views", {})
        evidence["validation"] = built_evidence.get("validation", {})

    def _log_strategy_selection(self, query: str, strategy: Dict[str, Any]) -> None:
        logger.info(
            "coconut.strategy.select query=%r strategy=%s reason=%s entropy=%s amplitudes=%s",
            (query or "")[:120],
            strategy["name"],
            strategy["reason"],
            strategy.get("entropy"),
            strategy.get("amplitudes"),
        )

    def _announce_evidence_builder(
        self, dashboard: Optional[LiveProgressDashboard]
    ) -> None:
        if dashboard:
            dashboard.add_agent(
                "Plan",
                "EvidenceBuilder",
                status="running",
                message="Multi-pass grounded evidence gathering...",
            )
            return
        self.console.print("  [dim]→ Multi-pass grounded evidence gathering...[/dim]")

    def _report_grounded_evidence(
        self, evidence: Dict[str, Any], dashboard: Optional[LiveProgressDashboard]
    ) -> None:
        message = (
            f"{len(evidence.get('file_contents', {}))} full files, "
            f"{len(evidence.get('skeletons', {}))} skeletons"
        )
        if dashboard:
            dashboard.update_agent(
                "EvidenceBuilder",
                status="completed",
                progress=1.0,
                message=message,
            )
            return
        self.console.print(f"  [green]✓ Grounded evidence ready ({message})[/green]")

    def _report_strict_evidence_failure(
        self, evidence: Dict[str, Any], exc: Exception
    ) -> None:
        error_msg = f"Strict evidence gathering failed: {exc}"
        logger.error(error_msg)
        evidence.setdefault("errors", []).append(error_msg)
        self.console.print(f"  [red]✗ {error_msg}[/red]")
        self._log_evidence_inventory(evidence, stage="builder_error")

    def _run_strict_evidence_builder(
        self,
        query: str,
        evidence: Dict[str, Any],
        dashboard: Optional[LiveProgressDashboard],
        strategy: Dict[str, Any],
        direct_file_path: Optional[str],
    ) -> None:
        try:
            self._announce_evidence_builder(dashboard)
            strategy_query = self._strategy_query_text(query, strategy["name"])
            built_evidence = self.evidence_builder.build(
                strategy_query, target_file=direct_file_path
            )
            self._merge_evidence_builder_output(evidence, built_evidence)
            if not (evidence.get("file_contents") or evidence.get("skeletons")):
                raise RuntimeError(
                    "Strict evidence mode: EvidenceBuilder produced no file contents/skeletons."
                )
            self._report_grounded_evidence(evidence, dashboard)
            self._log_evidence_inventory(evidence, stage="post_builder")
            if self._needs_web_search(query):
                self._do_web_search(query, evidence)
        except (AttributeError, RuntimeError, ValueError, TypeError, OSError) as exc:
            self._report_strict_evidence_failure(evidence, exc)
            raise RuntimeError(f"SAGUARO_STRICT_EVIDENCE_FAILED: {exc}") from exc

    def _complexity_subagent_slots(
        self, complexity_profile: Optional[ComplexityProfile]
    ) -> int:
        max_slots = self._max_configured_subagent_slots()
        if complexity_profile is None:
            return 1
        explicit_count = getattr(complexity_profile, "subagent_count", None)
        if isinstance(explicit_count, (int, float)):
            return int(max(1, min(max_slots, int(explicit_count))))
        score = int(getattr(complexity_profile, "score", 1) or 1)
        if score >= 9:
            return 4
        if score >= 7:
            return 3
        if score >= 5:
            return 2
        return 1

    @staticmethod
    def _max_configured_subagent_slots() -> int:
        try:
            return max(1, int(DYNAMIC_COCONUT_CONFIG.get("max_subagent_slots", 4)))
        except Exception:
            return 4

    def _run_multi_agent_coconut_gather(
        self, query: str, evidence: Dict[str, Any]
    ) -> None:
        complexity_profile = (
            evidence.get("complexity_profile") or self.current_complexity_profile
        )
        if self.multi_agent_gatherer is None or complexity_profile is None:
            return
        if not getattr(complexity_profile, "subagent_coconut", False):
            return

        candidate_files = list(dict.fromkeys(evidence.get("codebase_files", []) or []))
        if len(candidate_files) < 2:
            return

        self.console.print("  [dim]→ Adaptive multi-agent COCONUT gathering...[/dim]")
        try:
            gathered = self.multi_agent_gatherer.gather_evidence(
                query,
                candidate_files,
                complexity_profile=complexity_profile,
            )
        except (AttributeError, RuntimeError, ValueError, TypeError, OSError) as exc:
            logger.warning("Multi-agent gatherer failed: %s", exc)
            evidence.setdefault("errors", []).append(
                f"Multi-agent gatherer skipped: {exc}"
            )
            return

        gathered_files = gathered.get("file_contents") or {}
        if gathered_files:
            evidence.setdefault("file_contents", {}).update(gathered_files)

        summaries = [s for s in gathered.get("agent_summaries", []) if s]
        if summaries:
            joined = "\n\n".join(summaries)
            existing = str(evidence.get("subagent_analysis", "") or "").strip()
            evidence["subagent_analysis"] = (
                f"{existing}\n\n{joined}".strip() if existing else joined
            )

        for idx, raw_result in enumerate(gathered.get("all_results", []) or []):
            payload = self._extract_subagent_latent_payload(
                result=raw_result,
                source=f"gatherer_agent_{idx}",
            )
            self._append_subagent_latent_payload(evidence, payload)

        for key in (
            "coconut_refined",
            "coconut_paths",
            "coconut_amplitudes",
            "entanglement_correlation",
            "adaptive_allocation",
        ):
            value = gathered.get(key)
            if value is not None:
                evidence[key] = value

        evidence["multi_agent_gathering"] = {
            "total_agents": int(gathered.get("total_agents", 0) or 0),
            "files_analyzed": int(gathered.get("files_analyzed", 0) or 0),
            "total_tokens_used": int(gathered.get("total_tokens_used", 0) or 0),
        }

    def _gather_evidence_enhanced(
        self,
        query: str,
        evidence: Dict[str, Any],
        dashboard: Optional[LiveProgressDashboard] = None,
    ):
        """Enhanced evidence gathering with deterministic strict evidence building."""
        direct_file_path = self._extract_direct_file_path(query)
        strategy = self._select_tool_strategy(
            evidence.get("coconut_amplitudes"),
            evidence.get("question_type", "simple"),
            {
                "request_type": evidence.get("request_type"),
                "direct_file": bool(direct_file_path),
                "query_len": len(query or ""),
            },
        )
        evidence["tool_strategy"] = strategy["name"]
        evidence["tool_strategy_reason"] = strategy["reason"]
        self._log_strategy_selection(query, strategy)
        self._run_strict_evidence_builder(
            query=query,
            evidence=evidence,
            dashboard=dashboard,
            strategy=strategy,
            direct_file_path=direct_file_path,
        )
        self._run_multi_agent_coconut_gather(query=query, evidence=evidence)

    def _extend_with_lexical_file_candidates(
        self, query: str, relevant_files: List[str]
    ) -> None:
        raise RuntimeError(
            "SAGUARO_STRICT_FALLBACK_DISABLED: lexical candidate expansion is disabled."
        )

    def _discover_relevant_files_basic(self, query: str) -> List[str]:
        relevant_files: List[str] = []
        result = self.saguaro._api.query(query, k=10)
        relevant_files.extend(self._extract_saguaro_files(result))
        if len(relevant_files) < 3:
            raise RuntimeError(
                "SAGUARO_STRICT_FALLBACK_DISABLED: semantic recall below minimum "
                f"threshold ({len(relevant_files)} files)."
            )
        return list(dict.fromkeys(relevant_files))

    def _load_basic_file_evidence(self, evidence: Dict[str, Any]) -> None:
        for file_path in evidence["codebase_files"][:5]:
            try:
                content = self._execute_tool("read_file", {"path": file_path})
                if content and not str(content).startswith("Error"):
                    evidence["file_contents"][file_path] = str(content)[:5000]
                    self.files_read.add(file_path)
            except (OSError, RuntimeError, ValueError, TypeError) as exc:
                evidence["errors"].append(f"Failed to read {file_path}: {exc}")

    def _gather_evidence_basic(self, query: str, evidence: Dict[str, Any]):
        """Grounded evidence gathering with strict Saguaro-only discovery."""
        self.console.print("  [dim]→ Saguaro semantic search...[/dim]")
        try:
            relevant_files = self._discover_relevant_files_basic(query)
            evidence["codebase_files"] = relevant_files[:10]
            self.console.print(
                f"  [green]✓ Found {len(relevant_files)} files via Saguaro[/green]"
            )
        except (AttributeError, RuntimeError, ValueError, TypeError, OSError) as exc:
            evidence["errors"].append(f"Saguaro search failed: {exc}")
            raise RuntimeError(f"SAGUARO_STRICT_QUERY_FAILED: {exc}") from exc

        self._load_basic_file_evidence(evidence)

    def _handle_action(
        self, user_input: str, dashboard: Optional[LiveProgressDashboard] = None
    ) -> str:
        """
        Handle action requests (create, modify, delete) with enhanced execution.

        Enhanced features:
        - Smart context gathering
        - Parallel tool execution
        - Auto-verification with retry
        - Task memory recording
        """
        if dashboard:
            dashboard.update_phase(
                "Plan",
                status="in_progress",
                message="Gathering context for action...",
            )
        else:
            self.console.print("[cyan]Phase 1: Planning action...[/cyan]")

        # Gather context for the action
        if self.enhanced_mode:
            context = self._gather_action_context(user_input)
        else:
            context = {}

        if dashboard:
            dashboard.update_phase("Plan", status="completed")
            dashboard.update_phase(
                "Execute", status="in_progress", message="Generating action plan..."
            )
        else:
            self.console.print("[cyan]Phase 2: Generating action plan...[/cyan]")

        # Generate action plan
        action_plan = self._generate_action_plan(user_input, context)

        # Execute the action
        if not dashboard:
            self.console.print("[cyan]Phase 3: Executing action...[/cyan]")

        if self.enhanced_mode:
            results = self._execute_action_enhanced(
                action_plan, user_input, dashboard=dashboard
            )
        else:
            results = self._execute_action_basic(action_plan)

        # Synthesize response
        if dashboard:
            dashboard.update_phase("Execute", status="completed")
            dashboard.update_phase(
                "Synthesize", status="in_progress", message="Summarizing results..."
            )
        else:
            self.console.print("[cyan]Phase 4: Synthesizing response...[/cyan]")

        response = self._synthesize_action_result(
            user_input, action_plan, results, dashboard=dashboard
        )

        return response

    def _gather_action_context(self, task: str) -> Dict[str, Any]:
        """Gather context for an action using smart context manager."""
        bundle = self.context_manager.gather_context(task)

        # Load progressive context
        loaded_context = self.context_loader.load_context_for_task(
            task, initial_files=bundle.target_files
        )

        # Optimize to fit token budget
        optimized = self.context_optimizer.optimize(bundle, token_budget=80000)

        return {
            "bundle": optimized,
            "loaded_files": loaded_context,
            "summary": self.context_loader.get_context_summary(),
        }

    def _create_action_results(self) -> Dict[str, Any]:
        return {
            "files_written": [],
            "files_edited": [],
            "commands_run": [],
            "errors": [],
            "verification": {"passed": True, "issues": []},
            "chronicle": {},
            "legislation": {},
        }

    def _is_high_assurance_change(self) -> bool:
        return str(self.runtime_aal or "").upper() in {"AAL-0", "AAL-1"}

    def _has_mutating_actions(self, tool_calls: List[Dict[str, Any]]) -> bool:
        mutating_tools = {
            "write_file",
            "edit_file",
            "write_files",
            "delete_file",
            "move_file",
            "run_command",
            "apply_patch",
        }
        return any(str(call.get("tool") or "") in mutating_tools for call in tool_calls)

    def _resolve_execution_trace_id(self) -> str:
        return (
            self.current_compliance_context.get("trace_id")
            or self.current_task_id
            or "trace_unknown"
        )

    def _normalize_tool_call_args(self, tool_calls: List[Dict[str, Any]]) -> None:
        for tool_call in tool_calls:
            args = tool_call.setdefault("args", {})
            ensure_context_updates_arg(args)
            tool_name = tool_call.get("tool")
            if tool_name not in {"read_file", "read_files"} or "max_chars" not in args:
                continue
            requested_cap = self._read_requested_max_chars(args.get("max_chars"))
            has_line_window = (
                args.get("start_line") is not None or args.get("end_line") is not None
            )
            if requested_cap is None or requested_cap >= 4000 or has_line_window:
                continue
            logger.warning(
                "Ignoring tiny max_chars=%s for %s without line window.",
                requested_cap,
                tool_name,
            )
            args.pop("max_chars", None)

    def _read_requested_max_chars(self, max_chars_value: Any) -> Optional[int]:
        try:
            return int(max_chars_value)
        except (TypeError, ValueError):
            return None

    def _apply_pre_action_block(
        self, pre_action: Dict[str, Any], results: Dict[str, Any]
    ) -> bool:
        if pre_action.get("allowed", True):
            return False
        reason = pre_action.get("reason")
        results["errors"].append(f"AES pre-action governance block: {reason}")
        results["verification"]["passed"] = False
        results["verification"]["issues"].append(reason)
        return True

    def _record_prechange_snapshot(
        self, results: Dict[str, Any], trace_id: str
    ) -> None:
        snapshot_label = f"{trace_id}_prechange"
        try:
            snapshot_receipt = self.saguaro.create_chronicle_snapshot(
                label=snapshot_label
            )
            results["chronicle"]["pre_snapshot"] = snapshot_receipt
            self._record_reality_event(
                "chronicle_snapshot",
                phase="execute",
                status="ok",
                metadata={"label": snapshot_label, "receipt": snapshot_receipt},
            )
        except (
            AttributeError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            error = f"chronicle_snapshot_failed: {exc}"
            results["errors"].append(error)
            results["chronicle"]["snapshot_error"] = error
            self._record_reality_event(
                "chronicle_snapshot",
                phase="execute",
                status="error",
                metadata={"label": snapshot_label, "error": error},
            )

    def _run_parallel_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        dashboard: Optional[LiveProgressDashboard],
    ) -> List[Dict[str, Any]]:
        self.black_box.record_tool_plan(tool_calls)
        if dashboard:
            dashboard.add_agent(
                "Execute",
                "ParallelExecutor",
                status="running",
                message=f"Executing {len(tool_calls)} tools...",
            )
        else:
            self.console.print(
                f"  [dim]→ Executing {len(tool_calls)} tools in parallel...[/dim]"
            )
        tool_results = self.parallel_executor.execute_tools(tool_calls)
        self.black_box.record_tool_results(tool_results)
        self._apply_context_updates_from_results(tool_results)
        if dashboard:
            dashboard.update_agent("ParallelExecutor", status="completed", progress=1.0)
        return tool_results

    def _process_tool_result_success(
        self, result: Dict[str, Any], results: Dict[str, Any]
    ) -> None:
        from rich.panel import Panel

        tool = result["tool"]
        args = result.get("args", {})
        res_text = str(result.get("result", ""))
        truncated_res = self.agent.truncator.truncate(tool, args, res_text)
        result["result"] = truncated_res
        self._record_tool_result(tool, args, truncated_res)
        self.console.print(
            Panel(
                truncated_res,
                title=f"Tool Output: {tool}",
                border_style="green",
                expand=False,
            )
        )
        if tool == "write_file":
            file_path = args.get("file_path")
            results["files_written"].append(file_path)
            self.files_edited.add(file_path)
        elif tool == "edit_file":
            file_path = args.get("file_path")
            results["files_edited"].append(file_path)
            self.files_edited.add(file_path)
        elif tool == "run_command":
            results["commands_run"].append(args.get("command"))

    def _process_tool_result_failure(
        self, result: Dict[str, Any], results: Dict[str, Any]
    ) -> None:
        error_msg = result.get("error") or result.get("result") or "Unknown error"
        results["errors"].append(f"{result['tool']}: {error_msg}")
        self._record_tool_result(result["tool"], result.get("args", {}), error_msg)
        self.console.print(
            f"[bold red]Tool Error ({result['tool']}):[/bold red] {error_msg}"
        )

    def _process_executed_tool_results(
        self, tool_results: List[Dict[str, Any]], results: Dict[str, Any]
    ) -> None:
        for result in tool_results:
            if result["success"]:
                self._process_tool_result_success(result, results)
            else:
                self._process_tool_result_failure(result, results)

    def _verify_modified_files(
        self,
        modified_files: List[str],
        results: Dict[str, Any],
        dashboard: Optional[LiveProgressDashboard],
    ) -> None:
        if not modified_files:
            return
        if dashboard:
            dashboard.update_phase(
                "Observe", status="in_progress", message="Verifying changes..."
            )
            dashboard.add_agent(
                "Observe",
                "AutoVerifier",
                status="running",
                message=f"Verifying {len(modified_files)} files...",
            )
        else:
            self.console.print("  [dim]→ Auto-verifying changes...[/dim]")
        max_attempts = int(
            self.current_runtime_control.get("verification_max_attempts", 2) or 2
        )
        verification_passed = self.verification_loop.verify_with_retry(
            modified_files, max_attempts=max_attempts
        )
        results["verification"]["passed"] = verification_passed
        results["verification"]["max_attempts"] = max_attempts
        results["verification"]["runtime_posture"] = self.current_runtime_control.get(
            "posture"
        )
        if not verification_passed:
            results["verification"]["issues"].append("Verification failed after retry")
        self.black_box.record_verification(
            modified_files=modified_files,
            passed=verification_passed,
            issues=list(results["verification"]["issues"]),
        )
        if dashboard:
            v_status = "completed" if verification_passed else "failed"
            dashboard.update_agent("AutoVerifier", status=v_status, progress=1.0)
            dashboard.update_phase("Observe", status=v_status, progress=1.0)

    def _record_postchange_chronicle(
        self, results: Dict[str, Any], trace_id: str, task: str
    ) -> None:
        try:
            diff_payload = self.saguaro.create_chronicle_diff()
            results["chronicle"]["post_diff"] = diff_payload
            if isinstance(diff_payload, dict) and diff_payload.get("status") == "error":
                error = str(diff_payload.get("message", "chronicle_diff_failed"))
                results["errors"].append(error)
                results["chronicle"]["diff_error"] = error
                self._record_reality_event(
                    "chronicle_diff",
                    phase="observe",
                    status="error",
                    metadata={"error": error},
                )
                return
            delta_log_path = self.saguaro.write_chronicle_delta_log(
                diff_payload=diff_payload,
                trace_id=trace_id,
                task=task,
            )
            results["chronicle"]["delta_log"] = delta_log_path
            self._record_reality_event(
                "chronicle_diff",
                phase="observe",
                status="ok",
                metadata={"summary": diff_payload.get("summary")},
                artifacts={"delta_log": delta_log_path},
            )
        except (
            AttributeError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            error = f"chronicle_diff_failed: {exc}"
            results["errors"].append(error)
            results["chronicle"]["diff_error"] = error
            self._record_reality_event(
                "chronicle_diff",
                phase="observe",
                status="error",
                metadata={"error": error},
            )

    def _record_legislation_draft(self, results: Dict[str, Any], trace_id: str) -> None:
        try:
            results["legislation"] = self.saguaro.run_legislation_draft(
                reason=f"high_assurance_change:{trace_id}"
            )
        except (
            AttributeError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            results["legislation"] = {
                "status": "error",
                "message": f"legislation_draft_failed: {exc}",
            }

    def _run_high_assurance_finalize_automation(
        self, user_input: str, execution_result: Dict[str, Any]
    ) -> Optional[str]:
        if not self._is_high_assurance_change():
            return None
        if not isinstance(execution_result, dict):
            return "finalize_automation_missing_execution_result"

        trace_id = self._resolve_execution_trace_id()
        chronicle = execution_result.setdefault("chronicle", {})
        execution_result.setdefault("legislation", {})

        try:
            chronicle["finalize_snapshot"] = self.saguaro.create_chronicle_snapshot(
                label=f"{trace_id}_finalize"
            )
        except (
            AttributeError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            return f"chronicle_finalize_snapshot_failed: {exc}"

        self._record_postchange_chronicle(execution_result, trace_id, user_input)
        if chronicle.get("diff_error"):
            return str(chronicle.get("diff_error"))

        self._record_legislation_draft(execution_result, trace_id)
        legislation = execution_result.get("legislation") or {}
        if isinstance(legislation, dict) and legislation.get("status") == "error":
            return str(legislation.get("message", "legislation_draft_failed"))

        return None

    def _reflect_on_action_execution(
        self, results: Dict[str, Any], modified_files: List[str]
    ) -> None:
        if not self.enhanced_mode:
            return
        validity = (
            "Success" if results["verification"]["passed"] else "Failed Verification"
        )
        self.thinking_system.think(
            ThinkingType.REFLECTION,
            f"Execution complete. Verification: {validity}\nErrors: {len(results['errors'])}\nFiles changed: {len(modified_files)}",
        )

    def _execute_action_enhanced(
        self,
        action_plan: str,
        task: str,
        dashboard: Optional[LiveProgressDashboard] = None,
    ) -> Dict[str, Any]:
        """Execute action with parallel tools and auto-verification."""
        results = self._create_action_results()
        logger.debug(f"Generated action plan:\n{action_plan}")
        tool_calls = self._parse_action_plan(action_plan, task)
        logger.debug(f"Parsed tool calls: {tool_calls}")
        if not tool_calls:
            results["errors"].append("Could not parse action plan into tool calls")
            return results

        high_assurance_change = self._is_high_assurance_change()
        has_mutating_action = self._has_mutating_actions(tool_calls)
        trace_id = self._resolve_execution_trace_id()
        self._normalize_tool_call_args(tool_calls)
        results["compiled_plan"] = self._persist_compiled_mission_plan(
            action_plan=action_plan,
            tool_calls=tool_calls,
            task=task,
        )

        pre_action = self._pre_action_tool_checkpoint(tool_calls, task)
        if self._apply_pre_action_block(pre_action, results):
            return results
        if high_assurance_change and has_mutating_action:
            self._record_prechange_snapshot(results, trace_id)

        tool_results = self._run_parallel_tool_calls(tool_calls, dashboard)
        self._process_executed_tool_results(tool_results, results)

        modified_files = results["files_written"] + results["files_edited"]
        self._verify_modified_files(modified_files, results, dashboard)
        self._reflect_on_action_execution(results, modified_files)
        return results

    def _execute_action_basic(self, action_plan: str) -> Dict[str, Any]:
        """Basic action execution without enhanced features."""
        return {
            "files_written": [],
            "files_edited": [],
            "commands_run": [],
            "errors": ["Basic execution mode - action parsing not implemented"],
        }

    def _extract_literal_arg(self, node: ast.AST) -> tuple[bool, Any]:
        if isinstance(node, ast.Constant):  # Python 3.8+
            return True, node.value
        if isinstance(node, ast.Str):  # pragma: no cover - Python < 3.8 compatibility
            return True, node.s
        if isinstance(node, ast.Num):  # pragma: no cover - Python < 3.8 compatibility
            return True, node.n
        return False, None

    def _parse_tool_call_expression(self, call_str: str) -> Optional[Dict[str, Any]]:
        tree = ast.parse(call_str, mode="eval")
        if not isinstance(tree.body, ast.Call):
            logger.warning(f"Parsed expression is not a call: {call_str}")
            return None

        call_node = tree.body
        if not isinstance(call_node.func, ast.Name):
            logger.warning(f"Unsupported callable in tool expression: {call_str}")
            return None

        args: Dict[str, Any] = {}
        for keyword in call_node.keywords:
            if keyword.arg is None:
                continue
            is_literal, value = self._extract_literal_arg(keyword.value)
            if is_literal:
                args[keyword.arg] = value
                continue
            logger.warning(
                "Unsupported AST node type in tool args: %s. AST: %s",
                type(keyword.value),
                ast.dump(keyword.value),
            )

        return {"tool": call_node.func.id, "args": args}

    def _extract_tool_calls_from_response(self, response: str) -> List[Dict[str, Any]]:
        tool_calls: List[Dict[str, Any]] = []
        call_strings = re.findall(r"<tool_code>(.*?)</tool_code>", response, re.DOTALL)
        for call_str in call_strings:
            normalized = call_str.strip()
            if not normalized:
                continue
            try:
                parsed_call = self._parse_tool_call_expression(normalized)
            except (SyntaxError, ValueError) as exc:
                logger.error(
                    "Failed to parse tool call string: '%s'. Error: %s",
                    normalized,
                    exc,
                )
                continue
            if parsed_call:
                tool_calls.append(parsed_call)
        return tool_calls

    def _parse_action_plan(self, action_plan: str, task: str) -> List[Dict[str, Any]]:
        """
        Parse the action plan into executable tool calls using a robust XML-style format.
        """
        pressure_note = self._context_pressure_guidance(task)
        prompt = f"""Convert this action plan into a sequence of tool calls.

Action Plan:
{action_plan}

Original Task:
{task}

Available Tools:
- write_file(file_path: str, content: str, _context_updates: list[dict[str, str]])
- edit_file(file_path: str, instruction: str, _context_updates: list[dict[str, str]])
- run_command(command: str, _context_updates: list[dict[str, str]])
- delete_file(file_path: str, _context_updates: list[dict[str, str]])

Context Compression:
- `_context_updates` is REQUIRED on every tool call.
- Use `[]` if nothing should be compressed.
- Compress stale `[tcN]` tool results only.
- NEVER target results that do not contain `[tcN]`.
- {pressure_note}

Wrap each Python tool call in <tool_code> tags. For example:
<tool_code>write_file(file_path="foo.py", content="print('hello')", _context_updates=[])</tool_code>
<tool_code>run_command(command="pytest", _context_updates=[{{"tc2":"Old grep result is irrelevant"}}])</tool_code>

Output ONLY the <tool_code> blocks, with no other text or explanation.
"""

        messages = [
            {
                "role": "system",
                "content": "You are a tool call generator. You convert plans into Python function calls wrapped in XML tags.",
            },
            {"role": "user", "content": prompt},
        ]

        response = ""
        for chunk in self.brain.stream_chat(messages, max_tokens=4096, temperature=0.0):
            response += chunk

        logger.debug(f"Raw model response for tool parsing:\n{response}")

        tool_calls = self._extract_tool_calls_from_response(response)

        if not tool_calls:
            logger.warning("Could not parse any tool calls from model response.")
        else:
            logger.info(f"Successfully parsed {len(tool_calls)} tool calls.")

        return tool_calls

    def _get_master_system_prompt(
        self,
        context_type: str = "general",
        task: str = "",
        workset_files: Optional[List[str]] = None,
    ) -> str:
        """
        Returns the assembled master system prompt using PromptManager.
        """
        tracked_files = list(sorted(self.files_read | self.files_edited))
        if workset_files:
            tracked_files.extend(workset_files)
        tracked_files = list(dict.fromkeys(tracked_files))
        base_prompt = self.prompt_manager.get_master_prompt(
            agent_name=self.agent.name,
            context_type=context_type,
            task_text=task,
            workset_files=tracked_files or None,
            prompt_context=self._current_prompt_contract_context(tracked_files),
        )
        model_name = getattr(self.brain, "model_name", "")
        family_guidance = self.prompt_manager.get_model_family_compression_guidance(
            model_name
        )
        context_guidance = self._context_pressure_guidance(task)
        return (
            f"{base_prompt}\n\n"
            "## Context Compression\n"
            f"{context_guidance}\n"
            f"{family_guidance}\n"
        )

    def _handle_conversational(
        self, user_input: str, dashboard: Optional[Any] = None
    ) -> str:
        """Handle simple conversational requests."""
        if not dashboard:
            self.console.print("[dim]Conversational mode[/dim]")

        messages = self.history.get_messages()
        system_prompt = self._get_master_system_prompt(
            "conversational", task=user_input
        )

        chat_messages = [{"role": "system", "content": system_prompt}] + messages

        if dashboard:
            response = self._stream_into_dashboard(chat_messages, dashboard)
        else:
            response = self.agent._stream_response(chat_messages)

        self.history.add_message("assistant", response)
        return response

    def _stream_into_dashboard(
        self,
        messages: List[Dict[str, Any]],
        dashboard: LiveProgressDashboard,
        assistant_prefix: str = "",
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Stream a response directly into the live progress dashboard."""
        from rich.panel import Panel
        from rich.text import Text

        current_text = Text()
        active_panel_text = None
        active_panel = None

        def callback(event):
            nonlocal active_panel_text, active_panel

            if event.type == "content":
                if active_panel:
                    # Close the thinking panel
                    active_panel = None
                    active_panel_text = None
                    dashboard.stream_output.renderables.append(Text(""))  # Spacer

                # Check if we need to start a content panel
                has_content_panel = any(
                    isinstance(r, Panel) and r.title == "Assistant"
                    for r in dashboard.stream_output.renderables
                )
                if not has_content_panel:
                    dashboard.stream_output.renderables.append(
                        Panel(current_text, title="Assistant", border_style="green")
                    )

                current_text.append(event.content)
                dashboard.update_display()

            elif event.type == "thinking_start":
                active_panel_text = Text(style="dim italic")
                t_type = event.metadata.get("type", "reasoning")
                active_panel = Panel(
                    active_panel_text, title=f"Thinking ({t_type})", border_style="dim"
                )
                dashboard.stream_output.renderables.append(active_panel)
                dashboard.update_display()

            elif event.type == "thinking_chunk":
                if active_panel_text is not None:
                    active_panel_text.append(event.content)
                    dashboard.update_display()

            elif event.type == "thinking_end":
                if active_panel:
                    active_panel.border_style = "green"
                    active_panel.title = "Thought complete"
                    dashboard.update_display()

        return self.agent._stream_response(
            messages,
            assistant_prefix=assistant_prefix,
            callback=callback,
            is_streaming_ui=True,
            generation_kwargs=generation_kwargs,
        )

    def _resolve_pipeline_generation_kwargs(
        self,
        *,
        request_type: str,
        user_input: str,
        evidence: Optional[Dict[str, Any]] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.pipeline_manager is None:
            resolved = dict(overrides or {})
        else:
            resolved = self.pipeline_manager.resolve_generation_kwargs(
                request_type=request_type,
                user_input=user_input,
                overrides=overrides,
            )
        if not evidence:
            return resolved

        latent_prior = evidence.get("subagent_latent_merged")
        if isinstance(latent_prior, list) and latent_prior:
            resolved["latent_prior"] = [float(value) for value in latent_prior]
            resolved["subagent_latent_merged"] = [
                float(value) for value in latent_prior
            ]

        delta_watermark = (
            evidence.get("qsg_delta_watermark")
            or evidence.get("delta_watermark")
            or resolved.get("delta_watermark")
        )
        if isinstance(delta_watermark, dict) and delta_watermark:
            resolved["delta_watermark"] = dict(delta_watermark)
            resolved.setdefault("repo_delta", dict(delta_watermark))

        invariant_terms: list[str] = []
        for key in ("coconut_reranked_files", "latent_reranked_files"):
            for item in list(evidence.get(key) or []):
                if isinstance(item, dict):
                    value = item.get("path") or item.get("file")
                else:
                    value = item
                text = str(value or "").strip()
                if text:
                    invariant_terms.append(text)
        if invariant_terms:
            resolved["invariant_terms"] = invariant_terms
        resolved.setdefault("context_text", str(user_input or ""))
        return resolved

    def _synthesize_answer(
        self, user_input: str, evidence: Dict[str, Any], dashboard: Optional[Any] = None
    ) -> str:
        """Synthesize evidence into a comprehensive answer with COCONUT thinking (Cached)."""
        import hashlib

        # Check cache
        context_hash = ""
        try:
            context_summary = self.context_loader.get_context_summary() or ""
            context_hash = hashlib.sha256(context_summary.encode()).hexdigest()
            cached = self.response_cache.get_cached_response(user_input, context_hash)
            if cached:
                self.console.print("[green]Restored response from cache.[/green]")
                if dashboard:
                    dashboard.update_phase(
                        "Synthesize", status="completed", message="Restored from cache"
                    )
                return cached
        except (AttributeError, RuntimeError, ValueError, TypeError) as e:
            self.console.print(f"[dim]Cache check failed: {e}[/dim]")

        response = self._synthesize_answer_core(user_input, evidence, dashboard)

        # Cache result
        try:
            if context_hash:
                self.response_cache.cache_response(user_input, context_hash, response)
        except (AttributeError, RuntimeError, ValueError, TypeError):
            pass

        return response

    def _show_master_synthesis_panel(self) -> None:
        from rich import box
        from rich.panel import Panel

        self.console.print("\n")
        self.console.print(
            Panel(
                "[bold green]Phase 2: Master Synthesis[/bold green]\n"
                "[dim]Master agent analyzing all evidence with COCONUT deep reasoning...[/dim]",
                border_style="green",
                box=box.HEAVY,
                padding=(0, 2),
            )
        )
        self.console.print("")

    def _calculate_synthesis_evidence_budget(self, user_input: str) -> int:
        from core.token_budget import estimate_context_budget

        history_messages = self.history.get_messages()
        history_est = sum(
            len(str(m.get("content", ""))) // 4 for m in history_messages[-10:]
        )
        available_budget = estimate_context_budget(
            max_context=self.context_budget_allocator.total_budget,
            system_prompt_tokens=2000,
            user_input_tokens=len(user_input) // 4,
            history_tokens=history_est,
            response_reserve=50000,
        )
        evidence_budget = int(available_budget * 0.6)
        return min(
            evidence_budget,
            self.context_budget_allocator.get_budget("master")
            + self.context_budget_allocator.get_budget("coconut"),
        )

    def _prepare_synthesis_evidence_text(
        self, evidence: Dict[str, Any], evidence_budget: int
    ) -> str:
        self.console.print(
            f"  [cyan]→ Formatting evidence:[/cyan] [dim]{evidence_budget:,} token budget[/dim]"
        )
        evidence_text = self._format_evidence(evidence, token_budget=evidence_budget)
        self._log_evidence_inventory(evidence, stage="pre_synthesis")
        return evidence_text

    def _start_enhanced_synthesis_thinking(
        self, user_input: str, evidence: Dict[str, Any]
    ) -> None:
        if hasattr(self.agent, "renderer"):
            self.agent.renderer.print_thinking_start()
        else:
            self.console.print(
                "\n[bold magenta]Deep Thinking Phase (COCONUT)[/bold magenta]"
            )
        self.thinking_system.start_chain(
            task_id=self.current_task_id,
            compliance_context=self.current_compliance_context,
        )
        self.thinking_system.think(
            ThinkingType.UNDERSTANDING,
            f"Analyzing question: {user_input}...\nWith {len(evidence.get('codebase_files', []))} files of context.",
        )

    def _coconut_usage_decision(
        self,
        request_type: str,
        evidence: Dict[str, Any],
        complexity_profile: Optional[ComplexityProfile],
    ) -> tuple[bool, float]:
        complexity_score = evidence.get(
            "complexity_score",
            complexity_profile.score if complexity_profile else 0,
        )
        question_type = evidence.get("question_type", "simple")
        coconut_frequency = (
            complexity_profile.coconut_frequency
            if complexity_profile
            else "synthesis_only"
        )
        should_use = (
            self.thinking_system.coconut_enabled
            and coconut_frequency != "none"
            and (
                coconut_frequency == "synthesis_only"
                or complexity_score >= 6
                or question_type == "architecture"
            )
            and request_type not in ["simple", "conversational"]
        )
        return should_use, float(complexity_score)

    def _configure_coconut_profile(
        self,
        coconut: Any,
        complexity_profile: Optional[ComplexityProfile],
        evidence: Optional[Dict[str, Any]] = None,
    ) -> None:
        evidence = evidence or {}
        adaptive = (
            evidence.get("adaptive_complexity") or self.current_adaptive_complexity
        )
        if complexity_profile:
            coconut.config["num_paths"] = complexity_profile.coconut_paths
        elif "subagent_slots" in adaptive:
            coconut.config["num_paths"] = int(max(2, adaptive.get("subagent_slots", 2)))
        coconut.config["steps"] = self._resolve_coconut_depth(
            complexity_profile, adaptive
        )

    def _reshape_coconut_embeddings(self, embeddings: Any, embedding_dim: int) -> Any:
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings, dtype=np.float32)
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.shape[1] == embedding_dim:
            return embeddings
        if embeddings.shape[1] > embedding_dim:
            return embeddings[:, :embedding_dim]
        pad_width = ((0, 0), (0, embedding_dim - embeddings.shape[1]))
        return np.pad(embeddings, pad_width, mode="constant")

    def _collect_reinjected_latent_vectors(
        self, evidence: Dict[str, Any], target_dim: int
    ) -> List[np.ndarray]:
        vectors: List[np.ndarray] = []
        merged = self._coerce_latent_vector(
            evidence.get("subagent_latent_merged"), target_dim=target_dim
        )
        if merged is not None:
            vectors.append(merged)
        for signal in evidence.get("subagent_latent_signals", []) or []:
            if not isinstance(signal, dict):
                continue
            vec = self._coerce_latent_vector(signal.get("state"), target_dim=target_dim)
            if vec is not None:
                vectors.append(vec)
        latent_memory = getattr(self.agent, "latent_memory", None)
        if latent_memory is not None and hasattr(latent_memory, "get_merged_vector"):
            try:
                mem_vec = latent_memory.get_merged_vector(limit=3)
            except Exception:
                mem_vec = None
            vec = self._coerce_latent_vector(mem_vec, target_dim=target_dim)
            if vec is not None:
                vectors.append(vec)
        return vectors

    def _apply_reinjected_latent_to_embeddings(
        self, embeddings: np.ndarray, evidence: Dict[str, Any], embedding_dim: int
    ) -> tuple[np.ndarray, bool]:
        vectors = self._collect_reinjected_latent_vectors(
            evidence, target_dim=embedding_dim
        )
        if not vectors:
            evidence["latent_reinjection_applied"] = False
            return embeddings, False

        prior = np.mean(np.asarray(vectors, dtype=np.float32), axis=0).reshape(1, -1)
        norm = np.linalg.norm(prior, axis=1, keepdims=True)
        norm = np.where(norm <= 1e-8, 1.0, norm)
        prior = prior / norm
        blended = np.asarray(embeddings, dtype=np.float32)
        blended[0:1, :] = 0.78 * blended[0:1, :] + 0.22 * prior
        blended_norm = np.linalg.norm(blended[0:1, :], axis=1, keepdims=True)
        blended_norm = np.where(blended_norm <= 1e-8, 1.0, blended_norm)
        blended[0:1, :] = blended[0:1, :] / blended_norm
        evidence["subagent_latent_merged"] = [float(v) for v in prior.reshape(-1)]
        evidence["subagent_latent_vector_count"] = int(len(vectors))
        evidence["latent_reinjection_applied"] = True
        return blended, True

    def _store_coconut_synthesis_artifacts(
        self, evidence: Dict[str, Any], refined_embedding: Any, coconut_amplitudes: Any
    ) -> None:
        evidence["coconut_refined"] = refined_embedding
        evidence["coconut_amplitudes"] = (
            [float(x) for x in np.asarray(coconut_amplitudes).flatten()]
            if coconut_amplitudes is not None
            else []
        )
        reranked = self._rerank_evidence_with_coconut(
            evidence, np.asarray(refined_embedding).reshape(-1)
        )
        if reranked:
            evidence["coconut_reranked_files"] = reranked

    def _perform_coconut_synthesis_exploration(
        self,
        user_input: str,
        evidence: Dict[str, Any],
        complexity_profile: Optional[ComplexityProfile],
    ) -> Optional[Any]:
        try:
            coconut = self.thinking_system.coconut
            if coconut is None:
                self.console.print(
                    "  [yellow]→ COCONUT disabled or failed to initialize.[/yellow]"
                )
                raise ValueError("COCONUT unavailable")
            self._configure_coconut_profile(
                coconut,
                complexity_profile,
                evidence=evidence,
            )
            device_info = coconut.get_device_info()
            logger.info(f"COCONUT synthesis backend: {device_info}")
            query_text = (
                user_input + "\n" + str(evidence.get("codebase_files", []))[:500]
            )
            embeddings = self.brain.embeddings(query_text)
            embedding_dim = coconut.config.get("embedding_dim", 512)
            embeddings = self._reshape_coconut_embeddings(embeddings, embedding_dim)
            embeddings, _ = self._apply_reinjected_latent_to_embeddings(
                np.asarray(embeddings, dtype=np.float32),
                evidence=evidence,
                embedding_dim=embedding_dim,
            )
            adaptive_metrics = None
            max_steps = self._resolve_coconut_depth(
                complexity_profile,
                evidence.get("adaptive_complexity"),
            )
            if max_steps > 1 and hasattr(coconut, "explore_adaptive"):
                refined_embedding, adaptive_metrics = coconut.explore_adaptive(
                    embeddings,
                    min_steps=1,
                    max_steps=max_steps,
                )
            else:
                refined_embedding = coconut.explore(embeddings)
            coconut_amplitudes = coconut.amplitudes
            if (
                coconut_amplitudes is None
                and adaptive_metrics is not None
                and adaptive_metrics.path_amplitudes
            ):
                coconut_amplitudes = adaptive_metrics.path_amplitudes
            if adaptive_metrics is not None:
                evidence["coconut_adaptive_metrics"] = adaptive_metrics.to_dict()
            self._store_coconut_synthesis_artifacts(
                evidence, refined_embedding, coconut_amplitudes
            )
            self.console.print(
                f"  [green]✓ COCONUT exploration complete ({device_info.get('backend', 'unknown')} backend)[/green]"
            )
            return coconut_amplitudes
        except (RuntimeError, ValueError, TypeError) as exc:
            self.console.print(
                f"  [yellow]⚠ COCONUT exploration failed: {exc}[/yellow]"
            )
            logger.warning(f"COCONUT exploration error: {exc}", exc_info=True)
            return None

    def _render_coconut_paths(self, coconut_amplitudes: Any) -> None:
        if coconut_amplitudes is None:
            return
        path_summaries = [f"Path {i + 1}" for i in range(len(coconut_amplitudes))]
        if hasattr(self.agent, "renderer") and self.agent.renderer:
            self.agent.renderer.print_coconut_paths(
                coconut_amplitudes,
                path_summaries,
                num_paths=len(coconut_amplitudes),
            )
            return
        self.console.print(
            f"  [cyan]COCONUT paths explored with amplitudes: {coconut_amplitudes}[/cyan]"
        )

    def _run_synthesis_coconut_thinking(
        self,
        user_input: str,
        evidence: Dict[str, Any],
        request_type: str,
        complexity_profile: Optional[ComplexityProfile],
        coconut_amplitudes: Any,
    ) -> Any:
        if not self.enhanced_mode:
            return coconut_amplitudes
        self._start_enhanced_synthesis_thinking(user_input, evidence)
        should_use_coconut, complexity_score = self._coconut_usage_decision(
            request_type, evidence, complexity_profile
        )
        if should_use_coconut:
            self.console.print(
                f"  [dim]→ Exploring latent solution space (complexity={complexity_score:.1f})...[/dim]"
            )
            if (
                evidence.get("coconut_paths") is not None
                and coconut_amplitudes is not None
            ):
                self.console.print(
                    "  [green]✓ COCONUT paths and amplitudes from multi-agent gathering[/green]"
                )
            else:
                coconut_amplitudes = self._perform_coconut_synthesis_exploration(
                    user_input, evidence, complexity_profile
                )
            self._render_coconut_paths(coconut_amplitudes)
            return coconut_amplitudes
        if self.thinking_system.coconut_enabled:
            self.console.print(
                f"  [dim]→ Skipping COCONUT (complexity={complexity_score:.1f} < 10, type={request_type})[/dim]"
            )
        return coconut_amplitudes

    def _apply_synthesis_strategy(
        self, evidence: Dict[str, Any], coconut_amplitudes: Any
    ) -> Dict[str, Any]:
        strategy = self._select_tool_strategy(
            coconut_amplitudes,
            evidence.get("question_type", "simple"),
            {
                "request_type": evidence.get("request_type"),
                "direct_file": bool(evidence.get("primary_file")),
            },
        )
        evidence["tool_strategy"] = strategy["name"]
        evidence["tool_strategy_reason"] = strategy["reason"]
        logger.info(
            "coconut.strategy.synthesis strategy=%s reason=%s entropy=%s amplitudes=%s",
            strategy["name"],
            strategy["reason"],
            strategy.get("entropy"),
            strategy.get("amplitudes"),
        )
        return strategy

    def _insufficient_synthesis_evidence_response(
        self, total_evidence_chars: int
    ) -> str:
        self.console.print(
            "[red]⚠ Evidence gathering failed - insufficient data for synthesis[/red]"
        )
        self.console.print(f"[dim]Evidence size: {total_evidence_chars} chars[/dim]")
        return """I apologize, but the code analysis system encountered an issue gathering evidence for this question.

This could be due to:
1. The subagent stopped prematurely (check for errors above)
2. No relevant files were found in the codebase
3. The question requires information not present in the indexed codebase

Please try:
- Rephrasing your question more specifically
- Checking if the relevant code files exist
- Using /saguaro to verify the codebase is indexed correctly"""

    def _build_subagent_context_for_synthesis(
        self,
        evidence: Dict[str, Any],
        synthesis_strategy: Dict[str, Any],
        coconut_amplitudes: Any,
    ) -> str:
        subagent_context = ""
        if evidence.get("subagent_analysis"):
            subagent_context = f"""

## Subagent Analysis
The {evidence.get('subagent_type', 'specialist')} subagent produced:
{evidence['subagent_analysis']}
"""
        latent_signal_count = int(evidence.get("subagent_latent_signal_count", 0) or 0)
        latent_reinjections = int(evidence.get("subagent_latent_reinjections", 0) or 0)
        if latent_signal_count > 0:
            subagent_context += (
                "\n## Subagent Latent Signals\n"
                f"- Signals merged: {latent_signal_count}\n"
                f"- Reinjections observed: {latent_reinjections}\n"
                f"- Reinjected in synthesis: {bool(evidence.get('latent_reinjection_applied', False))}\n"
            )
        if coconut_amplitudes is None and not evidence.get("coconut_reranked_files"):
            return subagent_context
        coconut_insight = self._build_coconut_reasoning_insight(
            evidence, synthesis_strategy
        )
        return subagent_context + f"\n{coconut_insight}\n"

    def _build_synthesis_system_prompt(
        self,
        user_input: str,
        evidence: Dict[str, Any],
        evidence_text: str,
        request_type: str,
        capability_tier: str,
        subagent_context: str,
        prompt_workset_files: List[str],
        strict: bool = False,
    ) -> str:
        if request_type in {"creation", "modification"}:
            system_prompt = (
                self._get_master_system_prompt(
                    "synthesis", task=user_input, workset_files=prompt_workset_files
                )
                + f"""
### EVIDENCE
{evidence_text}
{subagent_context}

### TASK
Provide a detailed implementation roadmap with concrete file-level steps and verification commands.
"""
            )
        else:
            system_prompt = (
                self._get_master_system_prompt(
                    "synthesis",
                    task=user_input,
                    workset_files=prompt_workset_files,
                )
                + "\n\n"
                + self._build_grounded_synthesis_prompt(
                    evidence=evidence,
                    evidence_text=evidence_text,
                    capability_tier=capability_tier,
                    subagent_context=subagent_context,
                    strict=strict,
                )
            )
        return system_prompt + f"""

## Strategy Emphasis
{self._strategy_emphasis_text(evidence.get("tool_strategy", "implementation"))}
"""

    def _build_synthesis_messages(
        self,
        user_input: str,
        evidence: Dict[str, Any],
        evidence_text: str,
        synthesis_strategy: Dict[str, Any],
        coconut_amplitudes: Any,
    ) -> Dict[str, Any]:
        capability_tier = self._get_model_capability_tier()
        request_type = evidence.get("request_type", "question")
        prompt_workset_files = sorted(
            set(evidence.get("codebase_files", []))
            | set(evidence.get("file_contents", {}).keys())
        )
        subagent_context = self._build_subagent_context_for_synthesis(
            evidence, synthesis_strategy, coconut_amplitudes
        )
        system_prompt = self._build_synthesis_system_prompt(
            user_input=user_input,
            evidence=evidence,
            evidence_text=evidence_text,
            request_type=request_type,
            capability_tier=capability_tier,
            subagent_context=subagent_context,
            prompt_workset_files=prompt_workset_files,
        )
        user_prompt = f"""Question: {user_input}

IMPORTANT: Use detailed <thinking> blocks to analyze the evidence step-by-step, then provide your answer."""
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "capability_tier": capability_tier,
            "request_type": request_type,
            "prompt_workset_files": prompt_workset_files,
            "subagent_context": subagent_context,
        }

    def _generate_synthesis_response(
        self,
        user_input: str,
        evidence: Dict[str, Any],
        synthesis_payload: Dict[str, Any],
        dashboard: Optional[Any],
    ) -> str:
        messages = synthesis_payload["messages"]
        capability_tier = synthesis_payload["capability_tier"]
        request_type = synthesis_payload["request_type"]
        generation_kwargs = self._resolve_pipeline_generation_kwargs(
            request_type=request_type,
            user_input=user_input,
            evidence=evidence,
        )
        if capability_tier == "tier_1_minimal" and request_type == "question":
            deterministic_small = self._synthesize_for_small_model(
                user_input=user_input, evidence=evidence
            )
            if deterministic_small:
                return deterministic_small
            return self._synthesize_with_consistency_check(
                messages=messages,
                evidence=evidence,
                dashboard=dashboard,
                num_samples=3,
            )
        if dashboard:
            return self._stream_into_dashboard(
                messages,
                dashboard,
                generation_kwargs=generation_kwargs,
            )
        return self.agent._stream_response(
            messages,
            generation_kwargs=generation_kwargs,
        )

    def _sanitize_synthesis_response(
        self, response: str, evidence: Dict[str, Any]
    ) -> tuple[str, List[Any]]:
        response = self._deduplicate_response(response)
        response = self._deduplicate_response_v2(response)
        response = self._validate_markdown(response)
        response, violations = self.hallucination_gate.validate(response, evidence)
        return response, violations

    def _regenerate_synthesis_with_strict_grounding(
        self,
        user_input: str,
        evidence: Dict[str, Any],
        evidence_text: str,
        synthesis_payload: Dict[str, Any],
        dashboard: Optional[Any],
    ) -> str:
        synthesis_payload["messages"][0]["content"] = (
            self._build_synthesis_system_prompt(
                user_input=user_input,
                evidence=evidence,
                evidence_text=evidence_text,
                request_type=synthesis_payload["request_type"],
                capability_tier=synthesis_payload["capability_tier"],
                subagent_context=synthesis_payload["subagent_context"],
                prompt_workset_files=synthesis_payload["prompt_workset_files"],
                strict=True,
            )
        )
        if dashboard:
            response = self._stream_into_dashboard(
                synthesis_payload["messages"],
                dashboard,
                generation_kwargs=self._resolve_pipeline_generation_kwargs(
                    request_type=synthesis_payload["request_type"],
                    user_input=user_input,
                    evidence=evidence,
                ),
            )
        else:
            response = self.agent._stream_response(
                synthesis_payload["messages"],
                generation_kwargs=self._resolve_pipeline_generation_kwargs(
                    request_type=synthesis_payload["request_type"],
                    user_input=user_input,
                    evidence=evidence,
                ),
            )
        response, _ = self._sanitize_synthesis_response(response, evidence)
        return response

    def _handle_synthesis_hallucinations(
        self,
        user_input: str,
        response: str,
        violations: List[Any],
        evidence: Dict[str, Any],
        evidence_text: str,
        synthesis_payload: Dict[str, Any],
        dashboard: Optional[Any],
    ) -> str:
        if not violations:
            return response
        self.console.print(
            f"[yellow]⚠ Hallucination gate caught {len(violations)} unverified references[/yellow]"
        )
        logger.warning("Hallucination violations: %s", violations)
        if len(violations) <= 5:
            return response
        self.console.print(
            "[red]⚠ Regenerating with stricter grounding constraints...[/red]"
        )
        return self._regenerate_synthesis_with_strict_grounding(
            user_input=user_input,
            evidence=evidence,
            evidence_text=evidence_text,
            synthesis_payload=synthesis_payload,
            dashboard=dashboard,
        )

    def _load_missing_entities_for_synthesis(
        self, missing_entities: List[str], evidence: Dict[str, Any]
    ) -> None:
        for entity in missing_entities[:5]:
            try:
                search_result = self.semantic_engine.get_context_for_objective(
                    f"definition of {entity}"
                )
                if not search_result:
                    continue
                file_path = search_result[0]
                content = self._execute_tool("read_file", {"path": file_path})
                if not content or content.startswith("Error"):
                    continue
                evidence["file_contents"][file_path] = content[:10000]
                self.console.print(
                    f"    [green]✓ Loaded {file_path} for {entity}[/green]"
                )
            except (RuntimeError, ValueError, TypeError):
                continue

    def _resynthesize_with_expanded_context(
        self,
        user_input: str,
        evidence: Dict[str, Any],
        evidence_budget: int,
        synthesis_payload: Dict[str, Any],
        dashboard: Optional[Any],
    ) -> str:
        evidence_text = self._format_evidence(evidence, token_budget=evidence_budget)
        workset_files = sorted(
            set(synthesis_payload["prompt_workset_files"])
            | set(evidence.get("file_contents", {}).keys())
        )
        synthesis_payload["messages"][0]["content"] = (
            self._build_synthesis_system_prompt(
                user_input=user_input,
                evidence=evidence,
                evidence_text=evidence_text,
                request_type=synthesis_payload["request_type"],
                capability_tier=synthesis_payload["capability_tier"],
                subagent_context=synthesis_payload["subagent_context"],
                prompt_workset_files=workset_files,
            )
        )
        if dashboard:
            dashboard.stream_output.renderables.clear()
            response = self._stream_into_dashboard(
                synthesis_payload["messages"],
                dashboard,
                generation_kwargs=self._resolve_pipeline_generation_kwargs(
                    request_type=synthesis_payload["request_type"],
                    user_input=user_input,
                    evidence=evidence,
                ),
            )
        else:
            response = self.agent._stream_response(
                synthesis_payload["messages"],
                generation_kwargs=self._resolve_pipeline_generation_kwargs(
                    request_type=synthesis_payload["request_type"],
                    user_input=user_input,
                    evidence=evidence,
                ),
            )
        response, _ = self._sanitize_synthesis_response(response, evidence)
        return response

    def _expand_synthesis_context_if_needed(
        self,
        user_input: str,
        response: str,
        evidence: Dict[str, Any],
        evidence_budget: int,
        synthesis_payload: Dict[str, Any],
        dashboard: Optional[Any],
    ) -> str:
        if not self.enhanced_mode or not self._response_needs_more_context(response):
            return response
        self.console.print(
            "  [cyan]→ Detected incomplete answer, expanding context...[/cyan]"
        )
        missing_entities = self._extract_missing_entities(response)
        if not missing_entities:
            return response
        self.console.print(
            f"    [dim]Looking for: {', '.join(missing_entities[:3])}...[/dim]"
        )
        self._load_missing_entities_for_synthesis(missing_entities, evidence)
        return self._resynthesize_with_expanded_context(
            user_input=user_input,
            evidence=evidence,
            evidence_budget=evidence_budget,
            synthesis_payload=synthesis_payload,
            dashboard=dashboard,
        )

    def _finalize_synthesis_response(self, response: str) -> str:
        if self.enhanced_mode:
            return finalize_synthesis(
                response,
                self.thinking_system,
                self.brain,
                self.history.get_messages(),
                self.console,
            )
        return clean_response(response)

    def _synthesize_answer_core(
        self, user_input: str, evidence: Dict[str, Any], dashboard: Optional[Any] = None
    ) -> str:
        """Core synthesis logic."""
        self._show_master_synthesis_panel()
        evidence_budget = self._calculate_synthesis_evidence_budget(user_input)
        evidence_text = self._prepare_synthesis_evidence_text(evidence, evidence_budget)
        request_type = evidence.get("request_type", "question")
        coconut_amplitudes = evidence.get("coconut_amplitudes")
        complexity_profile = (
            evidence.get("complexity_profile") or self.current_complexity_profile
        )
        latent_prior = self._coerce_latent_vector(
            evidence.get("subagent_latent_merged")
        )
        if latent_prior is not None and not evidence.get("coconut_reranked_files"):
            latent_reranked = self._rerank_evidence_with_coconut(
                evidence, context_embedding=latent_prior
            )
            if latent_reranked:
                evidence["latent_reranked_files"] = latent_reranked
        coconut_amplitudes = self._run_synthesis_coconut_thinking(
            user_input=user_input,
            evidence=evidence,
            request_type=request_type,
            complexity_profile=complexity_profile,
            coconut_amplitudes=coconut_amplitudes,
        )
        synthesis_strategy = self._apply_synthesis_strategy(
            evidence, coconut_amplitudes
        )
        subagent_type = evidence.get("subagent_type")
        if subagent_type:
            self.console.print(
                f"  [dim]Synthesizing {subagent_type} subagent findings...[/dim]"
            )
        total_evidence_chars = len(evidence_text) if evidence_text else 0
        if total_evidence_chars < 100 and not evidence.get("subagent_analysis"):
            return self._insufficient_synthesis_evidence_response(total_evidence_chars)

        synthesis_payload = self._build_synthesis_messages(
            user_input=user_input,
            evidence=evidence,
            evidence_text=evidence_text,
            synthesis_strategy=synthesis_strategy,
            coconut_amplitudes=coconut_amplitudes,
        )
        response = self._generate_synthesis_response(
            user_input=user_input,
            evidence=evidence,
            synthesis_payload=synthesis_payload,
            dashboard=dashboard,
        )
        response, violations = self._sanitize_synthesis_response(response, evidence)
        response = self._handle_synthesis_hallucinations(
            user_input=user_input,
            response=response,
            violations=violations,
            evidence=evidence,
            evidence_text=evidence_text,
            synthesis_payload=synthesis_payload,
            dashboard=dashboard,
        )
        response = self._expand_synthesis_context_if_needed(
            user_input=user_input,
            response=response,
            evidence=evidence,
            evidence_budget=evidence_budget,
            synthesis_payload=synthesis_payload,
            dashboard=dashboard,
        )
        response = self._finalize_synthesis_response(response)
        self.history.add_message("assistant", response)
        return response

    def _resolve_small_model_primary_file(
        self, evidence: Dict[str, Any]
    ) -> Optional[str]:
        primary_file = evidence.get("primary_file")
        if primary_file:
            return primary_file
        return next(iter(evidence.get("file_contents", {}).keys()), None)

    def _collect_small_model_methods(
        self, evidence: Dict[str, Any], primary_file: str
    ) -> List[str]:
        methods = []
        for line in evidence.get("skeletons", {}).get(primary_file, "").splitlines():
            stripped = line.strip()
            if " class " in f" {stripped}" or " def " in f" {stripped}":
                methods.append(stripped)
        if methods:
            return methods
        for entity_name, info in evidence.get("entities", {}).items():
            if info.get("file") == primary_file:
                methods.append(entity_name)
        return methods

    def _collect_small_model_import_points(self, evidence: Dict[str, Any]) -> List[str]:
        import_points = []
        for path, imports in evidence.get("imports", {}).items():
            if imports:
                import_points.append(f"- `{path}` imports: {', '.join(imports[:8])}")
        return import_points

    def _synthesize_for_small_model(
        self, user_input: str, evidence: Dict[str, Any]
    ) -> str:
        """
        Deterministic synthesis path for tiny models.
        Trades stylistic quality for grounding reliability.
        """
        if not evidence.get("file_contents"):
            return ""

        primary_file = self._resolve_small_model_primary_file(evidence)
        if not primary_file:
            return ""

        primary_content = evidence["file_contents"].get(primary_file, "")
        purpose = (
            self._extract_docstring(primary_content) or "No module docstring found."
        )
        methods = self._collect_small_model_methods(evidence, primary_file)
        import_points = self._collect_small_model_import_points(evidence)
        evidence_files = sorted(
            set(evidence.get("codebase_files", []))
            | set(evidence.get("file_contents", {}).keys())
        )
        compliance = self.current_compliance_context or {}
        lines = [
            "## Compliance Context",
            f"- trace_id: {compliance.get('trace_id') or '[UNSET]'}",
            f"- evidence_bundle_id: {compliance.get('evidence_bundle_id') or '[UNSET]'}",
            f"- waiver_id: {compliance.get('waiver_id') or 'none'}",
            f"## Purpose\n{purpose}",
            "## Main Classes & Methods",
        ]
        (
            lines.extend(f"- `{m}`" for m in methods[:50])
            if methods
            else lines.append("- [NOT IN EVIDENCE]")
        )
        lines.append("## Integration Points")
        (
            lines.extend(import_points[:20])
            if import_points
            else lines.append("- [NOT IN EVIDENCE]")
        )
        lines.append("## Evidence Examined")
        lines.extend(f"- `{f}`" for f in evidence_files[:30])
        return "\n".join(lines).strip()

    def _extract_docstring(self, content: str) -> str:
        try:
            module = ast.parse(content)
            doc = ast.get_docstring(module)
            if doc:
                return doc.strip()
        except (SyntaxError, ValueError):
            pass

        match = re.search(r'"""(.*?)"""', content, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _synthesize_with_consistency_check(
        self,
        messages: List[Dict[str, str]],
        evidence: Dict[str, Any],
        dashboard: Optional[Any] = None,
        num_samples: int = 3,
    ) -> str:
        """Generate multiple grounded drafts and keep the one with fewest violations."""
        candidates: List[tuple[int, str]] = []
        sample_count = max(1, num_samples)

        for idx in range(sample_count):
            if dashboard and idx == 0:
                response = self._stream_into_dashboard(messages, dashboard)
            else:
                response = self.agent._stream_response(messages)

            response = self._deduplicate_response(response)
            response = self._deduplicate_response_v2(response)
            response = self._validate_markdown(response)
            response = clean_response(response)
            if re.fullmatch(
                r'\s*\{\s*"name"\s*:\s*"[a-zA-Z0-9_]+"\s*,\s*"arguments"\s*:\s*\{.*\}\s*\}\s*',
                response,
                flags=re.DOTALL,
            ):
                candidates.append((999, response))
                continue
            _, violations = self.hallucination_gate.validate(response, evidence)
            candidates.append((len(violations), response))

        candidates.sort(key=lambda item: item[0])
        best_violations, best_response = candidates[0]
        self.console.print(
            f"  [dim]Consistency check selected response with {best_violations} grounding violations[/dim]"
        )
        return best_response

    def _get_model_capability_tier(self) -> str:
        model = (getattr(self.brain, "model_name", "") or "").lower()
        if any(tag in model for tag in ["tiny", "mini", "1b", "2b"]):
            return "tier_1_minimal"
        if any(tag in model for tag in ["small", "3b", "7b", "8b"]):
            return "tier_2_standard"
        return "tier_3_advanced"

    def _build_grounded_synthesis_prompt(
        self,
        evidence: Dict[str, Any],
        evidence_text: str,
        capability_tier: str,
        subagent_context: str = "",
        strict: bool = False,
    ) -> str:
        entity_registry: List[str] = []
        for file_path, skeleton in evidence.get("skeletons", {}).items():
            for line in str(skeleton).splitlines():
                if " class " in f" {line}" or " def " in f" {line}":
                    entity_registry.append(f"- `{line.strip()}` in `{file_path}`")

        if not entity_registry:
            for name, info in evidence.get("entities", {}).items():
                file_path = info.get("file", "<unknown>")
                line = info.get("line")
                if line:
                    entity_registry.append(f"- `{name}` in `{file_path}:L{line}`")
                else:
                    entity_registry.append(f"- `{name}` in `{file_path}`")

        registry_block = "\n".join(entity_registry[:200]) or "- (no entities found)"
        compliance = self.current_compliance_context or {}
        compliance_block = (
            "## Compliance Context\n"
            f"- trace_id: {compliance.get('trace_id') or '[UNSET]'}\n"
            f"- evidence_bundle_id: {compliance.get('evidence_bundle_id') or '[UNSET]'}\n"
            f"- waiver_id: {compliance.get('waiver_id') or 'none'}"
        )
        strict_rules = (
            """
1. Mention only classes/functions in the entity registry.
2. If a symbol is missing from evidence, write `[NOT IN EVIDENCE]`.
3. Never invent signatures or behavior.
4. Cite file paths for every concrete claim.
"""
            if strict
            else """
1. Mention only classes/functions in the entity registry.
2. Cite file paths for concrete claims.
3. Avoid speculation if evidence is missing.
"""
        )

        if capability_tier == "tier_1_minimal":
            instruction_block = (
                "Return concise factual markdown with sections: Purpose, Main Entities, "
                "Execution Flow, Integration Points, Evidence Examined."
            )
        else:
            instruction_block = (
                "Return structured markdown with explicit citations and a final "
                "`## Evidence Examined` section."
            )

        return f"""You are an accuracy-first code analysis engine.

{compliance_block}

## Entity Registry
{registry_block}

## Grounding Rules
{strict_rules}

## Evidence
{evidence_text}
{subagent_context}

## Response Requirements
{instruction_block}
"""

    def _deduplicate_response_v2(self, response: str) -> str:
        if not response:
            return response

        chunks = re.split(r"(?=^#{1,4}\s+.+$)", response, flags=re.MULTILINE)
        seen_hashes = set()
        deduplicated: List[str] = []

        for chunk in chunks:
            normalized = re.sub(r"\s+", " ", chunk.lower().strip())
            if not normalized:
                continue
            digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            deduplicated.append(chunk)

        return "".join(deduplicated) if deduplicated else response

    def _validate_markdown(self, response: str) -> str:
        if not response:
            return response

        if response.count("```") % 2 != 0:
            response += "\n```"

        lines = response.splitlines()
        cleaned: List[str] = []
        for line in lines:
            if line.strip().lower() == "assistant":
                continue
            cleaned.append(line)
        return "\n".join(cleaned).strip()

    def _response_needs_more_context(self, response: str) -> bool:
        """Detect if the response indicates missing context."""
        indicators = [
            "NEED_MORE_CONTEXT:",
            "without seeing the implementation",
            "cannot determine without",
            "would need to see",
            "requires access to",
            "not shown in the provided",
            "implementation details are not available",
        ]

        response_lower = response.lower()

        # Check for JSON format
        if '"needs_more": true' in response_lower:
            return True

        return any(indicator.lower() in response_lower for indicator in indicators)

    def _extract_missing_entities(self, response: str) -> List[str]:
        """Extract entity names mentioned as missing from the response."""
        import re

        entities = []

        # Pattern 0: JSON object
        try:
            # Simple approach: find first { and last }
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    # Try to extract valid JSON by shrinking from the end
                    # (Handles cases with multiple JSON or trailing text)
                    for i in range(end, start, -1):
                        if response[i] == "}":
                            try:
                                data = json.loads(response[start : i + 1])
                                if isinstance(data, dict):
                                    j_entities = data.get("entities", [])
                                    if isinstance(j_entities, list):
                                        entities.extend([str(e) for e in j_entities])
                                    j_files = data.get("files", [])
                                    if isinstance(j_files, list):
                                        entities.extend([str(f) for f in j_files])
                                    break
                            except json.JSONDecodeError:
                                continue
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass
        except (AttributeError, TypeError):
            pass

        # Pattern 1: NEED_MORE_CONTEXT: EntityName
        pattern1 = re.findall(
            r"NEED_MORE_CONTEXT:\s*([A-Za-z_][A-Za-z0-9_\.]*)", response
        )
        entities.extend(pattern1)

        # Pattern 2: "implementation of ClassName"
        pattern2 = re.findall(r"implementation of\s+([A-Z][A-Za-z0-9_]+)", response)
        entities.extend(pattern2)

        # Pattern 3: "see the FunctionName method"
        pattern3 = re.findall(
            r"see the\s+([a-z_][a-z0-9_]+)\s+(?:method|function)", response
        )
        entities.extend(pattern3)

        return list(set(entities))  # Remove duplicates

    def _generate_action_plan(self, user_input: str, context: Dict[str, Any]) -> str:
        """Generate an action plan for modification requests."""
        # Deep Thinking: Architectural Impact
        if self.enhanced_mode:
            self.thinking_system.think(
                ThinkingType.REASONING,
                "Designing action plan via logical reasoning components...",
            )

        runtime_control = self._refresh_runtime_control(user_input)

        context_summary = ""
        if context.get("summary"):
            context_summary = f"\nContext:\n{context['summary']}"

        system_prompt = self._get_master_system_prompt("action", task=user_input)
        system_prompt += f"""
{context_summary}

### MISSION
Create a world-class, step-by-step action plan for the user's request.

### RUNTIME CONTROL
{self._runtime_control_prompt_directive(runtime_control)}

### PLANNING PROTOCOLS
1. **Dependency Analysis**: Identify all cross-component impacts.
2. **Fact-Checking**: Use Saguaro tools to verify the state of every file you intend to modify.
3. **Atomicity**: Break down complex changes into small, verifiable units of work.
4. **Verification**: Define precise steps to validate success (tests, linting, etc.).

Output ONLY the action plan in this structured format:
1. Files to modify/create:
2. Changes to make (with architectural rationale):
3. Tools to use:
4. Verification steps:"""

        user_prompt = f"""Request: {user_input}

Generate action plan:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        plan = self.agent._stream_response(messages)
        return plan

    def _synthesize_action_result(
        self,
        user_input: str,
        plan: str,
        results: Dict[str, Any],
        dashboard: Optional[Any] = None,
    ) -> str:
        """Synthesize action results into a response."""
        system_prompt = self._get_master_system_prompt("synthesis", task=user_input)
        system_prompt += f"""
### ACTION PLAN
{plan}

### EXECUTION METRICS
{json.dumps(results, indent=2)}

### MISSION
Provide a concise executive summary of the outcomes.
- Confirm successful implementations.
- Highlight any remaining technical debt or issues.
- Provide clear next steps for the user."""

        user_prompt = f"""Original Request: {user_input}

Provide summary:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if dashboard:
            response = self._stream_into_dashboard(messages, dashboard)
        else:
            response = self.agent._stream_response(messages)

        self.history.add_message("assistant", response)
        return response

    def _record_task_memory(
        self, task: str, task_type: str, start_time: float, tools_used: List[str]
    ):
        """Record task in memory for learning."""
        elapsed = time.time() - start_time

        memory = TaskMemory(
            task_id=self.current_task_id,
            task_type=task_type,
            description=task,
            files_modified=list(self.files_edited),
            tools_used=tools_used,
            success=True,  # Assume success if we got here
            execution_time=elapsed,
            iterations=1,
            timestamp=time.time(),
            approach=f"Unified loop: {task_type}",
            key_steps=tools_used,
            difficulties=[],
            tests_passed=True,
            syntax_valid=True,
        )

        self.memory_manager.remember(memory)

    def _format_evidence(
        self, evidence: Dict[str, Any], token_budget: int = 60000
    ) -> str:
        """
        Format evidence dict into readable text within token budget.

        Args:
            evidence: Evidence dictionary with file contents, summaries, etc.
            token_budget: Maximum tokens to use for formatting

        Returns:
            Formatted evidence string that fits within budget
        """
        from core.token_budget import TokenBudgetManager, smart_truncate

        budget_mgr = TokenBudgetManager(token_budget)
        parts: List[str] = []

        self._append_agent_summaries_section(
            evidence, parts, budget_mgr, smart_truncate
        )
        self._append_subagent_analysis_section(
            evidence, parts, budget_mgr, smart_truncate
        )
        self._append_file_list_section(evidence, parts, budget_mgr)
        self._append_workspace_map_section(evidence, parts, budget_mgr)
        self._append_import_map_section(evidence, parts, budget_mgr)
        self._append_dependency_graph_section(evidence, parts, budget_mgr)
        self._append_file_contents_section(evidence, parts, budget_mgr, smart_truncate)
        self._append_web_results_section(evidence, parts, budget_mgr, smart_truncate)

        # Log budget usage
        stats = budget_mgr.get_stats()
        self.console.print(
            f"  [dim]Evidence formatted: {stats['allocated']}/{stats['total_budget']} tokens ({stats['utilization']})[/dim]"
        )
        logger.info(
            json.dumps(
                {
                    "component": "evidence",
                    "event": "format",
                    "metrics": {
                        "allocated": stats["allocated"],
                        "total_budget": stats["total_budget"],
                        "remaining": stats["remaining"],
                        "utilization": stats["utilization"],
                        "items": stats.get("items", []),
                    },
                },
                sort_keys=True,
                ensure_ascii=True,
                default=str,
            )
        )

        return "\n".join(parts)

    def _append_agent_summaries_section(
        self,
        evidence: Dict[str, Any],
        parts: List[str],
        budget_mgr: Any,
        smart_truncate: Any,
    ) -> None:
        """Append high-priority multi-agent summaries within budget."""
        agent_summaries = evidence.get("agent_summaries")
        if not agent_summaries:
            return

        header = "## Multi-Agent Analysis\n"
        header_tokens = budget_mgr.count_tokens(header)
        if not budget_mgr.fits_in_budget(header_tokens):
            return

        parts.append(header)
        budget_mgr.allocate(header_tokens, "multi-agent-header")
        summaries_text = f"Analyzed by {len(agent_summaries)} parallel agents:\n\n"

        for i, summary in enumerate(agent_summaries[:5], 1):
            agent_section = f"### Agent {i} Findings\n{summary}\n\n"
            tokens = budget_mgr.count_tokens(agent_section)
            if budget_mgr.fits_in_budget(tokens):
                summaries_text += agent_section
                budget_mgr.allocate(tokens, f"agent_{i}_summary")
                continue

            truncated = smart_truncate(agent_section, budget_mgr.remaining())
            if len(truncated) > 50:
                summaries_text += truncated
                budget_mgr.allocate(
                    budget_mgr.count_tokens(truncated), f"agent_{i}_truncated"
                )
            break

        parts.append(summaries_text)

    def _append_subagent_analysis_section(
        self,
        evidence: Dict[str, Any],
        parts: List[str],
        budget_mgr: Any,
        smart_truncate: Any,
    ) -> None:
        """Append FileAnalysisSubagent output within remaining budget."""
        analysis = evidence.get("subagent_analysis")
        if not analysis:
            return

        header = "## File Analysis Subagent Findings\n"
        header_tokens = budget_mgr.count_tokens(header)
        if not budget_mgr.fits_in_budget(header_tokens):
            return

        parts.append(header)
        budget_mgr.allocate(header_tokens, "subagent_analysis_header")

        tokens = budget_mgr.count_tokens(analysis)
        if budget_mgr.fits_in_budget(tokens):
            parts.append(analysis)
            budget_mgr.allocate(tokens, "subagent_analysis")
            return

        truncated = smart_truncate(analysis, max(0, budget_mgr.remaining()))
        if truncated.strip():
            parts.append(truncated)
            budget_mgr.allocate(
                budget_mgr.count_tokens(truncated), "subagent_analysis_truncated"
            )

    def _append_file_list_section(
        self, evidence: Dict[str, Any], parts: List[str], budget_mgr: Any
    ) -> None:
        """Append a lightweight list of relevant files."""
        codebase_files = evidence.get("codebase_files")
        if not codebase_files:
            return

        file_list = "## Relevant Files\n" + "\n".join(
            f"- {f}" for f in codebase_files[:15]
        )
        tokens = budget_mgr.count_tokens(file_list)
        if budget_mgr.fits_in_budget(tokens):
            parts.append(file_list)
            budget_mgr.allocate(tokens, "file_list")

    def _append_workspace_map_section(
        self, evidence: Dict[str, Any], parts: List[str], budget_mgr: Any
    ) -> None:
        """Append summarized workspace structure metadata."""
        workspace_map = evidence.get("workspace_map") or {}
        if not workspace_map or budget_mgr.remaining() <= 800:
            return

        workspace_lines = ["## Workspace Structure"]
        top_dirs = workspace_map.get("top_dirs", []) or []
        key_paths = workspace_map.get("key_paths", []) or []
        if top_dirs:
            workspace_lines.append("Top directories:")
            workspace_lines.extend(f"- {d}" for d in top_dirs[:15])
        if key_paths:
            workspace_lines.append("Key paths:")
            workspace_lines.extend(f"- {p}" for p in key_paths[:20])

        tree_snippet = workspace_map.get("tree_snippet", "")
        if tree_snippet:
            workspace_lines.append("Tree snippet:")
            workspace_lines.append(tree_snippet[:1200])

        workspace_block = "\n".join(workspace_lines)
        tokens = budget_mgr.count_tokens(workspace_block)
        if budget_mgr.fits_in_budget(tokens):
            parts.append(workspace_block)
            budget_mgr.allocate(tokens, "workspace_map")

    def _append_import_map_section(
        self, evidence: Dict[str, Any], parts: List[str], budget_mgr: Any
    ) -> None:
        """Append condensed import relationships for top files."""
        imports = evidence.get("imports")
        if not imports or budget_mgr.remaining() <= 1000:
            return

        import_lines = ["## Import Map"]
        for file_path, file_imports in list(imports.items())[:12]:
            short_imports = ", ".join(file_imports[:10]) if file_imports else "(none)"
            import_lines.append(f"- {file_path}: {short_imports}")

        import_block = "\n".join(import_lines)
        tokens = budget_mgr.count_tokens(import_block)
        if budget_mgr.fits_in_budget(tokens):
            parts.append(import_block)
            budget_mgr.allocate(tokens, "import_map")

    def _append_dependency_graph_section(
        self, evidence: Dict[str, Any], parts: List[str], budget_mgr: Any
    ) -> None:
        """Append top dependency edges when available."""
        dependency_graph = evidence.get("dependency_graph")
        if not dependency_graph or budget_mgr.remaining() <= 1000:
            return

        edges = dependency_graph.get("edges", {})
        dep_lines = ["## Dependency Graph"]
        for source, targets in list(edges.items())[:12]:
            if not targets:
                continue
            dep_lines.append(f"- {source} -> {', '.join(targets[:6])}")

        dep_block = "\n".join(dep_lines)
        tokens = budget_mgr.count_tokens(dep_block)
        if budget_mgr.fits_in_budget(tokens):
            parts.append(dep_block)
            budget_mgr.allocate(tokens, "dependency_graph")

    def _append_file_contents_section(
        self,
        evidence: Dict[str, Any],
        parts: List[str],
        budget_mgr: Any,
        smart_truncate: Any,
    ) -> None:
        """Append prioritized file contents, honoring the remaining budget."""
        if not (evidence.get("file_contents") and budget_mgr.remaining() > 5000):
            return

        parts.append("\n## File Contents")
        ranked_files = self._rank_files_by_relevance(evidence)
        files_added = self._append_ranked_file_sections(
            ranked_files, parts, budget_mgr, smart_truncate
        )
        if files_added < len(ranked_files):
            parts.append(
                f"\n(Showing {files_added}/{len(ranked_files)} files due to token budget)"
            )

    def _append_ranked_file_sections(
        self,
        ranked_files: List[tuple],
        parts: List[str],
        budget_mgr: Any,
        smart_truncate: Any,
    ) -> int:
        """Append full or truncated skeleton views for ranked files."""
        files_added = 0
        for file_path, content in ranked_files:
            if self._try_append_full_file_content(
                file_path, content, parts, budget_mgr
            ):
                files_added += 1
            elif self._try_append_truncated_skeleton(
                file_path, content, parts, budget_mgr, smart_truncate
            ):
                files_added += 1

            if budget_mgr.remaining() < 1000:
                break

        return files_added

    def _try_append_full_file_content(
        self, file_path: str, content: str, parts: List[str], budget_mgr: Any
    ) -> bool:
        """Try appending full file content block."""
        file_section = f"\n### {file_path}\n```\n{content}\n```"
        tokens = budget_mgr.count_tokens(file_section)
        if not budget_mgr.fits_in_budget(tokens):
            return False

        parts.append(file_section)
        budget_mgr.allocate(tokens, f"file_{file_path}")
        return True

    def _try_append_truncated_skeleton(
        self,
        file_path: str,
        content: str,
        parts: List[str],
        budget_mgr: Any,
        smart_truncate: Any,
    ) -> bool:
        """Try appending truncated skeleton/entity content when full content won't fit."""
        if "[SKELETON]" not in content and "[ENTITY" not in content:
            return False

        truncated = smart_truncate(content, budget_mgr.remaining() - 100)
        file_section = f"\n### {file_path} [TRUNCATED]\n```\n{truncated}\n```"
        tokens = budget_mgr.count_tokens(file_section)
        if tokens >= budget_mgr.remaining():
            return False

        parts.append(file_section)
        budget_mgr.allocate(tokens, f"file_{file_path}_truncated")
        return True

    def _append_web_results_section(
        self,
        evidence: Dict[str, Any],
        parts: List[str],
        budget_mgr: Any,
        smart_truncate: Any,
    ) -> None:
        """Append web search results last, as lowest-priority context."""
        web_results = evidence.get("web_results")
        if not web_results or budget_mgr.remaining() <= 2000:
            return

        parts.append("\n## Web Search Results")
        for result in web_results:
            result_str = str(result)
            tokens = budget_mgr.count_tokens(result_str)
            if budget_mgr.fits_in_budget(tokens):
                parts.append(result_str)
                budget_mgr.allocate(tokens, "web_result")
                continue

            truncated = smart_truncate(result_str, budget_mgr.remaining())
            if len(truncated) > 100:
                parts.append(truncated)
                budget_mgr.allocate(
                    budget_mgr.count_tokens(truncated), "web_result_truncated"
                )
            break

    def _rank_files_by_relevance(self, evidence: Dict[str, Any]) -> List[tuple]:
        """
        Rank files by relevance for prioritized loading.

        Uses heuristics:
        - Files with key entities get priority
        - Files mentioned in subagent analysis get priority
        - Shorter files preferred (more context-efficient)
        - Full content preferred over skeletons

        Args:
            evidence: Evidence dictionary

        Returns:
            List of (file_path, content) tuples sorted by relevance
        """
        file_contents = evidence.get("file_contents", {})
        key_files = set(evidence.get("key_files", []))
        subagent_analysis = evidence.get("subagent_analysis", "")

        scored_files = []

        for file_path, content in file_contents.items():
            score = 0.0

            # Priority 1: Key files from subagent
            if file_path in key_files:
                score += 10.0

            # Priority 2: Mentioned in subagent analysis
            if file_path in subagent_analysis:
                score += 5.0

            # Priority 3: Full content (not skeleton) is more valuable
            if not ("[SKELETON]" in content[:50] or "[ENTITY" in content[:50]):
                score += 3.0

            # Priority 4: Shorter files are more context-efficient
            content_len = len(content)
            if content_len < 2000:
                score += 2.0
            elif content_len < 5000:
                score += 1.0

            # Priority 5: Core/important directories
            if any(pattern in file_path for pattern in ["/core/", "/src/", "/main"]):
                score += 1.5

            scored_files.append((score, file_path, content))

        # Sort by score (highest first)
        scored_files.sort(reverse=True, key=lambda x: x[0])

        # Return (file_path, content) tuples
        return [(fp, content) for _, fp, content in scored_files]

    def _iterative_search(self, query: str, max_rounds: int = 2) -> set:
        """
        Iterative semantic search like Claude Code.

        Rounds:
        1. Initial semantic search on query
        2. Extract imports/references from found files
        3. Search for those referenced entities
        4. Repeat to build comprehensive context
        """
        all_files = set()
        search_terms = {query}

        for round_num in range(max_rounds):
            self.console.print(
                f"    [dim]Round {round_num + 1}: searching for {len(search_terms)} terms...[/dim]"
            )
            new_files = self._search_files_for_terms(search_terms, k=5)
            newly_found = new_files - all_files
            self.console.print(f"    [dim]  Found {len(newly_found)} new files[/dim]")
            all_files.update(new_files)

            if round_num >= max_rounds - 1:
                continue

            search_terms = self._extract_references_from_files(list(newly_found)[:5])
            if not search_terms:
                break

        return all_files

    def _search_files_for_terms(self, search_terms: set, k: int) -> set:
        """Search for all terms and merge discovered files."""
        new_files = set()
        for term in search_terms:
            new_files.update(self._search_files_for_term(term, k=k))
        return new_files

    def _search_files_for_term(self, term: str, k: int) -> set:
        """Search one term with lightweight keyword expansion."""
        queries = [term]
        key_terms = self._extract_key_terms(term)
        if key_terms:
            queries.extend(key_terms[:2])
        search_results = self.parallel_search.multi_query_search(queries, k=k)
        return self._collect_files_from_search_results(search_results)

    def _collect_files_from_search_results(
        self, search_results: Dict[str, List[str]]
    ) -> set:
        """Collect file paths from multi-query search responses."""
        files = set()
        for search_files in search_results.values():
            files.update(search_files)
        return files

    def _iterative_search_with_lexical_fallback(
        self, query: str, all_files: set
    ) -> None:
        raise RuntimeError(
            "SAGUARO_STRICT_FALLBACK_DISABLED: iterative lexical fallback is disabled."
        )

    def _collect_files_from_grep_hits(self, term: str, all_files: set) -> None:
        raise RuntimeError(
            "SAGUARO_STRICT_FALLBACK_DISABLED: grep_search fallback is disabled."
        )

    def _collect_files_from_name_hits(self, term: str, all_files: set) -> None:
        raise RuntimeError(
            "SAGUARO_STRICT_FALLBACK_DISABLED: find_by_name fallback is disabled."
        )

    def _extract_references_from_files(self, file_paths: List[str]) -> set:
        """Extract imports and references from files to guide next search round."""
        references = set()
        import re

        for file_path in file_paths[:3]:  # Only check top 3 to avoid overhead
            try:
                # Quick read (first 100 lines)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = [next(f) for _ in range(100) if f]

                content = "".join(lines)

                # Extract imports
                # Python: from X import Y, import X
                python_imports = re.findall(
                    r"(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_.]*)", content
                )
                references.update(python_imports[:3])

                # Extract class names (potential search terms)
                class_names = re.findall(r"class\s+([A-Z][a-zA-Z0-9_]*)", content)
                references.update(class_names[:2])

                # Extract function calls that might be references
                # Look for capitalized names (likely classes/modules)
                caps_words = re.findall(r"\b([A-Z][a-zA-Z0-9_]{3,})\b", content)
                references.update(caps_words[:2])

            except (OSError, UnicodeDecodeError, StopIteration):
                pass

        # Filter out generic terms
        filtered = {
            ref
            for ref in references
            if len(ref) > 3
            and ref.lower()
            not in {"self", "true", "false", "none", "type", "dict", "list"}
        }

        return filtered

    def _identify_critical_files_fast(
        self, query: str, candidate_files: List[str]
    ) -> List[str]:
        """
        Quickly identify which files are most critical for the query.

        Uses heuristics:
        - Files mentioned in query
        - Files matching key terms
        - Files with 'core', 'main', 'bridge', 'client' in name
        """

        critical = []
        query.lower()

        # Extract key terms from query
        key_terms = self._extract_key_terms(query)

        for file_path in candidate_files[:15]:
            score = 0
            file_lower = file_path.lower()

            # Exact mentions in query
            if any(term in file_lower for term in key_terms):
                score += 10

            # Important file name patterns
            important_patterns = [
                "bridge",
                "client",
                "core",
                "main",
                "base",
                "native",
                "engine",
            ]
            if any(pattern in file_lower for pattern in important_patterns):
                score += 5

            # Files in top-level or important directories
            if "/core/" in file_path or "/src/" in file_path:
                score += 3

            # Python implementation files (not tests)
            if file_path.endswith(".py"):
                if "test" not in file_lower:
                    score += 3
                else:
                    score += 1

            if score >= 4:  # Lowered threshold to be more inclusive
                critical.append((score, file_path))

        # If zero files found as critical, pick top 3 candidates to ensure context
        if not critical:
            for file_path in candidate_files[:3]:
                critical.append((1, file_path))

        # Sort by score and return top files
        critical.sort(reverse=True)
        return [fp for _, fp in critical[:10]]  # Increased limit to 10

    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key technical terms from a query."""
        # Simple keyword extraction
        import re

        # Remove common words
        stop_words = {
            "how",
            "does",
            "the",
            "is",
            "what",
            "where",
            "why",
            "when",
            "can",
            "you",
            "tell",
            "me",
            "about",
            "a",
            "an",
            "in",
            "to",
            "for",
            "of",
            "and",
            "or",
            "this",
            "that",
            "it",
            "with",
            "explain",
        }

        words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", query.lower())
        key_terms = [w for w in words if w not in stop_words and len(w) > 2]

        return key_terms[:5]

    def _deduplicate_response(self, response: str) -> str:
        """
        Deduplicate response if model generated duplicate content.

        Sometimes models repeat their entire output. This detects and removes
        exact duplicates by looking for repeated sections.
        """
        if not response:
            return response

        response = self._strip_malformed_role_markers(response)
        response = self._deduplicate_thinking_blocks(response)
        response = self._deduplicate_assistant_sections(response)
        response = self._deduplicate_duplicate_halves(response)
        response = self._deduplicate_paragraphs(response)
        response = self._strip_repeated_markdown_headers(response)

        if len(response) > 500:
            response = self._sliding_window_deduplicate(response)

        return response

    def _strip_malformed_role_markers(self, response: str) -> str:
        """Remove malformed role transition markers emitted by some models."""
        response = re.sub(
            r"\banswer<\|end_of_role\|", "", response, flags=re.IGNORECASE
        )
        response = re.sub(r"<\|end_of_role\|", "", response)
        return re.sub(
            r"<\|start_header_id\|.*?<\|end_header_id\|", "", response, flags=re.DOTALL
        )

    def _deduplicate_thinking_blocks(self, response: str) -> str:
        """Remove duplicate <thinking> blocks while preserving order."""
        thinking_pattern = r"<thinking[^>]*>.*?</thinking>"
        thinking_blocks = re.findall(thinking_pattern, response, re.DOTALL)
        if len(thinking_blocks) < 2:
            return response

        unique_blocks = []
        seen = set()
        removed_count = 0
        for block in thinking_blocks:
            if block not in seen:
                seen.add(block)
                unique_blocks.append(block)
            else:
                removed_count += 1

        if removed_count == 0:
            return response

        response_parts = re.split(thinking_pattern, response, flags=re.DOTALL)
        rebuilt = []
        block_idx = 0
        for part in response_parts:
            rebuilt.append(part)
            if block_idx < len(unique_blocks):
                rebuilt.append(unique_blocks[block_idx])
                block_idx += 1

        self.console.print(
            f"  [yellow]⚠ Detected and removed {removed_count} duplicate thinking blocks[/yellow]"
        )
        return "".join(rebuilt)

    def _deduplicate_assistant_sections(self, response: str) -> str:
        """Remove duplicated trailing assistant sections."""
        assistant_parts = re.split(r"\n\nassistant\n", response)
        if len(assistant_parts) < 3:
            return response

        last = assistant_parts[-1].strip()
        second_last = assistant_parts[-2].strip()
        if not last or not second_last:
            return response

        if last == second_last:
            self.console.print(
                "  [yellow]⚠ Detected and removed duplicate assistant response[/yellow]"
            )
            return "\n\nassistant\n".join(assistant_parts[:-1])

        matches = sum(c1 == c2 for c1, c2 in zip(last, second_last))
        similarity = matches / max(len(last), len(second_last))
        if similarity > 0.85:
            self.console.print(
                f"  [yellow]⚠ Detected and removed similar duplicate response ({similarity:.1%} match)[/yellow]"
            )
            return "\n\nassistant\n".join(assistant_parts[:-1])

        return response

    def _deduplicate_duplicate_halves(self, response: str) -> str:
        """Trim second half when response is effectively duplicated."""
        lines = response.split("\n")
        if len(lines) <= 10:
            return response

        mid = len(lines) // 2
        first_half_lines = lines[:mid]
        second_half_lines = (
            lines[mid : mid * 2] if len(lines) >= mid * 2 else lines[mid:]
        )
        if len(first_half_lines) != len(second_half_lines):
            return response

        matches = sum(
            1
            for l1, l2 in zip(first_half_lines, second_half_lines)
            if l1.strip() == l2.strip()
        )
        if matches / len(first_half_lines) > 0.9:
            self.console.print(
                "  [yellow]⚠ Detected and removed duplicate half of response[/yellow]"
            )
            return "\n".join(first_half_lines)

        return response

    def _deduplicate_paragraphs(self, response: str) -> str:
        """Remove repeated paragraphs, keeping first occurrence."""
        paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
        if len(paragraphs) <= 2:
            return response

        last_para = paragraphs[-1]
        if len(last_para) <= 50:
            return response

        occurrences = sum(
            1
            for p in paragraphs
            if p == last_para
            or (len(p) > 50 and self._paragraph_similarity(p, last_para) > 0.9)
        )
        if occurrences <= 1:
            return response

        seen = set()
        unique_paragraphs = []
        for p in paragraphs:
            signature = p[:100] if len(p) > 100 else p
            if signature not in seen:
                seen.add(signature)
                unique_paragraphs.append(p)
            elif len(unique_paragraphs) > 0:
                self.console.print(
                    "  [yellow]⚠ Detected and removed duplicate paragraph[/yellow]"
                )

        return "\n\n".join(unique_paragraphs)

    def _strip_repeated_markdown_headers(self, response: str) -> str:
        """Strip repeated markdown headers such as duplicate '# Summary' lines."""
        seen_headers = set()
        unique_lines = []
        for line in response.split("\n"):
            if re.match(r"^#+\s+.*$", line):
                clean_header = line.strip().lower()
                if clean_header in seen_headers:
                    if hasattr(self, "console"):
                        self.console.print(
                            f"  [yellow]⚠ Stripped repeated header: {line}[/yellow]"
                        )
                    continue
                seen_headers.add(clean_header)
            unique_lines.append(line)
        return "\n".join(unique_lines)

    def _sliding_window_deduplicate(self, text: str, window_size: int = 150) -> str:
        """Detect and remove repeated sequences using a sliding window."""
        if len(text) < window_size * 2:
            return text

        # Check for immediate repetitions of windows
        pos = 0
        while pos + window_size * 2 <= len(text):
            window = text[pos : pos + window_size]
            next_window = text[pos + window_size : pos + window_size * 2]

            # Use fuzzy matching for windows
            if self._paragraph_similarity(window, next_window) > 0.9:
                # Found a repetition, jump over it
                text = text[: pos + window_size] + text[pos + window_size * 2 :]
                if hasattr(self, "console"):
                    self.console.print(
                        "  [yellow]⚠ Sliding window detected repetition, deduplicating...[/yellow]"
                    )
                # Don't increment pos, check again
                continue
            pos += 1
        return text

    def _paragraph_similarity(self, p1: str, p2: str) -> float:
        """Calculate similarity between two paragraphs."""
        if len(p1) == 0 or len(p2) == 0:
            return 0.0

        # Simple character-level similarity
        matches = sum(c1 == c2 for c1, c2 in zip(p1, p2))
        return matches / max(len(p1), len(p2))

    def _needs_web_search(self, query: str) -> bool:
        """Determine if web search is needed."""
        query_lower = query.lower()

        web_keywords = [
            "latest",
            "recent",
            "news",
            "current",
            "documentation",
            "tutorial",
            "example",
            "best practice",
            "stackoverflow",
            "github",
            "npm",
            "pypi",
            "package",
            "library",
            "framework",
        ]

        return any(kw in query_lower for kw in web_keywords)

    def _do_web_search(self, query: str, evidence: Dict[str, Any]):
        """Execute web search."""
        self.console.print("  [dim]→ Web search...[/dim]")
        try:
            results = self._execute_tool("web_search", {"query": query})
            if results and not str(results).startswith("Error"):
                evidence["web_results"].append(results)
                self.console.print("  [green]✓ Web search completed[/green]")
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            evidence["errors"].append(f"Web search failed: {e}")

    def _extract_direct_file_path(self, query: str) -> Optional[str]:
        """Check if the query contains a direct, existing file path."""
        match = re.search(r"([./a-zA-Z0-9_-]+\.(?:py|cc|cpp|h|js|ts|md))", query)
        if match:
            candidate = match.group(1)
            candidates = [
                candidate,
                os.path.join(self.saguaro.root_dir, candidate),
                os.path.join(os.getcwd(), candidate),
            ]
            for path in candidates:
                if os.path.exists(path):
                    rel = (
                        os.path.relpath(path, self.saguaro.root_dir)
                        if os.path.isabs(path)
                        else path
                    )
                    self.console.print(
                        f"  [green]✓ Direct file path detected:[/green] [bold]{rel}[/bold]"
                    )
                    return rel
        return None

    def _collect_streamed_text(
        self, messages: List[Dict[str, str]], max_tokens: int
    ) -> str:
        response = ""
        for chunk in self.brain.stream_chat(
            messages, max_tokens=max_tokens, temperature=0.0
        ):
            response += chunk
        return response

    def _extract_json_object_with_key(
        self, response: str, required_key: str
    ) -> Optional[Dict[str, Any]]:
        selected = None
        for match in re.finditer(r"\{.*?\}", response, re.DOTALL):
            potential = self._safe_json_loads(match.group(0))
            if not isinstance(potential, dict) or required_key not in potential:
                continue
            selected = potential
            if potential.get("entities") or potential.get("files"):
                return potential
        if selected is not None:
            return selected
        return self._extract_balanced_json_object(response, required_key)

    def _extract_balanced_json_object(
        self, response: str, required_key: str
    ) -> Optional[Dict[str, Any]]:
        start = response.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(response)):
            if response[i] == "{":
                depth += 1
            elif response[i] == "}":
                depth -= 1
            if depth != 0:
                continue
            potential = self._safe_json_loads(response[start : i + 1])
            if isinstance(potential, dict) and required_key in potential:
                return potential
            break
        return None

    def _extract_json_array(self, response: str) -> List[Any]:
        for match in re.finditer(r"\[.*?\]", response, re.DOTALL):
            potential = self._safe_json_loads(match.group(0))
            if isinstance(potential, list):
                return potential
        return self._extract_balanced_json_array(response)

    def _extract_balanced_json_array(self, response: str) -> List[Any]:
        start = response.find("[")
        if start == -1:
            return []
        depth = 0
        for i in range(start, len(response)):
            if response[i] == "[":
                depth += 1
            elif response[i] == "]":
                depth -= 1
            if depth != 0:
                continue
            potential = self._safe_json_loads(response[start : i + 1])
            return potential if isinstance(potential, list) else []
        return []

    def _safe_json_loads(self, content: str) -> Optional[Any]:
        try:
            return json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    def _build_context_expansion_messages(
        self, query: str, context_summary: str
    ) -> List[Dict[str, str]]:
        prompt = f"""You are analyzing a codebase to answer: "{query}"

Current context available:
{context_summary}

Do you need more specific information to answer this question comprehensively?

If YES, respond with a JSON object:
{{
  "needs_more": true,
  "entities": ["ClassName", "function_name", "MODULE_NAME"],
  "files": ["path/to/specific/file.py"],
  "reason": "brief explanation"
}}

If NO (you have enough context), respond:
{{"needs_more": false}}

Respond ONLY with valid JSON, no other text."""
        return [
            {
                "role": "system",
                "content": "You are a code analysis assistant. Output ONLY valid JSON.",
            },
            {"role": "user", "content": prompt},
        ]

    def _request_context_expansion(
        self, query: str, evidence: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        context_summary = self._build_context_summary(evidence)
        messages = self._build_context_expansion_messages(query, context_summary)
        response = self._collect_streamed_text(messages, max_tokens=500)
        try:
            return self._extract_json_object_with_key(response, "needs_more")
        except (RuntimeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.console.print(
                f"  [yellow]⚠ Context expansion parsing failed: {exc}[/yellow]"
            )
            return None

    def _load_requested_entity(
        self, entity: str, loaded_entities: set, evidence: Dict[str, Any]
    ) -> None:
        if entity in loaded_entities:
            return
        try:
            search_result = self.semantic_engine.get_context_for_objective(
                f"definition of {entity}"
            )
            if not search_result:
                return
            for file_path in search_result[:2]:
                try:
                    slice_target = f"{file_path}.{entity}"
                    sliced = self.saguaro_tools.slice(slice_target)
                    if not sliced or sliced.startswith("Error"):
                        continue
                    evidence["file_contents"][
                        f"{file_path}::{entity}"
                    ] = f"[ENTITY SLICE]\n{sliced}"
                    loaded_entities.add(entity)
                    self.console.print(
                        f"    [green]✓ Loaded {entity} from {file_path}[/green]"
                    )
                    return
                except (RuntimeError, ValueError, TypeError, OSError):
                    continue
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            self.console.print(
                f"    [yellow]⚠ Could not load entity {entity}: {exc}[/yellow]"
            )

    def _load_requested_file(self, file_path: str, evidence: Dict[str, Any]) -> None:
        if file_path in evidence["file_contents"]:
            return
        try:
            content = self._execute_tool("read_file", {"path": file_path})
            if not content or content.startswith("Error") or len(content) >= 15000:
                return
            evidence["file_contents"][file_path] = content
            self.files_read.add(file_path)
            self.console.print(f"    [green]✓ Loaded {file_path}[/green]")
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            self.console.print(
                f"    [yellow]⚠ Could not load {file_path}: {exc}[/yellow]"
            )

    def _apply_context_expansion_request(
        self,
        request: Dict[str, Any],
        round_num: int,
        loaded_entities: set,
        evidence: Dict[str, Any],
    ) -> None:
        entities = request.get("entities", [])
        files = request.get("files", [])
        reason = request.get("reason", "")
        self.console.print(
            f"  [cyan]→ Round {round_num + 1}: Loading {len(entities)} entities, {len(files)} files[/cyan]"
        )
        if reason:
            self.console.print(f"    [dim]Reason: {reason}[/dim]")
        for entity in entities:
            self._load_requested_entity(entity, loaded_entities, evidence)
        for file_path in files:
            self._load_requested_file(file_path, evidence)

    def _intelligent_context_expansion(
        self, query: str, evidence: Dict[str, Any], max_rounds: int = 2
    ):
        """
        Intelligent context expansion like Claude Code, but using Saguaro's semantic capabilities.

        Workflow:
        1. Show model the current context summary
        2. Ask: "Do you need more specific files/entities?"
        3. Parse response for entity names, file paths, class names
        4. Use Saguaro to find and load those entities precisely
        5. Repeat until satisfied or max_rounds
        """
        loaded_entities = set()
        for round_num in range(max_rounds):
            request = self._request_context_expansion(query, evidence)
            if not request:
                break
            if not request.get("needs_more", False):
                self.console.print("  [green]✓ Model has sufficient context[/green]")
                break
            self._apply_context_expansion_request(
                request=request,
                round_num=round_num,
                loaded_entities=loaded_entities,
                evidence=evidence,
            )

    def _build_context_summary(self, evidence: Dict[str, Any]) -> str:
        """Build a concise summary of currently loaded context."""
        summary_parts = []

        # Files loaded
        if evidence.get("codebase_files"):
            summary_parts.append(
                f"Found {len(evidence['codebase_files'])} relevant files:"
            )
            for f in evidence["codebase_files"][:10]:
                summary_parts.append(f"  - {f}")

        # Content details
        if evidence.get("file_contents"):
            summary_parts.append(
                f"\nLoaded content for {len(evidence['file_contents'])} files:"
            )
            for path, content in list(evidence["file_contents"].items())[:5]:
                content_type = (
                    "SKELETON"
                    if "[SKELETON]" in content[:50]
                    else "FULL" if "[ENTITY" not in content[:50] else "ENTITY"
                )
                lines = len(content.split("\n"))
                summary_parts.append(f"  - {path} ({content_type}, {lines} lines)")

        return "\n".join(summary_parts) if summary_parts else "No context loaded yet."

    def _build_dynamic_tooling_messages(
        self, query: str, evidence: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        evidence_summary = (
            f"Files found so far: {len(evidence.get('codebase_files', []))}"
        )
        pressure_note = self._context_pressure_guidance(query)
        prompt = f"""You are determining additional search steps.
Query: "{query}"
Current Status: {evidence_summary}

Do you need to run additional Saguaro tools to find exact code matches or files?
Available tools:
- saguaro_query(query, k): Semantic repository discovery
- skeleton(path): File structure view
- slice(target): Focused symbol extraction
- read_file(path): Read specific relevant file

All tool calls MUST include `_context_updates`.
Pass [] if nothing should be compressed.
Compress only stale `[tcN]` results.
{pressure_note}

Return a JSON list of tool calls if needed, or [] if you have enough info.
Example: [{{"tool": "saguaro_query", "args": {{"query": "class Foo implementation", "k": 5, "_context_updates": []}}}}]
"""
        return [
            {
                "role": "system",
                "content": "You are a tool usage optimizer. Output ONLY JSON.",
            },
            {"role": "user", "content": prompt},
        ]

    def _record_dynamic_tool_success(
        self, result: Dict[str, Any], evidence: Dict[str, Any]
    ) -> None:
        evidence["search_results"].append((result["tool"], result["result"]))
        self._record_tool_result(
            result["tool"], result.get("args", {}), str(result["result"])
        )
        if result["tool"] != "read_file":
            return
        path = result["args"].get("file_path") or result["args"].get("path")
        if path:
            evidence["file_contents"][path] = result["result"]

    def _record_dynamic_tool_failure(
        self, result: Dict[str, Any], evidence: Dict[str, Any]
    ) -> None:
        evidence["errors"].append(f"Dynamic tool error: {result.get('error')}")
        self._record_tool_result(
            result["tool"], result.get("args", {}), str(result.get("error"))
        )

    def _execute_dynamic_tool_calls(
        self, tool_calls: List[Dict[str, Any]], evidence: Dict[str, Any]
    ) -> None:
        for tool_call in tool_calls:
            ensure_context_updates_arg(tool_call.setdefault("args", {}))
        self.console.print(
            f"  [dim]→ Executing {len(tool_calls)} dynamic tool calls...[/dim]"
        )
        results = self.parallel_executor.execute_tools(tool_calls)
        self._apply_context_updates_from_results(results)
        for result in results:
            if result["success"]:
                self._record_dynamic_tool_success(result, evidence)
            else:
                self._record_dynamic_tool_failure(result, evidence)
        self.console.print("  [green]✓ Dynamic tooling finished[/green]")

    def _dynamic_tooling_loop(self, query: str, evidence: Dict[str, Any]):
        """
        Ask the model if it needs to run additional Saguaro tools
        to fill usage gaps.
        """
        messages = self._build_dynamic_tooling_messages(query, evidence)
        response = self._collect_streamed_text(messages, max_tokens=1000)
        try:
            tool_calls = self._extract_json_array(response)
            if tool_calls:
                self._execute_dynamic_tool_calls(tool_calls, evidence)
        except (RuntimeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.console.print(f"  [red]! Dynamic tooling failed: {exc}[/red]")

    def _calculate_complexity_score(
        self,
        query: str,
        num_files: int,
        skeletons: Dict[str, str],
        search_rounds: int,
        question_type: str,
    ) -> float:
        """
        Calculate dynamic complexity score for delegation decisions.

        Formula:
        - Base: 1.0 per file
        - Volume: 1.0 per 500 skeleton tokens
        - Iteration: 2.0 per search round
        - Intent: +5.0 for architecture/how-it-works
        """
        from core.token_budget import TokenBudgetManager

        budget_mgr = TokenBudgetManager(0)  # Only using for counting

        # 1. Base score (files)
        score = float(num_files)

        # 2. Volume weight (token density)
        total_tokens = sum(budget_mgr.count_tokens(s) for s in skeletons.values())
        score += total_tokens / 500.0

        # 3. Iteration weight
        score += search_rounds * 2.0

        # 4. Intent boost
        if (
            question_type == "architecture"
            or "how does" in query.lower()
            or "explain" in query.lower()
        ):
            score += 5.0

        return score

    def get_stats(self) -> Dict[str, Any]:
        """Get unified loop statistics."""
        return {
            "enhanced_mode": self.enhanced_mode,
            "files_read": len(self.files_read),
            "files_edited": len(self.files_edited),
            "task_memories": (
                len(self.memory_manager.memories)
                if hasattr(self.memory_manager, "memories")
                else 0
            ),
            "session_duration": time.time() - self.session_start,
        }
