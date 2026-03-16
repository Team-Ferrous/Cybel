import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Ensure local packages (`core`, `saguaro`, etc.) resolve when running
# `python cli/repl.py` directly from the repository root.
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from prompt_toolkit import HTML, PromptSession
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from cli.command_registry import CommandRegistry
from cli.commands.agent import (
    AgentCommand,
    CampaignCommand,
    CollaborateCommand,
    OwnershipCommand,
    PeersCommand,
)
from cli.commands.agent import (
    ThinkingCommand as BaseThinkingCommand,
)
from cli.commands.basic import ClearCommand, ExitCommand, HelpCommand, SaguaroCommand
from cli.commands.codebase_analysis import CodebaseAnalysisCommand
from cli.commands.config import ModeCommand, SettingsCommand
from cli.commands.dare import DareCommand
from cli.commands.dream import DreamCommand
from cli.commands.enhanced_tools import EnhancedToolsCommand, ToolMemoryCommand
from cli.commands.features import (
    CheckpointCommand,
    ModelCommand,
    ResetCommand,
    SessionCommand,
    SkillsCommand,
    StatsCommand,
    TimelineCommand,
    TreeCommand,
)
from cli.commands.manager import ManagerCommand
from cli.commands.memory import MemoryCommand
from cli.commands.models import ModelsCommand
from cli.commands.repl_commands import CoconutCommand, LogsCommand
from cli.commands.subagents import (
    AnalyzeCommand,
    DebugCommand,
    ImplementCommand,
    ResearchCommand,
    TestCommand,
)
from cli.commands.subagents import (
    PlanCommand as SubagentPlanCommand,
)
from cli.commands.swarm import SwarmCommand
from cli.commands.sync import SyncCommand
from cli.commands.thinking import (
    ApproveCommand,
    ArchiveCommand,
    ArchivesCommand,
    ArtifactsCommand,
    ChainsCommand,
    LoopCommand,
    PlanCommand,
    TaskCommand,
    VerifyCommand,
)
from cli.commands.unwired import UnwiredCommand
from cli.commands.wizard import WizardCommand
from cli.renderer import CLIRenderer, get_bottom_toolbar_text
from config.settings import (
    AGENTIC_THINKING,
    COLLABORATION_CONFIG,
    MASTER_MODEL,
    OWNERSHIP_CONFIG,
)
from core.agent import BaseAgent
from core.approval import PolicyProfile
from core.context import ContextManager
from core.env_manager import EnvironmentManager
from core.memory.project_memory import ProjectMemory
from core.orchestrator.loop_orchestrator import LoopOrchestrator
from core.utils.logger import (
    emit_structured_event,
    get_active_log_file,
    get_logger,
    setup_logging,
)
from domains.code_intelligence.saguaro_substrate import SaguaroSubstrate
from domains.task_execution.enhanced_loop import EnhancedAgenticLoop
from saguaro.sentinel.remediation import (
    format_repl_startup_toolchain_summary,
    run_repl_startup_toolchain_check,
)
from audit.runner.suite_certification import (
    assess_runtime_tuning,
    bootstrap_runtime_tuning,
    ensure_runtime_affinity,
    format_runtime_tuning_summary,
    has_benchmark_evidence,
    mark_runtime_tuning_deferred,
    resolve_runtime_tuning_bootstrap_policy,
    should_bootstrap_runtime_tuning,
)
from saguaro.indexing.auto_scaler import calibrate_runtime_profile

logger = get_logger(__name__)

EXIT_SUCCESS = 0
EXIT_POLICY_DENIED = 10
EXIT_TOOL_FAILURE = 20
EXIT_VERIFICATION_FAILED = 30
EXIT_TIMEOUT = 40
EXIT_INTERNAL_ERROR = 50


def _run_interactive_toolchain_check() -> None:
    report = run_repl_startup_toolchain_check(root_dir)
    lines = format_repl_startup_toolchain_summary(report)
    if report.get("cached") and not report.get("missing_profiles"):
        return
    for line in lines:
        print(line)


def _short_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=1.5,
        )
        return out.strip()
    except Exception:
        return "unknown"


class AgentREPL(BaseAgent):
    def __init__(self, agent_instance: BaseAgent | None = None):
        # Initialize Renderer first with default theme if needed, or custom
        self.renderer = CLIRenderer(None)  # uses internal console

        if agent_instance:
            # If an existing agent instance is provided, pass its serialized state
            # to BaseAgent's __init__ for proper reconstruction.
            if not isinstance(agent_instance, BaseAgent):
                raise TypeError(
                    "agent_instance must be an instance of BaseAgent or a subclass."
                )

            initial_state_data = agent_instance.to_dict()
            super().__init__(
                name=initial_state_data.get("name", "Anvil"),
                max_steps=initial_state_data.get("max_autonomous_steps", 15),
                output_format=initial_state_data.get("output_format", "text"),
                console=self.renderer.console,
                initial_state=initial_state_data,
            )
        else:
            # Normal initialization for a new REPL session
            super().__init__(name="Anvil", max_steps=15, console=self.renderer.console)

        # Saguaro specific substrate (REPL might need it directly for some commands)
        self.saguaro = SaguaroSubstrate()
        self.token_manager = ContextManager()

        # Command Registry for Slash Commands
        self.command_registry = CommandRegistry()
        self._register_commands()
        self.current_campaign_id = None

        # Thinking and loop control
        self.show_thinking = AGENTIC_THINKING.get("show_thinking", True)
        self.force_loop_type = None  # None = auto, "simple", or "enhanced"
        self.enhanced_loop_enabled = AGENTIC_THINKING.get("enhanced_loop_enabled", True)
        self.use_enhanced_tools = False  # Claude Code-style tool calling mode

        # Initialize Loop Orchestrator
        self.loop_orchestrator = LoopOrchestrator(
            self, self.saguaro, self.token_manager, self.renderer
        )
        self._repl_bus_agent_id = f"repl:{id(self)}"
        self._configure_progress_stream()

        # Ownership and collaboration infrastructure
        self.workset_manager = None
        self.ownership_registry = None
        self.instance_registry = None
        self.peer_discovery = None
        self.peer_transport = None
        self.repo_presence = None
        self.task_announcer = None
        self.collaboration_negotiator = None
        self.context_share_protocol = None
        self.agent_chat_channel = None
        self.collaboration_mode = "disabled"
        self.ownership_enabled = bool(OWNERSHIP_CONFIG.get("enabled", False))
        self.collaboration_enabled = bool(COLLABORATION_CONFIG.get("enabled", False))
        if self.ownership_enabled or self.collaboration_enabled:
            try:
                from core.subagent_communication import get_message_bus
                from shared_kernel.event_store import get_event_store

                event_store = get_event_store()

                if self.ownership_enabled:
                    from core.ownership.file_ownership import FileOwnershipRegistry
                    from core.ownership.file_ownership import build_trust_zone_resolver
                    from saguaro.workset import WorksetManager

                    self.workset_manager = WorksetManager(
                        saguaro_dir=".saguaro", repo_path="."
                    )
                    self.ownership_registry = FileOwnershipRegistry(
                        workset_manager=self.workset_manager,
                        message_bus=get_message_bus(),
                        event_store=event_store,
                        instance_id="local",
                        repo_policy_resolver=build_trust_zone_resolver(
                            local_zone="internal"
                        ),
                    )

                if self.collaboration_enabled:
                    from core.collaboration.agent_chat import AgentChatChannel
                    from core.collaboration.context_sharing import ContextShareProtocol
                    from core.collaboration.negotiation import CollaborationNegotiator
                    from core.collaboration.task_announcer import TaskAnnouncer

                    self.task_announcer = TaskAnnouncer(
                        transport=None,
                        overlap_threshold=float(
                            COLLABORATION_CONFIG.get("overlap_threshold", 0.75)
                        ),
                    )
                    self.collaboration_negotiator = CollaborationNegotiator(
                        event_store=event_store
                    )
                    self.context_share_protocol = ContextShareProtocol()
                    self.agent_chat_channel = AgentChatChannel(
                        transport=None,
                        event_store=event_store,
                    )
                    self.collaboration_mode = "local"
                    try:
                        from core.networking.instance_identity import InstanceRegistry
                        from core.networking.peer_discovery import PeerDiscovery
                        from core.networking.peer_transport import PeerTransport

                        self.instance_registry = InstanceRegistry(
                            anvil_dir=".anvil", project_root="."
                        )
                        self.peer_discovery = PeerDiscovery(
                            instance=self.instance_registry.identity,
                            method=str(
                                COLLABORATION_CONFIG.get("discovery_method", "auto")
                            ),
                            shared_peers_dir=".anvil/peers",
                            rendezvous_url=COLLABORATION_CONFIG.get("rendezvous_url"),
                        )
                        self.peer_transport = PeerTransport(
                            self.instance_registry.identity,
                            provider=COLLABORATION_CONFIG.get("transport_provider")
                            or (
                                "filesystem"
                                if str(
                                    COLLABORATION_CONFIG.get("discovery_method", "auto")
                                )
                                in {"auto", "filesystem"}
                                else "in_memory"
                            ),
                        )
                        self.task_announcer.transport = self.peer_transport
                        self.agent_chat_channel.transport = self.peer_transport
                        if self.peer_discovery is not None:
                            self.peer_discovery.start()
                        self.collaboration_mode = "networked"
                        from core.connectivity.repo_presence import RepoPresenceService

                        self.repo_presence = RepoPresenceService(
                            instance_registry=self.instance_registry,
                            peer_discovery=self.peer_discovery,
                            peer_transport=self.peer_transport,
                            ownership_registry=self.ownership_registry,
                            campaign_getter=self._campaign_presence_state,
                            capability_getter=self._presence_capability_state,
                        )
                        self.repo_presence.refresh()
                    except Exception as network_exc:
                        logger.debug(
                            "Collaboration running in local-only mode: %s",
                            network_exc,
                        )
            except Exception as exc:
                logger.debug("Ownership/collaboration bootstrap skipped: %s", exc)

        # Enhanced loop (lazy initialized)
        self._enhanced_loop = None
        self._enhanced_tool_loop = None
        self._unified_loop = None

        self.session = PromptSession(
            history=FileHistory(os.path.expanduser("~/.anvil_history")),
            style=Style.from_dict(
                {
                    "prompt": "ansicyan bold",
                }
            ),
            vi_mode=True,  # Enable Vim mode
        )

        self.project_memory = ProjectMemory(root_dir=".")

        # Proactive startup
        self.do_startup_scan = True
        self.env_info = None
        self.logging_enabled = AGENTIC_THINKING.get("logging_enabled", True)

    def ensure_environment_ready(self) -> None:
        affinity_report: dict[str, Any] = {}
        tuning_report: dict[str, Any] = {}
        with self.console.status("[bold green]Setting up environment...[/bold green]"):
            try:
                affinity_report = ensure_runtime_affinity()
                runtime_model = str(
                    getattr(getattr(self, "brain", None), "model_name", "")
                    or self.config.get("model", MASTER_MODEL)
                    or MASTER_MODEL
                ).strip()
                tuning_report = assess_runtime_tuning(
                    Path(root_dir),
                    models=[runtime_model],
                    invocation_source="repl_startup",
                )
                bootstrap_policy = resolve_runtime_tuning_bootstrap_policy()
                if not bool(
                    tuning_report.get("ready")
                ) and should_bootstrap_runtime_tuning(
                    tuning_report,
                    policy=bootstrap_policy,
                    has_prior_benchmark_evidence=has_benchmark_evidence(Path(root_dir)),
                ):
                    self.console.print(
                        "[dim cyan]Running runtime tuning calibration bootstrap.[/dim cyan]"
                    )
                    tuning_report = bootstrap_runtime_tuning(
                        Path(root_dir),
                        models=[runtime_model],
                        auto_run=True,
                        invocation_source="repl_startup",
                    )
                elif not bool(tuning_report.get("ready")):
                    tuning_report = mark_runtime_tuning_deferred(
                        tuning_report,
                        reason=f"bootstrap_policy={bootstrap_policy}",
                        policy=bootstrap_policy,
                    )
            except Exception as exc:
                tuning_report = {"status": "failed", "bootstrap_stderr": str(exc)}
            env_manager = EnvironmentManager()
            self.env_info = env_manager.ensure_ready(self.console)
            try:
                self.registry.register_mcp_tools()
            except Exception as e:
                self.console.print(
                    f"[dim yellow]⚠ MCP tools not fully registered: {e}[/dim yellow]"
                )
        if affinity_report.get("expanded"):
            before = len(list(affinity_report.get("before") or []))
            after = len(list(affinity_report.get("after") or []))
            self.console.print(
                f"[dim cyan]Runtime CPU affinity expanded: visible_threads {before} -> {after}[/dim cyan]"
            )
        for line in format_runtime_tuning_summary(tuning_report):
            self.console.print(f"[dim cyan]{line}[/dim cyan]")
        try:
            saguaro_profile = calibrate_runtime_profile(root_dir)
            layout = dict(saguaro_profile.get("selected_runtime_layout") or {})
            if layout:
                self.console.print(
                    "[dim cyan]"
                    f"Saguaro runtime profile: sessions={layout.get('max_concurrent_saguaro_sessions', 1)}, "
                    f"agents={layout.get('max_parallel_agents', 1)}, "
                    f"instances={layout.get('max_parallel_anvil_instances', 1)}."
                    "[/dim cyan]"
                )
        except Exception as exc:
            self.console.print(
                f"[dim yellow]Saguaro runtime profile calibration skipped: {exc}[/dim yellow]"
            )
        self.semantic_engine._indexed = True

    def _configure_progress_stream(self) -> None:
        try:
            bus = self.loop_orchestrator.unified_loop.message_bus
            bus.register_agent(
                self._repl_bus_agent_id,
                subscriptions=["progress"],
                metadata={"role": "repl"},
            )
            bus.register_handler("progress", self._on_progress_event)
        except Exception as exc:
            logger.debug("Progress stream unavailable: %s", exc)

    def _campaign_presence_state(self) -> dict[str, Any]:
        return {
            "campaign_id": self.current_campaign_id or "",
            "phase_id": getattr(self.loop_orchestrator, "current_phase", "") or "",
            "lane_id": "",
            "verification_state": "ready" if self.collaboration_enabled else "warming",
        }

    def _presence_capability_state(self) -> dict[str, Any]:
        getter = getattr(getattr(self, "brain", None), "get_runtime_status", None)
        runtime = dict(getter() or {}) if callable(getter) else {}
        return {
            "analysis_capacity": (
                1.0 if getattr(self, "saguaro", None) is not None else 0.2
            ),
            "verification_capacity": (
                1.0 if self.ownership_registry is not None else 0.3
            ),
            "runtime_symbol_digest": str(runtime.get("runtime_symbol_digest") or ""),
        }

    def _on_progress_event(self, message) -> None:
        payload = getattr(message, "payload", {}) or {}
        agent = payload.get("agent", "agent")
        event = payload.get("event", "update")
        tool = payload.get("tool")
        step = payload.get("step")
        details = []
        if step is not None:
            details.append(f"step={step}")
        if tool:
            details.append(f"tool={tool}")
        details_text = f" ({', '.join(details)})" if details else ""
        self.console.print(f"[dim][progress][/dim] {agent}: {event}{details_text}")

    @property
    def enhanced_loop(self):
        """Lazy-load enhanced loop."""
        if self._enhanced_loop is None and self.enhanced_loop_enabled:
            self._enhanced_loop = EnhancedAgenticLoop(self, console=self.console)
        return self._enhanced_loop

    def _register_commands(self):
        def register_many(category: str, commands: Iterable[Any]) -> None:
            for command in commands:
                self.command_registry.register(command, category=category)

        register_many(
            "mission",
            [
                HelpCommand(),
                AgentCommand(),
                CampaignCommand(),
                ManagerCommand(),
                WizardCommand(),
                ModeCommand(),
                SettingsCommand(),
                ModelCommand(),
                ModelsCommand(),
            ],
        )
        register_many(
            "campaigns",
            [
                TimelineCommand(),
                CheckpointCommand(),
                SessionCommand(),
                StatsCommand(),
                ResetCommand(),
                TreeCommand(),
                LogsCommand(),
            ],
        )
        register_many(
            "memory",
            [
                MemoryCommand(),
                ToolMemoryCommand(),
                DreamCommand(),
                SkillsCommand(),
            ],
        )
        register_many(
            "verification",
            [
                VerifyCommand(),
                ApproveCommand(),
                EnhancedToolsCommand(),
                SaguaroCommand(),
                UnwiredCommand(),
                CodebaseAnalysisCommand(),
            ],
        )
        register_many(
            "collaboration",
            [
                OwnershipCommand(),
                PeersCommand(),
                CollaborateCommand(),
                SyncCommand(),
                SwarmCommand(),
            ],
        )
        register_many(
            "diagnostics",
            [
                BaseThinkingCommand(),
                LoopCommand(),
                PlanCommand(),
                TaskCommand(),
                ArtifactsCommand(),
                ArchiveCommand(),
                ArchivesCommand(),
                AnalyzeCommand(),
                ResearchCommand(),
                DebugCommand(),
                ImplementCommand(),
                TestCommand(),
                SubagentPlanCommand(),
                DareCommand(),
                ClearCommand(),
                ExitCommand(),
                CoconutCommand(),
            ],
        )

    def run(self):
        try:
            self.ensure_environment_ready()
        except Exception as e:
            error_text = str(e)
            emit_structured_event(
                logger,
                component="repl",
                event="repl.startup.error",
                model=MASTER_MODEL,
                phase="startup",
                metrics={
                    "cwd": os.getcwd(),
                    "python": sys.executable,
                    "error": error_text,
                    "strict_mode_tensorflow_error": "TensorFlow-backed embeddings"
                    in error_text,
                },
                level=logging.ERROR,
            )
            raise

        emit_structured_event(
            logger,
            component="repl",
            event="repl.startup.ready",
            model=MASTER_MODEL,
            phase="startup",
            metrics={
                "cwd": os.getcwd(),
                "python": sys.executable,
                "mode": self.env_info.get("mode") if self.env_info else None,
                "venv_dir": self.env_info.get("venv_dir") if self.env_info else None,
                "anvil_root": (
                    self.env_info.get("anvil_root") if self.env_info else None
                ),
            },
            level=logging.INFO,
        )

        # Display Welcome Screen
        self.renderer.print_welcome_screen(MASTER_MODEL, "Unified Chat")
        self._render_startup_posture()

        while True:
            try:

                # Dynamic bottom toolbar
                def get_toolbar():
                    # Calculate usage on the fly
                    messages = self.history.get_messages()
                    stats = self.token_manager.get_stats(messages)
                    short_usage = stats.replace("Context Usage: ", "").replace(
                        "tokens ", ""
                    )

                    # Get current editing mode
                    # Default to Emacs mode for general cases or if not VI
                    mode_text = "Emacs"
                    if self.session.editing_mode == EditingMode.VI:
                        # prompt_toolkit's ViState has an input_mode attribute
                        # that tells us if it's NORMAL or INSERT.
                        # This depends on prompt_toolkit internals and might be brittle.
                        try:
                            if self.session.app.vi_state.input_mode.name == "NORMAL":
                                mode_text = "VI (Normal)"
                            else:
                                mode_text = "VI (Insert)"
                        except AttributeError:
                            # Fallback if vi_state or input_mode is not as expected
                            mode_text = "VI"

                    # Get COCONUT info
                    coconut_info = "COCONUT: Off"
                    if (
                        self.loop_orchestrator.thinking_system
                        and self.loop_orchestrator.thinking_system.coconut
                    ):
                        info = (
                            self.loop_orchestrator.thinking_system.coconut.get_device_info()
                        )
                        backend = info.get("backend", "cpu")
                        device = str(info.get("device", "cpu")).split(":")[-1]
                        coconut_info = f"🥥 {backend}({device})"

                    # CPU-only engine display for deterministic startup and render.
                    engine_info = "🤖 cpu"

                    return get_bottom_toolbar_text(
                        self.approval_manager.mode.value,
                        MASTER_MODEL,
                        context_usage=short_usage,
                        editing_mode=mode_text,  # Pass the editing mode
                        coconut_info=coconut_info,
                        engine_info=engine_info,
                    )

                prompt_html = HTML("<style fg='ansicyan'>anvil></style> ")

                user_input = self.session.prompt(
                    prompt_html, bottom_toolbar=get_toolbar
                )

                if not user_input.strip():
                    continue

                # Slash Commands
                if user_input.startswith("/"):
                    if self.command_registry.dispatch(user_input, self):
                        continue

                # Select loop type and run
                self.run_mission(user_input)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted by user[/yellow]")
                continue
            except EOFError:
                break
            except Exception:
                self.console.print_exception()

    @staticmethod
    def _deterministic_synthesis_lane_context(objective: str) -> dict[str, Any]:
        text = str(objective or "").strip()
        lowered = text.lower()
        explicit_prefixes = ("synth:", "deterministic synthesis:", "/synth ")
        env_enabled = str(os.getenv("ANVIL_DETERMINISTIC_SYNTHESIS", "0")).strip() == "1"
        prefix_enabled = any(lowered.startswith(prefix) for prefix in explicit_prefixes)
        keyword_enabled = "deterministic synthesis" in lowered
        enabled = bool(env_enabled or prefix_enabled or keyword_enabled)
        return {
            "enabled": enabled,
            "label": "deterministic-synthesis" if enabled else "standard-mission",
            "badge": "DET-SYNTH" if enabled else "STANDARD",
            "objective": text,
        }

    def run_mission(self, objective: str, interactive: bool = True) -> str:
        """Runs an agentic mission using the LoopOrchestrator."""
        mission_id = uuid.uuid4().hex[:12]
        self.current_mission_id = mission_id
        lane_context = self._deterministic_synthesis_lane_context(objective)
        intake_builder = getattr(self, "_build_compact_intake", None)
        if callable(intake_builder):
            intake = intake_builder(objective)
        else:
            intake = AgentREPL._build_compact_intake(self, objective)

        # Minimal mode indicator
        mode_status = "Enhanced" if self.enhanced_loop_enabled else "Standard"
        active_model = getattr(self.brain, "model_name", MASTER_MODEL)
        if interactive:
            self.console.print(f"\n[dim]{mode_status} mode • {active_model}[/dim]\n")
            if lane_context["enabled"]:
                self.console.print(
                    "[bold cyan]Deterministic synthesis lane active[/bold cyan] "
                    f"[dim]({lane_context['badge']})[/dim]"
                )
            if intake["show"]:
                intake_line = " | ".join(
                    [
                        f"Intake: {intake['request_type']}",
                        f"Surface: {intake['recommended_surface']}",
                        f"Campaign: {getattr(self, 'current_campaign_id', None) or 'none'}",
                    ]
                )
                if hasattr(self.renderer, "print_system"):
                    self.renderer.print_system(intake_line)
                else:
                    self.console.print(f"[dim]{intake_line}[/dim]")
            if hasattr(self.brain, "runtime_status"):
                try:
                    status = self.brain.runtime_status()
                    capability_vector = dict(status.get("capability_vector") or {})
                    controller_state = dict(status.get("controller_state") or {})
                    repo_coupled_runtime = dict(
                        status.get("repo_coupled_runtime") or {}
                    )
                    self.console.print(
                        "[dim]"
                        f"QSG {status.get('backend', 'native_qsg')} • "
                        f"digest {status.get('digest', 'unknown')[:18]} • "
                        f"threads {status.get('decode_threads', 'n/a')}/"
                        f"{status.get('batch_threads', 'n/a')} • "
                        f"OpenMP {status.get('openmp_enabled', False)} • "
                        f"AVX2 {status.get('avx2_enabled', False)} • "
                        f"ISA {capability_vector.get('native_isa_baseline', 'unknown')} • "
                        f"frontier {dict(controller_state.get('frontier') or {}).get('selected_mode', 'unknown')} • "
                        f"drift {dict(controller_state.get('drift') or {}).get('selected_mode', 'unknown')} • "
                        f"delta {dict(repo_coupled_runtime.get('delta_watermark') or {}).get('delta_id', '')}"
                        "[/dim]\n"
                    )
                except Exception:
                    pass

        start_perf = time.perf_counter()
        log_file = get_active_log_file()
        root_level = logging.getLevelName(logging.getLogger().getEffectiveLevel())

        emit_structured_event(
            logger,
            component="repl",
            event="repl.mission.start",
            mission_id=mission_id,
            model=active_model,
            phase="mission",
            metrics={
                "cwd": os.getcwd(),
                "python": sys.executable,
                "objective_chars": len(objective or ""),
                "mode_status": mode_status,
                "deterministic_synthesis_lane": lane_context["enabled"],
                "source_path": os.path.abspath(__file__),
                "git_sha": _short_git_sha(),
                "log_level": root_level,
                "log_file": log_file,
                "debug_enabled": root_level == "DEBUG",
            },
            level=logging.INFO,
        )

        # Run mission via centralized loop orchestrator
        try:
            if interactive:
                self.renderer.start_live_dashboard(active_model, mode_status)

            orchestrator_start = time.perf_counter()
            emit_structured_event(
                logger,
                component="repl",
                event="repl.orchestrator.run.start",
                mission_id=mission_id,
                model=active_model,
                phase="orchestrator",
                metrics={"objective_chars": len(objective or "")},
                level=logging.INFO,
            )
            if interactive:
                response = self.loop_orchestrator.run(objective)
            else:
                response = self.loop_orchestrator.unified_loop.run(
                    objective, dashboard=None
                )
            orchestrator_duration_ms = int(
                (time.perf_counter() - orchestrator_start) * 1000
            )
            emit_structured_event(
                logger,
                component="repl",
                event="repl.orchestrator.run.end",
                mission_id=mission_id,
                model=active_model,
                phase="orchestrator",
                duration_ms=orchestrator_duration_ms,
                metrics={"response_chars": len(response or "")},
                level=logging.INFO,
            )

            if interactive:
                # Print the final response after the dashboard is stopped
                self.renderer.stop_live_dashboard()
                if response:
                    self.renderer.print_response(response)
        except Exception as e:
            if interactive:
                self.renderer.stop_live_dashboard()
            elapsed_ms = int((time.perf_counter() - start_perf) * 1000)
            emit_structured_event(
                logger,
                component="repl",
                event="repl.mission.error",
                mission_id=mission_id,
                model=active_model,
                phase="mission",
                duration_ms=elapsed_ms,
                metrics={"error_type": type(e).__name__, "error": str(e)},
                level=logging.ERROR,
            )
            self.current_mission_id = None
            raise

        elapsed = time.perf_counter() - start_perf
        elapsed_ms = int(elapsed * 1000)
        emit_structured_event(
            logger,
            component="repl",
            event="repl.mission.complete",
            mission_id=mission_id,
            model=active_model,
            phase="mission",
            duration_ms=elapsed_ms,
            metrics={"mode_status": mode_status},
            level=logging.INFO,
        )
        self.current_mission_id = None

        # Minimal completion status (Claude Code style)
        if interactive:
            self.console.print(f"\n[dim]✓ Complete in {elapsed:.1f}s[/dim]\n")
        return response or ""

    def _build_compact_intake(self, objective: str) -> dict[str, Any]:
        unified_loop = getattr(self.loop_orchestrator, "unified_loop", None)
        request_type = "unknown"
        try:
            if unified_loop is not None:
                request_type = str(unified_loop._classify_request(objective))
        except Exception:
            request_type = "unknown"
        lowered = str(objective or "").lower()
        campaignish = any(
            token in lowered
            for token in ("campaign", "roadmap", "detach", "speculate", "governance")
        )
        recommended_surface = "mission"
        if campaignish or (
            request_type in {"creation", "modification"} and len(lowered) > 120
        ):
            recommended_surface = "campaign"
        return {
            "request_type": request_type,
            "recommended_surface": recommended_surface,
            "show": recommended_surface == "campaign" or request_type != "question",
        }

    def _render_startup_posture(self) -> None:
        if not self.env_info:
            return
        self.renderer.print_system(
            " | ".join(
                [
                    f"Mode: {self.env_info.get('mode', 'unknown')}",
                    f"Policy: {self.approval_manager.policy_profile.value}",
                    f"Saguaro: {self.env_info.get('saguaro_status', 'unknown')}",
                    f"Campaign: {self.current_campaign_id or 'none'}",
                ]
            )
        )
        timings = self.env_info.get("startup_timings") or {}
        if timings:
            self.renderer.print_system(
                "Startup timings: "
                f"venv={timings.get('venv_check_ms', 0)}ms "
                f"saguaro={timings.get('saguaro_check_ms', 0)}ms "
                f"ready={timings.get('environment_ready_ms', 0)}ms"
            )
        if self.env_info.get("saguaro_status") == "degraded":
            self.renderer.print_warning(
                "Saguaro degraded mode is active; semantic retrieval may fall back."
            )


def _coerce_policy_profile(raw: str | None) -> str | None:
    if not raw:
        return None
    normalized = raw.strip().lower()
    valid = {member.value for member in PolicyProfile}
    if normalized in valid:
        return normalized
    raise ValueError(
        f"Unknown policy profile '{raw}'. Expected one of: {', '.join(sorted(valid))}"
    )


def _collect_warnings(messages: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for msg in messages:
        content = str(msg.get("content", ""))
        if "warning" in content.lower() or "warn:" in content.lower():
            warnings.append(content[:300])
    return warnings


def _classify_exit(
    response: str,
    messages: list[dict[str, Any]],
    *,
    exit_on_warning: bool,
) -> dict[str, Any]:
    blob = "\n".join(str(m.get("content", "")) for m in messages[-80:])
    blob_lower = blob.lower()
    response_lower = (response or "").lower()
    warnings = _collect_warnings(messages)

    if "timed out" in blob_lower or "timed out" in response_lower:
        return {"status": "timeout", "exit_code": EXIT_TIMEOUT, "warnings": warnings}
    if (
        "execution was denied by the user" in blob_lower
        or "policy denied" in blob_lower
    ):
        return {
            "status": "policy_denied",
            "exit_code": EXIT_POLICY_DENIED,
            "warnings": warnings,
        }
    if "verification failed" in blob_lower:
        return {
            "status": "verification_failed",
            "exit_code": EXIT_VERIFICATION_FAILED,
            "warnings": warnings,
        }
    if "error executing" in blob_lower or "tool execution error" in blob_lower:
        return {
            "status": "tool_failure",
            "exit_code": EXIT_TOOL_FAILURE,
            "warnings": warnings,
        }
    if exit_on_warning and warnings:
        return {
            "status": "verification_failed",
            "exit_code": EXIT_VERIFICATION_FAILED,
            "warnings": warnings,
        }
    return {"status": "success", "exit_code": EXIT_SUCCESS, "warnings": warnings}


def _emit_scripted_output(
    *,
    payload: dict[str, Any],
    output_format: str,
    output_file: str | None,
    quiet: bool,
) -> None:
    if output_format == "json":
        rendered = json.dumps(payload, indent=2, sort_keys=True)
    else:
        rendered = str(payload.get("response", ""))

    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as handle:
            handle.write(rendered)

    if not quiet:
        print(rendered)


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sign_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _write_run_artifact(
    *,
    repl: AgentREPL,
    outcome_payload: dict[str, Any],
    output_file: str | None,
) -> dict[str, Any]:
    run_dir = os.path.join(".anvil", "runs")
    os.makedirs(run_dir, exist_ok=True)
    session_id = getattr(repl.history, "session_id", "unknown")
    run_id = f"{session_id}-{int(time.time())}"
    audit_path = os.path.join(run_dir, f"{run_id}.audit.json")
    exported_audit = repl.history.export_audit(audit_path)

    artifact = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "session_id": session_id,
        "policy_profile": repl.approval_manager.policy_profile.value,
        "output_file": output_file,
        "audit_path": exported_audit,
        "outcome": {
            "status": outcome_payload.get("status"),
            "exit_code": outcome_payload.get("exit_code"),
            "warnings": outcome_payload.get("warnings", []),
            "response_sha256": hashlib.sha256(
                str(outcome_payload.get("response", "")).encode("utf-8")
            ).hexdigest(),
        },
    }
    artifact["signature"] = _sign_payload(artifact)

    artifact_path = os.path.join(run_dir, f"{run_id}.artifact.json")
    with open(artifact_path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
    return {"artifact_path": artifact_path, "artifact_signature": artifact["signature"]}


def _run_mission_subcommand(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="anvil mission")
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Create a managed campaign and continue it in a detached worker.",
    )
    parser.add_argument(
        "--root-dir",
        default=".",
        help="Root directory for the managed campaign target repo.",
    )
    parser.add_argument(
        "--name",
        help="Optional campaign name. Defaults to the objective prefix.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format.",
    )
    parser.add_argument("objective_words", nargs="+", help="Mission objective.")
    args = parser.parse_args(argv)

    from core.campaign.runner import CampaignRunner

    objective = " ".join(args.objective_words).strip()
    runner = CampaignRunner()
    created = runner.create_autonomy_campaign(
        name=args.name or objective[:64] or "Mission",
        objective=objective,
        directives=[objective],
        root_dir=args.root_dir,
    )
    campaign_id = created["campaign_id"]
    payload: dict[str, Any] = {
        "campaign_id": campaign_id,
        "workspace": created["workspace"],
        "objective": objective,
        "detached": False,
    }
    if args.detach:
        payload["detached"] = True
        payload["worker"] = runner.detach_campaign(campaign_id)
    else:
        payload["event"] = runner.continue_autonomy_campaign(campaign_id)

    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.detach:
        print(
            f"Detached mission {campaign_id} "
            f"(pid={(payload.get('worker') or {}).get('pid')}, "
            f"state={(payload.get('worker') or {}).get('state')})"
        )
    else:
        print(
            f"Mission {campaign_id} -> "
            f"{(payload.get('event') or {}).get('to_state', 'unknown')}"
        )
    return 0


def main(argv: list[str] | None = None):
    argv = list(argv or [])
    if argv and argv[0] == "mission":
        return _run_mission_subcommand(argv[1:])

    parser = argparse.ArgumentParser(description="Anvil Agent REPL")
    parser.add_argument("prompt_words", nargs="*", help="Optional one-shot prompt")
    parser.add_argument(
        "--prompt",
        help="Run in scripted mode with this prompt (non-interactive).",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Scripted output format (default: text).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stdout output (still writes --output-file if provided).",
    )
    parser.add_argument(
        "--output-file",
        help="Write scripted output to a file path.",
    )
    parser.add_argument(
        "--exit-on-warning",
        action="store_true",
        help="Return verification_failed exit code when warnings are detected.",
    )
    parser.add_argument(
        "--policy-profile",
        choices=[member.value for member in PolicyProfile],
        help="Policy profile for this run: trusted|balanced|strict|regulated.",
    )
    parser.add_argument(
        "--dry-policy",
        action="store_true",
        help="Run only policy simulation for this prompt (no mission execution).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and policy-simulate only. Do not execute the mission.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Return a deterministic execution plan without running tools.",
    )
    parser.add_argument(
        "--replay-artifact",
        help="Replay a previously generated .artifact.json and return its recorded outcome.",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        default=None,
        help="Enable logging to .anvil/logs/anvil.log",
    )
    parser.add_argument(
        "--no-log", action="store_false", dest="log", help="Disable logging"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Set logging level to DEBUG"
    )

    args = parser.parse_args(argv)

    # Update settings based on flags
    if args.log is not None:
        AGENTIC_THINKING["logging_enabled"] = args.log

    if AGENTIC_THINKING.get("logging_enabled", True):
        level = logging.DEBUG if args.debug else logging.INFO
        console_level = logging.DEBUG if args.debug else None
        setup_logging(level=level, console_level=console_level)

    prompt = args.prompt
    if not prompt and args.prompt_words:
        prompt = " ".join(args.prompt_words).strip()
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()

    scripted_mode = bool(
        prompt
        or args.prompt
        or args.output_file
        or args.plan_only
        or args.dry_run
        or args.dry_policy
        or args.replay_artifact
        or args.format == "json"
    )

    if not scripted_mode:
        _run_interactive_toolchain_check()
        repl = AgentREPL()
        repl.run()
        return

    try:
        policy_profile = _coerce_policy_profile(args.policy_profile)
    except ValueError as exc:
        payload = {
            "schema_version": "1.0",
            "status": "policy_denied",
            "exit_code": EXIT_POLICY_DENIED,
            "error": str(exc),
            "response": "",
            "warnings": [],
        }
        _emit_scripted_output(
            payload=payload,
            output_format=args.format,
            output_file=args.output_file,
            quiet=args.quiet,
        )
        sys.exit(EXIT_POLICY_DENIED)

    if args.replay_artifact:
        try:
            with open(args.replay_artifact, encoding="utf-8") as handle:
                artifact = json.load(handle)
            payload = {
                "schema_version": "1.0",
                "status": artifact.get("outcome", {}).get("status", "tool_failure"),
                "exit_code": int(
                    artifact.get("outcome", {}).get("exit_code", EXIT_TOOL_FAILURE)
                ),
                "replayed_artifact": args.replay_artifact,
                "artifact_signature": artifact.get("signature"),
                "response": "",
                "warnings": artifact.get("outcome", {}).get("warnings", []),
                "dry_run": False,
                "plan_only": False,
                "replay": True,
            }
            _emit_scripted_output(
                payload=payload,
                output_format=args.format,
                output_file=args.output_file,
                quiet=args.quiet,
            )
            sys.exit(payload["exit_code"])
        except Exception as exc:
            payload = {
                "schema_version": "1.0",
                "status": "tool_failure",
                "exit_code": EXIT_TOOL_FAILURE,
                "error": f"Failed to replay artifact: {exc}",
                "response": "",
                "warnings": [],
            }
            _emit_scripted_output(
                payload=payload,
                output_format=args.format,
                output_file=args.output_file,
                quiet=args.quiet,
            )
            sys.exit(EXIT_TOOL_FAILURE)

    if not prompt:
        payload = {
            "schema_version": "1.0",
            "status": "tool_failure",
            "exit_code": EXIT_TOOL_FAILURE,
            "error": "No prompt provided for scripted mode.",
            "response": "",
            "warnings": [],
        }
        _emit_scripted_output(
            payload=payload,
            output_format=args.format,
            output_file=args.output_file,
            quiet=args.quiet,
        )
        sys.exit(EXIT_TOOL_FAILURE)

    repl = AgentREPL()
    if args.quiet:
        repl.console.quiet = True
        if hasattr(repl, "renderer") and hasattr(repl.renderer, "console"):
            repl.renderer.console.quiet = True
    if policy_profile:
        repl.approval_manager.set_policy_profile(policy_profile)
        repl.config.set("policy_profile", policy_profile)

    if args.plan_only:
        payload = {
            "schema_version": "1.0",
            "status": "plan_only",
            "exit_code": EXIT_SUCCESS,
            "prompt": prompt,
            "policy_profile": repl.approval_manager.policy_profile.value,
            "plan": [
                "Classify objective and gather semantic evidence.",
                "Execute constrained tool calls under active policy profile.",
                "Synthesize response with verification evidence and timeline.",
            ],
            "response": "",
            "warnings": [],
            "dry_run": bool(args.dry_run),
            "plan_only": True,
        }
        _emit_scripted_output(
            payload=payload,
            output_format=args.format,
            output_file=args.output_file,
            quiet=args.quiet,
        )
        sys.exit(EXIT_SUCCESS)

    if args.dry_run or args.dry_policy:
        simulations = [
            repl.approval_manager.simulate("read_file", {"path": "core/agent.py"}),
            repl.approval_manager.simulate(
                "run_command", {"command": "pytest -q", "max_runtime": 30}
            ),
            repl.approval_manager.simulate(
                "write_file", {"path": "tmp.txt", "content": "example"}
            ),
        ]
        payload = {
            "schema_version": "1.0",
            "status": "dry_policy" if args.dry_policy else "dry_run",
            "exit_code": EXIT_SUCCESS,
            "prompt": prompt,
            "policy_profile": repl.approval_manager.policy_profile.value,
            "simulations": simulations,
            "response": "",
            "warnings": [],
            "dry_run": True,
            "dry_policy": bool(args.dry_policy),
            "plan_only": False,
        }
        _emit_scripted_output(
            payload=payload,
            output_format=args.format,
            output_file=args.output_file,
            quiet=args.quiet,
        )
        sys.exit(EXIT_SUCCESS)

    try:
        repl.ensure_environment_ready()
        response = repl.run_mission(prompt, interactive=False)
        messages = repl.history.get_messages()
        classification = _classify_exit(
            response,
            messages,
            exit_on_warning=args.exit_on_warning,
        )
        payload = {
            "schema_version": "1.0",
            "status": classification["status"],
            "exit_code": classification["exit_code"],
            "prompt": prompt,
            "policy_profile": repl.approval_manager.policy_profile.value,
            "response": response,
            "warnings": classification["warnings"],
            "dry_run": False,
            "plan_only": False,
            "timeline": repl.history.get_timeline(limit=40),
            "session_id": getattr(repl.history, "session_id", None),
        }
        payload.update(
            _write_run_artifact(
                repl=repl,
                outcome_payload=payload,
                output_file=args.output_file,
            )
        )
        _emit_scripted_output(
            payload=payload,
            output_format=args.format,
            output_file=args.output_file,
            quiet=args.quiet,
        )
        sys.exit(classification["exit_code"])
    except Exception as exc:
        text = str(exc)
        status = "timeout" if "timed out" in text.lower() else "tool_failure"
        exit_code = EXIT_TIMEOUT if status == "timeout" else EXIT_INTERNAL_ERROR
        payload = {
            "schema_version": "1.0",
            "status": status,
            "exit_code": exit_code,
            "prompt": prompt,
            "policy_profile": repl.approval_manager.policy_profile.value,
            "response": "",
            "error": text,
            "warnings": [],
            "dry_run": False,
            "plan_only": False,
        }
        _emit_scripted_output(
            payload=payload,
            output_format=args.format,
            output_file=args.output_file,
            quiet=args.quiet,
        )
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
