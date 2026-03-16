from typing import List, Optional, Any
import json
from cli.commands.base import SlashCommand
from core.checkpoint import CheckpointManager
from core.session_manager import SessionManager
from config.settings import (
    GENERATION_PARAMS,
    GRANITE4_SAMPLING_PROFILES,
    QWEN35_SAMPLING_PROFILES,
)


class CheckpointCommand(SlashCommand):
    @property
    def name(self) -> str:
        return "checkpoint"

    @property
    def description(self) -> str:
        return "Manage checkpoints: list, save <name>, load <name>"

    def execute(self, args: List[str], context: Any) -> Optional[str]:
        if not hasattr(context, "checkpoint_manager"):
            context.checkpoint_manager = CheckpointManager()

        if not args or args[0] == "list":
            ckpts = context.checkpoint_manager.list_checkpoints()
            if not ckpts:
                return "No checkpoints found."
            lines = ["Checkpoints:"]
            for c in ckpts:
                lines.append(f" - {c['id']} ({c['timestamp']})")
            return "\n".join(lines)

        action = args[0]

        if action == "save":
            name = args[1] if len(args) > 1 else None
            state = {
                "history": context.history.get_messages(),
                "config": context.config.config,
            }
            saved_name = context.checkpoint_manager.save_checkpoint(name, state)
            return f"Checkpoint saved: {saved_name}"

        return "Usage: /checkpoint [list|save <name>|load <name>]"


class SessionCommand(SlashCommand):
    @property
    def name(self) -> str:
        return "session"

    @property
    def description(self) -> str:
        return "Manage sessions: list, save <name>, load <name>"

    def execute(self, args: List[str], context: Any) -> Optional[str]:
        # context is AgentREPL -> BaseAgent
        # We need a SessionManager instance.
        mgr = SessionManager()

        if not args or args[0] == "list":
            sessions = mgr.list_sessions()
            return f"Available sessions: {', '.join(sessions)}"

        action = args[0]
        if action == "save":
            if len(args) < 2:
                return "Usage: /session save <name>"
            name = args[1]
            try:
                mgr.save_agent_state(
                    context, name
                )  # 'context' is the AgentREPL instance (which is a BaseAgent)
                return f"Agent state for session '{name}' saved successfully."
            except Exception as e:
                return f"Error saving agent state: {e}"

        if action == "load":
            if len(args) < 2:
                return "Usage: /session load <name>"
            name = args[1]
            data = mgr.load_session(name)
            if not data:
                return f"Session '{name}' not found."

            # Restore state
            context.history.messages = data.get("history", [])
            # Config restoration?
            if "config" in data:
                context.config.config.update(data["config"])

            return f"Session '{name}' loaded with {len(context.history.messages)} messages."

        return "Usage: /session [list|save|load]"


class StatsCommand(SlashCommand):
    @property
    def name(self) -> str:
        return "stats"

    @property
    def aliases(self) -> List[str]:
        return ["perf", "performance"]

    @property
    def description(self) -> str:
        return "Show session and performance statistics"

    def execute(self, args: List[str], context: Any) -> Optional[str]:
        # Session stats
        msgs = context.history.get_messages()
        user_msgs = len([m for m in msgs if m["role"] == "user"])
        tool_msgs = len([m for m in msgs if m["role"] == "tool"])

        output = f"Session Stats:\nUser messages: {user_msgs}\nTool calls: {tool_msgs}\nTotal History: {len(msgs)}"

        brain = getattr(context, "brain", None)
        runtime_status = {}
        if brain is not None:
            runtime_fn = getattr(brain, "runtime_status", None)
            if callable(runtime_fn):
                try:
                    runtime_status = dict(runtime_fn())
                except Exception:
                    runtime_status = {}

        if args and args[0] == "qsg":
            if not runtime_status:
                return output + "\n\nNative QSG status: unavailable"
            capability_vector = dict(runtime_status.get("capability_vector") or {})
            controller_state = dict(runtime_status.get("controller_state") or {})
            performance_envelope = dict(
                runtime_status.get("performance_envelope") or {}
            )
            performance_twin = dict(runtime_status.get("performance_twin") or {})
            repo_coupled_runtime = dict(
                runtime_status.get("repo_coupled_runtime") or {}
            )
            lines = [
                output,
                "",
                "Native QSG Status:",
                f"Model: {runtime_status.get('model', 'unknown')}",
                f"Digest: {runtime_status.get('digest', 'unknown')}",
                f"Backend: {runtime_status.get('backend', 'unknown')}",
                f"Threads: decode={runtime_status.get('decode_threads', 0)} batch={runtime_status.get('batch_threads', 0)} ubatch={runtime_status.get('ubatch', 0)}",
                f"Throughput: prefill={runtime_status.get('prefill_throughput_tps', 0.0):.2f} tok/s decode={runtime_status.get('decode_throughput_tps', 0.0):.2f} tok/s",
                f"Latency: ttft={runtime_status.get('first_token_latency_seconds', 0.0) * 1000.0:.2f} ms p50={runtime_status.get('per_token_latency_p50_ms', 0.0):.2f} ms p95={runtime_status.get('per_token_latency_p95_ms', 0.0):.2f} ms",
                f"KV: used_cells={runtime_status.get('kv_used_cells', 0)} fragmentation={runtime_status.get('kv_fragmentation_ratio', 0.0):.4f} defrag={runtime_status.get('kv_defrag_count', 0)}",
                f"Capabilities: mmap={runtime_status.get('mmap_enabled', False)} openmp={runtime_status.get('openmp_enabled', False)} avx2={runtime_status.get('avx2_enabled', False)} avx512={runtime_status.get('avx512_enabled', False)}",
                f"Capability Vector: isa={capability_vector.get('native_isa_baseline', 'unknown')} strict={capability_vector.get('strict_path_stable', False)} abi_match={capability_vector.get('native_backend_abi_match', False)}",
                f"Controllers: frontier={dict(controller_state.get('frontier') or {}).get('selected_mode', 'unknown')} drift={dict(controller_state.get('drift') or {}).get('selected_mode', 'unknown')}",
                f"Envelope: drift_overhead={performance_envelope.get('drift_overhead_percent', 0.0):.2f}% queue_wait={performance_envelope.get('scheduler_queue_wait_ms', 0.0):.2f} ms hot_path_numpy={performance_envelope.get('numpy_hot_path_calls', 0)}",
                f"Twin: regime={performance_twin.get('predicted_regime', 'unknown')} risk={performance_twin.get('risk_level', 'unknown')} score={performance_twin.get('risk_score', 0.0):.2f}",
                f"Repo Coupling: delta={dict(repo_coupled_runtime.get('delta_watermark') or {}).get('delta_id', '')} authority={repo_coupled_runtime.get('delta_authority', 'unknown')}",
            ]
            return "\n".join(lines)
        if args and args[0] == "qsg":
            brain = getattr(context, "brain", None)
            if brain is None or not hasattr(brain, "runtime_status"):
                return output + "\n\nQSG runtime: unavailable"
            try:
                status = brain.runtime_status()
            except Exception as exc:
                return output + f"\n\nQSG runtime error: {exc}"
            return (
                output
                + "\n\nQSG Runtime:\n"
                + json.dumps(status, indent=2, sort_keys=True, default=str)
            )

        # Performance stats (if monitoring enabled)
        if (
            hasattr(context.agent, "perf_monitor")
            and context.agent.perf_monitor is not None
        ):
            monitor = context.agent.perf_monitor

            # Handle subcommands
            if args and args[0] == "clear":
                monitor.clear()
                return output + "\n\nPerformance statistics cleared."

            if args and args[0] == "baseline":
                baseline_tps = float(args[1]) if len(args) > 1 else 3.5
                context.console.print("\n" + monitor.get_speedup_estimate(baseline_tps))
                return output

            # Show full performance report
            context.console.print("\n" + monitor.get_report())
            return output
        brain = getattr(context, "brain", None)
        if brain is not None and hasattr(brain, "runtime_status"):
            try:
                status = brain.runtime_status()
                output += (
                    "\n\nQSG Runtime: "
                    f"{status.get('backend', 'unknown')} "
                    f"model={status.get('model', 'unknown')} "
                    f"digest={status.get('digest', 'unknown')} "
                    f"threads={status.get('decode_threads', 'n/a')}/{status.get('batch_threads', 'n/a')} "
                    f"ftl={status.get('first_token_latency_seconds', 0.0):.3f}s "
                    f"decode_tps={status.get('decode_throughput_tps', 0.0):.2f}"
                )
            except Exception:
                pass
        return output + "\n\nPerformance monitoring: Not enabled"


class ResetCommand(SlashCommand):
    @property
    def name(self) -> str:
        return "reset"

    @property
    def description(self) -> str:
        return "Reset conversation history"

    def execute(self, args: List[str], context: Any) -> Optional[str]:
        context.history.clear()
        return "Conversation history cleared."


class ModelCommand(SlashCommand):
    @property
    def name(self) -> str:
        return "model"

    @property
    def description(self) -> str:
        return "Switch model, sampling profiles, and runtime context"

    @staticmethod
    def _model_key(model_name: str) -> str:
        lowered = str(model_name or "").lower()
        if "qwen3.5" in lowered or "qwen35" in lowered:
            return "qwen35"
        if "granite4" in lowered:
            return "granite4"
        return "other"

    @classmethod
    def _profile_catalog(cls, model_name: str) -> tuple[str, dict]:
        model_key = cls._model_key(model_name)
        if model_key == "qwen35":
            return "qwen35_sampling_profile", QWEN35_SAMPLING_PROFILES
        if model_key == "granite4":
            return "granite4_sampling_profile", GRANITE4_SAMPLING_PROFILES
        return "", {}

    @staticmethod
    def _reload_brain(context: Any, model_name: str) -> None:
        from core.ollama_client import DeterministicOllama

        DeterministicOllama.clear_loader_cache(model_name)
        context.brain = DeterministicOllama(model_name)

    @staticmethod
    def _native_context_keys(model_name: str) -> tuple[str, str]:
        lowered = str(model_name or "").lower()
        if "qwen3.5" in lowered or "qwen35" in lowered:
            return ("qwen35_native_ctx_default", "qwen35_native_ctx_cap")
        if "granite4" in lowered:
            return ("granite4_native_ctx_default", "granite4_native_ctx_cap")
        return ("native_ctx_default", "native_ctx_cap")

    def execute(self, args: List[str], context: Any) -> Optional[str]:
        current_model = context.config.get("model") or getattr(
            getattr(context, "brain", None), "model_name", ""
        )

        if not args:
            profile_key, _ = self._profile_catalog(current_model)
            profile_name = (
                str(GENERATION_PARAMS.get(profile_key))
                if profile_key and GENERATION_PARAMS.get(profile_key) is not None
                else "n/a"
            )
            ctx_default_key, ctx_cap_key = self._native_context_keys(current_model)
            num_ctx = int(GENERATION_PARAMS.get("num_ctx", 400000))
            native_default = int(GENERATION_PARAMS.get(ctx_default_key, num_ctx))
            native_cap = int(GENERATION_PARAMS.get(ctx_cap_key, num_ctx))
            return (
                f"Current Model: {current_model}\n"
                f"Active Sampling Profile: {profile_name}\n"
                f"Context: num_ctx={num_ctx}, native_default={native_default}, native_cap={native_cap}"
            )

        if args[0] == "profile":
            profile_key, profiles = self._profile_catalog(current_model)
            if not profiles:
                return f"No model-specific sampling profiles configured for '{current_model}'."
            if len(args) == 1 or args[1] in {"list", "ls"}:
                active = str(GENERATION_PARAMS.get(profile_key, ""))
                lines = [f"Available profiles for {current_model}:"]
                for name in profiles:
                    marker = " (active)" if name == active else ""
                    lines.append(f" - {name}{marker}")
                return "\n".join(lines)

            profile_name = str(args[1]).strip()
            if profile_name not in profiles:
                return (
                    f"Unknown profile '{profile_name}'. "
                    f"Use '/model profile list' to see available profiles."
                )

            GENERATION_PARAMS[profile_key] = profile_name
            context.config.set(profile_key, profile_name)
            context.config.save()
            self._reload_brain(context, current_model)
            return f"Set {profile_key}={profile_name} for model '{current_model}'."

        if args[0] in {"context", "ctx"}:
            ctx_default_key, ctx_cap_key = self._native_context_keys(current_model)
            if len(args) == 1:
                num_ctx = int(GENERATION_PARAMS.get("num_ctx", 400000))
                native_default = int(GENERATION_PARAMS.get(ctx_default_key, num_ctx))
                native_cap = int(GENERATION_PARAMS.get(ctx_cap_key, num_ctx))
                return (
                    f"Context for {current_model}: "
                    f"num_ctx={num_ctx}, {ctx_default_key}={native_default}, {ctx_cap_key}={native_cap}"
                )

            try:
                new_ctx = int(str(args[1]).strip())
            except Exception:
                return "Usage: /model context <positive_int>"
            if new_ctx <= 0:
                return "Usage: /model context <positive_int>"

            GENERATION_PARAMS["num_ctx"] = new_ctx
            GENERATION_PARAMS["native_ctx_default"] = new_ctx
            GENERATION_PARAMS["native_ctx_cap"] = new_ctx
            GENERATION_PARAMS[ctx_default_key] = new_ctx
            GENERATION_PARAMS[ctx_cap_key] = new_ctx

            context.config.set("num_ctx", new_ctx)
            context.config.set("native_ctx_default", new_ctx)
            context.config.set("native_ctx_cap", new_ctx)
            context.config.set(ctx_default_key, new_ctx)
            context.config.set(ctx_cap_key, new_ctx)
            context.config.save()
            self._reload_brain(context, current_model)
            return (
                f"Set context for {current_model}: "
                f"num_ctx={new_ctx}, native_default={new_ctx}, native_cap={new_ctx}"
            )

        new_model = args[0]
        context.config.set("model", new_model)
        context.config.save()
        self._reload_brain(context, new_model)
        profile_key, _ = self._profile_catalog(new_model)
        active = (
            str(GENERATION_PARAMS.get(profile_key))
            if profile_key and GENERATION_PARAMS.get(profile_key) is not None
            else "n/a"
        )
        return f"Model switched to: {new_model} (sampling profile: {active})"


class TreeCommand(SlashCommand):
    @property
    def name(self) -> str:
        return "tree"

    @property
    def description(self) -> str:
        return "Show file tree"

    def execute(self, args: List[str], context: Any) -> Optional[str]:
        # Use simple os walk or tree command if available
        import os

        path = args[0] if args else "."

        tree_str = []
        for root, dirs, files in os.walk(path):
            level = root.replace(path, "").count(os.sep)
            indent = " " * 4 * (level)
            tree_str.append(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 4 * (level + 1)
            for f in files:
                tree_str.append(f"{subindent}{f}")

            # Limit depth/size for sanity
            if len(tree_str) > 50:
                tree_str.append("... (truncated)")
                break

        return "\n".join(tree_str)


class SkillsCommand(SlashCommand):
    @property
    def name(self) -> str:
        return "skills"

    @property
    def description(self) -> str:
        return "Manage skills: list, reload"

    def execute(self, args: List[str], context: Any) -> Optional[str]:
        # context is AgentREPL. It has 'registry' which has 'skill_manager'
        manager = context.registry.skill_manager

        if not args or args[0] == "list":
            skills = manager.list_skills()
            if not skills:
                return "No skills found in ~/.anvil/skills or ./.anvil/skills"

            lines = ["Available Skills:"]
            for s in skills:
                lines.append(f" - {s['name']}: {s['description']}")
                if s["triggers"]:
                    lines.append(f"   Triggers: {', '.join(s['triggers'])}")
            return "\n".join(lines)

        if args[0] == "reload":
            manager.discover_skills()
            return f"Skills reloaded. Found {len(manager.skills)} skills."

        return "Usage: /skills [list|reload]"


class TimelineCommand(SlashCommand):
    @property
    def name(self) -> str:
        return "timeline"

    @property
    def aliases(self) -> List[str]:
        return ["tl"]

    @property
    def description(self) -> str:
        return "Show recent timeline events or export audit bundle"

    def execute(self, args: List[str], context: Any) -> Optional[str]:
        if args and args[0] == "export":
            output_path = args[1] if len(args) > 1 else ".anvil/audit_export.json"
            exported = context.history.export_audit(output_path)
            if exported:
                return f"Audit exported to {exported}"
            return "Audit export unavailable (DB persistence disabled)."

        limit = 20
        if args:
            try:
                limit = max(1, min(int(args[0]), 500))
            except ValueError:
                return "Usage: /timeline [limit] | /timeline export [path]"

        events = context.history.get_timeline(limit=limit)
        if not events:
            return "No timeline events recorded yet."

        lines = [f"Timeline ({len(events)} events):"]
        for event in events:
            payload = event.get("payload") or {}
            lines.append(
                f"- {event.get('wall_clock')} | {event.get('event_type')} | "
                f"elapsed_ms={event.get('monotonic_elapsed_ms')} | "
                f"payload={json.dumps(payload, sort_keys=True)}"
            )
        return "\n".join(lines)
