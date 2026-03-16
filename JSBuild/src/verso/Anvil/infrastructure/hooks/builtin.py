from typing import Dict, Any
from infrastructure.hooks.base import Hook
from core.aes import AALClassifier, DomainDetector


class ToolAuditHook(Hook):
    """Audit all tool calls for security and logging."""

    @property
    def name(self) -> str:
        return "tool_audit"

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context.get("tool_name")
        context.get("tool_args")

        # Simple print audit for now (could log to file)
        # We use print because we might not have access to a rich console here
        # or we want it raw.
        # print(f"[AUDIT] Tool used: {tool_name} Args: {str(tool_args)[:50]}...")

        return context


class PrivacySafetyHook(Hook):
    """Prevent accidental leakage of secrets."""

    @property
    def name(self) -> str:
        return "privacy_safety"

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        tool_args = context.get("tool_args", {})

        # Check for simple keyword patterns in args
        # This is a basic example
        sensitive = ["api_key", "password", "secret"]
        for k, v in tool_args.items():
            if any(s in k.lower() for s in sensitive):
                # Mask it
                tool_args[k] = "***MASKED***"

        context["tool_args"] = tool_args
        return context


class TimingHook(Hook):
    """Measure execution time of tools."""

    @property
    def name(self) -> str:
        return "timing"

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # If pre_tool_use, mark start time
        import time

        start_time = context.get("start_time")

        # Check if we need to SET start time (pre-hook) or CALCULATE elapsed (post-hook)
        # start_time is None or missing means this is the pre-hook call
        if start_time is None or not isinstance(start_time, (int, float)):
            # Pre-tool: set start time
            context["start_time"] = time.time()
        else:
            # Post-tool: calculate elapsed - start_time is guaranteed to be numeric here
            elapsed = time.time() - start_time
            context["elapsed"] = elapsed
            # Reset start_time for next tool call
            context["start_time"] = None

        return context


class SaguaroSyncHook(Hook):
    """Automatically sync Saguaro index after codebase modifications."""

    @property
    def name(self) -> str:
        return "saguaro_sync"

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = context.get("tool_name")
        tool_args = context.get("tool_args", {}) or {}
        # Check for modify tools
        modifying_tools = [
            "write_file",
            "edit_file",
            "apply_patch",
            "write_files",
            "delete_file",
            "move_file",
            "rollback_file",
        ]

        if tool_name in modifying_tools:
            agent = context.get("agent")
            if not agent:
                return context

            changed_files: list[str] = []
            deleted_files: list[str] = []
            if hasattr(agent, "_extract_write_targets"):
                changed_files.extend(agent._extract_write_targets(tool_name, tool_args))

            if tool_name == "delete_file":
                deleted_files.extend(changed_files)
                changed_files = []
            elif tool_name == "move_file":
                src = tool_args.get("src")
                dst = tool_args.get("dst")
                changed_files = [dst] if isinstance(dst, str) and dst else []
                deleted_files = [src] if isinstance(src, str) and src else []

            substrate = getattr(getattr(agent, "registry", None), "saguaro", None)
            if substrate and hasattr(substrate, "sync"):
                try:
                    sync_payload = substrate.sync(
                        changed_files=changed_files,
                        deleted_files=deleted_files,
                        full=False,
                        reason=f"auto:{tool_name}",
                    )
                    context["saguaro_sync"] = sync_payload
                    if hasattr(agent, "console") and agent.console:
                        agent.console.print(
                            f"[dim cyan]Saguaro sync updated after {tool_name}[/dim cyan]"
                        )
                except Exception:
                    pass

        return context


class AALClassifyHook(Hook):
    """Attach deterministic AAL/domain metadata before subagent dispatch."""

    def __init__(self):
        self._aal_classifier = AALClassifier()
        self._domain_detector = DomainDetector()

    @property
    def name(self) -> str:
        return "aal_classify"

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        task_text = str(context.get("task") or "").strip()
        if not task_text:
            return context
        context.setdefault("aal", self._aal_classifier.classify_from_description(task_text))
        context.setdefault(
            "domains",
            sorted(self._domain_detector.detect_from_description(task_text)),
        )
        return context


class ChronicleHook(Hook):
    """Record Chronicle telemetry for post-dispatch governed tasks."""

    @property
    def name(self) -> str:
        return "chronicle"

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # This hook records metadata only; command execution remains in orchestrator.
        task = context.get("task")
        trace_id = context.get("trace_id")
        context["chronicle_receipt"] = {
            "task": task,
            "trace_id": trace_id,
            "recorded": bool(task),
        }
        return context
