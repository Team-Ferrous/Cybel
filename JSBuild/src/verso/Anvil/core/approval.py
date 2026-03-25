from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm


class ApprovalMode(Enum):
    SUGGEST = "suggest"  # Ask for everything
    AUTO_EDIT = "auto-edit"  # Auto-approve edits, ask for shell/dangerous
    FULL_AUTO = "full-auto"  # Auto-approve everything unless explicitly denied by policy
    PARANOID = "paranoid"  # Ask for everything including reads
    TRUSTED = "trusted"  # Log actions, never block unless policy denies


class PolicyProfile(Enum):
    TRUSTED = "trusted"
    BALANCED = "balanced"
    STRICT = "strict"
    REGULATED = "regulated"


READ_ONLY_TOOLS: Set[str] = {
    "read_file",
    "read_files",
    "list_dir",
    "skeleton",
    "slice",
    "query",
    "saguaro_query",
    "impact",
    "lsp_definition",
    "lsp_references",
    "lsp_diagnostics",
    "glob",
    "grep",
    "grep_search",
    "find_by_name",
    "list_backups",
}

WRITE_TOOLS: Set[str] = {
    "write_file",
    "edit_file",
    "write_files",
    "delete_file",
    "move_file",
    "apply_patch",
    "rollback_file",
}

SHELL_TOOLS: Set[str] = {
    "run_command",
    "send_command_input",
}

NETWORK_TOOLS: Set[str] = {
    "web_search",
    "web_fetch",
    "browser_visit",
    "browser_click",
    "browser_screenshot",
    "search_news",
    "search_finance",
    "search_arxiv",
    "fetch_arxiv_paper",
    "search_scholar",
    "search_patents",
    "search_reddit",
    "search_hackernews",
    "search_stackoverflow",
}


@dataclass
class PolicyConfig:
    profile: PolicyProfile
    allowed_tools: Optional[Set[str]] = None
    forbidden_paths: Set[str] = field(default_factory=set)
    network_policy: str = "allow"  # allow|restricted|deny
    max_command_runtime: int = 30
    max_diff_size: int = 120_000
    require_approval_for_shell: bool = True
    require_approval_for_write: bool = False


@dataclass
class PolicyDecision:
    allowed: bool
    requires_approval: bool
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _normalize_mode(mode: ApprovalMode | str) -> ApprovalMode:
    if isinstance(mode, ApprovalMode):
        return mode
    return ApprovalMode(str(mode))


def _normalize_profile(profile: PolicyProfile | str | None) -> PolicyProfile:
    if profile is None:
        return PolicyProfile.BALANCED
    if isinstance(profile, PolicyProfile):
        return profile
    return PolicyProfile(str(profile))


def _default_policy(profile: PolicyProfile) -> PolicyConfig:
    if profile == PolicyProfile.TRUSTED:
        return PolicyConfig(
            profile=profile,
            network_policy="restricted",
            max_command_runtime=45,
            require_approval_for_shell=False,
            require_approval_for_write=False,
        )
    if profile == PolicyProfile.STRICT:
        return PolicyConfig(
            profile=profile,
            network_policy="deny",
            max_command_runtime=20,
            require_approval_for_shell=True,
            require_approval_for_write=True,
        )
    if profile == PolicyProfile.REGULATED:
        return PolicyConfig(
            profile=profile,
            network_policy="deny",
            max_command_runtime=15,
            max_diff_size=40_000,
            require_approval_for_shell=True,
            require_approval_for_write=True,
        )

    # BALANCED default
    return PolicyConfig(
        profile=profile,
        network_policy="restricted",
        max_command_runtime=30,
        require_approval_for_shell=True,
        require_approval_for_write=False,
    )


class ApprovalManager:
    """Policy-aware approval manager with auditable decisions."""

    def __init__(
        self,
        mode: ApprovalMode = ApprovalMode.FULL_AUTO,
        *,
        policy_profile: PolicyProfile | str | None = None,
        policy_config: Optional[PolicyConfig] = None,
        audit_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        session_id: Optional[str] = None,
    ):
        self.mode = _normalize_mode(mode)
        self.console = Console()
        self.policy_profile = _normalize_profile(policy_profile)
        self.policy = policy_config or _default_policy(self.policy_profile)
        self.audit_callback = audit_callback
        self.session_id = session_id
        self.prompt_for_approval = os.getenv("ANVIL_PROMPT_APPROVAL", "0") == "1"

    def set_mode(self, mode: ApprovalMode | str) -> None:
        self.mode = _normalize_mode(mode)
        self.console.print(
            f"[bold yellow]Approval mode set to: {self.mode.value}[/bold yellow]"
        )

    def set_policy_profile(self, profile: PolicyProfile | str) -> None:
        self.policy_profile = _normalize_profile(profile)
        self.policy = _default_policy(self.policy_profile)
        self.console.print(
            f"[bold yellow]Policy profile set to: {self.policy_profile.value}[/bold yellow]"
        )

    def can_execute(self, tool_name: str, tool_args: Dict[str, Any]) -> bool:
        decision = self.evaluate(tool_name, tool_args)
        return decision.allowed and not decision.requires_approval

    def _extract_candidate_paths(self, tool_args: Dict[str, Any]) -> Set[str]:
        candidates: Set[str] = set()
        for key in ("path", "src", "dst", "cwd", "file_path"):
            value = tool_args.get(key)
            if isinstance(value, str):
                candidates.add(value)

        files_map = tool_args.get("files")
        if isinstance(files_map, dict):
            for file_path in files_map.keys():
                if isinstance(file_path, str):
                    candidates.add(file_path)

        return candidates

    def _violates_forbidden_path(self, tool_args: Dict[str, Any]) -> Optional[str]:
        if not self.policy.forbidden_paths:
            return None

        for raw_path in self._extract_candidate_paths(tool_args):
            normalized = str(Path(raw_path))
            for forbidden in self.policy.forbidden_paths:
                try:
                    if Path(normalized).is_absolute():
                        if str(Path(normalized)).startswith(str(Path(forbidden))):
                            return normalized
                    elif normalized.startswith(str(Path(forbidden))):
                        return normalized
                except Exception:
                    continue
        return None

    def _estimate_diff_size(self, tool_args: Dict[str, Any]) -> int:
        text_fields = ["content", "new_content", "patch", "command"]
        total = 0
        for key in text_fields:
            val = tool_args.get(key)
            if isinstance(val, str):
                total += len(val)
        files_map = tool_args.get("files")
        if isinstance(files_map, dict):
            total += sum(len(str(v)) for v in files_map.values())
        return total

    def _command_runtime(self, tool_args: Dict[str, Any]) -> Optional[int]:
        runtime_keys = ("timeout", "max_runtime", "runtime", "yield_time_ms")
        for key in runtime_keys:
            val = tool_args.get(key)
            if val is None:
                continue
            try:
                if key.endswith("_ms"):
                    return max(1, int(val) // 1000)
                return int(val)
            except (TypeError, ValueError):
                continue
        return None

    def evaluate(self, tool_name: str, tool_args: Dict[str, Any]) -> PolicyDecision:
        args = tool_args or {}

        if self.policy.allowed_tools is not None and tool_name not in self.policy.allowed_tools:
            return PolicyDecision(
                allowed=False,
                requires_approval=False,
                reason=f"Tool '{tool_name}' is not allowed by policy.",
                metadata={"rule": "allowed_tools"},
            )

        forbidden_hit = self._violates_forbidden_path(args)
        if forbidden_hit:
            return PolicyDecision(
                allowed=False,
                requires_approval=False,
                reason=f"Path '{forbidden_hit}' is forbidden by policy.",
                metadata={"rule": "forbidden_paths", "path": forbidden_hit},
            )

        if self.policy.network_policy == "deny" and tool_name in NETWORK_TOOLS:
            return PolicyDecision(
                allowed=False,
                requires_approval=False,
                reason=f"Network tool '{tool_name}' blocked by network policy.",
                metadata={"rule": "network_policy"},
            )

        runtime = self._command_runtime(args)
        if tool_name in SHELL_TOOLS and runtime is not None and runtime > self.policy.max_command_runtime:
            return PolicyDecision(
                allowed=False,
                requires_approval=False,
                reason=(
                    f"Requested runtime {runtime}s exceeds policy max "
                    f"{self.policy.max_command_runtime}s."
                ),
                metadata={"rule": "max_command_runtime", "requested": runtime},
            )

        estimated_size = self._estimate_diff_size(args)
        if tool_name in WRITE_TOOLS and estimated_size > self.policy.max_diff_size:
            return PolicyDecision(
                allowed=False,
                requires_approval=False,
                reason=(
                    f"Proposed write payload ({estimated_size} chars) exceeds policy "
                    f"max_diff_size ({self.policy.max_diff_size})."
                ),
                metadata={"rule": "max_diff_size", "estimated_size": estimated_size},
            )

        # Approval semantics are mode + profile aware.
        if self.mode == ApprovalMode.PARANOID or self.policy_profile == PolicyProfile.REGULATED:
            return PolicyDecision(
                allowed=True,
                requires_approval=True,
                reason="Paranoid/regulated mode requires explicit approval.",
            )

        if self.mode == ApprovalMode.SUGGEST:
            return PolicyDecision(
                allowed=True,
                requires_approval=True,
                reason="Suggest mode requires approval for every tool call.",
            )

        if tool_name in SHELL_TOOLS and self.policy.require_approval_for_shell:
            return PolicyDecision(
                allowed=True,
                requires_approval=True,
                reason="Shell execution requires approval by policy.",
                metadata={"rule": "require_approval_for_shell"},
            )

        if tool_name in WRITE_TOOLS and self.policy.require_approval_for_write:
            return PolicyDecision(
                allowed=True,
                requires_approval=True,
                reason="Write operation requires approval by policy.",
                metadata={"rule": "require_approval_for_write"},
            )

        if self.mode == ApprovalMode.AUTO_EDIT and tool_name in SHELL_TOOLS:
            return PolicyDecision(
                allowed=True,
                requires_approval=True,
                reason="AUTO_EDIT blocks shell execution without explicit approval.",
            )

        return PolicyDecision(allowed=True, requires_approval=False, reason="Allowed")

    def requires_approval(self, tool_name: str, tool_args: Dict[str, Any]) -> bool:
        decision = self.evaluate(tool_name, tool_args)
        return decision.allowed and decision.requires_approval

    def _sign_event(self, payload: Dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _emit_audit(
        self,
        *,
        tool_name: str,
        tool_args: Dict[str, Any],
        decision: PolicyDecision,
        final_decision: str,
    ) -> None:
        if self.audit_callback is None:
            return
        event = {
            "session_id": self.session_id,
            "tool_name": tool_name,
            "decision": final_decision,
            "reason": decision.reason,
            "policy_profile": self.policy_profile.value,
            "mode": self.mode.value,
            "metadata": decision.metadata,
            "tool_args": tool_args,
        }
        event["signature"] = self._sign_event(event)
        try:
            self.audit_callback(event)
        except Exception:
            # Never fail the main flow because audit sink failed.
            pass

    def request_approval(self, tool_name: str, tool_args: Dict[str, Any]) -> bool:
        decision = self.evaluate(tool_name, tool_args)

        if not decision.allowed:
            self.console.print(
                f"[bold red]Policy Denied[/bold red] {tool_name}: {decision.reason}"
            )
            self._emit_audit(
                tool_name=tool_name,
                tool_args=tool_args,
                decision=decision,
                final_decision="denied",
            )
            return False

        if not decision.requires_approval:
            self._emit_audit(
                tool_name=tool_name,
                tool_args=tool_args,
                decision=decision,
                final_decision="auto-approved",
            )
            return True

        self.console.print(
            Panel(
                f"Tool: {tool_name}\nReason: {decision.reason}",
                title="Approval Required",
                border_style="yellow",
            )
        )

        approved = True
        if self.prompt_for_approval:
            approved = Confirm.ask("Execute this tool?", default=False)

        self._emit_audit(
            tool_name=tool_name,
            tool_args=tool_args,
            decision=decision,
            final_decision="approved" if approved else "rejected",
        )
        return approved

    def simulate(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        decision = self.evaluate(tool_name, tool_args)
        return {
            "tool": tool_name,
            "allowed": decision.allowed,
            "requires_approval": decision.requires_approval,
            "reason": decision.reason,
            "profile": self.policy_profile.value,
            "mode": self.mode.value,
            "metadata": decision.metadata,
        }
