from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable


class IrreversibleActionType(str, Enum):
    FILESYSTEM_DESTRUCTIVE = "filesystem_destructive"
    BRANCH_REWRITE = "branch_rewrite"
    DEPLOYMENT_MUTATION = "deployment_mutation"
    SECRET_WRITE = "secret_write"
    REVERSIBLE = "reversible"


@dataclass(frozen=True)
class ActionEscalationResult:
    allowed: bool
    action_type: IrreversibleActionType
    reason: str
    requires_supervision: bool


class ActionEscalationEngine:
    """Classify irreversible actions and enforce high-AAL escalation policy."""

    HIGH_AAL = {"AAL-0", "AAL-1"}

    def classify(self, tool_name: str, arguments: Dict[str, Any]) -> IrreversibleActionType:
        name = str(tool_name or "").lower()
        args_text = " ".join(f"{k}={v}" for k, v in (arguments or {}).items()).lower()

        if name in {"delete_file", "rollback_file"}:
            return IrreversibleActionType.FILESYSTEM_DESTRUCTIVE
        if name == "run_command" and any(
            marker in args_text
            for marker in (
                "git reset --hard",
                "git push --force",
                "git rebase -i",
            )
        ):
            return IrreversibleActionType.BRANCH_REWRITE
        if name == "run_command" and any(
            marker in args_text
            for marker in (
                "kubectl apply",
                "terraform apply",
                "helm upgrade",
                "systemctl restart",
            )
        ):
            return IrreversibleActionType.DEPLOYMENT_MUTATION
        if name in {"write_file", "edit_file", "write_files"} and any(
            marker in args_text
            for marker in (
                "secret",
                "token",
                "password",
                "api_key",
            )
        ):
            return IrreversibleActionType.SECRET_WRITE
        return IrreversibleActionType.REVERSIBLE

    def evaluate(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        aal: str,
        review_signoff_token: str | None,
        rollback_plan_artifact: str | None,
        waiver_ids: Iterable[str] | None = None,
    ) -> ActionEscalationResult:
        action_type = self.classify(tool_name, arguments)
        normalized_aal = str(aal or "AAL-3").upper()
        waiver_set = {
            str(item).strip() for item in (waiver_ids or []) if str(item).strip()
        }

        if action_type == IrreversibleActionType.REVERSIBLE:
            return ActionEscalationResult(
                allowed=True,
                action_type=action_type,
                reason="Action is reversible.",
                requires_supervision=False,
            )

        if normalized_aal in self.HIGH_AAL and "irreversible-approved" not in waiver_set:
            if not str(rollback_plan_artifact or "").strip():
                return ActionEscalationResult(
                    allowed=False,
                    action_type=action_type,
                    reason="Rollback plan artifact is required for irreversible high-AAL actions.",
                    requires_supervision=True,
                )
            if not str(review_signoff_token or "").strip():
                return ActionEscalationResult(
                    allowed=False,
                    action_type=action_type,
                    reason="Review gate signoff token is required for irreversible high-AAL actions.",
                    requires_supervision=True,
                )

        return ActionEscalationResult(
            allowed=True,
            action_type=action_type,
            reason="Irreversible action approved under current policy.",
            requires_supervision=normalized_aal in self.HIGH_AAL,
        )
