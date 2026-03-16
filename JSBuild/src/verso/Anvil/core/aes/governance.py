from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, Optional


class GovernanceTier(str, Enum):
    """Hardcoded AES governance tiers."""

    L0_IMMUTABLE = "immutable"
    L1_GUARDED = "guarded"
    L2_SUPERVISED = "supervised"
    L3_ADVISORY = "advisory"


@dataclass(frozen=True)
class GovernanceResult:
    allowed: bool
    tier: GovernanceTier
    reason: str
    escalation_required: bool = False


class GovernanceEngine:
    """Enforce immutable/guarded/supervised behavior for runtime actions."""

    IMMUTABLE_RULES = [
        "Cannot disable saguaro verify for AAL-0/1 code",
        "Cannot skip Chronicle protocol for AAL-0/1 changes",
        "Cannot delete .saguaro.rules or standards/AES_RULES.json",
        "Cannot introduce bare except:pass in AAL-0/1",
        "Cannot remove type annotations from public APIs",
    ]

    HIGH_AAL = {"AAL-0", "AAL-1"}
    REPO_ROLE_DEFAULTS: Dict[str, GovernanceTier] = {
        "analysis_local": GovernanceTier.L0_IMMUTABLE,
        "analysis_external": GovernanceTier.L0_IMMUTABLE,
        "artifact_store": GovernanceTier.L1_GUARDED,
        "benchmark_fixture": GovernanceTier.L1_GUARDED,
        "target": GovernanceTier.L1_GUARDED,
    }
    QSG_DRIFT_OVERHEAD_LIMIT_PCT = 20.0

    @staticmethod
    def is_high_aal(aal: str) -> bool:
        return str(aal or "AAL-3").upper() in GovernanceEngine.HIGH_AAL

    def get_tier(self, aal: str, action_type: str) -> GovernanceTier:
        normalized_aal = str(aal or "AAL-3").upper()
        normalized_action = str(action_type or "").lower()

        if "architecture" in normalized_action:
            return GovernanceTier.L3_ADVISORY
        if normalized_action == "code_modification" and normalized_aal == "AAL-0":
            return GovernanceTier.L2_SUPERVISED
        if normalized_action == "code_modification" and normalized_aal in {
            "AAL-1",
            "AAL-2",
            "AAL-3",
        }:
            return GovernanceTier.L1_GUARDED
        return GovernanceTier.L3_ADVISORY

    def check_action(
        self,
        action: str,
        aal: str,
        action_type: str = "code_modification",
        waiver_ids: Iterable[str] | None = None,
        qsg_runtime_status: Optional[Dict[str, Any]] = None,
    ) -> GovernanceResult:
        text = str(action or "").lower()
        normalized_aal = str(aal or "AAL-3").upper()
        waiver_set = {
            str(item).strip() for item in (waiver_ids or []) if str(item).strip()
        }
        tier = self.get_tier(normalized_aal, action_type)

        immutable_matchers = (
            "disable saguaro verify",
            "skip chronicle",
            "delete .saguaro.rules",
            "delete standards/aes_rules.json",
            "except:pass",
            "remove type annotation",
        )
        if any(marker in text for marker in immutable_matchers):
            return GovernanceResult(
                allowed=False,
                tier=GovernanceTier.L0_IMMUTABLE,
                reason="AES immutable governance rule violation.",
                escalation_required=True,
            )

        if self.is_high_aal(normalized_aal) and {
            "fallback-waiver",
            "high-aal-override",
        }.isdisjoint(waiver_set):
            fallback_markers = ("fallback", "switch_to_faster_model", "warn-only")
            if any(marker in text for marker in fallback_markers):
                return GovernanceResult(
                    allowed=False,
                    tier=GovernanceTier.L2_SUPERVISED,
                    reason="High-AAL flow cannot downgrade safety without waiver.",
                    escalation_required=True,
                )

        if tier == GovernanceTier.L2_SUPERVISED:
            return GovernanceResult(
                allowed=False,
                tier=tier,
                reason="Supervised approval required for this high-assurance action.",
                escalation_required=True,
            )

        qsg_findings = self._qsg_runtime_findings(qsg_runtime_status)
        if qsg_findings:
            return GovernanceResult(
                allowed=False,
                tier=GovernanceTier.L1_GUARDED,
                reason=(
                    "QSG runtime contract violation: "
                    + "; ".join(qsg_findings[:3])
                ),
                escalation_required=True,
            )

        return GovernanceResult(
            allowed=True,
            tier=tier,
            reason="Action allowed under current governance tier.",
            escalation_required=False,
        )

    def build_task_envelope(
        self,
        *,
        objective: str,
        aal: str,
        action_type: str,
        repo_role: str,
        allowed_repos: Iterable[str],
        allowed_tools: Iterable[str],
        required_artifacts: Iterable[str],
    ) -> Dict[str, object]:
        return {
            "objective": objective,
            "aal": str(aal or "AAL-3").upper(),
            "action_type": action_type,
            "repo_role": repo_role,
            "allowed_repos": list(allowed_repos),
            "allowed_tools": list(allowed_tools),
            "required_artifacts": list(required_artifacts),
            "governance_tier": self.get_tier(aal, action_type).value,
        }

    def check_repo_action(
        self,
        *,
        action: str,
        aal: str,
        repo_role: str,
        runtime_state: str,
        write_policy: str,
        action_type: str = "code_modification",
        waiver_ids: Iterable[str] | None = None,
        qsg_runtime_status: Optional[Dict[str, Any]] = None,
    ) -> GovernanceResult:
        role = str(repo_role or "target").lower()
        state = str(runtime_state or "").upper()
        policy = str(write_policy or "").lower()
        if role in {"analysis_local", "analysis_external"} or policy == "immutable":
            return GovernanceResult(
                allowed=False,
                tier=GovernanceTier.L0_IMMUTABLE,
                reason="Analysis repositories are immutable under campaign governance.",
                escalation_required=True,
            )
        if policy == "phase_gated_write" and state not in {"DEVELOPMENT", "REMEDIATION"}:
            return GovernanceResult(
                allowed=False,
                tier=self.REPO_ROLE_DEFAULTS.get(role, GovernanceTier.L1_GUARDED),
                reason=f"Writes are not allowed in runtime state {state}.",
                escalation_required=True,
            )
        return self.check_action(
            action=action,
            aal=aal,
            action_type=action_type,
            waiver_ids=waiver_ids,
            qsg_runtime_status=qsg_runtime_status,
        )

    def _qsg_runtime_findings(
        self, runtime_status: Optional[Dict[str, Any]]
    ) -> list[str]:
        payload = dict(runtime_status or {})
        if not payload:
            return []
        findings: list[str] = []
        controller_state = dict(payload.get("controller_state") or {})
        repo_runtime = dict(payload.get("repo_coupled_runtime") or {})
        capability_vector = dict(payload.get("capability_vector") or {})
        performance_envelope = dict(payload.get("performance_envelope") or {})
        drift_state = dict(controller_state.get("drift") or {})

        strict_native = bool(
            payload.get("strict_native_qsg")
            or payload.get("strict_path_stable")
            or capability_vector.get("strict_path_stable")
        )
        abi_match = bool(
            payload.get("native_backend_abi_match")
            or capability_vector.get("native_backend_abi_match")
        )
        if strict_native and not abi_match:
            findings.append("strict_native_backend_abi_mismatch")

        if bool(payload.get("backend_module_required")) and not bool(
            payload.get("backend_module_loaded")
        ):
            findings.append("required_backend_module_not_loaded")

        if bool(payload.get("hot_path_numpy_detected")):
            findings.append("python_or_numpy_hot_path_detected")

        drift_overhead_pct = float(
            performance_envelope.get("drift_overhead_percent")
            or payload.get("drift_overhead_percent")
            or 0.0
        )
        if drift_overhead_pct > float(self.QSG_DRIFT_OVERHEAD_LIMIT_PCT):
            findings.append(
                f"drift_overhead_above_limit:{drift_overhead_pct:.2f}%"
            )

        if str(repo_runtime.get("delta_authority") or "").strip() not in {
            "",
            "state_ledger",
        }:
            findings.append("delta_authority_not_state_ledger")

        if int(repo_runtime.get("execution_capsule_version") or 2) < 2:
            findings.append("execution_capsule_abi_outdated")
        if int(repo_runtime.get("latent_packet_abi_version") or 2) < 2:
            findings.append("latent_packet_abi_outdated")

        if drift_state and str(drift_state.get("selected_mode") or "").strip() == "downgrade":
            findings.append("drift_controller_forced_downgrade")

        return findings
