"""Utilities for policy."""

import logging
import os
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class PolicyManager:
    """Unified Policy Enforcer for Sentinel.
    Evaluates violations against configured policies.
    """

    def __init__(self, repo_path: str) -> None:
        """Initialize the instance."""
        self.repo_path = repo_path
        self.config = self._load_policy()

    def _load_policy(self) -> dict[str, Any]:
        # Check standard location first
        policy_path = os.path.join(self.repo_path, ".saguaro/policy.yaml")
        if not os.path.exists(policy_path):
            # Check legacy location
            policy_path = os.path.join(self.repo_path, ".saguaro.policy.yaml")

        # Default policy
        default_policy = {
            "strict_mode": False,  # If True, violations = failure
            "drift_tolerance": 0.2,  # Semantic drift limit
            "semantic_drift_enabled": True,
            "semantic_decode_mode": "auto",
            "semantic_fail_open_on_decode_error": True,
            "excluded_rules": [],  # List of rule IDs to ignore
            "auto_fix": True,  # Allow auto-fixes by default
            "block_on_missing_artifacts": True,
            "guard_p2_in_aal_01": True,
            "aes_rollout_mode": "ratchet",
            "aes_blocking_rule_ids": [
                "AES-VIS-1",
                "AES-VIS-2",
                "AES-TR-1",
                "AES-TR-2",
                "AES-TR-3",
                "AES-REV-1",
                "AES-CR-2",
                "AES-SUP-1",
                "AES-SUP-2",
                "AES-CPP-2",
                "AES-CPP-3",
                "AES-CPP-4",
                "AES-CPP-5",
                "AES-HPC-2",
                "AES-HPC-3",
            ],
            "aes_guarded_rule_ids": ["AES-CPLX-1", "AES-ERR-1", "AES-ERR-2"],
            "aes_require_change_manifest": True,
            "aes_require_runtime_reports_for_hot_paths": True,
            "aes_authoritative_package_root": "saguaro",
            "aes_excluded_reference_roots": ["Saguaro/"],
            "excluded_paths": [
                ".anvil/",
                ".saguaro/",
                "build/",
                "dist/",
                "venv/",
                "__pycache__/",
                "Saguaro/",
                "core/native/build/",
                "saguaro/native/build_release/",
                "saguaro/native/build_test/",
            ],
        }

        if os.path.exists(policy_path):
            try:
                with open(policy_path) as f:
                    user_policy = yaml.safe_load(f) or {}
                    # Merge (simple override)
                    default_policy.update(user_policy)
            except Exception as e:
                logger.warning(f"Failed to load policy file {policy_path}: {e}")

        return default_policy

    def evaluate(self, violations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Process violations: filter based on policy."""
        active_violations = []
        rules = self.config.get("excluded_rules")
        excluded = set(rules if rules is not None else [])

        for v in violations:
            normalized = self._normalize_violation(v)
            rule_id = v.get("rule_id", "UNKNOWN")

            if rule_id in excluded:
                continue

            active_violations.append(normalized)

        return active_violations

    def should_fail(
        self, violations: list[dict[str, Any]], aal: str | None = None
    ) -> bool:
        """Determine if the process should exit with error."""
        if not violations:
            return False

        normalized_aal = str(aal or "").upper()
        if self.config.get("strict_mode"):
            return True

        for violation in violations:
            closure_level = str(violation.get("closure_level", "")).lower()
            severity = str(violation.get("severity", "")).upper()
            if closure_level == "blocking":
                return True
            if normalized_aal in {"AAL-0", "AAL-1"} and closure_level == "guarded":
                return True
            if severity in {"P0", "P1", "ERROR"}:
                return True
        return False

    def runtime_decision(
        self, violations: list[dict[str, Any]], aal: str | None = None
    ) -> dict[str, Any]:
        """Handle runtime decision."""
        normalized_aal = str(aal or "AAL-3").upper()
        blocking = [
            v
            for v in violations
            if str(v.get("closure_level", "")).lower() == "blocking"
        ]
        guarded = [
            v
            for v in violations
            if str(v.get("closure_level", "")).lower() == "guarded"
        ]
        should_fail = self.should_fail(violations, aal=normalized_aal)
        decision = "continue"
        reason = "no blocking policy outcomes"
        if should_fail:
            decision = "escalate" if normalized_aal in {"AAL-0", "AAL-1"} else "fail"
            reason = "blocking_or_guarded_policy_outcome"
        return {
            "decision": decision,
            "reason": reason,
            "aal": normalized_aal,
            "blocking_count": len(blocking),
            "guarded_count": len(guarded),
            "should_fail": should_fail,
        }

    def _normalize_violation(self, violation: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(violation)
        aal = str(normalized.get("aal") or "AAL-3").upper()
        severity = self._normalize_severity(str(normalized.get("severity", "")))
        normalized["severity"] = severity
        rule_id = str(normalized.get("rule_id", "UNKNOWN"))
        aes_status = str(normalized.get("status", "") or "").strip().lower()

        closure = normalized.get("closure_level")
        if not closure:
            closure = self._derive_closure_level(
                severity=severity,
                aal=aal,
                rule_id=rule_id,
            )
        closure = self._apply_aes_rollout(
            rule_id=rule_id,
            closure_level=str(closure).lower(),
            aes_status=aes_status,
        )
        normalized["closure_level"] = str(closure).lower()

        if "domain" not in normalized or not normalized["domain"]:
            normalized["domain"] = ["universal"]
        if "evidence_refs" not in normalized or normalized["evidence_refs"] is None:
            normalized["evidence_refs"] = []
        if aes_status:
            normalized["status"] = aes_status

        return normalized

    def _apply_aes_rollout(
        self, rule_id: str, closure_level: str, aes_status: str
    ) -> str:
        rollout_mode = str(self.config.get("aes_rollout_mode", "ratchet")).lower()
        blocking = set(self.config.get("aes_blocking_rule_ids", []) or [])
        guarded = set(self.config.get("aes_guarded_rule_ids", []) or [])

        if rule_id in blocking:
            return "blocking"
        if rule_id in guarded:
            return "guarded"
        if not aes_status:
            return closure_level

        if rollout_mode == "opt_in":
            if aes_status in {"blocking", "guarded"}:
                return "advisory"
            return "advisory"

        if rollout_mode == "ratchet":
            if aes_status == "blocking":
                return "blocking"
            if aes_status == "guarded":
                return "guarded"
            return "advisory"

        if rollout_mode == "fail_closed":
            if aes_status in {"advisory_pending_threshold", "advisory"}:
                return (
                    "guarded"
                    if aes_status == "advisory_pending_threshold"
                    else closure_level
                )
            return aes_status or closure_level

        return closure_level

    def _normalize_severity(self, severity: str) -> str:
        sev = (severity or "").upper()
        if sev in {"P0", "AAL-0"}:
            return "P0"
        if sev in {"P1", "AAL-1", "ERROR"}:
            return "P1"
        if sev in {"P2", "AAL-2", "WARN", "WARNING"}:
            return "P2"
        if sev in {"P3", "AAL-3", "INFO"}:
            return "P3"
        return "P2"

    def _derive_closure_level(self, severity: str, aal: str, rule_id: str) -> str:
        if severity in {"P0", "P1"}:
            return "blocking"
        if severity == "P2":
            if self.config.get("guard_p2_in_aal_01", True) and aal in {
                "AAL-0",
                "AAL-1",
            }:
                return "guarded"
            return "advisory"

        if (
            self.config.get("block_on_missing_artifacts")
            and rule_id in {"AES-TR-1", "AES-TR-2", "AES-TR-3", "AES-REV-1"}
            and aal in {"AAL-0", "AAL-1"}
        ):
            return "blocking"

        return "advisory"
