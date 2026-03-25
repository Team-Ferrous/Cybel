#!/usr/bin/env python3
"""Deterministic AES governance chaos drill for fail-closed verification."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from saguaro.sentinel.engines.aes import AESEngine
from saguaro.sentinel.policy import PolicyManager
from saguaro.sentinel.verifier import SentinelVerifier


@dataclass
class DrillScenario:
    name: str
    aal: str
    fail_closed: bool
    blocking_rule_ids: list[str]
    blocking_reasons: list[str]
    remediation: str


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _seed_repo(root: Path) -> None:
    _write(root / "standards" / "AES_RULES.json", "[]")


def _blocking_details(violations: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    blocking = [
        item
        for item in violations
        if str(item.get("closure_level", "")).lower() in {"blocking", "guarded"}
    ]
    return (
        sorted({str(item.get("rule_id", "UNKNOWN")) for item in blocking}),
        [str(item.get("message", "")) for item in blocking],
    )


def _scenario_engine_disabled() -> DrillScenario:
    with tempfile.TemporaryDirectory(prefix="aes-chaos-engine-") as tmp:
        repo = Path(tmp)
        _seed_repo(repo)
        _write(repo / "train.py", "loss.backward()\noptimizer.step()\n")

        requested = ["native", "ruff", "aes"]
        verifier = SentinelVerifier(str(repo), engines=requested)
        violations = verifier.verify_all(
            path_arg=str(repo / "train.py"),
            aal="1",
            require_trace=True,
            require_evidence=True,
            require_valid_waivers=True,
        )
        rule_ids, reasons = _blocking_details(violations)
        semantic_disabled = "semantic" not in requested
        fail_closed = semantic_disabled and any(
            rid in {"AES-TR-1", "AES-TR-2"} for rid in rule_ids
        )
        if semantic_disabled:
            reasons.insert(0, "Semantic engine intentionally disabled for drill.")

        return DrillScenario(
            name="engine_disabled",
            aal="AAL-1",
            fail_closed=fail_closed,
            blocking_rule_ids=rule_ids,
            blocking_reasons=reasons,
            remediation=(
                "Re-enable semantic engine and rerun verify "
                "with full engine set."
            ),
        )


def _scenario_malformed_evidence_bundle() -> DrillScenario:
    with tempfile.TemporaryDirectory(prefix="aes-chaos-evidence-") as tmp:
        repo = Path(tmp)
        _seed_repo(repo)
        _write(repo / "critical.py", "# _mm256_add_ps\n")
        _write(repo / "standards" / "evidence" / "bundle.json", "{not-json")

        engine = AESEngine(str(repo))
        engine.set_policy(
            {
                "block_on_missing_artifacts": True,
                "verify_context": {"require_evidence": True},
            }
        )
        violations = engine.run(path_arg=str(repo / "critical.py"))
        rule_ids, reasons = _blocking_details(violations)

        return DrillScenario(
            name="malformed_evidence_bundle",
            aal="AAL-0",
            fail_closed="AES-TR-2" in rule_ids,
            blocking_rule_ids=rule_ids,
            blocking_reasons=reasons,
            remediation=(
                "Regenerate evidence bundle JSON and ensure "
                "schema-valid payload."
            ),
        )


def _scenario_corrupt_waiver_expiry() -> DrillScenario:
    with tempfile.TemporaryDirectory(prefix="aes-chaos-waiver-") as tmp:
        repo = Path(tmp)
        _seed_repo(repo)
        _write(repo / "critical.py", "# _mm256_add_ps\n")
        waiver = {
            "waiver_id": "waiver-chaos",
            "rule_id": "AES-CR-2",
            "change_scope": "core/high_assurance.py",
            "compensating_control": "manual review",
            "risk_owner": "safety",
            "expiry": (date.today() - timedelta(days=3)).isoformat(),
            "remediation_ticket": "ENG-9999",
        }
        _write(repo / "standards" / "waivers" / "expired.json", json.dumps(waiver))

        engine = AESEngine(str(repo))
        engine.set_policy(
            {
                "block_on_missing_artifacts": False,
                "verify_context": {"require_valid_waivers": True},
            }
        )
        violations = engine.run(path_arg=str(repo / "critical.py"))
        rule_ids, reasons = _blocking_details(violations)

        return DrillScenario(
            name="corrupt_waiver_expiry",
            aal="AAL-0",
            fail_closed="AES-TR-3" in rule_ids,
            blocking_rule_ids=rule_ids,
            blocking_reasons=reasons,
            remediation=(
                "Renew waiver expiry or remove waiver after "
                "remediation closure."
            ),
        )


def _scenario_contradictory_policy_outcome() -> DrillScenario:
    with tempfile.TemporaryDirectory(prefix="aes-chaos-policy-") as tmp:
        repo = Path(tmp)
        policy = PolicyManager(str(repo))
        violations = [
            {
                "file": "critical.py",
                "line": 1,
                "rule_id": "AES-TR-1",
                "message": "Missing traceability records",
                "severity": "P1",
                "aal": "AAL-0",
                "closure_level": "blocking",
            }
        ]
        decision = policy.runtime_decision(violations, aal="AAL-0")
        fail_closed = bool(decision.get("should_fail")) and decision.get(
            "decision"
        ) in {
            "escalate",
            "fail",
        }

        return DrillScenario(
            name="contradictory_policy_outcome",
            aal="AAL-1",
            fail_closed=fail_closed,
            blocking_rule_ids=["AES-TR-1"],
            blocking_reasons=[str(decision.get("reason", ""))],
            remediation=(
                "Align runtime decision policy with closure levels "
                "for high-AAL paths."
            ),
        )


def run_drill(repo_root: str) -> dict[str, Any]:
    scenarios = [
        _scenario_engine_disabled(),
        _scenario_malformed_evidence_bundle(),
        _scenario_corrupt_waiver_expiry(),
        _scenario_contradictory_policy_outcome(),
    ]

    high_aal = [item for item in scenarios if item.aal in {"AAL-0", "AAL-1"}]
    fail_closed_high_aal = all(item.fail_closed for item in high_aal)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "repo": str(Path(repo_root).resolve()),
        "overall_pass": fail_closed_high_aal,
        "fail_closed_high_aal": fail_closed_high_aal,
        "scenario_count": len(scenarios),
        "fail_closed_count": sum(1 for item in scenarios if item.fail_closed),
        "scenarios": [asdict(item) for item in scenarios],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AES governance chaos drill")
    parser.add_argument("--repo", default=".", help="Repository root for metadata only")
    parser.add_argument("--out", help="Optional output JSON path")
    parser.add_argument("--json", action="store_true", help="Emit JSON to stdout")
    args = parser.parse_args()

    payload = run_drill(args.repo)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json or not args.out:
        print(json.dumps(payload, indent=2))

    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
