"""Tiered verification lane for campaign work."""

from __future__ import annotations

from typing import Any

from core.memory.fabric.models import MemoryFeedbackRecord


class VerificationLane:
    """Run tiered verification and feed results back into campaign memory."""

    def __init__(
        self,
        verifier,
        *,
        state_store=None,
        event_store=None,
        memory_fabric=None,
    ) -> None:
        self.verifier = verifier
        self.state_store = state_store
        self.event_store = event_store
        self.memory_fabric = memory_fabric

    def run(
        self,
        changed_files: list[str],
        *,
        campaign_id: str = "",
        task_packet_id: str = "",
        read_id: str = "",
        tier: str = "changed",
    ) -> dict[str, Any]:
        results = self.verifier.verify_changes(changed_files)
        stress_cases = self._stress_cases(
            changed_files,
            runtime_symbols=list(results.get("runtime_symbols", [])),
            counterexamples=list(results.get("counterexamples", [])),
        )
        stress_results = [
            {
                "label": case["label"],
                "files": case["files"],
                "result": self.verifier.verify_changes(case["files"]),
            }
            for case in stress_cases
        ]
        all_counterexamples = sorted(
            {
                str(counterexample)
                for batch in [results, *[item["result"] for item in stress_results]]
                for counterexample in list(batch.get("counterexamples", []))
                if str(counterexample).strip()
            }
        )
        passing_runs = len(
            [
                batch
                for batch in [results, *[item["result"] for item in stress_results]]
                if bool(batch.get("all_passed", False))
            ]
        )
        stress_case_count = 1 + len(stress_results)
        semantic_stability_score = round(
            max(
                0.0,
                min(
                    1.0,
                    passing_runs / max(1, stress_case_count)
                    - min(0.35, len(all_counterexamples) * 0.07),
                ),
            ),
            3,
        )
        promotion_blocked = (
            not bool(results.get("all_passed", False))
            or semantic_stability_score < 0.65
            or bool(all_counterexamples)
        )
        payload = {
            "tier": tier,
            "changed_files": list(changed_files),
            "syntax": results.get("syntax", {}),
            "lint": results.get("lint", {}),
            "changed_tests": results.get("tests", {}),
            "semantic_verify": results.get("sentinel", {}),
            "full_verify": {"passed": bool(results.get("all_passed", False))},
            "all_passed": bool(results.get("all_passed", False)),
            "runtime_symbols": list(results.get("runtime_symbols", [])),
            "counterexamples": all_counterexamples,
            "stress_cases": stress_results,
            "stress_case_count": stress_case_count,
            "semantic_stability_score": semantic_stability_score,
            "promotion_blocked": promotion_blocked,
        }
        if self.state_store is not None and campaign_id:
            self.state_store.record_telemetry(
                campaign_id,
                telemetry_kind="verification_lane",
                payload=payload,
                task_packet_id=task_packet_id or None,
            )
        if self.event_store is not None and campaign_id:
            self.event_store.emit(
                event_type="campaign.verification_lane",
                payload=payload,
                source="VerificationLane",
                run_id=campaign_id,
                links=[
                    {
                        "link_type": "file",
                        "target_type": "file",
                        "target_ref": file_path,
                    }
                    for file_path in changed_files
                ],
            )
        if self.memory_fabric is not None and read_id:
            score = 1.0 if payload["all_passed"] else 0.25
            self.memory_fabric.record_feedback(
                MemoryFeedbackRecord(
                    read_id=read_id,
                    consumer_system="verification_lane",
                    usefulness_score=score,
                    grounding_score=score,
                    citation_score=score,
                    token_savings_estimate=25.0 if payload["all_passed"] else 0.0,
                    outcome_json={
                        "task_packet_id": task_packet_id,
                        "campaign_id": campaign_id,
                        "all_passed": payload["all_passed"],
                        "promotion_blocked": promotion_blocked,
                    },
                )
            )
        return payload

    @staticmethod
    def _stress_cases(
        changed_files: list[str],
        *,
        runtime_symbols: list[str],
        counterexamples: list[str],
    ) -> list[dict[str, Any]]:
        ordered = [str(item) for item in changed_files if str(item).strip()]
        if not ordered:
            return []
        cases: list[dict[str, Any]] = [
            {"label": "sorted_scope", "files": sorted(set(ordered))},
        ]
        if len(ordered) > 1:
            cases.append({"label": "reverse_scope", "files": list(reversed(ordered))})
        if runtime_symbols or counterexamples:
            cases.append({"label": "counterexample_recheck", "files": ordered[:1]})
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for case in cases:
            key = (case["label"], tuple(case["files"]))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(case)
        return deduped[:3]
