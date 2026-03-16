"""Simulator and inverse-design planning hooks."""

from __future__ import annotations

from typing import Any


class SimulatorPlanner:
    """Suggests simulator-first work when evidence points to high uncertainty."""

    @staticmethod
    def estimate_shadow_preflight(
        *,
        required_metrics: list[str],
        command_count: int,
        shell_commands: int,
        timeout_budget_seconds: float,
        historical_verdicts: list[dict[str, Any]] | None = None,
        historical_counterexamples: list[str] | None = None,
        predicted_artifacts: list[str] | None = None,
    ) -> dict[str, Any]:
        known_metrics = {
            "correctness_pass",
            "determinism_pass",
            "replayability",
            "telemetry_contract_satisfied",
            "artifact_replayability",
            "benchmark_replayability",
            "simulation_parity",
            "latency_delta",
            "throughput_delta",
            "candidate_rank_stability",
        }
        required = [str(metric) for metric in required_metrics if str(metric).strip()]
        predicted_contract_gaps = sorted(
            {metric for metric in required if metric not in known_metrics}
        )
        historical = list(historical_verdicts or [])
        verification_failures = len(
            [
                item
                for item in historical
                if not bool(item.get("all_passed", item.get("verify_result", True)))
            ]
        )
        counterexamples = [str(item) for item in list(historical_counterexamples or []) if str(item).strip()]
        missing_artifacts = [
            artifact
            for artifact in list(predicted_artifacts or [])
            if artifact not in {"experiments", "telemetry", "whitepapers", "closure"}
        ]
        command_risk = min(
            0.45,
            command_count * 0.07
            + shell_commands * 0.09
            + min(0.18, timeout_budget_seconds / 600.0),
        )
        historical_penalty = min(0.3, verification_failures * 0.08)
        counterexample_penalty = min(0.25, len(counterexamples) * 0.05)
        contract_penalty = min(0.3, len(predicted_contract_gaps) * 0.08)
        artifact_penalty = min(0.2, len(missing_artifacts) * 0.05)
        shadow_success_probability = round(
            max(
                0.05,
                min(
                    0.98,
                    0.92
                    - command_risk
                    - historical_penalty
                    - counterexample_penalty
                    - contract_penalty
                    - artifact_penalty,
                ),
            ),
            3,
        )
        expected_value_density = round(
            max(
                0.0,
                shadow_success_probability
                / max(1.0, command_count + shell_commands * 0.5),
            ),
            3,
        )
        historical_failure_modes = sorted(
            {
                *(item for item in counterexamples),
                *(
                    f"missing_metric:{metric}"
                    for row in historical
                    for metric in list(row.get("missing_metrics") or [])
                ),
            }
        )[:6]
        return {
            "shadow_success_probability": shadow_success_probability,
            "predicted_contract_gaps": predicted_contract_gaps,
            "predicted_runtime_cost": round(
                max(0.05, command_count * 0.2 + timeout_budget_seconds * 0.01),
                3,
            ),
            "predicted_missing_artifacts": missing_artifacts,
            "historical_failure_modes": historical_failure_modes,
            "portfolio_budget": round(max(1.0, command_count + timeout_budget_seconds / 60.0), 3),
            "expected_value_density": expected_value_density,
        }

    def plan(
        self,
        objective: str,
        hypotheses: list[dict[str, Any]] | Any,
        *,
        funded_proposals: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        objective_text = objective.lower()
        plans: list[dict[str, Any]] = []
        hypothesis_list = list(hypotheses)
        funded = list(funded_proposals or [])
        average_cost = (
            sum(float(item.get("execution_cost_estimate") or 0.0) for item in funded) / max(1, len(funded))
            if funded
            else 1.0
        )
        preflight = self.estimate_shadow_preflight(
            required_metrics=[
                metric
                for proposal in funded
                for metric in list(
                    (proposal.get("telemetry_contract") or {}).get("required_metrics") or []
                )
            ],
            command_count=max(1, len(funded)),
            shell_commands=0,
            timeout_budget_seconds=max(30.0, average_cost * 20.0),
            historical_verdicts=[],
            historical_counterexamples=[
                str(proposal.get("funding_rationale") or "")
                for proposal in funded
                if "counterexample" in str(proposal.get("funding_rationale") or "").lower()
            ],
            predicted_artifacts=[
                artifact
                for proposal in funded
                for artifact in list(proposal.get("required_artifacts") or [])
            ],
        )

        if any(term in objective_text for term in {"simulator", "hardware", "mechanical", "physical"}):
            plans.append(
                {
                    "plan_id": "simulator_first",
                    "title": "Build simulator-first validation surface",
                    "rationale": (
                        "Objective implies hardware-sensitive or physical behavior; "
                        "simulation reduces uncertainty before implementation."
                    ),
                    "required_artifacts": ["experiments", "telemetry"],
                    "success_metrics": ["simulation_parity", "benchmark_replayability"],
                    "shadow_success_probability": preflight["shadow_success_probability"],
                    "predicted_contract_gaps": preflight["predicted_contract_gaps"],
                    "predicted_runtime_cost": preflight["predicted_runtime_cost"],
                    "historical_failure_modes": preflight["historical_failure_modes"],
                }
            )

        if any("hardware" in str(item.get("statement", "")).lower() for item in hypothesis_list):
            plans.append(
                {
                    "plan_id": "hardware_fit_eval",
                    "title": "Evaluate hardware-fit through synthetic benchmark harness",
                    "rationale": (
                        "Hypotheses include hardware-fit concerns that should be measured "
                        "before promotion."
                    ),
                    "required_artifacts": ["telemetry", "roadmap_draft"],
                    "success_metrics": ["latency_delta", "throughput_delta"],
                    "shadow_success_probability": max(
                        0.1,
                        round(preflight["shadow_success_probability"] - 0.05, 3),
                    ),
                    "predicted_contract_gaps": preflight["predicted_contract_gaps"],
                    "predicted_runtime_cost": preflight["predicted_runtime_cost"],
                    "historical_failure_modes": preflight["historical_failure_modes"],
                }
            )

        if any(
            "inverse" in str(item.get("statement", "")).lower()
            or "search space" in str(item.get("motivation", "")).lower()
            for item in hypothesis_list
        ):
            plans.append(
                {
                    "plan_id": "inverse_design_loop",
                    "title": "Model design space before committing implementation order",
                    "rationale": (
                        "Hypotheses indicate search-space uncertainty that benefits from "
                        "inverse-design or constrained simulation."
                    ),
                    "required_artifacts": ["experiments", "whitepapers"],
                    "success_metrics": ["candidate_rank_stability"],
                    "shadow_success_probability": max(
                        0.1,
                        round(preflight["shadow_success_probability"] - 0.08, 3),
                    ),
                    "predicted_contract_gaps": preflight["predicted_contract_gaps"],
                    "predicted_runtime_cost": preflight["predicted_runtime_cost"],
                    "historical_failure_modes": preflight["historical_failure_modes"],
                }
            )

        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for plan in plans:
            if plan["plan_id"] not in seen:
                seen.add(plan["plan_id"])
                deduped.append(plan)
        return deduped
