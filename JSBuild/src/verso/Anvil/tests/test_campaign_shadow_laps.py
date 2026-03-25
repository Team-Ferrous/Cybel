from __future__ import annotations

import sys
from uuid import uuid4

from core.campaign.control_plane import CampaignControlPlane
from core.research.simulator_planner import SimulatorPlanner


class _GhostVerifier:
    def verify_changes(self, modified_files):
        return {
            "syntax": {"passed": True, "errors": []},
            "tests": {"passed": True, "output": "", "skipped": False},
            "lint": {"passed": True, "warnings": []},
            "sentinel": {"passed": True, "violations": []},
            "all_passed": True,
            "runtime_symbols": [],
            "counterexamples": [],
        }


def test_simulator_planner_emits_shadow_lap_candidates() -> None:
    planner = SimulatorPlanner()

    plans = planner.plan(
        "Build simulator-first hardware optimization loop.",
        [
            {
                "statement": "Hardware telemetry hypothesis",
                "motivation": "Search space uncertainty should be reduced first.",
            },
            {
                "statement": "Inverse design candidate",
                "motivation": "Search space uncertainty benefits from inverse reasoning.",
            },
        ],
    )

    plan_ids = {plan["plan_id"] for plan in plans}

    assert {"simulator_first", "hardware_fit_eval", "inverse_design_loop"} <= plan_ids


def test_high_risk_speculation_runs_ghost_verifier(tmp_path) -> None:
    control = CampaignControlPlane.create(
        f"shadow-lap-{uuid4().hex[:8]}",
        "Shadow Laps",
        str(tmp_path / "campaigns"),
        objective="Kill bad branches before promotion.",
        root_dir=str(tmp_path),
    )
    editable = tmp_path / "editable.txt"
    editable.write_text("source\n", encoding="utf-8")

    control.state_store.record_roadmap_item(
        {
            "campaign_id": control.campaign_id,
            "item_id": "roadmap_shadow_lane",
            "phase_id": "development",
            "title": "Shadow-lap lane",
            "type": "experiment_lane",
            "repo_scope": ["target"],
            "owner_type": "ExperimentLane",
            "depends_on": [],
            "description": "Run speculative branches through a ghost verifier.",
            "objective": "Compare control and candidate lane.",
            "success_metrics": ["correctness_pass"],
            "required_evidence": [],
            "required_artifacts": ["experiments"],
            "telemetry_contract": {
                "required_metrics": ["correctness_pass", "determinism_pass"],
                "minimum_success_count": 1,
            },
            "allowed_writes": ["target"],
            "promotion_gate": {"minimum_score": 0.0},
            "exit_gate": {"branch_count": 2},
            "metadata": {
                "editable_scope": ["editable.txt"],
                "risk": {"risk_level": "high"},
                "speculation_variants": [
                    {
                        "name": "control",
                        "commands": [
                            {
                                "label": "control",
                                "argv": [
                                    sys.executable,
                                    "-c",
                                    "print('correctness_pass=1\\ndeterminism_pass=1')",
                                ],
                            }
                        ],
                    },
                    {
                        "name": "candidate",
                        "commands": [
                            {
                                "label": "candidate",
                                "argv": [
                                    sys.executable,
                                    "-c",
                                    "from pathlib import Path; Path('editable.txt').write_text('candidate\\n', encoding='utf-8'); print('correctness_pass=1\\ndeterminism_pass=1')",
                                ],
                            }
                        ],
                    },
                ],
            },
        }
    )

    comparison = control.run_speculative_roadmap_item(
        "roadmap_shadow_lane",
        verifier=_GhostVerifier(),
    )

    assert comparison["ghost_verifier"]["tier"] == "ghost_high_risk"
    assert comparison["ghost_verifier"]["all_passed"] is True
    assert len(comparison["branches"]) == 2
    assert set(comparison["comparison_metrics"]["verify_result"]) == {
        branch["lane_id"] for branch in comparison["branches"]
    }
