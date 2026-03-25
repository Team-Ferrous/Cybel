from __future__ import annotations

from uuid import uuid4

from core.campaign.control_plane import CampaignControlPlane
from core.research.hypothesis_ranker import HypothesisRanker


def test_hypothesis_ranker_prices_evidence_determinism_and_risk() -> None:
    ranker = HypothesisRanker()

    ranked = ranker.rank(
        [
            {
                "hypothesis_id": "risky",
                "statement": "Complex telemetry architecture change",
                "source_basis": ["telemetry", "audit"],
                "target_subsystems": ["telemetry"],
                "required_experiments": ["telemetry_contract_replay"],
                "risk": "high implementation complexity",
            },
            {
                "hypothesis_id": "balanced",
                "statement": "Deterministic telemetry contract for campaign runtime",
                "source_basis": ["telemetry", "repo analysis", "determinism", "audit"],
                "target_subsystems": ["campaign_runtime", "telemetry"],
                "required_experiments": [
                    "telemetry_contract_replay",
                    "artifact_resume_replay",
                    "analysis_pack_reuse_benchmark",
                ],
                "risk": "low implementation complexity",
            },
        ],
        objective="Improve native determinism and telemetry evidence.",
    )

    assert ranked[0]["hypothesis_id"] == "balanced"
    assert ranked[0]["promotable"] is True
    assert ranked[0]["innovation_score"] > ranked[1]["innovation_score"]
    assert ranked[0]["score_breakdown"]["determinism_bonus"] == 0.4
    assert ranked[1]["score_breakdown"]["risk_penalty"] == -0.9


def test_hypothesis_lab_generates_evidence_backed_exchange_candidates(tmp_path) -> None:
    control = CampaignControlPlane.create(
        f"eid-exchange-{uuid4().hex[:8]}",
        "EID Exchange",
        str(tmp_path / "campaigns"),
        objective="Improve campaign evidence exchange.",
        root_dir=str(tmp_path / "repo"),
    )

    hypotheses = control.hypothesis_lab.generate(
        "Improve campaign evidence exchange.",
        [
            {"topic": "telemetry", "claim_id": "claim-1"},
            {"topic": "repo analysis", "claim_id": "claim-2"},
        ],
    )
    persisted = control.state_store.fetchall(
        """
        SELECT hypothesis_id
        FROM hypotheses
        WHERE campaign_id = ?
        ORDER BY created_at ASC
        """,
        (control.campaign_id,),
    )

    statements = [item["statement"].lower() for item in hypotheses]

    assert len(hypotheses) == 3
    assert any("telemetry layer" in statement for statement in statements)
    assert any("repo analysis packs" in statement for statement in statements)
    assert len(persisted) == len(hypotheses)
