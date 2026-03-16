"""Campaign transition policy for managed autonomy workspaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TransitionDecision:
    current_state: str
    target_state: str
    cause: str
    action: str = "set_state"
    payload: dict[str, Any] = field(default_factory=dict)
    loop_id: str = ""


class CampaignTransitionPolicy:
    """Resolve campaign transitions from live state instead of inline branching."""

    def decide(self, control_plane: Any) -> TransitionDecision:
        snapshot = control_plane.snapshot()
        current = control_plane.state_store.get_campaign(control_plane.campaign_id) or {}
        state = str(
            current.get("current_state") or current.get("runtime_state") or "INTAKE"
        )
        artifact_map = control_plane._artifact_map()
        next_loop = control_plane.loop_scheduler.choose_next(state=state)
        loop_id = next_loop.loop_id if next_loop is not None else ""

        if state == "INTAKE":
            return TransitionDecision(
                current_state=state,
                target_state="REPO_INGESTION",
                cause="operator_continue",
                payload={"attached_repo_count": len(snapshot.get("repos", []))},
            )

        if state in {"REPO_INGESTION", "REPO_ACQUISITION"}:
            return TransitionDecision(
                current_state=state,
                target_state="RESEARCH",
                cause="repo_ingestion_accepted"
                if state == "REPO_INGESTION"
                else "repo_acquisition_accepted",
                payload={"loop_id": loop_id},
            )

        if state == "RESEARCH":
            return TransitionDecision(
                current_state=state,
                target_state="RESEARCH_RECONCILIATION",
                cause="research_digest_built",
                action="run_research",
                payload={"loop_id": loop_id},
                loop_id=loop_id,
            )

        if state == "RESEARCH_RECONCILIATION":
            return TransitionDecision(
                current_state=state,
                target_state="QUESTIONNAIRE_WAIT",
                cause="questionnaire_built",
                action="build_questionnaire",
            )

        if state == "QUESTIONNAIRE_WAIT":
            blocking_questions = control_plane.questionnaire.pending_blockers()
            top_blocker = blocking_questions[0] if blocking_questions else {}
            architecture_artifact = artifact_map.get(
                f"{control_plane.campaign_id}:architecture_questionnaire"
            )
            approved = bool(
                architecture_artifact
                and architecture_artifact["approval_state"] in {"approved", "accepted"}
            )
            if blocking_questions and not approved:
                raise ValueError(
                    "Blocking questionnaire items must be resolved or accepted before continuing: "
                    f"{top_blocker.get('question_id', 'unknown')}"
                )
            return TransitionDecision(
                current_state=state,
                target_state="FEATURE_MAP_WAIT",
                cause="feature_map_built",
                action="build_feature_map",
                payload={
                    "blocking_questions": len(blocking_questions),
                    "top_blocker_question_id": str(top_blocker.get("question_id") or ""),
                    "top_blocker_score": float(
                        (top_blocker.get("metadata") or {}).get("question_value_score") or 0.0
                    ),
                },
            )

        if state == "FEATURE_MAP_WAIT":
            feature_artifact = artifact_map.get(f"{control_plane.campaign_id}:feature_map")
            if feature_artifact is None or feature_artifact["approval_state"] not in {
                "approved",
                "accepted",
            }:
                raise ValueError("Feature map must be approved before continuing")
            return TransitionDecision(
                current_state=state,
                target_state="EID_LAB",
                cause="eid_completed",
                action="run_eid",
            )

        if state == "EID_LAB":
            return TransitionDecision(
                current_state=state,
                target_state="ROADMAP_WAIT",
                cause="roadmap_drafted",
                action="build_roadmap",
            )

        if state == "ROADMAP_WAIT":
            return TransitionDecision(
                current_state=state,
                target_state="DEVELOPMENT",
                cause="roadmap_promoted",
                action="promote_final_roadmap",
            )

        if state == "DEVELOPMENT":
            return TransitionDecision(
                current_state=state,
                target_state="AUDIT",
                cause="development_iteration_completed",
            )

        if state == "AUDIT":
            return TransitionDecision(
                current_state=state,
                target_state="CLOSURE",
                cause="closure_proof_emitted",
                action="build_completion_proof",
            )

        return TransitionDecision(
            current_state=state,
            target_state="DEVELOPMENT",
            cause="operator_continue",
        )
