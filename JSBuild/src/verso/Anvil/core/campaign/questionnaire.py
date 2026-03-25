"""Architecture questionnaire generation and persistence."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass
class ArchitectureQuestion:
    question_id: str
    question: str
    why_it_matters: str
    decision_scope: str
    blocking_level: str
    answer_mode: str
    default_policy: str = ""
    linked_roadmap_items: List[str] = field(default_factory=list)
    current_status: str = "open"
    answer: Optional[Dict[str, object]] = None
    metadata: Dict[str, object] = field(default_factory=dict)


class QuestionValueEstimator:
    """Scores architecture questions by branch-pruning leverage."""

    BLOCKING_WEIGHT = {
        "low": 0.25,
        "medium": 0.5,
        "high": 0.8,
        "critical": 1.0,
    }

    def score(
        self,
        *,
        question_text: str,
        blocking_level: str,
        linked_roadmap_items: list[str],
        metadata: Dict[str, object],
        risk_summary: Dict[str, object] | None,
    ) -> Dict[str, object]:
        blocking_weight = self.BLOCKING_WEIGHT.get(blocking_level, 0.5)
        linked_item_count = len(linked_roadmap_items)
        risk_weight = float((risk_summary or {}).get("high_risk_count") or 0) * 0.05
        branch_pruning_estimate = max(
            1,
            linked_item_count + (2 if "which" in question_text.lower() or "confirm" in question_text.lower() else 1),
        )
        question_value_score = round(
            min(
                1.0,
                blocking_weight * 0.55
                + min(0.3, linked_item_count * 0.08)
                + min(0.15, risk_weight)
                + min(0.1, len(question_text.split()) * 0.003),
            ),
            3,
        )
        seeded_score = float(metadata.get("question_value_score") or 0.0)
        seeded_pruning = int(metadata.get("branch_pruning_estimate") or 0)
        return {
            **metadata,
            "question_value_score": max(seeded_score, question_value_score),
            "branch_pruning_estimate": max(seeded_pruning, branch_pruning_estimate),
            "linked_item_count": linked_item_count,
            "risk_summary": dict(risk_summary or {}),
        }


class QuestionnaireBuilder:
    """Compile questionnaire artifacts from directives and unresolved unknowns."""

    def __init__(self, state_store, campaign_id: str):
        self.state_store = state_store
        self.campaign_id = campaign_id
        self.value_estimator = QuestionValueEstimator()

    @staticmethod
    def _question_id(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

    def build(
        self,
        *,
        directives: Iterable[str],
        unresolved_unknowns: Iterable[Dict[str, object]] | None = None,
        risk_summary: Dict[str, object] | None = None,
    ) -> List[ArchitectureQuestion]:
        questions: List[ArchitectureQuestion] = []
        for directive in directives:
            text = str(directive).strip()
            if not text:
                continue
            metadata = self.value_estimator.score(
                question_text=text,
                blocking_level="high",
                linked_roadmap_items=[],
                metadata={},
                risk_summary=risk_summary,
            )
            questions.append(
                ArchitectureQuestion(
                    question_id=self._question_id(text),
                    question=f"Confirm architecture stance: {text}",
                    why_it_matters="Campaign requires explicit architecture resolution before roadmap promotion.",
                    decision_scope="architecture",
                    blocking_level="high",
                    answer_mode="user",
                    default_policy="leave_open",
                    metadata=metadata,
                )
            )

        for item in unresolved_unknowns or []:
            question_text = str(item.get("question") or "").strip()
            if not question_text:
                continue
            question_metadata = self.value_estimator.score(
                question_text=question_text,
                blocking_level=str(item.get("blocking_level") or "medium"),
                linked_roadmap_items=list(item.get("linked_roadmap_items") or []),
                metadata=dict(item.get("metadata") or {}),
                risk_summary=risk_summary,
            )
            questions.append(
                ArchitectureQuestion(
                    question_id=str(item.get("question_id") or self._question_id(question_text)),
                    question=question_text,
                    why_it_matters=str(item.get("why_it_matters") or "Objective unknown requires adjudication."),
                    decision_scope=str(item.get("decision_scope") or "architecture"),
                    blocking_level=str(item.get("blocking_level") or "medium"),
                    answer_mode=str(item.get("answer_mode") or "specialist"),
                    default_policy=str(item.get("default_policy") or "route_to_specialist"),
                    linked_roadmap_items=list(item.get("linked_roadmap_items") or []),
                    metadata=question_metadata,
                )
            )
        questions.sort(
            key=lambda item: (
                -float(item.metadata.get("question_value_score") or 0.0),
                item.question_id,
            )
        )
        return questions

    def persist(self, questions: Iterable[ArchitectureQuestion]) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for question in questions:
            payload = asdict(question)
            payload["campaign_id"] = self.campaign_id
            self.state_store.record_question(payload)
            rows.append(payload)
        return rows

    def pending_blockers(self) -> List[Dict[str, object]]:
        questions = self.state_store.list_questions(self.campaign_id)
        pending = [
            question
            for question in questions
            if question["current_status"] not in {"answered", "waived"}
            and (
                question["blocking_level"] in {"high", "critical"}
                or float((question.get("metadata") or {}).get("question_value_score") or 0.0)
                >= 0.75
            )
        ]
        pending.sort(
            key=lambda item: (
                -float((item.get("metadata") or {}).get("question_value_score") or 0.0),
                str(item.get("question_id") or ""),
            )
        )
        return pending
