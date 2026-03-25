from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


@dataclass(frozen=True)
class ReviewGateResult:
    passed: bool
    required_reviews: int
    independent_reviews: int
    reasons: List[str]
    human_approval_blocking: bool


class ReviewGate:
    """Enforce reviewer independence and signoff expectations by AAL."""

    def __init__(self, matrix_path: str = "standards/review_matrix.yaml") -> None:
        self.matrix_path = Path(matrix_path)
        self._matrix = self._load_matrix()

    def _load_matrix(self) -> Dict[str, Dict[str, Any]]:
        if not self.matrix_path.exists():
            return {}
        try:
            data = yaml.safe_load(self.matrix_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
        levels = data.get("aal_levels") if isinstance(data, dict) else {}
        return levels if isinstance(levels, dict) else {}

    def evaluate(
        self,
        aal: str,
        reviewers: Iterable[Any] | None,
        author: str | None = None,
        irreversible_action: bool = False,
        signoff_token: str | None = None,
    ) -> ReviewGateResult:
        normalized = str(aal or "AAL-3").upper()
        cfg = dict(self._matrix.get(normalized, {}))

        required_reviews = int(cfg.get("independent_reviews", 0) or 0)
        require_ivv = bool(cfg.get("iv_and_v_required", False))
        require_human_block = bool(cfg.get("human_approval_blocking", False))

        independent_count = 0
        has_ivv = False
        for reviewer in reviewers or []:
            if isinstance(reviewer, dict):
                reviewer_name = str(reviewer.get("reviewer") or reviewer.get("name") or "").strip()
                is_independent = bool(reviewer.get("independent", False))
                role = str(reviewer.get("role") or "").lower()
            else:
                reviewer_name = str(reviewer).strip()
                is_independent = bool(reviewer_name)
                role = ""

            if not reviewer_name:
                continue
            if author and reviewer_name == author:
                continue
            if is_independent:
                independent_count += 1
            if role in {"ivv", "independent_verification", "independent_validation"}:
                has_ivv = True

        reasons: List[str] = []
        if independent_count < required_reviews:
            reasons.append(
                f"requires {required_reviews} independent reviews, found {independent_count}"
            )
        if require_ivv and not has_ivv:
            reasons.append("independent verification/validation reviewer missing")

        if (irreversible_action or require_human_block) and not str(signoff_token or "").strip():
            reasons.append("human signoff token is required")

        return ReviewGateResult(
            passed=not reasons,
            required_reviews=required_reviews,
            independent_reviews=independent_count,
            reasons=reasons,
            human_approval_blocking=require_human_block,
        )

    def evaluate_from_evidence(
        self,
        aal: str,
        evidence: Dict[str, Any] | None,
        author: str | None = None,
        irreversible_action: bool = False,
    ) -> ReviewGateResult:
        payload = dict(evidence or {})
        reviewers = payload.get("review_signoffs") or payload.get("reviewers") or []
        signoff_token = payload.get("review_signoff_token") or payload.get("signoff_token")
        return self.evaluate(
            aal=aal,
            reviewers=reviewers,
            author=author,
            irreversible_action=irreversible_action,
            signoff_token=signoff_token,
        )
