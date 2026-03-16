from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from core.aes import RedTeamProtocol


class SubagentQualityGate:
    """Validate subagent output before master synthesis consumes it."""

    _CITATION_RE = re.compile(
        r"([A-Za-z0-9_./-]+\.(?:py|cc|cpp|h|hpp|js|ts|md|json))(?::L(\d+))?"
    )
    _CODE_FENCE_RE = re.compile(r"```(?:[^\n]*)\n(.*?)```", re.DOTALL)

    def __init__(
        self,
        repo_root: str = ".",
        brain: Optional[Any] = None,
        thinking_system: Optional[Any] = None,
    ):
        self.repo_root = os.path.abspath(repo_root)
        self.brain = brain
        self.thinking_system = thinking_system
        self.red_team_protocol = RedTeamProtocol()

    def evaluate(
        self,
        subagent_payload: Dict[str, Any],
        original_query: str,
        complexity_score: int = 1,
    ) -> Dict[str, Any]:
        analysis_text = str(
            subagent_payload.get("subagent_analysis")
            or subagent_payload.get("summary")
            or subagent_payload.get("full_response")
            or ""
        )
        declared_files = subagent_payload.get("codebase_files") or subagent_payload.get(
            "files_read"
        )
        file_candidates = self._collect_candidate_files(declared_files, analysis_text)

        hallucination = self._hallucination_check(analysis_text, file_candidates)
        alignment = self._coconut_alignment_score(original_query, analysis_text)
        density = self._evidence_density_check(
            analysis_text, complexity_score, hallucination.get("citations", 0)
        )
        citation_existence = self._citation_existence_check(hallucination)
        compliance = self._compliance_check(subagent_payload)
        contradiction_density = self._contradiction_density_check(analysis_text)
        artifact_completeness = self._artifact_completeness_check(subagent_payload)
        red_team = self._red_team_checklist_check(subagent_payload)

        should_retry = (
            not hallucination["passed"]
            or alignment["score"] < 0.3
            or not density["passed"]
            or not citation_existence["passed"]
            or not compliance["passed"]
            or not contradiction_density["passed"]
            or not artifact_completeness["passed"]
            or not red_team["passed"]
        )
        accepted = not should_retry
        if alignment["score"] >= 0.7 and hallucination["passed"] and density["passed"]:
            confidence = "high"
        elif alignment["score"] >= 0.3 and hallucination["passed"]:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "accepted": accepted,
            "should_retry": should_retry,
            "confidence": confidence,
            "hallucination": hallucination,
            "alignment": alignment,
            "evidence_density": density,
            "citation_file_existence": citation_existence,
            "compliance": compliance,
            "contradiction_density": contradiction_density,
            "artifact_completeness": artifact_completeness,
            "red_team_checklist": red_team,
        }

    def _collect_candidate_files(
        self, declared_files: Optional[Iterable[str]], analysis_text: str
    ) -> List[str]:
        candidates: List[str] = []
        for path in declared_files or []:
            if isinstance(path, str) and path:
                candidates.append(path)
        for match in self._CITATION_RE.findall(analysis_text):
            candidates.append(match[0])
        # preserve order
        unique: List[str] = []
        seen = set()
        for item in candidates:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        return unique

    def _hallucination_check(
        self, text: str, file_candidates: List[str]
    ) -> Dict[str, Any]:
        citations = self._CITATION_RE.findall(text)
        missing_paths: List[str] = []
        invalid_lines: List[str] = []

        existing_paths: List[str] = []
        for file_path, line_str in citations:
            abs_path = self._resolve_path(file_path)
            if abs_path is None or not os.path.exists(abs_path):
                missing_paths.append(file_path)
                continue
            existing_paths.append(abs_path)
            if line_str:
                line_no = int(line_str)
                max_line = self._line_count(abs_path)
                if line_no < 1 or line_no > max_line:
                    invalid_lines.append(f"{file_path}:L{line_no} (max {max_line})")

        snippet_misses: List[str] = []
        snippets = [
            s.strip()
            for s in self._CODE_FENCE_RE.findall(text)
            if len(s.strip()) >= 20
        ][:5]
        searchable_files = existing_paths or [
            p for p in (self._resolve_path(f) for f in file_candidates) if p and os.path.exists(p)
        ]
        for snippet in snippets:
            if not self._snippet_exists(snippet, searchable_files[:8]):
                snippet_misses.append(snippet[:80].replace("\n", " "))

        return {
            "passed": not missing_paths and not invalid_lines and not snippet_misses,
            "citations": len(citations),
            "missing_paths": missing_paths,
            "invalid_lines": invalid_lines,
            "snippet_misses": snippet_misses,
        }

    def _coconut_alignment_score(self, query: str, response: str) -> Dict[str, Any]:
        query_emb = self._embed(query)
        response_emb = self._embed(response)

        if query_emb is None or response_emb is None:
            return {"score": 0.5, "band": "medium", "note": "embedding_unavailable"}

        if (
            self.thinking_system is not None
            and getattr(self.thinking_system, "coconut_enabled", False)
            and getattr(self.thinking_system, "coconut", None) is not None
        ):
            try:
                refined = self.thinking_system.deep_think(query_emb.reshape(1, -1))
                if refined is not None:
                    query_emb = np.asarray(refined, dtype=np.float32).reshape(-1)
            except Exception:
                pass

        score = self._cosine_similarity(query_emb, response_emb)
        if score < 0.3:
            band = "low"
        elif score < 0.7:
            band = "medium"
        else:
            band = "high"
        return {"score": float(score), "band": band}

    def _evidence_density_check(
        self, text: str, complexity_score: int, citations: int
    ) -> Dict[str, Any]:
        if complexity_score <= 3:
            minimum = 1
        elif complexity_score <= 5:
            minimum = 2
        elif complexity_score <= 7:
            minimum = 3
        else:
            minimum = 4
        passed = citations >= minimum and len(text.strip()) >= 20
        return {"passed": passed, "citations": citations, "minimum_required": minimum}

    def _compliance_check(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        compliance = payload.get("compliance") or {}
        requires_compliance = bool(
            payload.get("subagent_analysis")
            or payload.get("subagent_type")
            or payload.get("summary")
            or payload.get("full_response")
        )
        if not requires_compliance:
            return {"passed": True, "missing": [], "required": False}

        missing = [
            key for key in ("trace_id", "evidence_bundle_id") if not compliance.get(key)
        ]
        if "required_runtime_gates" in compliance and not isinstance(
            compliance.get("required_runtime_gates"), list
        ):
            missing.append("required_runtime_gates")
        return {"passed": not missing, "missing": missing, "required": True}

    def _citation_existence_check(self, hallucination: Dict[str, Any]) -> Dict[str, Any]:
        total = int(hallucination.get("citations", 0))
        missing = len(hallucination.get("missing_paths", []))
        valid = max(0, total - missing)
        ratio = float(valid / total) if total else 0.0
        # If no citations exist, defer to evidence density check.
        passed = total == 0 or ratio >= 0.8
        return {
            "passed": passed,
            "total_citations": total,
            "existing_citations": valid,
            "existence_ratio": ratio,
        }

    def _contradiction_density_check(self, text: str) -> Dict[str, Any]:
        normalized = str(text or "").strip()
        if not normalized:
            return {"passed": False, "density": 1.0, "markers": []}
        sentence_count = max(1, len(re.findall(r"[.!?]\s+", normalized)))
        markers: List[str] = []
        for marker in (
            "however",
            "but",
            "on the other hand",
            "contradiction",
            "in contrast",
        ):
            if marker in normalized.lower():
                markers.append(marker)
        density = len(markers) / sentence_count
        return {
            "passed": density <= 0.6,
            "density": density,
            "markers": markers,
            "sentences": sentence_count,
        }

    def _artifact_completeness_check(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        compliance = payload.get("compliance") or {}
        aal = str(payload.get("aal") or compliance.get("aal") or "AAL-3").upper()
        red_team_required = bool(compliance.get("red_team_required", False))
        artifacts = payload.get("artifacts") or {}
        explicit_required = payload.get("required_artifacts") or []
        required = [str(item).strip() for item in explicit_required if str(item).strip()]
        enforce = bool(required) or bool(artifacts) or red_team_required or aal in {"AAL-0", "AAL-1"}
        if not enforce:
            return {"passed": True, "required": [], "missing": [], "enforced": False}
        if aal in {"AAL-0", "AAL-1"}:
            required.extend(
                [
                    "traceability_record",
                    "verification_summary",
                    "review_signoff",
                    "rollback_plan",
                    "change_manifest",
                    "runtime_gates",
                ]
            )
        else:
            required.extend(["verification_summary"])
        if red_team_required or aal in {"AAL-0", "AAL-1"}:
            required.extend(self.red_team_protocol.required_artifacts(aal, True))
        required = list(dict.fromkeys(required))
        missing = [item for item in required if not artifacts.get(item)]
        return {
            "passed": not missing,
            "required": required,
            "missing": missing,
            "enforced": True,
        }

    def _red_team_checklist_check(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        compliance = payload.get("compliance") or {}
        aal = str(payload.get("aal") or compliance.get("aal") or "AAL-3").upper()
        red_team_required = bool(compliance.get("red_team_required", False))
        artifacts = payload.get("artifacts") or {}
        if not red_team_required and aal not in {"AAL-0", "AAL-1"}:
            return {"passed": True, "required": False, "missing_artifacts": []}
        validation = self.red_team_protocol.validate(artifacts, aal, red_team_required)
        return {
            "passed": validation.passed,
            "required": validation.required,
            "missing_artifacts": validation.missing_artifacts,
            "unresolved_critical_findings": validation.unresolved_critical_findings,
        }

    def _embed(self, text: str) -> Optional[np.ndarray]:
        if not self.brain or not text:
            return None
        try:
            if hasattr(self.brain, "get_embeddings"):
                emb = self.brain.get_embeddings(text)
            else:
                emb = self.brain.embeddings(text)
            arr = np.asarray(emb, dtype=np.float32).reshape(-1)
            if arr.size == 0 or not np.isfinite(arr).all():
                return None
            return arr
        except Exception:
            return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        dim = min(a.shape[0], b.shape[0])
        if dim == 0:
            return 0.0
        a = a[:dim]
        b = b[:dim]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-8:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _resolve_path(self, path: str) -> Optional[str]:
        if not path:
            return None
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(self.repo_root, path))

    def _line_count(self, abs_path: str) -> int:
        try:
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as handle:
                return sum(1 for _ in handle)
        except Exception:
            return 0

    def _snippet_exists(self, snippet: str, files: List[str]) -> bool:
        normalized = " ".join(snippet.split())
        for abs_path in files:
            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as handle:
                    content = handle.read()
                if normalized in " ".join(content.split()):
                    return True
            except Exception:
                continue
        return False
