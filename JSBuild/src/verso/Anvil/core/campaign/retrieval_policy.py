"""Saguaro-first retrieval policy and repo dossier briefing."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RetrievalDecision:
    route: str
    route_class: str
    query: str
    reason: str
    evidence_quality: str
    expected_latency_ms: float = 0.0
    token_savings_estimate: float = 0.0
    frontier_priority: float = 0.0
    repo_dossiers_present: int = 0
    artifact_hits: int = 0
    memory_hits: int = 0
    reuse_source: str = ""
    route_explanation: str = ""

    def to_dict(self) -> dict[str, str | float | int]:
        return {
            "route": self.route,
            "route_class": self.route_class,
            "query": self.query,
            "reason": self.reason,
            "evidence_quality": self.evidence_quality,
            "expected_latency_ms": round(self.expected_latency_ms, 3),
            "token_savings_estimate": round(self.token_savings_estimate, 3),
            "frontier_priority": round(self.frontier_priority, 3),
            "repo_dossiers_present": self.repo_dossiers_present,
            "artifact_hits": self.artifact_hits,
            "memory_hits": self.memory_hits,
            "reuse_source": self.reuse_source,
            "route_explanation": self.route_explanation,
        }


class RetrievalPolicyEngine:
    """Enforce Saguaro-first discovery and keep fallback reasons inspectable."""

    def __init__(self, state_store=None, event_store=None) -> None:
        self.state_store = state_store
        self.event_store = event_store

    def decide(
        self,
        *,
        campaign_id: str,
        query: str,
        fallback_reason: str = "",
        evidence_quality: str = "high",
        repo_dossiers_present: int = 0,
        artifact_hits: int = 0,
        memory_hits: int = 0,
        allow_browser: bool = True,
    ) -> RetrievalDecision:
        route = "fallback" if fallback_reason else "saguaro"
        if route == "fallback" and not fallback_reason.strip():
            raise ValueError("Fallback discovery requires an explicit reason.")
        route_class = "saguaro_slice"
        reuse_source = "saguaro"
        expected_latency_ms = 55.0
        token_savings_estimate = 1200.0
        route_explanation = "Default to Saguaro semantic retrieval for authoritative code discovery."
        reason = "saguaro_authoritative"
        quality_score = {"low": 0.35, "medium": 0.6, "high": 0.85, "campaign_brief": 0.8}.get(
            evidence_quality,
            0.6,
        )
        if fallback_reason:
            route_class = "browser_fetch" if allow_browser else "manual_escalation"
            reuse_source = "external"
            expected_latency_ms = 350.0 if allow_browser else 1000.0
            token_savings_estimate = 0.0
            route_explanation = (
                "Fallback retrieval was selected because authoritative internal evidence "
                f"was insufficient: {fallback_reason.strip()}."
            )
            reason = fallback_reason
        elif artifact_hits > 0:
            route_class = "artifact_cache"
            reuse_source = "artifact_cache"
            expected_latency_ms = 6.0
            token_savings_estimate = 2200.0
            route_explanation = "Existing campaign artifacts cover this query, so cached evidence is reused first."
            reason = "artifact_cache_reuse"
        elif memory_hits > 0:
            route_class = "memory_fabric"
            reuse_source = "memory_fabric"
            expected_latency_ms = 12.0
            token_savings_estimate = 1700.0
            route_explanation = "Memory Fabric already has linked evidence for this query, so replayable memory wins."
            reason = "memory_fabric_reuse"
        elif repo_dossiers_present > 0:
            route_class = "repo_dossier"
            reuse_source = "repo_dossier"
            expected_latency_ms = 18.0
            token_savings_estimate = 1400.0
            route_explanation = "Repo dossier summaries are available and are cheaper than a fresh browser fetch."
            reason = "repo_dossier_brief"
        frontier_priority = min(
            1.0,
            quality_score
            + min(0.18, repo_dossiers_present * 0.03)
            + min(0.12, artifact_hits * 0.04)
            + min(0.1, memory_hits * 0.05),
        )
        decision = RetrievalDecision(
            route=route,
            route_class=route_class,
            query=query,
            reason=reason,
            evidence_quality=evidence_quality,
            expected_latency_ms=expected_latency_ms,
            token_savings_estimate=token_savings_estimate,
            frontier_priority=frontier_priority,
            repo_dossiers_present=max(0, int(repo_dossiers_present)),
            artifact_hits=max(0, int(artifact_hits)),
            memory_hits=max(0, int(memory_hits)),
            reuse_source=reuse_source,
            route_explanation=route_explanation,
        )
        if self.state_store is not None:
            self.state_store.record_telemetry(
                campaign_id,
                telemetry_kind="retrieval_policy",
                payload=decision.to_dict(),
            )
        if self.event_store is not None:
            self.event_store.emit(
                event_type="campaign.retrieval_policy",
                payload=decision.to_dict(),
                source="RetrievalPolicyEngine",
                run_id=campaign_id,
                links=[
                    {
                        "link_type": "query",
                        "target_type": "campaign",
                        "target_ref": campaign_id,
                    }
                ],
            )
        return decision

    def render_repo_dossier_brief(
        self,
        *,
        campaign_id: str,
        repos: list[dict[str, Any]],
        repo_dossiers: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        repo_summaries: list[dict[str, Any]] = []
        for repo in repos:
            repo_id = str(repo.get("repo_id") or repo.get("name") or "repo")
            dossier = next(
                (
                    item
                    for item in repo_dossiers
                    if str(item.get("repo_id") or item.get("name") or "") == repo_id
                ),
                {},
            )
            repo_summaries.append(
                {
                    "repo_id": repo_id,
                    "role": str(repo.get("role") or "target"),
                    "path": str(repo.get("local_path") or repo.get("origin") or ""),
                    "entry_points": list(dossier.get("entry_points") or []),
                    "build_files": list(dossier.get("build_files") or []),
                    "test_files": list(dossier.get("test_files") or []),
                    "reuse_candidates": list(dossier.get("reuse_candidates") or []),
                    "risk_signals": list(dossier.get("risk_signals") or []),
                    "tech_stack": list(dossier.get("tech_stack") or []),
                }
            )

        lines = [
            "# Repo Dossier Brief",
            "",
            f"- Campaign: {campaign_id}",
            f"- Repos summarized: {len(repo_summaries)}",
            "",
        ]
        for repo in repo_summaries:
            lines.extend(
                [
                    f"## {repo['repo_id']}",
                    f"- Role: {repo['role']}",
                    f"- Path: {repo['path'] or 'n/a'}",
                    f"- Entry points: {', '.join(repo['entry_points']) or 'n/a'}",
                    f"- Build files: {', '.join(repo['build_files']) or 'n/a'}",
                    f"- Tests: {', '.join(repo['test_files']) or 'n/a'}",
                    f"- Tech stack: {', '.join(repo['tech_stack']) or 'n/a'}",
                    f"- Reuse candidates: {len(repo['reuse_candidates'])}",
                    f"- Risk signals: {len(repo['risk_signals'])}",
                    "",
                ]
            )
        payload = {
            "campaign_id": campaign_id,
            "repo_count": len(repo_summaries),
            "repos": repo_summaries,
            "brief_summary": " | ".join(
                [
                    f"{item['repo_id']}:{item['role']}:{len(item['reuse_candidates'])} reuse"
                    for item in repo_summaries
                ]
            ),
        }
        return "\n".join(lines).rstrip() + "\n", payload

    @staticmethod
    def load_repo_dossiers(workspace_root: str) -> list[dict[str, Any]]:
        dossier_path = os.path.join(workspace_root, "artifacts", "research", "repo_dossiers.json")
        if not os.path.exists(dossier_path):
            return []
        with open(dossier_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        return list(payload.get("repo_dossiers") or [])
