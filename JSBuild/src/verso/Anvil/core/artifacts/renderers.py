"""Markdown renderers for campaign-scoped artifacts."""

from __future__ import annotations

import json

from typing import Any, Dict, Iterable, List


def _lines(items: Iterable[str]) -> str:
    return "\n".join(items).strip() + "\n"


def render_intake_markdown(payload: Dict[str, Any]) -> str:
    directives = payload.get("directives") or []
    constraints = payload.get("constraints") or []
    directive_lines = [f"- {item}" for item in directives] or ["- None"]
    constraint_lines = [f"- {item}" for item in constraints] or ["- None"]
    return _lines(
        [
            f"# Intake Brief: {payload.get('campaign_name', 'Campaign')}",
            "",
            f"Objective: {payload.get('objective', '')}",
            "",
            "## Directives",
            *directive_lines,
            "",
            "## Constraints",
            *constraint_lines,
        ]
    )


def render_questionnaire_markdown(questions: List[Dict[str, Any]]) -> str:
    lines = ["# Architecture Questionnaire", ""]
    for item in questions:
        lines.extend(
            [
                f"## {item['question_id']}: {item['question']}",
                f"- Why it matters: {item.get('why_it_matters', 'TBD')}",
                f"- Decision scope: {item.get('decision_scope', 'architecture')}",
                f"- Blocking level: {item.get('blocking_level', 'unknown')}",
                f"- Answer mode: {item.get('answer_mode', 'manual')}",
                f"- Status: {item.get('current_status', 'open')}",
                "",
            ]
        )
    return _lines(lines)


def render_feature_map_markdown(features: List[Dict[str, Any]]) -> str:
    lines = ["# Feature Inventory", ""]
    for feature in features:
        state = feature.get("selection_state", feature.get("default_state", "defer"))
        marker = "[x]" if state == "selected" else "[ ]" if state == "unselected" else "defer"
        lines.append(
            f"- {marker} {feature['name']} ({feature['category']}): {feature['description']}"
        )
    return _lines(lines)


def render_roadmap_markdown(roadmap: Dict[str, Any]) -> str:
    lines = [f"# {roadmap.get('title', 'Campaign Roadmap')}", ""]
    for phase in roadmap.get("phases", []):
        lines.extend([f"## {phase['phase_id']}: {phase['title']}", ""])
        for item in phase.get("items", []):
            lines.append(f"- {item['item_id']}: {item['title']} [{item['type']}]")
        lines.append("")
    return _lines(lines)


def render_audit_markdown(audit: Dict[str, Any]) -> str:
    lines = ["# Audit Report", ""]
    for finding in audit.get("findings", []):
        lines.append(
            f"- {finding['severity']}: {finding['title']} ({finding.get('category', 'general')})"
        )
    if len(lines) == 2:
        lines.append("- No findings.")
    return _lines(lines)


def render_completion_markdown(payload: Dict[str, Any]) -> str:
    stop_lines = [f"- {item}" for item in payload.get("stop_reasons", [])] or ["- None"]
    return _lines(
        [
            "# Completion Proof",
            "",
            f"Runtime state: {payload.get('runtime_state', '')}",
            f"Closure decision: {payload.get('closure_status', '')}",
            "",
            "## Stop Reasons",
            *stop_lines,
        ]
    )


def render_whitepaper_markdown(payload: Dict[str, Any]) -> str:
    finding_lines = [f"- {item}" for item in payload.get("findings", [])] or ["- None"]
    return _lines(
        [
            f"# {payload.get('title', 'Campaign Whitepaper')}",
            "",
            payload.get("summary", ""),
            "",
            "## Findings",
            *finding_lines,
        ]
    )


def render_artifact_document(
    family: str, canonical_payload: Dict[str, Any] | List[Any]
) -> str:
    if family == "intake" and isinstance(canonical_payload, dict):
        return render_intake_markdown(canonical_payload)
    if family == "architecture":
        questions = canonical_payload.get("questions", canonical_payload)
        if isinstance(questions, list):
            return render_questionnaire_markdown(questions)
    if family == "feature_map":
        features = canonical_payload.get("features", canonical_payload)
        if isinstance(features, list):
            return render_feature_map_markdown(features)
    if family in {"roadmap_draft", "roadmap_final"} and isinstance(canonical_payload, dict):
        return render_roadmap_markdown(canonical_payload)
    if family == "audits" and isinstance(canonical_payload, dict):
        return render_audit_markdown(canonical_payload)
    if family == "closure" and isinstance(canonical_payload, dict):
        return render_completion_markdown(canonical_payload)
    if family == "whitepapers" and isinstance(canonical_payload, dict):
        return render_whitepaper_markdown(canonical_payload)
    return _lines(
        [
            f"# {family.replace('_', ' ').title()}",
            "",
            "```json",
            json.dumps(canonical_payload, indent=2, sort_keys=True, default=str),
            "```",
        ]
    )
