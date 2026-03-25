"""Links evidence to roadmap items and tasks."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


class EvidenceLinker:
    """Attaches evidence identifiers to downstream roadmap items."""

    def link(self, roadmap: Dict[str, Any], claims: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        evidence_ids = [item.get("claim_id") or item.get("finding_id") for item in claims]
        for phase in roadmap.get("phases", []):
            for item in phase.get("items", []):
                item.setdefault("required_evidence", [])
                item["required_evidence"] = list(dict.fromkeys(item["required_evidence"] + list(filter(None, evidence_ids[:3]))))
        return roadmap
