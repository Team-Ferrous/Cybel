"""Requirement validation engine over traceability and omnigraph state."""

from __future__ import annotations

import os
from typing import Any

from saguaro.omnigraph.store import OmniGraphStore
from saguaro.requirements.traceability import TraceabilityService
from saguaro.validation.witnesses import WitnessAggregator


class ValidationEngine:
    """Validate document requirements against repository artifacts."""

    def __init__(self, repo_path: str, graph_service: Any | None = None) -> None:
        self.repo_path = os.path.abspath(repo_path)
        self.traceability = TraceabilityService(repo_root=self.repo_path, graph_service=graph_service)
        self.omnigraph = OmniGraphStore(self.repo_path, graph_service=graph_service)
        self.witnesses = WitnessAggregator()

    def validate_docs(self, path: str = ".") -> dict[str, Any]:
        """Validate all extracted requirements."""
        traceability = self.traceability.build(path)
        omni = self.omnigraph.build(traceability_payload=traceability)
        results = []
        grouped_records = self._group_by_requirement(traceability.get("records", []))
        for requirement in traceability.get("requirements", []):
            records = grouped_records.get(requirement["id"], [])
            witnesses = self.witnesses.build(
                requirement,
                records,
                generation_id=str(traceability.get("generation_id") or "trace"),
            )
            state = self.witnesses.classify_state(records, witnesses)
            results.append(
                {
                    "requirement": requirement,
                    "records": records,
                    "witnesses": [item.to_dict() for item in witnesses],
                    "state": state,
                }
            )
        return {
            "status": "ok",
            "generation_id": traceability.get("generation_id"),
            "requirements": results,
            "summary": {
                "count": len(results),
                "implemented_witnessed": sum(1 for item in results if item["state"] == "implemented_witnessed"),
                "implemented_unwitnessed": sum(1 for item in results if item["state"] == "implemented_unwitnessed"),
                "partial": sum(1 for item in results if item["state"] == "partially_implemented"),
                "unimplemented": sum(1 for item in results if item["state"] == "unimplemented"),
                "omnigraph_nodes": int(omni.get("summary", {}).get("node_count", 0)),
                "omnigraph_relations": int(omni.get("summary", {}).get("relation_count", 0)),
            },
        }

    def validate_requirement(self, requirement_id: str) -> dict[str, Any]:
        """Validate a single requirement from cached traceability state."""
        traceability = self.traceability._load_payload()  # noqa: SLF001
        grouped_records = self._group_by_requirement(traceability.get("records", []))
        requirement = next(
            (item for item in traceability.get("requirements", []) if item["id"] == requirement_id),
            None,
        )
        if not requirement:
            return {"status": "missing", "requirement_id": requirement_id}
        records = grouped_records.get(requirement_id, [])
        witnesses = self.witnesses.build(
            requirement,
            records,
            generation_id=str(traceability.get("generation_id") or "trace"),
        )
        return {
            "status": "ok",
            "requirement": requirement,
            "records": records,
            "witnesses": [item.to_dict() for item in witnesses],
            "state": self.witnesses.classify_state(records, witnesses),
        }

    def gaps(self, path: str = ".") -> dict[str, Any]:
        """Return weak or missing coverage gaps."""
        report = self.validate_docs(path)
        gaps = [
            item
            for item in report.get("requirements", [])
            if item["state"] in {"unimplemented", "partially_implemented", "implemented_unwitnessed"}
        ]
        return {"status": "ok", "count": len(gaps), "gaps": gaps}

    @staticmethod
    def _group_by_requirement(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in records:
            grouped.setdefault(str(item.get("requirement_id") or ""), []).append(item)
        return grouped
