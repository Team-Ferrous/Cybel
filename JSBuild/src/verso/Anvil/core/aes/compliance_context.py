from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ComplianceContext:
    run_id: str
    aal: str = "AAL-3"
    domains: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    hot_paths: list[str] = field(default_factory=list)
    public_api_changes: list[str] = field(default_factory=list)
    dependency_changes: list[str] = field(default_factory=list)
    required_rule_ids: list[str] = field(default_factory=list)
    required_runtime_gates: list[str] = field(default_factory=list)
    trace_id: str | None = None
    evidence_bundle_id: str | None = None
    waiver_ids: list[str] = field(default_factory=list)
    red_team_required: bool = False

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "ComplianceContext":
        data = dict(payload or {})
        return cls(
            run_id=str(data.get("run_id") or data.get("trace_id") or "run"),
            aal=str(data.get("aal") or "AAL-3").upper(),
            domains=[str(item) for item in data.get("domains", []) or []],
            changed_files=[str(item) for item in data.get("changed_files", []) or []],
            hot_paths=[str(item) for item in data.get("hot_paths", []) or []],
            public_api_changes=[
                str(item) for item in data.get("public_api_changes", []) or []
            ],
            dependency_changes=[
                str(item) for item in data.get("dependency_changes", []) or []
            ],
            required_rule_ids=[
                str(item) for item in data.get("required_rule_ids", []) or []
            ],
            required_runtime_gates=[
                str(item) for item in data.get("required_runtime_gates", []) or []
            ],
            trace_id=str(data.get("trace_id") or data.get("run_id") or "run"),
            evidence_bundle_id=data.get("evidence_bundle_id"),
            waiver_ids=[str(item) for item in data.get("waiver_ids", []) or []],
            red_team_required=bool(data.get("red_team_required", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["domains"] = list(dict.fromkeys(payload["domains"]))
        payload["changed_files"] = list(dict.fromkeys(payload["changed_files"]))
        payload["hot_paths"] = list(dict.fromkeys(payload["hot_paths"]))
        payload["public_api_changes"] = list(
            dict.fromkeys(payload["public_api_changes"])
        )
        payload["dependency_changes"] = list(dict.fromkeys(payload["dependency_changes"]))
        payload["required_rule_ids"] = list(dict.fromkeys(payload["required_rule_ids"]))
        payload["required_runtime_gates"] = list(
            dict.fromkeys(payload["required_runtime_gates"])
        )
        payload["waiver_ids"] = list(dict.fromkeys(payload["waiver_ids"]))
        return payload

