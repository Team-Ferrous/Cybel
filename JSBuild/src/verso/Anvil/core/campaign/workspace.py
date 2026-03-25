"""Campaign workspace layout and metadata management."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

DEFAULT_CAMPAIGN_LAYOUT = {
    "repos/attached": "repos/attached",
    "repos/analysis_local": "repos/analysis_local",
    "repos/analysis_external": "repos/analysis_external",
    "repos/target": "repos/target",
    "artifacts/intake": "artifacts/intake",
    "artifacts/architecture": "artifacts/architecture",
    "artifacts/feature_map": "artifacts/feature_map",
    "artifacts/research": "artifacts/research",
    "artifacts/roadmap_draft": "artifacts/roadmap_draft",
    "artifacts/roadmap_final": "artifacts/roadmap_final",
    "artifacts/experiments": "artifacts/experiments",
    "artifacts/telemetry": "artifacts/telemetry",
    "artifacts/audits": "artifacts/audits",
    "artifacts/whitepapers": "artifacts/whitepapers",
    "artifacts/closure": "artifacts/closure",
    "logs": "logs",
    "reports": "reports",
    "phases": "phases",
}

PHASE_DIRECTORY_NAMES = {
    "intake": "01_intake",
    "research": "02_research",
    "eid": "03_eid",
    "questionnaire": "04_questionnaire",
    "feature_map": "05_feature_map",
    "roadmap_draft": "06_roadmap_draft",
    "development": "07_development",
    "analysis_upgrade": "08_analysis_upgrade",
    "deep_test_audit": "09_deep_test_audit",
    "convergence": "10_convergence",
}


@dataclass
class CampaignWorkspace:
    """Filesystem-backed campaign workspace with deterministic layout."""

    campaign_id: str
    base_dir: str = ".anvil/campaigns"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root_dir = os.path.abspath(os.path.join(self.base_dir, self.campaign_id))
        self.metadata_path = os.path.join(self.root_dir, "campaign.json")
        self.db_path = os.path.join(self.root_dir, "state.db")

    @classmethod
    def create(
        cls,
        campaign_id: str,
        *,
        base_dir: str = ".anvil/campaigns",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "CampaignWorkspace":
        workspace = cls(
            campaign_id=campaign_id, base_dir=base_dir, metadata=metadata or {}
        )
        workspace.ensure_layout()
        workspace.save_metadata()
        return workspace

    @classmethod
    def load(
        cls,
        campaign_id: str,
        *,
        base_dir: str = ".anvil/campaigns",
    ) -> "CampaignWorkspace":
        workspace = cls(campaign_id=campaign_id, base_dir=base_dir)
        workspace.ensure_layout()
        workspace.metadata = workspace.load_metadata()
        return workspace

    def ensure_layout(self) -> None:
        os.makedirs(self.root_dir, exist_ok=True)
        for relative in DEFAULT_CAMPAIGN_LAYOUT.values():
            os.makedirs(os.path.join(self.root_dir, relative), exist_ok=True)

    def ensure(self) -> None:
        self.ensure_layout()

    def load_metadata(self) -> Dict[str, Any]:
        if not os.path.exists(self.metadata_path):
            return {
                "campaign_id": self.campaign_id,
                "created_at": time.time(),
            }
        with open(self.metadata_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.setdefault("campaign_id", self.campaign_id)
        return payload

    def save_metadata(self, updates: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {
            "campaign_id": self.campaign_id,
            **self.metadata,
            **(updates or {}),
        }
        payload.setdefault("created_at", time.time())
        payload["updated_at"] = time.time()
        self.metadata = payload
        with open(self.metadata_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        return payload

    def phase_dir(self, phase_id: str) -> str:
        path = os.path.join(self.root_dir, "phases", self.phase_slug(phase_id))
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def phase_slug(phase_id: str) -> str:
        normalized = str(phase_id or "").strip().lower().replace("\\", "/").strip("/")
        if not normalized:
            return "00_unknown"
        if normalized in PHASE_DIRECTORY_NAMES:
            return PHASE_DIRECTORY_NAMES[normalized]
        if normalized[:2].isdigit() and "_" in normalized:
            return normalized
        return normalized.replace("/", "_")

    def phase_pack_paths(self, phase_id: str) -> Dict[str, str]:
        phase_dir = self.phase_dir(phase_id)
        fixed = {
            "phase": os.path.join(phase_dir, "phase.json"),
            "markdown": os.path.join(phase_dir, "phase.md"),
            "task_graph": os.path.join(phase_dir, "task_graph.json"),
            "decision_log": os.path.join(phase_dir, "decision_log.json"),
            "evidence_digest": os.path.join(phase_dir, "evidence_digest.json"),
            "exit_gate": os.path.join(phase_dir, "exit_gate.json"),
            "telemetry_summary": os.path.join(phase_dir, "telemetry_summary.json"),
        }
        for folder in ("evidence", "tasks", "decisions", "reports"):
            os.makedirs(os.path.join(phase_dir, folder), exist_ok=True)
        return fixed

    def phase_manifest_path(self, phase_id: str) -> str:
        return os.path.join(self.phase_dir(phase_id), "manifest.json")

    def write_phase_artifact(
        self,
        phase_id: str,
        relative_name: str,
        payload: Dict[str, Any],
    ) -> str:
        target = os.path.join(self.phase_dir(phase_id), relative_name)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        return target

    def artifact_family_dir(self, family: str) -> str:
        normalized = family.strip().replace("\\", "/").strip("/")
        path = os.path.join(self.root_dir, "artifacts", normalized)
        os.makedirs(path, exist_ok=True)
        return path

    def artifact_family_path(self, family: str) -> str:
        return self.artifact_family_dir(family)

    def repo_family_dir(self, family: str) -> str:
        normalized = family.strip().replace("\\", "/").strip("/")
        path = os.path.join(self.root_dir, "repos", normalized)
        os.makedirs(path, exist_ok=True)
        return path

    def repo_record_dir(self, family: str, repo_id: str) -> str:
        path = os.path.join(self.repo_family_dir(family), repo_id)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def campaign_dir(self) -> str:
        return self.root_dir

    @property
    def repo_cache_dir(self) -> str:
        path = os.path.join(self.base_dir, "repo_cache")
        os.makedirs(path, exist_ok=True)
        return path

    def write_json(self, relative_path: str, payload: Dict[str, Any]) -> str:
        target = os.path.join(self.root_dir, relative_path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        return target

    def summary(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "root_dir": self.root_dir,
            "db_path": self.db_path,
            "metadata_path": self.metadata_path,
            "metadata": dict(self.metadata),
        }
