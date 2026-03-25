"""Repo registration and write-policy enforcement for autonomy campaigns."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from core.campaign.workspace import CampaignWorkspace


TARGET_WRITE_STATES = {"DEVELOPMENT", "REMEDIATION"}


@dataclass
class CampaignRepoRecord:
    repo_id: str
    campaign_id: str
    name: str
    origin: str
    revision: str
    local_path: str
    role: str
    write_policy: str
    topic_tags: List[str] = field(default_factory=list)
    trust_level: str = "medium"
    ingestion_status: str = "registered"
    metadata: Dict[str, Any] = field(default_factory=dict)


class CampaignRepoRegistry:
    """Campaign-aware repo registry with deterministic write rules."""

    def __init__(self, workspace: CampaignWorkspace, state_store) -> None:
        self.workspace = workspace
        self.state_store = state_store

    @staticmethod
    def default_write_policy(role: str) -> str:
        role = str(role).strip().lower()
        defaults = {
            "target": "phase_gated_write",
            "analysis_local": "immutable",
            "analysis_external": "immutable",
            "artifact_store": "artifact_only",
            "benchmark_fixture": "sandboxed_write",
        }
        return defaults.get(role, "immutable")

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", value or "").strip("_").lower()
        return slug or "repo"

    def register_repo(
        self,
        *,
        name: str,
        local_path: str,
        role: str,
        origin: Optional[str] = None,
        revision: str = "",
        write_policy: Optional[str] = None,
        topic_tags: Optional[List[str]] = None,
        trust_level: str = "medium",
        ingestion_status: str = "registered",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CampaignRepoRecord:
        repo_id = self._slug(f"{role}_{name}")
        record = CampaignRepoRecord(
            repo_id=repo_id,
            campaign_id=self.workspace.campaign_id,
            name=name,
            origin=origin or local_path,
            revision=revision,
            local_path=os.path.abspath(local_path),
            role=role,
            write_policy=write_policy or self.default_write_policy(role),
            topic_tags=list(topic_tags or []),
            trust_level=trust_level,
            ingestion_status=ingestion_status,
            metadata=dict(metadata or {}),
        )
        self.state_store.register_repo(asdict(record))
        record_dir = self.workspace.repo_record_dir(self._repo_family(role), repo_id)
        with open(os.path.join(record_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(asdict(record), handle, indent=2, default=str)
        return record

    def list_repos(self) -> List[CampaignRepoRecord]:
        rows = self.state_store.list_repos(self.workspace.campaign_id)
        records: List[CampaignRepoRecord] = []
        for row in rows:
            topic_tags = json.loads(row.get("topic_tags_json") or "[]")
            metadata = json.loads(row.get("metadata_json") or "{}")
            records.append(
                CampaignRepoRecord(
                    repo_id=row["repo_id"],
                    campaign_id=row["campaign_id"],
                    name=row["name"],
                    origin=row.get("origin") or "",
                    revision=row.get("revision") or "",
                    local_path=row["local_path"],
                    role=row["role"],
                    write_policy=row["write_policy"],
                    topic_tags=topic_tags,
                    trust_level=row.get("trust_level", "medium"),
                    ingestion_status=row.get("ingestion_status", "registered"),
                    metadata=metadata,
                )
            )
        return records

    def resolve_repo_for_path(self, file_path: str) -> Optional[CampaignRepoRecord]:
        candidate = os.path.abspath(file_path)
        matches = [
            record
            for record in self.list_repos()
            if candidate == record.local_path or candidate.startswith(record.local_path + os.sep)
        ]
        if not matches:
            return None
        matches.sort(key=lambda record: len(record.local_path), reverse=True)
        return matches[0]

    def can_write(
        self,
        file_path: str,
        *,
        campaign_state: str,
    ) -> bool:
        record = self.resolve_repo_for_path(file_path)
        if record is None:
            return True

        policy = record.write_policy
        if policy == "immutable":
            return False
        if policy == "artifact_only":
            return "/artifacts/" in file_path.replace("\\", "/")
        if policy == "phase_gated_write":
            return campaign_state in TARGET_WRITE_STATES
        if policy == "sandboxed_write":
            return campaign_state in TARGET_WRITE_STATES | {"SOAK_TEST", "AUDIT"}
        return False

    def policy_snapshot(self) -> Dict[str, Any]:
        repos = self.list_repos()
        return {
            "campaign_id": self.workspace.campaign_id,
            "repo_roles": [repo.role for repo in repos],
            "repos": [asdict(repo) for repo in repos],
        }

    def record_ingestion_report(self, repo_id: str, report: Dict[str, Any]) -> None:
        self.state_store.execute(
            """
            INSERT INTO repo_ingestion_reports (
                campaign_id,
                repo_id,
                report_json,
                created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                self.workspace.campaign_id,
                repo_id,
                json.dumps(report, default=str),
                time.time(),
            ),
        )

    @staticmethod
    def _repo_family(role: str) -> str:
        normalized = role.strip().lower()
        if normalized == "target":
            return "target"
        if normalized == "analysis_local":
            return "analysis_local"
        if normalized == "analysis_external":
            return "analysis_external"
        return "attached"


RepoRegistry = CampaignRepoRegistry
RepoRecord = CampaignRepoRecord
