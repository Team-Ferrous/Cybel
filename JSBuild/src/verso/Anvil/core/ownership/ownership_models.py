"""Data models for file ownership and access decisions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

VALID_OWNERSHIP_MODES = {"exclusive", "shared_read", "collaborative"}
VALID_ACCESS_MODES = {
    "analysis_readonly",
    "analysis_extract_only",
    "target_plan_only",
    "target_write",
    "artifact_write",
    "audit_readonly",
}


@dataclass
class OwnershipRecord:
    file_path: str
    owner_agent_id: str
    owner_instance_id: str
    mode: str
    claimed_at: float
    heartbeat_at: float
    ttl_seconds: int
    campaign_id: Optional[str] = None
    repo_id: Optional[str] = None
    phase_id: Optional[str] = None
    task_id: Optional[str] = None
    agent_role: Optional[str] = None
    access_mode: Optional[str] = None
    workset_id: Optional[str] = None
    expires_at: Optional[float] = None
    lease_epoch: int = 0
    fencing_token: Optional[str] = None
    trust_zone: Optional[str] = None
    verification_state: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        if self.expires_at is not None:
            return time.time() > self.expires_at
        return (time.time() - self.heartbeat_at) > self.ttl_seconds


@dataclass
class DeniedFile:
    file_path: str
    current_owner: str
    ownership_mode: str
    reason: str
    repo_id: Optional[str] = None
    access_mode: Optional[str] = None
    required_trust_zone: Optional[str] = None
    required_lease_epoch: Optional[int] = None


@dataclass
class ClaimResult:
    success: bool
    granted_files: List[str] = field(default_factory=list)
    denied_files: List[DeniedFile] = field(default_factory=list)
    suggested_resolution: Optional[str] = None


@dataclass
class AccessDecision:
    allowed: bool
    reason: str
    owner_info: Optional[OwnershipRecord] = None
    can_negotiate: bool = False


@dataclass
class PhaseStatus:
    phase_id: str
    status: str
    files: List[str] = field(default_factory=list)
    predecessors: List[str] = field(default_factory=list)
    active_tasks: List[str] = field(default_factory=list)
    blocked_reason: Optional[str] = None


@dataclass
class PhaseStartResult:
    ready: bool
    phase_id: str
    blocked_reason: Optional[str] = None
    missing_predecessors: List[str] = field(default_factory=list)


def ownership_record_to_dict(record: OwnershipRecord) -> Dict[str, object]:
    return {
        "file_path": record.file_path,
        "owner_agent_id": record.owner_agent_id,
        "owner_instance_id": record.owner_instance_id,
        "mode": record.mode,
        "claimed_at": record.claimed_at,
        "heartbeat_at": record.heartbeat_at,
        "ttl_seconds": record.ttl_seconds,
        "campaign_id": record.campaign_id,
        "repo_id": record.repo_id,
        "phase_id": record.phase_id,
        "task_id": record.task_id,
        "agent_role": record.agent_role,
        "access_mode": record.access_mode,
        "workset_id": record.workset_id,
        "expires_at": record.expires_at,
        "lease_epoch": record.lease_epoch,
        "fencing_token": record.fencing_token,
        "trust_zone": record.trust_zone,
        "verification_state": record.verification_state,
        "is_expired": record.is_expired,
    }
