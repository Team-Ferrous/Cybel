"""Identity and repo-presence management for an Anvil instance."""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import socket
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AnvilInstance:
    instance_id: str
    hostname: str
    user: str
    project_root: str
    project_hash: str
    listen_address: str = "127.0.0.1:0"
    capabilities: List[str] = field(
        default_factory=lambda: ["ownership_v1", "collaboration_v1"]
    )
    repo_branch: str = ""
    repo_head: str = ""
    repo_dirty: bool = False
    discovery_method: str = "filesystem"
    transport_provider: str = "in_memory"
    trust_zone: str = "internal"
    current_campaign_id: str = ""
    current_phase_id: str = ""
    lane_id: str = ""
    active_claim_count: int = 0
    verification_state: str = "unknown"
    analysis_capacity: float = 0.0
    verification_capacity: float = 0.0
    runtime_symbol_digest: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def is_same_project(self, other: "AnvilInstance") -> bool:
        return self.project_hash == other.project_hash

    @property
    def promotable(self) -> bool:
        return self.verification_state in {"ready", "verified", "promotable"}

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "AnvilInstance":
        return AnvilInstance(
            instance_id=str(data["instance_id"]),
            hostname=str(data.get("hostname", "unknown")),
            user=str(data.get("user", "unknown")),
            project_root=str(data.get("project_root", ".")),
            project_hash=str(data.get("project_hash", "")),
            repo_branch=str(data.get("repo_branch", "")),
            repo_head=str(data.get("repo_head", "")),
            repo_dirty=bool(data.get("repo_dirty", False)),
            listen_address=str(data.get("listen_address", "127.0.0.1:0")),
            capabilities=list(
                data.get("capabilities", ["ownership_v1", "collaboration_v1"])
            ),
            discovery_method=str(data.get("discovery_method", "filesystem")),
            transport_provider=str(data.get("transport_provider", "in_memory")),
            trust_zone=str(data.get("trust_zone", "internal")),
            current_campaign_id=str(data.get("current_campaign_id", "")),
            current_phase_id=str(data.get("current_phase_id", "")),
            lane_id=str(data.get("lane_id", "")),
            active_claim_count=int(data.get("active_claim_count", 0)),
            verification_state=str(data.get("verification_state", "unknown")),
            analysis_capacity=float(data.get("analysis_capacity", 0.0)),
            verification_capacity=float(data.get("verification_capacity", 0.0)),
            runtime_symbol_digest=str(data.get("runtime_symbol_digest", "")),
            metadata=dict(data.get("metadata", {})),
            started_at=float(data.get("started_at", time.time())),
            last_seen=float(data.get("last_seen", time.time())),
        )

    def presence_summary(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "listen_address": self.listen_address,
            "project_hash": self.project_hash,
            "repo_branch": self.repo_branch,
            "repo_head": self.repo_head,
            "repo_dirty": self.repo_dirty,
            "trust_zone": self.trust_zone,
            "campaign_id": self.current_campaign_id,
            "phase_id": self.current_phase_id,
            "lane_id": self.lane_id,
            "active_claim_count": self.active_claim_count,
            "verification_state": self.verification_state,
            "analysis_capacity": self.analysis_capacity,
            "verification_capacity": self.verification_capacity,
            "transport_provider": self.transport_provider,
            "discovery_method": self.discovery_method,
            "runtime_symbol_digest": self.runtime_symbol_digest,
            "promotable": self.promotable,
        }


class InstanceRegistry:
    """Manages local instance identity persistence in ``.anvil/instance.json``."""

    def __init__(self, anvil_dir: str = ".anvil", project_root: str = "."):
        self.anvil_dir = Path(anvil_dir)
        self.project_root = Path(project_root).resolve()
        self.anvil_dir.mkdir(parents=True, exist_ok=True)
        self.identity_path = self.anvil_dir / "instance.json"
        self.identity = self._load_or_create()

    def _compute_project_hash(self) -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            )
            head = result.stdout.strip()
            if head:
                return head
        except Exception:
            pass

        fingerprint = f"{self.project_root}:{os.path.getmtime(self.project_root)}"
        return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()

    def _git_identity(self) -> tuple[str, str, bool]:
        try:
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:
            branch = ""
        try:
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:
            head = ""
        try:
            dirty = bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
            )
        except Exception:
            dirty = False
        return branch, head, dirty

    @staticmethod
    def _default_listen_address() -> str:
        host = "127.0.0.1"
        port = int(os.getenv("ANVIL_LISTEN_PORT", "0"))
        return f"{host}:{port}"

    def _create_identity(self) -> AnvilInstance:
        now = time.time()
        branch, head, dirty = self._git_identity()
        identity = AnvilInstance(
            instance_id=str(uuid.uuid4()),
            hostname=socket.gethostname(),
            user=getpass.getuser(),
            project_root=str(self.project_root),
            project_hash=self._compute_project_hash(),
            repo_branch=branch,
            repo_head=head,
            repo_dirty=dirty,
            listen_address=self._default_listen_address(),
            capabilities=["ownership_v1", "collaboration_v1"],
            started_at=now,
            last_seen=now,
        )
        self._save(identity)
        return identity

    def _load_or_create(self) -> AnvilInstance:
        if not self.identity_path.exists():
            return self._create_identity()

        try:
            payload = json.loads(self.identity_path.read_text(encoding="utf-8"))
            identity = AnvilInstance.from_dict(payload)
            identity.project_root = str(self.project_root)
            identity.project_hash = self._compute_project_hash()
            (
                identity.repo_branch,
                identity.repo_head,
                identity.repo_dirty,
            ) = self._git_identity()
            self._save(identity)
            return identity
        except Exception:
            return self._create_identity()

    def _save(self, identity: AnvilInstance) -> None:
        self.identity_path.write_text(
            json.dumps(identity.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def update_heartbeat(self) -> None:
        self.identity.last_seen = time.time()
        self._save(self.identity)

    def set_listen_address(self, listen_address: str) -> None:
        self.identity.listen_address = listen_address
        self._save(self.identity)

    def update_presence(self, **fields: Any) -> AnvilInstance:
        for key, value in fields.items():
            if not hasattr(self.identity, key):
                continue
            setattr(self.identity, key, value)
        self.identity.last_seen = time.time()
        self.identity.project_hash = self._compute_project_hash()
        (
            self.identity.repo_branch,
            self.identity.repo_head,
            self.identity.repo_dirty,
        ) = self._git_identity()
        self._save(self.identity)
        return self.identity
