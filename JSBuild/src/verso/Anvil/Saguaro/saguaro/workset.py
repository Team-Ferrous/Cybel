"""Utilities for workset."""

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from saguaro.governor import ContextBudgetExceeded, ContextGovernor

logger = logging.getLogger(__name__)


@dataclass
class WorksetConstraint:
    """Provide WorksetConstraint support."""
    type: str  # "read_only", "no_new_deps", "tests_must_pass"
    target: str  # e.g. "saguaro/core/**"


@dataclass
class Workset:
    """Provide Workset support."""
    id: str
    description: str
    files: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    constraints: list[WorksetConstraint] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    status: str = "active"  # active, locked, closed
    budget_usage: int = 0
    budget_limit: int = 0
    owner_agent_id: str | None = None
    owner_instance_id: str | None = None
    ownership_mode: str = "exclusive"  # exclusive, shared_read, collaborative
    phase_id: str | None = None
    parent_task_id: str | None = None
    lease_heartbeat: float = field(default_factory=time.time)
    lease_ttl_seconds: int = 300

    def to_json(self) -> str:
        """Handle to json."""
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(json_str: str) -> "Workset":
        """Handle from json."""
        data = json.loads(json_str)
        constraints = [WorksetConstraint(**c) for c in data.get("constraints", [])]
        data["constraints"] = constraints
        return Workset(**data)


class WorksetManager:
    """Provide WorksetManager support."""
    def __init__(self, saguaro_dir: str, repo_path: str = ".") -> None:
        """Initialize the instance."""
        self.saguaro_dir = saguaro_dir
        self.repo_path = os.path.abspath(repo_path)
        self.worksets_dir = os.path.join(saguaro_dir, "worksets")
        os.makedirs(self.worksets_dir, exist_ok=True)
        self.governor = ContextGovernor()
        self._lock = threading.RLock()

    def _estimate_file_tokens(self, rel_path: str) -> int:
        full_path = os.path.join(self.repo_path, rel_path)
        try:
            if not os.path.exists(full_path):
                return 0
            with open(full_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
                return self.governor.estimate_tokens(content)
        except Exception as e:
            logger.warning(f"Could not read {rel_path} for token estimation: {e}")
            return 0

    def create_workset(
        self,
        description: str,
        files: list[str],
        symbols: list[str] = None,
        constraints: list[dict] = None,
        allow_escalation: bool = False,
        owner_agent_id: str | None = None,
        owner_instance_id: str | None = None,
        ownership_mode: str = "exclusive",
        phase_id: str | None = None,
        parent_task_id: str | None = None,
        lease_ttl_seconds: int = 300,
    ) -> Workset:
        """Creates a new workset with budget enforcement."""
        # 1. Budget Check
        total_tokens = 0
        file_items = []
        for f in files:
            tokens = self._estimate_file_tokens(f)
            total_tokens += tokens
            file_items.append(
                {"name": f, "content": " " * (tokens * 4)}
            )  # Mock content for governor check

        # Check soft limit first
        is_safe, est, msg = self.governor.check_budget(file_items)

        if not is_safe and not allow_escalation:
            # If not safe (hard limit exceeded), we reject
            raise ContextBudgetExceeded(f"Workset creation failed: {msg}")

        # If warning (soft limit) but not hard limit, we allow it but log/warn
        if "WARNING" in msg:
            logger.warning(f"Workset near budget limit: {msg}")

        # Generate ID based on content hash + time for uniqueness
        content = f"{description}{sorted(files)}{time.time()}"
        ws_id = hashlib.sha256(content.encode()).hexdigest()[:12]

        real_constraints = []
        if constraints:
            for c in constraints:
                real_constraints.append(WorksetConstraint(**c))

        ws = Workset(
            id=ws_id,
            description=description,
            files=files,
            symbols=symbols or [],
            constraints=real_constraints,
            status="pending",  # Must acquire lease explicitly
            budget_usage=total_tokens,
            budget_limit=self.governor.hard_limit,
            owner_agent_id=owner_agent_id,
            owner_instance_id=owner_instance_id,
            ownership_mode=ownership_mode,
            phase_id=phase_id,
            parent_task_id=parent_task_id,
            lease_heartbeat=time.time(),
            lease_ttl_seconds=int(lease_ttl_seconds),
        )
        self._save(ws)
        return ws

    def expand_workset(
        self, ws_id: str, new_files: list[str], justification: str
    ) -> Workset:
        """Attempts to add files to an existing workset."""
        ws = self.get_workset(ws_id)
        if not ws:
            raise ValueError("Workset not found")

        # Calculate new usage
        current_files = set(ws.files)
        added_tokens = 0
        for f in new_files:
            if f not in current_files:
                added_tokens += self._estimate_file_tokens(f)

        new_usage = ws.budget_usage + added_tokens

        if new_usage > ws.budget_limit:
            # Check justification? For now, we just enforce the hard limit unless specific key words used?
            # Or maybe we allow escalation if justification is provided (mock logic)
            if (
                "CRITICAL" in justification.upper()
                or "SECURITY" in justification.upper()
            ):
                # Allow bump
                new_limit = self.governor.escalate(ws.budget_limit)
                ws.budget_limit = new_limit
                if new_usage > new_limit:
                    raise ContextBudgetExceeded(
                        f"Even with escalation, limit exceeded ({new_usage} > {new_limit})"
                    )
            else:
                raise ContextBudgetExceeded(
                    f"Expansion rejected. Budget {ws.budget_usage} -> {new_usage} exceeds limit {ws.budget_limit}. Justification insufficient."
                )

        # Apply expansion
        for f in new_files:
            if f not in ws.files:
                ws.files.append(f)

        ws.budget_usage = new_usage
        self._save(ws)
        return ws

    def get_workset(self, ws_id: str) -> Workset | None:
        """Get workset."""
        path = os.path.join(self.worksets_dir, f"{ws_id}.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return Workset.from_json(f.read())

    def list_worksets(self) -> list[Workset]:
        """List worksets."""
        worksets = []
        if not os.path.exists(self.worksets_dir):
            return []

        for f_name in os.listdir(self.worksets_dir):
            if f_name.endswith(".json"):
                with open(os.path.join(self.worksets_dir, f_name)) as f:
                    try:
                        worksets.append(Workset.from_json(f.read()))
                    except Exception:
                        continue  # Skip malformed
        return sorted(worksets, key=lambda w: w.created_at, reverse=True)

    def _save(self, workset: Workset) -> None:
        path = os.path.join(self.worksets_dir, f"{workset.id}.json")
        with open(path, "w") as f:
            f.write(workset.to_json())

    def check_conflicts(
        self,
        proposed_files: list[str],
        exclude_ws_id: str = None,
        ownership_mode: str = "exclusive",
    ) -> list[dict]:
        """Checks if any proposed files are already claimed by other ACTIVE worksets."""
        conflicts = []
        proposed_mode = ownership_mode or "exclusive"
        active = [
            w
            for w in self.list_worksets()
            if w.status == "active" and w.id != exclude_ws_id
        ]

        proposed_set = set(proposed_files)

        for ws in active:
            existing_set = set(ws.files)
            overlap = proposed_set.intersection(existing_set)
            if overlap:
                ws_mode = ws.ownership_mode or "exclusive"
                # shared_read/collaborative can coexist unless exclusive is involved.
                if proposed_mode in {"shared_read", "collaborative"} and ws_mode in {
                    "shared_read",
                    "collaborative",
                }:
                    continue
                conflicts.append(
                    {
                        "workset_id": ws.id,
                        "description": ws.description,
                        "overlapping_files": list(overlap),
                        "owner_agent_id": ws.owner_agent_id,
                        "owner_instance_id": ws.owner_instance_id,
                        "ownership_mode": ws_mode,
                    }
                )
        return conflicts

    def acquire_lease(self, ws_id: str) -> dict[str, Any]:
        """Attempts to lock the workset (set status to active) if no conflicts exist.

        Args:
            ws_id: The unique identifier of the workset.

        Returns:
            Dict containing 'success', 'message', and optionally 'conflicts' or the 'workset' instance.
        """
        ws = self.get_workset(ws_id)
        if not ws:
            return {"success": False, "message": f"Workset '{ws_id}' not found"}

        # Atomic conflict check: ensure no other active workset shares these files
        conflicts = self.check_conflicts(
            ws.files, exclude_ws_id=ws_id, ownership_mode=ws.ownership_mode
        )
        if conflicts:
            logger.warning(
                f"Lease acquisition failed for {ws_id}: conflicts with {conflicts}"
            )
            return {
                "success": False,
                "message": "Conflicts detected with other active worksets.",
                "conflicts": conflicts,
            }

        ws.status = "active"
        ws.lease_heartbeat = time.time()
        self._save(ws)
        logger.info(f"Lease acquired for workset {ws_id}")
        return {
            "success": True,
            "message": "Lease acquired successfully",
            "workset": ws,
        }

    def release_lease(self, ws_id: str) -> bool:
        """Releases the workset lease by setting its status back to closed.

        Args:
            ws_id: The unique identifier of the workset.

        Returns:
            True if released, False if not found.
        """
        ws = self.get_workset(ws_id)
        if ws:
            ws.status = "closed"
            ws.lease_heartbeat = time.time()
            self._save(ws)
            logger.info(f"Lease released for workset {ws_id}")
            return True
        return False

    def touch_lease(self, ws_id: str) -> bool:
        """Handle touch lease."""
        ws = self.get_workset(ws_id)
        if ws is None:
            return False
        ws.lease_heartbeat = time.time()
        self._save(ws)
        return True

    def reap_expired_leases(self) -> list[str]:
        """Handle reap expired leases."""
        expired_ids: list[str] = []
        now = time.time()
        for ws in self.list_worksets():
            if ws.status != "active":
                continue
            ttl = int(ws.lease_ttl_seconds or 300)
            heartbeat = float(ws.lease_heartbeat or ws.created_at)
            if (now - heartbeat) <= ttl:
                continue
            ws.status = "closed"
            ws.lease_heartbeat = now
            self._save(ws)
            expired_ids.append(ws.id)
        return expired_ids
