"""Authoritative file ownership registry for local and cross-instance coordination."""

from __future__ import annotations

import asyncio
import inspect
import os
import threading
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

from core.ownership.ownership_models import (
    AccessDecision,
    ClaimResult,
    DeniedFile,
    OwnershipRecord,
    VALID_ACCESS_MODES,
    VALID_OWNERSHIP_MODES,
    ownership_record_to_dict,
)


class FileOwnershipRegistry:
    """Central registry for file ownership with Workset persistence backing."""

    def __init__(
        self,
        workset_manager,
        message_bus=None,
        event_store=None,
        ownership_crdt=None,
        sync_protocol=None,
        instance_id: Optional[str] = None,
        default_ttl_seconds: int = 300,
        repo_policy_resolver=None,
        current_state_getter=None,
    ):
        self.workset_manager = workset_manager
        self.message_bus = message_bus
        self.event_store = event_store
        self.crdt = ownership_crdt
        self.sync = sync_protocol
        self.instance_id = instance_id or "local"
        self.default_ttl_seconds = int(default_ttl_seconds)
        self.repo_policy_resolver = repo_policy_resolver
        self.current_state_getter = current_state_getter

        # path -> [records]
        self._file_index: Dict[str, List[OwnershipRecord]] = {}
        self._agent_index: Dict[str, Set[str]] = defaultdict(set)
        self._phase_index: Dict[str, Set[str]] = defaultdict(set)
        self._workset_index: Dict[Tuple[str, str], str] = {}
        self._lease_epochs: Dict[str, int] = defaultdict(int)
        self._symbol_index: Dict[Tuple[str, str], OwnershipRecord] = {}

        self._lock = threading.RLock()
        self._rebuild_indexes()

    def _normalize_path(self, file_path: str) -> str:
        candidate = file_path or ""
        if not candidate:
            return candidate
        if os.path.isabs(candidate):
            try:
                candidate = os.path.relpath(candidate, self.workset_manager.repo_path)
            except Exception:
                pass
        return candidate.replace("\\", "/")

    def _emit_event(self, event_type: str, payload: Dict[str, object], source: str) -> None:
        if self.event_store is not None:
            try:
                self.event_store.emit(event_type=event_type, payload=payload, source=source)
            except Exception:
                pass

    def _publish(self, topic: str, sender: str, payload: Dict[str, object]) -> None:
        if self.message_bus is not None:
            try:
                self.message_bus.publish(topic=topic, sender=sender, payload=payload)
            except Exception:
                pass

    @staticmethod
    def _run_async(coro) -> None:
        try:
            running = asyncio.get_running_loop()
            running.create_task(coro)
        except RuntimeError:
            asyncio.run(coro)

    def _sync_entry(self, entry) -> None:
        if self.sync is None:
            return
        try:
            result = self.sync.on_ownership_change(entry)
            if inspect.isawaitable(result):
                self._run_async(result)
        except Exception:
            pass

    def _active_records(self, file_path: str) -> List[OwnershipRecord]:
        records = self._file_index.get(file_path, [])
        active = [rec for rec in records if not rec.is_expired]
        if len(active) != len(records):
            self._file_index[file_path] = active
        return active

    def _rebuild_indexes(self) -> None:
        with self._lock:
            self._file_index.clear()
            self._agent_index.clear()
            self._phase_index.clear()
            self._workset_index.clear()

            for ws in self.workset_manager.list_worksets():
                if ws.status != "active":
                    continue

                owner_agent_id = ws.owner_agent_id or "unknown"
                owner_instance_id = ws.owner_instance_id or self.instance_id
                mode = ws.ownership_mode if ws.ownership_mode in VALID_OWNERSHIP_MODES else "exclusive"
                ttl = int(ws.lease_ttl_seconds or self.default_ttl_seconds)
                heartbeat = float(ws.lease_heartbeat or ws.created_at)

                for file_path in ws.files:
                    normalized = self._normalize_path(file_path)
                    record = OwnershipRecord(
                        file_path=normalized,
                        owner_agent_id=owner_agent_id,
                        owner_instance_id=owner_instance_id,
                        mode=mode,
                        claimed_at=float(ws.created_at),
                        heartbeat_at=heartbeat,
                        ttl_seconds=ttl,
                        expires_at=heartbeat + ttl,
                        phase_id=ws.phase_id,
                        task_id=ws.parent_task_id,
                        workset_id=ws.id,
                        lease_epoch=int(self._lease_epochs.get(normalized, 0) or 0),
                    )
                    self._file_index.setdefault(normalized, []).append(record)
                    self._agent_index[owner_agent_id].add(normalized)
                    if ws.phase_id:
                        self._phase_index[ws.phase_id].add(normalized)
                    self._workset_index[(owner_agent_id, normalized)] = ws.id

    def _allow_claim(self, agent_id: str, mode: str, records: List[OwnershipRecord]) -> Optional[DeniedFile]:
        if not records:
            return None

        for record in records:
            if record.owner_agent_id == agent_id:
                continue

            if mode == "exclusive":
                return DeniedFile(
                    file_path=record.file_path,
                    current_owner=record.owner_agent_id,
                    ownership_mode=record.mode,
                    reason="exclusive_lock",
                    required_lease_epoch=record.lease_epoch,
                )

            if mode == "shared_read":
                if record.mode == "exclusive":
                    return DeniedFile(
                        file_path=record.file_path,
                        current_owner=record.owner_agent_id,
                        ownership_mode=record.mode,
                        reason="exclusive_lock",
                        required_lease_epoch=record.lease_epoch,
                    )

            if mode == "collaborative":
                if record.mode == "exclusive":
                    return DeniedFile(
                        file_path=record.file_path,
                        current_owner=record.owner_agent_id,
                        ownership_mode=record.mode,
                        reason="exclusive_lock",
                        required_lease_epoch=record.lease_epoch,
                    )

        return None

    def _claim_locally(
        self,
        agent_id: str,
        file_path: str,
        mode: str,
        phase_id: Optional[str],
        task_id: Optional[str],
        campaign_id: Optional[str],
        repo_id: Optional[str],
        agent_role: Optional[str],
        access_mode: Optional[str],
        trust_zone: Optional[str],
        verification_state: Optional[str],
    ) -> Optional[OwnershipRecord]:
        now = time.time()
        ws = self.workset_manager.create_workset(
            description=f"Ownership lease for {file_path}",
            files=[file_path],
            symbols=[],
            constraints=[],
            allow_escalation=True,
            owner_agent_id=agent_id,
            owner_instance_id=self.instance_id,
            ownership_mode=mode,
            phase_id=phase_id,
            parent_task_id=task_id,
            lease_ttl_seconds=self.default_ttl_seconds,
        )

        lease = self.workset_manager.acquire_lease(ws.id)
        if not lease.get("success"):
            return None

        self.workset_manager.touch_lease(ws.id)
        ws_active = self.workset_manager.get_workset(ws.id)
        heartbeat = float(ws_active.lease_heartbeat if ws_active else now)
        ttl = int(ws_active.lease_ttl_seconds if ws_active else self.default_ttl_seconds)
        lease_epoch = self._lease_epochs[file_path] + 1
        self._lease_epochs[file_path] = lease_epoch

        record = OwnershipRecord(
            file_path=file_path,
            owner_agent_id=agent_id,
            owner_instance_id=self.instance_id,
            mode=mode,
            claimed_at=now,
            heartbeat_at=heartbeat,
            ttl_seconds=ttl,
            campaign_id=campaign_id,
            repo_id=repo_id,
            phase_id=phase_id,
            task_id=task_id,
            agent_role=agent_role,
            access_mode=access_mode,
            workset_id=ws.id,
            expires_at=heartbeat + ttl,
            lease_epoch=lease_epoch,
            fencing_token=f"{self.instance_id}:{agent_id}:{file_path}:{lease_epoch}",
            trust_zone=trust_zone,
            verification_state=verification_state,
        )

        self._file_index.setdefault(file_path, []).append(record)
        self._agent_index[agent_id].add(file_path)
        if phase_id:
            self._phase_index[phase_id].add(file_path)
        self._workset_index[(agent_id, file_path)] = ws.id
        return record

    def claim_files(
        self,
        agent_id: str,
        files: List[str],
        mode: str = "exclusive",
        phase_id: Optional[str] = None,
        task_id: Optional[str] = None,
        campaign_id: Optional[str] = None,
        repo_id: Optional[str] = None,
        agent_role: Optional[str] = None,
        access_mode: Optional[str] = None,
        trust_zone: Optional[str] = None,
        verification_state: Optional[str] = None,
    ) -> ClaimResult:
        if mode not in VALID_OWNERSHIP_MODES:
            raise ValueError(f"Unsupported ownership mode: {mode}")
        if access_mode is not None and access_mode not in VALID_ACCESS_MODES:
            raise ValueError(f"Unsupported access mode: {access_mode}")

        normalized_files = [self._normalize_path(path) for path in files if path]
        granted: List[str] = []
        denied: List[DeniedFile] = []

        with self._lock:
            self.reap_expired_leases()

            for file_path in normalized_files:
                if callable(self.repo_policy_resolver):
                    allowed, reason = self.repo_policy_resolver(
                        file_path=file_path,
                        phase_id=phase_id,
                        task_id=task_id,
                        campaign_id=campaign_id,
                        repo_id=repo_id,
                        access_mode=access_mode,
                        campaign_state=self.current_state_getter() if callable(self.current_state_getter) else None,
                    )
                    if not allowed:
                        denied.append(
                            DeniedFile(
                                file_path=file_path,
                                current_owner="policy",
                                ownership_mode=mode,
                                reason=str(reason),
                                repo_id=repo_id,
                                access_mode=access_mode,
                                required_trust_zone=trust_zone,
                            )
                        )
                        continue
                active_records = self._active_records(file_path)
                denied_file = self._allow_claim(agent_id, mode, active_records)
                if denied_file is not None:
                    denied.append(denied_file)
                    continue

                if self.crdt is not None:
                    current = self.crdt.state.get(file_path)
                    if current and not current.is_tombstone:
                        if (
                            current.owner_agent_id != agent_id
                            or current.owner_instance_id != self.instance_id
                        ):
                            denied.append(
                                DeniedFile(
                                    file_path=file_path,
                                    current_owner=current.owner_agent_id,
                                    ownership_mode=current.mode,
                                    reason="instance_claim",
                                    required_lease_epoch=int(
                                        getattr(current, "logical_ts", 0) or 0
                                    ),
                                )
                            )
                            continue

                record = self._claim_locally(
                    agent_id=agent_id,
                    file_path=file_path,
                    mode=mode,
                    phase_id=phase_id,
                    task_id=task_id,
                    campaign_id=campaign_id,
                    repo_id=repo_id,
                    agent_role=agent_role,
                    access_mode=access_mode,
                    trust_zone=trust_zone,
                    verification_state=verification_state,
                )
                if record is None:
                    denied.append(
                        DeniedFile(
                            file_path=file_path,
                            current_owner="unknown",
                            ownership_mode=mode,
                            reason="lease_conflict",
                            repo_id=repo_id,
                            access_mode=access_mode,
                            required_trust_zone=trust_zone,
                        )
                    )
                    continue

                granted.append(file_path)

                self._emit_event(
                    event_type="ownership.claimed",
                    payload={
                        "agent_id": agent_id,
                        "file_path": file_path,
                        "mode": mode,
                        "phase_id": phase_id,
                        "task_id": task_id,
                        "campaign_id": campaign_id,
                        "repo_id": repo_id,
                        "access_mode": access_mode,
                    },
                    source=agent_id,
                )
                self._publish(
                    topic="ownership.claims",
                    sender=agent_id,
                    payload={
                        "action": "claimed",
                        "record": ownership_record_to_dict(record),
                    },
                )

                if self.crdt is not None:
                    try:
                        entry = self.crdt.claim(
                            file_path=file_path, agent_id=agent_id, mode=mode
                        )
                        self._sync_entry(entry)
                    except Exception:
                        self.release_files(agent_id=agent_id, files=[file_path])
                        denied.append(
                            DeniedFile(
                                file_path=file_path,
                                current_owner="remote",
                                ownership_mode=mode,
                                reason="crdt_tombstone",
                            )
                        )
                        if file_path in granted:
                            granted.remove(file_path)

        success = len(denied) == 0
        resolution = None
        if denied:
            resolution = "Retry with a compatible access mode or wait for lease expiry."
            self._publish(
                topic="ownership.conflicts",
                sender=agent_id,
                payload={
                    "action": "denied",
                    "agent_id": agent_id,
                    "denied_files": [denied_file.__dict__ for denied_file in denied],
                },
            )

        return ClaimResult(
            success=success,
            granted_files=granted,
            denied_files=denied,
            suggested_resolution=resolution,
        )

    def release_files(self, agent_id: str, files: Optional[List[str]] = None) -> None:
        with self._lock:
            if agent_id == "*":
                target_files = list(self._file_index.keys()) if files is None else [
                    self._normalize_path(path) for path in files
                ]
            elif files is None:
                target_files = list(self._agent_index.get(agent_id, set()))
            else:
                target_files = [self._normalize_path(path) for path in files]

            for file_path in target_files:
                existing = self._file_index.get(file_path, [])
                if not existing:
                    continue

                remaining: List[OwnershipRecord] = []
                for record in existing:
                    should_release = agent_id == "*" or record.owner_agent_id == agent_id
                    if not should_release:
                        remaining.append(record)
                        continue

                    ws_id = self._workset_index.pop((record.owner_agent_id, file_path), None)
                    if ws_id:
                        self.workset_manager.release_lease(ws_id)

                    self._emit_event(
                        event_type="ownership.released",
                        payload={
                            "agent_id": record.owner_agent_id,
                            "file_path": file_path,
                            "mode": record.mode,
                        },
                        source=record.owner_agent_id,
                    )
                    self._publish(
                        topic="ownership.claims",
                        sender=record.owner_agent_id,
                        payload={
                            "action": "released",
                            "record": ownership_record_to_dict(record),
                        },
                    )

                    self._agent_index[record.owner_agent_id].discard(file_path)
                    if record.phase_id:
                        self._phase_index[record.phase_id].discard(file_path)

                    if self.crdt is not None:
                        entry = self.crdt.release(file_path=file_path, agent_id=record.owner_agent_id)
                        self._sync_entry(entry)

                if remaining:
                    self._file_index[file_path] = remaining
                else:
                    self._file_index.pop(file_path, None)

    def release_all(self) -> None:
        self.release_files(agent_id="*", files=None)

    def query_ownership(self, files: List[str]) -> Dict[str, OwnershipRecord]:
        result: Dict[str, OwnershipRecord] = {}
        with self._lock:
            for file_path in [self._normalize_path(path) for path in files if path]:
                active = self._active_records(file_path)
                if not active:
                    continue
                result[file_path] = sorted(active, key=lambda rec: rec.claimed_at)[-1]
        return result

    def get_agent_files(self, agent_id: str) -> Set[str]:
        with self._lock:
            return set(self._agent_index.get(agent_id, set()))

    def get_phase_files(self, phase_id: str) -> Set[str]:
        with self._lock:
            return set(self._phase_index.get(phase_id, set()))

    def heartbeat(self, agent_id: str) -> None:
        with self._lock:
            now = time.time()
            for file_path in list(self._agent_index.get(agent_id, set())):
                for record in self._file_index.get(file_path, []):
                    if record.owner_agent_id != agent_id:
                        continue
                    record.heartbeat_at = now
                    record.expires_at = now + record.ttl_seconds
                    ws_id = self._workset_index.get((agent_id, file_path))
                    if ws_id:
                        self.workset_manager.touch_lease(ws_id)

            self._publish(
                topic="ownership.heartbeats",
                sender=agent_id,
                payload={"agent_id": agent_id, "timestamp": now},
            )

    def reap_expired_leases(self) -> None:
        with self._lock:
            expired: Dict[str, List[OwnershipRecord]] = defaultdict(list)
            for file_path, records in list(self._file_index.items()):
                active = []
                for record in records:
                    if record.is_expired:
                        expired[file_path].append(record)
                    else:
                        active.append(record)
                if active:
                    self._file_index[file_path] = active
                else:
                    self._file_index.pop(file_path, None)

            for file_path, records in expired.items():
                for record in records:
                    self._agent_index[record.owner_agent_id].discard(file_path)
                    if record.phase_id:
                        self._phase_index[record.phase_id].discard(file_path)
                    ws_id = self._workset_index.pop((record.owner_agent_id, file_path), None)
                    if ws_id:
                        self.workset_manager.release_lease(ws_id)
                    self._emit_event(
                        event_type="ownership.expired",
                        payload={
                            "agent_id": record.owner_agent_id,
                            "file_path": file_path,
                            "mode": record.mode,
                        },
                        source="ownership_reaper",
                    )

    def can_access(
        self,
        agent_id: str,
        file_path: str,
        access_type: str = "write",
        fencing_token: Optional[str] = None,
    ) -> AccessDecision:
        normalized = self._normalize_path(file_path)
        with self._lock:
            if callable(self.repo_policy_resolver) and access_type != "read":
                allowed, reason = self.repo_policy_resolver(
                    file_path=normalized,
                    phase_id=None,
                    task_id=None,
                    campaign_id=None,
                    repo_id=None,
                    access_mode="target_write",
                    campaign_state=self.current_state_getter() if callable(self.current_state_getter) else None,
                )
                if not allowed:
                    return AccessDecision(allowed=False, reason=str(reason))
            records = self._active_records(normalized)
            if not records:
                return AccessDecision(allowed=True, reason="unclaimed")

            owned = [rec for rec in records if rec.owner_agent_id == agent_id]
            if owned:
                owner_record = owned[0]
                if (
                    access_type != "read"
                    and fencing_token
                    and owner_record.fencing_token
                    and fencing_token != owner_record.fencing_token
                ):
                    return AccessDecision(
                        allowed=False,
                        reason="stale_fencing_token",
                        owner_info=owner_record,
                    )
                if access_type == "read":
                    return AccessDecision(allowed=True, reason="owner_read", owner_info=owner_record)
                return AccessDecision(allowed=True, reason="owner_write", owner_info=owner_record)

            # Any remote non-local owner blocks local write attempts.
            for record in records:
                if record.owner_instance_id != self.instance_id:
                    return AccessDecision(
                        allowed=False,
                        reason="instance_claim",
                        owner_info=record,
                        can_negotiate=(record.mode == "collaborative"),
                    )

            if access_type == "read":
                exclusive = next((rec for rec in records if rec.mode == "exclusive"), None)
                if exclusive is not None:
                    return AccessDecision(
                        allowed=False,
                        reason="exclusive_lock",
                        owner_info=exclusive,
                    )
                return AccessDecision(allowed=True, reason="shared_read")

            collaborative = next((rec for rec in records if rec.mode == "collaborative"), None)
            if collaborative is not None:
                return AccessDecision(
                    allowed=False,
                    reason="collaborative_lock",
                    owner_info=collaborative,
                    can_negotiate=True,
                )

            shared = next((rec for rec in records if rec.mode == "shared_read"), None)
            if shared is not None:
                return AccessDecision(
                    allowed=False,
                    reason="shared_read_owner_only",
                    owner_info=shared,
                )

            owner = records[0]
            return AccessDecision(
                allowed=False,
                reason="exclusive_lock",
                owner_info=owner,
            )

    def get_status_snapshot(self) -> Dict[str, object]:
        with self._lock:
            file_owners: Dict[str, List[Dict[str, object]]] = {}
            for file_path, records in self._file_index.items():
                active = [record for record in records if not record.is_expired]
                if not active:
                    continue
                file_owners[file_path] = [ownership_record_to_dict(record) for record in active]

            return {
                "instance_id": self.instance_id,
                "total_claimed_files": len(file_owners),
                "total_agents": len([agent for agent, files in self._agent_index.items() if files]),
                "file_owners": file_owners,
                "symbol_owners": {
                    f"{path}::{symbol}": ownership_record_to_dict(record)
                    for (path, symbol), record in self._symbol_index.items()
                    if not record.is_expired
                },
            }

    def current_fencing_token(self, file_path: str) -> Optional[str]:
        normalized = self._normalize_path(file_path)
        with self._lock:
            active = self._active_records(normalized)
            if not active:
                return None
            return sorted(active, key=lambda record: record.lease_epoch)[-1].fencing_token

    def claim_symbols(
        self,
        agent_id: str,
        file_path: str,
        symbols: List[str],
        *,
        phase_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> ClaimResult:
        normalized = self._normalize_path(file_path)
        granted: List[str] = []
        denied: List[DeniedFile] = []
        with self._lock:
            active_file_claims = self._active_records(normalized)
            denied_file = self._allow_claim(agent_id, "collaborative", active_file_claims)
            if denied_file is not None:
                return ClaimResult(success=False, granted_files=[], denied_files=[denied_file])
            for symbol in [str(item).strip() for item in symbols if str(item).strip()]:
                key = (normalized, symbol)
                current = self._symbol_index.get(key)
                if current is not None and not current.is_expired and current.owner_agent_id != agent_id:
                    denied.append(
                        DeniedFile(
                            file_path=f"{normalized}::{symbol}",
                            current_owner=current.owner_agent_id,
                            ownership_mode=current.mode,
                            reason="symbol_conflict",
                            required_lease_epoch=current.lease_epoch,
                        )
                    )
                    continue
                lease_epoch = self._lease_epochs[f"{normalized}::{symbol}"] + 1
                self._lease_epochs[f"{normalized}::{symbol}"] = lease_epoch
                self._symbol_index[key] = OwnershipRecord(
                    file_path=normalized,
                    owner_agent_id=agent_id,
                    owner_instance_id=self.instance_id,
                    mode="exclusive",
                    claimed_at=time.time(),
                    heartbeat_at=time.time(),
                    ttl_seconds=self.default_ttl_seconds,
                    phase_id=phase_id,
                    task_id=task_id,
                    lease_epoch=lease_epoch,
                    fencing_token=f"{self.instance_id}:{agent_id}:{normalized}::{symbol}:{lease_epoch}",
                )
                granted.append(f"{normalized}::{symbol}")
        return ClaimResult(
            success=not denied,
            granted_files=granted,
            denied_files=denied,
            suggested_resolution=(
                None if not denied else "Split work by symbol or wait for the active claim to expire."
            ),
        )

    def query_symbol_ownership(
        self,
        file_path: str,
        symbols: List[str],
    ) -> Dict[str, OwnershipRecord]:
        normalized = self._normalize_path(file_path)
        with self._lock:
            return {
                symbol: self._symbol_index[(normalized, symbol)]
                for symbol in symbols
                if (normalized, symbol) in self._symbol_index
                and not self._symbol_index[(normalized, symbol)].is_expired
            }


def build_trust_zone_resolver(
    *,
    local_zone: str = "internal",
    protected_prefixes: Optional[Iterable[str]] = None,
    quarantine_prefixes: Optional[Iterable[str]] = None,
):
    zone_order = {"external": 0, "campaign": 1, "internal": 2, "maintainer": 3}
    protected = tuple(
        protected_prefixes
        or ("core/native/", "core/architect/", "shared_kernel/", "saguaro/state/")
    )
    quarantine = tuple(quarantine_prefixes or (".anvil/quarantine/",))

    def resolver(**kwargs):
        file_path = str(kwargs.get("file_path") or "")
        access_mode = str(kwargs.get("access_mode") or "")
        required_zone = "internal"
        if any(file_path.startswith(prefix) for prefix in protected):
            required_zone = "maintainer"
        if any(file_path.startswith(prefix) for prefix in quarantine):
            required_zone = "maintainer"
        if access_mode in {"analysis_readonly", "analysis_extract_only", "audit_readonly"}:
            return True, f"policy.zone.assigned:{local_zone}"
        if zone_order.get(local_zone, 0) < zone_order.get(required_zone, 0):
            return False, f"policy.zone.denied:{required_zone}"
        return True, f"policy.zone.assigned:{required_zone}"

    return resolver
