"""Campaign-scoped structured artifact registry."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from core.artifacts.renderers import render_artifact_document
from core.campaign.state_store import CampaignStateStore
from core.campaign.workspace import CampaignWorkspace


class ArtifactApprovalState(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ACCEPTED = "accepted"


@dataclass
class ArtifactDependencyEdge:
    artifact_id: str
    depends_on_artifact_id: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class ArtifactProvenanceLink:
    kind: str
    target: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactVariant:
    name: str
    path: str
    content_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactApprovalRecord:
    state: str
    approved_by: str
    notes: str = ""
    approved_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactVersionRecord:
    version: int
    canonical_path: str
    rendered_path: str | None
    content_hash: str
    blocking: bool = False
    provenance: list[ArtifactProvenanceLink] = field(default_factory=list)
    dependencies: list[ArtifactDependencyEdge] = field(default_factory=list)
    variants: list[ArtifactVariant] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "canonical_path": self.canonical_path,
            "rendered_path": self.rendered_path,
            "content_hash": self.content_hash,
            "blocking": self.blocking,
            "provenance": [item.to_dict() for item in self.provenance],
            "dependencies": [item.to_dict() for item in self.dependencies],
            "variants": [item.to_dict() for item in self.variants],
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }


@dataclass
class ArtifactRecord:
    artifact_id: str
    family: str
    name: str
    latest_version: int = 0
    latest_approval_state: str = ArtifactApprovalState.PENDING.value
    blocking: bool = False
    status: str = "published"
    metadata: dict[str, Any] = field(default_factory=dict)
    versions: list[ArtifactVersionRecord] = field(default_factory=list)
    approvals: list[ArtifactApprovalRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "family": self.family,
            "name": self.name,
            "latest_version": self.latest_version,
            "latest_approval_state": self.latest_approval_state,
            "blocking": self.blocking,
            "status": self.status,
            "metadata": dict(self.metadata),
            "versions": [item.to_dict() for item in self.versions],
            "approvals": [item.to_dict() for item in self.approvals],
        }


class CampaignArtifactRegistry:
    """Stores versioned artifacts for campaign workspaces and simple fixtures."""

    def __init__(
        self,
        campaign_root_or_id: str,
        workspace: CampaignWorkspace | None = None,
        state_store: CampaignStateStore | None = None,
    ) -> None:
        self.state_store = state_store
        self.workspace = workspace
        if workspace is None:
            self.campaign_id = "campaign"
            self.root_dir = Path(campaign_root_or_id).resolve()
            self.index_path = self.root_dir / ".artifact_registry.json"
        else:
            self.campaign_id = campaign_root_or_id
            self.root_dir = Path(workspace.root_dir)
            self.index_path = self.root_dir / "artifacts" / "index.json"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    def publish(
        self,
        *,
        artifact_id: str,
        family: str,
        name: str,
        canonical_payload: dict[str, Any] | list[Any],
        provenance: list[ArtifactProvenanceLink] | None = None,
        dependencies: list[ArtifactDependencyEdge] | None = None,
        variants: list[ArtifactVariant] | None = None,
        rendered_document: str | None = None,
        approval_state: str | None = None,
        blocking: bool = False,
        status: str = "published",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactVersionRecord:
        index = self._load_index()
        current = self._record_from_payload(index.get(artifact_id))
        version = 1 if current is None else current.latest_version + 1
        family_dir = self._artifact_family_dir(family)
        base_name = f"{name}_v{version}"
        canonical_path = family_dir / f"{base_name}.json"
        canonical_path.write_text(
            json.dumps(canonical_payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        if rendered_document is None:
            rendered_document = render_artifact_document(family, canonical_payload)
        rendered_path = family_dir / f"{base_name}.md"
        rendered_path.write_text(rendered_document, encoding="utf-8")
        content_hash = hashlib.sha256(
            json.dumps(canonical_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        version_record = ArtifactVersionRecord(
            version=version,
            canonical_path=str(canonical_path),
            rendered_path=str(rendered_path),
            content_hash=content_hash,
            blocking=blocking,
            provenance=list(provenance or []),
            dependencies=list(dependencies or []),
            variants=list(variants or []),
            metadata=dict(metadata or {}),
        )
        record = current or ArtifactRecord(
            artifact_id=artifact_id,
            family=family,
            name=name,
        )
        record.family = family
        record.name = name
        record.latest_version = version
        record.blocking = blocking
        record.status = status
        record.metadata = dict(metadata or {})
        if approval_state is not None:
            record.latest_approval_state = approval_state
        elif not record.latest_approval_state:
            record.latest_approval_state = ArtifactApprovalState.PENDING.value
        record.versions.append(version_record)
        index[artifact_id] = record.to_dict()
        self._save_index(index)
        self._sync_state_store(record, version_record)
        return version_record

    def emit(
        self,
        family: str,
        name: str,
        canonical: dict[str, Any] | list[Any],
        rendered: str | None = None,
        *,
        artifact_id: str | None = None,
        blocking: bool = False,
        approval_state: str = ArtifactApprovalState.PENDING.value,
        status: str = "published",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactVersionRecord:
        return self.publish(
            artifact_id=artifact_id or f"{self.campaign_id}:{family}:{name}",
            family=family,
            name=name,
            canonical_payload=canonical,
            rendered_document=rendered,
            blocking=blocking,
            approval_state=approval_state,
            status=status,
            metadata=metadata,
        )

    def approve(
        self,
        artifact_id: str,
        *,
        approved_by: str,
        state: str = ArtifactApprovalState.APPROVED.value,
        notes: str = "",
    ) -> ArtifactApprovalRecord:
        index = self._load_index()
        record = self.get_artifact(artifact_id)
        approval = ArtifactApprovalRecord(
            state=state,
            approved_by=approved_by,
            notes=notes,
        )
        record.latest_approval_state = state
        record.approvals.append(approval)
        index[artifact_id] = record.to_dict()
        self._save_index(index)
        if self.state_store is not None:
            self.state_store.record_artifact_approval(
                artifact_id,
                state=state,
                approved_by=approved_by,
                notes=notes,
            )
        return approval

    def set_blocking(self, artifact_id: str, *, blocking: bool) -> None:
        index = self._load_index()
        record = self.get_artifact(artifact_id)
        record.blocking = blocking
        index[artifact_id] = record.to_dict()
        self._save_index(index)
        if self.state_store is not None:
            latest = record.versions[-1]
            self.state_store.upsert_artifact(
                {
                    "artifact_id": artifact_id,
                    "campaign_id": self.campaign_id,
                    "family": record.family,
                    "name": record.name,
                    "version": latest.version,
                    "canonical_path": latest.canonical_path,
                    "rendered_path": latest.rendered_path,
                    "approval_state": record.latest_approval_state,
                    "blocking": blocking,
                    "status": record.status,
                    "metadata": self._record_metadata(record, latest),
                }
            )

    def get_artifact(self, artifact_id: str) -> ArtifactRecord:
        index = self._load_index()
        payload = index.get(artifact_id)
        if payload is None:
            raise KeyError(f"Unknown artifact_id: {artifact_id}")
        return self._record_from_payload(payload)

    def list_artifacts(self) -> list[dict[str, Any]]:
        index = self._load_index()
        return [self._record_from_payload(payload).to_dict() for payload in index.values()]

    def list(self, family: str | None = None) -> list[dict[str, Any]]:
        artifacts = self.list_artifacts()
        if family is None:
            return artifacts
        return [item for item in artifacts if item["family"] == family]

    def get_latest_version(self, artifact_id: str) -> ArtifactVersionRecord:
        record = self.get_artifact(artifact_id)
        if not record.versions:
            raise KeyError(f"Artifact has no versions: {artifact_id}")
        return record.versions[-1]

    def latest(self, family: str, name: str) -> dict[str, Any] | None:
        for item in self.list_artifacts():
            if item["family"] == family and item["name"] == name:
                return item
        return None

    def get_blocking_artifacts(self) -> list[ArtifactRecord]:
        records: list[ArtifactRecord] = []
        for item in self._load_index().values():
            record = self._record_from_payload(item)
            if record.blocking and record.latest_approval_state != ArtifactApprovalState.APPROVED.value:
                records.append(record)
        return records

    def get_resume_pointer(self, artifact_id: str) -> int:
        return self.get_artifact(artifact_id).latest_version

    def export_index(self) -> dict[str, Any]:
        return self._load_index()

    def _artifact_family_dir(self, family: str) -> Path:
        if self.workspace is not None:
            return Path(self.workspace.artifact_family_dir(family))
        path = self.root_dir / family
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {}
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _save_index(self, payload: dict[str, Any]) -> None:
        self.index_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    def _record_from_payload(self, payload: dict[str, Any] | None) -> ArtifactRecord | None:
        if payload is None:
            return None
        versions = [
            ArtifactVersionRecord(
                version=int(item["version"]),
                canonical_path=item["canonical_path"],
                rendered_path=item.get("rendered_path"),
                content_hash=item["content_hash"],
                blocking=bool(item.get("blocking", False)),
                provenance=[
                    ArtifactProvenanceLink(**link)
                    for link in item.get("provenance", [])
                ],
                dependencies=[
                    ArtifactDependencyEdge(**edge)
                    for edge in item.get("dependencies", [])
                ],
                variants=[
                    ArtifactVariant(**variant)
                    for variant in item.get("variants", [])
                ],
                metadata=dict(item.get("metadata") or {}),
                created_at=float(item.get("created_at", time.time())),
            )
            for item in payload.get("versions", [])
        ]
        approvals = [
            ArtifactApprovalRecord(
                state=item["state"],
                approved_by=item["approved_by"],
                notes=item.get("notes", ""),
                approved_at=float(item.get("approved_at", time.time())),
            )
            for item in payload.get("approvals", [])
        ]
        return ArtifactRecord(
            artifact_id=payload["artifact_id"],
            family=payload["family"],
            name=payload["name"],
            latest_version=int(payload.get("latest_version", len(versions))),
            latest_approval_state=payload.get(
                "latest_approval_state",
                ArtifactApprovalState.PENDING.value,
            ),
            blocking=bool(payload.get("blocking", False)),
            status=payload.get("status", "published"),
            metadata=dict(payload.get("metadata") or {}),
            versions=versions,
            approvals=approvals,
        )

    def _record_metadata(
        self,
        record: ArtifactRecord,
        version: ArtifactVersionRecord,
    ) -> dict[str, Any]:
        return {
            **record.metadata,
            "provenance": [item.to_dict() for item in version.provenance],
            "dependencies": [item.to_dict() for item in version.dependencies],
            "variants": [item.to_dict() for item in version.variants],
            "approval_count": len(record.approvals),
        }

    def _sync_state_store(
        self,
        record: ArtifactRecord,
        version: ArtifactVersionRecord,
    ) -> None:
        if self.state_store is None:
            return
        self.state_store.upsert_artifact(
            {
                "artifact_id": record.artifact_id,
                "campaign_id": self.campaign_id,
                "family": record.family,
                "name": record.name,
                "version": version.version,
                "canonical_path": version.canonical_path,
                "rendered_path": version.rendered_path,
                "approval_state": record.latest_approval_state,
                "blocking": record.blocking,
                "status": record.status,
                "metadata": self._record_metadata(record, version),
            }
        )
        self.state_store.execute(
            """
            INSERT INTO artifact_versions (
                artifact_id,
                version,
                canonical_path,
                rendered_path,
                content_hash,
                metadata_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.artifact_id,
                version.version,
                version.canonical_path,
                version.rendered_path,
                version.content_hash,
                json.dumps(self._record_metadata(record, version), default=str),
                version.created_at,
            ),
        )


__all__ = [
    "ArtifactApprovalRecord",
    "ArtifactApprovalState",
    "ArtifactDependencyEdge",
    "ArtifactProvenanceLink",
    "ArtifactRecord",
    "ArtifactVariant",
    "ArtifactVersionRecord",
    "CampaignArtifactRegistry",
]
