"""Typed models for the Anvil Latent Memory Fabric."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import time
import uuid
from typing import Any, Dict


def stable_json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload or {}, default=str, sort_keys=True, separators=(",", ":"))


def now_ts() -> float:
    return time.time()


def canonical_hash_for(payload: Dict[str, Any], provenance: Dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(stable_json_dumps(payload).encode("utf-8"))
    digest.update(b"\n")
    digest.update(stable_json_dumps(provenance).encode("utf-8"))
    return f"sha256:{digest.hexdigest()}"


@dataclass(slots=True)
class MemoryObject:
    memory_id: str
    memory_kind: str
    campaign_id: str
    workspace_id: str = ""
    repo_context: str = ""
    session_id: str = ""
    source_system: str = ""
    created_at: float = field(default_factory=now_ts)
    observed_at: float = field(default_factory=now_ts)
    updated_at: float = field(default_factory=now_ts)
    schema_version: str = "almf.v1"
    canonical_hash: str = ""
    payload_json: Dict[str, Any] = field(default_factory=dict)
    summary_text: str = ""
    provenance_json: Dict[str, Any] = field(default_factory=dict)
    retention_class: str = "durable"
    importance_score: float = 0.5
    confidence_score: float = 0.5
    sensitivity_class: str = "internal"
    lifecycle_state: str = "active"
    hypothesis_id: str = ""
    document_id: str = ""
    claim_id: str = ""
    experiment_id: str = ""
    artifact_id: str = ""
    task_packet_id: str = ""
    lane_id: str = ""
    operator_id: str = ""
    thread_id: str = ""
    model_family: str = ""
    model_revision: str = ""

    def finalize(self) -> "MemoryObject":
        if not self.canonical_hash:
            self.canonical_hash = canonical_hash_for(
                self.payload_json,
                self.provenance_json,
            )
        if not self.summary_text:
            self.summary_text = stable_json_dumps(self.payload_json)[:240]
        return self


@dataclass(slots=True)
class MemoryEdge:
    src_memory_id: str
    dst_memory_id: str
    edge_type: str
    weight: float = 1.0
    valid_from: float = field(default_factory=now_ts)
    valid_to: float | None = None
    recorded_at: float = field(default_factory=now_ts)
    evidence_json: Dict[str, Any] = field(default_factory=dict)
    edge_id: str = field(default_factory=lambda: f"edge_{uuid.uuid4().hex}")


@dataclass(slots=True)
class LatentPackageRecord:
    latent_package_id: str
    memory_id: str
    branch_id: str = ""
    model_family: str = "qsg-python"
    model_revision: str = "v1"
    tokenizer_hash: str = ""
    adapter_hash: str = ""
    prompt_protocol_hash: str = ""
    hidden_dim: int = 0
    qsg_runtime_version: str = "qsg.v1"
    rope_config_hash: str = ""
    quantization_profile: str = ""
    capture_stage: str = ""
    tensor_format: str = "safetensors"
    tensor_uri: str = ""
    summary_text: str = ""
    compatibility_json: Dict[str, Any] = field(default_factory=dict)
    latent_packet_abi_version: int = 2
    execution_capsule_version: int = 2
    execution_capsule_id: str = ""
    capability_digest: str = ""
    delta_watermark_json: Dict[str, Any] = field(default_factory=dict)
    supporting_memory_ids: list[str] = field(default_factory=list)
    creation_reason: str = ""
    created_at: float = field(default_factory=now_ts)
    expires_at: float | None = None


@dataclass(slots=True)
class RepoDeltaMemoryRecord:
    delta_id: str = ""
    workspace_id: str = ""
    capability_digest: str = ""
    changed_paths: list[str] = field(default_factory=list)
    path_count: int = 0
    summary_text: str = ""
    semantic_impact_hint: str = ""
    source_stage: str = ""
    created_at: float = field(default_factory=now_ts)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "delta_id": str(self.delta_id),
            "workspace_id": str(self.workspace_id),
            "capability_digest": str(self.capability_digest),
            "changed_paths": list(self.changed_paths),
            "path_count": int(self.path_count or len(self.changed_paths)),
            "summary_text": str(self.summary_text or self.semantic_impact_hint),
            "semantic_impact_hint": str(self.semantic_impact_hint),
            "source_stage": str(self.source_stage),
            "created_at": float(self.created_at),
        }

    @classmethod
    def from_delta_watermark(
        cls,
        delta_watermark: Dict[str, Any] | None,
        *,
        capability_digest: str = "",
        summary_text: str = "",
        source_stage: str = "",
    ) -> "RepoDeltaMemoryRecord":
        payload = dict(delta_watermark or {})
        changed_paths = [
            str(path)
            for path in list(payload.get("changed_paths") or [])
            if str(path).strip()
        ]
        hint = str(summary_text or "").strip()
        if not hint:
            if changed_paths:
                preview = ", ".join(changed_paths[:3])
                suffix = " ..." if len(changed_paths) > 3 else ""
                hint = f"{len(changed_paths)} changed paths: {preview}{suffix}"
            else:
                hint = "no changed paths recorded"
        return cls(
            delta_id=str(payload.get("delta_id") or payload.get("event_id") or ""),
            workspace_id=str(payload.get("workspace_id") or ""),
            capability_digest=str(capability_digest),
            changed_paths=changed_paths,
            path_count=int(payload.get("path_count") or len(changed_paths)),
            summary_text=hint,
            semantic_impact_hint=hint,
            source_stage=str(source_stage),
            created_at=float(payload.get("created_at") or now_ts()),
        )


@dataclass(slots=True)
class MemoryReadRecord:
    campaign_id: str
    query_kind: str
    query_text: str
    planner_mode: str
    result_memory_ids_json: list[str]
    latency_ms: float
    created_at: float = field(default_factory=now_ts)
    read_id: str = field(default_factory=lambda: f"read_{uuid.uuid4().hex}")


@dataclass(slots=True)
class MemoryFeedbackRecord:
    read_id: str
    consumer_system: str
    usefulness_score: float
    grounding_score: float
    citation_score: float
    token_savings_estimate: float
    outcome_json: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=now_ts)
    feedback_id: str = field(default_factory=lambda: f"feedback_{uuid.uuid4().hex}")


def build_memory_object(
    memory_kind: str,
    payload_json: Dict[str, Any],
    *,
    campaign_id: str = "global",
    workspace_id: str = "",
    repo_context: str = "",
    session_id: str = "",
    source_system: str = "",
    summary_text: str = "",
    provenance_json: Dict[str, Any] | None = None,
    retention_class: str = "durable",
    importance_score: float = 0.5,
    confidence_score: float = 0.5,
    sensitivity_class: str = "internal",
    lifecycle_state: str = "active",
    observed_at: float | None = None,
    **extra_ids: Any,
) -> MemoryObject:
    created_at = now_ts()
    memory = MemoryObject(
        memory_id=f"mem_{uuid.uuid4().hex}",
        memory_kind=memory_kind,
        campaign_id=campaign_id,
        workspace_id=workspace_id,
        repo_context=repo_context,
        session_id=session_id,
        source_system=source_system,
        created_at=created_at,
        observed_at=created_at if observed_at is None else float(observed_at),
        updated_at=created_at,
        payload_json=dict(payload_json or {}),
        summary_text=summary_text,
        provenance_json=dict(provenance_json or {}),
        retention_class=retention_class,
        importance_score=float(importance_score),
        confidence_score=float(confidence_score),
        sensitivity_class=sensitivity_class,
        lifecycle_state=lifecycle_state,
        hypothesis_id=str(extra_ids.get("hypothesis_id") or ""),
        document_id=str(extra_ids.get("document_id") or ""),
        claim_id=str(extra_ids.get("claim_id") or ""),
        experiment_id=str(extra_ids.get("experiment_id") or ""),
        artifact_id=str(extra_ids.get("artifact_id") or ""),
        task_packet_id=str(extra_ids.get("task_packet_id") or ""),
        lane_id=str(extra_ids.get("lane_id") or ""),
        operator_id=str(extra_ids.get("operator_id") or ""),
        thread_id=str(extra_ids.get("thread_id") or ""),
        model_family=str(extra_ids.get("model_family") or ""),
        model_revision=str(extra_ids.get("model_revision") or ""),
    )
    return memory.finalize()
