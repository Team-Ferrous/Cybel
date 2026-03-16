"""Bridge ALMF latent packages to QSG replay surfaces."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Iterable, Optional

from core.memory.fabric import (
    LatentCompatibilityPolicy,
    LatentPackageRecord,
    MemoryTierPolicy,
    MemoryFabricStore,
    MemoryProjector,
    RepoDeltaMemoryRecord,
    RetentionPolicy,
)
from core.qsg.runtime_contracts import DeltaWatermark, MissionReplayDescriptor
from shared_kernel.event_store import get_event_store


class QSGLatentBridge:
    """Capture and replay latent packages through ALMF."""

    def __init__(
        self,
        store: MemoryFabricStore,
        projector: MemoryProjector,
    ) -> None:
        self.store = store
        self.projector = projector
        self.event_store = get_event_store()

    def capture_summary_package(
        self,
        *,
        memory_id: str,
        summary_text: str,
        capture_stage: str,
        model_family: str = "qsg-python",
        model_revision: str = "v1",
        prompt_protocol_hash: str = "almf.v1",
        supporting_memory_ids: Optional[Iterable[str]] = None,
        creation_reason: str = "",
        retention_class: str = "session",
        capability_digest: str = "",
        delta_watermark: dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        tensor = self.projector.latent_tensor(summary_text, hidden_dim=1)
        expires_at = RetentionPolicy.expires_at(
            retention_class,
            created_at=time.time(),
        )
        normalized_delta = DeltaWatermark.from_dict(delta_watermark)
        repo_delta_memory = RepoDeltaMemoryRecord.from_delta_watermark(
            normalized_delta.as_dict(),
            capability_digest=capability_digest,
            summary_text=summary_text,
            source_stage=capture_stage,
        )
        package = LatentPackageRecord(
            latent_package_id=f"latent_{uuid.uuid4().hex}",
            memory_id=memory_id,
            branch_id=f"{memory_id}:{capture_stage}",
            model_family=model_family,
            model_revision=model_revision,
            tokenizer_hash="tokenizer:unknown",
            adapter_hash="adapter:none",
            prompt_protocol_hash=prompt_protocol_hash,
            hidden_dim=int(tensor.shape[-1]),
            qsg_runtime_version="qsg.v1",
            quantization_profile="float16",
            capture_stage=capture_stage,
            summary_text=summary_text,
            latent_packet_abi_version=3,
            execution_capsule_version=3,
            capability_digest=str(capability_digest),
            delta_watermark_json=normalized_delta.as_dict(),
            compatibility_json={
                "model_family": model_family,
                "hidden_dim": int(tensor.shape[-1]),
                "tokenizer_hash": "tokenizer:unknown",
                "prompt_protocol_hash": prompt_protocol_hash,
                "qsg_runtime_version": "qsg.v1",
                "latent_packet_abi_version": 3,
                "execution_capsule_version": 3,
                "capability_digest": str(capability_digest),
                "delta_watermark": normalized_delta.as_dict(),
                "tensor_codec": "float16",
                "segments": [
                    {
                        "segment_id": f"{memory_id}:{capture_stage}:summary",
                        "segment_kind": "summary_state",
                        "row_start": 0,
                        "row_count": int(tensor.shape[0]),
                        "hidden_dim": int(tensor.shape[-1]),
                        "codec": "float16",
                        "importance": 0.8,
                    }
                ],
                "segment_count": 1,
                "repo_delta_memory": repo_delta_memory.as_dict(),
            },
            supporting_memory_ids=list(supporting_memory_ids or []),
            creation_reason=creation_reason,
            created_at=time.time(),
            expires_at=expires_at,
        )
        repo_delta_memory_id = self._persist_repo_delta_memory(
            memory_id=memory_id,
            repo_delta_memory=repo_delta_memory,
            source_system="QSGLatentBridge.capture_summary_package",
        )
        if repo_delta_memory_id:
            package.supporting_memory_ids.append(repo_delta_memory_id)
            package.compatibility_json["repo_delta_memory_id"] = repo_delta_memory_id
        self.store.put_latent_package(package, tensor=tensor)
        return self._capsule_descriptor(
            package=package,
            mode="captured",
            capability_digest=capability_digest,
            delta_watermark=normalized_delta.as_dict(),
        )

    def capture_engine_state(
        self,
        *,
        engine,
        memory_id: str,
        request_id: str,
        summary_text: str,
        capture_stage: str,
        model_family: str = "qsg-python",
        model_revision: str = "v1",
        prompt_protocol_hash: str = "almf.v1",
        supporting_memory_ids: Optional[Iterable[str]] = None,
        creation_reason: str = "",
    ) -> Dict[str, Any]:
        captured = engine.capture_latent_state(request_id)
        if captured is None:
            return self.capture_summary_package(
                memory_id=memory_id,
                summary_text=summary_text,
                capture_stage=capture_stage,
                model_family=model_family,
                model_revision=model_revision,
                prompt_protocol_hash=prompt_protocol_hash,
                supporting_memory_ids=supporting_memory_ids,
                creation_reason=creation_reason,
            )
        execution_capsule = dict(captured.get("execution_capsule") or {})
        latent_packet = dict(captured.get("latent_packet") or {})
        tensor = captured.get("tensor") or latent_packet.get("tensor") or [[0.0]]
        hidden_dim = int(
            captured.get("hidden_dim") or latent_packet.get("hidden_dim") or 0
        )
        capability_digest = str(latent_packet.get("capability_digest") or "")
        segments = list(latent_packet.get("segments") or [])
        delta_watermark = DeltaWatermark.from_dict(
            latent_packet.get("delta_watermark")
            or execution_capsule.get("delta_watermark")
            or {}
        )
        repo_delta_memory = RepoDeltaMemoryRecord.from_delta_watermark(
            delta_watermark.as_dict(),
            capability_digest=capability_digest,
            summary_text=summary_text,
            source_stage=capture_stage,
        )
        package = LatentPackageRecord(
            latent_package_id=f"latent_{uuid.uuid4().hex}",
            memory_id=memory_id,
            branch_id=f"{memory_id}:{capture_stage}",
            model_family=model_family,
            model_revision=model_revision,
            tokenizer_hash="tokenizer:unknown",
            adapter_hash="adapter:none",
            prompt_protocol_hash=prompt_protocol_hash,
            hidden_dim=hidden_dim,
            qsg_runtime_version="qsg.v1",
            quantization_profile="float16",
            capture_stage=capture_stage,
            summary_text=summary_text,
            latent_packet_abi_version=max(
                3, int(latent_packet.get("abi_version") or 2)
            ),
            execution_capsule_version=max(
                3, int(execution_capsule.get("version") or 2)
            ),
            execution_capsule_id=str(execution_capsule.get("capsule_id") or ""),
            capability_digest=capability_digest,
            delta_watermark_json=delta_watermark.as_dict(),
            compatibility_json={
                "model_family": model_family,
                "hidden_dim": hidden_dim,
                "tokenizer_hash": "tokenizer:unknown",
                "prompt_protocol_hash": prompt_protocol_hash,
                "qsg_runtime_version": "qsg.v1",
                "latent_packet_abi_version": max(
                    3, int(latent_packet.get("abi_version") or 2)
                ),
                "execution_capsule_version": max(
                    3, int(execution_capsule.get("version") or 2)
                ),
                "execution_capsule_id": str(execution_capsule.get("capsule_id") or ""),
                "capability_digest": capability_digest,
                "delta_watermark": delta_watermark.as_dict(),
                "tensor_codec": "float16",
                "segments": segments,
                "segment_count": len(segments),
                "repo_delta_memory": repo_delta_memory.as_dict(),
            },
            supporting_memory_ids=list(supporting_memory_ids or []),
            creation_reason=creation_reason,
            created_at=time.time(),
            expires_at=RetentionPolicy.expires_at("session", created_at=time.time()),
        )
        repo_delta_memory_id = self._persist_repo_delta_memory(
            memory_id=memory_id,
            repo_delta_memory=repo_delta_memory,
            source_system="QSGLatentBridge.capture_engine_state",
        )
        if repo_delta_memory_id:
            package.supporting_memory_ids.append(repo_delta_memory_id)
            package.compatibility_json["repo_delta_memory_id"] = repo_delta_memory_id
        self.store.put_latent_package(package, tensor=tensor)
        return {
            **self._capsule_descriptor(
                package=package,
                mode="captured",
                capability_digest=capability_digest,
                delta_watermark=delta_watermark.as_dict(),
            ),
            "request_id": request_id,
        }

    @staticmethod
    def _capsule_descriptor(
        *,
        package: LatentPackageRecord,
        mode: str,
        capability_digest: str,
        delta_watermark: dict[str, Any],
    ) -> Dict[str, Any]:
        capsule_id = str(package.execution_capsule_id or f"capsule_{package.latent_package_id[7:19]}")
        return {
            "capsule_id": capsule_id,
            "latent_package_id": package.latent_package_id,
            "supporting_memory_ids": sorted({str(item) for item in package.supporting_memory_ids}),
            "capability_digest": str(capability_digest),
            "delta_watermark": dict(delta_watermark or {}),
            "mode": mode,
        }

    def replay(
        self,
        *,
        engine,
        memory_id: str,
        model_family: str = "qsg-python",
        hidden_dim: Optional[int] = None,
        tokenizer_hash: str = "tokenizer:unknown",
        prompt_protocol_hash: str = "almf.v1",
        qsg_runtime_version: str = "qsg.v1",
        quantization_profile: str = "float32",
        capability_digest: str = "",
        delta_watermark: dict[str, Any] | None = None,
        target_request_id: str | None = None,
    ) -> Dict[str, Any]:
        request_id = target_request_id or f"replay_{uuid.uuid4().hex[:10]}"
        package = self.store.latest_latent_package(memory_id)
        if package is None:
            self.event_store.record_qsg_replay_event(
                request_id=request_id,
                stage="missing_package",
                payload={"memory_id": memory_id},
                source="QSGLatentBridge.replay",
            )
            return {
                "restored": False,
                "mode": "degraded",
                "reason": "missing_package",
                "memory_tier_decision": MemoryTierPolicy()
                .choose(purpose="mission_replay")
                .as_dict(),
            }
        compatibility_json = dict(package.get("compatibility_json") or {})
        effective_quantization_profile = str(
            quantization_profile
            or package.get("quantization_profile")
            or compatibility_json.get("tensor_codec")
            or ""
        )
        if effective_quantization_profile == "float32":
            effective_quantization_profile = str(
                package.get("quantization_profile")
                or compatibility_json.get("tensor_codec")
                or effective_quantization_profile
            )
        hidden = int(hidden_dim or package.get("hidden_dim") or 0)
        compatibility = LatentCompatibilityPolicy.evaluate(
            package,
            model_family=model_family,
            hidden_dim=hidden,
            tokenizer_hash=tokenizer_hash,
            prompt_protocol_hash=prompt_protocol_hash,
            qsg_runtime_version=qsg_runtime_version,
            quantization_profile=effective_quantization_profile,
            capability_digest=capability_digest,
            delta_watermark=delta_watermark,
        )
        runtime_status = {}
        metrics_snapshot = getattr(engine, "metrics_snapshot", None)
        if callable(metrics_snapshot):
            try:
                runtime_status = dict(metrics_snapshot() or {})
            except Exception:
                runtime_status = {}
        repo_delta_memory = dict(
            (package.get("compatibility_json") or {}).get("repo_delta_memory") or {}
        )
        memory_tier_decision = MemoryTierPolicy().choose(
            purpose="mission_replay",
            runtime_status=runtime_status,
            latent_package=package,
            compatibility=compatibility,
            repo_delta_memory=repo_delta_memory,
        )
        self.event_store.record_qsg_replay_event(
            request_id=request_id,
            stage="policy_evaluated",
            payload={
                "memory_id": memory_id,
                "latent_package_id": str(package.get("latent_package_id") or ""),
                "memory_tier_decision": memory_tier_decision.as_dict(),
                "repo_delta_memory": repo_delta_memory,
            },
            source="QSGLatentBridge.replay",
        )
        if not compatibility["compatible"]:
            descriptor = self._mission_replay_descriptor(
                request_id=request_id,
                memory_id=memory_id,
                package=package,
                memory_tier_decision=memory_tier_decision.as_dict(),
                restored=False,
                mode="degraded",
                replay_tape_path=self.event_store.export_replay_tape(request_id)[
                    "path"
                ],
            )
            self.event_store.record_qsg_replay_event(
                request_id=request_id,
                stage="compatibility_mismatch",
                payload={
                    "memory_id": memory_id,
                    "compatibility": compatibility,
                    "mission_replay_descriptor": descriptor.as_dict(),
                },
                source="QSGLatentBridge.replay",
            )
            return {
                "restored": False,
                "mode": "degraded",
                "reason": "compatibility_mismatch",
                "compatibility": compatibility,
                "memory_tier_decision": memory_tier_decision.as_dict(),
                "mission_replay_descriptor": descriptor.as_dict(),
            }
        tensor = self.store.load_latent_tensor(package)
        if tensor is None:
            descriptor = self._mission_replay_descriptor(
                request_id=request_id,
                memory_id=memory_id,
                package=package,
                memory_tier_decision=memory_tier_decision.as_dict(),
                restored=False,
                mode="degraded",
                replay_tape_path=self.event_store.export_replay_tape(request_id)[
                    "path"
                ],
            )
            self.event_store.record_qsg_replay_event(
                request_id=request_id,
                stage="missing_tensor",
                payload={
                    "memory_id": memory_id,
                    "mission_replay_descriptor": descriptor.as_dict(),
                },
                source="QSGLatentBridge.replay",
            )
            return {
                "restored": False,
                "mode": "degraded",
                "reason": "missing_tensor",
                "memory_tier_decision": memory_tier_decision.as_dict(),
                "mission_replay_descriptor": descriptor.as_dict(),
            }
        request_id = engine.restore_latent_state(
            {
                "request_id": request_id,
                "prompt": package.get("summary_text") or "latent replay",
                "options": {"replay_memory_id": memory_id},
                "tensor": tensor,
                "generated_tokens": 0,
                "phase_state": 0.0,
                "latent_packet": {
                    "abi_version": int(
                        (package.get("compatibility_json") or {}).get(
                            "latent_packet_abi_version",
                            2,
                        )
                    ),
                    "tensor": tensor,
                    "tensor_format": str(
                        (package.get("compatibility_json") or {}).get(
                            "tensor_format",
                            package.get("tensor_format") or "float32",
                        )
                    ),
                    "tensor_codec": str(
                        (package.get("compatibility_json") or {}).get(
                            "tensor_codec",
                            "float32",
                        )
                    ),
                    "generated_tokens": 0,
                    "phase_state": 0.0,
                    "hidden_dim": hidden,
                    "capability_digest": str(
                        (package.get("compatibility_json") or {}).get(
                            "capability_digest",
                            "",
                        )
                    ),
                    "delta_watermark": dict(
                        (package.get("compatibility_json") or {}).get(
                            "delta_watermark",
                            {},
                        )
                    ),
                    "execution_capsule_id": str(
                        (package.get("compatibility_json") or {}).get(
                            "execution_capsule_id",
                            "",
                        )
                    ),
                    "segments": list(
                        (package.get("compatibility_json") or {}).get("segments") or []
                    ),
                    "restore_segment_kinds": ["branch_state", "summary_state"],
                },
                "execution_capsule": {
                    "capsule_id": str(
                        (package.get("compatibility_json") or {}).get(
                            "execution_capsule_id",
                            "",
                        )
                    ),
                    "version": int(
                        (package.get("compatibility_json") or {}).get(
                            "execution_capsule_version",
                            2,
                        )
                    ),
                    "segment_index": list(
                        (package.get("compatibility_json") or {}).get("segments") or []
                    ),
                    "segment_count": int(
                        (package.get("compatibility_json") or {}).get("segment_count")
                        or 0
                    ),
                    "segment_kinds": [
                        str(item.get("segment_kind") or "")
                        for item in list(
                            (package.get("compatibility_json") or {}).get("segments")
                            or []
                        )
                    ],
                    "delta_watermark": dict(
                        (package.get("compatibility_json") or {}).get(
                            "delta_watermark",
                            {},
                        )
                    ),
                },
            },
            target_request_id=target_request_id,
        )
        self.event_store.record_qsg_replay_event(
            request_id=request_id,
            stage="restored",
            payload={
                "memory_id": memory_id,
                "latent_package_id": str(package.get("latent_package_id") or ""),
                "memory_tier_decision": memory_tier_decision.as_dict(),
            },
            source="QSGLatentBridge.replay",
        )
        replay_tape = self.event_store.export_replay_tape(request_id)
        descriptor = self._mission_replay_descriptor(
            request_id=request_id,
            memory_id=memory_id,
            package=package,
            memory_tier_decision=memory_tier_decision.as_dict(),
            restored=True,
            mode="exact",
            replay_tape_path=str(replay_tape.get("path") or ""),
        )
        return {
            "restored": True,
            "mode": "exact",
            "request_id": request_id,
            "latent_package_id": package["latent_package_id"],
            "memory_tier_decision": memory_tier_decision.as_dict(),
            "mission_replay_descriptor": descriptor.as_dict(),
        }

    def _persist_repo_delta_memory(
        self,
        *,
        memory_id: str,
        repo_delta_memory: RepoDeltaMemoryRecord,
        source_system: str,
    ) -> str:
        record = repo_delta_memory.as_dict()
        if not record["delta_id"] and not record["changed_paths"]:
            return ""
        parent_memory = self.store.get_memory(memory_id) or {}
        created = self.store.create_memory(
            memory_kind="repo_delta_memory",
            payload_json=record,
            campaign_id=str(parent_memory.get("campaign_id") or "global"),
            workspace_id=str(parent_memory.get("workspace_id") or ""),
            repo_context=str(parent_memory.get("repo_context") or ""),
            session_id=str(parent_memory.get("session_id") or ""),
            source_system=source_system,
            summary_text=str(record["semantic_impact_hint"]),
            retention_class=RetentionPolicy.default_for_kind("repo_delta_memory"),
            importance_score=0.7,
            confidence_score=0.8,
            lifecycle_state="captured",
        )
        return str(created.memory_id)

    def _mission_replay_descriptor(
        self,
        *,
        request_id: str,
        memory_id: str,
        package: Dict[str, Any],
        memory_tier_decision: dict[str, Any],
        restored: bool,
        mode: str,
        replay_tape_path: str,
    ) -> MissionReplayDescriptor:
        compatibility = dict(package.get("compatibility_json") or {})
        return MissionReplayDescriptor(
            request_id=request_id,
            memory_id=memory_id,
            latent_package_id=str(package.get("latent_package_id") or ""),
            capsule_id=str(compatibility.get("execution_capsule_id") or ""),
            capability_digest=str(compatibility.get("capability_digest") or ""),
            delta_watermark=dict(compatibility.get("delta_watermark") or {}),
            supporting_memory_ids=list(package.get("supporting_memory_ids_json") or []),
            memory_tier_decision=dict(memory_tier_decision or {}),
            replay_tape_path=str(replay_tape_path),
            replay_run_id=f"qsg-replay:{request_id}",
            mode=str(mode),
            restored=bool(restored),
        )
