"""Persistence helpers for the Anvil Latent Memory Fabric."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from safetensors.numpy import load_file, save_file

from core.campaign.state_store import CampaignStateStore
from core.memory.fabric.backends import (
    MemoryBackendProfile,
    resolve_memory_backend_profile,
)
from core.memory.fabric.models import (
    LatentPackageRecord,
    MemoryEdge,
    MemoryFeedbackRecord,
    MemoryObject,
    MemoryReadRecord,
    build_memory_object,
)
from core.native.qsg_state_kernels_wrapper import qsg_latent_decode_f16
from core.native.qsg_state_kernels_wrapper import qsg_latent_encode_f16


class MemoryFabricStore:
    """Typed storage facade over the campaign state store."""

    def __init__(
        self,
        state_store: CampaignStateStore,
        *,
        storage_root: str | None = None,
        backend_profile: MemoryBackendProfile | None = None,
    ) -> None:
        self.state_store = state_store
        self.backend_profile = backend_profile or resolve_memory_backend_profile(
            db_path=self.state_store.db_path,
            storage_root=storage_root,
        )
        self.storage_root = Path(self.backend_profile.storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_db_path(
        cls,
        db_path: str,
        *,
        storage_root: str | None = None,
        requested_backend: str = "auto",
        postgres_dsn: str | None = None,
        tenant_key: str = "default",
    ) -> "MemoryFabricStore":
        profile = resolve_memory_backend_profile(
            db_path=db_path,
            storage_root=storage_root,
            requested_backend=requested_backend,
            postgres_dsn=postgres_dsn,
            tenant_key=tenant_key,
        )
        return cls(
            CampaignStateStore(profile.canonical_dsn),
            storage_root=storage_root,
            backend_profile=profile,
        )

    @classmethod
    def from_profile(cls, profile: MemoryBackendProfile) -> "MemoryFabricStore":
        if profile.effective_backend != "sqlite":
            raise RuntimeError(
                "The current ALMF runtime only supports direct sqlite connections; "
                "enterprise profiles fall back to sqlite when postgres is unavailable."
            )
        return cls(
            CampaignStateStore(profile.canonical_dsn),
            storage_root=profile.storage_root,
            backend_profile=profile,
        )

    def create_memory(
        self,
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
        memory = build_memory_object(
            memory_kind,
            payload_json,
            campaign_id=campaign_id,
            workspace_id=workspace_id,
            repo_context=repo_context,
            session_id=session_id,
            source_system=source_system,
            summary_text=summary_text,
            provenance_json=provenance_json or {},
            retention_class=retention_class,
            importance_score=importance_score,
            confidence_score=confidence_score,
            sensitivity_class=sensitivity_class,
            lifecycle_state=lifecycle_state,
            observed_at=observed_at,
            **extra_ids,
        )
        self.upsert_memory(memory)
        return memory

    def upsert_memory(self, memory: MemoryObject) -> None:
        memory.finalize()
        self.state_store.execute(
            """
            INSERT INTO memory_objects (
                memory_id, campaign_id, memory_kind, workspace_id, repo_context, session_id,
                source_system, summary_text, payload_json, provenance_json, canonical_hash,
                importance_score, confidence_score, retention_class, sensitivity_class,
                lifecycle_state, schema_version, created_at, observed_at, updated_at,
                hypothesis_id, document_id, claim_id, experiment_id, artifact_id,
                task_packet_id, lane_id, operator_id, thread_id, model_family, model_revision
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                summary_text = excluded.summary_text,
                payload_json = excluded.payload_json,
                provenance_json = excluded.provenance_json,
                canonical_hash = excluded.canonical_hash,
                importance_score = excluded.importance_score,
                confidence_score = excluded.confidence_score,
                retention_class = excluded.retention_class,
                sensitivity_class = excluded.sensitivity_class,
                lifecycle_state = excluded.lifecycle_state,
                updated_at = excluded.updated_at,
                hypothesis_id = excluded.hypothesis_id,
                document_id = excluded.document_id,
                claim_id = excluded.claim_id,
                experiment_id = excluded.experiment_id,
                artifact_id = excluded.artifact_id,
                task_packet_id = excluded.task_packet_id,
                lane_id = excluded.lane_id,
                operator_id = excluded.operator_id,
                thread_id = excluded.thread_id,
                model_family = excluded.model_family,
                model_revision = excluded.model_revision
            """,
            (
                memory.memory_id,
                memory.campaign_id,
                memory.memory_kind,
                memory.workspace_id,
                memory.repo_context,
                memory.session_id,
                memory.source_system,
                memory.summary_text,
                json.dumps(memory.payload_json, default=str),
                json.dumps(memory.provenance_json, default=str),
                memory.canonical_hash,
                float(memory.importance_score),
                float(memory.confidence_score),
                memory.retention_class,
                memory.sensitivity_class,
                memory.lifecycle_state,
                memory.schema_version,
                float(memory.created_at),
                float(memory.observed_at),
                float(memory.updated_at),
                memory.hypothesis_id,
                memory.document_id,
                memory.claim_id,
                memory.experiment_id,
                memory.artifact_id,
                memory.task_packet_id,
                memory.lane_id,
                memory.operator_id,
                memory.thread_id,
                memory.model_family,
                memory.model_revision,
            ),
        )

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        row = self.state_store.fetchone(
            "SELECT * FROM memory_objects WHERE memory_id = ?",
            (memory_id,),
        )
        return self._decode_memory_row(row)

    def list_memories(
        self,
        campaign_id: str,
        *,
        memory_kinds: Optional[Iterable[str]] = None,
        repo_context: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM memory_objects WHERE campaign_id = ?"
        params: list[Any] = [campaign_id]
        if memory_kinds:
            kinds = list(memory_kinds)
            query += " AND memory_kind IN (%s)" % ",".join("?" for _ in kinds)
            params.extend(kinds)
        if repo_context is not None:
            query += " AND repo_context = ?"
            params.append(repo_context)
        if session_id is not None:
            query += " AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY observed_at DESC, created_at DESC"
        rows = self.state_store.fetchall(query, tuple(params))
        return [
            decoded
            for decoded in (self._decode_memory_row(row) for row in rows)
            if decoded
        ]

    def update_memory(self, memory: MemoryObject | Dict[str, Any]) -> None:
        if isinstance(memory, MemoryObject):
            payload = memory
        else:
            payload = build_memory_object(
                str(memory.get("memory_kind") or "unknown"),
                dict(memory.get("payload_json") or {}),
                campaign_id=str(memory.get("campaign_id") or "global"),
                workspace_id=str(memory.get("workspace_id") or ""),
                repo_context=str(memory.get("repo_context") or ""),
                session_id=str(memory.get("session_id") or ""),
                source_system=str(memory.get("source_system") or ""),
                summary_text=str(memory.get("summary_text") or ""),
                provenance_json=dict(memory.get("provenance_json") or {}),
                retention_class=str(memory.get("retention_class") or "durable"),
                importance_score=float(memory.get("importance_score") or 0.5),
                confidence_score=float(memory.get("confidence_score") or 0.5),
                sensitivity_class=str(memory.get("sensitivity_class") or "internal"),
                lifecycle_state=str(memory.get("lifecycle_state") or "active"),
                observed_at=float(
                    memory.get("observed_at") or memory.get("created_at") or time.time()
                ),
                hypothesis_id=str(memory.get("hypothesis_id") or ""),
                document_id=str(memory.get("document_id") or ""),
                claim_id=str(memory.get("claim_id") or ""),
                experiment_id=str(memory.get("experiment_id") or ""),
                artifact_id=str(memory.get("artifact_id") or ""),
                task_packet_id=str(memory.get("task_packet_id") or ""),
                lane_id=str(memory.get("lane_id") or ""),
                operator_id=str(memory.get("operator_id") or ""),
                thread_id=str(memory.get("thread_id") or ""),
                model_family=str(memory.get("model_family") or ""),
                model_revision=str(memory.get("model_revision") or ""),
            )
            payload.memory_id = str(memory.get("memory_id") or payload.memory_id)
            payload.created_at = float(memory.get("created_at") or payload.created_at)
            payload.updated_at = float(memory.get("updated_at") or time.time())
            payload.canonical_hash = str(
                memory.get("canonical_hash") or payload.canonical_hash
            )
        self.upsert_memory(payload)

    def search_text(
        self,
        campaign_id: str,
        query_text: str,
        *,
        memory_kinds: Optional[Iterable[str]] = None,
        limit: int = 25,
    ) -> List[Dict[str, Any]]:
        query = """
            SELECT * FROM memory_objects
            WHERE campaign_id = ? AND (summary_text LIKE ? OR payload_json LIKE ?)
        """
        like_query = f"%{query_text}%"
        params: list[Any] = [campaign_id, like_query, like_query]
        if memory_kinds:
            kinds = list(memory_kinds)
            query += " AND memory_kind IN (%s)" % ",".join("?" for _ in kinds)
            params.extend(kinds)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(int(limit))
        rows = self.state_store.fetchall(query, tuple(params))
        return [
            decoded
            for decoded in (self._decode_memory_row(row) for row in rows)
            if decoded
        ]

    def register_alias(
        self,
        memory_id: str,
        source_table: str,
        source_id: str,
        *,
        campaign_id: str,
    ) -> None:
        self.state_store.execute(
            """
            INSERT INTO memory_aliases (
                memory_id, source_table, source_id, campaign_id, created_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(campaign_id, source_table, source_id) DO UPDATE SET
                memory_id = excluded.memory_id
            """,
            (memory_id, source_table, source_id, campaign_id, float(time.time())),
        )

    def resolve_alias(
        self,
        *,
        campaign_id: str,
        source_table: str,
        source_id: str,
    ) -> Optional[str]:
        row = self.state_store.fetchone(
            """
            SELECT memory_id
            FROM memory_aliases
            WHERE campaign_id = ? AND source_table = ? AND source_id = ?
            LIMIT 1
            """,
            (campaign_id, source_table, source_id),
        )
        if row is None:
            return None
        return str(row["memory_id"])

    def add_edge(self, edge: MemoryEdge) -> None:
        self.state_store.execute(
            """
            INSERT INTO memory_edges (
                edge_id, src_memory_id, dst_memory_id, edge_type, weight,
                valid_from, valid_to, recorded_at, evidence_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(edge_id) DO UPDATE SET
                weight = excluded.weight,
                valid_from = excluded.valid_from,
                valid_to = excluded.valid_to,
                recorded_at = excluded.recorded_at,
                evidence_json = excluded.evidence_json
            """,
            (
                edge.edge_id,
                edge.src_memory_id,
                edge.dst_memory_id,
                edge.edge_type,
                float(edge.weight),
                float(edge.valid_from),
                None if edge.valid_to is None else float(edge.valid_to),
                float(edge.recorded_at),
                json.dumps(edge.evidence_json, default=str),
            ),
        )

    def list_edges(
        self,
        *,
        src_memory_id: Optional[str] = None,
        dst_memory_id: Optional[str] = None,
        edge_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM memory_edges WHERE 1 = 1"
        params: list[Any] = []
        if src_memory_id is not None:
            query += " AND src_memory_id = ?"
            params.append(src_memory_id)
        if dst_memory_id is not None:
            query += " AND dst_memory_id = ?"
            params.append(dst_memory_id)
        if edge_type is not None:
            query += " AND edge_type = ?"
            params.append(edge_type)
        query += " ORDER BY recorded_at ASC"
        rows = self.state_store.fetchall(query, tuple(params))
        return [
            {
                **row,
                "evidence_json": json.loads(row.get("evidence_json") or "{}"),
            }
            for row in rows
        ]

    def put_embedding(
        self,
        memory_id: str,
        *,
        embedding_family: str,
        embedding_version: str,
        vector: np.ndarray,
    ) -> str:
        path = self._vector_path(
            "embeddings", memory_id, embedding_family, embedding_version
        )
        array = np.asarray(vector, dtype=np.float32)
        np.save(path, array)
        norm = float(np.linalg.norm(array))
        self.state_store.execute(
            """
            INSERT INTO memory_embeddings (
                memory_id, embedding_family, embedding_version, dim, vector_uri, norm, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id, embedding_family, embedding_version) DO UPDATE SET
                dim = excluded.dim,
                vector_uri = excluded.vector_uri,
                norm = excluded.norm,
                created_at = excluded.created_at
            """,
            (
                memory_id,
                embedding_family,
                embedding_version,
                int(array.shape[-1]),
                str(path),
                norm,
                float(time.time()),
            ),
        )
        return str(path)

    def get_embedding(
        self,
        memory_id: str,
        *,
        embedding_family: str = "almf-dense",
    ) -> Optional[np.ndarray]:
        row = self.state_store.fetchone(
            """
            SELECT vector_uri
            FROM memory_embeddings
            WHERE memory_id = ? AND embedding_family = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (memory_id, embedding_family),
        )
        return self._load_vector(row)

    def put_multivector(
        self,
        memory_id: str,
        *,
        embedding_family: str,
        vectors: np.ndarray,
        indexing_mode: str = "token",
    ) -> str:
        path = self._vector_path("multivectors", memory_id, embedding_family, "mv")
        array = np.asarray(vectors, dtype=np.float32)
        np.save(path, array)
        self.state_store.execute(
            """
            INSERT INTO memory_multivectors (
                memory_id, embedding_family, token_count, vector_uri, indexing_mode, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id, embedding_family) DO UPDATE SET
                token_count = excluded.token_count,
                vector_uri = excluded.vector_uri,
                indexing_mode = excluded.indexing_mode,
                created_at = excluded.created_at
            """,
            (
                memory_id,
                embedding_family,
                int(array.shape[0]),
                str(path),
                indexing_mode,
                float(time.time()),
            ),
        )
        return str(path)

    def get_multivector(
        self,
        memory_id: str,
        *,
        embedding_family: str = "almf-token",
    ) -> Optional[np.ndarray]:
        row = self.state_store.fetchone(
            """
            SELECT vector_uri
            FROM memory_multivectors
            WHERE memory_id = ? AND embedding_family = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (memory_id, embedding_family),
        )
        return self._load_vector(row)

    def put_hd_bundle(
        self,
        memory_id: str,
        *,
        bundle_family: str,
        bundle_version: str,
        bundle: np.ndarray,
    ) -> str:
        path = self._vector_path("hd_bundles", memory_id, bundle_family, bundle_version)
        array = np.asarray(bundle)
        np.save(path, array)
        self.state_store.execute(
            """
            INSERT INTO memory_hd_bundles (
                memory_id, bundle_family, bundle_version, bundle_uri, created_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(memory_id, bundle_family, bundle_version) DO UPDATE SET
                bundle_uri = excluded.bundle_uri,
                created_at = excluded.created_at
            """,
            (
                memory_id,
                bundle_family,
                bundle_version,
                str(path),
                float(time.time()),
            ),
        )
        return str(path)

    def get_hd_bundle(
        self,
        memory_id: str,
        *,
        bundle_family: str = "almf-hd",
    ) -> Optional[np.ndarray]:
        row = self.state_store.fetchone(
            """
            SELECT bundle_uri
            FROM memory_hd_bundles
            WHERE memory_id = ? AND bundle_family = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (memory_id, bundle_family),
        )
        return self._load_vector(row, uri_key="bundle_uri")

    def put_latent_package(
        self,
        package: LatentPackageRecord,
        *,
        tensor: np.ndarray | None = None,
    ) -> str:
        package.compatibility_json = {
            **dict(package.compatibility_json or {}),
            "latent_packet_abi_version": int(package.latent_packet_abi_version),
            "execution_capsule_version": int(package.execution_capsule_version),
            "execution_capsule_id": str(package.execution_capsule_id),
            "capability_digest": str(package.capability_digest),
            "delta_watermark": dict(package.delta_watermark_json or {}),
        }
        if tensor is not None:
            tensor_array = np.asarray(tensor, dtype=np.float32)
            codec = (
                str(
                    dict(package.compatibility_json or {}).get("tensor_codec")
                    or package.quantization_profile
                    or "float32"
                )
                .strip()
                .lower()
            )
            package.tensor_format = (
                "native-f16" if codec == "float16" else "safetensors"
            )
            path = self._latent_path(
                package.memory_id,
                package.capture_stage or "capture",
                package.latent_package_id,
                tensor_format=package.tensor_format,
            )
            checksum = hashlib.sha256(tensor_array.tobytes()).hexdigest()
            package.compatibility_json = {
                **dict(package.compatibility_json or {}),
                "tensor_sha256": checksum,
                "tensor_shape": list(tensor_array.shape),
                "tensor_format": package.tensor_format,
                "tensor_codec": codec,
            }
            if codec == "float16":
                encoded = qsg_latent_encode_f16(tensor_array)
                np.save(
                    path,
                    {
                        "latent_f16": encoded,
                        "tensor_shape": np.asarray(tensor_array.shape, dtype=np.int32),
                    },
                    allow_pickle=True,
                )
            else:
                save_file(
                    {"latent": tensor_array},
                    str(path),
                    metadata={
                        "latent_package_id": package.latent_package_id,
                        "memory_id": package.memory_id,
                        "capture_stage": package.capture_stage,
                        "model_family": package.model_family,
                        "hidden_dim": str(
                            int(tensor_array.shape[-1]) if tensor_array.ndim else 1
                        ),
                        "checksum": checksum,
                    },
                )
            package.tensor_uri = str(path)
            if package.hidden_dim <= 0:
                package.hidden_dim = (
                    int(tensor_array.shape[-1]) if tensor_array.ndim else 1
                )
        self.state_store.execute(
            """
            INSERT INTO latent_packages (
                latent_package_id, memory_id, branch_id, model_family, model_revision,
                tokenizer_hash, adapter_hash, prompt_protocol_hash, hidden_dim,
                qsg_runtime_version, rope_config_hash, quantization_profile, capture_stage,
                tensor_format, tensor_uri, summary_text, compatibility_json, supporting_memory_ids_json,
                creation_reason, created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(latent_package_id) DO UPDATE SET
                tensor_uri = excluded.tensor_uri,
                summary_text = excluded.summary_text,
                compatibility_json = excluded.compatibility_json,
                supporting_memory_ids_json = excluded.supporting_memory_ids_json,
                creation_reason = excluded.creation_reason,
                expires_at = excluded.expires_at
            """,
            (
                package.latent_package_id,
                package.memory_id,
                package.branch_id,
                package.model_family,
                package.model_revision,
                package.tokenizer_hash,
                package.adapter_hash,
                package.prompt_protocol_hash,
                int(package.hidden_dim),
                package.qsg_runtime_version,
                package.rope_config_hash,
                package.quantization_profile,
                package.capture_stage,
                package.tensor_format,
                package.tensor_uri,
                package.summary_text,
                json.dumps(package.compatibility_json, default=str),
                json.dumps(package.supporting_memory_ids, default=str),
                package.creation_reason,
                float(package.created_at),
                None if package.expires_at is None else float(package.expires_at),
            ),
        )
        return package.tensor_uri

    def latest_latent_package(self, memory_id: str) -> Optional[Dict[str, Any]]:
        row = self.state_store.fetchone(
            """
            SELECT *
            FROM latent_packages
            WHERE memory_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (memory_id,),
        )
        if row is None:
            return None
        decoded = dict(row)
        decoded["compatibility_json"] = json.loads(
            decoded.get("compatibility_json") or "{}"
        )
        decoded["supporting_memory_ids_json"] = json.loads(
            decoded.get("supporting_memory_ids_json") or "[]"
        )
        return decoded

    def load_latent_tensor(self, package_row: Dict[str, Any]) -> Optional[np.ndarray]:
        tensor_uri = str(package_row.get("tensor_uri") or "")
        if not tensor_uri or not os.path.exists(tensor_uri):
            return None
        tensor_format = str(package_row.get("tensor_format") or "")
        if tensor_format == "native-f16" or tensor_uri.endswith(".npy"):
            payload = np.load(tensor_uri, allow_pickle=True).item()
            encoded = np.asarray(payload.get("latent_f16"), dtype=np.uint16)
            return qsg_latent_decode_f16(encoded)
        if tensor_format == "safetensors" or tensor_uri.endswith(".safetensors"):
            payload = load_file(tensor_uri)
            tensor = payload.get("latent")
            if tensor is None:
                return None
            return np.asarray(tensor)
        return np.load(tensor_uri)

    def list_latent_packages(
        self,
        *,
        memory_id: str | None = None,
        campaign_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        query = """
            SELECT latent_packages.*
            FROM latent_packages
            LEFT JOIN memory_objects ON memory_objects.memory_id = latent_packages.memory_id
            WHERE 1 = 1
        """
        params: list[Any] = []
        if memory_id is not None:
            query += " AND latent_packages.memory_id = ?"
            params.append(memory_id)
        if campaign_id is not None:
            query += " AND memory_objects.campaign_id = ?"
            params.append(campaign_id)
        query += " ORDER BY latent_packages.created_at DESC"
        rows = self.state_store.fetchall(query, tuple(params))
        decoded = []
        for row in rows:
            item = dict(row)
            item["compatibility_json"] = json.loads(
                item.get("compatibility_json") or "{}"
            )
            item["supporting_memory_ids_json"] = json.loads(
                item.get("supporting_memory_ids_json") or "[]"
            )
            decoded.append(item)
        return decoded

    def delete_latent_package(self, latent_package_id: str) -> None:
        self.state_store.execute(
            "DELETE FROM latent_packages WHERE latent_package_id = ?",
            (latent_package_id,),
        )

    def list_embedding_rows(
        self,
        *,
        memory_ids: Iterable[str],
    ) -> List[Dict[str, Any]]:
        memory_ids = [str(item) for item in memory_ids if item]
        if not memory_ids:
            return []
        rows = self.state_store.fetchall(
            "SELECT * FROM memory_embeddings WHERE memory_id IN (%s)"
            % ",".join("?" for _ in memory_ids),
            tuple(memory_ids),
        )
        return [dict(row) for row in rows]

    def list_multivector_rows(
        self,
        *,
        memory_ids: Iterable[str],
    ) -> List[Dict[str, Any]]:
        memory_ids = [str(item) for item in memory_ids if item]
        if not memory_ids:
            return []
        rows = self.state_store.fetchall(
            "SELECT * FROM memory_multivectors WHERE memory_id IN (%s)"
            % ",".join("?" for _ in memory_ids),
            tuple(memory_ids),
        )
        return [dict(row) for row in rows]

    def list_hd_bundle_rows(
        self,
        *,
        memory_ids: Iterable[str],
    ) -> List[Dict[str, Any]]:
        memory_ids = [str(item) for item in memory_ids if item]
        if not memory_ids:
            return []
        rows = self.state_store.fetchall(
            "SELECT * FROM memory_hd_bundles WHERE memory_id IN (%s)"
            % ",".join("?" for _ in memory_ids),
            tuple(memory_ids),
        )
        return [dict(row) for row in rows]

    def record_read(self, record: MemoryReadRecord) -> None:
        self.state_store.execute(
            """
            INSERT INTO memory_reads (
                read_id, campaign_id, query_kind, query_text, planner_mode,
                result_memory_ids_json, latency_ms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.read_id,
                record.campaign_id,
                record.query_kind,
                record.query_text,
                record.planner_mode,
                json.dumps(record.result_memory_ids_json, default=str),
                float(record.latency_ms),
                float(record.created_at),
            ),
        )

    def list_reads(self, campaign_id: str) -> List[Dict[str, Any]]:
        rows = self.state_store.fetchall(
            """
            SELECT *
            FROM memory_reads
            WHERE campaign_id = ?
            ORDER BY created_at ASC
            """,
            (campaign_id,),
        )
        decoded = []
        for row in rows:
            item = dict(row)
            item["result_memory_ids_json"] = json.loads(
                item.get("result_memory_ids_json") or "[]"
            )
            decoded.append(item)
        return decoded

    def record_feedback(self, feedback: MemoryFeedbackRecord) -> None:
        self.state_store.execute(
            """
            INSERT INTO memory_feedback (
                feedback_id, read_id, consumer_system, usefulness_score, grounding_score,
                citation_score, token_savings_estimate, outcome_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                feedback.feedback_id,
                feedback.read_id,
                feedback.consumer_system,
                float(feedback.usefulness_score),
                float(feedback.grounding_score),
                float(feedback.citation_score),
                float(feedback.token_savings_estimate),
                json.dumps(feedback.outcome_json, default=str),
                float(feedback.created_at),
            ),
        )

    def list_feedback(self, *, read_ids: Iterable[str]) -> List[Dict[str, Any]]:
        read_ids = [str(item) for item in read_ids if item]
        if not read_ids:
            return []
        rows = self.state_store.fetchall(
            "SELECT * FROM memory_feedback WHERE read_id IN (%s) ORDER BY created_at ASC"
            % ",".join("?" for _ in read_ids),
            tuple(read_ids),
        )
        decoded = []
        for row in rows:
            item = dict(row)
            item["outcome_json"] = json.loads(item.get("outcome_json") or "{}")
            decoded.append(item)
        return decoded

    def _vector_path(
        self,
        family_dir: str,
        memory_id: str,
        family: str,
        version: str,
    ) -> Path:
        root = Path(self.backend_profile.index_root)
        target_dir = root / family_dir / family
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / f"{memory_id}_{version}.npy"

    def _latent_path(
        self,
        memory_id: str,
        family: str,
        version: str,
        *,
        tensor_format: str = "safetensors",
    ) -> Path:
        target_dir = Path(self.backend_profile.blob_root) / "latent_packages" / family
        target_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".npy" if tensor_format == "native-f16" else ".safetensors"
        return target_dir / f"{memory_id}_{version}{suffix}"

    def _decode_memory_row(
        self, row: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        decoded = dict(row)
        decoded["payload_json"] = json.loads(decoded.get("payload_json") or "{}")
        decoded["provenance_json"] = json.loads(decoded.get("provenance_json") or "{}")
        return decoded

    @staticmethod
    def _load_vector(
        row: Optional[Dict[str, Any]],
        *,
        uri_key: str = "vector_uri",
    ) -> Optional[np.ndarray]:
        if row is None:
            return None
        vector_uri = str(row.get(uri_key) or "")
        if not vector_uri or not os.path.exists(vector_uri):
            return None
        return np.load(vector_uri)
