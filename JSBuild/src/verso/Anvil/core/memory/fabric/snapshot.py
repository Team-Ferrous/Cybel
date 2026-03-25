"""Snapshot and restore support for ALMF."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, Iterable, List

import numpy as np

from core.memory.fabric.models import LatentPackageRecord, MemoryObject, stable_json_dumps
from core.memory.fabric.store import MemoryFabricStore
from saguaro.storage.atomic_fs import atomic_write_json


class MemoryFabricSnapshotter:
    """Create campaign-scoped ALMF snapshots and restore them."""

    SCHEMA_VERSION = "almf.snapshot.v1"

    def __init__(self, store: MemoryFabricStore) -> None:
        self.store = store

    def snapshot_campaign(
        self,
        campaign_id: str,
        *,
        snapshot_id: str | None = None,
        include_latent_blobs: bool = True,
        include_indexes: bool = True,
    ) -> Dict[str, Any]:
        created_at = time.time()
        snapshot_name = snapshot_id or f"{campaign_id}_{int(created_at)}"
        snapshot_dir = Path(self.store.backend_profile.snapshot_root) / snapshot_name
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        memories = self.store.list_memories(campaign_id)
        memory_ids = {str(item.get("memory_id") or "") for item in memories}
        aliases = self.store.state_store.fetchall(
            """
            SELECT memory_id, source_table, source_id, campaign_id, created_at
            FROM memory_aliases
            WHERE campaign_id = ?
            ORDER BY created_at ASC
            """,
            (campaign_id,),
        )
        edges = self.store.state_store.fetchall(
            """
            SELECT *
            FROM memory_edges
            WHERE src_memory_id IN ({placeholders})
               OR dst_memory_id IN ({placeholders})
            ORDER BY recorded_at ASC
            """.format(placeholders=",".join("?" for _ in memory_ids or [""])),
            tuple(memory_ids) * 2,
        ) if memory_ids else []
        reads = self.store.list_reads(campaign_id)
        read_ids = [str(row.get("read_id") or "") for row in reads]
        feedback = self.store.list_feedback(read_ids=read_ids)

        embeddings = self.store.list_embedding_rows(memory_ids=memory_ids)
        multivectors = self.store.list_multivector_rows(memory_ids=memory_ids)
        hd_bundles = self.store.list_hd_bundle_rows(memory_ids=memory_ids)
        latent_packages = self.store.list_latent_packages(campaign_id=campaign_id)

        files = {
            "memory_objects": "memory_objects.json",
            "memory_aliases": "memory_aliases.json",
            "memory_edges": "memory_edges.json",
            "memory_reads": "memory_reads.json",
            "memory_feedback": "memory_feedback.json",
            "memory_embeddings": "memory_embeddings.json",
            "memory_multivectors": "memory_multivectors.json",
            "memory_hd_bundles": "memory_hd_bundles.json",
            "latent_packages": "latent_packages.json",
        }
        atomic_write_json(str(snapshot_dir / files["memory_objects"]), memories)
        atomic_write_json(str(snapshot_dir / files["memory_aliases"]), aliases)
        atomic_write_json(str(snapshot_dir / files["memory_edges"]), edges)
        atomic_write_json(str(snapshot_dir / files["memory_reads"]), reads)
        atomic_write_json(str(snapshot_dir / files["memory_feedback"]), feedback)

        copied_embeddings = self._copy_artifacts(
            rows=embeddings,
            field_name="vector_uri",
            snapshot_dir=snapshot_dir / "indexes" / "embeddings",
            include=include_indexes,
        )
        copied_multivectors = self._copy_artifacts(
            rows=multivectors,
            field_name="vector_uri",
            snapshot_dir=snapshot_dir / "indexes" / "multivectors",
            include=include_indexes,
        )
        copied_hd_bundles = self._copy_artifacts(
            rows=hd_bundles,
            field_name="bundle_uri",
            snapshot_dir=snapshot_dir / "indexes" / "hd_bundles",
            include=include_indexes,
        )
        copied_latent = self._copy_artifacts(
            rows=latent_packages,
            field_name="tensor_uri",
            snapshot_dir=snapshot_dir / "blobs" / "latent_packages",
            include=include_latent_blobs,
        )
        atomic_write_json(str(snapshot_dir / files["memory_embeddings"]), copied_embeddings)
        atomic_write_json(str(snapshot_dir / files["memory_multivectors"]), copied_multivectors)
        atomic_write_json(str(snapshot_dir / files["memory_hd_bundles"]), copied_hd_bundles)
        atomic_write_json(str(snapshot_dir / files["latent_packages"]), copied_latent)

        manifest = {
            "schema_version": self.SCHEMA_VERSION,
            "snapshot_id": snapshot_name,
            "campaign_id": campaign_id,
            "created_at": created_at,
            "backend_profile": self.store.backend_profile.as_dict(),
            "memory_count": len(memories),
            "latent_package_count": len(latent_packages),
            "index_artifact_count": len(embeddings) + len(multivectors) + len(hd_bundles),
            "includes": {
                "canonical_metadata": True,
                "latent_blobs": bool(include_latent_blobs),
                "retrieval_indexes": bool(include_indexes),
                "campaign_scoped_restore": True,
            },
            "files": files,
            "checksum": hashlib.sha256(
                stable_json_dumps(
                    {
                        "campaign_id": campaign_id,
                        "memory_ids": sorted(memory_ids),
                        "snapshot_id": snapshot_name,
                    }
                ).encode("utf-8")
            ).hexdigest(),
        }
        atomic_write_json(str(snapshot_dir / "manifest.json"), manifest)
        return {
            "snapshot_id": snapshot_name,
            "snapshot_dir": str(snapshot_dir),
            "manifest": manifest,
        }

    def restore_campaign(
        self,
        snapshot_dir: str,
        *,
        target_campaign_id: str | None = None,
        target_workspace_id: str | None = None,
        restore_latent_blobs: bool = True,
        restore_indexes: bool = True,
    ) -> Dict[str, Any]:
        root = Path(snapshot_dir)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        files = dict(manifest.get("files") or {})
        target_campaign = str(target_campaign_id or manifest.get("campaign_id") or "")

        memories = self._load_json(root / files["memory_objects"])
        aliases = self._load_json(root / files["memory_aliases"])
        edges = self._load_json(root / files["memory_edges"])
        reads = self._load_json(root / files["memory_reads"])
        feedback = self._load_json(root / files["memory_feedback"])
        embedding_rows = self._load_json(root / files["memory_embeddings"])
        multivector_rows = self._load_json(root / files["memory_multivectors"])
        hd_bundle_rows = self._load_json(root / files["memory_hd_bundles"])
        latent_rows = self._load_json(root / files["latent_packages"])

        restored_memory_ids: List[str] = []
        for row in memories:
            payload = dict(row)
            payload["campaign_id"] = target_campaign
            if target_workspace_id is not None:
                payload["workspace_id"] = target_workspace_id
            memory = MemoryObject(
                memory_id=str(payload["memory_id"]),
                memory_kind=str(payload["memory_kind"]),
                campaign_id=str(payload["campaign_id"]),
                workspace_id=str(payload.get("workspace_id") or ""),
                repo_context=str(payload.get("repo_context") or ""),
                session_id=str(payload.get("session_id") or ""),
                source_system=str(payload.get("source_system") or ""),
                created_at=float(payload.get("created_at") or time.time()),
                observed_at=float(payload.get("observed_at") or payload.get("created_at") or time.time()),
                updated_at=float(payload.get("updated_at") or payload.get("created_at") or time.time()),
                schema_version=str(payload.get("schema_version") or "almf.v1"),
                canonical_hash=str(payload.get("canonical_hash") or ""),
                payload_json=dict(payload.get("payload_json") or {}),
                summary_text=str(payload.get("summary_text") or ""),
                provenance_json=dict(payload.get("provenance_json") or {}),
                retention_class=str(payload.get("retention_class") or "durable"),
                importance_score=float(payload.get("importance_score") or 0.5),
                confidence_score=float(payload.get("confidence_score") or 0.5),
                sensitivity_class=str(payload.get("sensitivity_class") or "internal"),
                lifecycle_state=str(payload.get("lifecycle_state") or "active"),
                hypothesis_id=str(payload.get("hypothesis_id") or ""),
                document_id=str(payload.get("document_id") or ""),
                claim_id=str(payload.get("claim_id") or ""),
                experiment_id=str(payload.get("experiment_id") or ""),
                artifact_id=str(payload.get("artifact_id") or ""),
                task_packet_id=str(payload.get("task_packet_id") or ""),
                lane_id=str(payload.get("lane_id") or ""),
                operator_id=str(payload.get("operator_id") or ""),
                thread_id=str(payload.get("thread_id") or ""),
                model_family=str(payload.get("model_family") or ""),
                model_revision=str(payload.get("model_revision") or ""),
            ).finalize()
            self.store.upsert_memory(memory)
            restored_memory_ids.append(memory.memory_id)

        for row in aliases:
            self.store.register_alias(
                str(row["memory_id"]),
                str(row["source_table"]),
                str(row["source_id"]),
                campaign_id=target_campaign,
            )

        for row in edges:
            if (
                str(row.get("src_memory_id") or "") in restored_memory_ids
                and str(row.get("dst_memory_id") or "") in restored_memory_ids
            ):
                self.store.state_store.execute(
                    """
                    INSERT OR REPLACE INTO memory_edges (
                        edge_id, src_memory_id, dst_memory_id, edge_type, weight,
                        valid_from, valid_to, recorded_at, evidence_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["edge_id"],
                        row["src_memory_id"],
                        row["dst_memory_id"],
                        row["edge_type"],
                        float(row.get("weight") or 1.0),
                        float(row.get("valid_from") or time.time()),
                        row.get("valid_to"),
                        float(row.get("recorded_at") or time.time()),
                        json.dumps(row.get("evidence_json") or {}, default=str),
                    ),
                )

        for row in reads:
            self.store.state_store.execute(
                """
                INSERT OR REPLACE INTO memory_reads (
                    read_id, campaign_id, query_kind, query_text, planner_mode,
                    result_memory_ids_json, latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["read_id"],
                    target_campaign,
                    row["query_kind"],
                    row["query_text"],
                    row["planner_mode"],
                    json.dumps(row.get("result_memory_ids_json") or [], default=str),
                    float(row.get("latency_ms") or 0.0),
                    float(row.get("created_at") or time.time()),
                ),
            )

        for row in feedback:
            self.store.state_store.execute(
                """
                INSERT OR REPLACE INTO memory_feedback (
                    feedback_id, read_id, consumer_system, usefulness_score, grounding_score,
                    citation_score, token_savings_estimate, outcome_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["feedback_id"],
                    row["read_id"],
                    row["consumer_system"],
                    float(row.get("usefulness_score") or 0.0),
                    float(row.get("grounding_score") or 0.0),
                    float(row.get("citation_score") or 0.0),
                    float(row.get("token_savings_estimate") or 0.0),
                    json.dumps(row.get("outcome_json") or {}, default=str),
                    float(row.get("created_at") or time.time()),
                ),
            )

        if restore_indexes:
            self._restore_embeddings(embedding_rows, root)
            self._restore_multivectors(multivector_rows, root)
            self._restore_hd_bundles(hd_bundle_rows, root)
        if restore_latent_blobs:
            self._restore_latent_packages(latent_rows, root)

        return {
            "snapshot_id": str(manifest.get("snapshot_id") or ""),
            "target_campaign_id": target_campaign,
            "restored_memories": len(restored_memory_ids),
            "restored_aliases": len(aliases),
            "restored_edges": len(edges),
            "restored_reads": len(reads),
            "restored_feedback": len(feedback),
            "restored_indexes": bool(restore_indexes),
            "restored_latent_blobs": bool(restore_latent_blobs),
        }

    def _copy_artifacts(
        self,
        *,
        rows: Iterable[Dict[str, Any]],
        field_name: str,
        snapshot_dir: Path,
        include: bool,
    ) -> List[Dict[str, Any]]:
        copied_rows: List[Dict[str, Any]] = []
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            copied = dict(row)
            source = str(copied.get(field_name) or "")
            if include and source and os.path.exists(source):
                target = snapshot_dir / os.path.basename(source)
                shutil.copy2(source, target)
                copied[field_name] = str(target.relative_to(snapshot_dir.parent.parent))
            else:
                copied[field_name] = ""
            copied_rows.append(copied)
        return copied_rows

    def _restore_embeddings(self, rows: List[Dict[str, Any]], root: Path) -> None:
        for row in rows:
            rel = str(row.get("vector_uri") or "")
            if not rel:
                continue
            array = np.load(root / rel)
            self.store.put_embedding(
                str(row["memory_id"]),
                embedding_family=str(row["embedding_family"]),
                embedding_version=str(row["embedding_version"]),
                vector=array,
            )

    def _restore_multivectors(self, rows: List[Dict[str, Any]], root: Path) -> None:
        for row in rows:
            rel = str(row.get("vector_uri") or "")
            if not rel:
                continue
            array = np.load(root / rel)
            self.store.put_multivector(
                str(row["memory_id"]),
                embedding_family=str(row["embedding_family"]),
                vectors=array,
                indexing_mode=str(row.get("indexing_mode") or "token"),
            )

    def _restore_hd_bundles(self, rows: List[Dict[str, Any]], root: Path) -> None:
        for row in rows:
            rel = str(row.get("bundle_uri") or "")
            if not rel:
                continue
            array = np.load(root / rel)
            self.store.put_hd_bundle(
                str(row["memory_id"]),
                bundle_family=str(row["bundle_family"]),
                bundle_version=str(row["bundle_version"]),
                bundle=array,
            )

    def _restore_latent_packages(self, rows: List[Dict[str, Any]], root: Path) -> None:
        for row in rows:
            package = LatentPackageRecord(
                latent_package_id=str(row["latent_package_id"]),
                memory_id=str(row["memory_id"]),
                branch_id=str(row.get("branch_id") or ""),
                model_family=str(row.get("model_family") or ""),
                model_revision=str(row.get("model_revision") or ""),
                tokenizer_hash=str(row.get("tokenizer_hash") or ""),
                adapter_hash=str(row.get("adapter_hash") or ""),
                prompt_protocol_hash=str(row.get("prompt_protocol_hash") or ""),
                hidden_dim=int(row.get("hidden_dim") or 0),
                qsg_runtime_version=str(row.get("qsg_runtime_version") or ""),
                rope_config_hash=str(row.get("rope_config_hash") or ""),
                quantization_profile=str(row.get("quantization_profile") or ""),
                capture_stage=str(row.get("capture_stage") or ""),
                tensor_format=str(row.get("tensor_format") or "safetensors"),
                tensor_uri="",
                summary_text=str(row.get("summary_text") or ""),
                compatibility_json=dict(row.get("compatibility_json") or {}),
                supporting_memory_ids=list(
                    row.get("supporting_memory_ids_json")
                    or row.get("supporting_memory_ids")
                    or []
                ),
                creation_reason=str(row.get("creation_reason") or ""),
                created_at=float(row.get("created_at") or time.time()),
                expires_at=row.get("expires_at"),
            )
            tensor = None
            rel = str(row.get("tensor_uri") or "")
            if rel:
                tensor = self._load_snapshot_tensor(root / rel, package.tensor_format)
            self.store.put_latent_package(package, tensor=tensor)

    @staticmethod
    def _load_snapshot_tensor(path: Path, tensor_format: str) -> np.ndarray | None:
        if not path.exists():
            return None
        if tensor_format == "safetensors" or path.suffix == ".safetensors":
            from safetensors.numpy import load_file

            payload = load_file(str(path))
            return np.asarray(payload.get("latent"))
        return np.load(path)

    @staticmethod
    def _load_json(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        return list(json.loads(path.read_text(encoding="utf-8")) or [])
