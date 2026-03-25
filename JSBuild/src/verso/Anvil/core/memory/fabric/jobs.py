"""Consolidation jobs for ALMF."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha1
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List

from core.memory.fabric.community_builder import MemoryCommunityBuilder
from core.memory.fabric.models import MemoryEdge
from core.memory.fabric.projectors import MemoryProjector
from core.memory.fabric.store import MemoryFabricStore
from core.memory.fabric.temporal_tree import MemoryTemporalTreeBuilder


class MemoryConsolidationJobs:
    """Background and milestone-oriented ALMF consolidation jobs."""

    def __init__(self, store: MemoryFabricStore, projector: MemoryProjector) -> None:
        self.store = store
        self.projector = projector
        self.temporal_builder = MemoryTemporalTreeBuilder()
        self.community_builder = MemoryCommunityBuilder()

    def sensory_filter(
        self,
        campaign_id: str,
        *,
        importance_threshold: float = 0.2,
    ) -> Dict[str, Any]:
        candidates = []
        for memory in self.store.list_memories(campaign_id):
            if (
                str(memory.get("retention_class") or "") == "ephemeral"
                and float(memory.get("importance_score") or 0.0) <= importance_threshold
                and str(memory.get("lifecycle_state") or "active") == "active"
            ):
                memory["lifecycle_state"] = "filtered"
                self.store.update_memory(memory)
                candidates.append(str(memory.get("memory_id") or ""))
        return {"job": "sensory_filter", "filtered_memory_ids": candidates, "count": len(candidates)}

    def session_summarizer(self, campaign_id: str) -> Dict[str, Any]:
        by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for memory in self.store.list_memories(campaign_id):
            session_id = str(memory.get("session_id") or "")
            if session_id:
                by_session[session_id].append(memory)
        created = 0
        summary_ids: list[str] = []
        for session_id, items in sorted(by_session.items()):
            if len(items) < 2:
                continue
            source_id = f"session::{session_id}"
            memory_id = self.store.resolve_alias(
                campaign_id=campaign_id,
                source_table="memory_session_summary",
                source_id=source_id,
            )
            summary_text = " | ".join(
                str(item.get("summary_text") or "")[:120]
                for item in items[:5]
                if item.get("summary_text")
            )
            payload = {
                "session_id": session_id,
                "memory_ids": [str(item.get("memory_id") or "") for item in items],
                "summary_items": [str(item.get("summary_text") or "") for item in items[:8]],
            }
            if memory_id:
                summary = self.store.get_memory(memory_id)
                if summary is None:
                    continue
                summary["payload_json"] = payload
                summary["summary_text"] = summary_text
                summary["updated_at"] = time.time()
                self.store.update_memory(summary)
            else:
                summary = self.store.create_memory(
                    memory_kind="session_summary",
                    payload_json=payload,
                    campaign_id=campaign_id,
                    workspace_id=campaign_id,
                    session_id=session_id,
                    source_system="almf.jobs.session_summarizer",
                    summary_text=summary_text,
                    retention_class="durable",
                    importance_score=0.6,
                )
                self.store.register_alias(
                    summary.memory_id,
                    "memory_session_summary",
                    source_id,
                    campaign_id=campaign_id,
                )
                created += 1
                memory_id = summary.memory_id
            summary_ids.append(str(memory_id))
            self.projector.project_memory(self.store, self.store.get_memory(str(memory_id)))
            for item in items[:8]:
                self.store.add_edge(
                    MemoryEdge(
                        src_memory_id=str(memory_id),
                        dst_memory_id=str(item.get("memory_id") or ""),
                        edge_type="summarizes",
                        evidence_json={"job": "session_summarizer", "session_id": session_id},
                    )
                )
        return {"job": "session_summarizer", "summary_ids": summary_ids, "created": created}

    def claim_consolidator(self, campaign_id: str) -> Dict[str, Any]:
        claims = self.store.list_memories(campaign_id, memory_kinds=["research_claim"])
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for claim in claims:
            key = " ".join(str(claim.get("summary_text") or "").lower().split())
            groups[key].append(claim)
        duplicates = 0
        contradictions = 0
        cluster_ids: list[str] = []
        for key, items in groups.items():
            if not key:
                continue
            if len(items) > 1:
                duplicates += len(items) - 1
                avg_confidence = sum(
                    float(item.get("confidence_score") or 0.5) for item in items
                ) / float(len(items))
                cluster = self.store.create_memory(
                    memory_kind="claim_cluster",
                    payload_json={
                        "normalized_summary": key,
                        "claim_ids": [str(item.get("memory_id") or "") for item in items],
                        "duplicate_count": len(items),
                        "confidence_score": avg_confidence,
                    },
                    campaign_id=campaign_id,
                    workspace_id=campaign_id,
                    source_system="almf.jobs.claim_consolidator",
                    summary_text=f"Duplicate claim cluster for {key[:120]}",
                    retention_class="durable",
                    importance_score=0.7,
                    confidence_score=avg_confidence,
                )
                cluster_ids.append(cluster.memory_id)
                self.projector.project_memory(self.store, cluster)
                for item in items:
                    item["confidence_score"] = avg_confidence
                    self.store.update_memory(item)
                    self.store.add_edge(
                        MemoryEdge(
                            src_memory_id=cluster.memory_id,
                            dst_memory_id=str(item.get("memory_id") or ""),
                            edge_type="duplicates",
                            evidence_json={"job": "claim_consolidator"},
                        )
                    )
        by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for claim in claims:
            topic = str(claim.get("payload_json", {}).get("topic") or "").strip().lower()
            if topic:
                by_topic[topic].append(claim)
        negation_tokens = {"not", "never", "fails", "failure", "without", "missing"}
        for items in by_topic.values():
            for index, left in enumerate(items):
                left_tokens = set(str(left.get("summary_text") or "").lower().split())
                for right in items[index + 1 :]:
                    right_tokens = set(str(right.get("summary_text") or "").lower().split())
                    if bool(left_tokens & negation_tokens) == bool(right_tokens & negation_tokens):
                        continue
                    overlap = len(left_tokens & right_tokens)
                    if overlap < 3:
                        continue
                    contradictions += 1
                    self.store.add_edge(
                        MemoryEdge(
                            src_memory_id=str(left.get("memory_id") or ""),
                            dst_memory_id=str(right.get("memory_id") or ""),
                            edge_type="contradicts",
                            evidence_json={"job": "claim_consolidator", "topic_overlap": overlap},
                        )
                    )
        return {
            "job": "claim_consolidator",
            "duplicate_count": duplicates,
            "contradiction_count": contradictions,
            "cluster_ids": cluster_ids,
        }

    def temporal_tree_builder(self, campaign_id: str) -> Dict[str, Any]:
        tree = self.temporal_builder.build(self.store.list_memories(campaign_id))
        source_id = f"campaign::{campaign_id}"
        memory_id = self.store.resolve_alias(
            campaign_id=campaign_id,
            source_table="memory_temporal_tree",
            source_id=source_id,
        )
        if memory_id:
            summary = self.store.get_memory(memory_id)
            if summary:
                summary["payload_json"] = tree
                summary["summary_text"] = f"Temporal tree for {campaign_id} over {tree['total_days']} days"
                summary["updated_at"] = time.time()
                self.store.update_memory(summary)
                return {"job": "temporal_tree_builder", "memory_id": memory_id, "tree": tree}
        memory = self.store.create_memory(
            memory_kind="temporal_summary",
            payload_json=tree,
            campaign_id=campaign_id,
            workspace_id=campaign_id,
            source_system="almf.jobs.temporal_tree_builder",
            summary_text=f"Temporal tree for {campaign_id} over {tree['total_days']} days",
            retention_class="durable",
            importance_score=0.7,
        )
        self.store.register_alias(
            memory.memory_id,
            "memory_temporal_tree",
            source_id,
            campaign_id=campaign_id,
        )
        self.projector.project_memory(self.store, memory)
        return {"job": "temporal_tree_builder", "memory_id": memory.memory_id, "tree": tree}

    def latent_expiry_sweeper(
        self,
        campaign_id: str,
        *,
        mode: str = "archive",
    ) -> Dict[str, Any]:
        packages = self.store.list_latent_packages(campaign_id=campaign_id)
        archived = 0
        deleted = 0
        now = time.time()
        archive_root = Path(self.store.backend_profile.snapshot_root) / "latent_archive" / campaign_id
        archive_root.mkdir(parents=True, exist_ok=True)
        for package in packages:
            expires_at = package.get("expires_at")
            if expires_at is None or float(expires_at) > now:
                continue
            tensor_uri = str(package.get("tensor_uri") or "")
            if mode == "delete":
                if tensor_uri and os.path.exists(tensor_uri):
                    os.remove(tensor_uri)
                self.store.delete_latent_package(str(package.get("latent_package_id") or ""))
                deleted += 1
                continue
            if tensor_uri and os.path.exists(tensor_uri):
                target = archive_root / os.path.basename(tensor_uri)
                shutil.move(tensor_uri, target)
                self.store.state_store.execute(
                    """
                    UPDATE latent_packages
                    SET tensor_uri = ?
                    WHERE latent_package_id = ?
                    """,
                    (str(target), str(package.get("latent_package_id") or "")),
                )
            archived += 1
        return {
            "job": "latent_expiry_sweeper",
            "archived": archived,
            "deleted": deleted,
            "mode": mode,
        }

    def embedding_migrator(
        self,
        campaign_id: str,
        *,
        embedding_version: str = "v2",
    ) -> Dict[str, Any]:
        migrated = 0
        for memory in self.store.list_memories(campaign_id):
            text = self.projector.text_for(memory)
            vector = self.projector.dense_embedding(text)
            self.store.put_embedding(
                str(memory.get("memory_id") or ""),
                embedding_family="almf-dense",
                embedding_version=embedding_version,
                vector=vector,
            )
            migrated += 1
        return {
            "job": "embedding_migrator",
            "embedding_version": embedding_version,
            "migrated": migrated,
        }

    def graph_community_builder(self, campaign_id: str) -> Dict[str, Any]:
        community = self.community_builder.build(self.store.list_memories(campaign_id))
        source_id = f"campaign::{campaign_id}"
        memory_id = self.store.resolve_alias(
            campaign_id=campaign_id,
            source_table="memory_community_summary",
            source_id=source_id,
        )
        if memory_id:
            summary = self.store.get_memory(memory_id)
            if summary:
                summary["payload_json"] = community
                summary["summary_text"] = f"Community summary for {campaign_id} ({community['community_count']} groups)"
                summary["updated_at"] = time.time()
                self.store.update_memory(summary)
                return {"job": "graph_community_builder", "memory_id": memory_id, "community": community}
        memory = self.store.create_memory(
            memory_kind="community_summary",
            payload_json=community,
            campaign_id=campaign_id,
            workspace_id=campaign_id,
            source_system="almf.jobs.graph_community_builder",
            summary_text=f"Community summary for {campaign_id} ({community['community_count']} groups)",
            retention_class="durable",
            importance_score=0.75,
        )
        self.store.register_alias(
            memory.memory_id,
            "memory_community_summary",
            source_id,
            campaign_id=campaign_id,
        )
        self.projector.project_memory(self.store, memory)
        return {"job": "graph_community_builder", "memory_id": memory.memory_id, "community": community}

    def run_milestone_consolidation(self, campaign_id: str) -> Dict[str, Any]:
        report = {
            "campaign_id": campaign_id,
            "started_at": time.time(),
            "jobs": [
                self.sensory_filter(campaign_id),
                self.session_summarizer(campaign_id),
                self.claim_consolidator(campaign_id),
                self.temporal_tree_builder(campaign_id),
                self.latent_expiry_sweeper(campaign_id),
                self.embedding_migrator(campaign_id),
                self.graph_community_builder(campaign_id),
            ],
        }
        report["completed_at"] = time.time()
        report["job_count"] = len(report["jobs"])
        telemetry = self.store.create_memory(
            memory_kind="telemetry_snapshot",
            payload_json=report,
            campaign_id=campaign_id,
            workspace_id=campaign_id,
            source_system="almf.jobs.milestone_consolidation",
            summary_text=f"Milestone consolidation finished with {report['job_count']} jobs",
            retention_class="durable",
            importance_score=0.8,
        )
        self.projector.project_memory(self.store, telemetry)
        report["telemetry_memory_id"] = telemetry.memory_id
        return report
