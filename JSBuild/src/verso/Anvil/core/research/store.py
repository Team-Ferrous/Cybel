"""Campaign-local research database helpers."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from core.campaign.state_store import CampaignStateStore
from core.memory.fabric import MemoryEdge, MemoryFabricStore, MemoryProjector


class ResearchStore:
    """Typed helper around the campaign state store's research tables."""

    def __init__(self, campaign_id: str, state_store: CampaignStateStore):
        self.campaign_id = campaign_id
        self.state_store = state_store
        self.memory_fabric = MemoryFabricStore(state_store)
        self.memory_projector = MemoryProjector()

    def record_source(
        self,
        source_type: str,
        origin_url: str,
        metadata: dict[str, Any],
        *,
        repo_context: str = "",
    ) -> str:
        source_id = str(uuid.uuid4())
        digest = str(metadata.get("digest") or "")
        self.state_store.execute(
            """
            INSERT INTO research_sources (
                campaign_id, repo_context, source_id, source_type, origin_url,
                trust_level, digest, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(campaign_id, repo_context, source_type, origin_url, digest) DO UPDATE SET
                trust_level = excluded.trust_level,
                metadata_json = excluded.metadata_json
            """,
            (
                self.campaign_id,
                repo_context,
                source_id,
                source_type,
                origin_url,
                metadata.get("trust_level", "medium"),
                digest,
                json.dumps(metadata, default=str),
                time.time(),
            ),
        )
        row = self.state_store.fetchone(
            """
            SELECT source_id
            FROM research_sources
            WHERE campaign_id = ? AND repo_context = ? AND source_type = ? AND origin_url = ? AND COALESCE(digest, '') = ?
            LIMIT 1
            """,
            (
                self.campaign_id,
                repo_context,
                source_type,
                origin_url,
                digest,
            ),
        )
        source_id = str((row or {})["source_id"])
        self._mirror_memory(
            memory_kind="research_source",
            source_table="research_sources",
            source_id=source_id,
            payload={
                "source_id": source_id,
                "source_type": source_type,
                "origin_url": origin_url,
                "metadata": metadata,
            },
            summary_text=str(metadata.get("topic") or origin_url or source_type),
            repo_context=repo_context,
            source_system="research_store.record_source",
        )
        return source_id

    def record_document(
        self,
        source_id: str,
        title: str,
        normalized: dict[str, Any],
        path: str = "",
        *,
        repo_context: str = "",
    ) -> str:
        document_id = str(uuid.uuid4())
        digest = str(normalized.get("digest") or "")
        self.state_store.execute(
            """
            INSERT INTO research_documents (
                campaign_id, repo_context, document_id, source_id, title, path, digest,
                normalized_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(campaign_id, repo_context, source_id, title, digest) DO UPDATE SET
                path = excluded.path,
                normalized_json = excluded.normalized_json
            """,
            (
                self.campaign_id,
                repo_context,
                document_id,
                source_id,
                title,
                path,
                digest,
                json.dumps(normalized, default=str),
                time.time(),
            ),
        )
        row = self.state_store.fetchone(
            """
            SELECT document_id
            FROM research_documents
            WHERE campaign_id = ? AND repo_context = ? AND source_id = ? AND title = ? AND COALESCE(digest, '') = ?
            LIMIT 1
            """,
            (
                self.campaign_id,
                repo_context,
                source_id,
                title,
                digest,
            ),
        )
        document_id = str((row or {})["document_id"])
        document_memory = self._mirror_memory(
            memory_kind="research_document",
            source_table="research_documents",
            source_id=document_id,
            payload={
                "document_id": document_id,
                "source_id": source_id,
                "title": title,
                "path": path,
                "normalized": normalized,
            },
            summary_text=title,
            repo_context=repo_context,
            source_system="research_store.record_document",
            document_id=document_id,
        )
        source_memory_id = self.memory_fabric.resolve_alias(
            campaign_id=self.campaign_id,
            source_table="research_sources",
            source_id=source_id,
        )
        if source_memory_id:
            self.memory_fabric.add_edge(
                MemoryEdge(
                    src_memory_id=document_memory.memory_id,
                    dst_memory_id=source_memory_id,
                    edge_type="derived_from",
                    evidence_json={"source_id": source_id},
                )
            )
        return document_id

    def record_chunk(
        self,
        document_id: str,
        *,
        content: str,
        topic: str,
        position: int,
        metadata: dict[str, Any] | None = None,
        repo_context: str = "",
    ) -> str:
        chunk_id = str(uuid.uuid4())
        payload = {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "topic": topic,
            "position": position,
            "content": content,
            "metadata": dict(metadata or {}),
        }
        self.state_store.insert_json_row(
            "source_chunks",
            campaign_id=self.campaign_id,
            payload=payload,
            id_field="chunk_id",
            id_value=chunk_id,
            extra={"document_id": document_id, "repo_context": repo_context},
        )
        chunk_memory = self._mirror_memory(
            memory_kind="research_chunk",
            source_table="source_chunks",
            source_id=chunk_id,
            payload=payload,
            summary_text=f"{topic}: {content[:120]}",
            repo_context=repo_context,
            source_system="research_store.record_chunk",
            document_id=document_id,
        )
        document_memory_id = self.memory_fabric.resolve_alias(
            campaign_id=self.campaign_id,
            source_table="research_documents",
            source_id=document_id,
        )
        if document_memory_id:
            self.memory_fabric.add_edge(
                MemoryEdge(
                    src_memory_id=chunk_memory.memory_id,
                    dst_memory_id=document_memory_id,
                    edge_type="derived_from",
                    evidence_json={"document_id": document_id},
                )
            )
        return chunk_id

    def document_has_chunks(self, document_id: str, *, repo_context: str = "") -> bool:
        row = self.state_store.fetchone(
            """
            SELECT 1 AS present
            FROM source_chunks
            WHERE campaign_id = ? AND repo_context = ? AND document_id = ?
            LIMIT 1
            """,
            (self.campaign_id, repo_context, document_id),
        )
        return row is not None

    def record_claim(
        self,
        document_id: str,
        topic: str,
        summary: str,
        confidence: float,
        provenance: dict[str, Any],
        *,
        repo_context: str = "",
        complexity_score: float = 0.0,
        topic_hierarchy: str = "",
        applicability_score: float = 0.0,
        evidence_type: str = "implementation",
        utility_context: dict[str, Any] | None = None,
    ) -> str:
        claim_id = str(uuid.uuid4())
        normalized_provenance = {
            **dict(provenance or {}),
            "utility_context": dict(utility_context or {}),
        }
        self.state_store.execute(
            """
            INSERT INTO research_claims (
                campaign_id, repo_context, claim_id, document_id, topic, summary, confidence,
                complexity_score, topic_hierarchy, applicability_score, evidence_type,
                provenance_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(campaign_id, repo_context, document_id, topic, summary) DO UPDATE SET
                confidence = CASE
                    WHEN excluded.confidence > research_claims.confidence
                    THEN excluded.confidence
                    ELSE research_claims.confidence
                END,
                complexity_score = excluded.complexity_score,
                topic_hierarchy = excluded.topic_hierarchy,
                applicability_score = excluded.applicability_score,
                evidence_type = excluded.evidence_type,
                provenance_json = excluded.provenance_json
            """,
            (
                self.campaign_id,
                repo_context,
                claim_id,
                document_id,
                topic,
                summary,
                confidence,
                complexity_score,
                topic_hierarchy,
                applicability_score,
                evidence_type,
                json.dumps(normalized_provenance, default=str),
                time.time(),
            ),
        )
        row = self.state_store.fetchone(
            """
            SELECT claim_id
            FROM research_claims
            WHERE campaign_id = ? AND repo_context = ? AND document_id = ? AND topic = ? AND summary = ?
            LIMIT 1
            """,
            (self.campaign_id, repo_context, document_id, topic, summary),
        )
        claim_id = str((row or {})["claim_id"])
        claim_memory = self._mirror_memory(
            memory_kind="research_claim",
            source_table="research_claims",
            source_id=claim_id,
            payload={
                "claim_id": claim_id,
                "document_id": document_id,
                "topic": topic,
                "summary": summary,
                "confidence": confidence,
                "provenance": provenance,
                "complexity_score": complexity_score,
                "topic_hierarchy": topic_hierarchy,
                "applicability_score": applicability_score,
                "evidence_type": evidence_type,
                "utility_context": dict(utility_context or {}),
            },
            summary_text=summary,
            repo_context=repo_context,
            source_system="research_store.record_claim",
            document_id=document_id,
            claim_id=claim_id,
            importance_score=max(float(confidence), float(applicability_score)),
            confidence_score=float(confidence),
        )
        document_memory_id = self.memory_fabric.resolve_alias(
            campaign_id=self.campaign_id,
            source_table="research_documents",
            source_id=document_id,
        )
        if document_memory_id:
            self.memory_fabric.add_edge(
                MemoryEdge(
                    src_memory_id=claim_memory.memory_id,
                    dst_memory_id=document_memory_id,
                    edge_type="derived_from",
                    evidence_json={"document_id": document_id},
                )
            )
        return claim_id

    def record_usage_trace(
        self,
        document_id: str,
        *,
        repo_context: str,
        symbol_name: str,
        trace: dict[str, Any],
    ) -> None:
        self.state_store.record_usage_trace(
            self.campaign_id,
            repo_context,
            document_id,
            symbol_name,
            trace,
        )
        trace_memory = self._mirror_memory(
            memory_kind="repo_usage_trace",
            source_table="usage_traces",
            source_id=f"{document_id}:{symbol_name}:{trace.get('line', '')}",
            payload={
                "document_id": document_id,
                "symbol_name": symbol_name,
                "trace": trace,
            },
            summary_text=f"{symbol_name} observed in {repo_context}",
            repo_context=repo_context,
            source_system="research_store.record_usage_trace",
            document_id=document_id,
        )
        document_memory_id = self.memory_fabric.resolve_alias(
            campaign_id=self.campaign_id,
            source_table="research_documents",
            source_id=document_id,
        )
        if document_memory_id:
            self.memory_fabric.add_edge(
                MemoryEdge(
                    src_memory_id=trace_memory.memory_id,
                    dst_memory_id=document_memory_id,
                    edge_type="observed_in",
                    evidence_json={"symbol_name": symbol_name},
                )
            )

    def persist_clusters(self, clusters: list[dict[str, Any]]) -> None:
        self.state_store.execute(
            "DELETE FROM topic_clusters WHERE campaign_id = ?",
            (self.campaign_id,),
        )
        for cluster in clusters:
            self.state_store.execute(
                """
                INSERT INTO topic_clusters (
                    campaign_id, cluster_id, topic, label, members_json, score, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.campaign_id,
                    cluster["cluster_id"],
                    cluster["topic"],
                    cluster["label"],
                    json.dumps(cluster["members"], default=str),
                    float(cluster["score"]),
                    time.time(),
                ),
            )

    def list_claims(self) -> list[dict[str, Any]]:
        rows = self.state_store.fetchall(
            """
            SELECT claim_id, repo_context, document_id, topic, summary, confidence,
                   complexity_score, topic_hierarchy, applicability_score, evidence_type,
                   provenance_json, created_at
            FROM research_claims
            WHERE campaign_id = ?
            ORDER BY created_at ASC
            """,
            (self.campaign_id,),
        )
        claims: list[dict[str, Any]] = []
        for row in rows:
            claims.append(
                {
                    **row,
                    "provenance": json.loads(row["provenance_json"] or "{}"),
                }
            )
        return claims

    def _mirror_memory(
        self,
        *,
        memory_kind: str,
        source_table: str,
        source_id: str,
        payload: dict[str, Any],
        summary_text: str,
        repo_context: str,
        source_system: str,
        **extra: Any,
    ):
        existing_memory_id = self.memory_fabric.resolve_alias(
            campaign_id=self.campaign_id,
            source_table=source_table,
            source_id=source_id,
        )
        memory = self.memory_fabric.create_memory(
            memory_kind=memory_kind,
            payload_json=payload,
            campaign_id=self.campaign_id,
            workspace_id=self.campaign_id,
            repo_context=repo_context,
            source_system=source_system,
            summary_text=summary_text,
            provenance_json={"source_table": source_table, "source_id": source_id},
            **extra,
        )
        if existing_memory_id:
            memory.memory_id = existing_memory_id
            self.memory_fabric.upsert_memory(memory)
        self.memory_fabric.register_alias(
            memory.memory_id,
            source_table,
            source_id,
            campaign_id=self.campaign_id,
        )
        self.memory_projector.project_memory(
            self.memory_fabric,
            memory,
            include_multivector=memory_kind in {"research_chunk", "research_claim"},
        )
        return memory
