"""Audit helpers for ALMF governance."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict

from core.memory.fabric.policies import RetentionPolicy
from core.memory.fabric.store import MemoryFabricStore


class MemoryFabricAuditor:
    """Compute assurance summaries for ALMF state."""

    def __init__(self, store: MemoryFabricStore) -> None:
        self.store = store

    def audit_campaign(self, campaign_id: str) -> Dict[str, Any]:
        memories = self.store.list_memories(campaign_id)
        by_kind: dict[str, int] = defaultdict(int)
        provenance_covered = 0
        sensitivity_covered = 0
        retention_issues = []
        for memory in memories:
            memory_kind = str(memory.get("memory_kind") or "unknown")
            by_kind[memory_kind] += 1
            if memory.get("provenance_json"):
                provenance_covered += 1
            if str(memory.get("sensitivity_class") or ""):
                sensitivity_covered += 1
            validation = RetentionPolicy.validate_memory(
                memory_kind,
                retention_class=str(memory.get("retention_class") or ""),
                sensitivity_class=str(memory.get("sensitivity_class") or ""),
                provenance_present=bool(memory.get("provenance_json")),
            )
            if not validation["ok"]:
                retention_issues.append(
                    {
                        "memory_id": str(memory.get("memory_id") or ""),
                        "issues": validation["issues"],
                    }
                )
        orphan_embeddings = 0
        rows = self.store.state_store.fetchall(
            "SELECT memory_id FROM memory_embeddings",
            (),
        )
        for row in rows:
            if self.store.get_memory(str(row["memory_id"])) is None:
                orphan_embeddings += 1
        latent_rows = self.store.state_store.fetchall(
            "SELECT latent_package_id, expires_at FROM latent_packages",
            (),
        )
        expired_latent_packages = [
            row["latent_package_id"]
            for row in latent_rows
            if RetentionPolicy.is_expired(row.get("expires_at"))
        ]
        contradiction_edges = self.store.list_edges(edge_type="contradicts")
        return {
            "campaign_id": campaign_id,
            "memory_count": len(memories),
            "memory_kinds": dict(sorted(by_kind.items())),
            "provenance_coverage": (
                float(provenance_covered) / float(len(memories)) if memories else 1.0
            ),
            "sensitivity_coverage": (
                float(sensitivity_covered) / float(len(memories)) if memories else 1.0
            ),
            "orphan_embeddings": orphan_embeddings,
            "expired_latent_packages": expired_latent_packages,
            "contradiction_edges": len(contradiction_edges),
            "retention_issues": retention_issues,
        }
