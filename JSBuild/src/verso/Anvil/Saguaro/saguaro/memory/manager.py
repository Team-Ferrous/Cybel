"""Tiered Agentic Memory for SAGUARO.
Manages Working, Episodic, Semantic, and Preference memory namespaces.
"""

import json
import logging
import os
import time
from typing import Any

from core.memory.fabric import MemoryFabricStore, MemoryProjector

logger = logging.getLogger(__name__)


class MemoryTier:
    """Provide MemoryTier support."""
    WORKING = "working"  # Current context / task
    EPISODIC = "episodic"  # Past interactions
    SEMANTIC = "semantic"  # Factual knowledge (holographic)
    PREFERENCE = "preference"  # User-specific style/rules


class MemoryManager:
    """Provide MemoryManager support."""
    def __init__(self, storage_root: str) -> None:
        """Initialize the instance."""
        self.root = os.path.join(storage_root, "agentic_memory")
        os.makedirs(self.root, exist_ok=True)
        self.fabric_store = MemoryFabricStore.from_db_path(
            os.path.join(storage_root, "almf.db")
        )
        self.memory_projector = MemoryProjector()

    def _get_path(self, tier: str, namespace: str = "default") -> str:
        tier_dir = os.path.join(self.root, tier)
        os.makedirs(tier_dir, exist_ok=True)
        return os.path.join(tier_dir, f"{namespace}.json")

    def store(self, tier: str, content: Any, namespace: str = "default") -> None:
        """Handle store."""
        path = self._get_path(tier, namespace)
        data = {"timestamp": time.time(), "content": content}
        # In a real implementation, we would append or use a DB
        # For this prototype, we store as list items in JSON
        existing = self.retrieve(tier, namespace) or []
        existing.append(data)
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)
        memory = self.fabric_store.create_memory(
            memory_kind="operator_note",
            payload_json={"tier": tier, "namespace": namespace, "content": content},
            campaign_id="saguaro",
            workspace_id=namespace,
            session_id=namespace,
            source_system="saguaro.memory.manager",
            summary_text=str(content)[:160],
            retention_class="session" if tier == MemoryTier.WORKING else "durable",
        )
        self.memory_projector.project_memory(self.fabric_store, memory)

    def retrieve(self, tier: str, namespace: str = "default") -> list[dict[str, Any]]:
        """Handle retrieve."""
        path = self._get_path(tier, namespace)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return []

    def store_semantic(self, fact: str, embedding: Any | None = None) -> None:
        """Store semantic fact with optional holographic embedding."""
        if embedding is None:
            # Generate embedding via encoder
            from saguaro.encoder import encode_text

            embedding = encode_text(fact)

        from saguaro.ops.holographic import serialize_hd_state

        path = self._get_path(MemoryTier.SEMANTIC)
        # Store both text and serialized embedding
        entry = {
            "timestamp": time.time(),
            "fact": fact,
            "hd_bundle": (
                serialize_hd_state(embedding).hex()
                if hasattr(embedding, "numpy")
                else embedding.hex()
            ),
        }

        existing = self.retrieve(MemoryTier.SEMANTIC)
        existing.append(entry)
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)
        memory = self.fabric_store.create_memory(
            memory_kind="design_decision",
            payload_json={"fact": fact, "tier": MemoryTier.SEMANTIC},
            campaign_id="saguaro",
            workspace_id="default",
            source_system="saguaro.memory.manager",
            summary_text=fact,
            retention_class="durable",
        )
        self.memory_projector.project_memory(
            self.fabric_store,
            memory,
            include_multivector=True,
        )

    def clear(self, tier: str, namespace: str = "default") -> None:
        """Handle clear."""
        path = self._get_path(tier, namespace)
        if os.path.exists(path):
            os.remove(path)
