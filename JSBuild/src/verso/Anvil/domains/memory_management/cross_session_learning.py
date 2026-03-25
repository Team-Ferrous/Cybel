from typing import List, Dict, Any, Optional
from datetime import datetime

from core.memory.fabric import MemoryFabricStore, MemoryProjector
from domains.memory_management.backends.hybrid_backend import HybridMemoryBackend
from domains.memory_management.memory_types import MemoryTier


class Lesson:
    """Represents a learned experience from a previous task."""

    def __init__(
        self,
        objective: str,
        summary: str,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.objective = objective
        self.summary = summary
        self.success = success
        self.timestamp = datetime.now().isoformat()
        self.metadata = metadata or {}


class CrossSessionLearning:
    """
    Manages the persistence and retrieval of "lessons" across agent sessions.
    Uses the Hybrid Memory Backend for semantic lookup.
    """

    def __init__(self, memory_backend: Optional[HybridMemoryBackend] = None):
        self.backend = memory_backend or HybridMemoryBackend()
        self.fabric_store = MemoryFabricStore.from_db_path(".anvil/memory/almf.db")
        self.memory_projector = MemoryProjector()

    def record_lesson(self, lesson: Lesson):
        """Stores a lesson in episodic memory for future retrieval."""
        key = f"lesson_{lesson.timestamp}"

        # We store structured lesson in Semantic/Preference-like tier (Permanent)
        # But we also index it in Episodic for semantic search
        self.backend.write(
            MemoryTier.SEMANTIC,
            key,
            {
                "objective": lesson.objective,
                "summary": lesson.summary,
                "success": lesson.success,
                "metadata": lesson.metadata,
            },
        )
        self.backend.write(
            MemoryTier.WORKING,
            key,
            {
                "objective": lesson.objective,
                "summary": lesson.summary,
                "success": lesson.success,
                "metadata": lesson.metadata,
            },
        )

        # Manually force a compression-like indexing of this specific high-value lesson
        self.backend.compress_working_memory(
            summary=f"Lesson: {lesson.objective[:50]}..."
        )
        self.backend.save()
        memory = self.fabric_store.create_memory(
            memory_kind="task_memory",
            payload_json={
                "objective": lesson.objective,
                "summary": lesson.summary,
                "success": lesson.success,
                "metadata": lesson.metadata,
                "timestamp": lesson.timestamp,
            },
            campaign_id="cross_session",
            workspace_id="cross_session",
            source_system="cross_session_learning",
            summary_text=lesson.summary,
            lifecycle_state="completed" if lesson.success else "failed",
            importance_score=0.7,
            confidence_score=1.0 if lesson.success else 0.5,
        )
        self.fabric_store.register_alias(
            memory.memory_id,
            "cross_session_lessons",
            key,
            campaign_id="cross_session",
        )
        self.memory_projector.project_memory(
            self.fabric_store,
            memory,
            include_multivector=True,
        )

    def retrieve_similar_lessons(
        self, current_objective: str, k: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieves lessons from previous similar tasks."""
        # Search episodic timeline for similar objectives
        results = self.backend.query_episodic(current_objective, k=k)

        # Filter for successful lessons primarily
        relevant_lessons = []
        for r in results:
            if "Lesson" in r.get("summary", ""):
                relevant_lessons.append(r)

        return relevant_lessons
