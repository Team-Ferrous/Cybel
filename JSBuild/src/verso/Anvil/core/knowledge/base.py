from enum import Enum
import time
from typing import List, Dict


class KnowledgeCategory(Enum):
    CODE_PATTERN = "code_pattern"
    SOLUTION = "solution"
    BEST_PRACTICE = "best_practice"
    ERROR_FIX = "error_fix"
    API_USAGE = "api_usage"
    ARCHITECTURE = "architecture"
    PERFORMANCE_TIP = "performance_tip"


class KnowledgeBase:
    """Persistent knowledge base for agent learning."""

    def __init__(self, storage_path: str = ".anvil/knowledge"):
        self.storage_path = storage_path
        # Mock vector store for now
        self.knowledge_store: List[Dict] = []

    def store_knowledge(
        self, content: str, category: KnowledgeCategory, source: str, tags: List[str]
    ) -> str:
        """Store knowledge item."""

        # In a real implementation, this would generate embeddings
        # and store in a vector DB (like FAISS or Chroma).
        # Here we just store in list for demonstration.

        item = {
            "id": f"kb_{len(self.knowledge_store)}",
            "content": content,
            "category": category.value,
            "source": source,
            "tags": tags,
            "timestamp": time.time(),
        }
        self.knowledge_store.append(item)
        return item["id"]

    def search_knowledge(self, query: str, k: int = 5) -> List[Dict]:
        """Search knowledge base (naive keyword search for now)."""
        # Naive implementation: Check if query terms are in content
        results = []
        q_terms = query.lower().split()

        for item in self.knowledge_store:
            score = 0
            content_lower = item["content"].lower()
            for term in q_terms:
                if term in content_lower:
                    score += 1

            if score > 0:
                results.append((score, item))

        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results[:k]]
