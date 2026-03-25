import numpy as np
from typing import Dict, Any, List, Optional
from domains.memory_management.hd.compression import HDCompressor


class SharedMemoryNamespace:
    """
    Shared memory space for sequential swarm agents.
    Allows agents to pass data and context without redundant prompts.
    """

    def __init__(self, compressor: Optional[HDCompressor] = None):
        self.context: Dict[str, Any] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.compressor = compressor or HDCompressor()

    def update(self, agent_id: str, summary: str, data: Dict[str, Any]):
        """
        Update the shared space with results from an agent.
        """
        # Store structured data
        self.context[f"{agent_id}_result"] = data
        self.context[f"{agent_id}_summary"] = summary

        # Store semantic representation for retrieval
        embedding = self.compressor.encode_context(summary)
        self.embeddings[agent_id] = embedding

    def get_relevant_context(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve k most relevant prior agent results based on query.
        """
        if not self.embeddings:
            return []

        query_emb = self.compressor.encode_context(query)

        scores = []
        for agent_id, emb in self.embeddings.items():
            sim = self.compressor.bundle_op.similarity(query_emb, emb)
            scores.append((sim, agent_id))

        # Sort by similarity desc
        scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for _, agent_id in scores[:k]:
            results.append(
                {
                    "agent_id": agent_id,
                    "summary": self.context[f"{agent_id}_summary"],
                    "data": self.context[f"{agent_id}_result"],
                }
            )

        return results

    def clear(self):
        """Clear all shared context."""
        self.context.clear()
        self.embeddings.clear()
