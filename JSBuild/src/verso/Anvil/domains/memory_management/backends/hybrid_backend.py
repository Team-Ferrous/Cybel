import numpy as np
import os
import json
from typing import Any, Dict, List, Optional
from domains.memory_management.memory_types import MemoryTier
from domains.memory_management.hd.compression import HDCompressor
from domains.memory_management.persistence import MemoryPersistence


class ChromaDBBackend:
    """
    Accelerator for fast candidate retrieval.
    Secondary index to avoid full linear scans of HD vectors when possible.
    """

    def __init__(self, persist_dir: str = ".anvil/memory/chroma"):
        self.enabled = False
        self.persist_dir = persist_dir
        try:
            import chromadb

            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.collection = self.client.get_or_create_collection(
                name="episodic_memories"
            )
            self.enabled = True
        except ImportError:
            # print("Warning: ChromaDB not installed. Hybrid search will use HD linear scan.")
            pass

    def add(self, embedding: np.ndarray, metadata_id: int, content: str):
        if not self.enabled:
            return

        # ChromaDB expects float embeddings for indexing
        # We use a float32 representation of the HD vector (0, 1 -> 0.0, 1.0)
        # or a subset if dimension is too large for the HNSW index
        self.collection.add(
            embeddings=[embedding.astype(np.float32).tolist()],
            metadatas=[{"id": metadata_id, "content": content[:100]}],
            ids=[str(metadata_id)],
        )

    def search(self, query_emb: np.ndarray, k: int = 5) -> List[int]:
        if not self.enabled:
            return []

        results = self.collection.query(
            query_embeddings=[query_emb.astype(np.float32).tolist()], n_results=k
        )

        # Return internal IDs as integers
        return [int(id_str) for id_str in results["ids"][0]]


class HybridMemoryBackend:
    """
    Hybrid Memory Backend (HD Primary + ChromaDB Secondary).

    CRITICAL Design Principle:
    HD Vectors = GOLD STANDARD (Primary)
    ChromaDB = FAST LOOKUP ONLY (Secondary accelerator)
    """

    def __init__(
        self,
        hd_compressor: HDCompressor = None,
        *,
        base_dir: str = ".anvil/memory",
    ):
        self.compressor = hd_compressor if hd_compressor else HDCompressor()
        self.base_dir = base_dir
        self.chroma = ChromaDBBackend(persist_dir=os.path.join(base_dir, "chroma"))

        # Storage for each tier
        self.storage: Dict[MemoryTier, Dict[str, Any]] = {
            tier: {} for tier in MemoryTier
        }

        # Episodic specifically stores HD vectors
        self.episodic_timeline: List[np.ndarray] = []
        # Store metadata/raw text for re-ranking and retrieval
        self.episodic_metadata: List[Dict[str, Any]] = []

        # Persistence
        self.persistence = MemoryPersistence(base_dir)

        # Load state
        self.episodic_timeline = self.persistence.load(self.storage)
        self._load_metadata()

    def _load_metadata(self):
        """Loads episodic metadata for re-ranking."""
        meta_path = os.path.join(self.base_dir, "episodic_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    self.episodic_metadata = json.load(f)
            except Exception:
                self.episodic_metadata = []

    def _save_metadata(self):
        """Saves episodic metadata."""
        meta_path = os.path.join(self.base_dir, "episodic_meta.json")
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(self.episodic_metadata, f, indent=2)

    def save(self):
        """Save current state to disk."""
        self.persistence.save(self.storage, self.episodic_timeline)
        self._save_metadata()

    def write(self, tier: MemoryTier, key: str, value: Any) -> None:
        """Write item to specific tier."""
        self.storage[tier][key] = value

    def index_item(self, key: str, value: Any, *, summary: str) -> None:
        """Persist a high-value item into episodic memory immediately."""
        self.write(MemoryTier.WORKING, key, value)
        self.compress_working_memory(summary=summary)
        self.save()

    def read(self, tier: MemoryTier, key: str) -> Optional[Any]:
        """Read item from tier."""
        return self.storage[tier].get(key)

    def compress_working_memory(self, summary: str = "Working memory snapshot") -> int:
        """
        Compress contents of WORKING memory into EPISODIC memory.
        Returns number of items compressed.
        """
        working = self.storage[MemoryTier.WORKING]
        if not working:
            return 0

        # Convert all working memory items to text representation
        items = [f"{k}: {v}" for k, v in working.items()]
        content = "\n".join(items)

        # 1. Store in HD timeline (PRIMARY - gold standard)
        bundle = self.compressor.encode_sequence(items)
        self.episodic_timeline.append(bundle)

        meta_id = len(self.episodic_timeline) - 1
        self.episodic_metadata.append(
            {
                "id": meta_id,
                "summary": summary,
                "content": content,
                "timestamp": os.getenv("CURRENT_TIME", ""),
            }
        )

        # 2. Index in ChromaDB (SECONDARY - accelerator)
        # We quantize/map the HD vector to a searchable embedding for Chroma
        self.chroma.add(bundle, meta_id, content)

        count = len(working)
        working.clear()

        return count

    def query_episodic(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Query episodic timeline for relevant bundles.
        Uses Hybrid Search (ChromaDB candidates + HD Re-ranking).
        """
        query_vec = self.compressor.encode_context(query)

        candidates_ids = []
        if self.chroma.enabled:
            # 1. ChromaDB fast candidate retrieval (k*2 candidates)
            candidates_ids = self.chroma.search(query_vec, k=k * 2)

        # 2. HD re-ranking (GOLD STANDARD)
        # If no candidates from Chroma, scan full timeline (Fallback)
        ids_to_rank = (
            candidates_ids if candidates_ids else range(len(self.episodic_timeline))
        )

        ranked_results = []
        for i in ids_to_rank:
            if i >= len(self.episodic_timeline):
                continue
            bundle = self.episodic_timeline[i]
            sim = self.compressor.bundle_op.similarity(query_vec, bundle)
            ranked_results.append((sim, i))

        # Sort by HD similarity (Gold Standard)
        ranked_results.sort(key=lambda x: x[0], reverse=True)

        # Return top-k enriched with metadata
        results = []
        for score, idx in ranked_results[:k]:
            meta = self.episodic_metadata[idx].copy()
            meta["score"] = float(score)
            results.append(meta)

        return results


# UnifiedMemory alias for backward compatibility
UnifiedMemory = HybridMemoryBackend
