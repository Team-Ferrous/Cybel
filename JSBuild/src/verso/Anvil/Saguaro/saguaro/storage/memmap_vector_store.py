"""SAGUARO Memory-Mapped Vector Store
Enterprise-grade storage backend for Hyperdimensional Vectors with O(1) memory footprint.

This implementation uses numpy memory-mapped arrays to store vectors directly on disk,
allowing indexing of repositories of any size without running out of RAM.
"""

import json
import logging
import os
import re
import threading
from typing import Any

import numpy as np

from saguaro.errors import SaguaroStateCorruptionError
from saguaro.storage.atomic_fs import atomic_write_json

logger = logging.getLogger(__name__)
_IDENT_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")


class MemoryMappedVectorStore:
    """Memory-efficient vector storage using numpy memory-mapped arrays.

    Memory Characteristics:
    - Vectors are stored on disk, loaded on-demand by the OS
    - Only actively accessed pages are kept in RAM
    - Supports unlimited repository sizes (disk-bound only)
    - Thread-safe for concurrent writes

    Storage Format:
    - vectors.bin: Memory-mapped numpy array [capacity, dim]
    - metadata.json: JSON array of metadata dicts
    - index_meta.json: Store configuration (dim, count, capacity)
    """

    GROWTH_FACTOR = 2.0  # Double capacity when full
    INITIAL_CAPACITY = 10000  # Initial number of vectors

    def __init__(
        self,
        storage_path: str,
        dim: int,
        dark_space_ratio: float = 0.4,
        read_only: bool = False,
    ) -> None:
        """Initialize the memory-mapped vector store.

        Args:
            storage_path: Directory to store index files
            dim: Vector dimension
            dark_space_ratio: Reserved space ratio (for compatibility)
            read_only: If True, open existing index in read-only mode
        """
        self.storage_path = storage_path
        self.dim = dim
        self.dark_space_ratio = dark_space_ratio
        self.read_only = read_only

        self.vectors_path = os.path.join(storage_path, "vectors.bin")
        self.metadata_path = os.path.join(storage_path, "metadata.json")
        self.index_meta_path = os.path.join(storage_path, "index_meta.json")

        self._count: int = 0
        self._capacity: int = 0
        self._lookup: dict[tuple[str, str], int] = {}  # (file, entity_id_or_name) -> index
        self._entity_lookup: dict[str, int] = {}
        self._term_index: dict[str, set[int]] = {}
        self._indexes_dirty = True

        self._write_lock = threading.RLock()

        self._load()

    def _load(self) -> None:
        """Load or initialize the vector store."""
        os.makedirs(self.storage_path, exist_ok=True)

        if os.path.exists(self.index_meta_path):
            # Load existing store
            try:
                with open(self.index_meta_path) as f:
                    meta = json.load(f)

                stored_dim = meta.get("dim", self.dim)
                if stored_dim != self.dim:
                    logger.warning(
                        f"Dimension mismatch: stored={stored_dim}, requested={self.dim}. "
                        f"Using stored dimension."
                    )
                    self.dim = stored_dim

                self._count = meta.get("count", 0)
                self._capacity = meta.get("capacity", self.INITIAL_CAPACITY)

                # Open memory-mapped file
                mode = "r" if self.read_only else "r+"
                if os.path.exists(self.vectors_path):
                    self._vectors = np.memmap(
                        self.vectors_path,
                        dtype=np.float32,
                        mode=mode,
                        shape=(self._capacity, self.dim),
                    )
                else:
                    raise SaguaroStateCorruptionError(
                        f"Missing vectors file for vector store: {self.vectors_path}"
                    )

                # Load metadata
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path) as f:
                        self._metadata = json.load(f)
                else:
                    raise SaguaroStateCorruptionError(
                        f"Missing metadata file for vector store: {self.metadata_path}"
                    )
                self._validate_state()
                self._rebuild_indexes()

                logger.info(
                    f"Loaded MemoryMappedVectorStore: {self._count} vectors, "
                    f"capacity={self._capacity}, dim={self.dim}, "
                    f"lookup_size={len(self._lookup)}"
                )

            except Exception as e:
                raise SaguaroStateCorruptionError(
                    f"Failed to load existing store from {self.storage_path}: {e}"
                ) from e
        else:
            if any(
                os.path.exists(path)
                for path in (self.vectors_path, self.metadata_path)
            ):
                raise SaguaroStateCorruptionError(
                    "Incomplete vector store; index metadata is missing."
                )
            self._initialize_fresh()

    def _initialize_fresh(self) -> None:
        """Initialize a fresh vector store."""
        self._count = 0
        self._capacity = self.INITIAL_CAPACITY
        self._metadata = []
        self._lookup = {}
        self._entity_lookup = {}
        self._term_index = {}
        self._indexes_dirty = False
        self._create_vectors_file()
        logger.info(
            f"Initialized fresh MemoryMappedVectorStore: "
            f"capacity={self._capacity}, dim={self.dim}"
        )

    def _create_vectors_file(self) -> None:
        """Create or recreate the vectors memory-mapped file."""
        mode = "r" if self.read_only else "w+"
        self._vectors = np.memmap(
            self.vectors_path,
            dtype=np.float32,
            mode=mode,
            shape=(self._capacity, self.dim),
        )

    def _grow_capacity(self) -> None:
        """Double the capacity of the vector store."""
        if self.read_only:
            raise RuntimeError("Cannot grow capacity in read-only mode")

        new_capacity = int(self._capacity * self.GROWTH_FACTOR)
        logger.info(f"Growing vector store: {self._capacity} -> {new_capacity}")

        # Flush current memmap
        if self._vectors is not None:
            self._vectors.flush()
            del self._vectors

        # Create temporary file with new size
        temp_path = self.vectors_path + ".tmp"
        new_vectors = np.memmap(
            temp_path, dtype=np.float32, mode="w+", shape=(new_capacity, self.dim)
        )

        # Copy existing data
        old_vectors = np.memmap(
            self.vectors_path,
            dtype=np.float32,
            mode="r",
            shape=(self._capacity, self.dim),
        )
        new_vectors[: self._count] = old_vectors[: self._count]
        new_vectors.flush()
        del old_vectors
        del new_vectors

        # Replace old file
        os.replace(temp_path, self.vectors_path)

        self._capacity = new_capacity
        self._vectors = np.memmap(
            self.vectors_path,
            dtype=np.float32,
            mode="r+",
            shape=(self._capacity, self.dim),
        )

    @classmethod
    def _extract_terms(cls, text: str) -> set[str]:
        expanded = _CAMEL_RE.sub(r"\1 \2", str(text or ""))
        terms = set()
        for token in _IDENT_RE.findall(expanded):
            normalized = token.lower()
            if len(normalized) < 3 or normalized.isdigit():
                continue
            terms.add(normalized)
        return terms

    def _rebuild_indexes(self) -> None:
        self._lookup = {}
        self._entity_lookup = {}
        self._term_index = {}
        for idx, meta in enumerate(self._metadata[: self._count]):
            file = meta.get("file")
            identity = meta.get("entity_id") or meta.get("qualified_name") or meta.get("name")
            if file and identity:
                self._lookup[(file, identity)] = idx
            entity_id = meta.get("entity_id")
            if entity_id:
                self._entity_lookup[str(entity_id)] = idx
            raw_terms = list(meta.get("terms", []) or [])
            raw_terms.extend(
                [
                    meta.get("name", ""),
                    meta.get("qualified_name", ""),
                    meta.get("file", ""),
                    meta.get("type", ""),
                    meta.get("parent_symbol", ""),
                    meta.get("chunk_role", ""),
                    meta.get("file_role", ""),
                ]
            )
            for term in self._extract_terms(" ".join(str(item or "") for item in raw_terms)):
                self._term_index.setdefault(term, set()).add(idx)
        self._indexes_dirty = False

    def _ensure_indexes(self) -> None:
        if self._indexes_dirty:
            self._rebuild_indexes()

    def _candidate_indices(self, query_text: str | None, k: int) -> np.ndarray | None:
        if not query_text:
            return None
        self._ensure_indexes()
        candidate_scores: dict[int, float] = {}
        query_terms = self._extract_terms(query_text)
        if not query_terms:
            return None
        for term in query_terms:
            for idx in self._term_index.get(term, ()):
                candidate_scores[idx] = candidate_scores.get(idx, 0.0) + 1.0
        if not candidate_scores:
            return None
        ranked = sorted(
            candidate_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
        limit = min(self._count, max(64, max(1, int(k)) * 32))
        return np.asarray([idx for idx, _score in ranked[:limit]], dtype=np.int64)

    def _indices_for_entity_ids(self, candidate_ids: list[str] | None) -> np.ndarray | None:
        if not candidate_ids:
            return None
        self._ensure_indexes()
        indices = [
            self._entity_lookup[item]
            for item in candidate_ids
            if item in self._entity_lookup
        ]
        if not indices:
            return None
        return np.asarray(sorted(set(indices)), dtype=np.int64)

    def add(self, vector: np.ndarray, meta: dict[str, Any]) -> int:
        """Add a vector to the store.

        Args:
            vector: Vector to add [dim] or [1, dim]
            meta: Metadata dictionary

        Returns:
            Index of the added vector
        """
        if self.read_only:
            raise RuntimeError("Cannot add vectors in read-only mode")

        with self._write_lock:
            # Upsert Logic: Check if entity already exists
            file = meta.get("file")
            identity = meta.get("entity_id") or meta.get("qualified_name") or meta.get("name")
            key = (file, identity) if (file and identity) else None

            # Normalize vector shape
            vec = vector.flatten().astype(np.float32)
            if vec.shape[0] != self.dim:
                if vec.shape[0] < self.dim:
                    vec = np.pad(vec, (0, self.dim - vec.shape[0]))
                else:
                    vec = vec[: self.dim]

            if key and key in self._lookup:
                idx = self._lookup[key]
                self._vectors[idx] = vec
                self._metadata[idx] = meta
                self._indexes_dirty = True
                logger.debug(f"Upserted vector for {identity} in {file}")
                return idx

            # Ensure capacity
            if self._count >= self._capacity:
                self._grow_capacity()

            # Add to memmap
            idx = self._count
            self._vectors[idx] = vec
            self._metadata.append(meta)
            if key:
                self._lookup[key] = idx
            self._count += 1
            self._indexes_dirty = True

            return idx

    def add_batch(self, vectors: np.ndarray, metas: list[dict[str, Any]]) -> int:
        """Add a batch of vectors efficiently.

        Args:
            vectors: Vectors to add [N, dim]
            metas: List of metadata dicts

        Returns:
            Number of vectors added
        """
        if self.read_only:
            raise RuntimeError("Cannot add vectors in read-only mode")

        n = len(metas)
        if n == 0:
            return 0

        with self._write_lock:
            added_count = 0
            for i in range(n):
                meta = metas[i]
                vec = vectors[i].astype(np.float32)

                # Check for upsert
                file = meta.get("file")
                identity = (
                    meta.get("entity_id") or meta.get("qualified_name") or meta.get("name")
                )
                key = (file, identity) if (file and identity) else None

                if key and key in self._lookup:
                    idx = self._lookup[key]
                    self._vectors[idx] = vec
                    self._metadata[idx] = meta
                    self._indexes_dirty = True
                    continue

                # Ensure capacity
                if self._count >= self._capacity:
                    self._grow_capacity()

                # Append
                idx = self._count
                self._vectors[idx] = vec
                self._metadata.append(meta)
                if key:
                    self._lookup[key] = idx
                self._count += 1
                added_count += 1
                self._indexes_dirty = True

            return added_count

    def remove_file(self, file_path: str) -> int:
        """Remove all vectors associated with a file path."""
        if self.read_only:
            raise RuntimeError("Cannot remove vectors in read-only mode")

        with self._write_lock:
            keep_indices = [
                idx
                for idx, meta in enumerate(self._metadata[: self._count])
                if meta.get("file") != file_path
            ]
            removed = self._count - len(keep_indices)
            if removed <= 0:
                return 0

            if keep_indices:
                compact = np.asarray(self._vectors[keep_indices], dtype=np.float32)
                self._vectors[: len(keep_indices)] = compact
            self._count = len(keep_indices)
            self._metadata = [self._metadata[idx] for idx in keep_indices]
            self._lookup = {}
            self._entity_lookup = {}
            for idx, meta in enumerate(self._metadata):
                file = meta.get("file")
                identity = (
                    meta.get("entity_id")
                    or meta.get("qualified_name")
                    or meta.get("name")
                )
                if file and identity:
                    self._lookup[(file, identity)] = idx
                entity_id = meta.get("entity_id")
                if entity_id:
                    self._entity_lookup[str(entity_id)] = idx
            self._indexes_dirty = True
            return removed

    def save(self) -> None:
        """Flush vectors to disk and save metadata."""
        if self.read_only:
            return

        with self._write_lock:
            self._validate_state()
            # Flush memmap
            if self._vectors is not None:
                self._vectors.flush()

            # Save metadata
            atomic_write_json(
                self.metadata_path,
                self._metadata,
                indent=2,
                sort_keys=False,
            )

            # Save index meta
            atomic_write_json(
                self.index_meta_path,
                {
                    "dim": self.dim,
                    "count": self._count,
                    "capacity": self._capacity,
                    "version": 2,
                    "format": "memmap",
                },
                indent=2,
                sort_keys=True,
            )

            logger.debug(f"Saved VectorStore: {self._count} vectors")

    def query(
        self,
        query_vec: np.ndarray,
        k: int = 5,
        boost_map: dict[str, float] | None = None,
        query_text: str | None = None,
        candidate_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Find top-K most similar vectors using cosine similarity.

        Uses chunked processing for memory efficiency on large stores.

        Args:
            query_vec: Query vector [dim]
            k: Number of results to return
            boost_map: Optional name->boost mapping for result reranking

        Returns:
            List of result dicts with score, rank, and metadata
        """
        if self._count == 0:
            return []

        # Normalize query
        q = query_vec.flatten().astype(np.float32)
        if q.shape[0] != self.dim:
            if q.shape[0] < self.dim:
                q = np.pad(q, (0, self.dim - q.shape[0]))
            else:
                q = q[: self.dim]

        q_norm = np.linalg.norm(q)
        if q_norm < 1e-9:
            return []
        q = q / q_norm

        candidate_indices = self._indices_for_entity_ids(candidate_ids)
        if candidate_indices is None or candidate_indices.size == 0:
            candidate_indices = self._candidate_indices(query_text, k)
        if candidate_indices is None or candidate_indices.size == 0:
            score_indices = np.arange(self._count, dtype=np.int64)
        else:
            score_indices = candidate_indices

        # Chunked similarity computation for memory efficiency
        CHUNK_SIZE = 10000
        all_scores = np.zeros(score_indices.shape[0], dtype=np.float32)

        for start in range(0, score_indices.shape[0], CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, score_indices.shape[0])
            chunk_indices = score_indices[start:end]
            chunk = self._vectors[chunk_indices]

            # Compute norms
            chunk_norms = np.linalg.norm(chunk, axis=1)
            chunk_norms = np.maximum(chunk_norms, 1e-9)

            # Cosine similarity
            all_scores[start:end] = np.dot(chunk, q) / chunk_norms

        # Apply boost map
        if boost_map:
            for i, idx in enumerate(score_indices):
                meta = self._metadata[int(idx)]
                name = meta.get("name")
                if name and name in boost_map:
                    all_scores[i] += boost_map[name] * 0.2

        # Step 3: Refutation Layer (Literal Boosting)
        # If query contains high-entropy exact match tokens, boost results containing them.
        if query_text and len(query_text) > 3:
            literal_tokens = set(re.findall(r"[A-Za-z0-9_]{4,}", query_text))

            if literal_tokens:
                for i, idx in enumerate(score_indices):
                    meta = self._metadata[int(idx)]
                    name = meta.get("name", "")
                    file = meta.get("file", "")

                    # Check for literal matches in name or file path
                    match_score = 0
                    for token in literal_tokens:
                        if token in name:
                            match_score += 0.5
                        elif token in file:
                            match_score += 0.2

                    if match_score > 0:
                        # Cap boost to prevent completely overriding semantic signal
                        all_scores[i] *= 1.0 + min(match_score, 1.0)

        # Top-K
        k = min(k, score_indices.shape[0])
        top_indices = np.argpartition(all_scores, -k)[-k:]
        top_indices = top_indices[np.argsort(all_scores[top_indices])[::-1]]

        # Build results
        results = []
        for rank, idx in enumerate(top_indices):
            store_idx = int(score_indices[int(idx)])
            res = self._metadata[store_idx].copy()
            score = float(all_scores[idx])
            res["score"] = score
            res["rank"] = rank + 1
            res["candidate_pool"] = int(score_indices.shape[0])
            res["cpu_prefiltered"] = candidate_indices is not None
            res.setdefault("symbol_terms", list(res.get("symbol_terms", []) or []))
            res.setdefault("path_terms", list(res.get("path_terms", []) or []))
            res.setdefault("doc_terms", list(res.get("doc_terms", []) or []))
            res.setdefault("chunk_role", res.get("chunk_role"))
            res.setdefault("file_role", res.get("file_role"))
            res.setdefault("parent_symbol", res.get("parent_symbol"))

            # Explainability
            reasons = []
            if score > 0.8:
                reasons.append("High semantic similarity match.")
            elif score > 0.5:
                reasons.append("Moderate similarity; likely contextually relevant.")
            else:
                reasons.append("Low confidence match; potential conceptual overlap.")

            entity_type = res.get("type", "unknown")
            if entity_type == "file":
                reasons.append("Core module match.")
            elif entity_type == "class":
                reasons.append("Structural definition match.")
            elif entity_type == "function":
                reasons.append("Functional logic match.")

            file_path = res.get("file", "")
            if "tests" in file_path:
                reasons.append("Provides usage examples via tests.")
            elif "docs" in file_path:
                reasons.append("Documentation source.")

            res["reason"] = " ".join(reasons)
            results.append(res)

        return results

    def clear(self) -> None:
        """Clear all vectors and metadata."""
        if self.read_only:
            raise RuntimeError("Cannot clear in read-only mode")

        with self._write_lock:
            self._count = 0
            self._metadata = []
            self._lookup = {}
            self._entity_lookup = {}
            self._term_index = {}
            self._indexes_dirty = False
            self.save()

    def _validate_state(self) -> None:
        if self._count < 0:
            raise SaguaroStateCorruptionError("Vector store count cannot be negative.")
        if self._capacity < self._count:
            raise SaguaroStateCorruptionError(
                f"Vector store count {self._count} exceeds capacity {self._capacity}."
            )
        if len(self._metadata) < self._count:
            raise SaguaroStateCorruptionError(
                f"Vector metadata length {len(self._metadata)} is smaller than count {self._count}."
            )
        if os.path.exists(self.vectors_path):
            expected_size = self._capacity * self.dim * np.dtype(np.float32).itemsize
            actual_size = os.path.getsize(self.vectors_path)
            if actual_size != expected_size:
                raise SaguaroStateCorruptionError(
                    f"Vector store size mismatch for {self.vectors_path}: "
                    f"expected {expected_size}, got {actual_size}."
                )

    def __len__(self) -> int:
        """Return number of stored vectors."""
        return self._count

    def close(self) -> None:
        """Close the memory-mapped file."""
        if self._vectors is not None:
            del self._vectors
            self._vectors = None


# Alias for backward compatibility
VectorStore = MemoryMappedVectorStore
