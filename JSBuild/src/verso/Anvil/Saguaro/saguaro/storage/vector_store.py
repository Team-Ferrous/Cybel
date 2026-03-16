"""SAGUARO Vector Store
Simple storage backend for Hyperdimensional Vectors.

NOTE: This module now re-exports from memmap_vector_store for new indexes.
The original pickle-based class is renamed to LegacyVectorStore for backward compatibility.
"""

import json
import logging
import os
import pickle
from typing import Any

from saguaro.errors import SaguaroStateCorruptionError
from saguaro.storage.atomic_fs import atomic_write_json

logger = logging.getLogger(__name__)


class LegacyVectorStore:
    """Store vectors using the legacy pickle-backed format."""

    def __init__(self, storage_path: str, dim: int, dark_space_ratio: float = 0.4) -> None:
        """Initialize the legacy vector store."""
        self.storage_path = storage_path
        self.dim = dim
        self.dark_space_ratio = dark_space_ratio

        self.index_path = os.path.join(storage_path, "index.pkl")
        self.metadata_path = os.path.join(storage_path, "metadata.json")

        self.vectors = []
        self.metadata = []

        self._load()

    def _load(self) -> None:
        """Load vectors and metadata from disk with validation.

        Performs dimension validation to prevent inhomogeneous shape errors.
        Corrupt or mismatched entries are filtered out with warnings.
        """
        raw_vectors = []
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "rb") as f:
                    raw_vectors = pickle.load(f)
            except (pickle.UnpicklingError, EOFError) as e:
                raise SaguaroStateCorruptionError(
                    f"Failed to load legacy index {self.index_path}: {e}"
                ) from e

        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path) as f:
                    self.metadata = json.load(f)
            except json.JSONDecodeError as e:
                raise SaguaroStateCorruptionError(
                    f"Failed to load legacy metadata {self.metadata_path}: {e}"
                ) from e
        else:
            self.metadata = []

        # Validate and filter vectors to ensure homogeneous dimensions
        valid_vectors = []
        valid_metadata = []
        invalid_count = 0

        import numpy as np

        for i, vec in enumerate(raw_vectors):
            try:
                vec_array = np.asarray(vec, dtype=np.float32)
                if vec_array.ndim == 1 and vec_array.shape[0] == self.dim:
                    valid_vectors.append(vec_array)
                    if i < len(self.metadata):
                        valid_metadata.append(self.metadata[i])
                    else:
                        valid_metadata.append(
                            {"name": f"unknown_{i}", "type": "unknown"}
                        )
                else:
                    invalid_count += 1
                    logger.debug(
                        f"Filtered vector {i}: shape {vec_array.shape} != expected ({self.dim},)"
                    )
            except (ValueError, TypeError) as e:
                invalid_count += 1
                logger.debug(f"Filtered invalid vector {i}: {e}")

        if invalid_count > 0:
            logger.warning(
                f"Filtered {invalid_count} vectors with inconsistent dimensions. "
                f"Valid: {len(valid_vectors)}, Original: {len(raw_vectors)}"
            )

        self.vectors = valid_vectors
        self.metadata = valid_metadata

    def save(self) -> None:
        """Persist vectors and metadata to disk."""
        os.makedirs(self.storage_path, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.vectors, f)
        atomic_write_json(self.metadata_path, self.metadata, indent=2, sort_keys=False)

    def add(self, vector: Any, meta: dict[str, Any]) -> None:
        """Adds a vector to the store."""
        import numpy as np

        # Ensure vector matches dim
        if vector.shape[0] != self.dim:
            # Pad or truncate (shouldn't happen if pipeline is correct)
            if vector.shape[0] < self.dim:
                vector = np.pad(vector, (0, self.dim - vector.shape[0]))
            else:
                vector = vector[: self.dim]

        self.vectors.append(vector)
        self.metadata.append(meta)

    def remove_file(self, file_path: str) -> int:
        """Remove all vectors associated with a file."""
        keep_vectors = []
        keep_metadata = []
        removed = 0
        for vec, meta in zip(self.vectors, self.metadata):
            if meta.get("file") == file_path:
                removed += 1
                continue
            keep_vectors.append(vec)
            keep_metadata.append(meta)
        if removed:
            self.vectors = keep_vectors
            self.metadata = keep_metadata
        return removed

    def query(
        self, query_vec: Any, k: int = 5, boost_map: dict[str, float] = None
    ) -> list[dict[str, Any]]:
        """Naive linear scan for prototype.
        Returns top-K results with explanations.
        """
        if not self.vectors:
            return []

        import numpy as np

        # Cosine similarity
        # query_vec: [D]
        # database: [N, D]

        db = np.array(self.vectors)  # [N, D]

        # Norms
        q_norm = np.linalg.norm(query_vec)
        db_norm = np.linalg.norm(db, axis=1)

        scores = np.dot(db, query_vec) / (db_norm * q_norm + 1e-9)

        if boost_map:
            # Apply graph-based boosting
            for i, meta in enumerate(self.metadata):
                name = meta.get("name")
                if name and name in boost_map:
                    scores[i] += (
                        boost_map[name] * 0.2
                    )  # Weighted boost using 0.2 factor

        # Top K
        indices = np.argsort(scores)[::-1][:k]

        results = []
        for rank, idx in enumerate(indices):
            res = self.metadata[idx].copy()
            score = float(scores[idx])
            res["score"] = score

            # --- Explainability Layer ---
            # Basic heuristics to explain WHY this was retrieved
            reasons = []

            # 1. Score-based Confidence
            if score > 0.8:
                reasons.append("High semantic similarity match.")
            elif score > 0.5:
                reasons.append("Moderate similarity; likely contextually relevant.")
            else:
                reasons.append("Low confidence match; potential conceptual overlap.")

            # 2. Type-based Context
            entity_type = res.get("type", "unknown")
            if entity_type == "file":
                reasons.append("Core module match.")
            elif entity_type == "class":
                reasons.append("Structural definition match.")
            elif entity_type == "function":
                reasons.append("Functional logic match.")

            # 3. Path heuristic
            file_path = res.get("file", "")
            if "tests" in file_path:
                reasons.append("Provides usage examples via tests.")
            elif "docs" in file_path:
                reasons.append("Documentation source.")

            res["reason"] = " ".join(reasons)
            res["rank"] = rank + 1

            results.append(res)

        return results

    def clear(self) -> None:
        """Clear all stored vectors and metadata."""
        self.vectors = []
        self.metadata = []
        self.save()

    def __len__(self) -> int:
        """Return number of stored vectors."""
        return len(self.vectors)


# =============================================================================
# AUTO-DETECT FORMAT AND USE APPROPRIATE IMPLEMENTATION
# =============================================================================


def VectorStore(
    storage_path: str, dim: int, dark_space_ratio: float = 0.4, **kwargs: Any
) -> Any:
    """Return the authoritative vector-store implementation.

    The stabilization cut keeps native mmap stores as the canonical format and
    rejects legacy pickle stores rather than silently loading them.
    """

    index_meta_path = os.path.join(storage_path, "index_meta.json")
    legacy_index_path = os.path.join(storage_path, "index.pkl")

    if os.path.exists(index_meta_path):
        try:
            with open(index_meta_path) as f:
                meta = json.load(f)
            fmt = str(meta.get("format") or "").strip().lower()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read vector-store metadata for {storage_path}: {exc}"
            ) from exc
        if fmt != "native_mmap":
            raise RuntimeError(
                "Saguaro requires native_mmap vector-store format. "
                f"Found unsupported format '{fmt or 'unknown'}' in {index_meta_path}."
            )

    from saguaro.storage.native_vector_store import NativeMemoryMappedVectorStore

    if os.path.exists(legacy_index_path):
        raise RuntimeError(
            "Legacy pickle vector stores are not supported by the authoritative "
            f"Saguaro runtime: {legacy_index_path}"
        )

    logger.debug("Using NativeMemoryMappedVectorStore for %s", storage_path)
    active_dim = kwargs.pop("active_dim", None)
    total_dim = kwargs.pop("total_dim", None)
    return NativeMemoryMappedVectorStore(
        storage_path,
        dim,
        dark_space_ratio,
        active_dim=active_dim,
        total_dim=total_dim,
        **kwargs,
    )
