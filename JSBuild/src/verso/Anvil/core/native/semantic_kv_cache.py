"""
Semantic KV Cache with holographic compression for 1M+ token contexts.

Hybrid architecture:
- Recent 8K tokens: Full resolution in IncrementalKVCache
- Distant context: Compressed into 64 semantic crystals (4096-dim embeddings)
- Retrieval: Semantic similarity matching for context injection
"""

import numpy as np
from typing import Optional, Tuple
import tensorflow as tf

from core.native.incremental_kv_cache import IncrementalKVCache

try:
    from saguaro.native.ops.fused_coconut_ops import (
        fused_coconut_crystallize,
        fused_coconut_retrieve,
        fused_coconut_bfs_available,
    )

    SAGUARO_OPS_AVAILABLE = fused_coconut_bfs_available()
except ImportError:
    SAGUARO_OPS_AVAILABLE = False
    print(
        "Warning: Saguaro ops not available. SemanticKVCache will fallback to IncrementalKVCache."
    )


class SemanticKVCache:
    """
    Hybrid KV cache with semantic compression for massive contexts.

    Architecture:
        [Prompt] → [Crystal Store: 64 slots × 4096 dim]
                       ↓ compress (distant context)
                    [Recent Cache: 8K tokens]
                       ↓ generation
                    [New Tokens]

    Compression triggers every 16K tokens, compressing context beyond
    the recent window into semantic crystals.
    """

    def __init__(
        self,
        llm_ctx,
        max_seq_len: int = 32768,
        llm_obj=None,
        recent_window: int = 8192,
        max_crystals: int = 64,
        embedding_dim: int = 4096,
        compression_interval: int = 16384,
        crystallize_threshold: float = 0.85,
    ):
        """
        Initialize semantic KV cache.

        Args:
            llm_ctx: llama.cpp context object
            max_seq_len: Maximum sequence length for recent cache
            llm_obj: Full Llama object for resets
            recent_window: Number of recent tokens to keep uncompressed
            max_crystals: Maximum number of semantic crystals
            embedding_dim: Dimension of embeddings (must match model)
            compression_interval: Compress every N tokens
            crystallize_threshold: Minimum confidence to store crystal (0.0-1.0)
        """
        # Recent cache for uncompressed tokens
        self.recent_cache = IncrementalKVCache(llm_ctx, recent_window, llm_obj)

        # Semantic crystal store
        self.max_crystals = max_crystals
        self.embedding_dim = embedding_dim
        self.crystal_store = np.zeros((max_crystals, embedding_dim), dtype=np.float32)
        self.crystal_ages = np.zeros(max_crystals, dtype=np.int32)
        self.crystal_valid = np.zeros(max_crystals, dtype=np.bool_)
        self.compression_interval = compression_interval
        self.crystallize_threshold = crystallize_threshold

        # State tracking
        self.total_tokens_processed = 0
        self.last_compression_at = 0
        self.num_compressions = 0

        # Fallback mode if Saguaro ops unavailable
        self.saguaro_enabled = SAGUARO_OPS_AVAILABLE

        if not self.saguaro_enabled:
            print("Info: SemanticKVCache operating in fallback mode (no compression)")

    def reset(self):
        """Full reset - clear both recent cache and crystals."""
        self.recent_cache.reset()
        self.crystal_store.fill(0.0)
        self.crystal_ages.fill(0)
        self.crystal_valid.fill(False)
        self.total_tokens_processed = 0
        self.last_compression_at = 0
        self.num_compressions = 0

    def _extract_kv_embeddings(
        self, prompt_tokens: list, start: int, end: int, vocab_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Extract embeddings for tokens in range [start:end].

        This is a proxy for actual KV cache states - uses token embeddings
        as a simplified representation.

        Args:
            prompt_tokens: Full prompt token IDs
            start: Start index
            end: End index
            vocab_embeddings: Vocabulary embedding matrix [vocab_size, dim]

        Returns:
            Embeddings [end-start, dim]
        """
        token_slice = prompt_tokens[start:end]
        if len(token_slice) == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        # Get embeddings for tokens
        embeddings = vocab_embeddings[token_slice]
        return embeddings.astype(np.float32)

    def _compute_confidence(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute confidence scores for embeddings.

        Uses L2 norm as a simple confidence proxy. In production,
        this could use attention scores or model confidence.

        Args:
            embeddings: [num_tokens, dim]

        Returns:
            Confidence scores [num_tokens]
        """
        if len(embeddings) == 0:
            return np.array([], dtype=np.float32)

        # L2 norm as confidence proxy
        norms = np.linalg.norm(embeddings, axis=1)
        # Normalize to [0, 1]
        if norms.max() > 0:
            confidence = norms / norms.max()
        else:
            confidence = np.zeros(len(embeddings), dtype=np.float32)

        return confidence.astype(np.float32)

    def compress_distant_context(
        self,
        prompt_tokens: list,
        vocab_embeddings: np.ndarray,
        compress_before: int,
    ) -> int:
        """
        Compress tokens before compress_before index into crystals.

        Args:
            prompt_tokens: Full prompt token IDs
            vocab_embeddings: Vocabulary embeddings [vocab_size, dim]
            compress_before: Compress tokens [0:compress_before]

        Returns:
            Number of tokens compressed
        """
        if not self.saguaro_enabled:
            # Fallback: just track state without compression
            return 0

        if compress_before <= 0:
            return 0

        try:
            # Extract embeddings for distant context
            thought_paths = self._extract_kv_embeddings(
                prompt_tokens, 0, compress_before, vocab_embeddings
            )

            if len(thought_paths) == 0:
                return 0

            # Compute confidence for each token
            confidence_scores = self._compute_confidence(thought_paths)

            # Aggregate into single summary (mean pooling)
            # In production, could use attention-weighted pooling
            thought_summary = thought_paths.mean(axis=0, keepdims=True)  # [1, dim]
            summary_confidence = confidence_scores.mean(keepdims=True)  # [1]

            # Crystallize using Saguaro op
            updated_store, updated_ages, crystal_indices = fused_coconut_crystallize(
                thought_path=tf.constant(thought_summary, dtype=tf.float32),
                confidence=tf.constant(summary_confidence, dtype=tf.float32),
                crystal_store=tf.constant(self.crystal_store, dtype=tf.float32),
                crystal_ages=tf.constant(self.crystal_ages, dtype=tf.int32),
                crystallize_threshold=self.crystallize_threshold,
                max_crystals=self.max_crystals,
            )

            # Update state
            self.crystal_store = updated_store.numpy()
            self.crystal_ages = updated_ages.numpy()
            stored_idx = crystal_indices.numpy()[0]

            if stored_idx >= 0:
                self.crystal_valid[stored_idx] = True
                self.num_compressions += 1
                print(
                    f"Info: Compressed {compress_before} tokens into crystal #{stored_idx} "
                    f"(confidence: {summary_confidence[0]:.3f})"
                )

            return compress_before

        except Exception as e:
            print(f"Warning: Compression failed: {e}. Skipping compression.")
            return 0

    def retrieve_relevant_context(
        self, query_tokens: list, vocab_embeddings: np.ndarray, top_k: int = 3
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Retrieve relevant crystals for query.

        Args:
            query_tokens: Recent token IDs to query against
            vocab_embeddings: Vocabulary embeddings
            top_k: Number of crystals to retrieve

        Returns:
            Tuple of (retrieved_embedding, similarity_score)
        """
        if not self.saguaro_enabled or not self.crystal_valid.any():
            return None, 0.0

        try:
            # Create query embedding (mean of recent tokens)
            if len(query_tokens) == 0:
                return None, 0.0

            query_embs = vocab_embeddings[query_tokens[-min(32, len(query_tokens)) :]]
            query_summary = query_embs.mean(axis=0, keepdims=True)  # [1, dim]

            # Retrieve from crystal store
            retrieved, similarity = fused_coconut_retrieve(
                query=tf.constant(query_summary, dtype=tf.float32),
                crystal_store=tf.constant(self.crystal_store, dtype=tf.float32),
                crystal_valid=tf.constant(self.crystal_valid, dtype=tf.bool),
                top_k=min(top_k, self.crystal_valid.sum()),
            )

            retrieved_np = retrieved.numpy()[0]  # [dim]
            similarity_score = similarity.numpy()[0]

            if similarity_score > 0.5:  # Meaningful match
                print(
                    f"Info: Retrieved semantic context (similarity: {similarity_score:.3f})"
                )
                return retrieved_np, float(similarity_score)

            return None, 0.0

        except Exception as e:
            print(f"Warning: Retrieval failed: {e}")
            return None, 0.0

    def prepare_for_generation(
        self,
        prompt_tokens: list,
        vocab_embeddings: Optional[np.ndarray] = None,
        allow_reuse: bool = True,
    ) -> int:
        """
        Prepare cache for generation with semantic compression.

        Args:
            prompt_tokens: Token IDs to generate from
            vocab_embeddings: Vocabulary embeddings for compression (optional)
            allow_reuse: Allow KV cache reuse

        Returns:
            start_pos: Position to start decoding from
        """
        self.total_tokens_processed += len(prompt_tokens)

        # Check if compression is needed
        should_compress = (
            self.saguaro_enabled
            and vocab_embeddings is not None
            and (self.total_tokens_processed - self.last_compression_at)
            >= self.compression_interval
            and len(prompt_tokens) > self.recent_cache.max_seq_len
        )

        if should_compress:
            # Compress tokens beyond recent window
            compress_before = len(prompt_tokens) - self.recent_cache.max_seq_len
            num_compressed = self.compress_distant_context(
                prompt_tokens, vocab_embeddings, compress_before
            )

            if num_compressed > 0:
                self.last_compression_at = self.total_tokens_processed
                # Use only recent tokens for KV cache
                prompt_tokens = prompt_tokens[-self.recent_cache.max_seq_len :]

        # Prepare recent cache with (potentially trimmed) prompt
        start_pos = self.recent_cache.prepare_for_generation(prompt_tokens, allow_reuse)

        return start_pos

    def advance_position(self, n_tokens: int = 1, token_ids: Optional[list] = None):
        """
        Update position after generating tokens.

        Args:
            n_tokens: Number of tokens generated
            token_ids: Optional token IDs for tracking
        """
        self.recent_cache.advance_position(n_tokens, token_ids)
        self.total_tokens_processed += n_tokens

    def get_stats(self) -> dict:
        """Get cache statistics."""
        num_valid_crystals = self.crystal_valid.sum()
        return {
            "total_tokens_processed": self.total_tokens_processed,
            "num_compressions": self.num_compressions,
            "num_valid_crystals": int(num_valid_crystals),
            "max_crystals": self.max_crystals,
            "recent_cache_pos": self.recent_cache.current_pos,
            "recent_cache_valid": self.recent_cache.valid_until,
            "compression_enabled": self.saguaro_enabled,
        }


# Backward compatibility alias
SlidingWindowCache = SemanticKVCache
