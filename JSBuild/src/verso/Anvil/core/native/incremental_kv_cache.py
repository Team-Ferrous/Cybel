"""
Incremental KV Cache for llama.cpp with position tracking.
Eliminates costly full cache resets between generations.
"""

from typing import Optional

from core.native.llama_kv_compat import (
    kv_cache_clear,
    kv_cache_seq_add,
    kv_cache_seq_rm,
)


class IncrementalKVCache:
    """
    Manages KV cache state across multiple generations without full resets.
    Uses position offsets to track where attention states are valid.
    """

    def __init__(self, llm_ctx, max_seq_len: int = 32768, llm_obj=None):
        self.ctx = llm_ctx
        self.llm = llm_obj  # Full Llama object for proper resets
        self.max_seq_len = max_seq_len
        self.current_pos = 0  # Current position in cache
        self.valid_until = 0  # Positions [0, valid_until) contain valid data
        self.cached_tokens = []  # Track tokens in cache for prefix matching

    def reset(self):
        """Full reset - only when starting completely new conversation."""
        # CRITICAL FIX: Use llm.reset() instead of direct kv_cache_clear()
        # Direct kv_cache_clear() doesn't reset n_past counter, causing llama_decode -1 errors
        if self.llm:
            # Proper reset that clears cache AND resets internal position counter
            self.llm.reset()
        else:
            # Fallback for legacy code - but this can cause decode errors
            kv_cache_clear(self.ctx)

        self.current_pos = 0
        self.valid_until = 0
        self.cached_tokens = []  # Clear cached token tracking

    def trim_to(self, keep_tokens: int):
        """
        Trim cache to keep only the last N tokens.
        Useful for implementing sliding window attention.
        """
        if keep_tokens >= self.valid_until:
            return  # Nothing to trim

        # Remove tokens before position (valid_until - keep_tokens)
        remove_from = 0
        remove_to = self.valid_until - keep_tokens

        kv_cache_seq_rm(self.ctx, 0, remove_from, remove_to)

        # Shift remaining positions
        kv_cache_seq_add(
            self.ctx,
            0,  # seq_id
            remove_to,  # p0 (start of shift range)
            self.valid_until,  # p1 (end of shift range)
            -remove_to,  # delta (shift left by remove_to positions)
        )

        self.current_pos -= remove_to
        self.valid_until = keep_tokens

    def _find_common_prefix(self, cached: list, new_tokens: list) -> int:
        """
        Find the longest common prefix between cached tokens and new prompt.

        Args:
            cached: Previously cached token IDs
            new_tokens: New prompt token IDs

        Returns:
            Length of common prefix
        """
        min_len = min(len(cached), len(new_tokens))
        for i in range(min_len):
            if cached[i] != new_tokens[i]:
                return i
        return min_len

    def prepare_for_generation(
        self, prompt_tokens: list, allow_reuse: bool = True
    ) -> int:
        """
        Prepare cache for new generation.

        Args:
            prompt_tokens: Token IDs to generate from
            allow_reuse: If True, reuse existing cache when prompt is a prefix

        Returns:
            start_pos: Position to start decoding from (0 if full prompt, >0 if cached)
        """
        prompt_len = len(prompt_tokens)

        # OPTIMIZATION: Prefix caching for multi-turn conversations
        # Find common prefix between cached tokens and new prompt
        prefix_len = 0
        if allow_reuse and self.valid_until > 0 and len(self.cached_tokens) > 0:
            prefix_len = self._find_common_prefix(self.cached_tokens, prompt_tokens)

        if prefix_len > 0:
            # Reuse existing cache up to prefix_len
            # Only process tokens after the prefix
            if prefix_len < len(prompt_tokens):
                # Prompt extends cached prefix - keep cache and process extension
                self.current_pos = prefix_len
                self.valid_until = prefix_len
                # Update cached tokens to include full prompt
                self.cached_tokens = list(prompt_tokens)
                return prefix_len  # Start decoding from this position
            else:
                # Prompt is exact prefix of cache (e.g., retry/regeneration)
                self.current_pos = prefix_len
                self.valid_until = max(self.valid_until, prefix_len)
                self.cached_tokens = list(prompt_tokens)
                return prefix_len
        else:
            # No common prefix - reset cache for new conversation
            self.reset()

        # If prompt exceeds max length, trim from beginning (sliding window)
        if prompt_len > self.max_seq_len:
            overflow = prompt_len - self.max_seq_len
            prompt_tokens = prompt_tokens[overflow:]
            prompt_len = len(prompt_tokens)
            # Reset cache since we're dropping context
            self.reset()

        # Update position tracking and cache full prompt
        self.current_pos = prompt_len
        self.valid_until = prompt_len
        self.cached_tokens = list(prompt_tokens)

        return 0  # Start from beginning (no cache reuse)

    def advance_position(self, n_tokens: int = 1, token_ids: Optional[list] = None):
        """
        Update position after generating tokens.

        Args:
            n_tokens: Number of tokens generated
            token_ids: Optional list of generated token IDs to track in cache
        """
        self.current_pos += n_tokens
        self.valid_until = max(self.valid_until, self.current_pos)

        # Track generated tokens for prefix matching
        if token_ids is not None:
            self.cached_tokens.extend(token_ids)

        # Check for overflow
        if self.current_pos >= self.max_seq_len:
            # Trigger sliding window
            keep_tokens = self.max_seq_len // 2  # Keep last half
            self.trim_to(keep_tokens)


class SlidingWindowCache:
    """
    Implements fixed-size sliding window over KV cache.
    Memory-efficient for very long sequences.
    """

    def __init__(self, llm_ctx, window_size: int = 16384, llm_obj=None):
        self.ctx = llm_ctx
        self.llm = llm_obj  # Full Llama object for proper resets
        self.window_size = window_size
        self.total_processed = 0

    def update_with_overflow(self):
        """
        Called when cache is full. Keeps last window_size tokens,
        drops everything before.
        """
        # Remove first half of window
        remove_count = self.window_size // 2

        kv_cache_seq_rm(self.ctx, 0, 0, remove_count)

        # Shift remaining positions
        kv_cache_seq_add(
            self.ctx, 0, remove_count, self.window_size, -remove_count
        )

        self.total_processed += remove_count

        return self.window_size - remove_count  # New valid length


# Integration example for NativeInferenceEngine
"""
Modified generate() in engine.py:

def generate(self, prompt_tokens, max_new_tokens=20, ...):
    # BEFORE: self.reset_kv_cache()  # <-- REMOVE THIS

    # AFTER: Use incremental cache
    if not hasattr(self, 'kv_cache'):
        self.kv_cache = IncrementalKVCache(self.llm._ctx, self.context_length)

    # Prepare cache (reuses existing state when possible)
    start_pos = self.kv_cache.prepare_for_generation(prompt_tokens)

    # Generate tokens
    for token in self.llm.generate(prompt_tokens, ...):
        yield token
        self.kv_cache.advance_position(1)

        # Handle overflow with sliding window
        if self.kv_cache.current_pos >= self.context_length:
            self.kv_cache.trim_to(self.context_length // 2)
"""
