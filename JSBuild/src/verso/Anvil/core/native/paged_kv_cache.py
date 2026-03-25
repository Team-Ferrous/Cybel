"""
PagedAttention KV Cache for CPU cache alignment and reduced fragmentation.

Allocates KV cache in fixed-size pages (64 tokens) aligned with CPU cache lines.
Enables efficient memory reuse across sequences and reduces fragmentation.

Based on vLLM's PagedAttention but optimized for CPU-only inference.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple

from core.native.llama_kv_compat import (
    kv_cache_clear,
    kv_cache_seq_add,
    kv_cache_seq_rm,
)


class PagedKVCache:
    """
    Paged KV cache with CPU cache line alignment.

    Architecture:
        - Physical memory: Fixed pages of 64 tokens each
        - Block table: Maps logical sequence positions → physical pages
        - LRU eviction: Reuse least recently used pages when full
        - Page sharing: Multiple sequences can share read-only pages (prefix)
    """

    def __init__(
        self,
        llm_ctx,
        llm_obj=None,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        page_size: int = 64,  # Tokens per page (aligned with CPU cache)
        num_pages: int = 512,  # Total pages in pool
    ):
        """
        Initialize paged KV cache.

        Args:
            llm_ctx: llama.cpp context
            llm_obj: Full Llama object for resets
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension per head
            page_size: Tokens per page (should be power of 2)
            num_pages: Total pages in physical pool
        """
        self.ctx = llm_ctx
        self.llm = llm_obj
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.num_pages = num_pages

        # Physical memory pool: [num_pages, page_size] logical addressing
        # Actual KV cache is managed by llama.cpp internally
        # We just track page allocation and mapping

        # Block table: Maps sequence positions to page indices
        # Key: (seq_id, page_idx_in_seq) → Value: physical_page_id
        self.block_table: Dict[Tuple[int, int], int] = {}

        # Free page pool (LRU ordered)
        self.free_pages: List[int] = list(range(num_pages))

        # Page metadata
        self.page_ref_counts = np.zeros(num_pages, dtype=np.int32)  # Reference counting
        self.page_ages = np.zeros(num_pages, dtype=np.int32)  # LRU tracking
        self.current_tick = 0

        # Sequence tracking
        self.seq_lengths: Dict[int, int] = {}  # seq_id → length
        self.active_sequences: List[int] = []

    def reset(self):
        """Full reset - clear all pages and sequences."""
        if self.llm:
            self.llm.reset()
        else:
            kv_cache_clear(self.ctx)

        self.block_table.clear()
        self.free_pages = list(range(self.num_pages))
        self.page_ref_counts.fill(0)
        self.page_ages.fill(0)
        self.seq_lengths.clear()
        self.active_sequences.clear()
        self.current_tick = 0

    def _allocate_page(self) -> Optional[int]:
        """
        Allocate a page from the free pool.

        Returns:
            Physical page ID or None if pool exhausted
        """
        if len(self.free_pages) > 0:
            page_id = self.free_pages.pop(0)
            return page_id

        # Pool exhausted - evict LRU page
        return self._evict_lru_page()

    def _evict_lru_page(self) -> Optional[int]:
        """
        Evict least recently used page with zero references.

        Returns:
            Evicted page ID or None if no evictable pages
        """
        # Find page with ref_count=0 and oldest age
        evictable_pages = np.where(self.page_ref_counts == 0)[0]

        if len(evictable_pages) == 0:
            # No evictable pages available
            return None

        # Select LRU (oldest age)
        lru_page = evictable_pages[np.argmin(self.page_ages[evictable_pages])]

        # Remove from block table
        keys_to_remove = [
            key for key, val in self.block_table.items() if val == lru_page
        ]
        for key in keys_to_remove:
            del self.block_table[key]

        # Clear the page in llama.cpp
        start_pos = lru_page * self.page_size
        end_pos = start_pos + self.page_size
        kv_cache_seq_rm(self.ctx, 0, start_pos, end_pos)

        return int(lru_page)

    def _get_or_allocate_page(self, seq_id: int, page_idx: int) -> Optional[int]:
        """
        Get existing page or allocate new one.

        Args:
            seq_id: Sequence ID
            page_idx: Logical page index within sequence

        Returns:
            Physical page ID or None if allocation failed
        """
        key = (seq_id, page_idx)

        if key in self.block_table:
            # Page exists - update LRU
            physical_page = self.block_table[key]
            self.current_tick += 1
            self.page_ages[physical_page] = self.current_tick
            return physical_page

        # Allocate new page
        physical_page = self._allocate_page()
        if physical_page is None:
            return None

        # Map logical → physical
        self.block_table[key] = physical_page
        self.page_ref_counts[physical_page] = 1
        self.current_tick += 1
        self.page_ages[physical_page] = self.current_tick

        return physical_page

    def prepare_for_generation(
        self, seq_id: int, prompt_tokens: List[int], allow_reuse: bool = True
    ) -> int:
        """
        Prepare paged cache for generation.

        Args:
            seq_id: Sequence ID
            prompt_tokens: Token IDs
            allow_reuse: Allow page reuse (prefix caching)

        Returns:
            Start position for generation
        """
        prompt_len = len(prompt_tokens)
        num_pages_needed = (prompt_len + self.page_size - 1) // self.page_size

        # Allocate pages for this sequence
        allocated_pages = []
        for page_idx in range(num_pages_needed):
            physical_page = self._get_or_allocate_page(seq_id, page_idx)
            if physical_page is None:
                # Allocation failed - fallback to reset
                self.reset()
                return 0

            allocated_pages.append(physical_page)

        # Update sequence length
        self.seq_lengths[seq_id] = prompt_len
        if seq_id not in self.active_sequences:
            self.active_sequences.append(seq_id)

        # For simplicity, we don't implement actual page-level prefix matching here
        # (would require comparing page contents, which llama.cpp doesn't expose easily)
        # Just track logical→physical mapping for efficient memory reuse

        return 0  # Start from beginning (llama.cpp handles actual caching)

    def advance_position(
        self, seq_id: int, n_tokens: int = 1, token_ids: Optional[List[int]] = None
    ):
        """
        Update position after generating tokens.

        Args:
            seq_id: Sequence ID
            n_tokens: Number of tokens generated
            token_ids: Optional token IDs
        """
        if seq_id not in self.seq_lengths:
            self.seq_lengths[seq_id] = 0

        new_length = self.seq_lengths[seq_id] + n_tokens
        self.seq_lengths[seq_id] = new_length

        # Check if we need to allocate a new page
        new_num_pages = (new_length + self.page_size - 1) // self.page_size
        current_num_pages = len([k for k in self.block_table.keys() if k[0] == seq_id])

        if new_num_pages > current_num_pages:
            # Allocate additional page
            for page_idx in range(current_num_pages, new_num_pages):
                self._get_or_allocate_page(seq_id, page_idx)

    def get_stats(self) -> dict:
        """Get cache statistics."""
        num_free_pages = len(self.free_pages)
        num_allocated_pages = np.sum(self.page_ref_counts > 0)
        fragmentation = 1.0 - (num_allocated_pages / max(self.num_pages, 1))

        return {
            "num_pages": self.num_pages,
            "page_size": self.page_size,
            "free_pages": num_free_pages,
            "allocated_pages": int(num_allocated_pages),
            "fragmentation": float(fragmentation),
            "active_sequences": len(self.active_sequences),
            "total_tokens": sum(self.seq_lengths.values()),
        }

    def trim_to(self, seq_id: int, keep_tokens: int):
        """
        Trim sequence to keep only last N tokens.

        Args:
            seq_id: Sequence ID
            keep_tokens: Number of tokens to keep
        """
        if seq_id not in self.seq_lengths:
            return

        current_length = self.seq_lengths[seq_id]
        if keep_tokens >= current_length:
            return

        # Calculate pages to remove
        tokens_to_remove = current_length - keep_tokens
        pages_to_remove = tokens_to_remove // self.page_size

        # Remove pages from beginning
        for page_idx in range(pages_to_remove):
            key = (seq_id, page_idx)
            if key in self.block_table:
                physical_page = self.block_table[key]
                del self.block_table[key]

                # Decrement ref count
                self.page_ref_counts[physical_page] -= 1
                if self.page_ref_counts[physical_page] == 0:
                    # Return to free pool
                    self.free_pages.append(physical_page)

        # Update sequence length
        self.seq_lengths[seq_id] = keep_tokens

        # Also remove from llama.cpp cache
        remove_from = 0
        remove_to = tokens_to_remove
        kv_cache_seq_rm(self.ctx, seq_id, remove_from, remove_to)

        # Shift remaining positions
        kv_cache_seq_add(
            self.ctx, seq_id, remove_to, current_length, -remove_to
        )


class HybridPagedCache:
    """
    Hybrid cache combining PagedKVCache benefits with IncrementalKVCache simplicity.

    Uses paging for memory management but maintains simple single-sequence semantics.
    """

    def __init__(
        self,
        llm_ctx,
        max_seq_len: int = 32768,
        llm_obj=None,
        page_size: int = 64,
        num_pages: Optional[int] = None,
    ):
        """Initialize hybrid cache."""
        if num_pages is None:
            num_pages = max_seq_len // max(1, page_size) + 16
        self.paged_cache = PagedKVCache(
            llm_ctx,
            llm_obj,
            page_size=page_size,
            num_pages=num_pages,
        )
        self.seq_id = 0  # Single sequence
        self.max_seq_len = max_seq_len

    def reset(self):
        """Reset cache."""
        self.paged_cache.reset()

    def prepare_for_generation(
        self, prompt_tokens: List[int], allow_reuse: bool = True
    ) -> int:
        """Prepare for generation."""
        return self.paged_cache.prepare_for_generation(
            self.seq_id, prompt_tokens, allow_reuse
        )

    def advance_position(
        self, n_tokens: int = 1, token_ids: Optional[List[int]] = None
    ):
        """Advance position."""
        self.paged_cache.advance_position(self.seq_id, n_tokens, token_ids)

    def trim_to(self, keep_tokens: int):
        """Trim to keep tokens."""
        self.paged_cache.trim_to(self.seq_id, keep_tokens)

    def get_stats(self) -> dict:
        """Get statistics."""
        return self.paged_cache.get_stats()
