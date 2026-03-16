"""Backward-compatible memory-optimized engine facade.

The core implementation now lives in `saguaro.indexing.engine.IndexEngine` and
already supports low-memory operation with graceful backend fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from saguaro.indexing.engine import (
    IndexEngine,
)
from saguaro.indexing.engine import (
    process_batch_worker as _process_batch_worker,
)


@dataclass
class MemoryStats:
    """Provide MemoryStats support."""
    peak_rss_mb: float = 0.0
    current_rss_mb: float = 0.0
    batches_processed: int = 0
    vectors_indexed: int = 0
    files_indexed: int = 0
    gc_collections: int = 0


class MemoryOptimizedIndexEngine(IndexEngine):
    """Compatibility subclass that tracks additional memory stats."""

    def __init__(self, repo_path: str, saguaro_dir: str, config: dict) -> None:
        """Initialize the instance."""
        super().__init__(repo_path, saguaro_dir, config)
        self.memory_stats = MemoryStats()

    def ingest_worker_result(
        self, meta_list: list, vectors: Any
    ) -> tuple[int, int]:
        """Handle ingest worker result."""
        file_count, entity_count = super().ingest_worker_result(meta_list, vectors)
        self.memory_stats.vectors_indexed += entity_count
        self.memory_stats.files_indexed += file_count
        self.memory_stats.batches_processed += 1
        return file_count, entity_count

    def get_memory_stats(self) -> dict[str, Any]:
        """Get memory stats."""
        base = super().get_memory_stats()
        self.memory_stats.current_rss_mb = base.get("current_rss_mb", 0.0)
        self.memory_stats.peak_rss_mb = max(
            self.memory_stats.peak_rss_mb, self.memory_stats.current_rss_mb
        )

        base.update(
            {
                "peak_rss_mb": self.memory_stats.peak_rss_mb,
                "current_rss_mb": self.memory_stats.current_rss_mb,
                "batches_processed": self.memory_stats.batches_processed,
                "vectors_indexed": self.memory_stats.vectors_indexed,
                "files_indexed": self.memory_stats.files_indexed,
                "gc_collections": self.memory_stats.gc_collections,
            }
        )
        return base


def process_batch_worker_memory_optimized(
    file_paths: list,
    active_dim: int,
    total_dim: int,
    vocab_size: int,
) -> tuple[list, Any]:
    """Handle process batch worker memory optimized."""
    return _process_batch_worker(
        file_paths=file_paths,
        active_dim=active_dim,
        total_dim=total_dim,
        vocab_size=vocab_size,
        prefer_tensorflow=True,
    )


# Backward compatibility aliases
IndexEngine = MemoryOptimizedIndexEngine
process_batch_worker = process_batch_worker_memory_optimized
