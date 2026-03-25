"""Saguaro indexing engine with TensorFlow-first backend and graceful fallback.

This implementation keeps the historical public API used across the codebase while
removing hard dependency on TensorFlow-native custom ops for core indexing flow.
"""

from __future__ import annotations

import contextlib
import logging
import mmap
import os
import resource
from array import array
from typing import Any

from saguaro.analysis.code_graph import CodeGraph
from saguaro.indexing.auto_scaler import calibrate_runtime_profile, load_runtime_profile
from saguaro.indexing.native_worker import process_batch_worker_native
from saguaro.indexing.native_runtime import get_native_runtime
from saguaro.indexing.tracker import IndexTracker
from saguaro.parsing.parser import SAGUAROParser
from saguaro.storage.vector_store import VectorStore
from saguaro.utils.entity_ids import entity_identity
from saguaro.utils.float_vector import FloatVector

logger = logging.getLogger(__name__)


def _current_rss_mb() -> float:
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return float(usage.ru_maxrss) / 1024.0
    except Exception:
        return 0.0


class SharedProjectionManager:
    """File-backed mmap projection manager for worker process embedding."""

    RESOURCE_BASENAME = "projection_runtime.bin"

    def __init__(self, saguaro_dir: str) -> None:
        """Initialize the instance."""
        self._saguaro_dir = os.path.abspath(saguaro_dir)
        self._path = os.path.join(self._saguaro_dir, "runtime", self.RESOURCE_BASENAME)
        self._file_handle: Any | None = None
        self._map: mmap.mmap | None = None
        self._nbytes = 0

    def create(self, vocab_size: int, active_dim: int) -> None:
        """Handle create."""
        nbytes = vocab_size * active_dim * 4
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self.close()
        with open(self._path, "wb") as handle:
            handle.truncate(nbytes)
        self._file_handle = open(self._path, "r+b")
        self._map = mmap.mmap(self._file_handle.fileno(), 0, access=mmap.ACCESS_WRITE)
        self._nbytes = nbytes

    @property
    def resource_name(self) -> str:
        return self._path

    def get_projection(self) -> mmap.mmap:
        """Get projection."""
        if self._map is None:
            raise RuntimeError("Projection not initialized.")
        return self._map

    def write_bytes(self, payload: bytes | bytearray | memoryview) -> None:
        """Copy initialized projection bytes into the mapped projection file."""
        if self._map is None:
            raise RuntimeError("Projection not initialized.")
        view = memoryview(payload).cast("B")
        try:
            if len(view) != self._nbytes:
                raise ValueError(
                    f"Projection payload size mismatch: {len(view)} != {self._nbytes}"
                )
            self._map.seek(0)
            self._map.write(view)
            self._map.flush()
            self._map.seek(0)
        finally:
            view.release()

    def cleanup(self) -> None:
        """Handle cleanup."""
        path = self._path
        self.close()
        with contextlib.suppress(FileNotFoundError):
            os.remove(path)

    def close(self) -> None:
        """Handle close."""
        if self._map is not None:
            self._map.close()
            self._map = None
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
        self._nbytes = 0


def process_batch_worker(
    file_paths: list,
    active_dim: int,
    total_dim: int,
    vocab_size: int,
    prefer_tensorflow: bool = True,
) -> tuple[list, list[array] | None]:
    """Worker-safe batch indexer used by process pools."""
    del prefer_tensorflow, total_dim
    meta_list, vectors, _touched, _metrics = process_batch_worker_native(
        file_paths=file_paths,
        active_dim=active_dim,
        total_dim=active_dim,
        vocab_size=vocab_size,
        repo_path=os.path.commonpath(file_paths) if file_paths else os.getcwd(),
    )
    return meta_list, vectors


class IndexEngine:
    """Main indexing engine used by query/index workflows."""

    def __init__(self, repo_path: str, saguaro_dir: str, config: dict) -> None:
        """Initialize the instance."""
        self.repo_path = os.path.abspath(repo_path)
        self.saguaro_dir = os.path.abspath(saguaro_dir)
        self.config = config or {}

        self.active_dim = int(self.config.get("active_dim", 4096))
        self.total_dim = int(self.config.get("total_dim", 8192))
        self.vocab_size = int(self.config.get("vocab_size", 16384))

        self.parser = SAGUAROParser()
        self.store = VectorStore(
            storage_path=os.path.join(self.saguaro_dir, "vectors"),
            dim=self.active_dim,
            active_dim=self.active_dim,
            total_dim=self.total_dim,
        )
        self.tracker = IndexTracker(os.path.join(self.saguaro_dir, "tracking.json"))
        self.code_graph = CodeGraph(
            repo_path=self.repo_path,
            graph_path=os.path.join(self.saguaro_dir, "graph", "code_graph.json"),
        )

        self.projection_manager = SharedProjectionManager(self.saguaro_dir)
        self.native_runtime = get_native_runtime()
        self.projection_buffer: bytearray | None = None

        self.current_bundle = array("f", [0.0]) * self.active_dim
        self.bundle_count = 0
        self.BUNDLE_THRESHOLD = 256

    def _ensure_projection_buffer(self) -> bytearray:
        if self.projection_buffer is None:
            nbytes = self.vocab_size * self.active_dim * 4
            self.projection_buffer = bytearray(nbytes)
            self.native_runtime.init_projection(
                self.projection_buffer,
                self.vocab_size,
                self.active_dim,
                seed=42,
            )
        return self.projection_buffer

    def calibrate(self, file_paths: list) -> None:
        """Persist a runtime layout sized to the current repository and host."""
        _ = file_paths
        native_threads = 0
        with contextlib.suppress(Exception):
            native_threads = int(self.native_runtime.max_threads())
        calibrate_runtime_profile(
            self.repo_path,
            cpu_threads=os.cpu_count() or 1,
            native_threads=native_threads or None,
            force=False,
        )

    def _default_native_threads(self) -> int:
        profile = load_runtime_profile(self.repo_path)
        layout = dict(profile.get("selected_runtime_layout") or {})
        query_threads = int(layout.get("query_threads", 0) or 0)
        if query_threads > 0:
            return query_threads
        with contextlib.suppress(Exception):
            return int(self.native_runtime.default_threads())
        return max(1, int(os.cpu_count() or 1))

    def create_shared_projection(self) -> None:
        """Handle create shared projection."""
        self.projection_manager.create(self.vocab_size, self.active_dim)
        projection = bytearray(self.vocab_size * self.active_dim * 4)
        self.native_runtime.init_projection(
            projection,
            self.vocab_size,
            self.active_dim,
            seed=42,
        )
        self.projection_manager.write_bytes(projection)

    def cleanup_shared_projection(self) -> None:
        """Handle cleanup shared projection."""
        self.projection_manager.cleanup()

    def _update_bundle(self, vectors: Any) -> None:
        row_count = 0
        first_row: array | None = None
        for row in vectors:
            row_count += 1
            if first_row is None:
                raw = self.store._coerce_vector_bytes(row)
                first_row = array("f")
                first_row.frombytes(raw)
        if row_count == 0:
            return
        if first_row is not None:
            self.current_bundle = first_row
        self.bundle_count += row_count

        if self.bundle_count >= self.BUNDLE_THRESHOLD:
            self._crystallize()

    def ingest_worker_result(
        self, meta_list: list[dict[str, object]], vectors: Any
    ) -> tuple[int, int]:
        """Handle ingest worker result."""
        if vectors is None or len(meta_list) == 0:
            return 0, 0

        self._update_bundle(vectors)
        self.store.add_batch(vectors, meta_list)

        file_count = len(set(m.get("file") for m in meta_list if m.get("file")))
        return file_count, len(meta_list)

    def _crystallize(self) -> None:
        # Keep crystallization in-memory during indexing; commit persists store state once.
        self.current_bundle = array("f", [0.0]) * self.active_dim
        self.bundle_count = 0

    def get_state(self) -> array:
        """Get state."""
        return self.current_bundle

    def commit(self) -> None:
        """Handle commit."""
        if self.bundle_count > 0:
            self._crystallize()
        self.store.save()
        self.tracker.save()

    def encode_text(
        self, text: str, dim: int = None, weight_symbols: bool = True
    ) -> Any:
        """Handle encode text."""
        _ = weight_symbols
        target_dim = int(dim or self.total_dim)

        try:
            vectors = self.native_runtime.full_pipeline(
                texts=[str(text or "")[:4096]],
                projection_buffer=self._ensure_projection_buffer(),
                vocab_size=self.vocab_size,
                dim=self.active_dim,
                max_length=512,
                num_threads=self._default_native_threads(),
            )
            vec = FloatVector(vectors[0]) if vectors else FloatVector.zeros(self.active_dim)
        except Exception:
            vec = FloatVector.zeros(self.active_dim)

        current_dim = len(vec)
        if current_dim < target_dim:
            padded = FloatVector(vec)
            padded.extend([0.0] * (target_dim - current_dim))
            return padded
        if current_dim > target_dim:
            return FloatVector(vec[:target_dim])

        return FloatVector(vec)

    def index_batch(self, file_paths: list[str], force: bool = False) -> tuple[int, int]:
        """Handle index batch."""
        if not force:
            file_paths = self.tracker.filter_needs_indexing(file_paths)

        indexed_files = 0
        indexed_entities = 0
        touched: list[str] = []
        parsed_entities_by_file: dict[str, list[Any]] = {}
        changed_files = [os.path.abspath(path) for path in file_paths if path]

        for path in file_paths:
            try:
                entities = self.parser.parse_file(path)
                parsed_entities_by_file[os.path.abspath(path)] = entities or []
                if hasattr(self.store, "remove_file"):
                    self.store.remove_file(path)
                if not entities:
                    if os.path.exists(path):
                        touched.append(path)
                    continue

                local_entities = 0
                vectors_batch = []

                for entity in entities:
                    vec = self.encode_text(entity.content[:4096], dim=self.total_dim)
                    identity = entity_identity(
                        self.repo_path,
                        entity.file_path,
                        entity.name,
                        entity.type,
                        entity.start_line,
                    )
                    meta = {
                        "entity_id": identity["entity_id"],
                        "name": identity["display_name"],
                        "qualified_name": identity["qualified_name"],
                        "type": entity.type,
                        "file": entity.file_path,
                        "line": entity.start_line,
                        "end_line": entity.end_line,
                    }
                    self.store.add(vec, meta=meta)
                    vectors_batch.append(vec)
                    local_entities += 1

                if os.path.exists(path):
                    touched.append(path)

                if local_entities > 0:
                    indexed_files += 1
                    indexed_entities += local_entities
                    self._update_bundle(vectors_batch)
            except Exception as e:
                logger.debug("index_batch failed for %s: %s", path, e)

        if touched:
            self.tracker.update(touched)

        if changed_files:
            try:
                self.code_graph.build_incremental(
                    file_paths=changed_files,
                    parsed_entities_by_file=parsed_entities_by_file,
                )
            except Exception as e:
                logger.debug("code graph update failed: %s", e)

        return indexed_files, indexed_entities

    def compute_state(self) -> bytes:
        # Deterministic state snapshot from tracked hashes.
        """Handle compute state."""
        lines = []
        for path, data in sorted(self.tracker.state.items()):
            lines.append(f"{path}:{data.get('hash', '')}")
        if not lines:
            return b""

        vec = self.encode_text("\n".join(lines), dim=self.total_dim)
        return vec.tobytes()

    def get_memory_stats(self) -> dict[str, Any]:
        """Get memory stats."""
        return {
            "peak_rss_mb": _current_rss_mb(),
            "current_rss_mb": _current_rss_mb(),
            "batches_processed": 0,
            "vectors_indexed": len(self.store),
            "files_indexed": len(self.tracker.state),
            "gc_collections": 0,
            "store_count": len(self.store),
            "backend": "native_runtime",
        }


# Backward compatibility aliases
process_batch_worker_memory_optimized = process_batch_worker
MemoryOptimizedIndexEngine = IndexEngine
