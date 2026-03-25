"""Native-backed vector storage and query for Saguaro."""

from __future__ import annotations

import ctypes
import heapq
import json
import logging
import math
import mmap
import os
import re
import threading
from array import array
from pathlib import Path
from typing import Any, Iterable

from saguaro.errors import SaguaroStateCorruptionError
from saguaro.storage.atomic_fs import atomic_write_json

logger = logging.getLogger(__name__)
_IDENT_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")
_FLOAT_SIZE = 4
_LIB: ctypes.CDLL | None = None
_LIB_LOCK = threading.Lock()
_INT32_P = ctypes.POINTER(ctypes.c_int32)
_FLOAT_P = ctypes.POINTER(ctypes.c_float)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _native_lib_path() -> Path:
    return _repo_root() / "build" / "libanvil_saguaro_vector_store.so"


def _load_native_library() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB

    with _LIB_LOCK:
        if _LIB is not None:
            return _LIB
        lib_path = _native_lib_path()
        if not lib_path.exists():
            raise RuntimeError(
                "Prebuilt native vector-store library is required and runtime "
                f"compilation is disabled: missing {lib_path}"
            )
        lib = ctypes.CDLL(str(lib_path))
        lib.anvil_saguaro_query_cosine.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            _FLOAT_P,
            _INT32_P,
            ctypes.c_int,
            _FLOAT_P,
        ]
        lib.anvil_saguaro_query_cosine.restype = ctypes.c_int
        lib.anvil_saguaro_write_indexed_rows.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            _INT32_P,
            ctypes.c_int,
            _FLOAT_P,
        ]
        lib.anvil_saguaro_write_indexed_rows.restype = ctypes.c_int
        if hasattr(lib, "anvil_saguaro_write_indexed_rows_with_norms"):
            lib.anvil_saguaro_write_indexed_rows_with_norms.argtypes = [
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_int,
                _INT32_P,
                ctypes.c_int,
                _FLOAT_P,
                _FLOAT_P,
            ]
            lib.anvil_saguaro_write_indexed_rows_with_norms.restype = ctypes.c_int
        if hasattr(lib, "anvil_saguaro_open_store"):
            lib.anvil_saguaro_open_store.argtypes = [
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.anvil_saguaro_open_store.restype = ctypes.c_void_p
        if hasattr(lib, "anvil_saguaro_query_store"):
            lib.anvil_saguaro_query_store.argtypes = [
                ctypes.c_void_p,
                _FLOAT_P,
                _INT32_P,
                ctypes.c_int,
                _FLOAT_P,
            ]
            lib.anvil_saguaro_query_store.restype = ctypes.c_int
        if hasattr(lib, "anvil_saguaro_close_store"):
            lib.anvil_saguaro_close_store.argtypes = [ctypes.c_void_p]
            lib.anvil_saguaro_close_store.restype = None
        if hasattr(lib, "anvil_saguaro_perf_open_handles"):
            lib.anvil_saguaro_perf_open_handles.argtypes = []
            lib.anvil_saguaro_perf_open_handles.restype = ctypes.c_longlong
        if hasattr(lib, "anvil_saguaro_perf_remap_count"):
            lib.anvil_saguaro_perf_remap_count.argtypes = []
            lib.anvil_saguaro_perf_remap_count.restype = ctypes.c_longlong
        if hasattr(lib, "anvil_saguaro_perf_query_calls"):
            lib.anvil_saguaro_perf_query_calls.argtypes = []
            lib.anvil_saguaro_perf_query_calls.restype = ctypes.c_longlong
        _LIB = lib
        return lib


def native_vector_store_available() -> bool:
    try:
        _load_native_library()
        return True
    except Exception as exc:
        logger.debug("Native vector store unavailable: %s", exc)
        return False


def native_vector_store_perf_counters() -> dict[str, int]:
    try:
        lib = _load_native_library()
    except Exception as exc:
        return {"available": 0, "error": str(exc)}
    counters = {"available": 1}
    for key, symbol in (
        ("open_handles", "anvil_saguaro_perf_open_handles"),
        ("remap_count", "anvil_saguaro_perf_remap_count"),
        ("query_calls", "anvil_saguaro_perf_query_calls"),
    ):
        func = getattr(lib, symbol, None)
        counters[key] = int(func()) if func is not None else 0
    return counters


class NativeMemoryMappedVectorStore:
    """Raw mmap vector storage with native C++ cosine scoring."""

    GROWTH_FACTOR = 2.0
    INITIAL_CAPACITY = 10000
    SCHEMA_VERSION = 4

    def __init__(
        self,
        storage_path: str,
        dim: int,
        dark_space_ratio: float = 0.4,
        read_only: bool = False,
        *,
        active_dim: int | None = None,
        total_dim: int | None = None,
    ) -> None:
        self.storage_path = storage_path
        self.total_dim = int(total_dim or dim)
        self.active_dim = int(active_dim or dim)
        self.dim = self.total_dim
        self.dark_space_ratio = dark_space_ratio
        self.read_only = read_only

        self.vectors_path = os.path.join(storage_path, "vectors.bin")
        self.norms_path = os.path.join(storage_path, "norms.bin")
        self.metadata_path = os.path.join(storage_path, "metadata.json")
        self.index_meta_path = os.path.join(storage_path, "index_meta.json")

        self._count = 0
        self._capacity = 0
        self._storage_dim = max(1, self.active_dim)
        self._row_bytes = self._storage_dim * _FLOAT_SIZE
        self._format_version = self.SCHEMA_VERSION
        self._lookup: dict[tuple[str, str], int] = {}
        self._entity_lookup: dict[str, int] = {}
        self._term_index: dict[str, set[int]] = {}
        self._indexes_dirty = True
        self._metadata: list[dict[str, Any]] = []
        self._norms: list[float] = []
        self._file_handle: Any | None = None
        self._map: mmap.mmap | None = None
        self._write_lock = threading.RLock()
        self._native_handle: ctypes.c_void_p | None = None
        self._perf_counters = {
            "query_count": 0,
            "query_handle_opens": 0,
            "query_handle_closes": 0,
            "legacy_path_queries": 0,
        }

        self._load()

    def _load(self) -> None:
        os.makedirs(self.storage_path, exist_ok=True)
        if not os.path.exists(self.index_meta_path):
            if any(os.path.exists(path) for path in (self.vectors_path, self.metadata_path)):
                raise SaguaroStateCorruptionError(
                    "Incomplete vector store; index metadata is missing."
                )
            self._initialize_fresh()
            return

        try:
            with open(self.index_meta_path, encoding="utf-8") as handle:
                meta = json.load(handle) or {}
            self._format_version = int(meta.get("version", 3) or 3)
            stored_total_dim = int(meta.get("total_dim", meta.get("dim", self.total_dim)))
            stored_active_dim = int(meta.get("active_dim", self.active_dim))
            self.total_dim = stored_total_dim
            self.dim = stored_total_dim
            self.active_dim = min(stored_active_dim, stored_total_dim)
            self._storage_dim = self.active_dim if self._format_version >= 4 else stored_total_dim
            self._row_bytes = self._storage_dim * _FLOAT_SIZE
            self._count = int(meta.get("count", 0) or 0)
            self._capacity = int(meta.get("capacity", self.INITIAL_CAPACITY) or self.INITIAL_CAPACITY)

            if not os.path.exists(self.vectors_path):
                raise SaguaroStateCorruptionError(
                    f"Missing vectors file for vector store: {self.vectors_path}"
                )
            if not os.path.exists(self.metadata_path):
                raise SaguaroStateCorruptionError(
                    f"Missing metadata file for vector store: {self.metadata_path}"
                )
            with open(self.metadata_path, encoding="utf-8") as handle:
                self._metadata = json.load(handle) or []

            self._open_mapping()
            self._validate_state()
            self._load_norms()
            self._rebuild_indexes()
        except Exception as exc:
            raise SaguaroStateCorruptionError(
                f"Failed to load native vector store from {self.storage_path}: {exc}"
            ) from exc

    def _initialize_fresh(self) -> None:
        self._count = 0
        self._capacity = self.INITIAL_CAPACITY
        self._metadata = []
        self._norms = []
        self._lookup = {}
        self._entity_lookup = {}
        self._term_index = {}
        self._indexes_dirty = False
        self._format_version = self.SCHEMA_VERSION
        self._storage_dim = self.active_dim
        self._row_bytes = self._storage_dim * _FLOAT_SIZE
        self._create_vectors_file()

    def _create_vectors_file(self) -> None:
        self._close_query_handle()
        self._close_mapping()
        total_bytes = self._capacity * self._row_bytes
        with open(self.vectors_path, "wb") as handle:
            handle.truncate(total_bytes)
        self._open_mapping()

    def _open_mapping(self) -> None:
        self._close_mapping()
        mode = "rb" if self.read_only else "r+b"
        self._file_handle = open(self.vectors_path, mode)
        access = mmap.ACCESS_READ if self.read_only else mmap.ACCESS_WRITE
        self._map = mmap.mmap(self._file_handle.fileno(), 0, access=access)

    def _close_mapping(self) -> None:
        if self._map is not None:
            self._map.close()
            self._map = None
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def _close_query_handle(self) -> None:
        if self._native_handle is None:
            return
        lib = _load_native_library()
        if hasattr(lib, "anvil_saguaro_close_store"):
            lib.anvil_saguaro_close_store(self._native_handle)
        self._native_handle = None
        self._perf_counters["query_handle_closes"] += 1

    def _ensure_query_handle(self) -> ctypes.c_void_p | None:
        if self._native_handle is not None:
            return self._native_handle
        if self._format_version < 4 or self._count == 0:
            return None
        lib = _load_native_library()
        if not hasattr(lib, "anvil_saguaro_open_store"):
            return None
        norms_arg = self.norms_path.encode("utf-8") if os.path.exists(self.norms_path) else None
        handle = lib.anvil_saguaro_open_store(
            self.vectors_path.encode("utf-8"),
            norms_arg,
            self._storage_dim,
            self._count,
        )
        if not handle:
            return None
        self._native_handle = handle
        self._perf_counters["query_handle_opens"] += 1
        return handle

    def _load_norms(self) -> None:
        if os.path.exists(self.norms_path):
            raw = Path(self.norms_path).read_bytes()
            floats = array("f")
            floats.frombytes(raw)
            self._norms = list(floats[: self._count])
        else:
            self._norms = [self._compute_row_norm(idx) for idx in range(self._count)]

    def _write_norms(self) -> None:
        values = array("f", [float(item) for item in self._norms[: self._count]])
        with open(self.norms_path, "wb") as handle:
            handle.write(memoryview(values).cast("B"))

    def _grow_capacity(self) -> None:
        if self.read_only:
            raise RuntimeError("Cannot grow capacity in read-only mode")
        self._capacity = int(max(self._capacity + 1, self._capacity * self.GROWTH_FACTOR))
        total_bytes = self._capacity * self._row_bytes
        self._close_query_handle()
        self._close_mapping()
        with open(self.vectors_path, "r+b") as handle:
            handle.truncate(total_bytes)
        self._open_mapping()

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
                self._lookup[(str(file), str(identity))] = idx
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
            joined = " ".join(str(item or "") for item in raw_terms)
            for term in self._extract_terms(joined):
                self._term_index.setdefault(term, set()).add(idx)
        self._indexes_dirty = False

    def _ensure_indexes(self) -> None:
        if self._indexes_dirty:
            self._rebuild_indexes()

    def _candidate_indices(self, query_text: str | None, k: int) -> list[int] | None:
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
        ranked = sorted(candidate_scores.items(), key=lambda item: (-item[1], item[0]))
        limit = min(self._count, max(64, max(1, int(k)) * 32))
        return [idx for idx, _score in ranked[:limit]]

    def _indices_for_entity_ids(self, candidate_ids: list[str] | None) -> list[int] | None:
        if not candidate_ids:
            return None
        self._ensure_indexes()
        indices = [self._entity_lookup[item] for item in candidate_ids if item in self._entity_lookup]
        return sorted(set(indices)) if indices else None

    def _row_offset(self, idx: int) -> int:
        return int(idx) * self._row_bytes

    def _read_row_bytes(self, idx: int, *, target_dim: int | None = None) -> bytes:
        assert self._map is not None
        raw = self._map[self._row_offset(idx) : self._row_offset(idx) + self._row_bytes]
        if target_dim is None or int(target_dim) == self._storage_dim:
            return raw
        expected = int(target_dim) * _FLOAT_SIZE
        return raw[:expected] if len(raw) >= expected else raw + (b"\x00" * (expected - len(raw)))

    def _compute_row_norm(self, idx: int) -> float:
        values = array("f")
        values.frombytes(self._read_row_bytes(idx, target_dim=self.active_dim))
        return math.sqrt(sum(float(value) * float(value) for value in values))

    def _coerce_vector_bytes(self, value: Any, *, target_dim: int | None = None) -> bytes:
        dim = int(target_dim or self._storage_dim)
        expected_bytes = dim * _FLOAT_SIZE
        try:
            view = memoryview(value)
        except TypeError:
            view = None
        if view is not None and view.contiguous:
            try:
                raw = view.cast("B").tobytes()
            except TypeError:
                raw = b""
            if raw:
                if len(raw) >= expected_bytes:
                    return raw[:expected_bytes]
                return raw + (b"\x00" * (expected_bytes - len(raw)))

        values = array("f")
        for item in self._flatten_vector(value):
            values.append(float(item))
            if len(values) >= dim:
                break
        while len(values) < dim:
            values.append(0.0)
        return memoryview(values).cast("B").tobytes()

    def _flatten_vector(self, value: Any) -> Iterable[float]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, array):
            for item in value:
                yield float(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, (list, tuple, array)):
                    yield from self._flatten_vector(item)
                else:
                    yield float(item)
            return
        yield from (float(item) for item in value)

    def _norm_for_bytes(self, raw: bytes) -> float:
        values = array("f")
        values.frombytes(raw[: self.active_dim * _FLOAT_SIZE])
        return math.sqrt(sum(float(value) * float(value) for value in values))

    def _write_vector(self, idx: int, vector: Any) -> None:
        raw = self._coerce_vector_bytes(vector)
        assert self._map is not None
        start = self._row_offset(idx)
        self._map[start : start + self._row_bytes] = raw
        norm = self._norm_for_bytes(raw)
        if idx < len(self._norms):
            self._norms[idx] = norm
        else:
            self._norms.append(norm)

    def add(self, vector: Any, meta: dict[str, Any]) -> int:
        if self.read_only:
            raise RuntimeError("Cannot add vectors in read-only mode")
        with self._write_lock:
            self._close_query_handle()
            file = meta.get("file")
            identity = meta.get("entity_id") or meta.get("qualified_name") or meta.get("name")
            key = (str(file), str(identity)) if (file and identity) else None
            if key and key in self._lookup:
                idx = self._lookup[key]
                self._write_vector(idx, vector)
                self._metadata[idx] = meta
                self._indexes_dirty = True
                return idx
            if self._count >= self._capacity:
                self._grow_capacity()
            idx = self._count
            self._write_vector(idx, vector)
            self._metadata.append(meta)
            if key:
                self._lookup[key] = idx
            self._count += 1
            self._indexes_dirty = True
            return idx

    def _coerce_batch_rows(self, vectors: Any, row_count: int) -> array:
        rows = list(vectors)
        if len(rows) != row_count:
            raise ValueError(
                f"Vector batch row count mismatch: expected {row_count}, got {len(rows)}"
            )
        packed = array("f")
        for row in rows:
            raw = self._coerce_vector_bytes(row)
            values = array("f")
            values.frombytes(raw)
            packed.extend(values)
        return packed

    def add_batch(self, vectors: Any, metas: list[dict[str, Any]]) -> int:
        if self.read_only:
            raise RuntimeError("Cannot add vectors in read-only mode")
        if not metas:
            return 0
        row_buf = self._coerce_batch_rows(vectors, len(metas))
        with self._write_lock:
            self._close_query_handle()
            indices: list[int] = []
            pending_new = 0
            for meta in metas:
                file = meta.get("file")
                identity = meta.get("entity_id") or meta.get("qualified_name") or meta.get("name")
                key = (str(file), str(identity)) if (file and identity) else None
                if key and key in self._lookup:
                    idx = self._lookup[key]
                else:
                    idx = self._count + pending_new
                    pending_new += 1
                    if key:
                        self._lookup[key] = idx
                indices.append(idx)

            while self._count + pending_new > self._capacity:
                self._grow_capacity()

            lib = _load_native_library()
            idx_buf = (ctypes.c_int32 * len(indices))(*indices)
            norm_buf = (ctypes.c_float * len(indices))()
            idx_ptr = ctypes.cast(idx_buf, _INT32_P)
            row_ctypes = (ctypes.c_float * len(row_buf)).from_buffer(row_buf)
            row_ptr = ctypes.cast(row_ctypes, _FLOAT_P)
            norm_ptr = ctypes.cast(norm_buf, _FLOAT_P)
            if hasattr(lib, "anvil_saguaro_write_indexed_rows_with_norms"):
                rc = lib.anvil_saguaro_write_indexed_rows_with_norms(
                    self.vectors_path.encode("utf-8"),
                    self._storage_dim,
                    self._capacity,
                    idx_ptr,
                    len(indices),
                    row_ptr,
                    norm_ptr,
                )
            else:
                rc = lib.anvil_saguaro_write_indexed_rows(
                    self.vectors_path.encode("utf-8"),
                    self._storage_dim,
                    self._capacity,
                    idx_ptr,
                    len(indices),
                    row_ptr,
                )
            if rc < 0:
                raise RuntimeError("Native vector-store batch write failed.")

            last_pos_by_index = {idx: pos for pos, idx in enumerate(indices)}
            for idx in sorted(last_pos_by_index):
                pos = last_pos_by_index[idx]
                meta = metas[pos]
                norm = float(norm_buf[pos])
                if idx < len(self._metadata):
                    self._metadata[idx] = meta
                    self._norms[idx] = norm
                else:
                    self._metadata.append(meta)
                    self._norms.append(norm)
            self._count += pending_new
            self._indexes_dirty = True
            return pending_new

    def remove_file(self, file_path: str) -> int:
        return self.remove_files([file_path])

    def remove_files(self, file_paths: list[str]) -> int:
        if self.read_only:
            raise RuntimeError("Cannot remove vectors in read-only mode")
        if not file_paths:
            return 0
        with self._write_lock:
            self._close_query_handle()
            targets = {str(path) for path in file_paths if path}
            keep_indices = [
                idx
                for idx, meta in enumerate(self._metadata[: self._count])
                if str(meta.get("file") or "") not in targets
            ]
            removed = self._count - len(keep_indices)
            if removed <= 0:
                return 0
            assert self._map is not None
            for new_idx, old_idx in enumerate(keep_indices):
                if new_idx == old_idx:
                    continue
                new_off = self._row_offset(new_idx)
                old_off = self._row_offset(old_idx)
                self._map[new_off : new_off + self._row_bytes] = self._map[
                    old_off : old_off + self._row_bytes
                ]
            zero_row = b"\x00" * self._row_bytes
            for idx in range(len(keep_indices), self._count):
                off = self._row_offset(idx)
                self._map[off : off + self._row_bytes] = zero_row
            self._metadata = [self._metadata[idx] for idx in keep_indices]
            self._norms = [self._norms[idx] for idx in keep_indices]
            self._count = len(keep_indices)
            self._indexes_dirty = True
            self._rebuild_indexes()
            return removed

    def _migrate_legacy_layout(self) -> None:
        if self._format_version >= self.SCHEMA_VERSION or self.active_dim >= self.total_dim:
            return
        rows = [self._read_row_bytes(idx, target_dim=self.active_dim) for idx in range(self._count)]
        self._close_mapping()
        self._storage_dim = self.active_dim
        self._row_bytes = self._storage_dim * _FLOAT_SIZE
        with open(self.vectors_path, "wb") as handle:
            handle.truncate(self._capacity * self._row_bytes)
        self._open_mapping()
        assert self._map is not None
        for idx, raw in enumerate(rows):
            off = self._row_offset(idx)
            self._map[off : off + self._row_bytes] = raw
        self._norms = [self._norm_for_bytes(raw) for raw in rows]
        self._format_version = self.SCHEMA_VERSION

    def save(self) -> None:
        if self.read_only:
            return
        with self._write_lock:
            self._close_query_handle()
            if self._format_version < self.SCHEMA_VERSION:
                self._migrate_legacy_layout()
            self._validate_state()
            if self._map is not None:
                self._map.flush()
            self._write_norms()
            atomic_write_json(self.metadata_path, self._metadata, indent=2, sort_keys=False)
            atomic_write_json(
                self.index_meta_path,
                {
                    "dim": self.total_dim,
                    "active_dim": self.active_dim,
                    "total_dim": self.total_dim,
                    "storage_dim": self._storage_dim,
                    "count": self._count,
                    "capacity": self._capacity,
                    "version": self.SCHEMA_VERSION,
                    "format": "native_mmap",
                    "vector_layout": "active_only",
                    "norms_file": "norms.bin",
                },
                indent=2,
                sort_keys=True,
            )

    def _query_scores(self, raw_query: bytes, candidate_indices: list[int] | None) -> tuple[list[int], list[float]]:
        lib = _load_native_library()
        query_buf = (ctypes.c_float * self._storage_dim).from_buffer_copy(raw_query)
        idx_ptr = ctypes.cast(None, _INT32_P)
        active_count = 0
        if candidate_indices:
            idx_buf = (ctypes.c_int32 * len(candidate_indices))(*candidate_indices)
            idx_ptr = ctypes.cast(idx_buf, _INT32_P)
            active_count = len(candidate_indices)
        else:
            idx_buf = None
        score_count = active_count or self._count
        score_buf = (ctypes.c_float * score_count)()
        handle = self._ensure_query_handle()
        if handle is not None and hasattr(lib, "anvil_saguaro_query_store"):
            rc = lib.anvil_saguaro_query_store(
                handle,
                ctypes.cast(query_buf, _FLOAT_P),
                idx_ptr,
                active_count,
                ctypes.cast(score_buf, _FLOAT_P),
            )
        else:
            self._perf_counters["legacy_path_queries"] += 1
            rc = lib.anvil_saguaro_query_cosine(
                self.vectors_path.encode("utf-8"),
                self._storage_dim,
                self._count,
                ctypes.cast(query_buf, _FLOAT_P),
                idx_ptr,
                active_count,
                ctypes.cast(score_buf, _FLOAT_P),
            )
        if rc < 0:
            raise RuntimeError("Native vector-store query failed.")
        score_indices = candidate_indices if candidate_indices else list(range(self._count))
        return score_indices, [float(score_buf[i]) for i in range(score_count)]

    def query(
        self,
        query_vec: Any,
        k: int = 5,
        boost_map: dict[str, float] | None = None,
        query_text: str | None = None,
        candidate_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if self._count == 0:
            return []
        self._perf_counters["query_count"] += 1
        raw_query = self._coerce_vector_bytes(query_vec, target_dim=self._storage_dim)
        candidate_indices = self._indices_for_entity_ids(candidate_ids)
        if not candidate_indices:
            candidate_indices = self._candidate_indices(query_text, k)
        score_indices, all_scores = self._query_scores(raw_query, candidate_indices)

        if boost_map:
            for i, idx in enumerate(score_indices):
                name = str(self._metadata[int(idx)].get("name") or "")
                if name and name in boost_map:
                    all_scores[i] += float(boost_map[name]) * 0.2

        if query_text and len(query_text) > 3:
            literal_tokens = set(re.findall(r"[A-Za-z0-9_]{4,}", query_text))
            if literal_tokens:
                for i, idx in enumerate(score_indices):
                    meta = self._metadata[int(idx)]
                    name = str(meta.get("name") or "")
                    file = str(meta.get("file") or "")
                    match_score = 0.0
                    for token in literal_tokens:
                        if token in name:
                            match_score += 0.5
                        elif token in file:
                            match_score += 0.2
                    if match_score > 0.0:
                        all_scores[i] *= 1.0 + min(match_score, 1.0)

        best = heapq.nlargest(min(max(int(k), 1), len(all_scores)), enumerate(all_scores), key=lambda item: item[1])
        results = []
        for rank, (score_idx, score) in enumerate(best, start=1):
            store_idx = int(score_indices[score_idx])
            res = self._metadata[store_idx].copy()
            res["score"] = float(score)
            res["rank"] = rank
            res["candidate_pool"] = len(score_indices)
            res["cpu_prefiltered"] = bool(candidate_indices)
            res.setdefault("symbol_terms", list(res.get("symbol_terms", []) or []))
            res.setdefault("path_terms", list(res.get("path_terms", []) or []))
            res.setdefault("doc_terms", list(res.get("doc_terms", []) or []))
            res.setdefault("chunk_role", res.get("chunk_role"))
            res.setdefault("file_role", res.get("file_role"))
            res.setdefault("parent_symbol", res.get("parent_symbol"))

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
            file_path = str(res.get("file") or "")
            if "tests" in file_path:
                reasons.append("Provides usage examples via tests.")
            elif "docs" in file_path:
                reasons.append("Documentation source.")
            res["reason"] = " ".join(reasons)
            results.append(res)
        return results

    def clear(self) -> None:
        if self.read_only:
            raise RuntimeError("Cannot clear in read-only mode")
        with self._write_lock:
            self._close_query_handle()
            self._count = 0
            self._metadata = []
            self._norms = []
            self._lookup = {}
            self._entity_lookup = {}
            self._term_index = {}
            self._indexes_dirty = False
            self.save()

    def perf_counters(self) -> dict[str, int]:
        counters = dict(self._perf_counters)
        for key, value in native_vector_store_perf_counters().items():
            if key == "available":
                counters["native_perf_available"] = int(value)
            elif key == "error":
                counters["native_perf_error"] = str(value)
            else:
                counters[key] = int(value)
        return counters

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
            expected_size = self._capacity * self._row_bytes
            actual_size = os.path.getsize(self.vectors_path)
            if actual_size != expected_size:
                raise SaguaroStateCorruptionError(
                    f"Vector store size mismatch for {self.vectors_path}: expected {expected_size}, got {actual_size}."
                )

    def __len__(self) -> int:
        return self._count

    def close(self) -> None:
        self._close_query_handle()
        self._close_mapping()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
