from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from core.native.qsg_state_kernels_wrapper import (
    qsg_state_clone_cow,
    qsg_state_gather_rows,
    qsg_state_scatter_rows,
)


@dataclass(slots=True, frozen=True)
class RowRef:
    page_id: int
    row_idx: int


@dataclass(slots=True)
class StatePageMeta:
    page_id: int
    rows_used: int = 0
    refcount: int = 0
    generation: int = 0
    numa_node: int = 0


class QSGStatePager:
    """Paged row allocator for QSG continuous state."""

    def __init__(
        self,
        *,
        dim: int,
        state_page_rows: int,
        soft_compaction_threshold: float,
        hard_compaction_threshold: float,
    ) -> None:
        self._dim = int(max(1, dim))
        self._rows_per_page = int(max(1, state_page_rows))
        self._soft_threshold = float(max(0.0, soft_compaction_threshold))
        self._hard_threshold = float(max(self._soft_threshold, hard_compaction_threshold))

        self._pages: dict[int, np.ndarray] = {}
        self._meta: dict[int, StatePageMeta] = {}
        self._free_rows: dict[int, deque[int]] = {}
        self._request_rows: dict[str, list[RowRef]] = {}
        self._next_page_id = 0

        self._alloc_attempts = 0
        self._allocator_failures = 0
        self._compaction_count = 0
        self._cow_events = 0
        self._numpy_hot_path_calls = 0
        self._python_hot_path_calls = 0

    @property
    def dim(self) -> int:
        return self._dim

    def alloc_rows(self, request_id: str, n_rows: int) -> list[RowRef]:
        rows_needed = int(max(0, n_rows))
        self._alloc_attempts += rows_needed
        if rows_needed == 0:
            return []
        allocated: list[RowRef] = []
        try:
            for _ in range(rows_needed):
                allocated.append(self._alloc_single_row())
        except MemoryError:
            self._allocator_failures += 1
            for ref in allocated:
                self._free_row(ref)
            raise
        self._request_rows.setdefault(request_id, []).extend(allocated)
        return allocated

    def gather_active(self, slot_refs: list[RowRef]) -> np.ndarray:
        self._python_hot_path_calls += 1
        if not slot_refs:
            return np.zeros((0, self._dim), dtype=np.float32)
        refs = np.asarray([(r.page_id, r.row_idx) for r in slot_refs], dtype=np.int64)
        out = np.zeros((len(slot_refs), self._dim), dtype=np.float32)
        qsg_state_gather_rows(out, self._pages, refs, self._dim)
        self._numpy_hot_path_calls += 1
        return out

    def scatter_updates(self, slot_refs: list[RowRef], delta: np.ndarray) -> None:
        self._python_hot_path_calls += 1
        if not slot_refs:
            return
        values = np.asarray(delta, dtype=np.float32)
        if values.shape != (len(slot_refs), self._dim):
            raise ValueError(
                f"delta shape must be {(len(slot_refs), self._dim)}, got {values.shape}"
            )
        refs = np.asarray([(r.page_id, r.row_idx) for r in slot_refs], dtype=np.int64)
        qsg_state_scatter_rows(values, self._pages, refs, self._dim)
        self._numpy_hot_path_calls += 1

    def clone_cow(
        self,
        request_id: str,
        row_range: tuple[int, int] | None = None,
        *,
        target_request_id: str | None = None,
    ) -> list[RowRef]:
        src_refs = list(self._request_rows.get(request_id, []))
        if row_range is not None:
            start = max(0, int(row_range[0]))
            end = max(start, int(row_range[1]))
            src_refs = src_refs[start:end]
        if not src_refs:
            return []
        target = target_request_id or request_id
        dst_refs = self.alloc_rows(target, len(src_refs))
        src_array = np.asarray([(ref.page_id, ref.row_idx) for ref in src_refs], dtype=np.int64)
        dst_array = np.asarray([(ref.page_id, ref.row_idx) for ref in dst_refs], dtype=np.int64)
        qsg_state_clone_cow(self._pages, src_array, dst_array, self._dim)
        self._cow_events += len(dst_refs)
        self._numpy_hot_path_calls += 1
        return dst_refs

    def release_request(self, request_id: str) -> None:
        refs = self._request_rows.pop(request_id, [])
        for ref in refs:
            self._free_row(ref)

    def compact_if_needed(self, fragmentation_ratio_threshold: float | None = None) -> bool:
        threshold = (
            self._soft_threshold
            if fragmentation_ratio_threshold is None
            else float(max(0.0, fragmentation_ratio_threshold))
        )
        if self.fragmentation_ratio() <= threshold:
            return False
        snapshot = {
            request_id: self.gather_active(refs)
            for request_id, refs in self._request_rows.items()
            if refs
        }
        self._pages.clear()
        self._meta.clear()
        self._free_rows.clear()
        self._request_rows.clear()
        self._next_page_id = 0
        for request_id, values in snapshot.items():
            new_refs = self.alloc_rows(request_id, values.shape[0])
            self.scatter_updates(new_refs, values)
        self._compaction_count += 1
        return True

    def soft_compact_if_needed(self) -> bool:
        return self.compact_if_needed(self._soft_threshold)

    def hard_compact_if_needed(self) -> bool:
        return self.compact_if_needed(self._hard_threshold)

    def get_request_rows(self, request_id: str) -> list[RowRef]:
        return list(self._request_rows.get(request_id, []))

    def export_request_state(self, request_id: str) -> np.ndarray:
        refs = self.get_request_rows(request_id)
        if not refs:
            return np.zeros((0, self._dim), dtype=np.float32)
        return self.gather_active(refs)

    def import_request_state(
        self,
        request_id: str,
        values: np.ndarray,
    ) -> list[RowRef]:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.size == 0:
            self.release_request(request_id)
            return []
        if matrix.shape[1] != self._dim:
            raise ValueError(
                f"latent state dim must be {self._dim}, got {matrix.shape[1]}"
            )
        self.release_request(request_id)
        refs = self.alloc_rows(request_id, matrix.shape[0])
        self.scatter_updates(refs, matrix)
        return refs

    def fragmentation_ratio(self) -> float:
        committed_rows = len(self._pages) * self._rows_per_page
        if committed_rows <= 0:
            return 0.0
        used_rows = sum(meta.rows_used for meta in self._meta.values())
        return max(0.0, 1.0 - (float(used_rows) / float(committed_rows)))

    def metrics_snapshot(self) -> dict[str, float | int]:
        committed_rows = len(self._pages) * self._rows_per_page
        used_rows = sum(meta.rows_used for meta in self._meta.values())
        pages_in_use = sum(1 for meta in self._meta.values() if meta.rows_used > 0)
        return {
            "pages_total": len(self._pages),
            "pages_in_use": pages_in_use,
            "committed_rows": committed_rows,
            "used_rows": used_rows,
            "fragmentation_ratio": self.fragmentation_ratio(),
            "compaction_count": self._compaction_count,
            "cow_events": self._cow_events,
            "allocator_failures": self._allocator_failures,
            "alloc_attempts": self._alloc_attempts,
            "python_hot_path_calls": self._python_hot_path_calls,
            "numpy_hot_path_calls": self._numpy_hot_path_calls,
        }

    def _alloc_single_row(self) -> RowRef:
        page_id = self._find_page_with_capacity()
        if page_id is None:
            page_id = self._create_page()
        free_rows = self._free_rows[page_id]
        if not free_rows:
            raise MemoryError("state pager out of rows")
        row_idx = int(free_rows.popleft())
        meta = self._meta[page_id]
        meta.rows_used += 1
        meta.refcount = meta.rows_used
        return RowRef(page_id=page_id, row_idx=row_idx)

    def _free_row(self, ref: RowRef) -> None:
        meta = self._meta.get(ref.page_id)
        if meta is None:
            return
        self._pages[ref.page_id][ref.row_idx, :] = 0.0
        free_rows = self._free_rows.setdefault(ref.page_id, deque())
        free_rows.appendleft(int(ref.row_idx))
        meta.rows_used = max(0, meta.rows_used - 1)
        meta.refcount = meta.rows_used

    def _find_page_with_capacity(self) -> int | None:
        for page_id, free_rows in self._free_rows.items():
            if free_rows:
                return page_id
        return None

    def _create_page(self) -> int:
        page_id = self._next_page_id
        self._next_page_id += 1
        self._pages[page_id] = np.zeros(
            (self._rows_per_page, self._dim),
            dtype=np.float32,
        )
        self._meta[page_id] = StatePageMeta(page_id=page_id)
        self._free_rows[page_id] = deque(range(self._rows_per_page))
        return page_id
