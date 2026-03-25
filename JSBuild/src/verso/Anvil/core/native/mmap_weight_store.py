"""Optional mmap-oriented weight store for native QSG."""

from __future__ import annotations

import mmap
from typing import Optional

import numpy as np
from gguf import dequantize

from core.model.gguf_loader import GGUFModelLoader
from core.model.model_profile import ModelProfile
from core.native.weight_store import WeightStore


class MMapWeightStore(WeightStore):
    """WeightStore variant that prefers zero-copy tensor views from GGUF reader buffers."""

    def __init__(self, loader: GGUFModelLoader, profile: ModelProfile):
        super().__init__(loader=loader, profile=profile)
        self._mmap_dense_only = True
        # Phase 2 invariant: keep dense tensors zero-copy when possible.
        self._promote_f16_cache = False

    def get_tensor(self, name: str) -> Optional[np.ndarray]:
        if name in self._cache:
            return self._cache[name]

        tensor = self._tensor_index.get(name)
        if tensor is None:
            return None

        try:
            qtype = int(tensor.tensor_type.value)
        except Exception:
            return super().get_tensor(name)

        if qtype == 0:
            arr = np.asarray(tensor.data, dtype=np.float32)
        elif qtype == 1:
            arr = np.asarray(tensor.data, dtype=np.float16)
        elif self._mmap_dense_only:
            arr = super().get_tensor(name)
            if arr is None:
                return None
            self._cache[name] = arr
            return arr
        else:
            arr = np.asarray(dequantize(tensor.data, tensor.tensor_type))

        if arr.ndim == 1 and len(getattr(tensor, "shape", ())) > 1:
            expected = int(np.prod(tensor.shape))
            if arr.size == expected:
                arr = arr.reshape(tensor.shape)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        self._cache[name] = arr
        return arr


class MmapWeightStore:
    """Low-level mmap accessor used by tooling/tests."""

    def __init__(self, gguf_path: str):
        self._fd = open(gguf_path, "rb")
        self._mm = mmap.mmap(self._fd.fileno(), 0, access=mmap.ACCESS_READ)

    def get_bytes(self, offset: int, size: int) -> bytes:
        start = int(offset)
        end = start + int(size)
        return self._mm[start:end]

    def get_view(self, offset: int, size: int) -> memoryview:
        start = int(offset)
        end = start + int(size)
        return memoryview(self._mm)[start:end]

    def close(self) -> None:
        self._mm.close()
        self._fd.close()
