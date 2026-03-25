"""Minimal KV cache for native QSG generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.model.model_profile import ModelProfile


@dataclass
class KVEntry:
    k: np.ndarray
    v: np.ndarray


class NativeKVCache:
    def __init__(self, profile: ModelProfile, max_seq_len: int = 8192):
        self.profile = profile
        self.max_seq_len = max_seq_len
        self._caches: dict[int, KVEntry] = {}

    def append(self, layer_idx: int, k: np.ndarray, v: np.ndarray, pos: int) -> None:
        k_f = np.asarray(k, dtype=np.float32)
        v_f = np.asarray(v, dtype=np.float32)
        entry = self._caches.get(layer_idx)
        if entry is None or pos == 0:
            self._caches[layer_idx] = KVEntry(k=k_f.copy(), v=v_f.copy())
            return
        entry.k = np.concatenate([entry.k, k_f], axis=0)[-self.max_seq_len :]
        entry.v = np.concatenate([entry.v, v_f], axis=0)[-self.max_seq_len :]

    def get(self, layer_idx: int) -> tuple[np.ndarray, np.ndarray]:
        entry = self._caches.get(layer_idx)
        if entry is None:
            empty = np.zeros((0, self.profile.embedding_dim), dtype=np.float32)
            return empty, empty
        return entry.k, entry.v

    def reset(self) -> None:
        self._caches.clear()

    def get_current_length(self) -> int:
        if not self._caches:
            return 0
        first = next(iter(self._caches.values()))
        return int(first.k.shape[0])
