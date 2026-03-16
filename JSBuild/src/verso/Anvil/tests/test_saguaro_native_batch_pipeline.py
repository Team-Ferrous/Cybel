from __future__ import annotations

import numpy as np

from saguaro.indexing.native_indexer_bindings import NativeIndexer


class _DummyIndexer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def full_pipeline(self, **kwargs):
        texts = list(kwargs["texts"])
        self.calls.append(texts)
        target_dim = int(kwargs["target_dim"])
        return np.ones((len(texts), target_dim), dtype=np.float32)


def test_full_pipeline_batched_uses_one_native_call() -> None:
    dummy = _DummyIndexer()
    projection = np.zeros((8, 4), dtype=np.float32)

    output = NativeIndexer.full_pipeline_batched(
        dummy,
        texts=["alpha", "beta", "gamma"],
        projection=projection,
        vocab_size=8,
        batch_capacity=1,
        target_dim=6,
    )

    assert len(dummy.calls) == 1
    assert dummy.calls[0] == ["alpha", "beta", "gamma"]
    assert output.shape == (3, 6)
