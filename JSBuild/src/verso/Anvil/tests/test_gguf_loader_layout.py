from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from core.model.gguf_loader import GGUFModelLoader


def _make_loader_with_tensor(tensor) -> GGUFModelLoader:
    loader = GGUFModelLoader.__new__(GGUFModelLoader)
    loader._metadata = {}
    loader._embedding_tensor = None
    loader._vocab_tokens = None
    loader._reader = SimpleNamespace(tensors=[tensor], fields={})
    return loader


def test_get_tensor_preserves_reader_matrix_layout_for_f32():
    data = np.arange(8, dtype=np.float32).reshape(2, 4)
    tensor = SimpleNamespace(
        name="blk.0.example.weight",
        shape=(4, 2),  # GGUF metadata shape can be opposite of runtime matrix view.
        tensor_type=SimpleNamespace(value=0),
        data=data,
    )
    loader = _make_loader_with_tensor(tensor)

    got = loader.get_tensor("blk.0.example.weight")

    assert got is not None
    assert got.shape == (2, 4)
    np.testing.assert_array_equal(got, data)


def test_get_tensor_preserves_reader_matrix_layout_for_f16():
    data = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float16)
    tensor = SimpleNamespace(
        name="blk.0.example_f16.weight",
        shape=(3, 2),
        tensor_type=SimpleNamespace(value=1),
        data=data,
    )
    loader = _make_loader_with_tensor(tensor)

    got = loader.get_tensor("blk.0.example_f16.weight")

    assert got is not None
    assert got.shape == (2, 3)
    assert got.dtype == np.float16
    np.testing.assert_array_equal(got, data)
