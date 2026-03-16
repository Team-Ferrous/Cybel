from types import SimpleNamespace

import numpy as np

from core.model.gguf_loader import GGUFModelLoader


def _loader_stub() -> GGUFModelLoader:
    loader = GGUFModelLoader.__new__(GGUFModelLoader)
    loader.get_embedding_dim = lambda: 4  # type: ignore[method-assign]
    loader.get_layer_count = lambda: 3  # type: ignore[method-assign]
    return loader


def test_extract_propagator_auto_prefers_profile_strategy_mlp(monkeypatch):
    loader = _loader_stub()
    mlp_prop = np.full((4, 4), 3.0, dtype=np.float32)
    attn_prop = np.full((4, 4), 7.0, dtype=np.float32)

    monkeypatch.setattr(
        loader, "_extract_propagator_mlp", lambda rank, layers: mlp_prop
    )
    monkeypatch.setattr(
        loader, "_extract_propagator_attn", lambda rank, layers: attn_prop
    )

    profile = SimpleNamespace(propagator_strategy="mlp")
    result = loader.extract_propagator(strategy="auto", profile=profile)

    assert np.array_equal(result, mlp_prop)


def test_extract_propagator_mlp_falls_back_to_attn_when_empty(monkeypatch):
    loader = _loader_stub()
    attn_prop = np.full((4, 4), 2.0, dtype=np.float32)

    monkeypatch.setattr(loader, "_extract_propagator_mlp", lambda rank, layers: None)
    monkeypatch.setattr(
        loader, "_extract_propagator_attn", lambda rank, layers: attn_prop
    )

    profile = SimpleNamespace(propagator_strategy="mlp")
    result = loader.extract_propagator(strategy="auto", profile=profile)

    assert np.array_equal(result, attn_prop)


def test_extract_propagator_mlp_builds_nontrivial_transition(monkeypatch):
    loader = _loader_stub()
    loader._svd_approximation = lambda matrix, rank: matrix  # type: ignore[method-assign]

    gate_by_layer = {
        0: np.array(
            [
                [1.0, 0.5, 0.0, 0.0],
                [0.0, 1.0, 0.5, 0.0],
                [0.0, 0.0, 1.0, 0.5],
                [0.5, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        1: np.eye(4, dtype=np.float32) * 2.0,
    }
    down_by_layer = {
        0: np.array(
            [
                [0.8, 0.0, 0.0, 0.0],
                [0.0, 0.8, 0.0, 0.0],
                [0.0, 0.0, 0.8, 0.0],
                [0.0, 0.0, 0.0, 0.8],
            ],
            dtype=np.float32,
        ),
        1: np.eye(4, dtype=np.float32) * 0.5,
    }

    def _first_tensor(layer_idx, suffixes):
        if suffixes[0].startswith("ffn_gate"):
            return gate_by_layer.get(layer_idx)
        return down_by_layer.get(layer_idx)

    monkeypatch.setattr(loader, "_first_tensor_for_keys", _first_tensor)

    propagator = loader._extract_propagator_mlp(rank=4, layers=3)

    assert propagator is not None
    assert propagator.shape == (4, 4)
    assert np.isfinite(propagator).all()
    assert float(np.linalg.norm(propagator)) > 0.0
