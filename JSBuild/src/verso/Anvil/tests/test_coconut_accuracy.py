import numpy as np

from core.reasoning.coconut import ContinuousThoughtBlock


class _UniformBackend:
    def __init__(self):
        self.calls = 0
        self.last_amplitudes = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

    def get_device_info(self):
        return {"backend": "test"}

    def explore(self, embedding):
        self.calls += 1
        # Keep amplitudes uniform to trigger diversification fallback.
        self.last_amplitudes = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        return embedding + (self.calls * 0.01)


class _SignaledBackend:
    def __init__(self):
        self.calls = 0
        self.last_amplitudes = np.array([0.7, 0.2, 0.1], dtype=np.float32)

    def get_device_info(self):
        return {"backend": "test"}

    def explore(self, embedding):
        self.calls += 1
        self.last_amplitudes = np.array([0.7, 0.2, 0.1], dtype=np.float32)
        return embedding


def test_coconut_fallback_diversifies_paths(monkeypatch):
    backend = _UniformBackend()
    monkeypatch.setattr("core.reasoning.coconut.get_backend", lambda *_a, **_k: backend)

    coconut = ContinuousThoughtBlock(embedding_dim=8, num_paths=4, steps=2)
    emb = np.ones((1, 8), dtype=np.float32)
    out = coconut.explore(emb)

    assert out.shape == emb.shape
    # 1 primary path + 4 diversified paths
    assert backend.calls == 5
    assert coconut.amplitudes is not None
    assert float(np.max(coconut.amplitudes) - np.min(coconut.amplitudes)) > 0.0


def test_coconut_keeps_primary_when_amplitudes_have_signal(monkeypatch):
    backend = _SignaledBackend()
    monkeypatch.setattr("core.reasoning.coconut.get_backend", lambda *_a, **_k: backend)

    coconut = ContinuousThoughtBlock(embedding_dim=8, num_paths=4, steps=2)
    emb = np.ones((1, 8), dtype=np.float32)
    out = coconut.explore(emb)

    assert out.shape == emb.shape
    assert backend.calls == 1
