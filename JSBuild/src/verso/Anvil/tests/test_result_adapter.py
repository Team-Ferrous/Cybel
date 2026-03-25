import numpy as np

from core.reasoning.result_adapter import ResultEmbeddingAdapter


class _BrainStub:
    def embeddings(self, text: str):
        base = np.arange(8, dtype=np.float32)
        scale = float((len(text) % 7) + 1)
        return (base + 1.0) * scale


def test_result_adapter_injects_and_normalizes():
    adapter = ResultEmbeddingAdapter(
        brain=_BrainStub(), embedding_dim=8, residual_weight=0.6
    )
    pre = np.ones((1, 8), dtype=np.float32)
    merged = adapter.inject(
        tool_result="Found matching implementation in core/x.py",
        pre_tool_state=pre,
        tool_name="saguaro_query",
    )
    assert merged.shape == (1, 8)
    assert np.all(np.isfinite(merged))
    norm = float(np.linalg.norm(merged[0]))
    assert abs(norm - 1.0) < 1e-5


def test_result_adapter_projects_with_dimension_coercion():
    adapter = ResultEmbeddingAdapter(brain=_BrainStub(), embedding_dim=6)
    projected = adapter.project_text("short text")
    assert projected.shape == (1, 6)
    assert np.all(np.isfinite(projected))
