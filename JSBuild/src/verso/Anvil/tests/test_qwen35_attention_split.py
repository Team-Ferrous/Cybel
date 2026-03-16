from types import SimpleNamespace

import numpy as np
import core.native.qsg_forward as qsg_forward_module

from core.native.qsg_forward import QSGForwardPass


def test_split_qwen_q_and_gate_handles_multi_token_interleaved_batches():
    q_full = np.array(
        [
            [1.0, 2.0, 11.0, 12.0, 3.0, 4.0, 13.0, 14.0],
            [5.0, 6.0, 15.0, 16.0, 7.0, 8.0, 17.0, 18.0],
        ],
        dtype=np.float32,
    )

    q, gate = QSGForwardPass._split_qwen_q_and_gate(q_full, head_dim_hint=2)

    assert gate is not None
    np.testing.assert_array_equal(
        q,
        np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        gate,
        np.array(
            [
                [11.0, 12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0, 18.0],
            ],
            dtype=np.float32,
        ),
    )


def test_split_qwen_q_and_gate_uses_interleaved_head_layout():
    # Layout per head is [q_head, gate_head].
    q_full = np.array(
        [
            [
                1.0, 2.0, 101.0, 102.0,   # head 0
                3.0, 4.0, 103.0, 104.0,   # head 1
                5.0, 6.0, 105.0, 106.0,   # head 2
                7.0, 8.0, 107.0, 108.0,   # head 3
            ]
        ],
        dtype=np.float32,
    )

    q, gate = QSGForwardPass._split_qwen_q_and_gate(q_full, head_dim_hint=2)

    assert gate is not None
    np.testing.assert_array_equal(q, np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.float32))
    np.testing.assert_array_equal(
        gate,
        np.array([[101, 102, 103, 104, 105, 106, 107, 108]], dtype=np.float32),
    )


def test_split_qwen_q_and_gate_falls_back_to_half_split_when_head_dim_unknown():
    q_full = np.arange(12, dtype=np.float32).reshape(1, 12)

    q, gate = QSGForwardPass._split_qwen_q_and_gate(q_full, head_dim_hint=5)

    assert gate is not None
    np.testing.assert_array_equal(q, q_full[:, :6])
    np.testing.assert_array_equal(gate, q_full[:, 6:12])


def test_split_qwen_q_and_gate_returns_none_for_non_matrix_input():
    q_full = np.arange(8, dtype=np.float32)

    q, gate = QSGForwardPass._split_qwen_q_and_gate(q_full, head_dim_hint=2)

    assert gate is None
    np.testing.assert_array_equal(q, q_full)


def test_split_qwen_q_and_gate_falls_back_when_pair_layout_does_not_fit_total_dim():
    q_full = np.arange(10, dtype=np.float32).reshape(1, 10)

    q, gate = QSGForwardPass._split_qwen_q_and_gate(q_full, head_dim_hint=3)

    assert gate is not None
    np.testing.assert_array_equal(q, q_full[:, :5])
    np.testing.assert_array_equal(gate, q_full[:, 5:10])


def test_qwen_delta_step_matches_reference_orientation():
    # Non-symmetric state catches transpose mistakes.
    state = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    q = np.array([0.7, -0.2], dtype=np.float32)
    k = np.array([1.3, 0.4], dtype=np.float32)
    v = np.array([0.5, -0.1], dtype=np.float32)
    gate = -0.3
    beta = 0.6

    out, next_state = QSGForwardPass._qwen_delta_step(state, q, k, v, gate, beta)

    gated = state * np.exp(gate)
    expected_sk = gated @ k
    expected_d = (v - expected_sk) * beta
    expected_state = gated + np.outer(expected_d, k)
    expected_out = expected_state @ q

    np.testing.assert_allclose(next_state, expected_state.astype(np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out, expected_out.astype(np.float32), rtol=1e-6, atol=1e-6)


def test_qwen_delta_step_clips_gate_before_exponentiating():
    state = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float32)
    q = np.array([0.2, -0.4], dtype=np.float32)
    k = np.array([0.9, -0.3], dtype=np.float32)
    v = np.array([0.1, 0.7], dtype=np.float32)
    beta = 0.25

    out_hi, next_state_hi = QSGForwardPass._qwen_delta_step(
        state,
        q,
        k,
        v,
        gate=1e6,
        beta=beta,
    )
    out_cap, next_state_cap = QSGForwardPass._qwen_delta_step(
        state,
        q,
        k,
        v,
        gate=60.0,
        beta=beta,
    )

    np.testing.assert_allclose(next_state_hi, next_state_cap, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out_hi, out_cap, rtol=1e-6, atol=1e-6)

    out_lo, next_state_lo = QSGForwardPass._qwen_delta_step(
        state,
        q,
        k,
        v,
        gate=-1e6,
        beta=beta,
    )
    out_floor, next_state_floor = QSGForwardPass._qwen_delta_step(
        state,
        q,
        k,
        v,
        gate=-60.0,
        beta=beta,
    )

    np.testing.assert_allclose(next_state_lo, next_state_floor, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out_lo, out_floor, rtol=1e-6, atol=1e-6)


def test_qwen_forward_normalizes_false_like_mrope_metadata(monkeypatch):
    monkeypatch.setattr(
        qsg_forward_module,
        "NativeKVCache",
        lambda profile, max_seq_len: object(),  # noqa: ARG005
    )
    monkeypatch.setattr(qsg_forward_module, "_NativeKVCacheWrapper", None)

    metadata = {
        "qwen35.rope.mrope_interleaved": "0",
        "qwen35.rope.scaling.finetuned": "false",
    }
    weight_store = SimpleNamespace(
        loader=SimpleNamespace(get_metadata=lambda: metadata),
    )
    profile = SimpleNamespace(architecture="qwen35", embedding_dim=8)

    forward = QSGForwardPass(weight_store, profile, max_seq_len=16)

    assert forward.qwen_is_mrope is False
    assert forward.rope_finetuned is False
