import numpy as np
import pytest

from core.native.cpu_speculative_decode import CPUSpeculativeDecoder, SpeculativeConfig


class _LLMStub:
    @staticmethod
    def token_eos() -> int:
        return -1


class _EngineStub:
    def __init__(self):
        self.llm = _LLMStub()

    def generate_stream(self, *_args, **_kwargs):
        yield 0


class _BridgeStub:
    def expand_paths(self, hidden_state, num_paths, noise_scale=0.01):
        return np.repeat(hidden_state[:, None, :], num_paths, axis=1)

    def evolve_paths(self, paths, *_args):
        paths += 0.0

    def score_paths(self, paths, _context):
        batch, num_paths, _ = paths.shape
        return np.ones((batch, num_paths), dtype=np.float32)


def test_logits_proxy_state_is_used_for_coconut_drafts(monkeypatch):
    decoder = CPUSpeculativeDecoder(
        native_engine=_EngineStub(),
        coconut_bridge=object(),
        config=SpeculativeConfig(
            num_candidates=2,
            acceptance_threshold=0.5,
            max_draft_length=1,
            use_coconut_drafts=True,
        ),
    )

    vocab_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    logits = np.array([0.0, 1.5, 2.0, -1.0], dtype=np.float32)

    monkeypatch.setattr(
        decoder, "_get_logits_for_tokens", lambda _tokens: logits.copy()
    )

    captured = {}

    def _capture_context(context_embedding, *_args, **_kwargs):
        captured["context"] = np.asarray(context_embedding, dtype=np.float32).copy()
        return [2, 1], [0.9, 0.1]

    monkeypatch.setattr(decoder, "_draft_candidates_coconut_with_scores", _capture_context)
    monkeypatch.setattr(
        decoder, "_verify_candidates_batch", lambda *_a, **_k: (1, [0.9])
    )

    outputs = list(
        decoder.generate_speculative(
            prompt_tokens=[0],
            max_new_tokens=1,
            temperature=1.0,
            coconut_resources={
                "vocab_embeddings": vocab_embeddings,
                "lm_head_weight": vocab_embeddings,
                "evolution_weights": {
                    "norm_gamma": np.ones((3,), dtype=np.float32),
                    "norm_beta": np.zeros((3,), dtype=np.float32),
                    "w1": np.eye(3, dtype=np.float32),
                    "b1": np.zeros((3,), dtype=np.float32),
                    "w2": np.eye(3, dtype=np.float32),
                    "b2": np.zeros((3,), dtype=np.float32),
                    "hidden_dim": 3,
                },
            },
        )
    )
    assert outputs == [2]
    assert "context" in captured

    centered = logits - logits.max()
    weights = np.exp(centered) / np.exp(centered).sum()
    expected = (weights[:, None] * vocab_embeddings).sum(axis=0, keepdims=True)
    np.testing.assert_allclose(captured["context"], expected, atol=1e-6)
    # Verify this is not simply the last-token embedding.
    assert not np.allclose(captured["context"], vocab_embeddings[0:1])


def test_verify_candidates_uses_prefix_distribution_for_each_step(monkeypatch):
    decoder = CPUSpeculativeDecoder(
        native_engine=_EngineStub(),
        coconut_bridge=None,
        config=SpeculativeConfig(
            num_candidates=3,
            acceptance_threshold=0.2,
            max_draft_length=3,
            use_coconut_drafts=False,
        ),
    )

    calls = []

    def _logits_for_prefix(tokens):
        key = tuple(tokens)
        calls.append(key)
        logits = np.zeros((8,), dtype=np.float32)
        if key == (10,):
            logits[2] = 6.0
        elif key == (10, 2):
            logits[3] = 6.0
        elif key == (10, 2, 3):
            logits[4] = 6.0
        return logits

    monkeypatch.setattr(decoder, "_get_logits_for_tokens", _logits_for_prefix)

    accepted_length, probs = decoder._verify_candidates_batch(
        prompt_tokens=[10],
        candidate_sequences=[[2], [2, 3], [2, 3, 4]],
        temperature=1.0,
    )

    assert accepted_length == 3
    assert len(probs) == 3
    assert calls == [(10,), (10, 2), (10, 2, 3)]


def test_strict_mode_raises_when_coconut_drafts_unavailable():
    decoder = CPUSpeculativeDecoder(
        native_engine=_EngineStub(),
        coconut_bridge=None,
        config=SpeculativeConfig(
            num_candidates=2,
            acceptance_threshold=0.5,
            max_draft_length=1,
            use_coconut_drafts=True,
            strict_native_only=True,
        ),
    )

    with pytest.raises(
        RuntimeError, match="disallows top-k fallback when no COCONUT drafts"
    ):
        list(
            decoder.generate_speculative(
                prompt_tokens=[7],
                max_new_tokens=1,
                temperature=1.0,
                coconut_resources=None,
            )
        )


def test_strict_mode_raises_when_verification_rejects_all(monkeypatch):
    decoder = CPUSpeculativeDecoder(
        native_engine=_EngineStub(),
        coconut_bridge=object(),
        config=SpeculativeConfig(
            num_candidates=2,
            acceptance_threshold=0.9,
            max_draft_length=2,
            use_coconut_drafts=True,
            strict_native_only=True,
        ),
    )

    monkeypatch.setattr(
        decoder,
        "_draft_candidates_coconut_with_scores",
        lambda *_a, **_k: ([1, 2], [0.6, 0.4]),
    )
    monkeypatch.setattr(
        decoder, "_verify_candidates_batch", lambda *_a, **_k: (0, [0.1, 0.1])
    )
    monkeypatch.setattr(
        decoder,
        "_get_logits_for_tokens",
        lambda _tokens: np.array([0.0, 2.0, 1.0], dtype=np.float32),
    )

    with pytest.raises(
        RuntimeError,
        match="Speculative verification rejected all candidates and standard-generation fallback is disabled",
    ):
        list(
            decoder.generate_speculative(
                prompt_tokens=[1],
                max_new_tokens=1,
                temperature=1.0,
                coconut_resources={
                    "vocab_embeddings": np.eye(3, dtype=np.float32),
                    "lm_head_weight": np.eye(3, dtype=np.float32),
                    "evolution_weights": {
                        "norm_gamma": np.ones((3,), dtype=np.float32),
                        "norm_beta": np.zeros((3,), dtype=np.float32),
                        "w1": np.eye(3, dtype=np.float32),
                        "b1": np.zeros((3,), dtype=np.float32),
                        "w2": np.eye(3, dtype=np.float32),
                        "b2": np.zeros((3,), dtype=np.float32),
                        "hidden_dim": 3,
                    },
                },
            )
        )


def test_strict_mode_accepts_when_draft_probs_are_forwarded(monkeypatch):
    decoder = CPUSpeculativeDecoder(
        native_engine=_EngineStub(),
        coconut_bridge=object(),
        config=SpeculativeConfig(
            num_candidates=2,
            acceptance_threshold=0.9,
            max_draft_length=1,
            use_coconut_drafts=True,
            strict_native_only=True,
        ),
    )

    monkeypatch.setattr(
        decoder,
        "_draft_candidates_coconut_with_scores",
        lambda *_a, **_k: ([1], [0.2]),
    )
    monkeypatch.setattr(
        decoder,
        "_get_logits_for_tokens",
        lambda _tokens: np.array(
            [-20.0, np.log(0.2), np.log(0.8)], dtype=np.float32
        ),
    )

    outputs = list(
        decoder.generate_speculative(
            prompt_tokens=[7],
            max_new_tokens=1,
            temperature=1.0,
            coconut_resources={
                "vocab_embeddings": np.eye(3, dtype=np.float32),
                "lm_head_weight": np.eye(3, dtype=np.float32),
                "evolution_weights": {
                    "norm_gamma": np.ones((3,), dtype=np.float32),
                    "norm_beta": np.zeros((3,), dtype=np.float32),
                    "w1": np.eye(3, dtype=np.float32),
                    "b1": np.zeros((3,), dtype=np.float32),
                    "w2": np.eye(3, dtype=np.float32),
                    "b2": np.zeros((3,), dtype=np.float32),
                    "hidden_dim": 3,
                },
            },
        )
    )

    assert outputs == [1]


def test_coconut_drafts_form_a_short_sequence_not_parallel_alternatives():
    decoder = CPUSpeculativeDecoder(
        native_engine=_EngineStub(),
        coconut_bridge=_BridgeStub(),
        config=SpeculativeConfig(
            num_candidates=3,
            acceptance_threshold=0.5,
            max_draft_length=3,
            use_coconut_drafts=True,
            strict_native_only=True,
        ),
    )

    vocab_embeddings = np.eye(3, dtype=np.float32)
    draft_tokens, draft_probs = decoder._draft_candidates_coconut_with_scores(
        context_embedding=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        vocab_embeddings=vocab_embeddings,
        lm_head_weight=vocab_embeddings,
        evolution_weights={
            "norm_gamma": np.ones((3,), dtype=np.float32),
            "norm_beta": np.zeros((3,), dtype=np.float32),
            "w1": np.eye(3, dtype=np.float32),
            "b1": np.zeros((3,), dtype=np.float32),
            "w2": np.eye(3, dtype=np.float32),
            "b2": np.zeros((3,), dtype=np.float32),
            "hidden_dim": 3,
        },
        num_candidates=decoder.config.num_candidates,
    )

    assert len(draft_tokens) == 3
    assert len(draft_probs) == 3
    assert draft_tokens == [0, 0, 0]


def test_top_k_fallback_produces_one_token_sequence():
    decoder = CPUSpeculativeDecoder(
        native_engine=_EngineStub(),
        coconut_bridge=None,
        config=SpeculativeConfig(use_coconut_drafts=False),
    )

    tokens, probs = decoder._draft_sequence_top_k_with_scores(
        np.array([0.2, 0.7, 0.1], dtype=np.float32)
    )

    assert tokens == [1]
    assert len(probs) == 1
