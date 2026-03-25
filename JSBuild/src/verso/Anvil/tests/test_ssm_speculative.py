from __future__ import annotations

import numpy as np
import pytest

from core.native.cpu_speculative_decode import (
    CPUSpeculativeDecoder,
    SSMSelfSpeculativeDecoder,
    SpeculativeConfig,
    SpeculativeFallbackDisabledError,
)


class _LLMStub:
    @staticmethod
    def token_eos() -> int:
        return -1


class _EngineStub:
    def __init__(self, fallback_tokens: list[int] | None = None):
        self.llm = _LLMStub()
        self._fallback_tokens = list(fallback_tokens or [42])

    def generate_stream(self, *_args, **kwargs):
        max_new_tokens = int(kwargs.get("max_new_tokens", 1))
        for token in self._fallback_tokens[:max_new_tokens]:
            yield token

    def token_eos(self) -> int:
        return self.llm.token_eos()


def _minimal_coconut_resources() -> dict:
    ident = np.eye(3, dtype=np.float32)
    return {
        "vocab_embeddings": ident,
        "lm_head_weight": ident,
        "evolution_weights": {
            "norm_gamma": np.ones((3,), dtype=np.float32),
            "norm_beta": np.zeros((3,), dtype=np.float32),
            "w1": ident,
            "b1": np.zeros((3,), dtype=np.float32),
            "w2": ident,
            "b2": np.zeros((3,), dtype=np.float32),
            "hidden_dim": 3,
        },
    }


def test_ssm_speculative_accepts_verified_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    decoder = CPUSpeculativeDecoder(
        native_engine=_EngineStub(fallback_tokens=[99]),
        coconut_bridge=None,
        config=SpeculativeConfig(
            use_coconut_drafts=False,
            fallback_to_top_k=True,
            max_draft_length=4,
            acceptance_threshold=0.5,
        ),
    )

    monkeypatch.setattr(
        decoder,
        "_draft_sequence_top_k_with_scores",
        lambda _logits: ([5, 6], [0.9, 0.8]),
    )
    monkeypatch.setattr(
        decoder,
        "_get_logits_for_tokens",
        lambda _tokens: np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
    )
    monkeypatch.setattr(
        decoder,
        "_verify_candidates_batch",
        lambda *_args, **_kwargs: (2, [1.0, 1.0]),
    )

    generated = list(
        decoder.generate_speculative(
            prompt_tokens=[1, 2],
            max_new_tokens=2,
            temperature=0.0,
        )
    )

    assert generated == [5, 6]


def test_ssm_speculative_rejection_falls_back_to_standard_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    decoder = CPUSpeculativeDecoder(
        native_engine=_EngineStub(fallback_tokens=[77]),
        coconut_bridge=None,
        config=SpeculativeConfig(
            use_coconut_drafts=False,
            fallback_to_top_k=True,
            fallback_to_standard_generation=True,
            max_draft_length=2,
        ),
    )

    monkeypatch.setattr(
        decoder,
        "_draft_sequence_top_k_with_scores",
        lambda _logits: ([9], [0.6]),
    )
    monkeypatch.setattr(
        decoder,
        "_get_logits_for_tokens",
        lambda _tokens: np.asarray([0.3, 0.4, 0.3], dtype=np.float32),
    )
    monkeypatch.setattr(
        decoder,
        "_verify_candidates_batch",
        lambda *_args, **_kwargs: (0, [0.05]),
    )

    generated = list(
        decoder.generate_speculative(
            prompt_tokens=[3],
            max_new_tokens=1,
            temperature=0.0,
        )
    )

    assert generated == [77]


def test_ssm_speculative_strict_mode_raises_on_full_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    decoder = CPUSpeculativeDecoder(
        native_engine=_EngineStub(fallback_tokens=[88]),
        coconut_bridge=object(),
        config=SpeculativeConfig(
            use_coconut_drafts=True,
            strict_native_only=True,
            max_draft_length=1,
            acceptance_threshold=0.95,
        ),
    )

    monkeypatch.setattr(
        decoder,
        "_get_logits_for_tokens",
        lambda _tokens: np.asarray([0.1, 0.7, 0.2], dtype=np.float32),
    )
    monkeypatch.setattr(
        decoder,
        "_draft_candidates_coconut_with_scores",
        lambda *_args, **_kwargs: ([1], [0.9]),
    )
    monkeypatch.setattr(
        decoder,
        "_verify_candidates_batch",
        lambda *_args, **_kwargs: (0, [0.01]),
    )

    with pytest.raises(
        SpeculativeFallbackDisabledError,
        match="rejected all candidates",
    ):
        list(
            decoder.generate_speculative(
                prompt_tokens=[1],
                max_new_tokens=1,
                temperature=0.0,
                coconut_resources=_minimal_coconut_resources(),
            )
        )


def test_ssm_self_spec_decoder_generates_tokens_from_verify_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _EngineStub(fallback_tokens=[88])
    decoder = SSMSelfSpeculativeDecoder(
        native_engine=engine,
        max_draft_length=3,
        acceptance_threshold=0.5,
    )

    # Deterministic logits: token 2 preferred first, then EOS.
    logits_map = {
        (1,): np.asarray([0.1, 0.2, 0.9], dtype=np.float32),
        (1, 2): np.asarray([0.1, 0.9, 0.2], dtype=np.float32),
    }
    monkeypatch.setattr(
        decoder,
        "_engine_logits",
        lambda tokens: logits_map.get(tuple(tokens), np.asarray([0.1, 0.9, 0.2], dtype=np.float32)),
    )

    generated = list(
        decoder.generate(
            prompt_tokens=[1],
            max_new_tokens=2,
            temperature=0.0,
        )
    )
    assert generated == [2, 1]


def test_ssm_softmax_is_finite_for_non_finite_logits() -> None:
    decoder = SSMSelfSpeculativeDecoder(native_engine=_EngineStub(), max_draft_length=2)

    probs = decoder._softmax(
        np.asarray([np.nan, np.inf, -np.inf], dtype=np.float32),
        temperature=0.6,
    )

    assert np.all(np.isfinite(probs))
    assert np.all(probs >= 0.0)
    np.testing.assert_allclose(float(np.sum(probs)), 1.0, atol=1e-5)


def test_ssm_draft_uses_finite_argmax_when_temperature_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = SSMSelfSpeculativeDecoder(native_engine=_EngineStub(), max_draft_length=1)
    monkeypatch.setattr(
        decoder,
        "_engine_logits",
        lambda _tokens: np.asarray([np.nan, 4.0, -np.inf], dtype=np.float32),
    )

    drafted, draft_probs = decoder.draft(
        prompt_tokens=[1],
        max_draft_tokens=1,
        temperature=0.0,
    )

    assert drafted == [1]
    assert len(draft_probs) == 1


def test_ssm_apply_logits_processor_sanitizes_non_finite_outputs() -> None:
    decoder = SSMSelfSpeculativeDecoder(native_engine=_EngineStub(), max_draft_length=1)
    logits = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)

    def _bad_processor(_tokens, _scores):
        return np.asarray([np.nan, np.inf, -np.inf], dtype=np.float32)

    updated = decoder._apply_logits_processor([1], logits, [_bad_processor])

    assert updated.shape == logits.shape
    assert np.all(np.isfinite(updated))
