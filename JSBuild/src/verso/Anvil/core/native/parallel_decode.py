"""Parallel decoding helpers for the native QSG engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from core.native import simd_ops_wrapper as simd_ops


@dataclass
class JacobiDecodeResult:
    tokens: list[int]
    accepted: int


class JacobiDecoder:
    """Greedy multi-token drafting with single-pass verification."""

    def __init__(self, width: int = 4):
        self.width = max(1, int(width))

    def decode(
        self,
        engine,
        prompt_tokens: list[int],
        max_tokens: int,
        temperature: float,
        logits_processor: Optional[Callable] = None,
    ) -> JacobiDecodeResult:
        if max_tokens <= 0:
            return JacobiDecodeResult(tokens=[], accepted=0)

        draft: list[int] = []
        working = list(prompt_tokens)
        for _ in range(min(self.width, max_tokens)):
            logits = engine._get_logits_for_tokens(working)
            if logits_processor is not None:
                logits = engine._apply_logits_processors(working, logits, logits_processor)
            # Jacobi drafting must stay deterministic in strict native mode so the
            # verification pass is not guaranteed to reject temperature-sampled
            # tokens at position zero.
            token = int(simd_ops.argmax(logits))
            draft.append(token)
            working.append(token)
            if token == engine.token_eos():
                break

        if not draft:
            return JacobiDecodeResult(tokens=[], accepted=0)

        accepted = 0
        for idx, token in enumerate(draft):
            prefix = prompt_tokens + draft[:idx]
            logits = engine._get_logits_for_tokens(prefix)
            if logits_processor is not None:
                logits = engine._apply_logits_processors(prefix, logits, logits_processor)
            predicted = int(simd_ops.argmax(logits))
            if predicted != token:
                break
            accepted += 1
            if token == engine.token_eos():
                break

        return JacobiDecodeResult(tokens=draft[:accepted], accepted=accepted)
