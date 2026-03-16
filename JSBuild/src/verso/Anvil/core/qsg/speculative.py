from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.qsg.generator import QSGGenerator


@dataclass
class _DraftBundle:
    generator: QSGGenerator
    token_labels: list[str]

    def generate_drafts(
        self,
        context: np.ndarray,
        *,
        num_drafts: int,
        draft_length: int,
    ) -> list[np.ndarray]:
        context_embedding = np.asarray(context, dtype=np.float32)
        if context_embedding.ndim == 1:
            context_embedding = context_embedding.reshape(1, -1)
        drafts: list[np.ndarray] = []
        for _ in range(max(0, int(num_drafts))):
            tokens, _ = self.generator.generate_draft(
                context_embedding,
                seq_len=max(1, int(draft_length)),
                oracle_context={"tokens": self.token_labels},
            )
            drafts.append(np.asarray(tokens[0], dtype=np.int32))
        return drafts


class SpeculativeQSG:
    """Legacy speculative pipeline facade built on the unified QSG generator."""

    def __init__(self, adapter) -> None:
        self.adapter = adapter
        token_labels = [str(token) for token in list(adapter.loader.get_vocab_tokens())]
        self._generator = QSGGenerator(
            adapter.config,
            adapter.vocab_embeddings,
            propagator=adapter.propagator,
            lm_head_weight=adapter.lm_head_weight,
        )
        self.drafter = _DraftBundle(self._generator, token_labels)

    def generate(self, prompt: str, *, max_tokens: int = 16) -> str:
        context = self.adapter.get_embeddings(prompt)
        drafts = self.drafter.generate_drafts(
            context,
            num_drafts=max(1, int(getattr(self.adapter.config, "speculative_drafts", 1))),
            draft_length=max(1, int(max_tokens)),
        )
        lines = [str(prompt)]
        for index, draft in enumerate(drafts, start=1):
            token_line = " ".join(self._token_label(int(token_id)) for token_id in draft)
            lines.append(f"Block {index}: {token_line}")
        return "\n".join(lines)

    def _token_label(self, token_id: int) -> str:
        vocab = list(self.adapter.loader.get_vocab_tokens())
        if 0 <= token_id < len(vocab):
            return str(vocab[token_id])
        return f"<tok:{token_id}>"
