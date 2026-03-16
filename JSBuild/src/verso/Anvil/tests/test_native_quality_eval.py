from __future__ import annotations

import json
from pathlib import Path

from audit.eval import native_logits


class _FakeEngine:
    def __init__(self, model: str, context_length: int = 2048):  # noqa: ARG002
        self.model = model
        self.context_length = context_length
        self.profile = type("Profile", (), {"vocab_size": 6})()

    def reset_kv_cache(self) -> None:
        return None

    def close(self) -> None:
        return None

    def tokenize(self, text: str) -> list[int]:
        mapping = {
            "alpha": [0, 1, 2],
            "beta": [0, 2, 3],
            "throughput": [1],
            "time to first token": [2],
            "cache hit ratio": [3],
            "context length": [4],
            "perf_event_paranoid": [2],
            "OMP_PLACES": [1],
            "transparent huge pages": [3],
            "top_p": [4],
            "model digest": [2],
            "git branch name": [1],
            "wall clock timestamp": [3],
            "prompt": [0, 1],
        }
        return list(mapping.get(text, [0, 1, 2]))

    def prepare_prompt_tokens(self, prompt: str) -> list[int]:
        return [0, 1]

    def _get_logits_for_tokens(self, prefix_tokens: list[int]) -> list[float]:
        target = (sum(prefix_tokens) + len(prefix_tokens)) % 5 + 1
        logits = [-4.0] * 6
        logits[target] = 4.0
        logits[(target + 1) % 6] = 2.0
        return logits


def test_evaluate_perplexity_and_confidence(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(native_logits, "NativeQSGEngine", _FakeEngine)

    perplexity_path = tmp_path / "perplexity.jsonl"
    perplexity_path.write_text(
        json.dumps({"sample_id": "p1", "text": "alpha"})
        + "\n"
        + json.dumps({"sample_id": "p2", "text": "beta"})
        + "\n",
        encoding="utf-8",
    )
    confidence_path = tmp_path / "confidence.jsonl"
    confidence_path.write_text(
        json.dumps(
            {
                "sample_id": "c1",
                "prompt": "prompt",
                "options": {
                    "A": "throughput",
                    "B": "time to first token",
                    "C": "cache hit ratio",
                    "D": "context length",
                },
                "correct_option": "B",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    ppl = native_logits.evaluate_perplexity(
        model="qwen3.5:4b",
        corpus_path=perplexity_path,
        context_length=128,
    )
    conf = native_logits.evaluate_confidence(
        model="qwen3.5:4b",
        corpus_path=confidence_path,
        context_length=128,
    )

    assert ppl["tokens_scored"] > 0
    assert ppl["perplexity"] > 0.0
    assert "documents" in ppl
    assert 0.0 <= conf["mean_token_confidence"] <= 1.0
    assert conf["entropy_p95"] >= 0.0
    assert "records" in conf
    assert conf["tokens_scored"] > 0
    assert conf["records"][0]["format"] == "mcq"
    assert "option_probabilities" in conf["records"][0]
