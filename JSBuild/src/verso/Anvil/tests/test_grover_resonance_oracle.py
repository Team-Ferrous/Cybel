import numpy as np

from core.qsg.grover import GroverAmplifier


class _SemanticEngine:
    def get_context_for_objective(self, _context: str):
        return ["core/engine.py", "core/coconut.py"]


def test_resonance_oracle_uses_latent_and_repo_sources() -> None:
    amplifier = GroverAmplifier(semantic_engine=_SemanticEngine())
    logits = np.asarray([[0.2, 0.2, 0.19, 0.18]], dtype=np.float32)
    tokens = ["engine", "coconut", "banana", "apple"]
    token_embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [-1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    telemetry: dict[str, float] = {}

    resonant = amplifier.amplify_with_resonance(
        logits,
        tokens,
        "engine coconut repo delta",
        iterations=1,
        token_embeddings=token_embeddings,
        latent_prior=[1.0, 0.0],
        repo_delta={
            "changed_paths": ["core/engine.py"],
            "summary_text": "engine change",
        },
        invariant_terms=["coconut"],
        telemetry_sink=telemetry,
    )

    top_two = set(np.argsort(resonant[0])[-2:].tolist())
    assert {0, 1}.issubset(top_two)
    assert telemetry["grover_oracle_source_count"] >= 3.0
    assert telemetry["grover_resonance_mean"] > 0.0
