from typing import Any, Optional, Tuple

import numpy as np

from core.qsg.config import QSGConfig
from core.qsg.grover import GroverAmplifier
from core.qsg.jacobi_refiner import JacobiRefiner
from core.qsg.hopfield_vocab import HopfieldVocab
from core.qsg.holographic_encoder import HolographicEncoder
from core.qsg.phase_controller import AdaptivePhaseController
from core.reasoning.coconut import ContinuousThoughtBlock

# from core.qsg.ollama_adapter import OllamaQSGAdapter # Circular import prevention


class QSGGenerator:
    """
    Quantum-Inspired Speculative Generator (QSG).

    Orchestrates the 5-phase parallel generation pipeline:
    1. Context Entanglement  (via Adapter/Embeddings)
    2. Vocabulary Superposition (Hopfield)
    3. Position Coherence (Jacobi)
    4. Grover Amplification
    5. Final Selection
    """

    def __init__(
        self,
        config: QSGConfig,
        vocab_embeddings: np.ndarray,
        propagator: Optional[np.ndarray] = None,
        lm_head_weight: Optional[np.ndarray] = None,
    ):
        self.config = config
        self.vocab_embeddings = vocab_embeddings
        self.propagator = propagator

        # Use LM head weight for token probs if available, else fall back to embeddings
        hopfield_patterns = (
            lm_head_weight if lm_head_weight is not None else vocab_embeddings
        )

        # Primitives
        self.hopfield = HopfieldVocab(hopfield_patterns, beta=config.hopfield_beta)

        self.grover = GroverAmplifier()
        self.jacobi = JacobiRefiner(iterations=config.jacobi_iterations)

        # HSE Encoder
        self.encoder = HolographicEncoder(dim=vocab_embeddings.shape[1])

        # Reasoning
        self.thought_block = (
            ContinuousThoughtBlock(embedding_dim=vocab_embeddings.shape[1])
            if config.use_coconut_reasoning
            else None
        )

        # Phase Controller
        self.phase_controller = AdaptivePhaseController(config)
        self.last_generation_trace: dict[str, Any] = {}

    def generate_draft(
        self,
        context_embedding: np.ndarray,
        seq_len: int,
        oracle_context: Optional[dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a parallel draft sequence.

        Args:
           context_embedding: [1, Dim] or [Batch, Dim] embedding of the prompt context.
           seq_len: Number of tokens to generate in parallel.

        Returns:
           tokens: [Batch, SeqLen] integer token IDs.
           probs: [Batch, SeqLen, Vocab] probabilities.
        """
        # 0. CoCoNut Reasoning (Latent Thought)
        if self.thought_block and self.config.use_coconut_reasoning:
            context_embedding = self.thought_block.explore(context_embedding)

        # 1. Project Context to Future Positions
        # Simple heuristic: Decay context influence over time or position
        # Real impl would use MPS Entanglement (Roadmap Phase 2.1)
        # For now: Repeat context for all positions + noise/position embedding

        if len(context_embedding.shape) == 3:
            # [Batch, ContextLen, Dim] -> [Batch, Dim] using HSE
            context_embedding = self.encoder.encode(context_embedding)

        # 1.5 State Evolution (Propagator)
        # Predict next state: |psi_next> = U * |psi_current>
        # if self.propagator is not None:
        #      original_state = context_embedding.copy()
        #      # context_embedding: [Batch, Dim]
        #      # propagator: [Dim, Dim]
        #      # Result: [Batch, Dim]
        #      propagated_state = context_embedding @ self.propagator.T

        #      # Phase 3: Tunable Coherence
        #      # Interpolate between current state and predicted next state
        #      # coherence_range (0-100) -> alpha (0.0-1.0)
        #      alpha = np.clip(self.config.coherence_range / 100.0, 0.0, 1.0)
        #      context_embedding = (1.0 - alpha) * original_state + alpha * propagated_state

        batch_size, dim = context_embedding.shape

        # [Batch, SeqLen, Dim]
        # We add some random perturbation to break symmetry for different positions
        # Simulates "superposition" of potentials
        position_states = np.tile(context_embedding[:, np.newaxis, :], (1, seq_len, 1))
        # noise = np.random.normal(0, 1e-4, position_states.shape).astype(np.float32)
        # position_states += noise

        # Flatten for processing: [Batch*SeqLen, Dim]
        flat_states = position_states.reshape(-1, dim)

        # 2. Vocabulary Superposition (Hopfield)
        # Get probabilities for each position based on state similarity
        # [Batch*SeqLen, Vocab]
        probs = self.hopfield.get_token_probs(flat_states)

        # Reshape to [Batch, SeqLen, Vocab]
        probs = probs.reshape(batch_size, seq_len, -1)

        # 3. Position Coherence (Jacobi Refinement)
        # Smooth probabilities based on neighbors
        latent_prior = np.mean(position_states, axis=1, keepdims=True)
        latent_prior = np.broadcast_to(latent_prior, position_states.shape)
        probs = self.jacobi.refine(
            probs,
            branch_priors=np.mean(latent_prior, axis=-1, keepdims=True),
            frontier_width=getattr(self.config, "speculative_drafts", 4),
        )

        # 4. Grover Amplification
        # Sharpen the distribution towards likely tokens
        # We treat the probs as if they were result of quantum measurement
        # Convert back to amplitude space implicitly
        # Actually grover works on amplitudes, but our wrapper handles logits/probs conversion

        # Need to work on flattened [Batch*SeqLen, Vocab] again for simplicity
        # or adapt Grover to handle 3D. Our grover.amplify_logits handles 1D or 2D.
        # Let's verify grover implementation... well amplify_logits takes logits.
        # We have probs.

        # Probs -> Logits approx
        logits = np.log(np.maximum(probs, 1.0e-10))

        # Apply Grover to each token position
        # We can loop or vectorize. For MVP, loop over seq_len
        # (Since Grover.amplify_logits does internal softmax/norm, we pass logits)
        amplified_probs_list = []
        token_labels = [
            str(token)
            for token in (
                (oracle_context or {}).get("tokens")
                or list(range(int(probs.shape[-1])))
            )
        ]
        repo_delta = dict((oracle_context or {}).get("repo_delta") or {})
        invariant_terms = list((oracle_context or {}).get("invariant_terms") or [])
        grover_telemetry: list[dict[str, float]] = []
        for i in range(seq_len):
            pos_logits = logits[:, i, :]  # [Batch, Vocab]
            telemetry_sink: dict[str, float] = {}
            amp_probs = self.grover.amplify_with_resonance(
                pos_logits,
                token_labels,
                context_text=str((oracle_context or {}).get("context_text") or ""),
                iterations=self.config.grover_iterations,
                top_k_oracle=getattr(self.config, "speculative_drafts", 4),
                token_embeddings=self.vocab_embeddings,
                latent_prior=(
                    (oracle_context or {}).get("latent_prior")
                    or (context_embedding[0].tolist() if batch_size > 0 else None)
                ),
                repo_delta=repo_delta,
                invariant_terms=invariant_terms,
                telemetry_sink=telemetry_sink,
            )
            grover_telemetry.append(dict(telemetry_sink))
            amplified_probs_list.append(amp_probs)

        final_probs = np.stack(amplified_probs_list, axis=1)  # [Batch, SeqLen, Vocab]

        # 5. Selection (Measurement)
        # Update phase based on average confidence (max prob at each pos)
        avg_confidence = np.mean(np.max(final_probs, axis=-1))
        self.phase_controller.update_phase(avg_confidence)

        # Get dynamic parameters
        dyn_params = self.phase_controller.get_parameters()
        temp = dyn_params["temperature"]

        # Greedy for now (Top-1)
        # Or sample? Configurable.
        # [Batch, SeqLen]
        if temp < 1e-5:
            tokens = np.argmax(final_probs, axis=-1)
        else:
            # Vectorized Gumbel-max sampling for 100k vocab
            # [Batch, SeqLen, Vocab]
            logits = np.log(np.maximum(final_probs, 1.0e-10)) / temp
            gumbel_noise = np.random.gumbel(size=logits.shape).astype(np.float32)
            tokens = np.argmax(logits + gumbel_noise, axis=-1)

        self.last_generation_trace = {
            "jacobi_frontier": self.jacobi.last_frontier_state.as_dict(),
            "grover_steps": grover_telemetry,
        }
        return tokens, final_probs
