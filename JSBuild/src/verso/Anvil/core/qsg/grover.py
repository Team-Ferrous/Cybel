from dataclasses import dataclass
import os
import re
from typing import Any, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from core.simd.simd_ops import SIMDOps
from core.memory.scratch_pool import ScratchPool
from core.analysis.semantic import SemanticEngine

_TOKEN_TERM_RE = re.compile(r"[a-z0-9_]+")


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    norm = float(np.linalg.norm(arr))
    if norm <= 1.0e-8:
        return np.zeros_like(arr)
    return arr / norm


def _extract_terms(values: Iterable[str]) -> set[str]:
    terms: set[str] = set()
    for value in values:
        text = str(value or "").strip().lower()
        if not text:
            continue
        terms.update(_TOKEN_TERM_RE.findall(text))
    return terms


@dataclass(slots=True)
class ResonanceOracleField:
    resonance_scores: np.ndarray
    oracle_mask: np.ndarray
    latent_bias: np.ndarray
    diagnostics: dict[str, float]


class GroverAmplifier:
    """
    Quantum-inspired Grover Amplitude Amplification for token selection.

    Amplifies the amplitude of states (tokens) that satisfy an oracle condition.
    Enhanced version uses 'Semantic Resonance' with the Saguaro index.
    """

    def __init__(
        self,
        semantic_engine: Optional[SemanticEngine] = None,
        resonance_timeout_ms: int = 0,
    ):
        self.simd = SIMDOps()
        self.scratch = ScratchPool()
        self.semantic_engine = semantic_engine
        self.resonance_timeout_ms = int(max(0, resonance_timeout_ms))
        self.last_oracle_telemetry: dict[str, float] = {}

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        max_vals = np.max(values, axis=-1, keepdims=True)
        exp_vals = np.exp(np.clip(values - max_vals, -80.0, 80.0))
        denom = np.sum(exp_vals, axis=-1, keepdims=True)
        denom = np.where(denom <= 1.0e-12, 1.0, denom)
        return exp_vals / denom

    @staticmethod
    def _coerce_vector(
        payload: Optional[Sequence[float]],
        *,
        target_dim: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        if payload is None:
            return None
        try:
            arr = np.asarray(list(payload), dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if arr.size <= 0 or not np.all(np.isfinite(arr)):
            return None
        if target_dim is not None and target_dim > 0:
            if arr.size > target_dim:
                arr = arr[:target_dim]
            elif arr.size < target_dim:
                arr = np.pad(arr, (0, target_dim - arr.size))
        return arr

    def _get_semantic_context(
        self,
        context_text: str,
        *,
        timeout_ms: Optional[int] = None,
    ) -> list[str]:
        if not self.semantic_engine:
            return []
        timeout_budget_ms = (
            self.resonance_timeout_ms if timeout_ms is None else int(max(0, timeout_ms))
        )
        if timeout_budget_ms > 0:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.semantic_engine.get_context_for_objective, context_text
                )
                try:
                    result = future.result(timeout=timeout_budget_ms / 1000.0)
                except TimeoutError:
                    return []
        else:
            result = self.semantic_engine.get_context_for_objective(context_text)
        return [str(item) for item in list(result or []) if str(item).strip()]

    def build_oracle_field(
        self,
        *,
        logits: np.ndarray,
        tokens: Sequence[str],
        context_text: str,
        token_embeddings: Optional[np.ndarray] = None,
        latent_prior: Optional[Sequence[float]] = None,
        repo_delta: Optional[Mapping[str, Any]] = None,
        invariant_terms: Optional[Sequence[str]] = None,
        semantic_context: Optional[Sequence[str]] = None,
        top_k_oracle: Optional[int] = None,
        latent_bias_alpha: float = 0.35,
        timeout_ms: Optional[int] = None,
    ) -> ResonanceOracleField:
        logits_arr = np.asarray(logits, dtype=np.float32)
        if logits_arr.ndim == 1:
            logits_arr = logits_arr.reshape(1, -1)
        vocab_size = int(logits_arr.shape[-1])
        oracle_k = int(top_k_oracle or 0)
        if oracle_k <= 0:
            oracle_k = min(8, vocab_size)
        oracle_k = max(1, min(oracle_k, vocab_size))
        top_indices = np.argpartition(logits_arr[0], -oracle_k)[-oracle_k:]

        scores = np.zeros((vocab_size,), dtype=np.float32)
        token_terms = [_extract_terms([token]) for token in list(tokens)]

        repo_payload = dict(repo_delta or {})
        repo_paths = [
            str(path)
            for path in list(repo_payload.get("changed_paths") or [])
            if str(path).strip()
        ]
        repo_terms = _extract_terms(
            repo_paths
            + [os.path.basename(path) for path in repo_paths]
            + [str(repo_payload.get("summary_text") or "")]
        )
        invariant_set = _extract_terms(list(invariant_terms or []))
        semantic_values = [
            str(item) for item in list(semantic_context or []) if str(item).strip()
        ] or self._get_semantic_context(context_text, timeout_ms=timeout_ms)
        semantic_terms = _extract_terms(semantic_values)
        context_terms = _extract_terms([context_text])

        latent_signal = np.zeros((vocab_size,), dtype=np.float32)
        embeddings = None
        if token_embeddings is not None:
            embeddings = np.asarray(token_embeddings, dtype=np.float32)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            if embeddings.shape[0] < vocab_size:
                embeddings = None
        prior = None
        if embeddings is not None:
            prior = self._coerce_vector(
                latent_prior, target_dim=int(embeddings.shape[1])
            )
            if prior is not None:
                prior = _normalize_vector(prior)
                indexed = embeddings[top_indices]
                indexed_norm = np.linalg.norm(indexed, axis=1, keepdims=True)
                indexed_norm = np.where(indexed_norm <= 1.0e-8, 1.0, indexed_norm)
                indexed = indexed / indexed_norm
                similarities = np.maximum(indexed @ prior, 0.0)
                latent_signal[top_indices] = np.asarray(similarities, dtype=np.float32)

        source_count = (
            int(bool(semantic_terms)) + int(bool(repo_terms)) + int(prior is not None)
        )
        source_count += int(bool(invariant_set or context_terms))

        for idx in top_indices.tolist():
            lexical_terms = token_terms[idx]
            if not lexical_terms and idx < len(tokens):
                token_value = str(tokens[idx] or "").strip().lower()
                if token_value:
                    lexical_terms = {token_value}
            if not lexical_terms:
                continue
            repo_overlap = (
                float(len(lexical_terms & repo_terms)) / float(len(lexical_terms))
                if repo_terms
                else 0.0
            )
            semantic_overlap = (
                float(len(lexical_terms & semantic_terms)) / float(len(lexical_terms))
                if semantic_terms
                else 0.0
            )
            invariant_overlap = (
                float(len(lexical_terms & (invariant_set | context_terms)))
                / float(len(lexical_terms))
                if (invariant_set or context_terms)
                else 0.0
            )
            scores[idx] = (
                0.58 * float(latent_signal[idx])
                + 0.22 * float(repo_overlap)
                + 0.12 * float(semantic_overlap)
                + 0.08 * float(invariant_overlap)
            )

        if float(np.max(scores)) <= 1.0e-8:
            fallback_idx = int(np.argmax(logits_arr[0]))
            scores[fallback_idx] = 1.0

        normalized_scores = self._softmax(scores.reshape(1, -1))[0]
        ranked = np.argpartition(normalized_scores, -oracle_k)[-oracle_k:]
        oracle_mask = np.zeros((1, vocab_size), dtype=bool)
        oracle_mask[0, ranked] = True
        topk_overlap = float(
            len(set(top_indices.tolist()) & set(ranked.tolist()))
        ) / float(max(1, oracle_k))
        diagnostics = {
            "grover_oracle_source_count": float(source_count),
            "grover_resonance_mean": float(np.mean(normalized_scores[ranked])),
            "grover_resonance_topk_overlap": float(topk_overlap),
            "grover_latent_bias_alpha": float(max(0.0, min(1.0, latent_bias_alpha))),
        }
        return ResonanceOracleField(
            resonance_scores=normalized_scores,
            oracle_mask=oracle_mask,
            latent_bias=normalized_scores.reshape(1, -1),
            diagnostics=diagnostics,
        )

    def amplify(
        self, amplitudes: np.ndarray, oracle_mask: np.ndarray, iterations: int = 1
    ) -> np.ndarray:
        """Standard Grover Amplitude Amplification."""
        state = amplitudes.astype(np.float32, copy=True)

        for _ in range(iterations):
            # 1. Oracle: Flip phase
            np.negative(state, out=state, where=oracle_mask)

            # 2. Diffusion: Invert about mean
            mean_amp = np.mean(state)
            two_mean = 2.0 * mean_amp
            state = two_mean - state

        return state

    def amplify_with_resonance(
        self,
        logits: np.ndarray,
        tokens: List[str],
        context_text: str,
        iterations: int = 1,
        model_profile: Optional[Any] = None,
        top_k_oracle: Optional[int] = None,
        damping: Optional[float] = None,
        timeout_ms: Optional[int] = None,
        token_embeddings: Optional[np.ndarray] = None,
        latent_prior: Optional[Sequence[float]] = None,
        repo_delta: Optional[Mapping[str, Any]] = None,
        invariant_terms: Optional[Sequence[str]] = None,
        semantic_context: Optional[Sequence[str]] = None,
        telemetry_sink: Optional[dict[str, float]] = None,
        latent_bias_alpha: float = 0.35,
    ) -> np.ndarray:
        """
        Grover Amplification using Semantic Resonance as the oracle.
        1. Compute resonance(token, codebase_context)
        2. Mark tokens with resonance > threshold.
        3. Amplify.
        """
        has_oracle_sources = any(
            (
                self.semantic_engine is not None,
                token_embeddings is not None and latent_prior is not None,
                repo_delta,
                invariant_terms,
                semantic_context,
            )
        )
        if not has_oracle_sources:
            return self.amplify_logits(
                logits,
                iterations=iterations,
                model_profile=model_profile,
                top_k_oracle=top_k_oracle,
                damping=damping,
            )

        # 1. Get Probs & Amplitudes
        max_logit = np.max(logits, axis=-1, keepdims=True)
        probs = np.exp(logits - max_logit)
        probs /= np.sum(probs, axis=-1, keepdims=True)
        initial_amplitudes = np.sqrt(probs)
        state = initial_amplitudes.copy()

        oracle = self.build_oracle_field(
            logits=logits,
            tokens=tokens,
            context_text=context_text,
            token_embeddings=token_embeddings,
            latent_prior=latent_prior,
            repo_delta=repo_delta,
            invariant_terms=invariant_terms,
            semantic_context=semantic_context,
            top_k_oracle=top_k_oracle,
            latent_bias_alpha=latent_bias_alpha,
            timeout_ms=timeout_ms,
        )
        self.last_oracle_telemetry = dict(oracle.diagnostics)
        if telemetry_sink is not None:
            telemetry_sink.update(self.last_oracle_telemetry)

        # 3. Grover Iterations
        for _ in range(iterations):
            np.negative(state, out=state, where=oracle.oracle_mask)
            inner_product = np.sum(initial_amplitudes * state, axis=-1, keepdims=True)
            state = 2.0 * inner_product * initial_amplitudes - state

        # 4. Back to Probs
        final_probs = state**2
        sum_probs = np.sum(final_probs, axis=-1, keepdims=True)
        normalized_probs = final_probs / sum_probs
        alpha = float(max(0.0, min(1.0, latent_bias_alpha)))
        if alpha > 0.0:
            normalized_probs = ((1.0 - alpha) * normalized_probs) + (
                alpha * oracle.latent_bias
            )
            renorm = np.sum(normalized_probs, axis=-1, keepdims=True)
            renorm = np.where(renorm <= 1.0e-12, 1.0, renorm)
            normalized_probs = normalized_probs / renorm
        return normalized_probs

    def amplify_logits(
        self,
        logits: np.ndarray,
        iterations: int = 1,
        model_profile: Optional[Any] = None,
        top_k_oracle: Optional[int] = None,
        damping: Optional[float] = None,
    ) -> np.ndarray:
        """Profile-aware Grover amplification with adaptive top-k oracle."""
        # Handle 1D input (single sample)
        input_was_1d = logits.ndim == 1
        if input_was_1d:
            logits = logits[np.newaxis, :]  # Add batch dimension

        effective_iterations = max(1, int(iterations))
        if model_profile is not None and getattr(model_profile, "gqa", False):
            effective_iterations = min(effective_iterations, 2)

        max_logit = np.max(logits, axis=-1, keepdims=True)
        probs = np.exp(logits - max_logit)
        probs /= np.sum(probs, axis=-1, keepdims=True)
        initial_amplitudes = np.sqrt(probs)
        state = initial_amplitudes.copy()

        mask = np.zeros_like(initial_amplitudes, dtype=bool)
        oracle_k = int(top_k_oracle or 0)
        if oracle_k <= 0 and model_profile is not None:
            oracle_k = int(getattr(model_profile, "grover_top_k", 1))
        if oracle_k <= 0:
            oracle_k = 1
        oracle_k = max(1, min(oracle_k, initial_amplitudes.shape[-1]))

        damping_factor = float(damping or 0.0)
        if damping_factor <= 0.0 and model_profile is not None:
            damping_factor = float(getattr(model_profile, "grover_damping", 1.0))
        if damping_factor <= 0.0:
            damping_factor = 1.0

        if oracle_k == 1:
            max_indices = np.argmax(initial_amplitudes, axis=-1)
            batch_indices = np.arange(initial_amplitudes.shape[0])
            mask[batch_indices, max_indices] = True
        else:
            for batch_idx in range(initial_amplitudes.shape[0]):
                top_indices = np.argpartition(initial_amplitudes[batch_idx], -oracle_k)[
                    -oracle_k:
                ]
                mask[batch_idx, top_indices] = True

        for _ in range(effective_iterations):
            np.negative(state, out=state, where=mask)
            inner_product = np.sum(initial_amplitudes * state, axis=-1, keepdims=True)
            state = 2.0 * inner_product * initial_amplitudes - state

        # Damping prevents over-sharpening on GQA/tied-embedding models.
        if damping_factor < 0.999:
            state = (damping_factor * state) + (
                (1.0 - damping_factor) * initial_amplitudes
            )

        final_probs = state**2
        sum_probs = np.sum(final_probs, axis=-1, keepdims=True)
        result = final_probs / sum_probs

        # Return same dimensionality as input
        if input_was_1d:
            result = result[0]

        return result
