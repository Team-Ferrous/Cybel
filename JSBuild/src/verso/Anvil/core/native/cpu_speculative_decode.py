"""
CPU-Optimized Speculative Decoding for llamacpp + COCONUT + QSG.

Uses COCONUT multi-path reasoning to generate draft candidates (no separate model),
then verifies them in parallel using llama.cpp batch processing.

Key differences from standard speculative decoding:
1. No separate draft model (uses COCONUT paths as candidates)
2. CPU-optimized batch verification
3. Integrated with QSG pipeline
"""

import numpy as np
from typing import Any, Callable, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

try:
    from core.native import simd_ops_wrapper as native_simd_ops
except Exception:
    native_simd_ops = None

try:
    from core.simd.simd_ops import SIMDOps
except Exception:
    SIMDOps = None

_FLOAT32_FLOOR = np.finfo(np.float32).min


def _sanitize_logits(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float32)
    if x.size <= 0:
        return x
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.float32)
    safe_floor = float(np.min(x[finite])) - 1.0e4
    return np.where(finite, x, safe_floor).astype(np.float32, copy=False)


def _safe_argmax(logits: np.ndarray, fallback_token: int = 0) -> int:
    x = np.asarray(logits, dtype=np.float32)
    if x.size <= 0:
        return int(fallback_token)
    fallback = int(np.clip(int(fallback_token), 0, int(x.size) - 1))
    finite = np.isfinite(x)
    if not np.any(finite):
        return fallback
    return int(np.argmax(np.where(finite, x, _FLOAT32_FLOOR)))


def _stable_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float32)
    if x.size <= 0:
        return x

    finite = np.isfinite(x)
    finite_count = int(np.sum(finite))
    if finite_count <= 0:
        return np.full(x.shape, 1.0 / float(x.size), dtype=np.float32)

    finite_min = float(np.min(x[finite]))
    safe_floor = finite_min - 1.0e4
    x = np.where(finite, x, safe_floor)
    x = x / max(float(temperature), 1e-8)
    x = x - float(np.max(x))
    x = np.clip(x, -80.0, 0.0)
    exp_x = np.exp(x).astype(np.float32, copy=False)
    total = float(np.sum(exp_x))
    if total <= 0.0 or not np.isfinite(total):
        result = np.zeros_like(exp_x)
        result[finite] = 1.0 / float(finite_count)
        return result

    probs = exp_x / total
    if not np.all(np.isfinite(probs)) or np.any(probs < 0):
        result = np.zeros_like(probs)
        result[finite] = 1.0 / float(finite_count)
        return result
    return probs


def _sample_token(
    logits: np.ndarray,
    temperature: float,
    fallback_token: int,
) -> int:
    sanitized = _sanitize_logits(logits)
    if sanitized.size <= 0:
        return int(fallback_token)
    if temperature <= 0.0:
        return _safe_argmax(sanitized, fallback_token=fallback_token)
    if native_simd_ops is None:
        raise RuntimeError(
            "Native SIMD sampler is unavailable; Python sampling fallback is disabled."
        )
    try:
        return int(
            native_simd_ops.sample_token(
                logits=sanitized,
                temperature=max(float(temperature), 1e-8),
                eos_token=int(fallback_token),
            )
        )
    except Exception as exc:
        raise RuntimeError(
            "Native SIMD sampler failed; Python sampling fallback is disabled."
        ) from exc


class SpeculativeDecodeError(RuntimeError):
    """Base class for speculative decode failures."""


class CoconutDraftUnavailableError(SpeculativeDecodeError):
    """Raised when COCONUT drafting cannot produce candidates."""


class SpeculativeFallbackDisabledError(SpeculativeDecodeError):
    """Raised when strict mode disallows fallback generation."""


@dataclass
class SpeculativeConfig:
    """Configuration for CPU speculative decoding."""

    num_candidates: int = 4  # Number of candidate tokens to draft
    acceptance_threshold: float = 0.7  # Min probability ratio to accept
    max_draft_length: int = 4  # Max tokens to draft ahead
    use_coconut_drafts: bool = True  # Use COCONUT paths for drafting
    fallback_to_top_k: bool = False  # Legacy compatibility path; off by default.
    fallback_to_standard_generation: bool = False  # Legacy compatibility path; off by default.
    strict_native_only: bool = False  # Disable fallback ladder and fail-fast when enabled.


class CPUSpeculativeDecoder:
    """
    CPU-optimized speculative decoder using COCONUT for draft generation.

    Architecture:
        1. Draft Phase: Generate candidates using COCONUT multi-path reasoning
        2. Verification Phase: Parallel verification with llama.cpp
        3. Acceptance: Accept longest valid prefix based on probability ratios
    """

    def __init__(
        self,
        native_engine,
        coconut_bridge=None,
        config: Optional[SpeculativeConfig] = None,
    ):
        """
        Initialize CPU speculative decoder.

        Args:
            native_engine: NativeInferenceEngine instance
            coconut_bridge: CoconutNativeBridge for draft generation
            config: Speculative decoding configuration
        """
        self.engine = native_engine
        self.coconut_bridge = coconut_bridge
        self.config = config or SpeculativeConfig()
        if self.config.strict_native_only:
            self.config.fallback_to_top_k = False
            self.config.fallback_to_standard_generation = False

        # Statistics
        self.total_drafted = 0
        self.total_accepted = 0
        self.acceptance_rate = 0.0
        self._simd_ops = None
        if SIMDOps is not None:
            try:
                simd = SIMDOps()
                if getattr(simd, "available", False):
                    self._simd_ops = simd
            except Exception:
                self._simd_ops = None

    def _draft_candidates_coconut(
        self,
        context_embedding: np.ndarray,
        vocab_embeddings: np.ndarray,
        lm_head_weight: np.ndarray,
        evolution_weights: dict,
        num_candidates: int = 4,
    ) -> List[int]:
        """
        Draft a speculative token sequence using COCONUT multi-path reasoning.

        Instead of a separate draft model, we use COCONUT's multiple
        reasoning paths to generate a short continuation that can be
        verified by the native target model.

        Args:
            context_embedding: Last token embedding [1, dim]
            vocab_embeddings: Full vocabulary embeddings [vocab_size, dim]
            lm_head_weight: LM head projection [vocab_size, dim]
            evolution_weights: COCONUT evolution weights
            num_candidates: Number of candidates to generate

        Returns:
            List of draft token IDs
        """
        candidates, _ = self._draft_candidates_coconut_with_scores(
            context_embedding=context_embedding,
            vocab_embeddings=vocab_embeddings,
            lm_head_weight=lm_head_weight,
            evolution_weights=evolution_weights,
            num_candidates=num_candidates,
        )
        return candidates

    def _draft_candidates_coconut_with_scores(
        self,
        context_embedding: np.ndarray,
        vocab_embeddings: np.ndarray,
        lm_head_weight: np.ndarray,
        evolution_weights: dict,
        num_candidates: int = 4,
    ) -> Tuple[List[int], List[float]]:
        """Draft a short speculative continuation with approximate probabilities."""
        if self.coconut_bridge is None:
            if self.config.strict_native_only:
                raise CoconutDraftUnavailableError(
                    "Strict speculative mode requires an active COCONUT bridge."
                )
            return [], []

        try:
            draft_tokens: List[int] = []
            draft_probs: List[float] = []
            current_context = np.asarray(context_embedding, dtype=np.float32).reshape(
                1, -1
            )
            num_paths = int(max(1, num_candidates))
            max_steps = int(max(1, self.config.max_draft_length))
            w = evolution_weights

            for _ in range(max_steps):
                paths = self.coconut_bridge.expand_paths(
                    current_context, num_paths=num_paths, noise_scale=0.02
                )
                self.coconut_bridge.evolve_paths(
                    paths,
                    w["norm_gamma"],
                    w["norm_beta"],
                    w["w1"],
                    w["b1"],
                    w["w2"],
                    w["b2"],
                    w["hidden_dim"],
                )

                amplitudes = self.coconut_bridge.score_paths(paths, current_context)
                batch_paths = np.asarray(paths[0], dtype=np.float32)
                max_paths = int(min(num_paths, batch_paths.shape[0]))
                if max_paths <= 0:
                    break

                weights = np.asarray(amplitudes[0][:max_paths], dtype=np.float32)
                weight_sum = float(np.sum(weights))
                if weight_sum <= 1e-12:
                    weights = np.full((max_paths,), 1.0 / float(max_paths), dtype=np.float32)
                else:
                    weights = weights / weight_sum

                aggregated_state = np.sum(
                    batch_paths[:max_paths] * weights[:, None], axis=0, keepdims=True
                )
                logits = (aggregated_state @ lm_head_weight.T)[0]
                probs = self._softmax(np.asarray(logits, dtype=np.float32), temperature=1.0)
                token_id = _safe_argmax(
                    logits, fallback_token=int(self.engine.llm.token_eos())
                )
                token_prob = float(probs[token_id])

                draft_tokens.append(token_id)
                draft_probs.append(token_prob)

                if token_id == self.engine.llm.token_eos():
                    break

                current_context = self._logits_to_proxy_state(
                    logits=np.asarray(logits, dtype=np.float32),
                    vocab_embeddings=vocab_embeddings,
                    top_k=64,
                )

            return draft_tokens, draft_probs
        except (KeyError, IndexError, TypeError, ValueError, RuntimeError) as exc:
            if self.config.strict_native_only:
                raise CoconutDraftUnavailableError(
                    "Strict speculative mode failed while generating COCONUT drafts."
                ) from exc
            logger.warning(
                "COCONUT draft generation failed; falling back to compatibility path: %s",
                exc,
            )
            return [], []

    def _draft_candidates_top_k(
        self,
        logits: np.ndarray,
        k: int = 4,
    ) -> List[int]:
        """
        Fallback: Draft candidates from top-k logits.

        Args:
            logits: Logits from last token [vocab_size]
            k: Number of top candidates

        Returns:
            List of top-k token IDs
        """
        candidates, _ = self._draft_candidates_top_k_with_scores(logits=logits, k=k)
        return candidates

    def _draft_candidates_top_k_with_scores(
        self,
        logits: np.ndarray,
        k: int = 4,
    ) -> Tuple[List[int], List[float]]:
        """Draft top-k candidates with normalized candidate scores."""
        logits_f = np.asarray(logits, dtype=np.float32)
        logits_f = np.where(np.isfinite(logits_f), logits_f, _FLOAT32_FLOOR)
        safe_k = int(max(1, min(k, logits_f.shape[-1])))
        top_k_indices = np.argpartition(logits_f, -safe_k)[-safe_k:]
        top_k_indices = top_k_indices[np.argsort(-logits_f[top_k_indices])]
        candidate_scores = logits_f[top_k_indices]
        candidate_probs = self._softmax(
            np.asarray(candidate_scores, dtype=np.float32), temperature=1.0
        )
        return top_k_indices.tolist(), candidate_probs.tolist()

    def _draft_sequence_top_k_with_scores(
        self,
        logits: np.ndarray,
    ) -> Tuple[List[int], List[float]]:
        """Compatibility fallback: produce a one-token draft from target logits."""
        candidates, probs = self._draft_candidates_top_k_with_scores(logits=logits, k=1)
        return candidates[:1], probs[:1]

    def _verify_candidates_batch(
        self,
        prompt_tokens: List[int],
        candidate_sequences: List[List[int]],
        temperature: float = 0.8,
        draft_probs: Optional[List[float]] = None,
    ) -> Tuple[int, List[float]]:
        """
        Verify candidate sequences in parallel using llama.cpp.

        Args:
            prompt_tokens: Current context tokens
            candidate_sequences: List of candidate token sequences to verify
            temperature: Sampling temperature
            draft_probs: Draft model probabilities for each candidate step

        Returns:
            Tuple of (accepted_length, acceptance_ratios)
        """
        if len(candidate_sequences) == 0:
            return 0, []

        # For CPU optimization, we verify sequentially but efficiently
        # (llama.cpp doesn't expose batch API easily via Python bindings)
        accepted_length = 0
        acceptance_ratios = []

        for i, candidate_seq in enumerate(candidate_sequences):
            if len(candidate_seq) == 0:
                break

            # Verify token i in the draft sequence against the model distribution
            # at position i (conditioned on the prefix before that token).
            candidate_token = candidate_seq[-1]
            prefix_tokens = prompt_tokens + candidate_seq[:-1]

            try:
                # Get logits for the next-token distribution of the prefix.
                logits = self._get_logits_for_tokens(prefix_tokens)

                if logits is None:
                    break

                # Compute probability of the proposed token at this timestep.
                probs = self._softmax(logits, temperature)
                candidate_prob = float(probs[candidate_token])
                draft_prob = (
                    float(draft_probs[i])
                    if draft_probs is not None and i < len(draft_probs)
                    else 1.0
                )
                draft_prob = max(draft_prob, 1e-8)
                accept_ratio = min(1.0, candidate_prob / draft_prob)
                acceptance_ratios.append(float(accept_ratio))

                # Check acceptance threshold using draft-vs-target ratio.
                if accept_ratio < self.config.acceptance_threshold:
                    break

                accepted_length = i + 1

            except (AttributeError, IndexError, TypeError, ValueError, RuntimeError):
                break

        return accepted_length, acceptance_ratios

    def _get_logits_for_tokens(self, tokens: List[int]) -> Optional[np.ndarray]:
        """
        Get logits from llama.cpp for a token sequence.
        Uses KV cache for efficiency.
        """
        try:
            # Prepare for generation (prefix matching)
            start_pos = 0
            if self.engine.kv_cache_manager:
                start_pos = self.engine.kv_cache_manager.prepare_for_generation(
                    tokens, allow_reuse=True
                )

            # If we have a cache hit, only evaluate the new part
            if start_pos > 0:
                self.engine.llm.n_past = start_pos
                tokens_to_eval = tokens[start_pos:]
                self.engine.llm.eval(tokens_to_eval)
            else:
                self.engine.llm.reset()
                self.engine.llm.eval(tokens)

            logits = self.engine.llm.scores[self.engine.llm.n_tokens - 1]
            return np.array(logits, dtype=np.float32)
        except (AttributeError, IndexError, TypeError, ValueError, RuntimeError):
            return None

    def _softmax(self, logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Compute softmax with temperature, NaN-safe."""
        return _stable_softmax(logits, temperature=temperature)

    def _logits_to_proxy_state(
        self,
        logits: np.ndarray,
        vocab_embeddings: np.ndarray,
        top_k: int = 64,
    ) -> np.ndarray:
        """
        Build a context proxy from logits via weighted vocab embedding mixture.

        This avoids using the raw last-token embedding as a hidden-state proxy.
        """
        if logits is None or vocab_embeddings is None or len(vocab_embeddings) == 0:
            return np.zeros((1, 1), dtype=np.float32)

        logits_f = np.asarray(logits, dtype=np.float32)
        logits_f = np.where(np.isfinite(logits_f), logits_f, _FLOAT32_FLOOR)
        safe_k = int(max(1, min(top_k, logits_f.shape[-1], vocab_embeddings.shape[0])))
        top_idx = np.argpartition(logits_f, -safe_k)[-safe_k:]
        top_logits = logits_f[top_idx]
        top_logits = top_logits - np.max(top_logits)
        weights = np.exp(top_logits)
        weights_sum = float(np.sum(weights))
        if weights_sum <= 1e-12:
            weights = np.ones_like(weights, dtype=np.float32) / float(len(weights))
        else:
            weights = weights / weights_sum
        proxy = (weights[:, None] * vocab_embeddings[top_idx]).sum(
            axis=0, keepdims=True
        )
        return proxy.astype(np.float32, copy=False)

    def generate_speculative(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int,
        temperature: float = 0.8,
        coconut_resources: Optional[dict] = None,
        logits_processor: Optional[Callable] = None,
    ):
        """
        Generate tokens using CPU-optimized speculative decoding.

        Args:
            prompt_tokens: Initial prompt tokens
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            coconut_resources: Dict with vocab_embeddings, lm_head_weight, evolution_weights
            logits_processor: Optional logits processor chain

        Yields:
            Generated token IDs
        """
        current_tokens = list(prompt_tokens)
        generated_count = 0

        while generated_count < max_new_tokens:
            # 1. DRAFT PHASE
            # Generate candidates using COCONUT or top-k
            candidates = []
            draft_probs: List[float] = []

            if (
                self.config.use_coconut_drafts
                and coconut_resources is not None
                and self.coconut_bridge is not None
            ):
                try:
                    vocab_embeddings = coconut_resources["vocab_embeddings"]
                    logits = self._get_logits_for_tokens(current_tokens)
                    context_emb = self._logits_to_proxy_state(
                        logits=logits,
                        vocab_embeddings=vocab_embeddings,
                        top_k=64,
                    )
                    if context_emb.size <= 1:
                        last_token = current_tokens[-1]
                        context_emb = vocab_embeddings[last_token : last_token + 1]

                    candidates, draft_probs = self._draft_candidates_coconut_with_scores(
                        context_emb,
                        vocab_embeddings,
                        coconut_resources["lm_head_weight"],
                        coconut_resources["evolution_weights"],
                        num_candidates=self.config.num_candidates,
                    )
                except (
                    CoconutDraftUnavailableError,
                    KeyError,
                    IndexError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                ) as exc:
                    if self.config.strict_native_only:
                        raise CoconutDraftUnavailableError(
                            "Strict speculative mode failed to draft COCONUT candidates."
                        ) from exc
                    logger.warning(
                        "COCONUT drafting failed; continuing with fallback candidates: %s",
                        exc,
                    )
                    candidates = []
                    draft_probs = []

            # Fallback to top-k if COCONUT drafting failed
            if len(candidates) == 0 and self.config.strict_native_only:
                raise SpeculativeFallbackDisabledError(
                    "Strict speculative mode disallows top-k fallback when no COCONUT drafts are available."
                )

            if len(candidates) == 0 and self.config.fallback_to_top_k:
                try:
                    logits = self._get_logits_for_tokens(current_tokens)
                    if logits is not None:
                        candidates, draft_probs = self._draft_sequence_top_k_with_scores(
                            logits
                        )
                except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
                    logger.warning("Top-k fallback candidate drafting failed: %s", exc)
                    candidates = []
                    draft_probs = []

            # If no candidates, fallback to standard generation
            if len(candidates) == 0:
                if (
                    self.config.strict_native_only
                    or not self.config.fallback_to_standard_generation
                ):
                    raise SpeculativeFallbackDisabledError(
                        "No speculative candidates available and standard-generation fallback is disabled."
                    )
                # Standard single-token generation
                for token in self.engine.generate_stream(
                    current_tokens,
                    max_new_tokens=1,
                    temperature=temperature,
                    logits_processor=logits_processor,
                ):
                    current_tokens.append(token)
                    generated_count += 1
                    yield token
                    if token == self.engine.llm.token_eos():
                        return
                continue

            # 2. VERIFICATION PHASE
            # Verify prefixes of a single speculative continuation.
            draft_sequences = []
            for draft_len in range(
                1, min(len(candidates) + 1, self.config.max_draft_length + 1)
            ):
                draft_sequences.append(candidates[:draft_len])

            # Verify candidates
            accepted_length, probs = self._verify_candidates_batch(
                current_tokens,
                draft_sequences,
                temperature,
                draft_probs=draft_probs,
            )

            # 3. ACCEPTANCE PHASE
            if accepted_length > 0:
                # Accept the verified prefix
                accepted_tokens = candidates[:accepted_length]
                for token in accepted_tokens:
                    current_tokens.append(token)
                    generated_count += 1
                    yield token

                    if token == self.engine.llm.token_eos():
                        return

                    if generated_count >= max_new_tokens:
                        return

                # Update statistics
                self.total_drafted += len(candidates)
                self.total_accepted += accepted_length
                self.acceptance_rate = self.total_accepted / max(self.total_drafted, 1)

            else:
                # Rejection: Generate one token normally
                if (
                    self.config.strict_native_only
                    or not self.config.fallback_to_standard_generation
                ):
                    raise SpeculativeFallbackDisabledError(
                        "Speculative verification rejected all candidates and standard-generation fallback is disabled."
                    )
                for token in self.engine.generate_stream(
                    current_tokens,
                    max_new_tokens=1,
                    temperature=temperature,
                    logits_processor=logits_processor,
                ):
                    current_tokens.append(token)
                    generated_count += 1
                    yield token

                    if token == self.engine.llm.token_eos():
                        return

    def get_stats(self) -> dict:
        """Get speculative decoding statistics."""
        return {
            "total_drafted": self.total_drafted,
            "total_accepted": self.total_accepted,
            "acceptance_rate": self.acceptance_rate,
            "average_speedup": 1.0 + self.acceptance_rate,  # Approximate
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self.total_drafted = 0
        self.total_accepted = 0
        self.acceptance_rate = 0.0


class SSMSelfSpeculativeDecoder:
    """Self-speculative decoder that drafts and verifies using the same native engine logits."""

    def __init__(
        self,
        native_engine: Any,
        max_draft_length: int = 4,
        acceptance_threshold: float = 0.7,
    ):
        self.engine = native_engine
        self.max_draft_length = int(max(1, max_draft_length))
        self.acceptance_threshold = float(np.clip(acceptance_threshold, 0.0, 1.0))

    def _softmax(self, logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        return _stable_softmax(logits, temperature=temperature)

    def _apply_logits_processor(
        self,
        tokens: List[int],
        logits: np.ndarray,
        logits_processor: Optional[Callable],
    ) -> np.ndarray:
        out = _sanitize_logits(logits)
        if logits_processor is None:
            return out
        if isinstance(logits_processor, list):
            for processor in logits_processor:
                next_out = processor(tokens, out)
                if next_out is None:
                    continue
                out = _sanitize_logits(next_out)
            return out
        next_out = logits_processor(tokens, out)
        if next_out is None:
            return out
        return _sanitize_logits(next_out)

    def _engine_logits(self, tokens: List[int]) -> np.ndarray:
        if hasattr(self.engine, "_get_logits_for_tokens"):
            logits = self.engine._get_logits_for_tokens(tokens)
            return np.asarray(logits, dtype=np.float32)
        if hasattr(self.engine, "_get_logits"):
            logits = self.engine._get_logits(tokens, start_pos=0)
            return np.asarray(logits, dtype=np.float32)
        raise AttributeError("Native engine does not expose _get_logits_for_tokens/_get_logits")

    def draft(
        self,
        prompt_tokens: List[int],
        max_draft_tokens: Optional[int] = None,
        temperature: float = 0.8,
        logits_processor: Optional[Callable] = None,
    ) -> Tuple[List[int], List[float]]:
        """Draft a short continuation from current model logits."""
        tokens = [int(t) for t in prompt_tokens]
        draft_len = int(max_draft_tokens) if max_draft_tokens is not None else self.max_draft_length
        draft_len = int(max(1, min(draft_len, self.max_draft_length)))
        drafted: List[int] = []
        draft_probs: List[float] = []

        for _ in range(draft_len):
            logits = self._engine_logits(tokens)
            logits = self._apply_logits_processor(tokens, logits, logits_processor)
            probs = self._softmax(logits, temperature=max(temperature, 1e-8))
            token = _sample_token(
                logits=logits,
                temperature=float(temperature),
                fallback_token=int(self.engine.token_eos()),
            )
            drafted.append(token)
            draft_probs.append(float(probs[token]))
            tokens.append(token)
            if token == int(self.engine.token_eos()):
                break
        return drafted, draft_probs

    def verify(
        self,
        prompt_tokens: List[int],
        draft_tokens: List[int],
        draft_probs: Optional[List[float]] = None,
        temperature: float = 0.8,
        logits_processor: Optional[Callable] = None,
    ) -> Tuple[int, List[float]]:
        """Verify drafted tokens against target logits and return accepted prefix length."""
        accepted = 0
        ratios: List[float] = []
        context = [int(t) for t in prompt_tokens]

        for i, token in enumerate(draft_tokens):
            logits = self._engine_logits(context)
            logits = self._apply_logits_processor(context, logits, logits_processor)
            probs = self._softmax(logits, temperature=max(temperature, 1e-8))
            target_prob = float(probs[int(token)])
            proposal_prob = (
                float(draft_probs[i])
                if draft_probs is not None and i < len(draft_probs)
                else 1.0
            )
            ratio = min(1.0, target_prob / max(proposal_prob, 1e-8))
            ratios.append(ratio)
            if ratio < self.acceptance_threshold:
                break
            accepted += 1
            context.append(int(token))
        return accepted, ratios

    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int,
        temperature: float = 0.8,
        logits_processor: Optional[Callable] = None,
    ):
        """Generate tokens using draft/verify loops over native logits."""
        tokens = [int(t) for t in prompt_tokens]
        produced = 0
        while produced < int(max_new_tokens):
            drafted, probs = self.draft(
                prompt_tokens=tokens,
                max_draft_tokens=min(self.max_draft_length, int(max_new_tokens) - produced),
                temperature=temperature,
                logits_processor=logits_processor,
            )
            if not drafted:
                break
            accepted, _ = self.verify(
                prompt_tokens=tokens,
                draft_tokens=drafted,
                draft_probs=probs,
                temperature=temperature,
                logits_processor=logits_processor,
            )
            if accepted <= 0:
                logits = self._engine_logits(tokens)
                logits = self._apply_logits_processor(tokens, logits, logits_processor)
                next_token = _sample_token(
                    logits=logits,
                    temperature=float(temperature),
                    fallback_token=int(self.engine.token_eos()),
                )
                tokens.append(next_token)
                produced += 1
                yield next_token
                if next_token == int(self.engine.token_eos()):
                    return
                continue

            for token in drafted[:accepted]:
                tokens.append(int(token))
                produced += 1
                yield int(token)
                if int(token) == int(self.engine.token_eos()):
                    return
                if produced >= int(max_new_tokens):
                    return
