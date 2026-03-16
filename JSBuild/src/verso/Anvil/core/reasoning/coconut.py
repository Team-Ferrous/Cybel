"""
CoCoNut: Continuous Chain of Thought with CPU-native backend.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from .backends import get_backend
from .adaptive_exploration import (
    AdaptiveExplorationMetrics,
    amplitude_confidence,
    amplitude_entropy,
)

logger = logging.getLogger(__name__)


class ContinuousThoughtBlock:
    """
    CoCoNut: Continuous Chain of Thought.
    Explores latent reasoning paths before projecting to vocabulary.

    This class acts as a high-level wrapper that delegates to the native backend.
    """

    def __init__(
        self,
        embedding_dim: int = 4096,
        num_paths: int = 12,
        steps: int = 6,
        use_gpu: bool = False,
        backend: str = "auto",
        gpu_device: str = "/CPU:0",
        use_fft: bool = True,
        persistent_freq_state: bool = True,
        deterministic: bool = True,
        **extra_config,
    ):
        """
        Initialize COCONUT reasoning block.

        Args:
            embedding_dim: Dimension of thought embeddings
            num_paths: Number of parallel reasoning paths to explore
            steps: Depth of search (thought iterations)
            use_gpu: Ignored in CPU-only mode
            backend: Specific backend to use ("auto" or "native")
            gpu_device: Device hint string
            use_fft: Enable FFT-based evolution (if supported by backend)
            persistent_freq_state: Enable frequency domain persistence
            deterministic: Enable deterministic operations
        """
        self.config = {
            "embedding_dim": embedding_dim,
            "num_paths": num_paths,
            "steps": steps,
            "gpu_device": gpu_device,
            "use_fft": use_fft,
            "persistent_freq_state": persistent_freq_state,
            "deterministic": deterministic,
            **extra_config,
        }

        # Select backend
        effective_backend = backend if use_gpu else "native"
        self.backend = get_backend(effective_backend, **self.config)

        logger.info(
            f"Initialized COCONUT with {self.backend.get_device_info()['backend']} backend"
        )
        self.last_adaptive_metrics: Optional[AdaptiveExplorationMetrics] = None
        self.last_session_record: Optional[Dict[str, Any]] = None

    def explore(self, context_embedding: np.ndarray) -> np.ndarray:
        """
        Explore latent reasoning paths using the selected backend.

        Args:
            context_embedding: Input embedding [Batch, Dim] or [Batch, Seq, Dim]

        Returns:
            refined_embedding: Enhanced embedding [Batch, Dim]
        """
        self._sync_backend_runtime_config()
        embedding = np.asarray(context_embedding, dtype=np.float32)
        normalized = self._normalize_embedding(embedding)

        # Primary backend path
        primary = self.backend.explore(normalized)
        primary_amplitudes = getattr(self.backend, "last_amplitudes", None)
        get_last_session_record = getattr(self.backend, "get_last_session_record", None)
        self.last_session_record = (
            get_last_session_record() if callable(get_last_session_record) else None
        )

        # If amplitudes are already meaningful, keep backend behavior.
        if self._has_signal(primary_amplitudes):
            return primary

        # Fallback: explicitly diversify path seeds and score outputs.
        num_paths = max(2, int(self.config.get("num_paths", 12)))
        candidates = []
        scores = []
        for idx in range(num_paths):
            rotated = self._rotate_embedding(normalized, idx, num_paths)
            candidate = self.backend.explore(rotated)
            candidates.append(candidate)
            scores.append(self._path_score(candidate, normalized))

        amplitudes = self._softmax(np.array(scores, dtype=np.float32))
        try:
            self.backend.last_amplitudes = amplitudes
        except Exception:
            pass

        best_idx = int(np.argmax(amplitudes))
        return candidates[best_idx]

    def explore_adaptive(
        self,
        context_embedding: np.ndarray,
        min_steps: int = 1,
        max_steps: Optional[int] = None,
        entropy_threshold: Optional[float] = None,
        confidence_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, AdaptiveExplorationMetrics]:
        """
        Adaptive-depth exploration driven by path entropy/confidence feedback.
        """
        self._sync_backend_runtime_config()

        configured_steps = int(self.config.get("steps", 6))
        configured_min = int(self.config.get("adaptive_min_steps", min_steps))
        configured_max = int(
            self.config.get(
                "adaptive_max_steps",
                max_steps if max_steps is not None else configured_steps,
            )
        )
        min_steps = max(1, configured_min)
        max_steps = max(min_steps, configured_max)

        entropy_threshold = float(
            self.config.get(
                "adaptive_entropy_threshold",
                0.3 if entropy_threshold is None else entropy_threshold,
            )
        )
        confidence_threshold = float(
            self.config.get(
                "adaptive_confidence_threshold",
                0.85 if confidence_threshold is None else confidence_threshold,
            )
        )

        normalized = self._normalize_embedding(
            np.asarray(context_embedding, dtype=np.float32)
        )
        refined = normalized
        actual_steps = 0
        step_entropies: List[float] = []
        step_confidences: List[float] = []
        termination_reason = "max_steps_reached"
        final_entropy: Optional[float] = None
        final_confidence: Optional[float] = None
        final_amplitudes: Optional[np.ndarray] = None

        for _ in range(max_steps):
            actual_steps += 1
            refined = self.backend.explore(refined)
            amplitudes = getattr(self.backend, "last_amplitudes", None)
            get_last_session_record = getattr(
                self.backend, "get_last_session_record", None
            )
            self.last_session_record = (
                get_last_session_record() if callable(get_last_session_record) else None
            )
            final_amplitudes = (
                np.asarray(amplitudes, dtype=np.float32)
                if amplitudes is not None
                else None
            )

            entropy_val = amplitude_entropy(final_amplitudes)
            confidence_val = amplitude_confidence(final_amplitudes)
            final_entropy = entropy_val
            final_confidence = confidence_val

            if entropy_val is not None:
                step_entropies.append(float(entropy_val))
            if confidence_val is not None:
                step_confidences.append(float(confidence_val))

            if actual_steps < min_steps:
                continue

            if confidence_val is not None and confidence_val >= confidence_threshold:
                termination_reason = "confidence_threshold"
                break

            if entropy_val is not None and entropy_val <= entropy_threshold:
                termination_reason = "entropy_threshold"
                break

        metrics = AdaptiveExplorationMetrics(
            actual_steps=actual_steps,
            max_steps=max_steps,
            min_steps=min_steps,
            final_entropy=final_entropy,
            final_confidence=final_confidence,
            path_amplitudes=(
                [float(v) for v in np.asarray(final_amplitudes).reshape(-1)]
                if final_amplitudes is not None
                else None
            ),
            step_entropies=step_entropies,
            step_confidences=step_confidences,
            termination_reason=termination_reason,
        )
        self.last_adaptive_metrics = metrics
        return refined, metrics

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        arr = embedding.astype(np.float32, copy=True)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.ndim == 2:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            return arr / norms

        if arr.ndim >= 3:
            flat = arr.reshape(arr.shape[0], -1)
            norms = np.linalg.norm(flat, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            return arr / norms.reshape((arr.shape[0],) + (1,) * (arr.ndim - 1))

        return arr

    def _rotate_embedding(
        self, embedding: np.ndarray, idx: int, total_paths: int
    ) -> np.ndarray:
        phase = (2.0 * np.pi * float(idx)) / float(max(total_paths, 1))
        if embedding.ndim < 2:
            return embedding

        # Deterministic high-dimensional "rotation" via roll + phase blend.
        rolled = np.roll(embedding, shift=idx + 1, axis=-1)
        mixed = np.cos(phase) * embedding + np.sin(phase) * rolled
        return self._normalize_embedding(mixed)

    def _path_score(self, candidate: np.ndarray, reference: np.ndarray) -> float:
        cand = np.asarray(candidate, dtype=np.float32)
        ref = np.asarray(reference, dtype=np.float32)
        if cand.shape != ref.shape:
            try:
                ref = np.broadcast_to(ref, cand.shape)
            except Exception:
                pass

        cand_flat = cand.reshape(cand.shape[0], -1)
        ref_flat = ref.reshape(ref.shape[0], -1)
        cand_norm = np.linalg.norm(cand_flat, axis=1) + 1e-8
        ref_norm = np.linalg.norm(ref_flat, axis=1) + 1e-8
        cos = np.sum(cand_flat * ref_flat, axis=1) / (cand_norm * ref_norm)
        return float(np.mean(cos))

    def _softmax(self, scores: np.ndarray, temperature: float = 0.7) -> np.ndarray:
        temp = max(temperature, 1e-4)
        normalized = scores / temp
        normalized -= np.max(normalized)
        exp_scores = np.exp(normalized)
        denom = np.sum(exp_scores)
        if denom <= 0:
            return np.ones_like(scores) / float(len(scores))
        return exp_scores / denom

    def _has_signal(self, amplitudes: Optional[np.ndarray]) -> bool:
        if amplitudes is None:
            return False
        arr = np.asarray(amplitudes, dtype=np.float32)
        if arr.size == 0:
            return False
        if not np.isfinite(arr).all():
            return False
        return float(np.max(arr) - np.min(arr)) > 0.01

    def _path_entropy(self, amplitudes: Optional[np.ndarray]) -> Optional[float]:
        return amplitude_entropy(amplitudes)

    def _sync_backend_runtime_config(self) -> None:
        """Keep backend runtime knobs aligned with mutable config values."""
        try:
            backend_paths = int(self.config.get("num_paths", 12))
            if hasattr(self.backend, "num_paths"):
                self.backend.num_paths = max(1, backend_paths)
            backend_steps = int(self.config.get("steps", 6))
            if hasattr(self.backend, "steps"):
                self.backend.steps = max(1, backend_steps)
        except Exception:
            # Runtime config sync is best-effort only.
            pass

    @property
    def amplitudes(self) -> Optional[np.ndarray]:
        """Return path amplitudes from the last exploration if available."""
        return getattr(self.backend, "last_amplitudes", None)

    @property
    def session_record(self) -> Optional[Dict[str, Any]]:
        return self.last_session_record

    def get_device_info(self) -> Dict[str, Any]:
        """Return information about the current backend and device."""
        return self.backend.get_device_info()

    def __repr__(self):
        info = self.get_device_info()
        return (
            f"ContinuousThoughtBlock(backend={info['backend']}, "
            f"paths={self.config['num_paths']}, steps={self.config['steps']})"
        )
