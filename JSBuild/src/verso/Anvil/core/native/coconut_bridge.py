import numpy as np
import ctypes
import os
from typing import Tuple, Optional
import logging

from config.settings import PERFORMANCE_CONFIG
from core.native.native_ops import load_native_library, native_lib_candidates

logger = logging.getLogger(__name__)


class NativeCoconutBridgeUnavailableError(RuntimeError):
    """Raised when strict native bridge mode is enabled and C++ bridge is unavailable."""


class CoconutNativeBridge:
    def __init__(
        self,
        lib_path: str = None,
        embedding_dim: int = 4096,
        use_gpu: bool = None,
        strict_native: Optional[bool] = None,
    ):
        if use_gpu:
            logger.info("COCONUT GPU mode is disabled in CPU-only configuration.")
        self.use_gpu = False
        self.lib = None
        self.native_available = False
        self.strict_native = (
            bool(strict_native)
            if strict_native is not None
            else bool(
                PERFORMANCE_CONFIG.get("strict_coconut_bridge", False)
                or PERFORMANCE_CONFIG.get("strict_native_qsg", False)
            )
        )

        if not self.use_gpu:
            if lib_path is None:
                candidate = native_lib_candidates()
                lib_path = next(
                    (str(path) for path in candidate if path.exists()),
                    str(candidate[0]),
                )

            if not os.path.exists(lib_path):
                message = (
                    "Native COCONUT bridge library not found at "
                    f"{lib_path}. Native-only execution cannot continue."
                )
                raise NativeCoconutBridgeUnavailableError(message)

            self.lib = load_native_library()
            self.native_available = True

            # Define function signatures
            self.lib.coconut_expand_paths_c.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # hidden_state
                ctypes.POINTER(ctypes.c_float),  # paths
                ctypes.c_int64,  # batch_size
                ctypes.c_int64,  # num_paths
                ctypes.c_int64,  # dim
                ctypes.c_float,  # noise_scale
            ]

            self.lib.coconut_evolve_paths_c.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # paths
                ctypes.POINTER(ctypes.c_float),  # norm_gamma
                ctypes.POINTER(ctypes.c_float),  # norm_beta
                ctypes.POINTER(ctypes.c_float),  # dense1_weight
                ctypes.POINTER(ctypes.c_float),  # dense1_bias
                ctypes.POINTER(ctypes.c_float),  # dense2_weight
                ctypes.POINTER(ctypes.c_float),  # dense2_bias
                ctypes.c_int64,  # batch_size
                ctypes.c_int64,  # num_paths
                ctypes.c_int64,  # dim
                ctypes.c_int64,  # hidden_dim
                ctypes.POINTER(ctypes.c_float),  # work_buffer
            ]

            self.lib.coconut_amplitude_score_c.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # paths
                ctypes.POINTER(ctypes.c_float),  # context
                ctypes.POINTER(ctypes.c_float),  # amplitudes
                ctypes.c_int64,  # batch_size
                ctypes.c_int64,  # num_paths
                ctypes.c_int64,  # dim
            ]

            self.lib.coconut_aggregate_paths_c.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # paths
                ctypes.POINTER(ctypes.c_float),  # amplitudes
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_int64,  # batch_size
                ctypes.c_int64,  # num_paths
                ctypes.c_int64,  # dim
            ]

    def _ensure_native_available(self, operation: str) -> None:
        if not self.native_available:
            raise NativeCoconutBridgeUnavailableError(
                f"Native COCONUT bridge unavailable for '{operation}'."
            )

    def expand_paths(
        self, hidden_state: np.ndarray, num_paths: int, noise_scale: float = 0.01
    ) -> np.ndarray:
        self._ensure_native_available("expand_paths")

        batch_size, dim = hidden_state.shape
        paths = np.zeros((batch_size, num_paths, dim), dtype=np.float32)

        self.lib.coconut_expand_paths_c(
            hidden_state.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            paths.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            batch_size,
            num_paths,
            dim,
            noise_scale,
        )
        return paths

    def evolve_paths(
        self,
        paths: np.ndarray,
        norm_gamma: np.ndarray,
        norm_beta: np.ndarray,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
        hidden_dim: int,
    ) -> np.ndarray:
        self._ensure_native_available("evolve_paths")

        batch_size, num_paths, dim = paths.shape
        work_buffer = np.zeros((batch_size * num_paths, hidden_dim), dtype=np.float32)

        self.lib.coconut_evolve_paths_c(
            paths.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            norm_gamma.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            norm_beta.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            w1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            w2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            batch_size,
            num_paths,
            dim,
            hidden_dim,
            work_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return paths

    def score_paths(self, paths: np.ndarray, context: np.ndarray) -> np.ndarray:
        self._ensure_native_available("score_paths")

        batch_size, num_paths, dim = paths.shape
        amplitudes = np.zeros((batch_size, num_paths), dtype=np.float32)

        self.lib.coconut_amplitude_score_c(
            paths.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            context.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            amplitudes.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            batch_size,
            num_paths,
            dim,
        )
        return amplitudes

    def aggregate_paths(self, paths: np.ndarray, amplitudes: np.ndarray) -> np.ndarray:
        self._ensure_native_available("aggregate_paths")

        batch_size, num_paths, dim = paths.shape
        output = np.zeros((batch_size, dim), dtype=np.float32)

        self.lib.coconut_aggregate_paths_c(
            paths.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            amplitudes.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            batch_size,
            num_paths,
            dim,
        )
        return output

    def verify_paths_with_consistency(
        self,
        paths: np.ndarray,
        amplitudes: np.ndarray,
        verification_threshold: float = 0.7,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Self-consistency verification for multi-path reasoning.

        Computes pairwise agreement between paths and aggregates based on
        consistency scores. Implements simplified DeepSeek-R1 style verification.

        Args:
            paths: Multi-path states [batch, num_paths, dim]
            amplitudes: Path quality scores [batch, num_paths]
            verification_threshold: Minimum agreement for high confidence

        Returns:
            Tuple of:
                - verified_output: Consistency-weighted aggregated output [batch, dim]
                - confidence_scores: Verification confidence per batch [batch]
        """
        batch_size, num_paths, dim = paths.shape

        if num_paths < 2:
            # Not enough paths for consistency check
            return self.aggregate_paths(paths, amplitudes), amplitudes.mean(axis=1)

        # Compute pairwise cosine similarities between paths
        # paths: [batch, num_paths, dim]
        verified_outputs = []
        confidence_scores = []

        for b in range(batch_size):
            batch_paths = paths[b]  # [num_paths, dim]
            batch_amps = amplitudes[b]  # [num_paths]

            # Normalize paths for cosine similarity
            norms = np.linalg.norm(batch_paths, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)  # Avoid division by zero
            normalized_paths = batch_paths / norms  # [num_paths, dim]

            # Pairwise similarity matrix [num_paths, num_paths]
            similarity_matrix = normalized_paths @ normalized_paths.T

            # Compute agreement scores for each path
            # Agreement = average similarity with other high-amplitude paths
            agreement_scores = np.zeros(num_paths, dtype=np.float32)
            for i in range(num_paths):
                # Weight similarities by amplitudes of other paths
                weighted_similarities = similarity_matrix[i] * batch_amps
                agreement_scores[i] = weighted_similarities.sum() / (
                    batch_amps.sum() + 1e-8
                )

            # Combine amplitude and agreement for final weights
            # Final weight = amplitude * agreement
            combined_weights = batch_amps * agreement_scores

            # Normalize weights
            weight_sum = combined_weights.sum()
            if weight_sum > 1e-8:
                combined_weights /= weight_sum
            else:
                combined_weights = np.ones(num_paths, dtype=np.float32) / num_paths

            # Aggregate paths using combined weights
            verified_output = (batch_paths.T @ combined_weights).reshape(dim)

            # Confidence = average agreement score (weighted by amplitudes)
            confidence = (agreement_scores * batch_amps).sum() / (
                batch_amps.sum() + 1e-8
            )

            verified_outputs.append(verified_output)
            confidence_scores.append(confidence)

        return np.array(verified_outputs, dtype=np.float32), np.array(
            confidence_scores, dtype=np.float32
        )
