import time

import numpy as np
from core.simd.simd_ops import SIMDOps
from core.qsg.runtime_contracts import BranchCoherenceMatrix, JacobiFrontierState


class JacobiRefiner:
    """
    Jacobi Iterative Refinement for Sequence Consistency.

    Treats the sequence generation as a system of linear equations A*x = b,
    where x are the token embeddings and A represents the constraints (grammar, context).
    Iteratively solves for x (tokens) to minimize inconsistency energy.
    """

    def __init__(self, iterations: int = 2):
        self.iterations = iterations
        self.simd = SIMDOps()
        self.last_frontier_state = JacobiFrontierState()

    @staticmethod
    def _normalize_probs(probs: np.ndarray) -> np.ndarray:
        normalized = np.asarray(probs, dtype=np.float32)
        row_sums = normalized.sum(axis=-1, keepdims=True)
        row_sums = np.where(row_sums < 1.0e-12, 1.0, row_sums)
        normalized = normalized / row_sums
        if np.any(np.isnan(normalized)):
            normalized = np.nan_to_num(normalized, nan=1.0 / normalized.shape[-1])
        return normalized

    def _build_neighbor_coherence(self, refined: np.ndarray) -> np.ndarray:
        left = np.roll(refined, shift=1, axis=1)
        right = np.roll(refined, shift=-1, axis=1)
        left[:, 0, :] = refined[:, 0, :]
        right[:, -1, :] = refined[:, -1, :]
        return 0.5 * (left + right)

    def verify_frontier(
        self,
        draft_probs: np.ndarray,
        coherence_matrix: np.ndarray = None,
        *,
        branch_priors: np.ndarray | None = None,
        frontier_width: int | None = None,
        replay_costs: np.ndarray | None = None,
    ) -> JacobiFrontierState:
        started = time.perf_counter()
        probs = self._normalize_probs(draft_probs)
        batch_size, seq_len, vocab_size = probs.shape
        width = int(frontier_width or min(4, vocab_size))
        width = max(1, min(width, vocab_size))

        neighbors = self._build_neighbor_coherence(probs)
        if coherence_matrix is not None:
            coherence = np.asarray(coherence_matrix, dtype=np.float32)
            if coherence.ndim == 2 and coherence.shape == (seq_len, seq_len):
                weighted_neighbors = np.einsum("ij,bjv->biv", coherence, probs)
                weighted_neighbors = self._normalize_probs(weighted_neighbors)
                neighbors = 0.5 * (neighbors + weighted_neighbors)
        combined = 0.7 * probs + 0.3 * neighbors

        if branch_priors is not None:
            priors = np.asarray(branch_priors, dtype=np.float32)
            if priors.ndim == 2:
                priors = priors[:, :, np.newaxis]
            if priors.ndim == 3 and priors.shape[:2] == (batch_size, seq_len):
                if priors.shape[-1] == 1:
                    priors = np.broadcast_to(priors, combined.shape)
                elif priors.shape[-1] != vocab_size:
                    priors = None
            else:
                priors = None
            if priors is not None:
                combined = combined + (0.1 * np.maximum(priors, 0.0))

        if replay_costs is not None:
            costs = np.asarray(replay_costs, dtype=np.float32)
            if costs.ndim == 2 and costs.shape == (batch_size, seq_len):
                costs = costs[:, :, np.newaxis]
            if costs.ndim == 3 and costs.shape[:2] == (batch_size, seq_len):
                if costs.shape[-1] == 1:
                    costs = np.broadcast_to(costs, combined.shape)
                elif costs.shape[-1] != vocab_size:
                    costs = None
            else:
                costs = None
            if costs is not None:
                combined = np.maximum(combined - (0.05 * np.maximum(costs, 0.0)), 0.0)

        combined = self._normalize_probs(combined)
        top_indices = np.argpartition(combined, -width, axis=-1)[..., -width:]
        top_scores = np.take_along_axis(combined, top_indices, axis=-1)
        top_scores = self._normalize_probs(top_scores)
        entropy = -np.sum(top_scores * np.log(top_scores + 1.0e-10), axis=-1)
        threshold = np.max(top_scores, axis=-1, keepdims=True) * 0.30
        survivors = top_scores >= threshold
        survival_rate = float(np.mean(survivors))
        verify_cost_ms = float((time.perf_counter() - started) * 1000.0)

        coherence_preview = []
        for row in range(min(4, seq_len)):
            if coherence_matrix is not None and np.asarray(coherence_matrix).shape == (
                seq_len,
                seq_len,
            ):
                coherence_preview.append(
                    [
                        float(v)
                        for v in np.asarray(coherence_matrix)[row, : min(4, seq_len)]
                    ]
                )
            else:
                coherence_preview.append(
                    [
                        float(v)
                        for v in np.mean(
                            neighbors[:, row, : min(4, vocab_size)], axis=0
                        )
                    ]
                )
        return JacobiFrontierState(
            frontier_width=width,
            branch_survival_rate=survival_rate,
            verify_cost_ms=verify_cost_ms,
            branch_entropy=float(np.mean(entropy)),
            branch_scores=(
                top_scores[0].astype(np.float32).tolist() if batch_size > 0 else []
            ),
            surviving_tokens=(
                top_indices[0].astype(np.int32).tolist() if batch_size > 0 else []
            ),
            coherence=BranchCoherenceMatrix(
                matrix=coherence_preview,
                source=(
                    "explicit" if coherence_matrix is not None else "neighbor_agreement"
                ),
            ),
        )

    def refine(
        self,
        draft_probs: np.ndarray,
        coherence_matrix: np.ndarray = None,
        *,
        branch_priors: np.ndarray | None = None,
        frontier_width: int | None = None,
        replay_costs: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Apply Jacobi updates to refine token probabilities based on neighbor consistency.

        Args:
           draft_probs: [Batch, SeqLen, Vocab] Initial probabilities
           coherence_matrix: [SeqLen, SeqLen] (Optional) Matrix indicating position dependencies.
                             If None, assumes simple windowed dependencies (tridiagonal).

        Returns:
            Refined probabilities [Batch, SeqLen, Vocab]
        """
        # Simplified simulation of Jacobi refinement for integration
        # Real implementation would use the HighNoon attention mask as coherence matrix

        # We simulate "energy minimization" by smoothing probability spikes
        # that disagree with neighbors (like a CRF or MRF)

        refined = self._normalize_probs(draft_probs.copy())
        frontier = self.verify_frontier(
            refined,
            coherence_matrix,
            branch_priors=branch_priors,
            frontier_width=frontier_width,
            replay_costs=replay_costs,
        )
        self.last_frontier_state = frontier

        width = max(1, int(frontier.frontier_width or 1))
        for _ in range(self.iterations):
            neighbors = self._build_neighbor_coherence(refined)
            if coherence_matrix is not None:
                coherence = np.asarray(coherence_matrix, dtype=np.float32)
                if coherence.ndim == 2 and coherence.shape == (
                    refined.shape[1],
                    refined.shape[1],
                ):
                    weighted_neighbors = np.einsum("ij,bjv->biv", coherence, refined)
                    weighted_neighbors = self._normalize_probs(weighted_neighbors)
                    neighbors = 0.5 * (neighbors + weighted_neighbors)

            candidate_scores = self._normalize_probs(
                (0.72 * refined) + (0.28 * neighbors)
            )
            top_indices = np.argpartition(candidate_scores, -width, axis=-1)[
                ..., -width:
            ]
            mask = np.zeros_like(candidate_scores, dtype=bool)
            np.put_along_axis(mask, top_indices, True, axis=-1)
            candidate_scores = np.where(mask, candidate_scores, candidate_scores * 0.15)
            refined = self._normalize_probs(candidate_scores)

        return refined

    def refine_logits(self, logits: np.ndarray) -> np.ndarray:
        """
        Directly refine logits (more common for generation).
        1. Softmax -> Probs
        2. Refine
        3. Log -> Logits
        """
        # 1. Softmax
        # (This duplicates logic effectively, but needed for interface)
        max_vals = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - max_vals)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # 2. Refine
        refined_probs = self.refine(probs)

        # 3. Back to logits (approx)
        # Add epsilon to avoid log(0)
        refined_logits = np.log(refined_probs + 1e-10)

        return refined_logits
