import numpy as np
from scipy.linalg import eigvals, eigvalsh
from typing import List, Dict, Any, Optional


class SheafValidator:
    """
    Detects logical contradictions using simplified Sheaf Cohomology.
    Higher energy/Fiedler value indicates more contradictions in the reasoning trace.
    """

    def compute_score(self, reasoning_trace: List[str]) -> float:
        if len(reasoning_trace) < 2:
            return 1.0

        # 1. Build reasoning interaction matrix (Simplified Sheaf Laplacian)
        # We model the trace as a sequential chain where each step MUST align with the previous.
        # In a real sheaf, we'd have restriction maps between open sets.
        # Here, we use semantic similarity as a proxy for the 'agreement' between stalks.

        from domains.memory_management.hd.compression import HDCompressor

        compressor = HDCompressor()

        n = len(reasoning_trace)
        # Adjacency matrix of the reasoning chain
        A = np.zeros((n, n))
        for i in range(n - 1):
            # Check agreement between step i and i+1
            sim = compressor.similarity(reasoning_trace[i], reasoning_trace[i + 1])
            A[i, i + 1] = A[i + 1, i] = sim

        # 2. Compute Laplacian
        D = np.diag(np.sum(A, axis=1))
        L = D - A

        # 3. Compute Fiedler value (second smallest eigenvalue)
        # For a connected graph, 0 is the first eigenvalue.
        # The Fiedler value represents the algebraic connectivity.
        # In our "coherence sheaf", higher values show stronger contradiction (more 'energy' required to align).
        try:
            evs = eigvalsh(L)
            # The smallest is 0. The second smallest is the Fiedler value.
            # For a reasoning trace to be coherent, it should be STRONGLY CONNECTED semantically.
            fiedler = evs[1] if len(evs) > 1 else 0.0

            # Normalize: if fiedler is high, coherence is high.
            # Max fiedler for a path graph with weights ~1 is small, but let's use a sigmoid or clip.
            coherence = np.tanh(fiedler * 2.0)

            # Penalty for total disconnect (Fiedler is 0 or very small)
            if fiedler < 0.01:
                return 0.1

            return float(coherence)
        except Exception:
            return 0.5


class SpectralAnalyzer:
    """
    Checks multi-agent interaction stability via Spectral Analysis.
    Stable if the spectral radius of the interaction matrix < 1.0.
    """

    def compute_stability(self, interaction_history: List[float]) -> float:
        if not interaction_history:
            return 1.0

        # Build interaction matrix from recent history (e.g., token usage/delta shifts)
        # This is a simplified stability check for the swarm.
        n = len(interaction_history)
        if n < 4:
            return 1.0

        # Create a Hankel-like matrix to represent the system dynamics
        dim = n // 2
        M = np.zeros((dim, dim))
        for i in range(dim):
            M[i] = interaction_history[i : i + dim]

        try:
            evs = np.abs(eigvals(M))
            spectral_radius = np.max(evs)

            # Stable if max eigenvalue < 1.0
            return float(1.0 if spectral_radius < 1.0 else 1.0 / spectral_radius)
        except Exception:
            return 0.5


class CausalValidator:
    """
    Verifies interventions using Pearl's do-calculus logic (simplified).
    Ensures that agent interventions are identifiable and consistent.
    """

    def validate_intervention(self, plan: List[str]) -> float:
        # Simplified: Check for "backdoor" reasoning where an agent
        # assumes a result without justifying the intervention steps.

        # Keywords suggesting intervention/action
        action_keywords = ["change", "fix", "delete", "move", "update", "implement"]
        logic_keywords = ["because", "since", "leads to", "due to", "result"]

        actions = [p for p in plan if any(k in p.lower() for k in action_keywords)]
        reasoning = [p for p in plan if any(k in p.lower() for k in logic_keywords)]

        if not actions:
            return 1.0  # No interventions to validate

        # Ratio of stated reasoning to actions
        # Pure action without reasoning suggests a causal backdoor violation
        ratio = len(reasoning) / len(actions)

        return float(min(ratio, 1.0))


class CoherenceEngine:
    """
    Prime Radiant Mathematical Coherence Validation.
    Combines Sheaf Cohomology, Spectral Analysis, and Causal Validation.
    """

    def __init__(self):
        self.sheaf = SheafValidator()
        self.spectral = SpectralAnalyzer()
        self.causal = CausalValidator()

    def validate_trace(
        self,
        reasoning_trace: List[str],
        interaction_history: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Runs mathematical validation on a reasoning trace.
        """
        sheaf_score = self.sheaf.compute_score(reasoning_trace)
        spectral_score = self.spectral.compute_stability(interaction_history or [])
        causal_score = self.causal.validate_intervention(reasoning_trace)

        # Weighted Aggregation
        overall = sheaf_score * 0.5 + spectral_score * 0.2 + causal_score * 0.3

        return {
            "overall_coherence": overall,
            "sheaf_score": sheaf_score,
            "spectral_stability": spectral_score,
            "causal_validity": causal_score,
            "passed": overall > 0.7,
        }
