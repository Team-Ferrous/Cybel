import numpy as np
from core.simd.simd_ops import SIMDOps

# Try to use optimized attention kernels if available
try:
    from core.native.fast_attention_wrapper import (
        OptimizedHopfieldVocab as _OptimizedHopfieldVocab,
    )

    FAST_ATTENTION_AVAILABLE = True
except ImportError:
    FAST_ATTENTION_AVAILABLE = False
    _OptimizedHopfieldVocab = None

try:
    from config.settings import PERFORMANCE_CONFIG
except ImportError:
    PERFORMANCE_CONFIG = {"fast_attention_kernels": False}


class HopfieldVocab:
    """
    Continuous Hopfield Network for Vocabulary Association.

    Stores vocabulary embeddings as patterns. Given a query vector (context state),
    retrieves the most similar token embeddings via energy minimization.

    Ref: "Hopfield Networks is all you need" (Ramsauer et al. 2020)
    """

    def __init__(self, vocab_embeddings: np.ndarray, beta: float = 1.0):
        """
        Args:
            vocab_embeddings: [VocabSize, Dim] matrix of token embeddings.
            beta: Inverse temperature parameter.
        """
        self.patterns = vocab_embeddings
        self.beta = beta
        self.simd = SIMDOps()

    def retrieve(self, query: np.ndarray) -> np.ndarray:
        """
        Retrieve associated pattern (update state) for a query.

        Args:
            query: [Batch, Dim] query vectors.

        Returns:
            [Batch, Dim] updated state vectors (mixture of patterns).
        """
        # 1. Similarity: logits = beta * Query @ Patterns.T
        # [Batch, Dim] @ [Dim, Vocab] -> [Batch, Vocab]
        logits = self.beta * (query @ self.patterns.T)

        # 2. Attention (Softmax): p = softmax(logits)
        # Use SIMD exp for speed
        # Subtract max for stability
        max_logits = np.max(logits, axis=-1, keepdims=True)
        shifted = logits - max_logits

        # SIMD In-place exp
        # Flatten for SIMD op if necessary, or just call on array
        # Our wrapper handles numpy arrays
        # Note: SIMD wrapper might expect float32
        if shifted.dtype != np.float32:
            shifted = shifted.astype(np.float32)

        self.simd.exp_inplace(shifted)

        sums = np.sum(shifted, axis=-1, keepdims=True)
        attention = shifted / sums

        # 3. Update: State = Attention @ Patterns
        # [Batch, Vocab] @ [Vocab, Dim] -> [Batch, Dim]
        output = attention @ self.patterns

        return output

    def get_token_probs(self, query: np.ndarray) -> np.ndarray:
        """
        Get probability distribution over tokens for a query state.
        Basically step 1 & 2 of retrieval.
        """
        logits = self.beta * (query @ self.patterns.T)

        max_logits = np.max(logits, axis=-1, keepdims=True)
        shifted = logits - max_logits

        if shifted.dtype != np.float32:
            shifted = shifted.astype(np.float32)

        self.simd.exp_inplace(shifted)
        sums = np.sum(shifted, axis=-1, keepdims=True)
        probs = shifted / sums

        return probs


# Factory function to get the best available implementation
def create_hopfield_vocab(vocab_embeddings: np.ndarray, beta: float = 1.0):
    """
    Create a Hopfield vocabulary network with the best available implementation.

    Args:
        vocab_embeddings: [VocabSize, Dim] matrix of token embeddings
        beta: Inverse temperature parameter

    Returns:
        HopfieldVocab instance (optimized if available)
    """
    if (
        PERFORMANCE_CONFIG.get("fast_attention_kernels", False)
        and FAST_ATTENTION_AVAILABLE
    ):
        try:
            return _OptimizedHopfieldVocab(vocab_embeddings, beta)
        except Exception as e:
            print(f"Warning: Could not initialize optimized Hopfield network: {e}")
            print("Falling back to standard implementation.")
            return HopfieldVocab(vocab_embeddings, beta)
    else:
        return HopfieldVocab(vocab_embeddings, beta)
