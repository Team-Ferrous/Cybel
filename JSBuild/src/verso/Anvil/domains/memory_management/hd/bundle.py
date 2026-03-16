import numpy as np
from typing import List


class HolographicBundle:
    """
    Holographic Bundle for Hyperdimensional Computing.
    """

    def __init__(self, dimension: int = 10000):
        self.dimension = dimension

    def generate_random_vector(self) -> np.ndarray:
        """Generate a random dense bipolar vector (stored as int8 0/1 for XOR efficiency)."""
        return np.random.randint(0, 2, self.dimension, dtype=np.int8)

    def bind(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Bind two vectors using XOR."""
        return np.bitwise_xor(v1, v2)

    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Superpose multiple vectors using Majority Rule."""
        if not vectors:
            return np.zeros(self.dimension, dtype=np.int8)

        matrix = np.stack(vectors)
        sums = np.sum(matrix, axis=0)

        threshold = len(vectors) / 2.0
        result = (sums > threshold).astype(np.int8)

        return result

    def permute(self, vector: np.ndarray, shifts: int = 1) -> np.ndarray:
        """Permute vector (cyclic shift) to encode sequence/order."""
        return np.roll(vector, shifts)

    def similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute Hamming similarity (normalized)."""
        diffs = np.bitwise_xor(v1, v2)
        dist = np.sum(diffs)
        return 1.0 - (dist / self.dimension)
