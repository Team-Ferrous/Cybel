import numpy as np
import numpy.fft as fft
from typing import Dict
import ctypes
from pathlib import Path


class HolographicEncoder:
    """
    Implements Holographic State Evolution (HSE) encoding using Circular Convolution.
    Binds token embeddings with position embeddings to create a position-aware
    superposition state.

    Formula: |ψ> = Sum( Token_i * Position_i )
    where * is circular convolution.

    Uses accelerated C++ SIMD kernels if available.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.rng = np.random.default_rng(seed=42)
        self._position_cache: Dict[int, np.ndarray] = {}

        # Load accelerated library
        self.lib = self._load_simd_lib()
        if self.lib:
            print("HolographicEncoder: Using accelerated C++ SIMD kernels.")

    def _load_simd_lib(self):
        """Try to load the libhnn_simd.so library."""
        try:
            lib_path = Path(__file__).parent.parent / "simd" / "libhnn_simd.so"
            if not lib_path.exists():
                return None

            lib = ctypes.CDLL(str(lib_path))

            # Define function signatures
            # void circular_convolution_batched(const float* a, const float* b, float* out, int batch, int seq, int hd_dim)
            lib.circular_convolution_batched.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]

            # void simd_hadamard_product_batched(const float* a, const float* b, float* out, int batch, int seq, int hd_dim)
            lib.simd_hadamard_product_batched.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]

            return lib
        except Exception as e:
            print(f"Warning: Could not load SIMD library: {e}")
            return None

    def _get_position_vector(self, idx: int) -> np.ndarray:
        """Get or generate a random hyperdimensional position vector."""
        if idx not in self._position_cache:
            vec = self.rng.standard_normal(self.dim).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-10)
            self._position_cache[idx] = vec
        return self._position_cache[idx]

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Encode a sequence of embeddings into a single holographic state vector.
        """
        if embeddings.ndim == 3:
            batch_size, seq_len, dim = embeddings.shape
            assert dim == self.dim

            if self.lib:
                return self._encode_batch_cpp(embeddings)
            else:
                states = np.zeros((batch_size, dim), dtype=np.float32)
                for b in range(batch_size):
                    states[b] = self._encode_sequence_numpy(embeddings[b])
                return states

        elif embeddings.ndim == 2:
            return self._encode_sequence_numpy(embeddings).reshape(1, -1)
        else:
            raise ValueError(f"Invalid embedding shape: {embeddings.shape}")

    def _encode_batch_cpp(self, embeddings: np.ndarray) -> np.ndarray:
        """Faster batch encoding using SIMD C++ kernels."""
        batch_size, seq_len, dim = embeddings.shape

        # Prepare position vectors as a single block
        pos_vectors = np.stack(
            [self._get_position_vector(i) for i in range(seq_len)]
        ).astype(np.float32)

        # Output buffer for all bindings: [batch, seq, dim]
        bound_output = np.zeros((batch_size, seq_len, dim), dtype=np.float32)

        # Call C++: tokens_hd * position_vectors -> bound_hd (Circular Conv)
        self.lib.circular_convolution_batched(
            embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            pos_vectors.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            bound_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            batch_size,
            seq_len,
            dim,
        )

        # Sum across sequence for superposition
        state = np.sum(bound_output, axis=1)

        # Normalize each batch item
        norms = np.linalg.norm(state, axis=1, keepdims=True)
        return state / (norms + 1e-10)

    def _encode_sequence_numpy(self, seq_embeddings: np.ndarray) -> np.ndarray:
        """Fallback NumPy implementation using Circular Convolution with padding."""
        seq_len, dim = seq_embeddings.shape
        state = np.zeros(dim, dtype=np.float32)

        # Calculate next power of 2
        n_pow2 = 1 << (dim - 1).bit_length()

        for t in range(seq_len):
            token_vec = seq_embeddings[t]
            pos_vec = self._get_position_vector(t)

            # Pad to next power of 2
            a_pad = np.pad(token_vec, (0, n_pow2 - dim))
            b_pad = np.pad(pos_vec, (0, n_pow2 - dim))

            ft_token = fft.fft(a_pad)
            ft_pos = fft.fft(b_pad)
            convolved = fft.ifft(ft_token * ft_pos)
            state += np.real(convolved[:dim])

        norm = np.linalg.norm(state)
        if norm > 1e-9:
            state = state / norm

        return state
