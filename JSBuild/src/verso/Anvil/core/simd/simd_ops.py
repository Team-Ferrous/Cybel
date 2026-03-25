import ctypes
import os
import platform
import numpy as np
from typing import Optional


class SIMDOps:
    """
    Python wrapper for HighNoon SIMD operations.
    """

    _lib: Optional[ctypes.CDLL] = None

    def __init__(self):
        self._load_library()

    @property
    def available(self) -> bool:
        return SIMDOps._lib is not None

    @property
    def lib(self):
        return SIMDOps._lib

    def _load_library(self):
        if SIMDOps._lib is not None:
            return

        # Determine library extension based on OS
        system = platform.system()
        if system == "Linux":
            ext = ".so"
        elif system == "Darwin":
            ext = ".dylib"
        elif system == "Windows":
            ext = ".dll"
        else:
            raise RuntimeError(f"Unsupported operating system: {system}")

        # Path to the shared library
        # Assuming the lib is built in core/simd/build or core/native/build
        # We'll look in plausible locations
        base_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(base_dir, f"libhnn_simd{ext}"),
            os.path.join(base_dir, "build", f"libhnn_simd{ext}"),
            os.path.join(
                os.path.dirname(base_dir), "native", "build", f"libhnn_simd{ext}"
            ),
        ]

        lib_path = None
        for path in possible_paths:
            if os.path.exists(path):
                lib_path = path
                break

        if lib_path is None:
            # Fallback for dev environment - maybe it's not built yet?
            # We won't raise error immediately to allow class instantiation if users just want to check existence
            print(
                f"Warning: HighNoon SIMD library not found. Checked: {possible_paths}"
            )
            return

        try:
            SIMDOps._lib = ctypes.CDLL(lib_path)
            self._setup_signatures()
        except OSError as e:
            print(f"Error loading HighNoon SIMD library: {e}")

    # ... (previous signatures retained in implementation below)

    def _setup_signatures(self):
        if SIMDOps._lib is None:
            return

        # Basic primitives
        # void simd_exp_inplace(float* data, int64_t size)
        SIMDOps._lib.simd_exp_inplace.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
        ]
        SIMDOps._lib.simd_exp_inplace.restype = None

        # void simd_log_inplace(float* data, int64_t size)
        SIMDOps._lib.simd_log_inplace.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
        ]
        SIMDOps._lib.simd_log_inplace.restype = None

        # void simd_sigmoid_inplace(float* data, int64_t size)
        SIMDOps._lib.simd_sigmoid_inplace.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
        ]
        SIMDOps._lib.simd_sigmoid_inplace.restype = None

        # void simd_softmax_inplace(float* data, int64_t size)
        SIMDOps._lib.simd_softmax_inplace.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
        ]
        SIMDOps._lib.simd_softmax_inplace.restype = None

        # void simd_silu_inplace(float* data, int64_t size)
        SIMDOps._lib.simd_silu_inplace.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
        ]
        SIMDOps._lib.simd_silu_inplace.restype = None

        # void simd_hadamard_product(float* a, float* b, float* out, int64_t size)
        SIMDOps._lib.simd_hadamard_product.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
        ]
        SIMDOps._lib.simd_hadamard_product.restype = None

        # void simd_hadamard_product_batched(const float* a, const float* b, float* out, int batch, int seq, int hd_dim)
        SIMDOps._lib.simd_hadamard_product_batched.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        SIMDOps._lib.simd_hadamard_product_batched.restype = None

        # float simd_dot_product(const float* a, const float* b, int64_t size)
        SIMDOps._lib.simd_dot_product.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
        ]
        SIMDOps._lib.simd_dot_product.restype = ctypes.c_float

        # float simd_cosine_similarity(const float* a, const float* b, int64_t size)
        SIMDOps._lib.simd_cosine_similarity.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
        ]
        SIMDOps._lib.simd_cosine_similarity.restype = ctypes.c_float

        # void circular_convolution_batched(const float* a, const float* b, float* out, int batch, int seq, int hd_dim)
        SIMDOps._lib.circular_convolution_batched.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        SIMDOps._lib.circular_convolution_batched.restype = None

        # --- NEW KERNELS ---

        # void ssm_scan(...)
        SIMDOps._lib.ssm_scan.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        SIMDOps._lib.ssm_scan.restype = None

        # void simd_flash_attention_forward(...)
        SIMDOps._lib.simd_flash_attention_forward.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_bool,
        ]
        SIMDOps._lib.simd_flash_attention_forward.restype = None

        # void simd_qssm_forward(...)
        SIMDOps._lib.simd_qssm_forward.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        SIMDOps._lib.simd_qssm_forward.restype = None

    # ... (Previous wrapper methods) ...

    def exp_inplace(self, data: np.ndarray) -> None:
        """Computes exp(x) in-place for the given array."""
        if SIMDOps._lib is None:
            # Fallback to numpy if lib not loaded
            np.exp(data, out=data)
            return

        data_contiguous = np.ascontiguousarray(data, dtype=np.float32)
        ptr = data_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        SIMDOps._lib.simd_exp_inplace(ptr, data_contiguous.size)

        # If a copy was made, update the original array
        if data_contiguous is not data:
            np.copyto(data, data_contiguous)

    def log_inplace(self, data: np.ndarray) -> None:
        """Computes log(x) in-place for the given array."""
        if SIMDOps._lib is None:
            np.log(data, out=data)
            return

        data_contiguous = np.ascontiguousarray(data, dtype=np.float32)
        ptr = data_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        SIMDOps._lib.simd_log_inplace(ptr, data_contiguous.size)

        if data_contiguous is not data:
            np.copyto(data, data_contiguous)

    def sigmoid_inplace(self, data: np.ndarray) -> None:
        """Computes sigmoid(x) in-place for the given array."""
        if SIMDOps._lib is None:
            # 1 / (1 + exp(-x))
            coeffs = 1.0 / (1.0 + np.exp(-data))
            np.copyto(data, coeffs)
            return

        data_contiguous = np.ascontiguousarray(data, dtype=np.float32)
        ptr = data_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        SIMDOps._lib.simd_sigmoid_inplace(ptr, data_contiguous.size)

        if data_contiguous is not data:
            np.copyto(data, data_contiguous)

    def softmax(self, data: np.ndarray) -> np.ndarray:
        """Computes softmax(x) in-place (returns input for convenience)."""
        if SIMDOps._lib is None:
            shift = data - np.max(data)
            exps = np.exp(shift)
            return exps / np.sum(exps)

        data_contiguous = np.ascontiguousarray(data, dtype=np.float32)
        ptr = data_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        SIMDOps._lib.simd_softmax_inplace(ptr, data_contiguous.size)
        return data_contiguous

    def silu_inplace(self, data: np.ndarray) -> None:
        """Computes x * sigmoid(x) in-place."""
        if SIMDOps._lib is None:
            data[:] = data * (1 / (1 + np.exp(-data)))
            return

        data_contiguous = np.ascontiguousarray(data, dtype=np.float32)
        # Some builds expose simd_silu_inplace with inconsistent numerics; compose
        # SiLU from the validated sigmoid kernel for stable behavior.
        original = data_contiguous.copy()
        ptr = data_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        SIMDOps._lib.simd_sigmoid_inplace(ptr, data_contiguous.size)
        data_contiguous *= original
        if data_contiguous is not data:
            np.copyto(data, data_contiguous)

    def dot_product(self, a: np.ndarray, b: np.ndarray) -> float:
        """Computes dot(a, b) using SIMD when available."""
        a_arr = np.ascontiguousarray(a, dtype=np.float32).reshape(-1)
        b_arr = np.ascontiguousarray(b, dtype=np.float32).reshape(-1)
        if a_arr.size != b_arr.size:
            raise ValueError(
                f"dot_product shape mismatch: {a_arr.size} != {b_arr.size}"
            )
        if a_arr.size == 0:
            return 0.0
        if SIMDOps._lib is None:
            return float(np.dot(a_arr, b_arr))

        a_ptr = a_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        return float(SIMDOps._lib.simd_dot_product(a_ptr, b_ptr, a_arr.size))

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Computes cosine similarity using SIMD when available."""
        a_arr = np.ascontiguousarray(a, dtype=np.float32).reshape(-1)
        b_arr = np.ascontiguousarray(b, dtype=np.float32).reshape(-1)
        if a_arr.size != b_arr.size:
            raise ValueError(
                f"cosine_similarity shape mismatch: {a_arr.size} != {b_arr.size}"
            )
        if a_arr.size == 0:
            return 0.0

        if SIMDOps._lib is None:
            denom = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
            if denom <= 1e-12:
                return 0.0
            return float(np.dot(a_arr, b_arr) / denom)

        a_ptr = a_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        return float(SIMDOps._lib.simd_cosine_similarity(a_ptr, b_ptr, a_arr.size))

    def hadamard_product(self, a_ptr, b_ptr, out_ptr, size):
        """Computes element-wise multiplication: out = a * b"""
        if SIMDOps._lib:
            SIMDOps._lib.simd_hadamard_product(
                a_ptr, b_ptr, out_ptr, ctypes.c_int64(size)
            )
        else:
            # Fallback to NumPy
            import numpy as np

            a_arr = np.frombuffer(
                (ctypes.c_float * size).from_address(ctypes.addressof(a_ptr.contents)),
                dtype=np.float32,
            )
            b_arr = np.frombuffer(
                (ctypes.c_float * size).from_address(ctypes.addressof(b_ptr.contents)),
                dtype=np.float32,
            )
            out_arr = np.frombuffer(
                (ctypes.c_float * size).from_address(
                    ctypes.addressof(out_ptr.contents)
                ),
                dtype=np.float32,
            )
            out_arr[:] = a_arr * b_arr

    def hadamard_product_batched(self, a_ptr, b_ptr, out_ptr, batch, seq, hd_dim):
        """Computes batched element-wise multiplication with broadcasting."""
        if SIMDOps._lib:
            SIMDOps._lib.simd_hadamard_product_batched(
                a_ptr, b_ptr, out_ptr, batch, seq, hd_dim
            )
        else:
            # Fallback to NumPy broadcasting
            import numpy as np

            size_a = batch * seq * hd_dim
            size_b = seq * hd_dim
            # Cast pins to numpy arrays
            # This is complex for broadcasting, but we do it per-batch for simplicity in fallback
            a = np.frombuffer(
                (ctypes.c_float * size_a).from_address(
                    ctypes.addressof(a_ptr.contents)
                ),
                dtype=np.float32,
            ).reshape(batch, seq, hd_dim)
            b = np.frombuffer(
                (ctypes.c_float * size_b).from_address(
                    ctypes.addressof(b_ptr.contents)
                ),
                dtype=np.float32,
            ).reshape(seq, hd_dim)
            out = np.frombuffer(
                (ctypes.c_float * size_a).from_address(
                    ctypes.addressof(out_ptr.contents)
                ),
                dtype=np.float32,
            ).reshape(batch, seq, hd_dim)
            out[:] = a * b[np.newaxis, :, :]

    def circular_convolution(self, a_ptr, b_ptr, out_ptr, hd_dim):
        """Computes batched circular convolution: out = a ⊛ b"""
        if SIMDOps._lib:
            SIMDOps._lib.circular_convolution_batched(
                a_ptr, b_ptr, out_ptr, 1, 1, hd_dim
            )
        else:
            # Fallback to NumPy FFT
            import numpy as np

            a_arr = np.frombuffer(
                (ctypes.c_float * hd_dim).from_address(
                    ctypes.addressof(a_ptr.contents)
                ),
                dtype=np.float32,
            )
            b_arr = np.frombuffer(
                (ctypes.c_float * hd_dim).from_address(
                    ctypes.addressof(b_ptr.contents)
                ),
                dtype=np.float32,
            )
            out_arr = np.frombuffer(
                (ctypes.c_float * hd_dim).from_address(
                    ctypes.addressof(out_ptr.contents)
                ),
                dtype=np.float32,
            )
            out_arr[:] = np.real(
                np.fft.ifft(np.fft.fft(a_arr) * np.fft.fft(b_arr))
            ).astype(np.float32)

    def ssm_scan(self, x, A_log, dt, B, C, D, output, h_final=None):
        """Mamba SSM Scan."""
        if SIMDOps._lib is None:
            return

        batch, seq, d_inner = x.shape
        state_dim = B.shape[2]

        # Ensure contiguous pointers
        x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        A_ptr = A_log.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        dt_ptr = dt.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        C_ptr = C.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        D_ptr = D.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        h_ptr = (
            h_final.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            if h_final is not None
            else ctypes.POINTER(ctypes.c_float)()
        )

        SIMDOps._lib.ssm_scan(
            x_ptr,
            A_ptr,
            dt_ptr,
            B_ptr,
            C_ptr,
            D_ptr,
            out_ptr,
            h_ptr,
            batch,
            seq,
            d_inner,
            state_dim,
        )

    def flash_attention(self, q, k, v, output, scale=0.0, causal=True):
        """Unified Flash Attention."""
        if SIMDOps._lib is None:
            return

        batch, heads, seq, head_dim = q.shape
        kv_seq = k.shape[2]

        q_ptr = q.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        k_ptr = k.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        v_ptr = v.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        SIMDOps._lib.simd_flash_attention_forward(
            q_ptr,
            k_ptr,
            v_ptr,
            out_ptr,
            batch,
            heads,
            head_dim,
            seq,
            kv_seq,
            scale,
            causal,
        )
