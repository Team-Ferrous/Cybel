"""ctypes bindings for quantized native matvec kernels.

Optimized for repeated calls — validation and data pointers are cached
on first use, so subsequent matvec/matmul calls have minimal Python overhead.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
from gguf import GGMLQuantizationType, dequantize

from core.native.native_ops import load_native_library

try:
    from config.settings import PERFORMANCE_CONFIG
except Exception:
    PERFORMANCE_CONFIG = {}


# =========================================================================
# Block specs: (elements_per_block, bytes_per_block)
# =========================================================================
QTYPE_Q8_0 = GGMLQuantizationType.Q8_0.value
QTYPE_Q4_K = GGMLQuantizationType.Q4_K.value
QTYPE_Q6_K = GGMLQuantizationType.Q6_K.value
QTYPE_Q4_K_R4 = 112  # Anvil custom repacked layout (4-row block-major)
QTYPE_Q6_K_R4 = 114  # Anvil custom repacked layout (4-row block-major)
QTYPE_Q6_K_LM = 214  # Anvil custom repacked layout for LM-head decode

_BLOCK_SPECS: dict[int, tuple[int, int]] = {
    QTYPE_Q4_K: (256, 144),
    QTYPE_Q6_K: (256, 210),
    QTYPE_Q8_0: (32, 34),
    QTYPE_Q4_K_R4: (256, 144),
    QTYPE_Q6_K_R4: (256, 210),
    QTYPE_Q6_K_LM: (256, 276),
}


@dataclass
class QuantizedMatrix:
    """GGUF-backed quantized matrix stored in row-wise packed blocks.

    After the first `matvec`/`matmul` call, the packed data pointer and
    layout validation are cached so future calls jump straight to the
    C kernel with no Python overhead.
    """

    name: str
    qtype: int
    shape: tuple[int, int]  # (input_dim, output_dim)
    data: np.ndarray  # uint8 packed rows, shape [output_dim, bytes_per_row]
    interleave_factor: int = 1

    # ---- cached state (set lazily) ----
    _validated: bool = field(default=False, init=False, repr=False)
    _packed: np.ndarray | None = field(default=None, init=False, repr=False)
    _data_ptr: int = field(default=0, init=False, repr=False)  # raw void* for ctypes
    _inverse_row_permutation: np.ndarray | None = field(
        default=None, init=False, repr=False
    )

    @property
    def input_dim(self) -> int:
        return int(self.shape[0])

    @property
    def output_dim(self) -> int:
        return int(self.shape[1])

    def ensure_packed(self) -> np.ndarray:
        """Return the packed uint8 data, ensuring it's contiguous and validated."""
        if self._packed is not None:
            return self._packed
        arr = self.data
        if arr.dtype != np.uint8:
            arr = np.asarray(arr, dtype=np.uint8)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        # Validate layout once
        spec = _BLOCK_SPECS.get(self.qtype)
        if spec is not None:
            block_elems, block_bytes = spec
            if self.input_dim % block_elems == 0:
                expected_row_bytes = (self.input_dim // block_elems) * block_bytes
                if arr.shape == (self.output_dim, expected_row_bytes):
                    self._validated = True
        self._packed = arr
        self._data_ptr = arr.ctypes.data
        return arr


_LIB: Optional[ctypes.CDLL] = None

# Pre-resolved C function references keyed by qtype
_MATVEC_FN: dict[int, object] = {}
_MATMUL_FN: dict[int, object] = {}


def _strict_native_qsg_enabled() -> bool:
    raw = os.getenv("ANVIL_STRICT_NATIVE_QSG")
    if raw is not None:
        normalized = str(raw).strip().lower()
        return normalized not in {"0", "false", "no", "off"}
    return bool(PERFORMANCE_CONFIG.get("strict_native_qsg", False))


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        lib = load_native_library()
        _float_p = ctypes.POINTER(ctypes.c_float)
        _voidp = ctypes.c_void_p
        _int = ctypes.c_int

        for fn_name in (
            "simd_matvec_q4k",
            "simd_matvec_q6k",
            "simd_matvec_q8_0",
            "simd_matvec_q4k_r4",
            "simd_matvec_q6k_r4",
            "simd_matvec_q6k_lm",
        ):
            if not hasattr(lib, fn_name):
                continue
            fn = getattr(lib, fn_name)
            fn.argtypes = [_float_p, _voidp, _float_p, _int, _int]
            fn.restype = None

        for fn_name in (
            "simd_matmul_q4k",
            "simd_matmul_q6k",
            "simd_matmul_q8_0",
            "simd_matmul_q4k_r4",
            "simd_matmul_q6k_r4",
        ):
            if not hasattr(lib, fn_name):
                continue
            fn = getattr(lib, fn_name)
            fn.argtypes = [_float_p, _voidp, _float_p, _int, _int, _int]
            fn.restype = None

        if hasattr(lib, "simd_repack_q4k_r4"):
            lib.simd_repack_q4k_r4.argtypes = [_voidp, _voidp, _int, _int]
            lib.simd_repack_q4k_r4.restype = _int
        if hasattr(lib, "simd_repack_q6k_r4"):
            lib.simd_repack_q6k_r4.argtypes = [_voidp, _voidp, _int, _int]
            lib.simd_repack_q6k_r4.restype = _int
        if hasattr(lib, "simd_repack_q6k_lm"):
            lib.simd_repack_q6k_lm.argtypes = [_voidp, _voidp, _int, _int]
            lib.simd_repack_q6k_lm.restype = _int

        # Cache function dispatch tables
        if hasattr(lib, "simd_matvec_q4k"):
            _MATVEC_FN[QTYPE_Q4_K] = lib.simd_matvec_q4k
        if hasattr(lib, "simd_matvec_q4k_r4"):
            _MATVEC_FN[QTYPE_Q4_K_R4] = lib.simd_matvec_q4k_r4
        if hasattr(lib, "simd_matvec_q6k"):
            _MATVEC_FN[QTYPE_Q6_K] = lib.simd_matvec_q6k
        if hasattr(lib, "simd_matvec_q6k_r4"):
            _MATVEC_FN[QTYPE_Q6_K_R4] = lib.simd_matvec_q6k_r4
        if hasattr(lib, "simd_matvec_q6k_lm"):
            _MATVEC_FN[QTYPE_Q6_K_LM] = lib.simd_matvec_q6k_lm
        if hasattr(lib, "simd_matvec_q8_0"):
            _MATVEC_FN[QTYPE_Q8_0] = lib.simd_matvec_q8_0

        if hasattr(lib, "simd_matmul_q4k"):
            _MATMUL_FN[QTYPE_Q4_K] = lib.simd_matmul_q4k
        if hasattr(lib, "simd_matmul_q4k_r4"):
            _MATMUL_FN[QTYPE_Q4_K_R4] = lib.simd_matmul_q4k_r4
        if hasattr(lib, "simd_matmul_q6k"):
            _MATMUL_FN[QTYPE_Q6_K] = lib.simd_matmul_q6k
        if hasattr(lib, "simd_matmul_q6k_r4"):
            _MATMUL_FN[QTYPE_Q6_K_R4] = lib.simd_matmul_q6k_r4
        if hasattr(lib, "simd_matmul_q8_0"):
            _MATMUL_FN[QTYPE_Q8_0] = lib.simd_matmul_q8_0

        # Fused expert SwiGLU FFN
        try:
            lib.simd_fused_expert_swiglu.argtypes = [
                _float_p,
                _int,  # x, in_dim
                _voidp,
                _int,
                _int,  # gate_data, gate_qtype, hidden_dim
                _voidp,
                _int,  # up_data, up_qtype
                _voidp,
                _int,
                _int,  # down_data, down_qtype, out_dim
                _float_p,  # output
            ]
            lib.simd_fused_expert_swiglu.restype = None
        except AttributeError:
            pass

        # Fused multi-expert MoE FFN
        try:
            _int_p = ctypes.POINTER(ctypes.c_int)
            _voidpp = ctypes.POINTER(ctypes.c_void_p)
            lib.simd_fused_moe_ffn.argtypes = [
                _float_p,
                _int,  # x, in_dim
                _int_p,
                _float_p,
                _int,  # expert_indices, expert_weights, top_k
                _voidpp,
                _int,
                _int,  # gate_ptrs, gate_qtype, hidden_dim
                _voidpp,
                _int,  # up_ptrs, up_qtype
                _voidpp,
                _int,
                _int,  # down_ptrs, down_qtype, out_dim
                _float_p,  # output
            ]
            lib.simd_fused_moe_ffn.restype = None
        except AttributeError:
            pass

        _LIB = lib
    return _LIB


def is_quantized_matrix(value: object) -> bool:
    return isinstance(value, QuantizedMatrix)


def interleave_quantized_rows(matrix: QuantizedMatrix, factor: int) -> QuantizedMatrix:
    """Return a row-interleaved QuantizedMatrix with inverse permutation metadata."""
    interleave = int(max(1, factor))
    if interleave <= 1:
        return matrix
    packed = matrix.ensure_packed()
    rows = int(packed.shape[0])
    if rows <= 1:
        return matrix

    order = np.concatenate(
        [
            np.arange(offset, rows, interleave, dtype=np.int64)
            for offset in range(interleave)
        ]
    )
    if order.shape[0] != rows or np.array_equal(order, np.arange(rows, dtype=np.int64)):
        return matrix

    interleaved = np.ascontiguousarray(packed[order])
    result = QuantizedMatrix(
        name=matrix.name,
        qtype=matrix.qtype,
        shape=matrix.shape,
        data=interleaved,
        interleave_factor=interleave,
    )
    result._validated = bool(matrix._validated)
    result._inverse_row_permutation = np.argsort(order)
    return result


def _requires_output_deinterleave(matrix: QuantizedMatrix) -> bool:
    perm = matrix._inverse_row_permutation
    return isinstance(perm, np.ndarray) and perm.size > 0


def has_q4k_r4_support() -> bool:
    lib = _lib()
    return (
        hasattr(lib, "simd_repack_q4k_r4")
        and QTYPE_Q4_K_R4 in _MATVEC_FN
        and QTYPE_Q4_K_R4 in _MATMUL_FN
    )


def has_q6k_r4_support() -> bool:
    lib = _lib()
    return (
        hasattr(lib, "simd_repack_q6k_r4")
        and QTYPE_Q6_K_R4 in _MATVEC_FN
        and QTYPE_Q6_K_R4 in _MATMUL_FN
    )


def has_q6k_lm_support() -> bool:
    lib = _lib()
    return hasattr(lib, "simd_repack_q6k_lm") and QTYPE_Q6_K_LM in _MATVEC_FN


def repack_q4k_r4(matrix: QuantizedMatrix) -> QuantizedMatrix:
    """Repack canonical Q4_K rows into 4-row block-major layout for multi-row GEMV."""
    if matrix.qtype == QTYPE_Q4_K_R4:
        return matrix
    if matrix.qtype != QTYPE_Q4_K:
        return matrix
    if not has_q4k_r4_support():
        return matrix

    packed = matrix.ensure_packed()
    if not matrix._validated:
        return matrix
    if _requires_output_deinterleave(matrix):
        return matrix

    cached = getattr(matrix, "_repacked_q4k_r4", None)
    if isinstance(cached, QuantizedMatrix):
        return cached

    dst = np.empty_like(packed, dtype=np.uint8)
    ok = _lib().simd_repack_q4k_r4(
        ctypes.c_void_p(packed.ctypes.data),
        ctypes.c_void_p(dst.ctypes.data),
        int(matrix.input_dim),
        int(matrix.output_dim),
    )
    if int(ok) != 1:
        return matrix

    out = QuantizedMatrix(
        name=f"{matrix.name}::q4k_r4",
        qtype=QTYPE_Q4_K_R4,
        shape=matrix.shape,
        data=np.ascontiguousarray(dst, dtype=np.uint8),
        interleave_factor=1,
    )
    out._validated = True
    matrix._repacked_q4k_r4 = out
    return out


def repack_q6k_r4(matrix: QuantizedMatrix) -> QuantizedMatrix:
    """Repack canonical Q6_K rows into 4-row block-major layout for multi-row GEMV."""
    if matrix.qtype == QTYPE_Q6_K_R4:
        return matrix
    if matrix.qtype != QTYPE_Q6_K:
        return matrix
    if not has_q6k_r4_support():
        return matrix

    packed = matrix.ensure_packed()
    if not matrix._validated:
        return matrix
    if _requires_output_deinterleave(matrix):
        return matrix

    cached = getattr(matrix, "_repacked_q6k_r4", None)
    if isinstance(cached, QuantizedMatrix):
        return cached

    dst = np.empty_like(packed, dtype=np.uint8)
    ok = _lib().simd_repack_q6k_r4(
        ctypes.c_void_p(packed.ctypes.data),
        ctypes.c_void_p(dst.ctypes.data),
        int(matrix.input_dim),
        int(matrix.output_dim),
    )
    if int(ok) != 1:
        return matrix

    out = QuantizedMatrix(
        name=f"{matrix.name}::q6k_r4",
        qtype=QTYPE_Q6_K_R4,
        shape=matrix.shape,
        data=np.ascontiguousarray(dst, dtype=np.uint8),
        interleave_factor=1,
    )
    out._validated = True
    matrix._repacked_q6k_r4 = out
    return out


def repack_q6k_lm(matrix: QuantizedMatrix) -> QuantizedMatrix:
    """Repack canonical Q6_K rows into an expanded LM-head decode layout."""
    if matrix.qtype == QTYPE_Q6_K_LM:
        return matrix
    if matrix.qtype != QTYPE_Q6_K:
        return matrix
    if not has_q6k_lm_support():
        return matrix

    packed = matrix.ensure_packed()
    if not matrix._validated:
        return matrix
    if _requires_output_deinterleave(matrix):
        return matrix

    cached = getattr(matrix, "_repacked_q6k_lm", None)
    if isinstance(cached, QuantizedMatrix):
        return cached

    blocks_per_row = int(matrix.input_dim) // 256
    dst = np.empty((int(matrix.output_dim), blocks_per_row * 276), dtype=np.uint8)
    ok = _lib().simd_repack_q6k_lm(
        ctypes.c_void_p(packed.ctypes.data),
        ctypes.c_void_p(dst.ctypes.data),
        int(matrix.input_dim),
        int(matrix.output_dim),
    )
    if int(ok) != 1:
        return matrix

    out = QuantizedMatrix(
        name=f"{matrix.name}::q6k_lm",
        qtype=QTYPE_Q6_K_LM,
        shape=matrix.shape,
        data=np.ascontiguousarray(dst, dtype=np.uint8),
        interleave_factor=1,
    )
    out._validated = True
    matrix._repacked_q6k_lm = out
    return out


# =========================================================================
# Fast path helpers
# =========================================================================


def _ensure_f32_vec(x: np.ndarray, target_dim: int) -> np.ndarray:
    """Convert x to a contiguous float32 vector of exact length target_dim."""
    if (
        x.dtype == np.float32
        and x.ndim == 1
        and x.shape[0] == target_dim
        and x.flags["C_CONTIGUOUS"]
    ):
        return x
    vec = np.asarray(x, dtype=np.float32).reshape(-1)
    if vec.shape[0] == target_dim:
        return np.ascontiguousarray(vec)
    if vec.shape[0] > target_dim:
        return np.ascontiguousarray(vec[:target_dim])
    out = np.zeros((target_dim,), dtype=np.float32)
    out[: vec.shape[0]] = vec
    return out


# =========================================================================
# Core compute functions
# =========================================================================


def matvec(x: np.ndarray, matrix: QuantizedMatrix) -> np.ndarray:
    """Compute y = x @ W where W is GGUF quantized and shaped [in, out]."""
    # Ensure lib is loaded and function tables populated
    _lib()

    x_f = _ensure_f32_vec(x, matrix.input_dim)
    packed = matrix.ensure_packed()

    if not matrix._validated:
        raise ValueError(
            f"Invalid quantized layout for '{matrix.name}': "
            f"qtype={matrix.qtype} shape={packed.shape}"
        )

    fn = _MATVEC_FN.get(matrix.qtype)
    if fn is None:
        raise ValueError(
            f"Unsupported quantization type for '{matrix.name}': qtype={matrix.qtype}"
        )

    out = np.empty((matrix.output_dim,), dtype=np.float32)
    fn(
        x_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_void_p(matrix._data_ptr),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        matrix.input_dim,
        matrix.output_dim,
    )
    if matrix._inverse_row_permutation is not None:
        out = out[matrix._inverse_row_permutation]
    return out


def matmul(x: np.ndarray, matrix: QuantizedMatrix) -> np.ndarray:
    """Compute Y = X @ W where X is [batch, in] and W is GGUF quantized [in, out]."""
    x_f = np.asarray(x, dtype=np.float32)
    if x_f.ndim == 1:
        return matvec(x_f, matrix)
    if x_f.ndim != 2:
        raise ValueError(f"matmul expects 2D input, got shape={x_f.shape}")

    if x_f.shape[1] != matrix.input_dim:
        if x_f.shape[1] > matrix.input_dim:
            x_f = x_f[:, : matrix.input_dim]
        else:
            padded = np.zeros((x_f.shape[0], matrix.input_dim), dtype=np.float32)
            padded[:, : x_f.shape[1]] = x_f
            x_f = padded

    if not x_f.flags["C_CONTIGUOUS"]:
        x_f = np.ascontiguousarray(x_f)
    if x_f.shape[0] == 1:
        return matvec(x_f[0], matrix).reshape(1, matrix.output_dim)

    # Ensure lib and cached packed data
    _lib()
    packed = matrix.ensure_packed()

    if not matrix._validated:
        raise ValueError(
            f"Invalid quantized layout for '{matrix.name}': "
            f"qtype={matrix.qtype} shape={packed.shape}"
        )

    fn = _MATMUL_FN.get(matrix.qtype)
    if fn is None:
        raise ValueError(
            f"Unsupported quantization type for '{matrix.name}': qtype={matrix.qtype}"
        )

    out = np.empty((x_f.shape[0], matrix.output_dim), dtype=np.float32)
    fn(
        x_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_void_p(matrix._data_ptr),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        x_f.shape[0],
        matrix.input_dim,
        matrix.output_dim,
    )
    if matrix._inverse_row_permutation is not None:
        out = out[:, matrix._inverse_row_permutation]
    return out


def dequantize_rows(matrix: QuantizedMatrix, row_ids: Iterable[int]) -> np.ndarray:
    """Dequantize selected rows (typically embedding lookups)."""
    idx = np.asarray(list(row_ids), dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return np.zeros((0, matrix.input_dim), dtype=np.float32)
    clipped = np.clip(idx, 0, matrix.output_dim - 1)
    if matrix._inverse_row_permutation is not None:
        clipped = matrix._inverse_row_permutation[clipped]
    packed = matrix.ensure_packed()
    rows = packed[clipped]
    if not rows.flags["C_CONTIGUOUS"]:
        rows = np.ascontiguousarray(rows)
    return _dequantize_to_f32(matrix.qtype, rows)


def _dequantize_to_f32(qtype: int, packed: np.ndarray) -> np.ndarray:
    try:
        enum_type = GGMLQuantizationType(qtype)
    except ValueError:
        raise ValueError(f"Unsupported GGUF quantization type: {qtype}") from None
    return np.asarray(dequantize(packed, enum_type), dtype=np.float32)


def dequantize_full(matrix: QuantizedMatrix) -> np.ndarray:
    """Dequantize the entire matrix to dense float32 [output_dim, input_dim]."""
    packed = matrix.ensure_packed()
    if not packed.flags["C_CONTIGUOUS"]:
        packed = np.ascontiguousarray(packed)
    dense = _dequantize_to_f32(matrix.qtype, packed)
    if dense.ndim == 1:
        dense = dense.reshape(matrix.output_dim, matrix.input_dim)
    elif dense.shape != (matrix.output_dim, matrix.input_dim):
        total = matrix.output_dim * matrix.input_dim
        if dense.size >= total:
            dense = dense.flatten()[:total].reshape(matrix.output_dim, matrix.input_dim)
    return np.ascontiguousarray(dense, dtype=np.float32)


def fused_expert_swiglu(
    x: np.ndarray,
    gate_matrix: QuantizedMatrix,
    up_matrix: QuantizedMatrix,
    down_matrix: QuantizedMatrix,
) -> np.ndarray:
    """Fused expert SwiGLU FFN: gate->up->SiLU(gate)*up->down in one C call.

    Eliminates 5 Python/ctypes round-trips compared to 3 separate matvec + swiglu calls.
    """

    def _python_fallback(x_in: np.ndarray) -> np.ndarray:
        from core.native import simd_ops_wrapper as simd_ops

        gate = matvec(x_in, gate_matrix)
        up = matvec(x_in, up_matrix)
        hidden = simd_ops.swiglu(gate, up)
        return matvec(hidden, down_matrix)

    strict_native = _strict_native_qsg_enabled()
    lib = _lib()
    fn = getattr(lib, "simd_fused_expert_swiglu", None)
    if fn is None:
        if strict_native:
            raise RuntimeError(
                "Strict native QSG requires simd_fused_expert_swiglu; Python fallback is disabled."
            )
        return _python_fallback(x)

    x_f = _ensure_f32_vec(x, gate_matrix.input_dim)
    gate_packed = gate_matrix.ensure_packed()
    up_packed = up_matrix.ensure_packed()
    down_packed = down_matrix.ensure_packed()

    if not (gate_matrix._validated and up_matrix._validated and down_matrix._validated):
        if strict_native:
            raise RuntimeError(
                "Strict native QSG requires validated fused expert matrices; Python fallback is disabled."
            )
        return _python_fallback(x_f)

    # Fused expert kernels assume canonical row order for the gate/up hidden channels.
    # If those matrices were row-interleaved, use the mathematically safe path.
    if _requires_output_deinterleave(gate_matrix) or _requires_output_deinterleave(
        up_matrix
    ):
        if strict_native:
            raise RuntimeError(
                "Strict native QSG requires fused expert kernels without row-deinterleave fallback."
            )
        return _python_fallback(x_f)

    out = np.empty((down_matrix.output_dim,), dtype=np.float32)
    fn(
        x_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gate_matrix.input_dim,
        ctypes.c_void_p(gate_matrix._data_ptr),
        gate_matrix.qtype,
        gate_matrix.output_dim,
        ctypes.c_void_p(up_matrix._data_ptr),
        up_matrix.qtype,
        ctypes.c_void_p(down_matrix._data_ptr),
        down_matrix.qtype,
        down_matrix.output_dim,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if _requires_output_deinterleave(down_matrix):
        out = out[down_matrix._inverse_row_permutation]
    return out


def fused_moe_ffn(
    x: np.ndarray,
    expert_weights: Sequence[float],
    gate_matrices: Sequence[QuantizedMatrix],
    up_matrices: Sequence[QuantizedMatrix],
    down_matrices: Sequence[QuantizedMatrix],
) -> np.ndarray:
    """Fused top-k MoE FFN for quantized experts in a single C call."""
    k = len(expert_weights)
    if (
        k == 0
        or len(gate_matrices) != k
        or len(up_matrices) != k
        or len(down_matrices) != k
    ):
        raise ValueError("fused_moe_ffn expects non-empty, equal-length expert lists")

    if not all(
        isinstance(m, QuantizedMatrix)
        for m in (*gate_matrices, *up_matrices, *down_matrices)
    ):
        raise ValueError("fused_moe_ffn only supports QuantizedMatrix expert weights")

    gate_qtype = gate_matrices[0].qtype
    up_qtype = up_matrices[0].qtype
    down_qtype = down_matrices[0].qtype
    if (
        gate_qtype not in {QTYPE_Q8_0, QTYPE_Q4_K, QTYPE_Q4_K_R4, QTYPE_Q6_K}
        or up_qtype not in {QTYPE_Q8_0, QTYPE_Q4_K, QTYPE_Q4_K_R4, QTYPE_Q6_K}
        or down_qtype not in {QTYPE_Q8_0, QTYPE_Q4_K, QTYPE_Q4_K_R4, QTYPE_Q6_K}
    ):
        raise ValueError("fused_moe_ffn only supports Q8_0/Q4_K/Q4_K_R4/Q6_K experts")

    in_dim = gate_matrices[0].input_dim
    hidden_dim = gate_matrices[0].output_dim
    out_dim = down_matrices[0].output_dim

    for idx in range(k):
        g = gate_matrices[idx]
        u = up_matrices[idx]
        d = down_matrices[idx]
        if g.qtype != gate_qtype or u.qtype != up_qtype or d.qtype != down_qtype:
            raise ValueError("fused_moe_ffn requires uniform qtype per projection kind")
        if g.input_dim != in_dim or u.input_dim != in_dim:
            raise ValueError(
                "fused_moe_ffn requires consistent input dims across experts"
            )
        if g.output_dim != hidden_dim or u.output_dim != hidden_dim:
            raise ValueError(
                "fused_moe_ffn requires consistent hidden dims across experts"
            )
        if d.input_dim != hidden_dim or d.output_dim != out_dim:
            raise ValueError(
                "fused_moe_ffn requires down projection dims matching hidden/out dims"
            )

    x_f = _ensure_f32_vec(x, in_dim)
    weights_f = np.ascontiguousarray(
        np.asarray(expert_weights, dtype=np.float32).reshape(-1)
    )
    if weights_f.shape[0] != k:
        raise ValueError("fused_moe_ffn expert weight count mismatch")

    for matrix in (*gate_matrices, *up_matrices, *down_matrices):
        matrix.ensure_packed()
        if not matrix._validated:
            raise ValueError(f"Invalid quantized layout for '{matrix.name}'")

    lib = _lib()
    fn = getattr(lib, "simd_fused_moe_ffn", None)
    needs_python_fallback = any(
        _requires_output_deinterleave(m)
        for m in (*gate_matrices, *up_matrices, *down_matrices)
    )
    strict_native = _strict_native_qsg_enabled()
    if fn is None or needs_python_fallback:
        if strict_native:
            if fn is None:
                raise RuntimeError(
                    "Strict native QSG requires simd_fused_moe_ffn; Python fallback is disabled."
                )
            raise RuntimeError(
                "Strict native QSG requires fused MoE kernels without row-deinterleave fallback."
            )
        out = np.zeros((out_dim,), dtype=np.float32)
        for idx, weight in enumerate(weights_f.tolist()):
            if weight <= 0.0:
                continue
            out += float(weight) * fused_expert_swiglu(
                x_f,
                gate_matrices[idx],
                up_matrices[idx],
                down_matrices[idx],
            )
        return out

    expert_indices = np.arange(k, dtype=np.int32)
    gate_ptrs = (ctypes.c_void_p * k)(
        *[ctypes.c_void_p(m._data_ptr) for m in gate_matrices]
    )
    up_ptrs = (ctypes.c_void_p * k)(
        *[ctypes.c_void_p(m._data_ptr) for m in up_matrices]
    )
    down_ptrs = (ctypes.c_void_p * k)(
        *[ctypes.c_void_p(m._data_ptr) for m in down_matrices]
    )
    out = np.empty((out_dim,), dtype=np.float32)
    fn(
        x_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        in_dim,
        expert_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        weights_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        k,
        gate_ptrs,
        gate_qtype,
        hidden_dim,
        up_ptrs,
        up_qtype,
        down_ptrs,
        down_qtype,
        out_dim,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return out
