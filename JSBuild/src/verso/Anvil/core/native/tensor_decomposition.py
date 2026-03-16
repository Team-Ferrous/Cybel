"""Tensor decomposition helpers for MPO compression and inference."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from core.native.native_ops import load_native_library


@dataclass(frozen=True)
class MPOFactors:
    """Matrix Product Operator factors for a 2D matrix."""

    cores: list[np.ndarray]
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    ranks: tuple[int, ...]

    @property
    def input_dim(self) -> int:
        return int(np.prod(self.input_shape))

    @property
    def output_dim(self) -> int:
        return int(np.prod(self.output_shape))


@dataclass(frozen=True)
class SVDCompressedLayer:
    """Low-rank layer representation used by TensorNetworkCompressor API."""

    u: np.ndarray
    s: np.ndarray
    vt: np.ndarray


def mpo_compress(
    matrix: np.ndarray,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    max_rank: int = 64,
) -> MPOFactors:
    """Compress a dense matrix into MPO/TT format using TT-SVD."""
    w = np.asarray(matrix, dtype=np.float32)
    in_shape = tuple(int(v) for v in input_shape)
    out_shape = tuple(int(v) for v in output_shape)
    if len(in_shape) != len(out_shape):
        raise ValueError("input_shape and output_shape must have the same rank")
    if any(v <= 0 for v in in_shape + out_shape):
        raise ValueError("input_shape/output_shape values must be positive")
    if w.ndim != 2:
        raise ValueError("matrix must be 2D")
    if w.shape != (int(np.prod(out_shape)), int(np.prod(in_shape))):
        raise ValueError(
            f"matrix shape mismatch: expected {(int(np.prod(out_shape)), int(np.prod(in_shape)))}, got {w.shape}"
        )

    n = len(in_shape)
    max_rank = int(max(1, max_rank))
    tensor = w.reshape(*out_shape, *in_shape)
    perm = []
    for i in range(n):
        perm.extend([i, n + i])  # [o1, i1, o2, i2, ...]
    tensor = np.transpose(tensor, axes=perm)

    cores: list[np.ndarray] = []
    ranks = [1]
    work = np.asarray(tensor, dtype=np.float32)
    r_prev = 1

    for idx in range(n - 1):
        d = out_shape[idx] * in_shape[idx]
        work = work.reshape(r_prev * d, -1)
        u, s, vt = np.linalg.svd(work, full_matrices=False)
        r = min(max_rank, int(s.shape[0]))
        u = u[:, :r]
        s = s[:r]
        vt = vt[:r, :]
        core = u.reshape(r_prev, out_shape[idx], in_shape[idx], r).astype(np.float32)
        cores.append(core)
        work = (s[:, None] * vt).astype(np.float32)
        r_prev = r
        ranks.append(r_prev)

    core_last = work.reshape(r_prev, out_shape[-1], in_shape[-1], 1).astype(np.float32)
    cores.append(core_last)
    ranks.append(1)
    return MPOFactors(
        cores=cores,
        input_shape=in_shape,
        output_shape=out_shape,
        ranks=tuple(int(r) for r in ranks),
    )


def mpo_to_matrix(factors: MPOFactors) -> np.ndarray:
    """Reconstruct dense matrix from MPO factors."""
    if not factors.cores:
        raise ValueError("MPOFactors.cores cannot be empty")

    op = np.asarray(factors.cores[0], dtype=np.float32)
    for core in factors.cores[1:]:
        op = np.tensordot(op, np.asarray(core, dtype=np.float32), axes=([-1], [0]))
    if op.shape[0] != 1 or op.shape[-1] != 1:
        raise ValueError("Invalid MPO ranks: expected boundary ranks to be 1")
    op = np.squeeze(op, axis=(0, -1))

    n = len(factors.output_shape)
    if op.ndim != 2 * n:
        raise ValueError("Invalid MPO tensor rank after contraction")
    out_axes = list(range(0, 2 * n, 2))
    in_axes = list(range(1, 2 * n, 2))
    op = np.transpose(op, axes=out_axes + in_axes)
    return op.reshape(factors.output_dim, factors.input_dim).astype(np.float32)


def mpo_matvec(factors: MPOFactors, x: np.ndarray) -> np.ndarray:
    """Apply MPO matrix to a vector, using optional native kernel when available."""
    x_f = np.asarray(x, dtype=np.float32).reshape(-1)
    if x_f.shape[0] != factors.input_dim:
        raise ValueError(f"Expected x dim {factors.input_dim}, got {x_f.shape[0]}")

    native = _native_mpo_matvec(factors=factors, x=x_f)
    if native is not None:
        return native
    return mpo_to_matrix(factors) @ x_f


def _native_mpo_matvec(factors: MPOFactors, x: np.ndarray) -> Optional[np.ndarray]:
    """Best-effort native MPO matvec call if symbol is present.

    The native kernel currently supports a Kronecker-style factor chain:
    y = (A_n ⊗ ... ⊗ A_1) x where each A_i is 2D [rows_i, cols_i].
    Our general MPO factors are TT cores [r_l, out_i, in_i, r_r]. We only use the
    native path when all ranks are 1 so each core collapses to a 2D factor.
    """
    try:
        lib = load_native_library()
    except Exception:
        return None
    fn = getattr(lib, "tensor_mpo_matvec_f32", None)
    if fn is None:
        return None

    # Only rank-1 MPOs can be lowered to the current native ABI.
    if any(int(c.shape[0]) != 1 or int(c.shape[3]) != 1 for c in factors.cores):
        return None

    # ABI:
    # tensor_mpo_matvec_f32(
    #   const float* x,
    #   const float* const* factors,
    #   const int* rows,
    #   const int* cols,
    #   int num_factors,
    #   float* y
    # ) -> int
    try:
        float_p = ctypes.POINTER(ctypes.c_float)
        float_pp = ctypes.POINTER(float_p)
        int_p = ctypes.POINTER(ctypes.c_int)
        fn.argtypes = [float_p, float_pp, int_p, int_p, ctypes.c_int, float_p]
        fn.restype = ctypes.c_int
    except Exception:
        return None

    factor_mats: list[np.ndarray] = []
    rows: list[int] = []
    cols: list[int] = []
    for core in factors.cores:
        c = np.asarray(core, dtype=np.float32)
        if c.ndim != 4:
            return None
        # ranks are guaranteed 1 above
        mat = np.ascontiguousarray(c[0, :, :, 0], dtype=np.float32)
        factor_mats.append(mat)
        rows.append(int(mat.shape[0]))
        cols.append(int(mat.shape[1]))

    if not factor_mats:
        return None

    factor_ptrs = (float_p * len(factor_mats))(
        *[mat.ctypes.data_as(float_p) for mat in factor_mats]
    )
    rows_arr = np.ascontiguousarray(np.asarray(rows, dtype=np.int32))
    cols_arr = np.ascontiguousarray(np.asarray(cols, dtype=np.int32))
    x_f = np.ascontiguousarray(x, dtype=np.float32)
    out = np.empty((factors.output_dim,), dtype=np.float32)

    ok = fn(
        x_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(factor_ptrs, float_pp),
        rows_arr.ctypes.data_as(int_p),
        cols_arr.ctypes.data_as(int_p),
        len(factor_mats),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if int(ok) != 1:
        return None
    return out


class TensorNetworkCompressor:
    """Convenience API used by higher-level code/tests."""

    def compress_layer(self, weight: np.ndarray, bond_dim: int = 64) -> SVDCompressedLayer:
        weight_f = np.asarray(weight, dtype=np.float32)
        if weight_f.ndim != 2:
            raise ValueError("weight must be 2D")
        u, s, vt = np.linalg.svd(weight_f, full_matrices=False)
        rank = int(max(1, min(int(bond_dim), s.shape[0])))
        return SVDCompressedLayer(
            u=np.asarray(u[:, :rank], dtype=np.float32),
            s=np.asarray(s[:rank], dtype=np.float32),
            vt=np.asarray(vt[:rank, :], dtype=np.float32),
        )

    def reconstruct_layer(self, factors: SVDCompressedLayer | MPOFactors) -> np.ndarray:
        if isinstance(factors, MPOFactors):
            return mpo_to_matrix(factors)
        return (factors.u * factors.s[np.newaxis, :]) @ factors.vt

    def matvec_mpo(
        self,
        x: np.ndarray,
        factors: SVDCompressedLayer | MPOFactors,
    ) -> np.ndarray:
        if isinstance(factors, MPOFactors):
            return mpo_matvec(factors, x)
        vec = np.asarray(x, dtype=np.float32).reshape(-1)
        proj = factors.vt @ vec
        return (factors.u * factors.s[np.newaxis, :]) @ proj

    @staticmethod
    def _factor_dim(dim: int, parts: int = 4) -> tuple[int, ...]:
        dim = int(dim)
        if dim <= 0:
            raise ValueError("dimension must be positive")
        parts = int(max(1, parts))
        vals = [1] * parts
        n = dim
        p = 2
        while p * p <= n:
            while n % p == 0:
                idx = int(np.argmin(vals))
                vals[idx] *= p
                n //= p
            p += 1
        if n > 1:
            idx = int(np.argmin(vals))
            vals[idx] *= n
        return tuple(int(v) for v in vals)
