"""ctypes bindings for the native SIMD helper ops.

Optimized for minimal Python overhead — library handle and function
references are cached on first call, numpy array conversions are
short-circuited when inputs are already float32 + contiguous.
"""

from __future__ import annotations

import ctypes
from array import array
import json
from typing import Optional

import numpy as np

from core.native.native_ops import load_native_library


_LIB: Optional[ctypes.CDLL] = None
_FLOAT_P = ctypes.POINTER(ctypes.c_float)
_INT_P = ctypes.POINTER(ctypes.c_int)
_INT = ctypes.c_int


def _as_f32(value: np.ndarray) -> np.ndarray:
    if value.dtype == np.float32 and value.flags["C_CONTIGUOUS"]:
        return value
    return np.ascontiguousarray(value, dtype=np.float32)


def _coerce_1d_float_buffer(
    value,
) -> tuple[object, int, tuple[int, ...] | None]:
    if isinstance(value, np.ndarray):
        flat = _as_f32(value).reshape(-1)
        length = int(flat.size)
        return flat, length, tuple(int(dim) for dim in value.shape)

    if isinstance(value, array):
        if value.typecode != "f":
            value = array("f", (float(item) for item in value))
        length = len(value)
        address, _ = value.buffer_info()
        return (ctypes.c_float * length).from_address(address), length, None

    if isinstance(value, list):
        values = [float(item) for item in value]
        length = len(values)
        return (ctypes.c_float * length)(*values), length, None

    if isinstance(value, tuple):
        values = [float(item) for item in value]
        length = len(values)
        return (ctypes.c_float * length)(*values), length, None

    if isinstance(value, ctypes.Array):
        length = len(value)
        return value, length, None

    values = [float(item) for item in value]
    length = len(values)
    return (ctypes.c_float * length)(*values), length, None


def _coerce_1d_int_buffer(value) -> tuple[object, int]:
    if isinstance(value, array):
        if value.typecode != "i":
            value = array("i", (int(item) for item in value))
        length = len(value)
        address, _ = value.buffer_info()
        return (ctypes.c_int * length).from_address(address), length

    if isinstance(value, list):
        values = [int(item) for item in value]
        length = len(values)
        return (ctypes.c_int * length)(*values), length

    if isinstance(value, tuple):
        values = [int(item) for item in value]
        length = len(values)
        return (ctypes.c_int * length)(*values), length

    if isinstance(value, ctypes.Array):
        length = len(value)
        return value, length

    values = [int(item) for item in value]
    length = len(values)
    return (ctypes.c_int * length)(*values), length


def _float_ptr(buffer) -> ctypes.POINTER(ctypes.c_float):
    if isinstance(buffer, np.ndarray):
        return buffer.ctypes.data_as(_FLOAT_P)
    return ctypes.cast(buffer, _FLOAT_P)


def _int_ptr(buffer) -> ctypes.POINTER(ctypes.c_int):
    return ctypes.cast(buffer, _INT_P)


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    lib = load_native_library()

    # --- Float32 ops ---
    lib.simd_matmul_f32.argtypes = [_FLOAT_P, _FLOAT_P, _FLOAT_P, _INT, _INT, _INT]
    lib.simd_matvec_f32.argtypes = [_FLOAT_P, _FLOAT_P, _FLOAT_P, _INT, _INT]
    lib.simd_rmsnorm_f32.argtypes = [_FLOAT_P, _FLOAT_P, _INT, ctypes.c_float]
    lib.simd_swiglu_f32.argtypes = [_FLOAT_P, _FLOAT_P, _FLOAT_P, _INT]
    lib.simd_softmax_f32.argtypes = [_FLOAT_P, _INT]
    try:
        lib.simd_sanitize_logits_f32.argtypes = [_FLOAT_P, _INT]
        lib.simd_argmax_f32.argtypes = [_FLOAT_P, _INT]
        lib.simd_argmax_f32.restype = _INT
        lib.simd_score_token_f32.argtypes = [
            _FLOAT_P,
            _INT,
            ctypes.c_float,
            _INT,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.simd_score_token_f32.restype = _INT
        lib.simd_sample_token_f32.argtypes = [
            _FLOAT_P,
            _INT,
            ctypes.c_float,
            _INT,
            ctypes.c_float,
            _INT,
            ctypes.c_float,
        ]
        lib.simd_sample_token_f32.restype = _INT
        lib.simd_seed_rng_f32.argtypes = [_INT]
        lib.simd_seed_rng_f32.restype = None
        lib.simd_suppress_tokens_f32.argtypes = [_FLOAT_P, _INT, _INT_P, _INT]
        lib.simd_suppress_tokens_f32.restype = None
        lib.simd_apply_token_penalties_f32.argtypes = [
            _FLOAT_P,
            _INT,
            _INT_P,
            _INT,
            ctypes.c_float,
            ctypes.c_float,
        ]
        lib.simd_apply_token_penalties_f32.restype = None
        lib.simd_postprocess_sample_f32.argtypes = [
            _FLOAT_P,
            _INT,
            _INT_P,
            _INT,
            _INT_P,
            _INT,
            ctypes.c_float,
            ctypes.c_float,
            _INT,
            ctypes.c_float,
            _INT,
            ctypes.c_float,
            _INT,
            ctypes.c_float,
        ]
        lib.simd_postprocess_sample_f32.restype = _INT
        lib.simd_postprocess_score_token_f32.argtypes = [
            _FLOAT_P,
            _INT,
            _INT_P,
            _INT,
            _INT_P,
            _INT,
            ctypes.c_float,
            ctypes.c_float,
            _INT,
            ctypes.c_float,
            _INT,
            ctypes.c_float,
            _INT,
            ctypes.c_float,
            _INT,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.simd_postprocess_score_token_f32.restype = _INT
        lib.simd_qsg_postprocess_sample_f32.argtypes = [
            _FLOAT_P,
            _INT,
            _INT_P,
            _INT,
            _INT_P,
            _INT,
            _INT_P,
            _INT,
            _INT,
            _INT,
            ctypes.c_float,
            _INT,
            _INT,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            _INT,
            ctypes.c_float,
            _INT,
            ctypes.c_float,
            _INT,
            ctypes.c_float,
        ]
        lib.simd_qsg_postprocess_sample_f32.restype = _INT
        lib.simd_qsg_postprocess_score_token_f32.argtypes = [
            _FLOAT_P,
            _INT,
            _INT_P,
            _INT,
            _INT_P,
            _INT,
            _INT,
            _INT,
            ctypes.c_float,
            _INT,
            _INT,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            _INT,
            ctypes.c_float,
            _INT,
            ctypes.c_float,
            _INT,
            ctypes.c_float,
            _INT,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.simd_qsg_postprocess_score_token_f32.restype = _INT
    except AttributeError:
        pass
    lib.simd_rope_f32.argtypes = [_FLOAT_P, _FLOAT_P, _INT, _INT, _INT, _INT, ctypes.c_float]
    try:
        lib.batch_rope_f32.argtypes = [_FLOAT_P, _FLOAT_P, _INT, _INT, _INT, _INT, ctypes.c_float]
    except AttributeError:
        pass
    try:
        lib.simd_fast_exp_f32.argtypes = [_FLOAT_P, _INT]
    except AttributeError:
        pass
    lib.simd_ssm_step_f32.argtypes = [_FLOAT_P, _FLOAT_P, _FLOAT_P, _INT]
    lib.simd_ssm_parallel_scan_f32.argtypes = [_FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P, _INT, _INT]

    # --- Quantized matmul ops ---
    try:
        lib.simd_matvec_q8_0.argtypes = [_FLOAT_P, ctypes.c_void_p, _FLOAT_P, _INT, _INT]
        lib.simd_matvec_q4k.argtypes = [_FLOAT_P, ctypes.c_void_p, _FLOAT_P, _INT, _INT]
    except AttributeError:
        pass

    try:
        lib.anvil_set_num_threads.argtypes = [_INT]
        lib.anvil_set_num_threads.restype = None
        lib.anvil_get_num_threads.argtypes = []
        lib.anvil_get_num_threads.restype = _INT
        lib.anvil_get_num_threads_for_path.argtypes = [_INT]
        lib.anvil_get_num_threads_for_path.restype = _INT
        lib.anvil_set_thread_mode.argtypes = [_INT]
        lib.anvil_set_thread_mode.restype = None
        lib.anvil_get_thread_mode.argtypes = []
        lib.anvil_get_thread_mode.restype = _INT
        lib.anvil_get_num_procs.argtypes = []
        lib.anvil_get_num_procs.restype = _INT
        lib.anvil_detect_physical_cores.argtypes = []
        lib.anvil_detect_physical_cores.restype = _INT
        lib.anvil_get_p_core_count.argtypes = []
        lib.anvil_get_p_core_count.restype = _INT
        lib.anvil_get_omp_max_threads.argtypes = []
        lib.anvil_get_omp_max_threads.restype = _INT
        lib.anvil_get_omp_dynamic.argtypes = []
        lib.anvil_get_omp_dynamic.restype = _INT
        lib.anvil_get_omp_active_levels.argtypes = []
        lib.anvil_get_omp_active_levels.restype = _INT
        lib.anvil_set_thread_affinity.argtypes = [_INT]
        lib.anvil_set_thread_affinity.restype = _INT
        lib.anvil_get_affinity_mode.argtypes = []
        lib.anvil_get_affinity_mode.restype = _INT
        lib.anvil_configure_affinity_mode.argtypes = [_INT]
        lib.anvil_configure_affinity_mode.restype = _INT
        lib.anvil_get_l3_domain_count.argtypes = []
        lib.anvil_get_l3_domain_count.restype = _INT
        lib.anvil_bind_worker_thread.argtypes = [_INT, _INT]
        lib.anvil_bind_worker_thread.restype = _INT
        lib.anvil_sample_thread_cpu.argtypes = []
        lib.anvil_sample_thread_cpu.restype = _INT
        lib.anvil_get_thread_migration_count.argtypes = []
        lib.anvil_get_thread_migration_count.restype = _INT
        lib.anvil_get_last_cpu.argtypes = []
        lib.anvil_get_last_cpu.restype = _INT
        lib.anvil_refresh_topology.argtypes = []
        lib.anvil_refresh_topology.restype = _INT
        lib.anvil_topology_export_json.argtypes = [ctypes.c_char_p, _INT]
        lib.anvil_topology_export_json.restype = _INT
        lib.anvil_affinity_plan_export_json.argtypes = [ctypes.c_char_p, _INT]
        lib.anvil_affinity_plan_export_json.restype = _INT
        lib.anvil_openmp_enabled.argtypes = []
        lib.anvil_openmp_enabled.restype = _INT
        lib.anvil_compiled_with_avx2.argtypes = []
        lib.anvil_compiled_with_avx2.restype = _INT
        lib.anvil_compiled_with_avx512.argtypes = []
        lib.anvil_compiled_with_avx512.restype = _INT
        if hasattr(lib, "anvil_compiled_with_amx"):
            lib.anvil_compiled_with_amx.argtypes = []
            lib.anvil_compiled_with_amx.restype = _INT
        if hasattr(lib, "anvil_runtime_amx_available"):
            lib.anvil_runtime_amx_available.argtypes = []
            lib.anvil_runtime_amx_available.restype = _INT
        lib.anvil_reset_qsg_sampling_stats.argtypes = []
        lib.anvil_reset_qsg_sampling_stats.restype = None
        lib.anvil_qsg_sampling_stats_json.argtypes = [ctypes.c_char_p, _INT]
        lib.anvil_qsg_sampling_stats_json.restype = _INT
    except AttributeError:
        pass
    try:
        lib.simd_pin_to_p_cores.argtypes = [_INT]
        lib.simd_pin_to_p_cores.restype = _INT
        lib.simd_get_p_core_count.argtypes = []
        lib.simd_get_p_core_count.restype = _INT
    except AttributeError:
        pass

    required_symbols = (
        "simd_sanitize_logits_f32",
        "simd_argmax_f32",
        "simd_score_token_f32",
        "simd_sample_token_f32",
        "simd_suppress_tokens_f32",
        "simd_apply_token_penalties_f32",
        "simd_postprocess_sample_f32",
        "simd_qsg_postprocess_sample_f32",
        "simd_postprocess_score_token_f32",
        "simd_qsg_postprocess_score_token_f32",
        "simd_fast_exp_f32",
    )
    missing = [name for name in required_symbols if not hasattr(lib, name)]
    if missing:
        raise RuntimeError(
            "Native-only QSG requires SIMD symbols: "
            + ", ".join(missing)
        )

    _LIB = lib
    return lib


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_f = _as_f32(a)
    b_f = _as_f32(b)
    if a_f.ndim != 2 or b_f.ndim != 2:
        raise ValueError("matmul expects 2D arrays")
    if a_f.shape[1] != b_f.shape[0]:
        raise ValueError(f"matmul shape mismatch: {a_f.shape} x {b_f.shape}")
    out = np.empty((a_f.shape[0], b_f.shape[1]), dtype=np.float32)
    _lib().simd_matmul_f32(
        a_f.ctypes.data_as(_FLOAT_P),
        b_f.ctypes.data_as(_FLOAT_P),
        out.ctypes.data_as(_FLOAT_P),
        a_f.shape[0],
        a_f.shape[1],
        b_f.shape[1],
    )
    return out


def matvec(x: np.ndarray, a: np.ndarray) -> np.ndarray:
    x_f = _as_f32(x).reshape(-1)
    a_f = _as_f32(a)
    if a_f.ndim != 2 or a_f.shape[0] != x_f.shape[0]:
        raise ValueError(f"matvec shape mismatch: {x_f.shape} vs {a_f.shape}")
    out = np.empty((a_f.shape[1],), dtype=np.float32)
    _lib().simd_matvec_f32(
        x_f.ctypes.data_as(_FLOAT_P),
        a_f.ctypes.data_as(_FLOAT_P),
        out.ctypes.data_as(_FLOAT_P),
        x_f.shape[0],
        a_f.shape[1],
    )
    return out


def rmsnorm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x_f = _as_f32(x)
    gamma_f = _as_f32(gamma).reshape(-1)
    if x_f.ndim == 1:
        # In-place on a copy
        out = x_f.copy()
        _lib().simd_rmsnorm_f32(
            out.ctypes.data_as(_FLOAT_P),
            gamma_f.ctypes.data_as(_FLOAT_P),
            out.shape[0],
            eps,
        )
        return out
    out = np.empty_like(x_f)
    lib = _lib()
    for idx in range(x_f.shape[0]):
        row = x_f[idx].copy()
        lib.simd_rmsnorm_f32(
            row.ctypes.data_as(_FLOAT_P),
            gamma_f.ctypes.data_as(_FLOAT_P),
            row.shape[0],
            eps,
        )
        out[idx] = row
    return out


def swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    gate_f = _as_f32(gate)
    up_f = _as_f32(up)
    if gate_f.shape != up_f.shape:
        raise ValueError(f"swiglu shape mismatch: {gate_f.shape} vs {up_f.shape}")
    out = np.empty_like(gate_f)
    lib = _lib()
    if gate_f.ndim == 1:
        lib.simd_swiglu_f32(
            gate_f.ctypes.data_as(_FLOAT_P),
            up_f.ctypes.data_as(_FLOAT_P),
            out.ctypes.data_as(_FLOAT_P),
            gate_f.shape[0],
        )
        return out
    for idx in range(gate_f.shape[0]):
        lib.simd_swiglu_f32(
            gate_f[idx].ctypes.data_as(_FLOAT_P),
            up_f[idx].ctypes.data_as(_FLOAT_P),
            out[idx].ctypes.data_as(_FLOAT_P),
            gate_f.shape[1],
        )
    return out


def softmax(scores: np.ndarray) -> np.ndarray:
    scores_f = _as_f32(scores).reshape(-1).copy()
    _lib().simd_softmax_f32(
        scores_f.ctypes.data_as(_FLOAT_P),
        scores_f.shape[0],
    )
    return scores_f.reshape(scores.shape)


def sanitize_logits(logits: np.ndarray):
    logits_f, length, original_shape = _coerce_1d_float_buffer(logits)
    lib = _lib()
    fn = getattr(lib, "simd_sanitize_logits_f32", None)
    if fn is None:
        raise RuntimeError(
            "Native-only QSG requires simd_sanitize_logits_f32; no NumPy fallback is allowed."
        )
    fn(_float_ptr(logits_f), length)
    if isinstance(logits, (array, np.ndarray, ctypes.Array)):
        return logits
    values = [float(logits_f[idx]) for idx in range(length)]
    if original_shape is not None and original_shape != (length,):
        return np.asarray(values, dtype=np.float32).reshape(original_shape)
    return values


def sanitize_logits_inplace(logits):
    logits_f, length, _ = _coerce_1d_float_buffer(logits)
    if length <= 0:
        return logits
    lib = _lib()
    fn = getattr(lib, "simd_sanitize_logits_f32", None)
    if fn is None:
        raise RuntimeError(
            "Native-only QSG requires simd_sanitize_logits_f32; no NumPy fallback is allowed."
        )
    fn(_float_ptr(logits_f), length)
    return logits


def argmax(values: np.ndarray) -> int:
    vals, length, _ = _coerce_1d_float_buffer(values)
    lib = _lib()
    fn = getattr(lib, "simd_argmax_f32", None)
    if fn is None:
        raise RuntimeError(
            "Native-only QSG requires simd_argmax_f32; no NumPy fallback is allowed."
    )
    return int(fn(_float_ptr(vals), length))


def score_token(
    logits,
    *,
    token_id: int,
    temperature: float,
) -> tuple[int, float]:
    logits_f, length, _ = _coerce_1d_float_buffer(logits)
    lib = _lib()
    fn = getattr(lib, "simd_score_token_f32", None)
    if fn is None:
        raise RuntimeError(
            "Native-only QSG requires simd_score_token_f32; no Python fallback is allowed."
        )
    greedy_token = ctypes.c_int(0)
    token_prob = ctypes.c_float(0.0)
    fn(
        _float_ptr(logits_f),
        length,
        float(temperature),
        int(token_id),
        ctypes.byref(greedy_token),
        ctypes.byref(token_prob),
    )
    return int(greedy_token.value), float(token_prob.value)


def postprocess_and_score(
    logits,
    *,
    token_id: int,
    suppressed_ids,
    token_history,
    presence_penalty: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    temperature: float,
    eos_token: int,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
) -> tuple[int, float]:
    logits_f, length, _ = _coerce_1d_float_buffer(logits)
    suppressed_buf, suppressed_len = _coerce_1d_int_buffer(suppressed_ids)
    history_buf, history_len = _coerce_1d_int_buffer(token_history)
    fn = getattr(_lib(), "simd_postprocess_score_token_f32", None)
    if fn is None:
        raise RuntimeError(
            "Native-only QSG requires simd_postprocess_score_token_f32; "
            "no Python fallback is allowed."
        )
    greedy_token = ctypes.c_int(0)
    token_prob = ctypes.c_float(0.0)
    fn(
        _float_ptr(logits_f),
        length,
        _int_ptr(suppressed_buf),
        suppressed_len,
        _int_ptr(history_buf),
        history_len,
        float(presence_penalty),
        float(repetition_penalty),
        int(no_repeat_ngram_size),
        float(temperature),
        int(eos_token),
        float(top_p),
        int(top_k),
        float(min_p),
        int(token_id),
        ctypes.byref(greedy_token),
        ctypes.byref(token_prob),
    )
    return int(greedy_token.value), float(token_prob.value)


def qsg_postprocess_and_score(
    logits,
    *,
    suppressed_ids,
    token_history,
    use_coconut: bool,
    coconut_paths: int,
    coconut_alpha: float,
    use_grover: bool,
    grover_top_k: int,
    grover_damping: float,
    presence_penalty: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    temperature: float,
    eos_token: int,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    token_id: int = -1,
) -> tuple[int, float]:
    logits_f, length, _ = _coerce_1d_float_buffer(logits)
    suppressed_buf, suppressed_len = _coerce_1d_int_buffer(suppressed_ids)
    history_buf, history_len = _coerce_1d_int_buffer(token_history)
    fn = getattr(_lib(), "simd_qsg_postprocess_score_token_f32", None)
    if fn is None:
        raise RuntimeError(
            "Native-only QSG requires simd_qsg_postprocess_score_token_f32; "
            "no Python fallback is allowed."
        )
    greedy_token = ctypes.c_int(0)
    token_prob = ctypes.c_float(0.0)
    fn(
        _float_ptr(logits_f),
        length,
        _int_ptr(suppressed_buf),
        suppressed_len,
        _int_ptr(history_buf),
        history_len,
        int(bool(use_coconut)),
        int(coconut_paths),
        float(coconut_alpha),
        int(bool(use_grover)),
        int(grover_top_k),
        float(grover_damping),
        float(presence_penalty),
        float(repetition_penalty),
        int(no_repeat_ngram_size),
        float(temperature),
        int(eos_token),
        float(top_p),
        int(top_k),
        float(min_p),
        int(token_id),
        ctypes.byref(greedy_token),
        ctypes.byref(token_prob),
    )
    return int(greedy_token.value), float(token_prob.value)


def sample_token(
    logits: np.ndarray,
    temperature: float,
    eos_token: int,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
) -> int:
    logits_f, length, _ = _coerce_1d_float_buffer(logits)
    lib = _lib()
    fn = getattr(lib, "simd_sample_token_f32", None)
    if fn is None:
        raise RuntimeError(
            "Native-only QSG requires simd_sample_token_f32; no Python fallback is allowed."
        )
    return int(
        fn(
            _float_ptr(logits_f),
            length,
            float(temperature),
            int(eos_token),
            float(top_p),
            int(top_k),
            float(min_p),
        )
    )


def seed_rng(seed: int) -> None:
    fn = getattr(_lib(), "simd_seed_rng_f32", None)
    if fn is None:
        raise RuntimeError(
            "Native-only QSG requires simd_seed_rng_f32; sampler seeding is unavailable."
        )
    fn(int(seed))


def suppress_tokens_inplace(logits, suppressed_ids) -> object:
    logits_f, length, _ = _coerce_1d_float_buffer(logits)
    ids_buf, ids_len = _coerce_1d_int_buffer(suppressed_ids)
    if length <= 0 or ids_len <= 0:
        return logits
    fn = getattr(_lib(), "simd_suppress_tokens_f32", None)
    if fn is None:
        raise RuntimeError(
            "Native-only QSG requires simd_suppress_tokens_f32; no Python fallback is allowed."
        )
    fn(_float_ptr(logits_f), length, _int_ptr(ids_buf), ids_len)
    return logits


def apply_token_penalties_inplace(
    logits,
    token_history,
    presence_penalty: float,
    repetition_penalty: float,
) -> object:
    logits_f, length, _ = _coerce_1d_float_buffer(logits)
    history_buf, history_len = _coerce_1d_int_buffer(token_history)
    if length <= 0 or history_len <= 0:
        return logits
    fn = getattr(_lib(), "simd_apply_token_penalties_f32", None)
    if fn is None:
        raise RuntimeError(
            "Native-only QSG requires simd_apply_token_penalties_f32; no Python fallback is allowed."
        )
    fn(
        _float_ptr(logits_f),
        length,
        _int_ptr(history_buf),
        history_len,
        float(presence_penalty),
        float(repetition_penalty),
    )
    return logits


def postprocess_and_sample(
    logits,
    *,
    suppressed_ids,
    token_history,
    presence_penalty: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    temperature: float,
    eos_token: int,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
) -> int:
    logits_f, length, _ = _coerce_1d_float_buffer(logits)
    suppressed_buf, suppressed_len = _coerce_1d_int_buffer(suppressed_ids)
    history_buf, history_len = _coerce_1d_int_buffer(token_history)
    fn = getattr(_lib(), "simd_postprocess_sample_f32", None)
    if fn is None:
        raise RuntimeError(
            "Native-only QSG requires simd_postprocess_sample_f32; no Python fallback is allowed."
        )
    return int(
        fn(
            _float_ptr(logits_f),
            length,
            _int_ptr(suppressed_buf),
            suppressed_len,
            _int_ptr(history_buf),
            history_len,
            float(presence_penalty),
            float(repetition_penalty),
            int(no_repeat_ngram_size),
            float(temperature),
            int(eos_token),
            float(top_p),
            int(top_k),
            float(min_p),
        )
    )


def qsg_postprocess_and_sample(
    logits,
    *,
    suppressed_ids,
    token_history,
    use_coconut: bool,
    coconut_paths: int,
    coconut_alpha: float,
    use_grover: bool,
    grover_top_k: int,
    grover_damping: float,
    presence_penalty: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    temperature: float,
    eos_token: int,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    grammar_allowed_ids=(),
) -> int:
    logits_f, length, _ = _coerce_1d_float_buffer(logits)
    suppressed_buf, suppressed_len = _coerce_1d_int_buffer(suppressed_ids)
    history_buf, history_len = _coerce_1d_int_buffer(token_history)
    grammar_buf, grammar_len = _coerce_1d_int_buffer(grammar_allowed_ids)
    fn = getattr(_lib(), "simd_qsg_postprocess_sample_f32", None)
    if fn is None:
        raise RuntimeError(
            "Native-only QSG requires simd_qsg_postprocess_sample_f32; no Python fallback is allowed."
        )
    return int(
        fn(
            _float_ptr(logits_f),
            length,
            _int_ptr(suppressed_buf),
            suppressed_len,
            _int_ptr(history_buf),
            history_len,
            _int_ptr(grammar_buf),
            grammar_len,
            int(bool(use_coconut)),
            int(coconut_paths),
            float(coconut_alpha),
            int(bool(use_grover)),
            int(grover_top_k),
            float(grover_damping),
            float(presence_penalty),
            float(repetition_penalty),
            int(no_repeat_ngram_size),
            float(temperature),
            int(eos_token),
            float(top_p),
            int(top_k),
            float(min_p),
        )
    )


def rope(
    q: np.ndarray,
    k: np.ndarray,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    pos: int,
    theta: float = 10000.0,
) -> None:
    q_f = _as_f32(q).reshape(n_heads, head_dim)
    k_f = _as_f32(k).reshape(n_kv_heads, head_dim)
    _lib().simd_rope_f32(
        q_f.ctypes.data_as(_FLOAT_P),
        k_f.ctypes.data_as(_FLOAT_P),
        n_heads,
        n_kv_heads,
        head_dim,
        pos,
        theta,
    )
    np.copyto(q, q_f.reshape(q.shape))
    np.copyto(k, k_f.reshape(k.shape))


def batch_rope(
    q: np.ndarray,
    k: np.ndarray,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    pos: int,
    theta: float = 10000.0,
) -> None:
    q_f = _as_f32(q).reshape(n_heads, head_dim)
    k_f = _as_f32(k).reshape(n_kv_heads, head_dim)
    lib = _lib()
    fn = getattr(lib, "batch_rope_f32", None)
    if fn is None:
        rope(q_f, k_f, n_heads, n_kv_heads, head_dim, pos, theta)
    else:
        fn(
            q_f.ctypes.data_as(_FLOAT_P),
            k_f.ctypes.data_as(_FLOAT_P),
            n_heads,
            n_kv_heads,
            head_dim,
            pos,
            theta,
        )
    np.copyto(q, q_f.reshape(q.shape))
    np.copyto(k, k_f.reshape(k.shape))


def fast_exp(x: np.ndarray) -> np.ndarray:
    x_f = _as_f32(x).reshape(-1).copy()
    lib = _lib()
    fn = getattr(lib, "simd_fast_exp_f32", None)
    if fn is None:
        raise RuntimeError(
            "Native-only QSG requires simd_fast_exp_f32; no NumPy fallback is allowed."
        )
    fn(x_f.ctypes.data_as(_FLOAT_P), x_f.shape[0])
    return x_f.reshape(np.asarray(x).shape)


def ssm_step(h: np.ndarray, a: np.ndarray, x_proj: np.ndarray) -> np.ndarray:
    """SSM state update: h = a * h + x_proj (linear recurrence, in-place on h)."""
    h_f = _as_f32(h).reshape(-1).copy()
    a_f = _as_f32(a).reshape(-1)
    x_f = _as_f32(x_proj).reshape(-1)
    _lib().simd_ssm_step_f32(
        h_f.ctypes.data_as(_FLOAT_P),
        a_f.ctypes.data_as(_FLOAT_P),
        x_f.ctypes.data_as(_FLOAT_P),
        h_f.shape[0],
    )
    return h_f


def ssm_parallel_scan(
    a: np.ndarray,
    alphas: np.ndarray,
    h_init: np.ndarray | None = None,
) -> np.ndarray:
    a_f = _as_f32(a).reshape(-1)
    alpha_f = _as_f32(alphas)
    if alpha_f.ndim != 2:
        raise ValueError("ssm_parallel_scan expects [seq, state]")
    init = None
    if h_init is not None:
        init = _as_f32(h_init).reshape(-1)
    out = np.empty_like(alpha_f)
    _lib().simd_ssm_parallel_scan_f32(
        out.ctypes.data_as(_FLOAT_P),
        a_f.ctypes.data_as(_FLOAT_P),
        alpha_f.ctypes.data_as(_FLOAT_P),
        None
        if init is None
        else init.ctypes.data_as(_FLOAT_P),
        alpha_f.shape[0],
        alpha_f.shape[1],
    )
    return out


def set_num_threads(n: int) -> None:
    lib = _lib()
    fn = getattr(lib, "anvil_set_num_threads", None)
    if fn is None:
        return
    fn(int(n))


def get_num_threads() -> int:
    lib = _lib()
    fn = getattr(lib, "anvil_get_num_threads", None)
    if fn is None:
        return 1
    return int(fn())


def get_num_threads_for_path(decode_path: bool = True) -> int:
    lib = _lib()
    fn = getattr(lib, "anvil_get_num_threads_for_path", None)
    if fn is not None:
        return int(fn(1 if decode_path else 0))
    return get_num_threads()


def set_thread_mode(decode_path: bool = True) -> None:
    lib = _lib()
    fn = getattr(lib, "anvil_set_thread_mode", None)
    if fn is None:
        return
    fn(1 if decode_path else 0)


def get_thread_mode() -> int:
    lib = _lib()
    fn = getattr(lib, "anvil_get_thread_mode", None)
    if fn is None:
        return -1
    return int(fn())


def get_num_procs() -> int:
    lib = _lib()
    fn = getattr(lib, "anvil_get_num_procs", None)
    if fn is None:
        return 1
    return int(fn())


def detect_physical_cores() -> int:
    lib = _lib()
    fn = getattr(lib, "anvil_detect_physical_cores", None)
    if fn is None:
        return 0
    return int(fn())


def openmp_enabled() -> bool:
    lib = _lib()
    fn = getattr(lib, "anvil_openmp_enabled", None)
    if fn is None:
        return False
    return bool(fn())


def get_omp_max_threads() -> int:
    lib = _lib()
    fn = getattr(lib, "anvil_get_omp_max_threads", None)
    if fn is None:
        return get_num_threads()
    return int(fn())


def get_omp_dynamic() -> bool:
    lib = _lib()
    fn = getattr(lib, "anvil_get_omp_dynamic", None)
    if fn is None:
        return False
    return bool(fn())


def get_omp_active_levels() -> int:
    lib = _lib()
    fn = getattr(lib, "anvil_get_omp_active_levels", None)
    if fn is None:
        return 0
    return int(fn())


def compiled_with_avx2() -> bool:
    lib = _lib()
    fn = getattr(lib, "anvil_compiled_with_avx2", None)
    if fn is None:
        return False
    return bool(fn())


def compiled_with_avx512() -> bool:
    lib = _lib()
    fn = getattr(lib, "anvil_compiled_with_avx512", None)
    if fn is None:
        return False
    return bool(fn())


def compiled_with_amx() -> bool:
    lib = _lib()
    fn = getattr(lib, "anvil_compiled_with_amx", None)
    if fn is None:
        return False
    return bool(fn())


def runtime_amx_available() -> bool:
    lib = _lib()
    fn = getattr(lib, "anvil_runtime_amx_available", None)
    if fn is None:
        return False
    return bool(fn())


def get_p_core_count() -> int:
    lib = _lib()
    native_fn = getattr(lib, "anvil_get_p_core_count", None)
    if native_fn is not None:
        return int(native_fn())
    fn = getattr(lib, "simd_get_p_core_count", None)
    if fn is None:
        return 0
    return int(fn())


def set_thread_affinity(use_p_cores_only: bool = False) -> bool:
    lib = _lib()
    fn = getattr(lib, "anvil_set_thread_affinity", None)
    if fn is not None:
        return int(fn(1 if use_p_cores_only else 0)) == 1
    return pin_to_p_cores() if use_p_cores_only else False


def get_affinity_mode() -> int:
    lib = _lib()
    fn = getattr(lib, "anvil_get_affinity_mode", None)
    if fn is None:
        return 1
    return int(fn())


def configure_affinity_mode(mode: int) -> int:
    lib = _lib()
    fn = getattr(lib, "anvil_configure_affinity_mode", None)
    if fn is None:
        return int(mode)
    return int(fn(int(mode)))


def get_l3_domain_count() -> int:
    lib = _lib()
    fn = getattr(lib, "anvil_get_l3_domain_count", None)
    if fn is None:
        return 0
    return int(fn())


def bind_worker_thread(worker_tid: int, decode_path: bool) -> bool:
    lib = _lib()
    fn = getattr(lib, "anvil_bind_worker_thread", None)
    if fn is None:
        return False
    return int(fn(int(worker_tid), 1 if decode_path else 0)) == 1


def sample_thread_cpu() -> int:
    lib = _lib()
    fn = getattr(lib, "anvil_sample_thread_cpu", None)
    if fn is None:
        return -1
    return int(fn())


def get_thread_migration_count() -> int:
    lib = _lib()
    fn = getattr(lib, "anvil_get_thread_migration_count", None)
    if fn is None:
        return 0
    return int(fn())


def get_last_cpu() -> int:
    lib = _lib()
    fn = getattr(lib, "anvil_get_last_cpu", None)
    if fn is None:
        return -1
    return int(fn())


def refresh_topology() -> bool:
    lib = _lib()
    fn = getattr(lib, "anvil_refresh_topology", None)
    if fn is None:
        return False
    return int(fn()) == 1


def export_topology_json() -> str:
    lib = _lib()
    fn = getattr(lib, "anvil_topology_export_json", None)
    if fn is None:
        return ""
    buf_size = 32768
    buf = ctypes.create_string_buffer(buf_size)
    written = int(fn(buf, int(buf_size)))
    if written <= 0:
        return ""
    return buf.value.decode("utf-8", errors="replace")


def export_affinity_plan_json() -> str:
    lib = _lib()
    fn = getattr(lib, "anvil_affinity_plan_export_json", None)
    if fn is None:
        return ""
    buf_size = 8192
    buf = ctypes.create_string_buffer(buf_size)
    written = int(fn(buf, int(buf_size)))
    if written <= 0:
        return ""
    return buf.value.decode("utf-8", errors="replace")


def reset_qsg_sampling_stats() -> None:
    lib = _lib()
    fn = getattr(lib, "anvil_reset_qsg_sampling_stats", None)
    if fn is None:
        return
    fn()


def get_qsg_sampling_stats() -> dict[str, float | int]:
    lib = _lib()
    fn = getattr(lib, "anvil_qsg_sampling_stats_json", None)
    if fn is None:
        return {}
    buf_size = 4096
    buf = ctypes.create_string_buffer(buf_size)
    written = int(fn(buf, int(buf_size)))
    if written <= 0:
        return {}
    try:
        payload = json.loads(buf.value.decode("utf-8", errors="replace"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def pin_to_p_cores(n_p_cores: int | None = None) -> bool:
    lib = _lib()
    fn = getattr(lib, "simd_pin_to_p_cores", None)
    if fn is None:
        return False
    n = int(n_p_cores) if n_p_cores is not None else 0
    return int(fn(n)) == 1
