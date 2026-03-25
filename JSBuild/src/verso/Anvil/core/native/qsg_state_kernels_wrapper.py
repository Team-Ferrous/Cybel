from __future__ import annotations

import ctypes
from collections.abc import Mapping, Sequence
from typing import Any

from core.native.native_ops import load_native_library

_FLOAT_P = ctypes.POINTER(ctypes.c_float)
_UINT16_P = ctypes.POINTER(ctypes.c_uint16)
_LIB: ctypes.CDLL | None = None


class _RowRef(ctypes.Structure):
    _fields_ = [
        ("page_id", ctypes.c_int32),
        ("row_idx", ctypes.c_int32),
    ]


class _CompactMove(ctypes.Structure):
    _fields_ = [
        ("src", _RowRef),
        ("dst", _RowRef),
    ]


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    lib = load_native_library()
    required = (
        "qsg_state_gather_rows",
        "qsg_state_scatter_rows",
        "qsg_state_clone_cow",
        "qsg_state_compact",
        "qsg_state_weighted_merge",
        "qsg_latent_encode_f16",
        "qsg_latent_decode_f16",
    )
    missing = [symbol for symbol in required if not hasattr(lib, symbol)]
    if missing:
        raise RuntimeError(
            "Native-only QSG state kernels require symbols: " + ", ".join(missing)
        )

    pages_pp = ctypes.POINTER(_FLOAT_P)
    row_refs_p = ctypes.POINTER(_RowRef)
    compact_moves_p = ctypes.POINTER(_CompactMove)

    lib.qsg_state_gather_rows.argtypes = [
        _FLOAT_P,
        pages_pp,
        row_refs_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.qsg_state_gather_rows.restype = None

    lib.qsg_state_scatter_rows.argtypes = [
        _FLOAT_P,
        pages_pp,
        row_refs_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.qsg_state_scatter_rows.restype = None

    lib.qsg_state_clone_cow.argtypes = [
        pages_pp,
        row_refs_p,
        row_refs_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.qsg_state_clone_cow.restype = None

    lib.qsg_state_compact.argtypes = [
        pages_pp,
        compact_moves_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.qsg_state_compact.restype = None

    lib.qsg_state_weighted_merge.argtypes = [
        _FLOAT_P,
        _FLOAT_P,
        _FLOAT_P,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.qsg_state_weighted_merge.restype = None

    lib.qsg_latent_encode_f16.argtypes = [
        _FLOAT_P,
        _UINT16_P,
        ctypes.c_int32,
    ]
    lib.qsg_latent_encode_f16.restype = None

    lib.qsg_latent_decode_f16.argtypes = [
        _UINT16_P,
        _FLOAT_P,
        ctypes.c_int32,
    ]
    lib.qsg_latent_decode_f16.restype = None

    _LIB = lib
    return lib


def _as_row_refs(refs: Sequence[tuple[int, int]] | Any) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    for row in refs:
        if len(row) != 2:
            raise ValueError("row refs must have shape [N, 2]")
        page_id = int(row[0])
        row_idx = int(row[1])
        if page_id < 0 or row_idx < 0:
            raise ValueError("row refs must be non-negative")
        rows.append((page_id, row_idx))
    return rows


def _row_refs_buffer(
    ref_rows: list[tuple[int, int]],
) -> tuple[ctypes.Array[_RowRef], int]:
    n_refs = len(ref_rows)
    rows = (_RowRef * n_refs)()
    for idx in range(n_refs):
        rows[idx].page_id = int(ref_rows[idx][0])
        rows[idx].row_idx = int(ref_rows[idx][1])
    return rows, n_refs


def _pages_pointer_table(
    pages: Mapping[int, Any],
    *,
    dim: int,
) -> tuple[ctypes.Array[_FLOAT_P], int]:
    if not pages:
        raise RuntimeError(
            "Native-only QSG state kernels require at least one allocated page."
        )
    page_ids = [int(page_id) for page_id in pages]
    max_page_id = max(page_ids)
    table_size = max_page_id + 1
    ptr_table = (_FLOAT_P * table_size)()
    rows_per_page: int | None = None
    for page_id, page in pages.items():
        page_id = int(page_id)
        page_array = page
        page_dtype = str(getattr(page_array, "dtype", ""))
        if page_dtype != "float32":
            raise TypeError("state pages must be float32")
        page_shape = tuple(getattr(page_array, "shape", ()))
        if len(page_shape) != 2:
            raise TypeError("state pages must be 2D tensors")
        page_flags = getattr(page_array, "flags", None)
        if page_flags is None or not bool(page_flags["C_CONTIGUOUS"]):
            raise TypeError("state pages must be C-contiguous")
        if int(page_shape[1]) < dim:
            raise ValueError(
                f"state page has dim={page_shape[1]} but requested dim={dim}"
            )
        if rows_per_page is None:
            rows_per_page = int(page_shape[0])
        elif int(page_shape[0]) != rows_per_page:
            raise ValueError("all state pages must share the same row count")
        if not hasattr(page_array, "ctypes"):
            raise TypeError("state pages must expose ctypes pointers")
        ptr_table[page_id] = page_array.ctypes.data_as(_FLOAT_P)
    if rows_per_page is None:
        raise RuntimeError("unable to determine rows_per_page for state pages")
    return ptr_table, int(rows_per_page)


def qsg_state_gather_rows(
    dst: Any,
    pages: Mapping[int, Any],
    refs: Sequence[tuple[int, int]] | Any,
    dim: int,
) -> None:
    ref_rows = _as_row_refs(refs)
    dim = int(dim)
    if not ref_rows:
        return
    dst_arr = dst
    dst_dtype = str(getattr(dst_arr, "dtype", ""))
    dst_flags = getattr(dst_arr, "flags", None)
    dst_shape = tuple(getattr(dst_arr, "shape", ()))
    if dst_dtype != "float32" or dst_flags is None or not dst_flags["C_CONTIGUOUS"]:
        raise TypeError("dst must be a contiguous float32 array")
    if dst_shape != (len(ref_rows), dim):
        raise ValueError(
            f"dst must have shape ({len(ref_rows)}, {dim}), got {dst_shape}"
        )
    if not hasattr(dst_arr, "ctypes"):
        raise TypeError("dst must expose ctypes pointers")
    ptr_table, rows_per_page = _pages_pointer_table(pages, dim=dim)
    row_refs, n_refs = _row_refs_buffer(ref_rows)
    _lib().qsg_state_gather_rows(
        dst_arr.ctypes.data_as(_FLOAT_P),
        ptr_table,
        row_refs,
        n_refs,
        rows_per_page,
        dim,
    )


def qsg_state_scatter_rows(
    src: Any,
    pages: Mapping[int, Any],
    refs: Sequence[tuple[int, int]] | Any,
    dim: int,
) -> None:
    ref_rows = _as_row_refs(refs)
    dim = int(dim)
    if not ref_rows:
        return
    src_arr = src
    src_dtype = str(getattr(src_arr, "dtype", ""))
    src_flags = getattr(src_arr, "flags", None)
    src_shape = tuple(getattr(src_arr, "shape", ()))
    if src_dtype != "float32" or src_flags is None or not src_flags["C_CONTIGUOUS"]:
        raise TypeError("src must be a contiguous float32 array")
    if src_shape != (len(ref_rows), dim):
        raise ValueError(
            f"src must have shape ({len(ref_rows)}, {dim}), got {src_shape}"
        )
    if not hasattr(src_arr, "ctypes"):
        raise TypeError("src must expose ctypes pointers")
    ptr_table, rows_per_page = _pages_pointer_table(pages, dim=dim)
    row_refs, n_refs = _row_refs_buffer(ref_rows)
    _lib().qsg_state_scatter_rows(
        src_arr.ctypes.data_as(_FLOAT_P),
        ptr_table,
        row_refs,
        n_refs,
        rows_per_page,
        dim,
    )


def qsg_state_clone_cow(
    pages: Mapping[int, Any],
    src_refs: Sequence[tuple[int, int]] | Any,
    dst_refs: Sequence[tuple[int, int]] | Any,
    dim: int,
) -> None:
    src_rows = _as_row_refs(src_refs)
    dst_rows = _as_row_refs(dst_refs)
    if len(src_rows) != len(dst_rows):
        raise ValueError("src_refs and dst_refs must have same length")
    if not src_rows:
        return
    dim = int(dim)
    ptr_table, rows_per_page = _pages_pointer_table(pages, dim=dim)
    src_buf, n_rows = _row_refs_buffer(src_rows)
    dst_buf, _ = _row_refs_buffer(dst_rows)
    _lib().qsg_state_clone_cow(
        ptr_table,
        src_buf,
        dst_buf,
        n_rows,
        rows_per_page,
        dim,
    )


def qsg_state_compact(
    pages: Mapping[int, Any],
    move_src_refs: Sequence[tuple[int, int]] | Any,
    move_dst_refs: Sequence[tuple[int, int]] | Any,
    dim: int,
) -> None:
    src_rows = _as_row_refs(move_src_refs)
    dst_rows = _as_row_refs(move_dst_refs)
    if len(src_rows) != len(dst_rows):
        raise ValueError("move_src_refs and move_dst_refs must have same length")
    if not src_rows:
        return
    dim = int(dim)
    ptr_table, rows_per_page = _pages_pointer_table(pages, dim=dim)
    n_moves = len(src_rows)
    moves = (_CompactMove * n_moves)()
    for idx in range(n_moves):
        moves[idx].src.page_id = int(src_rows[idx][0])
        moves[idx].src.row_idx = int(src_rows[idx][1])
        moves[idx].dst.page_id = int(dst_rows[idx][0])
        moves[idx].dst.row_idx = int(dst_rows[idx][1])
    _lib().qsg_state_compact(
        ptr_table,
        moves,
        n_moves,
        rows_per_page,
        dim,
    )


def qsg_state_weighted_merge(src: Any, weights: Any) -> Any:
    src_arr = src
    weights_arr = weights
    src_shape = tuple(getattr(src_arr, "shape", ()))
    if len(src_shape) != 2:
        raise ValueError("src must have shape [rows, dim]")
    if str(getattr(src_arr, "dtype", "")) != "float32":
        raise TypeError("src must be float32")
    src_flags = getattr(src_arr, "flags", None)
    if src_flags is None or not bool(src_flags["C_CONTIGUOUS"]):
        raise TypeError("src must be C-contiguous")
    if str(getattr(weights_arr, "dtype", "")) != "float32":
        raise TypeError("weights must be float32")
    weight_shape = tuple(getattr(weights_arr, "shape", ()))
    if weight_shape != (src_shape[0],):
        raise ValueError(
            f"weights must have shape ({src_shape[0]},), got {weight_shape}"
        )
    weight_flags = getattr(weights_arr, "flags", None)
    if weight_flags is None or not bool(weight_flags["C_CONTIGUOUS"]):
        raise TypeError("weights must be C-contiguous")

    import numpy as np

    dst = np.zeros((src_shape[1],), dtype=np.float32)
    _lib().qsg_state_weighted_merge(
        dst.ctypes.data_as(_FLOAT_P),
        src_arr.ctypes.data_as(_FLOAT_P),
        weights_arr.ctypes.data_as(_FLOAT_P),
        int(src_shape[0]),
        int(src_shape[1]),
    )
    return dst


def qsg_latent_encode_f16(src: Any) -> Any:
    src_arr = src
    src_shape = tuple(getattr(src_arr, "shape", ()))
    if str(getattr(src_arr, "dtype", "")) != "float32":
        raise TypeError("src must be float32")
    src_flags = getattr(src_arr, "flags", None)
    if src_flags is None or not bool(src_flags["C_CONTIGUOUS"]):
        raise TypeError("src must be C-contiguous")

    import numpy as np

    encoded = np.zeros(src_shape, dtype=np.uint16)
    count = int(encoded.size)
    _lib().qsg_latent_encode_f16(
        src_arr.ctypes.data_as(_FLOAT_P),
        encoded.ctypes.data_as(_UINT16_P),
        count,
    )
    return encoded


def qsg_latent_decode_f16(src: Any) -> Any:
    src_arr = src
    src_shape = tuple(getattr(src_arr, "shape", ()))
    if str(getattr(src_arr, "dtype", "")) != "uint16":
        raise TypeError("src must be uint16")
    src_flags = getattr(src_arr, "flags", None)
    if src_flags is None or not bool(src_flags["C_CONTIGUOUS"]):
        raise TypeError("src must be C-contiguous")

    import numpy as np

    decoded = np.zeros(src_shape, dtype=np.float32)
    count = int(decoded.size)
    _lib().qsg_latent_decode_f16(
        src_arr.ctypes.data_as(_UINT16_P),
        decoded.ctypes.data_as(_FLOAT_P),
        count,
    )
    return decoded
