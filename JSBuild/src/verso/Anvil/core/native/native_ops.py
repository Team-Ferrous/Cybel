"""Helpers for loading the unified native ops shared library."""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
from typing import Iterable

_LIB_CACHE: dict[str, ctypes.CDLL] = {}
_BACKEND_LIB_CACHE: dict[str, ctypes.CDLL] = {}


def _normalize_backend_name(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def native_lib_candidates(extra: Iterable[str] | None = None) -> list[Path]:
    base_dir = Path(__file__).resolve().parent
    names = [
        "libanvil_native_ops.so",
        "libfast_attention.so",
        "libcoconut_bridge.so",
    ]
    if extra:
        names = list(extra) + names
    return [base_dir / name for name in names]


def backend_module_candidates(backend_name: str) -> list[Path]:
    base_dir = Path(__file__).resolve().parent
    normalized = _normalize_backend_name(backend_name)
    if not normalized:
        return []
    return [
        base_dir / f"libanvil_backend_{normalized}.so",
        base_dir / "build" / f"libanvil_backend_{normalized}.so",
    ]


def load_native_library(extra: Iterable[str] | None = None) -> ctypes.CDLL:
    """Load the unified native ops library, falling back to legacy names."""
    candidates = native_lib_candidates(extra)
    for candidate in candidates:
        key = os.fspath(candidate)
        if key in _LIB_CACHE:
            return _LIB_CACHE[key]
        if not candidate.exists():
            continue
        lib = ctypes.CDLL(key)
        _LIB_CACHE[key] = lib
        return lib

    searched = ", ".join(os.fspath(path) for path in candidates)
    raise FileNotFoundError(f"Native ops library not found. Checked: {searched}")


def loaded_library_path(extra: Iterable[str] | None = None) -> Path:
    candidates = native_lib_candidates(extra)
    for candidate in candidates:
        key = os.fspath(candidate)
        if key in _LIB_CACHE:
            return candidate
        if candidate.exists():
            return candidate
    searched = ", ".join(os.fspath(path) for path in candidates)
    raise FileNotFoundError(f"Native ops library not found. Checked: {searched}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_char_ptr_symbol(lib: ctypes.CDLL, symbol: str) -> str:
    getter = getattr(lib, symbol, None)
    if getter is None:
        return ""
    try:
        getter.argtypes = []
        getter.restype = ctypes.c_char_p
        raw = getter()
    except Exception:
        return ""
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="ignore")
    return str(raw)


def _read_int_symbol(lib: ctypes.CDLL, symbol: str, *, default: int = 0) -> int:
    getter = getattr(lib, symbol, None)
    if getter is None:
        return int(default)
    try:
        getter.argtypes = []
        getter.restype = ctypes.c_int32
        return int(getter())
    except Exception:
        return int(default)


def get_native_library_info(extra: Iterable[str] | None = None) -> dict[str, object]:
    lib = load_native_library(extra)
    path = loaded_library_path(extra)
    build_id = _read_char_ptr_symbol(lib, "anvil_native_build_id")
    compat_alias_csv = _read_char_ptr_symbol(lib, "anvil_native_compat_alias_csv")
    compat_aliases = [
        item.strip() for item in compat_alias_csv.split(",") if item.strip()
    ]
    optional_isa_leaves_csv = _read_char_ptr_symbol(
        lib, "anvil_native_optional_isa_leaves"
    )
    optional_isa_leaves = [
        item.strip() for item in optional_isa_leaves_csv.split(",") if item.strip()
    ]
    return {
        "loaded_native_library": os.fspath(path),
        "native_build_id": build_id,
        "native_build_sha256": _sha256_file(path),
        "native_split_layout": _read_char_ptr_symbol(lib, "anvil_native_split_layout"),
        "native_public_load_target": _read_char_ptr_symbol(
            lib,
            "anvil_native_public_load_target",
        ),
        "native_runtime_core_target": _read_char_ptr_symbol(
            lib,
            "anvil_native_runtime_core_target",
        ),
        "native_split_abi_version": _read_int_symbol(
            lib,
            "anvil_native_split_abi_version",
            default=0,
        ),
        "native_isa_baseline": _read_char_ptr_symbol(
            lib,
            "anvil_native_isa_baseline",
        ),
        "native_optional_isa_leaves_csv": optional_isa_leaves_csv,
        "native_optional_isa_leaves": optional_isa_leaves,
        "native_compiled_with_amx": bool(
            _read_int_symbol(lib, "anvil_compiled_with_amx", default=0)
        ),
        "native_runtime_amx_available": bool(
            _read_int_symbol(lib, "anvil_runtime_amx_available", default=0)
        ),
        "native_compat_alias_csv": compat_alias_csv,
        "native_compat_aliases": compat_aliases,
    }


def resolve_backend_module_selection(
    *,
    backend_name: str = "",
    model_name: str = "",
    architecture: str = "",
    family: str = "",
) -> dict[str, str]:
    requested = _normalize_backend_name(backend_name)
    model = str(model_name or "").strip().lower()
    arch = str(architecture or "").strip().lower()
    fam = str(family or "").strip().lower()

    if requested:
        return {
            "backend_module": requested,
            "backend_module_requested": requested,
            "backend_selection_source": "explicit",
            "backend_selection_reason": "backend_name",
            "backend_selection_model_name": model,
            "backend_selection_architecture": arch,
            "backend_selection_family": fam,
        }

    if "qwen3.5:4b" in model:
        return {
            "backend_module": "qwen35",
            "backend_module_requested": "",
            "backend_selection_source": "model_name_exact",
            "backend_selection_reason": "qwen3.5:4b",
            "backend_selection_model_name": model,
            "backend_selection_architecture": arch,
            "backend_selection_family": fam,
        }
    if "qwen3.5:9b" in model:
        return {
            "backend_module": "qwen35",
            "backend_module_requested": "",
            "backend_selection_source": "model_name_exact",
            "backend_selection_reason": "qwen3.5:9b",
            "backend_selection_model_name": model,
            "backend_selection_architecture": arch,
            "backend_selection_family": fam,
        }
    if fam == "qwen" or "qwen" in arch or "qwen3.5" in model or "qwen35" in model:
        reason = "family" if fam == "qwen" else "architecture_or_model_name"
        return {
            "backend_module": "qwen35",
            "backend_module_requested": "",
            "backend_selection_source": "inferred",
            "backend_selection_reason": reason,
            "backend_selection_model_name": model,
            "backend_selection_architecture": arch,
            "backend_selection_family": fam,
        }
    if fam == "granite" or "granite" in arch or "granite4:tiny-h" in model:
        reason = "family" if fam == "granite" else "architecture_or_model_name"
        return {
            "backend_module": "granite4_tinyh",
            "backend_module_requested": "",
            "backend_selection_source": "inferred",
            "backend_selection_reason": reason,
            "backend_selection_model_name": model,
            "backend_selection_architecture": arch,
            "backend_selection_family": fam,
        }
    return {
        "backend_module": "",
        "backend_module_requested": "",
        "backend_selection_source": "unresolved",
        "backend_selection_reason": "no_match",
        "backend_selection_model_name": model,
        "backend_selection_architecture": arch,
        "backend_selection_family": fam,
    }


def resolve_backend_module_name(
    *,
    backend_name: str = "",
    model_name: str = "",
    architecture: str = "",
    family: str = "",
) -> str:
    selection = resolve_backend_module_selection(
        backend_name=backend_name,
        model_name=model_name,
        architecture=architecture,
        family=family,
    )
    return str(selection.get("backend_module", ""))


def load_backend_module(backend_name: str) -> tuple[ctypes.CDLL, Path]:
    normalized = _normalize_backend_name(backend_name)
    if not normalized:
        raise RuntimeError("Backend module name is required.")
    candidates = backend_module_candidates(normalized)
    for candidate in candidates:
        key = os.fspath(candidate)
        if key in _BACKEND_LIB_CACHE:
            return _BACKEND_LIB_CACHE[key], candidate
        if not candidate.exists():
            continue
        lib = ctypes.CDLL(key)
        _BACKEND_LIB_CACHE[key] = lib
        return lib, candidate
    searched = ", ".join(os.fspath(path) for path in candidates)
    raise FileNotFoundError(
        f"Native backend module '{normalized}' not found. Checked: {searched}"
    )


def get_native_backend_info(
    *,
    backend_name: str,
    model_name: str = "",
    architecture: str = "",
    family: str = "",
) -> dict[str, object]:
    selection = resolve_backend_module_selection(
        backend_name=backend_name,
        model_name=model_name,
        architecture=architecture,
        family=family,
    )
    resolved = str(selection.get("backend_module", ""))
    candidates = [os.fspath(path) for path in backend_module_candidates(resolved)]
    backend_metadata = {
        "backend_module": resolved,
        "backend_module_loaded": False,
        "backend_module_library": "",
        "backend_module_marker_symbol": "",
        "backend_module_marker": 0,
        "backend_module_name_symbol": "",
        "backend_module_name": "",
        "backend_module_build_id_symbol": "",
        "backend_module_build_id": "",
        "backend_module_abi_symbol": "",
        "backend_module_abi_version": 0,
        "backend_module_candidates": candidates,
        "backend_module_error": "",
    }
    backend_metadata.update(selection)

    if not resolved:
        backend_metadata["backend_module_error"] = "no_backend_resolution"
        return backend_metadata

    marker_symbol = f"anvil_backend_{resolved}_marker"
    name_symbol = f"anvil_backend_{resolved}_name"
    build_id_symbol = f"anvil_backend_{resolved}_build_id"
    abi_symbol = f"anvil_backend_{resolved}_abi_version"

    backend_metadata.update(
        {
            "backend_module_marker_symbol": marker_symbol,
            "backend_module_name_symbol": name_symbol,
            "backend_module_build_id_symbol": build_id_symbol,
            "backend_module_abi_symbol": abi_symbol,
        }
    )

    try:
        lib, path = load_backend_module(resolved)
    except Exception as exc:
        backend_metadata["backend_module_error"] = str(exc)
        return backend_metadata

    marker_value = _read_int_symbol(lib, marker_symbol, default=0)
    backend_metadata.update(
        {
            "backend_module_loaded": True,
            "backend_module_library": os.fspath(path),
            "backend_module_marker": marker_value,
            "backend_module_name": _read_char_ptr_symbol(lib, name_symbol),
            "backend_module_build_id": _read_char_ptr_symbol(lib, build_id_symbol),
            "backend_module_abi_version": _read_int_symbol(lib, abi_symbol, default=0),
        }
    )
    return backend_metadata
