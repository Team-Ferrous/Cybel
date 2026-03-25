"""
Compatibility helpers for llama.cpp KV cache APIs across versions.

Newer llama.cpp builds replaced `llama_kv_cache_*` with `llama_memory_*`.
These wrappers allow runtime dispatch without hard-coding one API generation.
"""

from __future__ import annotations

from typing import Optional

from llama_cpp import llama_cpp

_KV_API_BACKEND: Optional[str] = None


def _symbol_available(name: str) -> bool:
    symbol = getattr(llama_cpp, name, None)
    if symbol is None:
        return False
    # Our patched llama_cpp wrapper marks missing bindings with this attribute.
    return not bool(getattr(symbol, "_llama_missing_symbol", False))


def _detect_backend() -> str:
    if all(
        _symbol_available(name)
        for name in (
            "llama_get_memory",
            "llama_memory_clear",
            "llama_memory_seq_rm",
            "llama_memory_seq_add",
        )
    ):
        return "memory"
    if all(
        _symbol_available(name)
        for name in (
            "llama_kv_cache_clear",
            "llama_kv_cache_seq_rm",
            "llama_kv_cache_seq_add",
        )
    ):
        return "kv_cache"
    if all(
        _symbol_available(name)
        for name in (
            "llama_kv_self_clear",
            "llama_kv_self_seq_rm",
            "llama_kv_self_seq_add",
        )
    ):
        return "kv_self"
    raise RuntimeError(
        "No compatible llama KV/memory API found. Expected one of "
        "{llama_memory_*, llama_kv_cache_*, llama_kv_self_*}."
    )


def _backend() -> str:
    global _KV_API_BACKEND
    if _KV_API_BACKEND is None:
        _KV_API_BACKEND = _detect_backend()
    return _KV_API_BACKEND


def _memory_handle(ctx):
    mem = llama_cpp.llama_get_memory(ctx)
    if not mem:
        raise RuntimeError("llama_get_memory returned null memory handle")
    return mem


def kv_cache_clear(ctx) -> None:
    backend = _backend()
    if backend == "memory":
        llama_cpp.llama_memory_clear(_memory_handle(ctx), True)
        return
    if backend == "kv_cache":
        llama_cpp.llama_kv_cache_clear(ctx)
        return
    llama_cpp.llama_kv_self_clear(ctx)


def kv_cache_seq_rm(ctx, seq_id: int, p0: int, p1: int) -> None:
    backend = _backend()
    if backend == "memory":
        ok = llama_cpp.llama_memory_seq_rm(_memory_handle(ctx), seq_id, p0, p1)
        if ok is False:
            raise RuntimeError(
                f"llama_memory_seq_rm failed for seq_id={seq_id}, range=[{p0}, {p1})"
            )
        return
    if backend == "kv_cache":
        llama_cpp.llama_kv_cache_seq_rm(ctx, seq_id, p0, p1)
        return
    ok = llama_cpp.llama_kv_self_seq_rm(ctx, seq_id, p0, p1)
    if ok is False:
        raise RuntimeError(
            f"llama_kv_self_seq_rm failed for seq_id={seq_id}, range=[{p0}, {p1})"
        )


def kv_cache_seq_add(ctx, seq_id: int, p0: int, p1: int, delta: int) -> None:
    backend = _backend()
    if backend == "memory":
        llama_cpp.llama_memory_seq_add(_memory_handle(ctx), seq_id, p0, p1, delta)
        return
    if backend == "kv_cache":
        llama_cpp.llama_kv_cache_seq_add(ctx, seq_id, p0, p1, delta)
        return
    llama_cpp.llama_kv_self_seq_add(ctx, seq_id, p0, p1, delta)

