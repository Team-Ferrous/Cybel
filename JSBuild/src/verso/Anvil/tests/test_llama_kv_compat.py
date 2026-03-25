from types import SimpleNamespace

import pytest

import core.native.llama_kv_compat as compat


def _set_backend(monkeypatch, fake_llama_cpp):
    monkeypatch.setattr(compat, "llama_cpp", fake_llama_cpp, raising=False)
    monkeypatch.setattr(compat, "_KV_API_BACKEND", None, raising=False)


def test_memory_backend_dispatch(monkeypatch):
    calls = []

    fake = SimpleNamespace(
        llama_get_memory=lambda ctx: f"mem:{ctx}",
        llama_memory_clear=lambda mem, data: calls.append(("clear", mem, data)),
        llama_memory_seq_rm=lambda mem, seq_id, p0, p1: calls.append(
            ("rm", mem, seq_id, p0, p1)
        )
        or True,
        llama_memory_seq_add=lambda mem, seq_id, p0, p1, delta: calls.append(
            ("add", mem, seq_id, p0, p1, delta)
        ),
    )
    _set_backend(monkeypatch, fake)

    compat.kv_cache_clear("ctx")
    compat.kv_cache_seq_rm("ctx", 0, 1, 3)
    compat.kv_cache_seq_add("ctx", 0, 1, 3, -1)

    assert calls[0] == ("clear", "mem:ctx", True)
    assert calls[1] == ("rm", "mem:ctx", 0, 1, 3)
    assert calls[2] == ("add", "mem:ctx", 0, 1, 3, -1)


def test_memory_backend_rm_failure_raises(monkeypatch):
    fake = SimpleNamespace(
        llama_get_memory=lambda ctx: "mem",
        llama_memory_clear=lambda mem, data: None,
        llama_memory_seq_rm=lambda mem, seq_id, p0, p1: False,
        llama_memory_seq_add=lambda mem, seq_id, p0, p1, delta: None,
    )
    _set_backend(monkeypatch, fake)

    with pytest.raises(RuntimeError, match="llama_memory_seq_rm failed"):
        compat.kv_cache_seq_rm("ctx", 0, 0, 4)


def test_kv_cache_backend_dispatch(monkeypatch):
    calls = []

    fake = SimpleNamespace(
        llama_kv_cache_clear=lambda ctx: calls.append(("clear", ctx)),
        llama_kv_cache_seq_rm=lambda ctx, seq_id, p0, p1: calls.append(
            ("rm", ctx, seq_id, p0, p1)
        ),
        llama_kv_cache_seq_add=lambda ctx, seq_id, p0, p1, delta: calls.append(
            ("add", ctx, seq_id, p0, p1, delta)
        ),
    )
    _set_backend(monkeypatch, fake)

    compat.kv_cache_clear("ctx")
    compat.kv_cache_seq_rm("ctx", 2, 3, 4)
    compat.kv_cache_seq_add("ctx", 2, 3, 4, -2)

    assert calls == [
        ("clear", "ctx"),
        ("rm", "ctx", 2, 3, 4),
        ("add", "ctx", 2, 3, 4, -2),
    ]


def test_kv_self_backend_dispatch(monkeypatch):
    calls = []

    fake = SimpleNamespace(
        llama_kv_self_clear=lambda ctx: calls.append(("clear", ctx)),
        llama_kv_self_seq_rm=lambda ctx, seq_id, p0, p1: calls.append(
            ("rm", ctx, seq_id, p0, p1)
        )
        or True,
        llama_kv_self_seq_add=lambda ctx, seq_id, p0, p1, delta: calls.append(
            ("add", ctx, seq_id, p0, p1, delta)
        ),
    )
    _set_backend(monkeypatch, fake)

    compat.kv_cache_clear("ctx")
    compat.kv_cache_seq_rm("ctx", 7, 1, 9)
    compat.kv_cache_seq_add("ctx", 7, 1, 9, -3)

    assert calls == [
        ("clear", "ctx"),
        ("rm", "ctx", 7, 1, 9),
        ("add", "ctx", 7, 1, 9, -3),
    ]


def test_no_backend_raises(monkeypatch):
    _set_backend(monkeypatch, SimpleNamespace())
    with pytest.raises(RuntimeError, match="No compatible llama KV/memory API"):
        compat.kv_cache_clear("ctx")


def test_missing_symbol_marker_is_honored(monkeypatch):
    def _missing(*args, **kwargs):
        return None

    _missing._llama_missing_symbol = "llama_memory_seq_rm"

    fake = SimpleNamespace(
        llama_get_memory=lambda ctx: "mem",
        llama_memory_clear=lambda mem, data: None,
        llama_memory_seq_rm=_missing,
        llama_memory_seq_add=lambda mem, seq_id, p0, p1, delta: None,
    )
    _set_backend(monkeypatch, fake)

    with pytest.raises(RuntimeError, match="No compatible llama KV/memory API"):
        compat.kv_cache_clear("ctx")
