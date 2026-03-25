"""Zero-NumPy native runtime for Saguaro indexing hot paths."""

from __future__ import annotations

import contextlib
import ctypes
import json
import logging
import os
from array import array

from saguaro.native.loader import (
    core_library_candidates,
    manifest_candidates,
    resolve_core_library,
)

logger = logging.getLogger(__name__)

_FLOAT_SIZE = 4
_FLOAT_P = ctypes.POINTER(ctypes.c_float)
_INT32_P = ctypes.POINTER(ctypes.c_int32)


class NativeRuntimeError(RuntimeError):
    """Raised when the native indexing runtime is unavailable or fails."""


def _raw_ptr_from_buffer(buffer: object) -> tuple[memoryview, ctypes.Array, ctypes.c_void_p]:
    view = memoryview(buffer)
    if view.readonly:
        raise NativeRuntimeError("Native runtime requires a writable projection buffer.")
    byte_view = view.cast("B")
    raw = (ctypes.c_uint8 * len(byte_view)).from_buffer(byte_view)
    return byte_view, raw, ctypes.cast(raw, ctypes.c_void_p)


class NativeIndexRuntime:
    """Thin ctypes ABI for the native indexing hot path."""

    _lib: ctypes.CDLL | None = None
    _instance: NativeIndexRuntime | None = None

    def __new__(cls) -> NativeIndexRuntime:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._lib_path = ""
        self._manifest: dict[str, object] = {}
        self._load_library()
        self._manifest = self._load_manifest()
        self._bind_functions()
        self._initialized = True

    def _find_library(self) -> str:
        resolved = resolve_core_library(prefer_tf_free=True, required=False)
        if resolved is not None:
            return str(resolved)
        searched = [str(candidate) for candidate in core_library_candidates(prefer_tf_free=True)]
        raise NativeRuntimeError(
            "Could not find native Saguaro library via canonical loader. "
            f"Searched: {searched}"
        )

    def _load_library(self) -> None:
        if NativeIndexRuntime._lib is not None:
            return
        lib_path = self._find_library()
        try:
            NativeIndexRuntime._lib = ctypes.CDLL(lib_path)
            self._lib_path = lib_path
        except OSError as exc:
            raise NativeRuntimeError(f"Failed to load {lib_path}: {exc}") from exc

    def _load_manifest(self) -> dict[str, object]:
        for path in manifest_candidates(self._lib_path or None):
            if not os.path.exists(path):
                continue
            try:
                with open(path, encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    payload["_path"] = str(path)
                    return payload
            except Exception:
                continue
        return {}

    def _require_symbol(self, name: str) -> ctypes._CFuncPtr:
        lib = NativeIndexRuntime._lib
        assert lib is not None
        if not hasattr(lib, name):
            raise NativeRuntimeError(
                f"Native ABI mismatch: missing required symbol {name}. Rebuild native libraries."
            )
        return getattr(lib, name)

    def _optional_symbol(self, name: str) -> ctypes._CFuncPtr | None:
        lib = NativeIndexRuntime._lib
        assert lib is not None
        if not hasattr(lib, name):
            return None
        return getattr(lib, name)

    def _bind_functions(self) -> None:
        init_projection = self._require_symbol("saguaro_native_init_projection")
        init_projection.argtypes = [
            _FLOAT_P,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
        ]
        init_projection.restype = None

        full_pipeline = self._require_symbol("saguaro_native_full_pipeline")
        full_pipeline.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            _FLOAT_P,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            _FLOAT_P,
            ctypes.c_int,
        ]
        full_pipeline.restype = ctypes.c_int

        max_threads = self._optional_symbol("saguaro_native_max_threads")
        if max_threads is not None:
            max_threads.argtypes = []
            max_threads.restype = ctypes.c_int
        self._max_threads = max_threads

        trie_create = self._require_symbol("saguaro_native_trie_create")
        trie_create.argtypes = []
        trie_create.restype = ctypes.c_void_p

        trie_destroy = self._require_symbol("saguaro_native_trie_destroy")
        trie_destroy.argtypes = [ctypes.c_void_p]
        trie_destroy.restype = None

        trie_build = self._require_symbol("saguaro_native_trie_build_from_table")
        trie_build.argtypes = [
            ctypes.c_void_p,
            _INT32_P,
            _INT32_P,
            _INT32_P,
            ctypes.c_int,
        ]
        trie_build.restype = None

        match_capture_names = self._optional_symbol("saguaro_native_match_capture_names")
        if match_capture_names is not None:
            match_capture_names.argtypes = [
                _INT32_P,
                _INT32_P,
                _INT32_P,
                ctypes.c_int,
                _INT32_P,
                _INT32_P,
                _INT32_P,
                ctypes.c_int,
                _INT32_P,
            ]
            match_capture_names.restype = ctypes.c_int
        self._match_capture_names = match_capture_names

    @property
    def build_manifest(self) -> dict[str, object]:
        return dict(self._manifest)

    def max_threads(self) -> int:
        if self._max_threads is None:
            return max(1, int(os.cpu_count() or 1))
        try:
            return max(1, int(self._max_threads()))
        except Exception:
            return max(1, int(os.cpu_count() or 1))

    def default_threads(self) -> int:
        env_override = os.getenv("SAGUARO_NATIVE_NUM_THREADS") or os.getenv(
            "OMP_NUM_THREADS"
        )
        if env_override:
            try:
                return max(1, min(int(env_override), self.max_threads()))
            except ValueError:
                pass
        if hasattr(os, "sched_getaffinity"):
            try:
                return max(1, min(len(os.sched_getaffinity(0)), self.max_threads()))
            except Exception:
                pass
        return self.max_threads()

    def init_projection(
        self,
        projection_buffer: object,
        vocab_size: int,
        dim: int,
        seed: int = 42,
    ) -> None:
        expected_bytes = int(vocab_size) * int(dim) * _FLOAT_SIZE
        view, raw, ptr = _raw_ptr_from_buffer(projection_buffer)
        try:
            if len(view) < expected_bytes:
                raise NativeRuntimeError(
                    f"Projection buffer too small: {len(view)} < expected {expected_bytes}"
                )
            lib = NativeIndexRuntime._lib
            assert lib is not None
            lib.saguaro_native_init_projection(
                ctypes.cast(ptr, _FLOAT_P),
                int(vocab_size),
                int(dim),
                ctypes.c_uint64(int(seed)),
            )
            _ = raw
        finally:
            with contextlib.suppress(Exception):
                view.release()

    def create_trie(self) -> ctypes.c_void_p:
        lib = NativeIndexRuntime._lib
        assert lib is not None
        return lib.saguaro_native_trie_create()

    def destroy_trie(self, trie: ctypes.c_void_p | None) -> None:
        if not trie:
            return
        lib = NativeIndexRuntime._lib
        assert lib is not None
        lib.saguaro_native_trie_destroy(trie)

    def build_trie_from_table(
        self,
        trie: ctypes.c_void_p,
        offsets: list[int],
        tokens: list[int],
        superword_ids: list[int],
    ) -> None:
        if not trie:
            raise NativeRuntimeError("Trie handle is required.")
        if not superword_ids:
            return
        offsets_arr = (ctypes.c_int32 * len(offsets))(*offsets)
        tokens_arr = (ctypes.c_int32 * len(tokens))(*tokens)
        ids_arr = (ctypes.c_int32 * len(superword_ids))(*superword_ids)
        lib = NativeIndexRuntime._lib
        assert lib is not None
        lib.saguaro_native_trie_build_from_table(
            trie,
            ctypes.cast(offsets_arr, _INT32_P),
            ctypes.cast(tokens_arr, _INT32_P),
            ctypes.cast(ids_arr, _INT32_P),
            len(superword_ids),
        )

    def full_pipeline(
        self,
        texts: list[str],
        projection_buffer: object,
        vocab_size: int,
        dim: int,
        *,
        max_length: int = 512,
        trie: ctypes.c_void_p | None = None,
        num_threads: int = 0,
    ) -> list[array]:
        if not texts:
            return []

        expected_bytes = int(vocab_size) * int(dim) * _FLOAT_SIZE
        view, raw, projection_ptr = _raw_ptr_from_buffer(projection_buffer)
        output_view: memoryview | None = None
        try:
            if len(view) < expected_bytes:
                raise NativeRuntimeError(
                    f"Projection buffer too small: {len(view)} < expected {expected_bytes}"
                )

            batch_size = len(texts)
            encoded = [text.encode("utf-8") for text in texts]
            text_ptrs = (ctypes.c_char_p * batch_size)(*encoded)
            text_lengths = (ctypes.c_int * batch_size)(*[len(text) for text in encoded])

            output = array("f", [0.0]) * (batch_size * int(dim))
            output_view = memoryview(output)
            output_raw = (ctypes.c_uint8 * len(output_view.cast("B"))).from_buffer(
                output_view.cast("B")
            )

            lib = NativeIndexRuntime._lib
            assert lib is not None
            rc = lib.saguaro_native_full_pipeline(
                text_ptrs,
                text_lengths,
                batch_size,
                ctypes.cast(projection_ptr, _FLOAT_P),
                int(vocab_size),
                int(dim),
                int(max_length),
                trie or None,
                ctypes.cast(output_raw, _FLOAT_P),
                int(num_threads),
            )
            _ = raw
            if rc != 0:
                raise NativeRuntimeError(f"Native full pipeline failed with code {rc}")

            rows: list[array] = []
            for index in range(batch_size):
                start = index * int(dim)
                end = start + int(dim)
                rows.append(output[start:end])
            return rows
        finally:
            if output_view is not None:
                with contextlib.suppress(Exception):
                    output_view.release()
            with contextlib.suppress(Exception):
                view.release()

    def match_capture_names(
        self,
        *,
        def_starts: list[int],
        def_ends: list[int],
        def_type_ids: list[int],
        name_starts: list[int],
        name_ends: list[int],
        name_type_ids: list[int],
    ) -> list[int]:
        def_count = len(def_starts)
        if def_count == 0:
            return []
        if len(def_ends) != def_count or len(def_type_ids) != def_count:
            raise NativeRuntimeError("Definition capture arrays must have identical lengths.")
        name_count = len(name_starts)
        if len(name_ends) != name_count or len(name_type_ids) != name_count:
            raise NativeRuntimeError("Name capture arrays must have identical lengths.")
        if self._match_capture_names is None:
            raise NativeRuntimeError(
                "Native ABI mismatch: missing required symbol "
                "saguaro_native_match_capture_names. Rebuild native libraries."
            )

        def_starts_arr = (ctypes.c_int32 * def_count)(*def_starts)
        def_ends_arr = (ctypes.c_int32 * def_count)(*def_ends)
        def_types_arr = (ctypes.c_int32 * def_count)(*def_type_ids)
        if name_count:
            name_starts_arr = (ctypes.c_int32 * name_count)(*name_starts)
            name_ends_arr = (ctypes.c_int32 * name_count)(*name_ends)
            name_types_arr = (ctypes.c_int32 * name_count)(*name_type_ids)
        else:
            name_starts_arr = ctypes.POINTER(ctypes.c_int32)()
            name_ends_arr = ctypes.POINTER(ctypes.c_int32)()
            name_types_arr = ctypes.POINTER(ctypes.c_int32)()
        output_arr = (ctypes.c_int32 * def_count)()

        rc = self._match_capture_names(
            ctypes.cast(def_starts_arr, _INT32_P),
            ctypes.cast(def_ends_arr, _INT32_P),
            ctypes.cast(def_types_arr, _INT32_P),
            def_count,
            ctypes.cast(name_starts_arr, _INT32_P),
            ctypes.cast(name_ends_arr, _INT32_P),
            ctypes.cast(name_types_arr, _INT32_P),
            name_count,
            ctypes.cast(output_arr, _INT32_P),
        )
        if rc != 0:
            raise NativeRuntimeError(f"Native capture matcher failed with code {rc}")
        return [int(value) for value in output_arr]


_RUNTIME: NativeIndexRuntime | None = None


def get_native_runtime() -> NativeIndexRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = NativeIndexRuntime()
    return _RUNTIME
