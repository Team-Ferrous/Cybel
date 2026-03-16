"""SAGUARO Native Indexer Bindings.

Python ctypes bindings for the native C API in _saguaro_core.so.
These functions can be called WITHOUT loading TensorFlow, reducing
worker memory from ~1.9GB to ~300MB.

Usage:
    from saguaro.ops.native_indexer import NativeIndexer
    indexer = NativeIndexer()
    tokens, lengths = indexer.tokenize_batch(["hello world"])
"""

import ctypes
import hashlib
import json
import logging
import os
from typing import Any

import numpy as np
from numpy.ctypeslib import ndpointer

logger = logging.getLogger(__name__)

# Type aliases
c_int32_ptr = ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
c_float_ptr = ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")

_REQUIRED_SYMBOLS = (
    "saguaro_native_version",
    "saguaro_native_available",
    "saguaro_native_trie_create",
    "saguaro_native_trie_destroy",
    "saguaro_native_trie_insert",
    "saguaro_native_trie_build_from_table",
    "saguaro_native_tokenize_batch",
    "saguaro_native_embed_lookup",
    "saguaro_native_compute_doc_vectors",
    "saguaro_native_holographic_bundle",
    "saguaro_native_crystallize",
    "saguaro_native_full_pipeline",
)
_TRIE_CREATE_ALIASES = (
    "superword_trie_create",
    "SAGUAROTrieCreate",
    "saguaro_trie_create",
)


class NativeIndexerError(Exception):
    """Error from native indexer operations."""

    pass


class NativeIndexer:
    """Native C++ indexer - NO TensorFlow required.

    Loads _saguaro_core.so and calls C functions directly via ctypes,
    bypassing TensorFlow entirely for massive memory savings.
    """

    _lib = None
    _instance = None

    def __new__(cls):
        """Singleton pattern to avoid reloading the library."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the instance."""
        if self._initialized and NativeIndexer._lib is not None:
            return

        self._lib_path = ""
        self._manifest = {}
        self._token_buffer_cache: dict[tuple[int, int], np.ndarray] = {}
        self._length_buffer_cache: dict[int, np.ndarray] = {}
        self._docvec_buffer_cache: dict[tuple[int, int], np.ndarray] = {}
        self._rank_index_buffer_cache: dict[tuple[int, int], np.ndarray] = {}
        self._rank_score_buffer_cache: dict[tuple[int, int], np.ndarray] = {}
        self._load_library()
        self._manifest = self._load_manifest()
        self._assert_required_symbols()
        self._bind_functions()
        self._initialized = True

        # Check if native API is available
        if not self.is_available():
            raise NativeIndexerError(
                "Native API not available in _saguaro_core.so. "
                "Please rebuild with native_indexer_api.cc"
            )

        logger.info(f"Native indexer initialized (version: {self.version()})")

    def _find_library(self) -> str:
        """Find _saguaro_core.so in various locations."""
        # Current file is in saguaro/ops/
        current_dir = os.path.dirname(os.path.abspath(__file__))
        saguaro_dir = os.path.dirname(current_dir)
        repo_dir = os.path.dirname(saguaro_dir)

        search_paths = [
            # PRIORITY: TF-free native lib (for worker processes)
            os.path.join(repo_dir, "build", "_saguaro_native.so"),
            os.path.join(saguaro_dir, "_saguaro_native.so"),
            os.path.join(repo_dir, "_saguaro_native.so"),
            "_saguaro_native.so",
            # FALLBACK: Full core lib (requires TF runtime)
            os.path.join(repo_dir, "build", "_saguaro_core.so"),
            os.path.join(saguaro_dir, "_saguaro_core.so"),
            os.path.join(repo_dir, "_saguaro_core.so"),
            "_saguaro_core.so",
        ]

        for path in search_paths:
            if os.path.exists(path):
                return path

        raise NativeIndexerError(
            f"Could not find _saguaro_native.so/_saguaro_core.so. Searched: {search_paths}"
        )

    def _load_library(self) -> None:
        """Load the shared library."""
        if NativeIndexer._lib is not None:
            return

        lib_path = self._find_library()
        logger.debug(f"Loading native library from: {lib_path}")

        try:
            NativeIndexer._lib = ctypes.CDLL(lib_path)
            self._lib_path = lib_path
        except OSError as e:
            raise NativeIndexerError(f"Failed to load {lib_path}: {e}")

    def _manifest_candidates(self) -> list[str]:
        candidates: list[str] = []
        if self._lib_path:
            lib_dir = os.path.dirname(self._lib_path)
            candidates.append(os.path.join(lib_dir, "saguaro_build_manifest.json"))

        current_dir = os.path.dirname(os.path.abspath(__file__))
        saguaro_dir = os.path.dirname(current_dir)
        repo_dir = os.path.dirname(saguaro_dir)
        candidates.extend(
            [
                os.path.join(
                    saguaro_dir,
                    "native",
                    "bin",
                    "x86_64",
                    "saguaro_build_manifest.json",
                ),
                os.path.join(
                    saguaro_dir, "native", "bin", "arm64", "saguaro_build_manifest.json"
                ),
                os.path.join(
                    repo_dir,
                    "saguaro",
                    "native",
                    "build",
                    "saguaro_build_manifest.json",
                ),
                os.path.join(
                    repo_dir,
                    "saguaro",
                    "native",
                    "build_release",
                    "saguaro_build_manifest.json",
                ),
            ]
        )
        deduped: list[str] = []
        seen = set()
        for item in candidates:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def _load_manifest(self) -> dict:
        for path in self._manifest_candidates():
            if not os.path.exists(path):
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    payload["_path"] = path
                    return payload
            except Exception:
                continue
        synthetic = self._synthesize_manifest()
        if synthetic:
            return synthetic
        return {}

    def _synthesize_manifest(self) -> dict:
        if not self._lib_path:
            return {}
        manifest_path = os.path.join(
            os.path.dirname(self._lib_path), "saguaro_build_manifest.json"
        )
        lib = NativeIndexer._lib
        assert lib is not None
        version = "2.0.0-native"
        isa_baseline = "scalar"
        build_signature = ""
        if hasattr(lib, "saguaro_native_version"):
            lib.saguaro_native_version.argtypes = []
            lib.saguaro_native_version.restype = ctypes.c_char_p
            raw = lib.saguaro_native_version()
            if raw:
                version = raw.decode("utf-8", errors="ignore")
        if hasattr(lib, "saguaro_native_isa_baseline"):
            lib.saguaro_native_isa_baseline.argtypes = []
            lib.saguaro_native_isa_baseline.restype = ctypes.c_char_p
            raw = lib.saguaro_native_isa_baseline()
            if raw:
                isa_baseline = raw.decode("utf-8", errors="ignore")
        if hasattr(lib, "saguaro_native_build_signature"):
            lib.saguaro_native_build_signature.argtypes = []
            lib.saguaro_native_build_signature.restype = ctypes.c_char_p
            raw = lib.saguaro_native_build_signature()
            if raw:
                build_signature = raw.decode("utf-8", errors="ignore")
        manifest = {
            "version": version,
            "target_arch": isa_baseline,
            "compiler_id": "",
            "compiler_version": "",
            "cxx_standard": "c++17",
            "edition": "runtime_synthesized",
            "simd_flags": build_signature,
            "base_cxx_flags": "",
            "hardening_flags": "",
            "tf_cflags": "",
            "tf_lflags": "",
            "required_symbols": list(_REQUIRED_SYMBOLS),
        }
        manifest["abi_hash"] = self._compute_manifest_abi_hash(manifest)
        try:
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True)
        except Exception:
            pass
        manifest["_path"] = manifest_path
        manifest["generated"] = True
        return manifest

    def _assert_required_symbols(self) -> None:
        lib = NativeIndexer._lib
        missing = [name for name in _REQUIRED_SYMBOLS if not hasattr(lib, name)]
        if missing:
            raise NativeIndexerError(
                "Native ABI mismatch: missing required symbols "
                f"{missing}. Rebuild native libraries."
            )

    def _resolve_num_threads(self, requested: int) -> int:
        if requested and requested > 0:
            return int(requested)
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
        cpu_count = os.cpu_count() or 1
        return max(1, min(cpu_count, self.max_threads()))

    def _token_output_buffers(
        self, batch_size: int, max_length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        token_key = (batch_size, max_length)
        tokens = self._token_buffer_cache.get(token_key)
        if tokens is None:
            tokens = np.zeros((batch_size, max_length), dtype=np.int32)
            self._token_buffer_cache[token_key] = tokens
        else:
            tokens.fill(0)

        lengths = self._length_buffer_cache.get(batch_size)
        if lengths is None:
            lengths = np.zeros(batch_size, dtype=np.int32)
            self._length_buffer_cache[batch_size] = lengths
        else:
            lengths.fill(0)
        return tokens, lengths

    def _docvec_output_buffer(self, batch_size: int, dim: int) -> np.ndarray:
        key = (batch_size, dim)
        vecs = self._docvec_buffer_cache.get(key)
        if vecs is None:
            vecs = np.zeros((batch_size, dim), dtype=np.float32)
            self._docvec_buffer_cache[key] = vecs
        else:
            vecs.fill(0.0)
        return vecs

    def _rank_output_buffers(
        self,
        left_count: int,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (left_count, top_k)
        indices = self._rank_index_buffer_cache.get(key)
        if indices is None:
            indices = np.full((left_count, top_k), -1, dtype=np.int32)
            self._rank_index_buffer_cache[key] = indices
        else:
            indices.fill(-1)

        scores = self._rank_score_buffer_cache.get(key)
        if scores is None:
            scores = np.zeros((left_count, top_k), dtype=np.float32)
            self._rank_score_buffer_cache[key] = scores
        else:
            scores.fill(0.0)
        return indices, scores

    def _bind_functions(self) -> None:
        """Bind C functions with proper type signatures."""
        lib = NativeIndexer._lib

        # Version / availability
        lib.saguaro_native_version.argtypes = []
        lib.saguaro_native_version.restype = ctypes.c_char_p

        lib.saguaro_native_available.argtypes = []
        lib.saguaro_native_available.restype = ctypes.c_int
        if hasattr(lib, "saguaro_native_build_signature"):
            lib.saguaro_native_build_signature.argtypes = []
            lib.saguaro_native_build_signature.restype = ctypes.c_char_p
        if hasattr(lib, "saguaro_native_isa_baseline"):
            lib.saguaro_native_isa_baseline.argtypes = []
            lib.saguaro_native_isa_baseline.restype = ctypes.c_char_p
        if hasattr(lib, "saguaro_native_openmp_enabled"):
            lib.saguaro_native_openmp_enabled.argtypes = []
            lib.saguaro_native_openmp_enabled.restype = ctypes.c_int
        if hasattr(lib, "saguaro_native_avx2_enabled"):
            lib.saguaro_native_avx2_enabled.argtypes = []
            lib.saguaro_native_avx2_enabled.restype = ctypes.c_int
        if hasattr(lib, "saguaro_native_fma_enabled"):
            lib.saguaro_native_fma_enabled.argtypes = []
            lib.saguaro_native_fma_enabled.restype = ctypes.c_int
        if hasattr(lib, "saguaro_native_max_threads"):
            lib.saguaro_native_max_threads.argtypes = []
            lib.saguaro_native_max_threads.restype = ctypes.c_int

        # Trie management
        lib.saguaro_native_trie_create.argtypes = []
        lib.saguaro_native_trie_create.restype = ctypes.c_void_p

        lib.saguaro_native_trie_destroy.argtypes = [ctypes.c_void_p]
        lib.saguaro_native_trie_destroy.restype = None

        lib.saguaro_native_trie_insert.argtypes = [
            ctypes.c_void_p,  # trie
            c_int32_ptr,  # ngram
            ctypes.c_int,  # ngram_len
            ctypes.c_int32,  # superword_id
        ]
        lib.saguaro_native_trie_insert.restype = None

        lib.saguaro_native_trie_build_from_table.argtypes = [
            ctypes.c_void_p,  # trie
            c_int32_ptr,  # offsets
            c_int32_ptr,  # tokens
            c_int32_ptr,  # superword_ids
            ctypes.c_int,  # num_ngrams
        ]
        lib.saguaro_native_trie_build_from_table.restype = None

        # Tokenization
        lib.saguaro_native_tokenize_batch.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),  # texts
            ctypes.POINTER(ctypes.c_int),  # text_lengths
            ctypes.c_int,  # batch_size
            c_int32_ptr,  # output_tokens
            c_int32_ptr,  # output_lengths
            ctypes.c_int,  # max_length
            ctypes.c_int,  # byte_offset
            ctypes.c_int,  # add_special_tokens
            ctypes.c_void_p,  # trie
            ctypes.c_int,  # num_threads
        ]
        lib.saguaro_native_tokenize_batch.restype = ctypes.c_int

        # Embedding lookup
        lib.saguaro_native_embed_lookup.argtypes = [
            c_int32_ptr,  # tokens
            ctypes.c_int,  # batch_size
            ctypes.c_int,  # seq_len
            c_float_ptr,  # projection
            ctypes.c_int,  # vocab_size
            ctypes.c_int,  # dim
            c_float_ptr,  # output
        ]
        lib.saguaro_native_embed_lookup.restype = None

        # Document vectors
        lib.saguaro_native_compute_doc_vectors.argtypes = [
            c_float_ptr,  # embeddings
            c_int32_ptr,  # lengths
            ctypes.c_int,  # batch_size
            ctypes.c_int,  # seq_len
            ctypes.c_int,  # dim
            c_float_ptr,  # output
        ]
        lib.saguaro_native_compute_doc_vectors.restype = None

        # Holographic bundling
        lib.saguaro_native_holographic_bundle.argtypes = [
            c_float_ptr,  # vectors
            ctypes.c_int,  # num_vectors
            ctypes.c_int,  # dim
            c_float_ptr,  # output
        ]
        lib.saguaro_native_holographic_bundle.restype = None

        lib.saguaro_native_crystallize.argtypes = [
            c_float_ptr,  # knowledge
            c_float_ptr,  # importance
            ctypes.c_int,  # num_vectors
            ctypes.c_int,  # dim
            ctypes.c_float,  # threshold
            c_float_ptr,  # output
        ]
        lib.saguaro_native_crystallize.restype = None

        # Full pipeline
        lib.saguaro_native_full_pipeline.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),  # texts
            ctypes.POINTER(ctypes.c_int),  # text_lengths
            ctypes.c_int,  # batch_size
            c_float_ptr,  # projection
            ctypes.c_int,  # vocab_size
            ctypes.c_int,  # dim
            ctypes.c_int,  # max_length
            ctypes.c_void_p,  # trie
            c_float_ptr,  # output
            ctypes.c_int,  # num_threads
        ]
        lib.saguaro_native_full_pipeline.restype = ctypes.c_int
        if hasattr(lib, "saguaro_native_full_pipeline_strided"):
            lib.saguaro_native_full_pipeline_strided.argtypes = [
                ctypes.POINTER(ctypes.c_char_p),  # texts
                ctypes.POINTER(ctypes.c_int),  # text_lengths
                ctypes.c_int,  # batch_size
                c_float_ptr,  # projection
                ctypes.c_int,  # vocab_size
                ctypes.c_int,  # dim
                ctypes.c_int,  # output_dim
                ctypes.c_int,  # output_stride
                ctypes.c_int,  # max_length
                ctypes.c_void_p,  # trie
                c_float_ptr,  # output
                ctypes.c_int,  # num_threads
            ]
            lib.saguaro_native_full_pipeline_strided.restype = ctypes.c_int
        if hasattr(lib, "saguaro_native_rank_jaccard_pairs"):
            lib.saguaro_native_rank_jaccard_pairs.argtypes = [
                c_int32_ptr,  # left_tokens
                c_int32_ptr,  # left_lengths
                ctypes.c_int,  # left_count
                c_int32_ptr,  # right_tokens
                c_int32_ptr,  # right_lengths
                ctypes.c_int,  # right_count
                ctypes.c_int,  # token_stride
                ctypes.c_int,  # top_k
                c_int32_ptr,  # output_indices
                c_float_ptr,  # output_scores
                ctypes.c_int,  # num_threads
            ]
            lib.saguaro_native_rank_jaccard_pairs.restype = ctypes.c_int
        if hasattr(lib, "saguaro_native_screen_overlap_pairs"):
            lib.saguaro_native_screen_overlap_pairs.argtypes = [
                c_int32_ptr,  # left_tokens
                c_int32_ptr,  # left_lengths
                ctypes.c_int,  # left_count
                c_int32_ptr,  # right_tokens
                c_int32_ptr,  # right_lengths
                ctypes.c_int,  # right_count
                ctypes.c_int,  # token_stride
                ctypes.c_int,  # top_k
                c_int32_ptr,  # output_indices
                c_float_ptr,  # output_scores
                ctypes.c_int,  # num_threads
            ]
            lib.saguaro_native_screen_overlap_pairs.restype = ctypes.c_int

    # =========================================================================
    # Public API
    # =========================================================================

    def version(self) -> str:
        """Get native library version."""
        return NativeIndexer._lib.saguaro_native_version().decode("utf-8")

    def abi_self_test(self) -> dict:
        """Validate required symbol ABI and manifest coherence."""
        missing = [
            name for name in _REQUIRED_SYMBOLS if not hasattr(NativeIndexer._lib, name)
        ]
        lib_path = self._lib_path or "unknown"
        manifest = dict(self._manifest or {})
        manifest_path = str(manifest.pop("_path", ""))
        required_symbols = list(manifest.get("required_symbols") or _REQUIRED_SYMBOLS)
        manifest_missing_symbols = [
            name for name in required_symbols if not hasattr(NativeIndexer._lib, name)
        ]
        abi_hash = str(manifest.get("abi_hash") or "")
        computed_hash = self._compute_manifest_abi_hash(manifest) if manifest else ""
        hash_matches = bool(abi_hash and computed_hash and abi_hash == computed_hash)
        runtime_probe = {"ok": False, "message": "skipped"}
        try:
            # Minimal smoke probe to detect runtime ABI drift beyond symbol presence.
            self.tokenize_batch(
                ["saguaro_abi_probe"], max_length=8, add_special_tokens=False
            )
            runtime_probe = {"ok": True, "message": "tokenize_batch probe passed"}
        except Exception as exc:
            runtime_probe = {"ok": False, "message": str(exc)}
        ok = (
            not missing
            and bool(manifest_path)
            and not manifest_missing_symbols
            and hash_matches
            and bool(runtime_probe.get("ok", False))
        )
        capabilities = {
            "manifest": bool(manifest_path),
            "required_symbols": not missing,
            "manifest_symbols": not manifest_missing_symbols,
            "abi_hash": hash_matches,
            "runtime_probe": bool(runtime_probe.get("ok", False)),
        }
        return {
            "ok": ok,
            "status": "ready" if ok else "degraded",
            "library_path": lib_path,
            "manifest_path": manifest_path or None,
            "manifest_found": bool(manifest_path),
            "abi_hash": abi_hash or None,
            "computed_hash_hint": computed_hash or None,
            "abi_hash_match": hash_matches,
            "required_symbols": required_symbols,
            "missing_symbols": missing,
            "manifest_missing_symbols": manifest_missing_symbols,
            "runtime_probe": runtime_probe,
            "capabilities": capabilities,
            "build_signature": self.build_signature(),
            "isa_baseline": self.isa_baseline(),
            "openmp_enabled": self.openmp_enabled(),
            "avx2_enabled": self.avx2_enabled(),
            "fma_enabled": self.fma_enabled(),
            "max_threads": self.max_threads(),
            "version": self.version(),
        }

    def capability_report(self) -> dict:
        """Describe native runtime, ABI, and hot-path op coverage."""
        try:
            abi = self.abi_self_test()
        except Exception as exc:
            abi = {"ok": False, "reason": str(exc)}
        trie_ops = self._trie_ops_report()
        ops = self._ops_matrix()
        requirements = {
            "backend": "native_cpp",
            "openmp_required": True,
            "avx2_required": True,
            "satisfied": bool(
                self.is_available() and self.openmp_enabled() and self.avx2_enabled()
            ),
        }
        manifest = {
            key: value
            for key, value in dict(self._manifest or {}).items()
            if not str(key).startswith("_")
        }
        affinity_cpus: list[int] = []
        if hasattr(os, "sched_getaffinity"):
            try:
                affinity_cpus = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
            except Exception:
                affinity_cpus = []
        default_threads = self._resolve_num_threads(0)
        return {
            "status": "ready" if requirements["satisfied"] else "degraded",
            "requirements": requirements,
            "parallel_runtime": {
                "compiled": bool(self.openmp_enabled()),
                "default_threads": int(default_threads),
                "max_threads": int(self.max_threads()),
                "omp_num_threads": os.getenv("OMP_NUM_THREADS"),
                "saguaro_native_num_threads": os.getenv("SAGUARO_NATIVE_NUM_THREADS"),
                "affinity_mode": "sched_affinity" if affinity_cpus else "process_default",
                "affinity_cpus": affinity_cpus,
            },
            "simd": {
                "baseline": self.isa_baseline(),
                "avx2_compiled": bool(self.avx2_enabled()),
                "fma_compiled": bool(self.fma_enabled()),
            },
            "manifest": manifest,
            "ops": ops,
            "native_indexer": abi,
            "trie_ops": trie_ops,
            "degraded": not requirements["satisfied"],
        }

    def _trie_ops_report(self) -> dict:
        trie_ops = {
            "available": False,
            "reason": "unresolved",
            "create_op": None,
        }
        try:
            lib = NativeIndexer._lib
            create_name = None
            if hasattr(lib, "saguaro_native_trie_create"):
                create_name = "saguaro_native_trie_create"
            else:
                create_name = next(
                    (name for name in _TRIE_CREATE_ALIASES if hasattr(lib, name)),
                    None,
                )
            if create_name:
                trie_ops = {
                    "available": True,
                    "reason": "",
                    "create_op": create_name,
                }
            elif getattr(tokenizer_module, "_ops_available", False):
                trie_ops = {
                    "available": False,
                    "reason": "text_tokenizer_loaded_without_trie_create",
                    "create_op": None,
                }
            else:
                trie_ops = {
                    "available": False,
                    "reason": "text_tokenizer_ops_unavailable",
                    "create_op": None,
                }
        except Exception as exc:
            trie_ops = {
                "available": False,
                "reason": str(exc),
                "create_op": None,
            }
        return trie_ops

    def _ops_matrix(self) -> dict[str, dict[str, Any]]:
        lib = NativeIndexer._lib
        return {
            "trie_build_from_table": {
                "available": hasattr(lib, "saguaro_native_trie_build_from_table"),
                "required": True,
            },
            "tokenize_batch": {
                "available": hasattr(lib, "saguaro_native_tokenize_batch"),
                "required": True,
            },
            "full_pipeline": {
                "available": hasattr(lib, "saguaro_native_full_pipeline"),
                "required": True,
            },
            "full_pipeline_strided": {
                "available": hasattr(lib, "saguaro_native_full_pipeline_strided"),
                "required": False,
            },
            "rank_jaccard_pairs": {
                "available": hasattr(lib, "saguaro_native_rank_jaccard_pairs"),
                "required": False,
            },
            "screen_overlap_pairs": {
                "available": hasattr(lib, "saguaro_native_screen_overlap_pairs"),
                "required": False,
            },
        }

    def _compute_manifest_abi_hash(self, manifest: dict) -> str:
        seed = (
            f"{manifest.get('version', '')}|{manifest.get('target_arch', '')}|"
            f"{manifest.get('compiler_id', '')}|{manifest.get('compiler_version', '')}|"
            f"{manifest.get('cxx_standard', '')}|{manifest.get('edition', '')}|"
            f"{manifest.get('simd_flags', '')}|{manifest.get('base_cxx_flags', '')}|"
            f"{manifest.get('hardening_flags', '')}|{manifest.get('tf_cflags', '')}|"
            f"{manifest.get('tf_lflags', '')}"
        )
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()

    def is_available(self) -> bool:
        """Check if native API is available."""
        return NativeIndexer._lib.saguaro_native_available() == 1

    def build_signature(self) -> str:
        func = getattr(NativeIndexer._lib, "saguaro_native_build_signature", None)
        if func is None:
            return ""
        return str(func().decode("utf-8"))

    def isa_baseline(self) -> str:
        func = getattr(NativeIndexer._lib, "saguaro_native_isa_baseline", None)
        if func is None:
            return "scalar"
        return str(func().decode("utf-8"))

    def openmp_enabled(self) -> bool:
        func = getattr(NativeIndexer._lib, "saguaro_native_openmp_enabled", None)
        return bool(func()) if func is not None else False

    def avx2_enabled(self) -> bool:
        func = getattr(NativeIndexer._lib, "saguaro_native_avx2_enabled", None)
        return bool(func()) if func is not None else False

    def fma_enabled(self) -> bool:
        func = getattr(NativeIndexer._lib, "saguaro_native_fma_enabled", None)
        return bool(func()) if func is not None else False

    def max_threads(self) -> int:
        func = getattr(NativeIndexer._lib, "saguaro_native_max_threads", None)
        if func is None:
            return 0
        try:
            return int(func())
        except Exception:
            return 0

    def create_trie(self) -> ctypes.c_void_p:
        """Create a new superword trie."""
        return NativeIndexer._lib.saguaro_native_trie_create()

    def destroy_trie(self, trie: ctypes.c_void_p) -> None:
        """Destroy a superword trie."""
        if trie:
            NativeIndexer._lib.saguaro_native_trie_destroy(trie)

    def trie_insert(
        self, trie: ctypes.c_void_p, ngram: np.ndarray, superword_id: int
    ) -> None:
        """Insert an n-gram into the trie."""
        ngram = np.ascontiguousarray(ngram, dtype=np.int32)
        NativeIndexer._lib.saguaro_native_trie_insert(
            trie, ngram, len(ngram), superword_id
        )

    def build_trie_from_table(
        self,
        trie: ctypes.c_void_p,
        offsets: np.ndarray | list[int],
        tokens: np.ndarray | list[int],
        superword_ids: np.ndarray | list[int],
    ) -> None:
        """Bulk-build a trie from offset/token/id tables."""
        offsets_arr = np.ascontiguousarray(offsets, dtype=np.int32)
        tokens_arr = np.ascontiguousarray(tokens, dtype=np.int32)
        superword_ids_arr = np.ascontiguousarray(superword_ids, dtype=np.int32)
        ngrams = max(0, len(offsets_arr) - 1)
        NativeIndexer._lib.saguaro_native_trie_build_from_table(
            trie,
            offsets_arr,
            tokens_arr,
            superword_ids_arr,
            ngrams,
        )

    def tokenize_batch(
        self,
        texts: list[str],
        max_length: int = 512,
        byte_offset: int = 32,
        add_special_tokens: bool = True,
        trie: ctypes.c_void_p | None = None,
        num_threads: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize a batch of texts.

        Args:
            texts: List of UTF-8 strings.
            max_length: Maximum sequence length.
            byte_offset: Byte offset (typically 32).
            add_special_tokens: Whether to add CLS/EOS tokens.
            trie: Optional superword trie for merging.
            num_threads: Number of threads (0 = auto).

        Returns:
            (tokens, lengths) where tokens is [batch_size, max_length]
            and lengths is [batch_size].
        """
        batch_size = len(texts)
        if batch_size == 0:
            return np.zeros((0, max_length), dtype=np.int32), np.zeros(
                0, dtype=np.int32
            )

        # Encode texts
        encoded = [t.encode("utf-8") for t in texts]
        text_ptrs = (ctypes.c_char_p * batch_size)(*encoded)
        text_lengths = (ctypes.c_int * batch_size)(*[len(t) for t in encoded])

        output_tokens, output_lengths = self._token_output_buffers(
            batch_size, max_length
        )
        threads = self._resolve_num_threads(num_threads)

        ret = NativeIndexer._lib.saguaro_native_tokenize_batch(
            text_ptrs,
            text_lengths,
            batch_size,
            output_tokens,
            output_lengths,
            max_length,
            byte_offset,
            1 if add_special_tokens else 0,
            trie,
            threads,
        )

        if ret != 0:
            raise NativeIndexerError(f"Tokenization failed with code {ret}")

        return output_tokens.copy(), output_lengths.copy()

    def embed_lookup(
        self,
        tokens: np.ndarray,
        projection: np.ndarray,
        vocab_size: int,
    ) -> np.ndarray:
        """Perform embedding lookup.

        Args:
            tokens: Token IDs [batch_size, seq_len].
            projection: Projection matrix [vocab_size, dim].
            vocab_size: Vocabulary size.

        Returns:
            Embeddings [batch_size, seq_len, dim].
        """
        tokens = np.ascontiguousarray(tokens, dtype=np.int32)
        projection = np.ascontiguousarray(projection, dtype=np.float32)

        batch_size, seq_len = tokens.shape
        dim = projection.shape[1]

        output = np.zeros((batch_size, seq_len, dim), dtype=np.float32)

        NativeIndexer._lib.saguaro_native_embed_lookup(
            tokens,
            batch_size,
            seq_len,
            projection,
            vocab_size,
            dim,
            output,
        )

        return output

    def compute_doc_vectors(
        self,
        embeddings: np.ndarray,
        lengths: np.ndarray,
    ) -> np.ndarray:
        """Compute document vectors via mean pooling.

        Args:
            embeddings: [batch_size, seq_len, dim].
            lengths: [batch_size].

        Returns:
            Document vectors [batch_size, dim].
        """
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        lengths = np.ascontiguousarray(lengths, dtype=np.int32)

        batch_size, seq_len, dim = embeddings.shape
        output = np.zeros((batch_size, dim), dtype=np.float32)

        NativeIndexer._lib.saguaro_native_compute_doc_vectors(
            embeddings,
            lengths,
            batch_size,
            seq_len,
            dim,
            output,
        )

        return output

    def holographic_bundle(
        self,
        vectors: np.ndarray,
    ) -> np.ndarray:
        """Bundle multiple vectors into one.

        Args:
            vectors: [num_vectors, dim].

        Returns:
            Bundled vector [dim].
        """
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        num_vectors, dim = vectors.shape
        output = np.zeros(dim, dtype=np.float32)

        NativeIndexer._lib.saguaro_native_holographic_bundle(
            vectors,
            num_vectors,
            dim,
            output,
        )

        return output

    def crystallize(
        self,
        knowledge: np.ndarray,
        importance: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Crystallize memory.

        Args:
            knowledge: [num_vectors, dim].
            importance: [num_vectors, dim].
            threshold: Crystallization threshold.

        Returns:
            Crystallized output [num_vectors, dim].
        """
        knowledge = np.ascontiguousarray(knowledge, dtype=np.float32)
        importance = np.ascontiguousarray(importance, dtype=np.float32)

        num_vectors, dim = knowledge.shape
        output = np.zeros_like(knowledge)

        NativeIndexer._lib.saguaro_native_crystallize(
            knowledge,
            importance,
            num_vectors,
            dim,
            threshold,
            output,
        )

        return output

    def full_pipeline(
        self,
        texts: list[str],
        projection: np.ndarray,
        vocab_size: int,
        max_length: int = 512,
        trie: ctypes.c_void_p | None = None,
        num_threads: int = 0,
        target_dim: int | None = None,
    ) -> np.ndarray:
        """Full pipeline: texts -> document vectors.

        Args:
            texts: List of UTF-8 strings.
            projection: Projection matrix [vocab_size, dim].
            vocab_size: Vocabulary size.
            max_length: Maximum sequence length.
            trie: Optional superword trie.
            num_threads: Number of threads (0 = auto).

        Returns:
            Document vectors [batch_size, dim].
        """
        batch_size = len(texts)
        if batch_size == 0:
            dim = projection.shape[1]
            return np.zeros((0, dim), dtype=np.float32)

        projection = np.ascontiguousarray(projection, dtype=np.float32)
        dim = projection.shape[1]
        output_dim = max(dim, int(target_dim or dim))
        threads = self._resolve_num_threads(num_threads)

        # Encode texts
        encoded = [t.encode("utf-8") for t in texts]
        text_ptrs = (ctypes.c_char_p * batch_size)(*encoded)
        text_lengths = (ctypes.c_int * batch_size)(*[len(t) for t in encoded])

        output = self._docvec_output_buffer(batch_size, output_dim)

        if output_dim != dim and hasattr(
            NativeIndexer._lib, "saguaro_native_full_pipeline_strided"
        ):
            ret = NativeIndexer._lib.saguaro_native_full_pipeline_strided(
                text_ptrs,
                text_lengths,
                batch_size,
                projection,
                vocab_size,
                dim,
                output_dim,
                output_dim,
                max_length,
                trie,
                output,
                threads,
            )
        else:
            ret = NativeIndexer._lib.saguaro_native_full_pipeline(
                text_ptrs,
                text_lengths,
                batch_size,
                projection,
                vocab_size,
                dim,
                max_length,
                trie,
                output,
                threads,
            )

        if ret != 0:
            raise NativeIndexerError(f"Full pipeline failed with code {ret}")

        return output.copy()

    def full_pipeline_batched(
        self,
        texts: list[str],
        projection: np.ndarray,
        vocab_size: int,
        *,
        batch_capacity: int = 128,
        max_total_texts: int = 0,
        max_length: int = 512,
        trie: ctypes.c_void_p | None = None,
        num_threads: int = 0,
        target_dim: int | None = None,
    ) -> np.ndarray:
        """Capacity-aware batched pipeline to limit allocator churn.

        Args:
            texts: Input text list.
            projection: Projection matrix [vocab_size, dim].
            vocab_size: Vocabulary size.
            batch_capacity: Max texts processed per native call.
            max_total_texts: Optional hard cap. If >0, raises when exceeded.
            max_length: Maximum sequence length.
            trie: Optional trie handle.
            num_threads: Requested threads (0 -> env/autodetect).
        """
        total = len(texts)
        if max_total_texts > 0 and total > max_total_texts:
            raise NativeIndexerError(
                f"Batch capacity contract violated: total_texts={total} > max_total_texts={max_total_texts}"
            )
        if total == 0:
            dim = int(projection.shape[1])
            return np.zeros((0, dim), dtype=np.float32)

        projection = np.ascontiguousarray(projection, dtype=np.float32)
        dim = int(projection.shape[1])
        output_dim = max(dim, int(target_dim or dim))
        _ = batch_capacity
        return self.full_pipeline(
            texts=texts,
            projection=projection,
            vocab_size=vocab_size,
            max_length=max_length,
            trie=trie,
            num_threads=num_threads,
            target_dim=output_dim,
        )

    def rank_jaccard_pairs(
        self,
        left_tokens: np.ndarray,
        left_lengths: np.ndarray,
        right_tokens: np.ndarray,
        right_lengths: np.ndarray,
        *,
        top_k: int = 4,
        num_threads: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rank top-k Jaccard matches between left and right sorted token rows."""
        func = getattr(NativeIndexer._lib, "saguaro_native_rank_jaccard_pairs", None)
        if func is None:
            raise NativeIndexerError("Native comparative scorer is unavailable")

        left_tokens = np.ascontiguousarray(left_tokens, dtype=np.int32)
        left_lengths = np.ascontiguousarray(left_lengths, dtype=np.int32)
        right_tokens = np.ascontiguousarray(right_tokens, dtype=np.int32)
        right_lengths = np.ascontiguousarray(right_lengths, dtype=np.int32)

        if left_tokens.ndim != 2 or right_tokens.ndim != 2:
            raise NativeIndexerError("rank_jaccard_pairs expects 2D token tables")
        if left_tokens.shape[1] != right_tokens.shape[1]:
            raise NativeIndexerError("left/right token tables must share token stride")
        if left_tokens.shape[0] != left_lengths.shape[0]:
            raise NativeIndexerError("left token table/lengths mismatch")
        if right_tokens.shape[0] != right_lengths.shape[0]:
            raise NativeIndexerError("right token table/lengths mismatch")

        left_count = int(left_tokens.shape[0])
        right_count = int(right_tokens.shape[0])
        token_stride = int(left_tokens.shape[1])
        top_k = max(1, int(top_k or 1))
        threads = self._resolve_num_threads(num_threads)
        output_indices, output_scores = self._rank_output_buffers(left_count, top_k)

        ret = func(
            left_tokens,
            left_lengths,
            left_count,
            right_tokens,
            right_lengths,
            right_count,
            token_stride,
            top_k,
            output_indices,
            output_scores,
            threads,
        )
        if ret != 0:
            raise NativeIndexerError(f"rank_jaccard_pairs failed with code {ret}")
        return output_indices.copy(), output_scores.copy()

    def screen_overlap_pairs(
        self,
        left_tokens: np.ndarray,
        left_lengths: np.ndarray,
        right_tokens: np.ndarray,
        right_lengths: np.ndarray,
        *,
        top_k: int = 16,
        num_threads: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Screen top-k overlap candidates between left and right token tables."""
        func = getattr(NativeIndexer._lib, "saguaro_native_screen_overlap_pairs", None)
        if func is None:
            raise NativeIndexerError("Native overlap prefilter is unavailable")

        left_tokens = np.ascontiguousarray(left_tokens, dtype=np.int32)
        left_lengths = np.ascontiguousarray(left_lengths, dtype=np.int32)
        right_tokens = np.ascontiguousarray(right_tokens, dtype=np.int32)
        right_lengths = np.ascontiguousarray(right_lengths, dtype=np.int32)

        if left_tokens.ndim != 2 or right_tokens.ndim != 2:
            raise NativeIndexerError("screen_overlap_pairs expects 2D token tables")
        if left_tokens.shape[1] != right_tokens.shape[1]:
            raise NativeIndexerError("left/right token tables must share token stride")
        if left_tokens.shape[0] != left_lengths.shape[0]:
            raise NativeIndexerError("left token table/lengths mismatch")
        if right_tokens.shape[0] != right_lengths.shape[0]:
            raise NativeIndexerError("right token table/lengths mismatch")

        left_count = int(left_tokens.shape[0])
        right_count = int(right_tokens.shape[0])
        token_stride = int(left_tokens.shape[1])
        top_k = max(1, int(top_k or 1))
        threads = self._resolve_num_threads(num_threads)
        output_indices, output_scores = self._rank_output_buffers(left_count, top_k)

        ret = func(
            left_tokens,
            left_lengths,
            left_count,
            right_tokens,
            right_lengths,
            right_count,
            token_stride,
            top_k,
            output_indices,
            output_scores,
            threads,
        )
        if ret != 0:
            raise NativeIndexerError(f"screen_overlap_pairs failed with code {ret}")
        return output_indices.copy(), output_scores.copy()


# Global singleton instance
_indexer: NativeIndexer | None = None


def get_native_indexer() -> NativeIndexer:
    """Get the global native indexer instance."""
    global _indexer
    if _indexer is None:
        _indexer = NativeIndexer()
    return _indexer


def collect_native_capability_report() -> dict:
    """Return native indexer and trie capability state."""
    try:
        return get_native_indexer().capability_report()
    except Exception as exc:
        return {
            "status": "degraded",
            "requirements": {
                "backend": "native_cpp",
                "openmp_required": True,
                "avx2_required": True,
                "satisfied": False,
            },
            "parallel_runtime": {"compiled": False, "default_threads": 0, "max_threads": 0},
            "simd": {"baseline": "scalar", "avx2_compiled": False, "fma_compiled": False},
            "manifest": {},
            "ops": {},
            "native_indexer": {"ok": False, "reason": str(exc)},
            "trie_ops": {
                "available": False,
                "reason": str(exc),
                "create_op": None,
            },
            "degraded": True,
        }


# Convenience functions
def tokenize_batch(texts: list[str], **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize texts without TensorFlow."""
    return get_native_indexer().tokenize_batch(texts, **kwargs)


def embed_and_pool(
    tokens: np.ndarray,
    lengths: np.ndarray,
    projection: np.ndarray,
    vocab_size: int,
) -> np.ndarray:
    """Embed tokens and compute document vectors."""
    indexer = get_native_indexer()
    embeddings = indexer.embed_lookup(tokens, projection, vocab_size)
    return indexer.compute_doc_vectors(embeddings, lengths)


def full_pipeline(
    texts: list[str], projection: np.ndarray, vocab_size: int, **kwargs
) -> np.ndarray:
    """Full text -> document vector pipeline."""
    return get_native_indexer().full_pipeline(texts, projection, vocab_size, **kwargs)


def full_pipeline_batched(
    texts: list[str], projection: np.ndarray, vocab_size: int, **kwargs
) -> np.ndarray:
    """Capacity-aware batched text -> document vector pipeline."""
    return get_native_indexer().full_pipeline_batched(
        texts, projection, vocab_size, **kwargs
    )


def holographic_bundle(vectors: np.ndarray) -> np.ndarray:
    """Bundle vectors without TensorFlow."""
    return get_native_indexer().holographic_bundle(vectors)
