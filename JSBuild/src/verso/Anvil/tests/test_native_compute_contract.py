from __future__ import annotations

import numpy as np

from saguaro.indexing.native_indexer_bindings import NativeIndexer


class _FakeNativeLib:
    def saguaro_native_available(self) -> int:
        return 1

    def saguaro_native_version(self) -> bytes:
        return b"2.1.0-test"

    def saguaro_native_build_signature(self) -> bytes:
        return b"compiler=gcc;openmp=1;avx2=1;fma=1"

    def saguaro_native_isa_baseline(self) -> bytes:
        return b"avx2"

    def saguaro_native_openmp_enabled(self) -> int:
        return 1

    def saguaro_native_avx2_enabled(self) -> int:
        return 1

    def saguaro_native_fma_enabled(self) -> int:
        return 1

    def saguaro_native_max_threads(self) -> int:
        return 16


class _FakeRankJaccardPairs:
    def __call__(
        self,
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
        num_threads,
    ) -> int:
        assert left_count == 2
        assert right_count == 2
        assert token_stride == 3
        assert top_k == 1
        output_indices[0, 0] = 0
        output_scores[0, 0] = 1.0
        output_indices[1, 0] = 1
        output_scores[1, 0] = 0.5
        return 0


class _FakeScreenOverlapPairs:
    def __call__(
        self,
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
        num_threads,
    ) -> int:
        assert left_count == 2
        assert right_count == 3
        assert token_stride == 3
        assert top_k == 2
        output_indices[0, 0] = 1
        output_scores[0, 0] = 1.0
        output_indices[0, 1] = 0
        output_scores[0, 1] = 0.5
        output_indices[1, 0] = 2
        output_scores[1, 0] = 1.0
        return 0


def test_native_indexer_capability_report_surfaces_openmp_and_avx2(monkeypatch) -> None:
    indexer = NativeIndexer.__new__(NativeIndexer)
    indexer._initialized = True
    indexer._lib_path = "/tmp/_saguaro_core.so"
    indexer._manifest = {
        "_path": "/tmp/saguaro_build_manifest.json",
        "abi_hash": "abc123",
        "compiler_id": "GNU",
        "compiler_version": "13.2",
        "simd_flags": "-mavx2 -mfma",
        "base_cxx_flags": "-O3 -march=native",
    }

    monkeypatch.setattr(NativeIndexer, "_instance", indexer, raising=False)
    monkeypatch.setattr(NativeIndexer, "_lib", _FakeNativeLib(), raising=False)

    report = indexer.capability_report()

    assert report["status"] == "ready"
    assert report["requirements"]["satisfied"] is True
    assert report["parallel_runtime"]["compiled"] is True
    assert report["parallel_runtime"]["max_threads"] == 16
    assert report["simd"]["baseline"] == "avx2"
    assert report["simd"]["avx2_compiled"] is True
    assert report["manifest"]["simd_flags"] == "-mavx2 -mfma"


def test_native_indexer_rank_jaccard_pairs_uses_native_kernel(monkeypatch) -> None:
    indexer = NativeIndexer.__new__(NativeIndexer)
    indexer._initialized = True
    indexer._rank_index_buffer_cache = {}
    indexer._rank_score_buffer_cache = {}
    monkeypatch.setattr(indexer, "_resolve_num_threads", lambda requested: 4)

    fake_lib = _FakeNativeLib()
    fake_lib.saguaro_native_rank_jaccard_pairs = _FakeRankJaccardPairs()
    monkeypatch.setattr(NativeIndexer, "_lib", fake_lib, raising=False)

    indices, scores = indexer.rank_jaccard_pairs(
        np.array([[1, 2, 3], [4, 5, 0]], dtype=np.int32),
        np.array([3, 2], dtype=np.int32),
        np.array([[1, 2, 3], [4, 6, 0]], dtype=np.int32),
        np.array([3, 2], dtype=np.int32),
        top_k=1,
    )

    assert indices.tolist() == [[0], [1]]
    assert scores.tolist() == [[1.0], [0.5]]


def test_native_indexer_screen_overlap_pairs_uses_native_kernel(monkeypatch) -> None:
    indexer = NativeIndexer.__new__(NativeIndexer)
    indexer._initialized = True
    indexer._rank_index_buffer_cache = {}
    indexer._rank_score_buffer_cache = {}
    monkeypatch.setattr(indexer, "_resolve_num_threads", lambda requested: 4)

    fake_lib = _FakeNativeLib()
    fake_lib.saguaro_native_screen_overlap_pairs = _FakeScreenOverlapPairs()
    monkeypatch.setattr(NativeIndexer, "_lib", fake_lib, raising=False)

    indices, scores = indexer.screen_overlap_pairs(
        np.array([[1, 2, 0], [7, 8, 0]], dtype=np.int32),
        np.array([2, 2], dtype=np.int32),
        np.array([[2, 9, 0], [1, 2, 0], [7, 8, 0]], dtype=np.int32),
        np.array([2, 2, 2], dtype=np.int32),
        top_k=2,
    )

    assert indices.tolist() == [[1, 0], [2, -1]]
    assert scores.tolist() == [[1.0, 0.5], [1.0, 0.0]]
