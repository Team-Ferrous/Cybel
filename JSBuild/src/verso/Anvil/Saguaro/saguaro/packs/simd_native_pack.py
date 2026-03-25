"""SIMD/native performance pack."""

from .base import PackSpec

SIMD_NATIVE_PACK = PackSpec(
    name="simd_native_pack",
    description="SIMD kernels, AVX2-style intrinsics, memory layout, and native hot paths.",
    keywords=["avx", "avx2", "simd", "intrin", "vectorized", "aligned", "stride"],
    languages=["cpp", "c", "rust"],
    file_hints=["kernel", "simd", "avx", "intrinsics", "native"],
)
