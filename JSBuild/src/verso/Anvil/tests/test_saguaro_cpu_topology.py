from __future__ import annotations

from saguaro.cpu import get_architecture_pack


def test_architecture_pack_normalizes_common_aliases() -> None:
    avx2 = get_architecture_pack("amd64")
    neon = get_architecture_pack("aarch64")

    assert avx2.arch == "x86_64-avx2"
    assert avx2.preferred_alignment == 32
    assert neon.arch == "arm64-neon"
    assert neon.vector_bits == 128
