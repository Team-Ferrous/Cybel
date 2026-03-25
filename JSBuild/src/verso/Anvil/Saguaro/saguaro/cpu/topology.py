"""Architecture packs for static CPU analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ArchitecturePack:
    """Static architecture assumptions for the CPU scan."""

    arch: str
    isa_family: str
    vector_bits: int
    lane_width_bits: int
    preferred_alignment: int
    cache_line_bytes: int
    prefetch_distance: int
    gather_penalty: float
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_PACKS = {
    "x86_64-avx2": ArchitecturePack(
        arch="x86_64-avx2",
        isa_family="avx2",
        vector_bits=256,
        lane_width_bits=32,
        preferred_alignment=32,
        cache_line_bytes=64,
        prefetch_distance=2,
        gather_penalty=1.4,
        notes=("Default CPU-first pack for local Anvil benchmarking.",),
    ),
    "x86_64-avx512": ArchitecturePack(
        arch="x86_64-avx512",
        isa_family="avx512",
        vector_bits=512,
        lane_width_bits=32,
        preferred_alignment=64,
        cache_line_bytes=64,
        prefetch_distance=3,
        gather_penalty=1.2,
        notes=("Assumes wide vectors but tighter register-pressure budgets.",),
    ),
    "arm64-neon": ArchitecturePack(
        arch="arm64-neon",
        isa_family="neon",
        vector_bits=128,
        lane_width_bits=32,
        preferred_alignment=16,
        cache_line_bytes=64,
        prefetch_distance=1,
        gather_penalty=1.8,
        notes=("Models NEON as a narrower SIMD target with weaker gather behavior.",),
    ),
}


def normalize_arch_name(arch: str | None) -> str:
    """Normalize CLI and runtime architecture labels."""

    raw = str(arch or "x86_64-avx2").strip().lower()
    aliases = {
        "amd64": "x86_64-avx2",
        "x86_64": "x86_64-avx2",
        "x86_64-avx2": "x86_64-avx2",
        "x86_64-avx512": "x86_64-avx512",
        "arm64": "arm64-neon",
        "aarch64": "arm64-neon",
        "arm64-neon": "arm64-neon",
    }
    return aliases.get(raw, raw)


def get_architecture_pack(arch: str | None) -> ArchitecturePack:
    """Return one of the supported architecture packs."""

    normalized = normalize_arch_name(arch)
    return _PACKS.get(normalized, _PACKS["x86_64-avx2"])
