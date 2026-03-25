"""Canonical native-library resolution for Saguaro."""

from __future__ import annotations

import os
import platform
from pathlib import Path

_ARCH_MAP = {
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "aarch64": "arm64",
    "arm64": "arm64",
}


def native_arch_dir() -> str:
    """Return the normalized architecture directory used by native builds."""
    return _ARCH_MAP.get(platform.machine().lower(), platform.machine().lower())


def repo_root() -> Path:
    """Return the repository root for the current checkout."""
    return Path(__file__).resolve().parents[3]


def authoritative_root() -> Path:
    """Return the authoritative top-level Saguaro root."""
    return repo_root() / "Saguaro"


def native_package_dir() -> Path:
    """Return the lowercase package native root."""
    return authoritative_root() / "saguaro" / "native"


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    ordered: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


def _core_name_order(prefer_tf_free: bool) -> list[str]:
    if prefer_tf_free:
        return ["_saguaro_native.so", "_saguaro_core.so"]
    return ["_saguaro_core.so", "_saguaro_native.so"]


def core_library_candidates(*, prefer_tf_free: bool = False) -> list[Path]:
    """Return canonical core-library candidates ordered by authority."""
    root = repo_root()
    authority = authoritative_root()
    native_dir = native_package_dir()
    arch = native_arch_dir()

    candidates: list[Path] = []
    override_keys = (
        ("SAGUARO_NATIVE_BINARY", prefer_tf_free),
        ("SAGUARO_CORE_BINARY", True),
    )
    for env_name, enabled in override_keys:
        if not enabled:
            continue
        override = os.getenv(env_name)
        if override:
            candidates.append(Path(override))

    roots = [
        authority / "build",
        authority / "saguaro",
        root / "build",
        root / "saguaro",
        root,
        native_dir / "bin" / arch,
        native_dir / "build",
    ]
    for stale_dir in sorted(native_dir.glob("build.stale-*")):
        roots.append(stale_dir)

    for name in _core_name_order(prefer_tf_free):
        for base in roots:
            candidates.append(base / name)
    return _dedupe_paths(candidates)


def resolve_core_library(
    *,
    prefer_tf_free: bool = False,
    required: bool = False,
) -> Path | None:
    """Resolve the canonical Saguaro shared library path."""
    candidates = core_library_candidates(prefer_tf_free=prefer_tf_free)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if required:
        raise RuntimeError(
            "Saguaro native library not found. Searched: "
            + ", ".join(str(candidate) for candidate in candidates)
        )
    return None


def resolve_op_library_path(
    lib_basename: str,
    *,
    prefer_consolidated: bool = True,
    target_arch: str | None = None,
    module_file: str | None = None,
) -> Path:
    """Resolve a custom-op shared object, preferring the unified core binary."""
    if prefer_consolidated:
        resolved = resolve_core_library(prefer_tf_free=False, required=False)
        if resolved is not None:
            return resolved

    raw = Path(str(lib_basename or ""))
    base_name = raw.name
    stem = raw.stem if raw.suffix == ".so" else base_name
    arch = target_arch or native_arch_dir()
    native_dir = native_package_dir()
    module_dir = Path(module_file).resolve().parent if module_file else native_dir / "ops"
    candidates = [
        native_dir / "bin" / arch / f"{stem}.{arch}.so",
        native_dir / "bin" / arch / f"{stem}.so",
        native_dir / "ops" / f"{stem}.{arch}.so",
        native_dir / "ops" / f"{stem}.so",
        module_dir / f"{stem}.{arch}.so",
        module_dir / f"{stem}.so",
    ]
    for candidate in _dedupe_paths(candidates):
        if candidate.exists():
            return candidate
    return _dedupe_paths(candidates)[0]


def manifest_candidates(library_path: str | os.PathLike[str] | None = None) -> list[Path]:
    """Return build-manifest candidates aligned to the unified loader paths."""
    root = repo_root()
    authority = authoritative_root()
    candidates: list[Path] = []
    if library_path:
        candidates.append(Path(library_path).resolve().parent / "saguaro_build_manifest.json")
    candidates.extend(
        [
            authority / "build" / "saguaro_build_manifest.json",
            authority / "saguaro" / "native" / "build" / "saguaro_build_manifest.json",
            root / "build" / "saguaro_build_manifest.json",
            root / "saguaro" / "native" / "build" / "saguaro_build_manifest.json",
            root / "saguaro" / "native" / "build_release" / "saguaro_build_manifest.json",
            root / "saguaro" / "native" / "build_test" / "saguaro_build_manifest.json",
            root / "saguaro" / "native" / "bin" / native_arch_dir() / "saguaro_build_manifest.json",
        ]
    )
    return _dedupe_paths(candidates)
