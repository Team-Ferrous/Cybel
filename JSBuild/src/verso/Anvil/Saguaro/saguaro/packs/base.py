"""Domain pack registry and diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from typing import Any


@dataclass(slots=True)
class PackSpec:
    """Declarative pack definition."""

    name: str
    description: str
    keywords: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    file_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PackManager:
    """Manage optional scientific/domain understanding packs."""

    def __init__(self, repo_path: str) -> None:
        self.repo_path = os.path.abspath(repo_path)
        self.base_dir = os.path.join(self.repo_path, ".saguaro", "packs")
        self.enabled_path = os.path.join(self.base_dir, "enabled.json")
        self._registry = self._load_registry()

    def list(self) -> list[dict[str, Any]]:
        enabled = set(self.enabled())
        rows = []
        for spec in self._registry.values():
            row = spec.to_dict()
            row["enabled"] = spec.name in enabled
            rows.append(row)
        return rows

    def enable(self, name: str) -> dict[str, Any]:
        os.makedirs(self.base_dir, exist_ok=True)
        if name not in self._registry:
            return {"status": "missing", "pack": name}
        enabled = set(self.enabled())
        enabled.add(name)
        with open(self.enabled_path, "w", encoding="utf-8") as handle:
            json.dump(sorted(enabled), handle, indent=2)
        return {"status": "ok", "enabled": sorted(enabled)}

    def enabled(self) -> list[str]:
        if not os.path.exists(self.enabled_path):
            return []
        with open(self.enabled_path, encoding="utf-8") as handle:
            data = json.load(handle)
        return sorted(str(item) for item in data)

    def diagnose(self, path: str = ".") -> dict[str, Any]:
        root = os.path.abspath(path if os.path.isabs(path) else os.path.join(self.repo_path, path))
        evidence: dict[str, list[str]] = {name: [] for name in self._registry}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                item
                for item in dirnames
                if item not in {".git", ".saguaro", ".anvil", "venv", ".venv", "__pycache__", "Saguaro"}
            ]
            for filename in filenames:
                rel_file = os.path.relpath(os.path.join(dirpath, filename), self.repo_path).replace("\\", "/")
                lower = rel_file.lower()
                try:
                    with open(os.path.join(dirpath, filename), encoding="utf-8", errors="ignore") as handle:
                        head = handle.read(4096).lower()
                except OSError:
                    continue
                for name, spec in self._registry.items():
                    if any(hint in lower for hint in spec.file_hints) or any(keyword in head for keyword in spec.keywords):
                        evidence[name].append(rel_file)
        results = []
        for name, files in evidence.items():
            results.append(
                {
                    "pack": name,
                    "matches": sorted(files)[:20],
                    "count": len(files),
                    "enabled": name in set(self.enabled()),
                }
            )
        return {"status": "ok", "packs": results}

    @staticmethod
    def _load_registry() -> dict[str, PackSpec]:
        from .cfd_pack import CFD_PACK
        from .chemistry_pack import CHEMISTRY_PACK
        from .jax_pack import JAX_PACK
        from .physics_pack import PHYSICS_PACK
        from .quantum_pack import QUANTUM_PACK
        from .simd_native_pack import SIMD_NATIVE_PACK
        from .tensorflow_pack import TENSORFLOW_PACK
        from .torch_pack import TORCH_PACK

        return {
            spec.name: spec
            for spec in [
                TORCH_PACK,
                JAX_PACK,
                TENSORFLOW_PACK,
                SIMD_NATIVE_PACK,
                PHYSICS_PACK,
                CHEMISTRY_PACK,
                QUANTUM_PACK,
                CFD_PACK,
            ]
        }
