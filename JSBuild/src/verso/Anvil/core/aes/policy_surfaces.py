from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.aes.rule_registry import AESRuleRegistry

_DEFAULT_SELECTORS: dict[str, dict[str, Any]] = {
    "AES-PY-1": {
        "exclude_path_prefixes": ["saguaro/native/", "core/native/", "saguaro/ops/"],
        "surface": "public_python_functions",
    },
    "AES-ERR-1": {
        "exclude_path_prefixes": ["saguaro/native/", "core/native/", "saguaro/ops/"],
        "include_path_tokens": ["api", "routes", "service"],
        "public_export_marker": "__all__",
        "surface": "public_error_contracts",
    },
}


@dataclass(frozen=True, slots=True)
class SurfacePolicy:
    rule_id: str
    selectors: dict[str, Any]

    @property
    def exclude_path_prefixes(self) -> tuple[str, ...]:
        values = self.selectors.get("exclude_path_prefixes", [])
        return tuple(str(item).lower() for item in values)

    @property
    def include_path_tokens(self) -> tuple[str, ...]:
        values = self.selectors.get("include_path_tokens", [])
        return tuple(str(item).lower() for item in values)

    @property
    def public_export_marker(self) -> str:
        return str(self.selectors.get("public_export_marker", "__all__"))


def _normalize_path(filepath: str) -> str:
    return filepath.replace("\\", "/").lower().lstrip("./")


def _resolve_repo_root(filepath: str) -> Path | None:
    path = Path(filepath)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.extend(path.parents)
    else:
        cwd = Path.cwd().resolve()
        candidates.extend([cwd, *cwd.parents])

    for candidate in candidates:
        if (candidate / "standards" / "AES_RULES.json").exists():
            return candidate
    return None


@lru_cache(maxsize=32)
def _load_registry(repo_root: str) -> AESRuleRegistry:
    registry = AESRuleRegistry()
    registry.load(str(Path(repo_root) / "standards" / "AES_RULES.json"))
    return registry


def get_surface_policy(rule_id: str, filepath: str) -> SurfacePolicy:
    selectors = dict(_DEFAULT_SELECTORS.get(rule_id, {}))
    repo_root = _resolve_repo_root(filepath)
    if repo_root is not None:
        try:
            rule = _load_registry(str(repo_root)).get_rule(rule_id)
        except Exception:
            rule = None
        if rule is not None and rule.selectors:
            selectors.update(rule.selectors)
    return SurfacePolicy(rule_id=rule_id, selectors=selectors)


def is_excluded_path(rule_id: str, filepath: str) -> bool:
    normalized = _normalize_path(filepath)
    policy = get_surface_policy(rule_id, filepath)
    return any(
        normalized.startswith(prefix) or f"/{prefix}" in f"/{normalized}"
        for prefix in policy.exclude_path_prefixes
    )


def is_error_contract_surface(filepath: str, source: str) -> bool:
    policy = get_surface_policy("AES-ERR-1", filepath)
    normalized = _normalize_path(filepath)
    if is_excluded_path("AES-ERR-1", filepath):
        return False
    if policy.public_export_marker and policy.public_export_marker in source:
        return True
    return any(token in normalized for token in policy.include_path_tokens)
