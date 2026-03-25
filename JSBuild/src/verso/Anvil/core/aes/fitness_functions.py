from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

LAYER_ORDER = {
    "orchestration": 0,
    "domain": 1,
    "kernel": 2,
}


FORBIDDEN_IMPORT_PREFIXES = {
    "core.aes": ("agents.", "agents"),
}


def _violation(rule_id: str, file: str, message: str, line: int = 1) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "file": file,
        "line": line,
        "message": message,
        "severity": "P1",
    }


def _python_files(repo_path: Path) -> list[Path]:
    return sorted(path for path in repo_path.rglob("*.py") if ".git" not in path.parts and "venv" not in path.parts)


def _module_name(repo_path: Path, file_path: Path) -> str:
    rel = file_path.relative_to(repo_path)
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _collect_imports(repo_path: Path) -> dict[str, list[tuple[str, int]]]:
    graph: dict[str, list[tuple[str, int]]] = {}
    for file_path in _python_files(repo_path):
        module = _module_name(repo_path, file_path)
        imports: list[tuple[str, int]] = []
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            graph[module] = imports
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append((alias.name, getattr(node, "lineno", 1)))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append((node.module, getattr(node, "lineno", 1)))
        graph[module] = imports
    return graph


def check_no_cyclic_dependencies(repo_path: str) -> list[dict[str, Any]]:
    root = Path(repo_path)
    graph = _collect_imports(root)
    adjacency: dict[str, set[str]] = {}
    modules = set(graph)
    for module, imports in graph.items():
        adjacency[module] = set()
        for imported, _line in imports:
            for candidate in modules:
                if imported == candidate or imported.startswith(candidate + "."):
                    adjacency[module].add(candidate)

    visited: set[str] = set()
    stack: set[str] = set()
    violations: list[dict[str, Any]] = []

    def dfs(node: str, trail: list[str]) -> None:
        if node in stack:
            cycle = trail[trail.index(node) :] + [node]
            violations.append(
                _violation(
                    "AES-ARCH-3",
                    node,
                    f"Import cycle detected: {' -> '.join(cycle)}",
                )
            )
            return
        if node in visited:
            return
        visited.add(node)
        stack.add(node)
        for nxt in adjacency.get(node, set()):
            dfs(nxt, trail + [nxt])
        stack.remove(node)

    for module in sorted(adjacency):
        dfs(module, [module])

    # Deduplicate by message.
    dedup: dict[str, dict[str, Any]] = {}
    for item in violations:
        dedup[item["message"]] = item
    return list(dedup.values())


def _layer_for_module(module_name: str) -> str | None:
    if module_name.startswith(("agents", "core", "cli")):
        return "orchestration"
    if module_name.startswith("domains"):
        return "domain"
    if module_name.startswith(("shared_kernel", "core.native", "core.simd")):
        return "kernel"
    return None


def check_layering_rules(repo_path: str) -> list[dict[str, Any]]:
    root = Path(repo_path)
    graph = _collect_imports(root)
    violations: list[dict[str, Any]] = []
    for module, imports in graph.items():
        source_layer = _layer_for_module(module)
        if source_layer is None:
            continue
        for imported, line in imports:
            target_layer = _layer_for_module(imported)
            if target_layer is None:
                continue
            if LAYER_ORDER[source_layer] > LAYER_ORDER[target_layer]:
                violations.append(
                    _violation(
                        "AES-ARCH-3",
                        module,
                        f"Layering violation: {module} ({source_layer}) imports {imported} ({target_layer})",
                        line=line,
                    )
                )
    return violations


def check_forbidden_imports(repo_path: str) -> list[dict[str, Any]]:
    root = Path(repo_path)
    graph = _collect_imports(root)
    violations: list[dict[str, Any]] = []
    for module, imports in graph.items():
        for guarded_prefix, forbidden_prefixes in FORBIDDEN_IMPORT_PREFIXES.items():
            if not module.startswith(guarded_prefix):
                continue
            for imported, line in imports:
                if imported.startswith(forbidden_prefixes):
                    violations.append(
                        _violation(
                            "AES-ARCH-3",
                            module,
                            f"Forbidden import: {module} must not import {imported}",
                            line=line,
                        )
                    )
    return violations


def _collect_public_signatures(repo_path: Path) -> dict[str, str]:
    signatures: dict[str, str] = {}
    for file_path in _python_files(repo_path):
        module = _module_name(repo_path, file_path)
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                args = [arg.arg for arg in node.args.args]
                signatures[f"{module}.{node.name}"] = f"({', '.join(args)})"
    return signatures


def check_abi_boundary_stability(
    repo_path: str,
    baseline_path: str = ".anvil/aes_abi_baseline.json",
) -> list[dict[str, Any]]:
    root = Path(repo_path)
    current = _collect_public_signatures(root)
    baseline_file = root / baseline_path
    if not baseline_file.exists():
        return []

    try:
        baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return [_violation("AES-ARCH-3", baseline_path, "Invalid ABI baseline JSON format")]

    violations: list[dict[str, Any]] = []
    for symbol, old_sig in baseline.items():
        new_sig = current.get(symbol)
        if new_sig is None:
            violations.append(_violation("AES-ARCH-3", symbol, "Public API symbol removed from current build"))
        elif new_sig != old_sig:
            violations.append(
                _violation(
                    "AES-ARCH-3",
                    symbol,
                    f"Public API signature changed: {old_sig} -> {new_sig}",
                )
            )
    return violations


def run_all_fitness_checks(repo_path: str) -> dict[str, Any]:
    checks = {
        "no_cyclic_dependencies": check_no_cyclic_dependencies(repo_path),
        "layering_rules": check_layering_rules(repo_path),
        "forbidden_imports": check_forbidden_imports(repo_path),
        "abi_boundary_stability": check_abi_boundary_stability(repo_path),
    }
    violations = [item for entries in checks.values() for item in entries]
    return {
        "passed": len(violations) == 0,
        "violations": violations,
        "checks": checks,
    }
