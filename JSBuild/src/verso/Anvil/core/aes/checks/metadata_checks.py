from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


def _violation(rule_id: str, filepath: str, line: int, message: str) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "filepath": filepath,
        "line": line,
        "message": message,
    }


def check_catalog_authority(source: str, filepath: str) -> list[dict[str, Any]]:
    rel = filepath.replace("\\", "/")
    if rel.endswith("standards/AES_RULES.json") or rel == "standards/AES_RULES.json":
        missing = [
            field
            for field in ("execution_mode", "source_version", "source_refs", "status")
            if field not in source
        ]
        if missing:
            return [
                _violation(
                    "AES-ARCH-4",
                    filepath,
                    1,
                    f"Compiled AES rules missing authoritative catalog fields: {', '.join(missing)}",
                )
            ]
        return []

    if Path(filepath).name == "setup.py" and "saguaro=saguaro.cli:main" not in source:
        return [
            _violation(
                "AES-ARCH-4",
                filepath,
                1,
                "Lowercase saguaro console entry point must remain authoritative.",
            )
        ]
    return []


def check_no_silent_fallback_markers(source: str, filepath: str) -> list[dict[str, Any]]:
    tree = ast.parse(source)
    violations: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if not _is_broad_exception(node):
            continue
        if _handler_has_raise(node) or _handler_has_logging(node):
            continue
        if _handler_is_silent(node):
            line = getattr(node, "lineno", 1)
            if node.body:
                line = getattr(node.body[0], "lineno", line)
            violations.append(
                _violation(
                    "AES-CR-1",
                    filepath,
                    line,
                    "Suspicious silent fallback or masked exception path detected.",
                )
            )
    return violations


def check_no_verification_bypass_markers(
    source: str, filepath: str
) -> list[dict[str, Any]]:
    tree = ast.parse(source)
    violations: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = node.value
            for target in node.targets:
                if _is_verification_bypass_target(target) and _is_truthy(value):
                    violations.append(
                        _violation(
                            "AES-CR-3",
                            filepath,
                            getattr(target, "lineno", getattr(node, "lineno", 1)),
                            "Fallback paths MUST NOT silently bypass required verification.",
                        )
                    )
        elif isinstance(node, ast.AnnAssign):
            if _is_verification_bypass_target(node.target) and _is_truthy(node.value):
                violations.append(
                    _violation(
                        "AES-CR-3",
                        filepath,
                        getattr(node.target, "lineno", getattr(node, "lineno", 1)),
                        "Fallback paths MUST NOT silently bypass required verification.",
                    )
                )
        elif isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg and _matches_verification_bypass_name(keyword.arg) and _is_truthy(
                    keyword.value
                ):
                    violations.append(
                        _violation(
                            "AES-CR-3",
                            filepath,
                            getattr(keyword, "lineno", getattr(node, "lineno", 1)),
                            "Fallback paths MUST NOT silently bypass required verification.",
                        )
                    )
    return violations


def _is_broad_exception(node: ast.ExceptHandler) -> bool:
    if node.type is None:
        return True
    return _exception_name(node.type) in {"Exception", "BaseException"}


def _exception_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _exception_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _handler_has_raise(node: ast.ExceptHandler) -> bool:
    return any(isinstance(child, ast.Raise) for child in ast.walk(node))


def _handler_has_logging(node: ast.ExceptHandler) -> bool:
    logging_names = {
        "logging.exception",
        "logging.error",
        "logging.warning",
        "logging.info",
        "logging.debug",
        "logger.exception",
        "logger.error",
        "logger.warning",
        "logger.info",
        "logger.debug",
        "log.exception",
        "log.error",
        "log.warning",
        "log.info",
        "log.debug",
    }
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if _call_name(child.func) in logging_names:
            return True
    return False


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        base = _call_name(func.value)
        return f"{base}.{func.attr}" if base else func.attr
    return ""


def _handler_is_silent(node: ast.ExceptHandler) -> bool:
    for child in node.body:
        if isinstance(child, ast.Pass):
            return True
        if isinstance(child, ast.Return):
            if child.value is None:
                return True
            if isinstance(child.value, ast.Constant) and child.value.value is None:
                return True
    return False


def _is_verification_bypass_target(target: ast.AST) -> bool:
    if isinstance(target, ast.Name):
        return _matches_verification_bypass_name(target.id)
    if isinstance(target, ast.Attribute):
        return _matches_verification_bypass_name(target.attr)
    return False


def _matches_verification_bypass_name(name: str) -> bool:
    normalized = str(name or "").strip().lower()
    if not normalized:
        return False
    has_action = any(token in normalized for token in ("skip", "disable", "bypass"))
    has_verification = any(
        token in normalized for token in ("verify", "verification", "validation", "check")
    )
    return has_action and has_verification


def _is_truthy(node: ast.AST | None) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.Constant):
        return bool(node.value)
    if isinstance(node, ast.NameConstant):  # pragma: no cover - py<3.8 compatibility shape
        return bool(node.value)
    return False
