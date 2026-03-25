import ast
import inspect
import json
from pathlib import Path
from typing import Any

from core.aes.policy_surfaces import is_excluded_path

_VISUALS_RULES_ANCHOR = "standards/AES_RULES.json"
_VISUAL_PACKS: tuple[tuple[str, str], ...] = (
    ("v1", "aes_visuals/v1/directives.json"),
    ("v2", "aes_visuals/v2/directives.json"),
)
_TOP_LEVEL_STRING_FIELDS = (
    "schema_version",
    "artifact",
    "generated_on",
    "owner",
)
_DIRECTIVE_STRING_FIELDS = ("directive_id", "title", "rationale")
_DIRECTIVE_LIST_FIELDS = (
    "enforcement_targets",
    "implementation_patterns",
    "verification_checks",
    "source_refs",
)


def _violation(rule_id: str, filepath: str, line: int, message: str) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "filepath": filepath,
        "line": line,
        "message": message,
    }


def _normalize_relpath(filepath: str) -> str:
    return Path(filepath).as_posix().lstrip("./")


def _is_visuals_anchor(filepath: str) -> bool:
    normalized = _normalize_relpath(filepath)
    return normalized == _VISUALS_RULES_ANCHOR or normalized.endswith(
        f"/{_VISUALS_RULES_ANCHOR}"
    )


def _resolve_repo_root(filepath: str) -> Path | None:
    path = Path(filepath)
    if path.is_absolute():
        normalized = path.as_posix()
        suffix = f"/{_VISUALS_RULES_ANCHOR}"
        if normalized.endswith(suffix):
            return path.resolve().parents[1]

    frame = inspect.currentframe()
    try:
        cursor = frame.f_back if frame else None
        while cursor is not None:
            caller_self = cursor.f_locals.get("self")
            repo_path = getattr(caller_self, "repo_path", None)
            if isinstance(repo_path, str) and repo_path:
                return Path(repo_path).resolve()
            cursor = cursor.f_back
    finally:
        del frame

    cwd = Path.cwd().resolve()
    if (cwd / _VISUALS_RULES_ANCHOR).exists():
        return cwd
    return None


def _validate_visual_pack_payload(payload: Any, expected_version: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["Top-level JSON payload must be an object."]

    for field in _TOP_LEVEL_STRING_FIELDS:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"Missing non-empty string field '{field}'.")

    profile = payload.get("profile")
    if not isinstance(profile, str) or not profile.strip():
        errors.append("Missing non-empty string field 'profile'.")
    elif profile.strip() != expected_version:
        errors.append(
            f"Field 'profile' must equal '{expected_version}' (found '{profile}')."
        )

    upstream_context = payload.get("upstream_context")
    if not isinstance(upstream_context, dict) or not upstream_context:
        errors.append("Field 'upstream_context' must be a non-empty object.")
    else:
        for key, value in upstream_context.items():
            if not isinstance(key, str) or not key.strip():
                errors.append("Field 'upstream_context' must use non-empty string keys.")
                break
            if not isinstance(value, str) or not value.strip():
                errors.append(
                    "Field 'upstream_context' must use non-empty string values."
                )
                break

    directives = payload.get("directives")
    if not isinstance(directives, list) or not directives:
        errors.append("Field 'directives' must be a non-empty list.")
        return errors

    for idx, directive in enumerate(directives):
        if not isinstance(directive, dict):
            errors.append(f"Directive #{idx} must be an object.")
            continue
        for field in _DIRECTIVE_STRING_FIELDS:
            value = directive.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(
                    f"Directive #{idx} is missing non-empty string field '{field}'."
                )
        for field in _DIRECTIVE_LIST_FIELDS:
            value = directive.get(field)
            if not isinstance(value, list) or not value:
                errors.append(
                    f"Directive #{idx} is missing non-empty list field '{field}'."
                )
                continue
            if any(not isinstance(item, str) or not item.strip() for item in value):
                errors.append(
                    f"Directive #{idx} field '{field}' must contain only non-empty strings."
                )
    return errors


def check_aes_visuals_pack_presence(source: str, filepath: str) -> list[dict[str, Any]]:
    del source
    if not _is_visuals_anchor(filepath):
        return []
    repo_root = _resolve_repo_root(filepath)
    if repo_root is None:
        return []

    violations: list[dict[str, Any]] = []
    for _version, pack_relpath in _VISUAL_PACKS:
        if not (repo_root / pack_relpath).is_file():
            violations.append(
                _violation(
                    "AES-VIS-1",
                    pack_relpath,
                    1,
                    f"Missing model-readable visuals governance pack: {pack_relpath}",
                )
            )
    return violations


def check_aes_visuals_pack_shape(source: str, filepath: str) -> list[dict[str, Any]]:
    del source
    if not _is_visuals_anchor(filepath):
        return []
    repo_root = _resolve_repo_root(filepath)
    if repo_root is None:
        return []

    violations: list[dict[str, Any]] = []
    for version, pack_relpath in _VISUAL_PACKS:
        pack_path = repo_root / pack_relpath
        if not pack_path.is_file():
            continue
        try:
            payload = json.loads(pack_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            violations.append(
                _violation(
                    "AES-VIS-2",
                    pack_relpath,
                    1,
                    f"Invalid JSON for visuals governance pack ({pack_relpath}): {exc}",
                )
            )
            continue

        for error in _validate_visual_pack_payload(payload, expected_version=version):
            violations.append(_violation("AES-VIS-2", pack_relpath, 1, error))
    return violations


def check_no_bare_except(source: str, filepath: str) -> list[dict[str, Any]]:
    tree = ast.parse(source)
    violations: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            violations.append(
                _violation("AES-CR-2", filepath, node.lineno, "Bare except is not allowed")
            )
    return violations


def check_type_annotations(source: str, filepath: str) -> list[dict[str, Any]]:
    if is_excluded_path("AES-PY-1", filepath):
        return []
    tree = ast.parse(source)
    violations: list[dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            missing_return = node.returns is None
            missing_arg = any(arg.annotation is None for arg in node.args.args)
            if missing_return or missing_arg:
                violations.append(
                    _violation(
                        "AES-PY-1",
                        filepath,
                        node.lineno,
                        f"Public function '{node.name}' is missing type annotations",
                    )
                )
    return violations


def check_no_dynamic_execution(source: str, filepath: str) -> list[dict[str, Any]]:
    tree = ast.parse(source)
    violations: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"eval", "exec"}:
                violations.append(
                    _violation(
                        "AES-PY-4",
                        filepath,
                        node.lineno,
                        f"{node.func.id}() is forbidden outside explicitly sandboxed contexts",
                    )
                )
    return violations


def check_no_eval_exec(source: str, filepath: str) -> list[dict[str, Any]]:
    tree = ast.parse(source)
    violations: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"eval", "exec"}:
                violations.append(
                    _violation(
                        "AES-SEC-2",
                        filepath,
                        node.lineno,
                        f"{node.func.id}() requires explicit sandbox justification",
                    )
                )
    return violations


def check_no_wildcard_imports(source: str, filepath: str) -> list[dict[str, Any]]:
    tree = ast.parse(source)
    violations: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(
            alias.name == "*" for alias in node.names
        ):
            violations.append(
                _violation(
                    "AES-PY-5",
                    filepath,
                    node.lineno,
                    "Wildcard imports are forbidden in governed Python code",
                )
            )
    return violations


def check_suspicious_exception_none_returns(
    source: str, filepath: str
) -> list[dict[str, Any]]:
    tree = ast.parse(source)
    violations: list[dict[str, Any]] = []
    parents: dict[ast.AST, ast.AST] = {}

    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if _handler_has_raise(node):
            continue
        if _handler_has_logging(node):
            continue
        return_node = _find_none_return(node)
        if return_node is None:
            continue
        if _function_return_allows_none(node, parents):
            continue
        violations.append(
            _violation(
                "AES-ERR-2",
                filepath,
                return_node.lineno,
                "Exception handler returns None without logging, raising, or an Optional return contract.",
            )
        )
    return violations


def _find_none_return(handler: ast.ExceptHandler) -> ast.Return | None:
    for node in ast.walk(handler):
        if not isinstance(node, ast.Return):
            continue
        if node.value is None:
            return node
        if isinstance(node.value, ast.Constant) and node.value.value is None:
            return node
    return None


def _handler_has_raise(handler: ast.ExceptHandler) -> bool:
    return any(isinstance(node, ast.Raise) for node in ast.walk(handler))


def _handler_has_logging(handler: ast.ExceptHandler) -> bool:
    logging_names = {
        "logging.exception",
        "logging.error",
        "logging.warning",
        "logger.exception",
        "logger.error",
        "logger.warning",
        "log.exception",
        "log.error",
        "log.warning",
    }
    for node in ast.walk(handler):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func) in logging_names:
            return True
    return False


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        base = _call_name(func.value)
        return f"{base}.{func.attr}" if base else func.attr
    return ""


def _function_return_allows_none(
    handler: ast.ExceptHandler, parents: dict[ast.AST, ast.AST]
) -> bool:
    cursor: ast.AST | None = handler
    while cursor is not None and not isinstance(
        cursor, (ast.FunctionDef, ast.AsyncFunctionDef)
    ):
        cursor = parents.get(cursor)

    if not isinstance(cursor, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    return _annotation_allows_none(cursor.returns)


def _annotation_allows_none(annotation: ast.AST | None) -> bool:
    if annotation is None:
        return False
    if isinstance(annotation, ast.Name):
        return annotation.id in {"Optional", "None"}
    if isinstance(annotation, ast.Constant):
        return annotation.value is None
    if isinstance(annotation, ast.Attribute):
        return annotation.attr == "Optional"
    if isinstance(annotation, ast.Subscript):
        if isinstance(annotation.value, ast.Name) and annotation.value.id == "Optional":
            return True
        if isinstance(annotation.value, ast.Attribute) and annotation.value.attr == "Optional":
            return True
        slice_value = annotation.slice
        if isinstance(slice_value, ast.Tuple):
            return any(_annotation_allows_none(item) for item in slice_value.elts)
        return _annotation_allows_none(slice_value)
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return _annotation_allows_none(annotation.left) or _annotation_allows_none(
            annotation.right
        )
    if isinstance(annotation, ast.Tuple):
        return any(_annotation_allows_none(item) for item in annotation.elts)
    return False


def check_context_managed_open(source: str, filepath: str) -> list[dict[str, Any]]:
    tree = ast.parse(source)
    managed_lines: set[int] = set()
    violations: list[dict[str, Any]] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                context_expr = item.context_expr
                if (
                    isinstance(context_expr, ast.Call)
                    and isinstance(context_expr.func, ast.Name)
                    and context_expr.func.id == "open"
                ):
                    managed_lines.add(context_expr.lineno)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "open":
            continue
        if node.lineno in managed_lines:
            continue
        violations.append(
            _violation(
                "AES-PY-6",
                filepath,
                node.lineno,
                "open() must be used via a context manager in governed Python code",
            )
        )
    return violations


def check_complexity_bounds(source: str, filepath: str) -> list[dict[str, Any]]:
    tree = ast.parse(source)
    violations: list[dict[str, Any]] = []
    branch_nodes = (
        ast.If,
        ast.For,
        ast.While,
        ast.Try,
        ast.With,
        ast.BoolOp,
        ast.Match,
        ast.comprehension,
    )
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            complexity = 1 + sum(
                1 for child in ast.walk(node) if isinstance(child, branch_nodes)
            )
            if complexity > 10:
                violations.append(
                    _violation(
                        "AES-CPLX-1",
                        filepath,
                        node.lineno,
                        f"Function '{node.name}' exceeds complexity bound ({complexity} > 10)",
                    )
                )
    return violations


def check_error_contracts(source: str, filepath: str) -> list[dict[str, Any]]:
    if is_excluded_path("AES-ERR-1", filepath):
        return []
    tree = ast.parse(source)
    violations: list[dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            docstring = ast.get_docstring(node) or ""
            has_decorator_contract = any(
                (
                    isinstance(decorator, ast.Name)
                    and decorator.id == "error_contract"
                )
                or (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Name)
                    and decorator.func.id == "error_contract"
                )
                for decorator in node.decorator_list
            )
            if (
                "Raises" not in docstring
                and "Errors" not in docstring
                and not has_decorator_contract
            ):
                violations.append(
                    _violation(
                        "AES-ERR-1",
                        filepath,
                        node.lineno,
                        f"Public function '{node.name}' is missing an error contract",
                    )
                )
    return violations


def check_mutable_defaults(source: str, filepath: str) -> list[dict[str, Any]]:
    tree = ast.parse(source)
    violations: list[dict[str, Any]] = []
    mutable_nodes = (ast.List, ast.Dict, ast.Set)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for default in node.args.defaults:
                if isinstance(default, mutable_nodes):
                    violations.append(
                        _violation(
                            "AES-PY-3",
                            filepath,
                            node.lineno,
                            f"Function '{node.name}' uses a mutable default argument",
                        )
                    )
    return violations
