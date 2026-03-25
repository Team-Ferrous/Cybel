import ast
from typing import Any


def _contains_finite_check(nodes: list[ast.stmt]) -> bool:
    for node in nodes:
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                text = ast.unparse(child)
                if "isfinite" in text or "torch.isfinite" in text:
                    return True
    return False


def check_gradient_health_gate(source: str, filepath: str) -> list[dict[str, Any]]:
    if "core/aes/checks/" in filepath.replace("\\", "/"):
        return []
    tree = ast.parse(source)
    violations: list[dict[str, Any]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for index, stmt in enumerate(node.body):
            for child in ast.walk(stmt):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                    if child.func.attr == "step":
                        window_start = max(0, index - 5)
                        if not _contains_finite_check(node.body[window_start:index]):
                            violations.append(
                                {
                                    "rule_id": "AES-ML-1",
                                    "filepath": filepath,
                                    "line": child.lineno,
                                    "message": "optimizer.step() missing nearby finite gradient check",
                                }
                            )
    return violations


def check_stable_numerics(source: str, filepath: str) -> list[dict[str, Any]]:
    tree = ast.parse(source)
    violations: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "exp":
                owner = ast.unparse(node.func.value)
                if owner in {"torch", "tf", "tensorflow", "np", "numpy"}:
                    violations.append(
                        {
                            "rule_id": "AES-ML-2",
                            "filepath": filepath,
                            "line": node.lineno,
                            "message": f"Unbounded exponential call detected via {owner}.exp(...)",
                        }
                    )
    return violations


def check_data_validation(source: str, filepath: str) -> list[dict[str, Any]]:
    lowered = source.lower()
    has_ingest = any(token in lowered for token in ("dataloader", "dataset", "load_data", "read_csv"))
    has_validation = any(token in lowered for token in ("shape", "dtype", "schema", "validate"))
    if has_ingest and not has_validation:
        return [
            {
                "rule_id": "AES-ML-4",
                "filepath": filepath,
                "line": 1,
                "message": "Data ingest markers found without schema/shape/dtype validation markers",
            }
        ]
    return []


def check_reproducibility_manifest(source: str, filepath: str) -> list[dict[str, Any]]:
    lowered = source.lower()
    uses_training = any(token in lowered for token in ("optimizer", "train", "epoch"))
    has_seed = "seed" in lowered
    has_version = "version" in lowered or "requirements" in lowered
    if uses_training and (not has_seed or not has_version):
        return [
            {
                "rule_id": "AES-ML-5",
                "filepath": filepath,
                "line": 1,
                "message": "Training flow is missing seed and/or version manifest markers",
            }
        ]
    return []
