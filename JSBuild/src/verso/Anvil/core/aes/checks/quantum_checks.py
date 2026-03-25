import ast
from typing import Any


def _violation(rule_id: str, filepath: str, line: int, message: str) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "filepath": filepath,
        "line": line,
        "message": message,
    }


def check_no_magic_angles(source: str, filepath: str) -> list[dict[str, Any]]:
    tree = ast.parse(source)
    violations: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"rx", "ry", "rz"} and node.args:
                if isinstance(node.args[0], ast.Constant) and isinstance(
                    node.args[0].value, (int, float)
                ):
                    violations.append(
                        _violation(
                            "AES-QC-2",
                            filepath,
                            node.lineno,
                            f"Magic angle literal used in {node.func.attr} gate",
                        )
                    )
    return violations


def check_transpilation_required(source: str, filepath: str) -> list[dict[str, Any]]:
    lowered = source.lower()
    if ".run(" in lowered and "transpile(" not in lowered and "compile(" not in lowered:
        return [
            _violation(
                "AES-QC-4",
                filepath,
                1,
                "Backend execution detected without transpile/compile marker",
            )
        ]
    return []


def check_noise_model_present(source: str, filepath: str) -> list[dict[str, Any]]:
    lowered = source.lower()
    if "simulator" in lowered and "noise" not in lowered:
        return [
            _violation(
                "AES-QC-3",
                filepath,
                1,
                "Simulator usage detected without explicit noise model marker",
            )
        ]
    return []


def check_shot_sufficiency(source: str, filepath: str) -> list[dict[str, Any]]:
    lowered = source.lower()
    if "shots=" in lowered:
        for token in lowered.split("shots=")[1:]:
            digits = "".join(ch for ch in token if ch.isdigit())
            if digits and int(digits[:6]) < 128:
                return [
                    _violation(
                        "AES-QC-5",
                        filepath,
                        1,
                        "Shot count below minimum confidence threshold (128)",
                    )
                ]
    return []
