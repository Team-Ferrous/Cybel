from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from typing import Any


_OPERATOR_TOKENS = {
    "+",
    "-",
    "*",
    "/",
    "//",
    "%",
    "**",
    "=",
    "==",
    "!=",
    "<",
    ">",
    "<=",
    ">=",
    "and",
    "or",
    "not",
    "if",
    "for",
    "while",
    "try",
    "except",
    "with",
    "return",
}


def compute_halstead_metrics(source: str) -> dict[str, float]:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|==|!=|<=|>=|\*\*|//|[+\-*/%=<>]", source)
    operators = [tok for tok in tokens if tok in _OPERATOR_TOKENS]
    operands = [tok for tok in tokens if tok not in _OPERATOR_TOKENS]

    n1 = len(set(operators))
    n2 = len(set(operands))
    N1 = len(operators)
    N2 = len(operands)
    vocabulary = max(1, n1 + n2)
    length = N1 + N2
    volume = float(length * math.log2(vocabulary)) if length else 0.0
    difficulty = float((n1 / 2.0) * (N2 / max(1, n2))) if n1 else 0.0
    effort = float(volume * difficulty)

    return {
        "n1_distinct_operators": float(n1),
        "n2_distinct_operands": float(n2),
        "N1_total_operators": float(N1),
        "N2_total_operands": float(N2),
        "vocabulary": float(vocabulary),
        "length": float(length),
        "volume": volume,
        "difficulty": difficulty,
        "effort": effort,
    }


def _python_files(repo_path: Path) -> list[Path]:
    return sorted(path for path in repo_path.rglob("*.py") if "venv" not in path.parts and ".git" not in path.parts)


def compute_dsqi(repo_path: str) -> dict[str, Any]:
    root = Path(repo_path)
    files = _python_files(root)
    if not files:
        return {
            "score": 1.0,
            "files": 0,
            "avg_branches_per_function": 0.0,
            "avg_function_length": 0.0,
        }

    function_count = 0
    branch_count = 0
    total_function_lines = 0
    branch_nodes = (ast.If, ast.For, ast.While, ast.Try, ast.Match, ast.BoolOp, ast.comprehension)

    for file_path in files:
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_count += 1
                end_line = getattr(node, "end_lineno", node.lineno)
                total_function_lines += max(1, end_line - node.lineno + 1)
                branch_count += sum(1 for child in ast.walk(node) if isinstance(child, branch_nodes))

    if function_count == 0:
        return {
            "score": 1.0,
            "files": len(files),
            "avg_branches_per_function": 0.0,
            "avg_function_length": 0.0,
        }

    avg_branches = branch_count / function_count
    avg_length = total_function_lines / function_count
    penalty = min(1.0, (avg_branches / 10.0) + (avg_length / 120.0))
    score = max(0.0, round(1.0 - penalty, 4))

    return {
        "score": score,
        "files": len(files),
        "functions": function_count,
        "avg_branches_per_function": round(avg_branches, 4),
        "avg_function_length": round(avg_length, 4),
    }
