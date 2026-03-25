"""Heuristic static FLOP counting for ML/DL-oriented code paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import ast
import os
from typing import Any, Callable


@dataclass(slots=True)
class FLOPEstimate:
    """FLOP estimate for a tensor operation."""

    operation: str
    formula: str
    estimated_flops: int | str
    input_shapes: list[str]
    output_shape: str | None
    file_path: str
    line: int
    confidence: float = 0.5
    evidence: list[str] = field(default_factory=list)


class FLOPCounter:
    """Counts FLOPs for ML/DL operations using static shape analysis."""

    FLOP_FORMULAS: dict[str, Callable[..., int | str]] = {
        "matmul": lambda m="M", n="N", k="K": f"2 * {m} * {n} * {k}",
        "conv2d": lambda h="H_out", w="W_out", cout="C_out", cin="C_in", kh="K_h", kw="K_w": f"2 * {h} * {w} * {cout} * {cin} * {kh} * {kw}",
        "attention": lambda n="n", d="d": f"4 * {n}^2 * {d} + 2 * {n} * {d}^2",
        "linear_attention": lambda n="n", d="d", k="k": f"4 * {n} * {d} * {k}",
        "flash_attention": lambda n="n", d="d": f"~2 * {n}^2 * {d}",
        "mqa_attention": lambda n="n", d="d", h="h": f"2 * {n}^2 * {d} + 2 * {n} * {d}^2 / {h}",
        "gqa_attention": lambda n="n", d="d", g="g": f"2 * {n}^2 * {d} + 2 * {n} * {d}^2 / {g}",
        "layernorm": lambda n="n", d="d": f"7 * {n} * {d}",
        "rmsnorm": lambda n="n", d="d": f"5 * {n} * {d}",
        "embedding": lambda n="n", d="d": f"{n} * {d}",
        "cross_entropy_loss": lambda n="n", c="c": f"2 * {n} * {c}",
        "mse_loss": lambda n="n": f"4 * {n}",
        "activation": lambda n="n": f"{n}",
    }

    def __init__(self, repo_path: str | None = None) -> None:
        self.repo_path = os.path.abspath(repo_path or ".")

    def count_function(self, file_path: str, symbol: str) -> list[FLOPEstimate]:
        """Estimate FLOPs for tensor operations in a target function."""
        abs_path = file_path if os.path.isabs(file_path) else os.path.join(self.repo_path, file_path)
        source = self._read_text(abs_path)
        fn_node = self._find_function_node(source, symbol)
        if fn_node is None:
            return []

        estimates: list[FLOPEstimate] = []
        for node in ast.walk(fn_node):
            item = self._detect_matmul(node, abs_path)
            if item:
                estimates.append(item)
                continue
            item = self._detect_conv2d(node, abs_path)
            if item:
                estimates.append(item)
                continue
            item = self._detect_attention(node, abs_path)
            if item:
                estimates.append(item)
                continue
            item = self._detect_norm(node, abs_path)
            if item:
                estimates.append(item)
                continue
            item = self._detect_embedding(node, abs_path)
            if item:
                estimates.append(item)
                continue
            item = self._detect_loss(node, abs_path)
            if item:
                estimates.append(item)
                continue
            item = self._detect_activation(node, abs_path)
            if item:
                estimates.append(item)

        return estimates

    def count_pipeline(self, trace: Any) -> dict[str, Any]:
        """Estimate aggregate FLOPs across pipeline stages."""
        per_stage: list[dict[str, Any]] = []
        numeric_total = 0
        symbolic_terms: list[str] = []
        confidence_total = 0.0
        confidence_count = 0
        evidence: list[str] = []

        for stage in list(getattr(trace, "stages", []) or []):
            file_path = str(getattr(stage, "file_path", "") or "")
            symbol = str(getattr(stage, "name", "") or "")
            if not file_path or not symbol:
                continue

            estimates = self.count_function(file_path=file_path, symbol=symbol)
            if not estimates:
                continue

            stage_numeric = 0
            stage_symbolic: list[str] = []
            for item in estimates:
                flops = item.estimated_flops
                if isinstance(flops, int):
                    numeric_total += flops
                    stage_numeric += flops
                else:
                    symbolic_terms.append(str(flops))
                    stage_symbolic.append(str(flops))
                confidence_total += item.confidence
                confidence_count += 1
                evidence.extend(item.evidence)

            per_stage.append(
                {
                    "stage_id": getattr(stage, "id", ""),
                    "stage_name": symbol,
                    "numeric_flops": stage_numeric,
                    "symbolic_flops": stage_symbolic,
                    "operations": [asdict(item) for item in estimates],
                }
            )

        confidence = round(confidence_total / max(1, confidence_count), 3)
        return {
            "total_numeric_flops": numeric_total if numeric_total > 0 else None,
            "symbolic_total": " + ".join(symbolic_terms[:25]) if symbolic_terms else None,
            "confidence": confidence,
            "evidence": sorted(set(evidence))[:30],
            "per_stage": per_stage,
        }

    def _detect_matmul(self, node: Any, file_path: str) -> FLOPEstimate | None:
        """Detect matrix multiplication patterns."""
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            expr = self.FLOP_FORMULAS["matmul"]()
            return FLOPEstimate(
                operation="matmul",
                formula="2 * M * N * K",
                estimated_flops=expr,
                input_shapes=["[M, K]", "[K, N]"],
                output_shape="[M, N]",
                file_path=file_path,
                line=int(getattr(node, "lineno", 0) or 0),
                confidence=0.85,
                evidence=["Detected '@' operator"],
            )

        if isinstance(node, ast.Call):
            name = self._call_name(node).lower()
            if name in {"matmul", "mm", "bmm", "dot", "einsum"}:
                expr = self.FLOP_FORMULAS["matmul"]()
                return FLOPEstimate(
                    operation="matmul",
                    formula="2 * M * N * K",
                    estimated_flops=expr,
                    input_shapes=["unknown", "unknown"],
                    output_shape=None,
                    file_path=file_path,
                    line=int(getattr(node, "lineno", 0) or 0),
                    confidence=0.7,
                    evidence=[f"Detected call '{name}'"],
                )
        return None

    def _detect_conv2d(self, node: Any, file_path: str) -> FLOPEstimate | None:
        """Detect convolution-like operations."""
        if not isinstance(node, ast.Call):
            return None
        name = self._call_name(node).lower()
        if "conv" not in name:
            return None

        expr = self.FLOP_FORMULAS["conv2d"]()
        return FLOPEstimate(
            operation="conv2d",
            formula="2 * H_out * W_out * C_out * C_in * K_h * K_w",
            estimated_flops=expr,
            input_shapes=["[B, C_in, H, W]", "[C_out, C_in, K_h, K_w]"],
            output_shape="[B, C_out, H_out, W_out]",
            file_path=file_path,
            line=int(getattr(node, "lineno", 0) or 0),
            confidence=0.68,
            evidence=[f"Detected call '{name}'"],
        )

    def _detect_attention(self, node: Any, file_path: str) -> FLOPEstimate | None:
        """Detect self-attention and attention-variant operations."""
        if not isinstance(node, ast.Call):
            return None
        name = self._call_name(node).lower()
        if not any(token in name for token in ("attention", "attn", "flash", "mqa", "gqa")):
            return None

        formula = str(self.FLOP_FORMULAS["attention"]())
        op = "attention"
        confidence = 0.66
        if "linear" in name:
            formula = str(self.FLOP_FORMULAS["linear_attention"]())
            op = "linear_attention"
            confidence = 0.68
        if "flash" in name or "flash_attn" in name:
            formula = str(self.FLOP_FORMULAS["flash_attention"]())
            op = "flash_attention"
            confidence = 0.74
        if "mqa" in name:
            formula = str(self.FLOP_FORMULAS["mqa_attention"]())
            op = "mqa_attention"
            confidence = 0.71
        if "gqa" in name:
            formula = str(self.FLOP_FORMULAS["gqa_attention"]())
            op = "gqa_attention"
            confidence = 0.71

        return FLOPEstimate(
            operation=op,
            formula=formula,
            estimated_flops=formula,
            input_shapes=["[B, n, d]", "[B, n, d]", "[B, n, d]"],
            output_shape="[B, n, d]",
            file_path=file_path,
            line=int(getattr(node, "lineno", 0) or 0),
            confidence=confidence,
            evidence=[f"Detected attention-like call '{name}'"],
        )

    def _detect_norm(self, node: Any, file_path: str) -> FLOPEstimate | None:
        """Detect layer normalization and RMS normalization operations."""
        if not isinstance(node, ast.Call):
            return None
        name = self._call_name(node).lower()
        if not any(token in name for token in ("layernorm", "layer_norm", "rmsnorm", "rms_norm")):
            return None

        op = "layernorm"
        if "rms" in name:
            op = "rmsnorm"
        formula = str(self.FLOP_FORMULAS[op]())
        return FLOPEstimate(
            operation=op,
            formula=formula,
            estimated_flops=formula,
            input_shapes=["[B, n, d]"],
            output_shape="[B, n, d]",
            file_path=file_path,
            line=int(getattr(node, "lineno", 0) or 0),
            confidence=0.62,
            evidence=[f"Detected normalization call '{name}'"],
        )

    def _detect_embedding(self, node: Any, file_path: str) -> FLOPEstimate | None:
        """Detect embedding lookups as memory-bound FLOP approximations."""
        if not isinstance(node, ast.Call):
            return None
        name = self._call_name(node).lower()
        if not any(token in name for token in ("embedding", "embed", "lookup")):
            return None

        formula = str(self.FLOP_FORMULAS["embedding"]())
        return FLOPEstimate(
            operation="embedding",
            formula=formula,
            estimated_flops=formula,
            input_shapes=["[B, n]"],
            output_shape="[B, n, d]",
            file_path=file_path,
            line=int(getattr(node, "lineno", 0) or 0),
            confidence=0.6,
            evidence=[f"Detected embedding-like call '{name}'"],
        )

    def _detect_loss(self, node: Any, file_path: str) -> FLOPEstimate | None:
        """Detect loss functions with lightweight symbolic FLOP formulas."""
        if not isinstance(node, ast.Call):
            return None
        name = self._call_name(node).lower()
        if not any(token in name for token in ("loss", "cross_entropy", "nll", "mse", "kl_div")):
            return None

        op = "cross_entropy_loss"
        if "mse" in name:
            op = "mse_loss"
        formula = str(self.FLOP_FORMULAS[op]())
        return FLOPEstimate(
            operation=op,
            formula=formula,
            estimated_flops=formula,
            input_shapes=["[B, C]"],
            output_shape="[1]",
            file_path=file_path,
            line=int(getattr(node, "lineno", 0) or 0),
            confidence=0.58,
            evidence=[f"Detected loss call '{name}'"],
        )

    def _detect_activation(self, node: Any, file_path: str) -> FLOPEstimate | None:
        """Detect common activation/elementwise operations."""
        if not isinstance(node, ast.Call):
            return None
        name = self._call_name(node).lower()
        if name not in {"relu", "gelu", "silu", "softmax", "tanh", "sigmoid"}:
            return None

        formula = "n" if name != "softmax" else "5 * n"
        return FLOPEstimate(
            operation="activation",
            formula=formula,
            estimated_flops=formula,
            input_shapes=["[n]"],
            output_shape="[n]",
            file_path=file_path,
            line=int(getattr(node, "lineno", 0) or 0),
            confidence=0.55,
            evidence=[f"Detected activation call '{name}'"],
        )

    def _symbolic_flop_expression(self, shapes: list[str], op: str) -> str:
        """Build a symbolic FLOP expression when dimensions are unknown."""
        op_key = (op or "").strip().lower()
        if op_key in self.FLOP_FORMULAS:
            value = self.FLOP_FORMULAS[op_key]()
            return str(value)

        if shapes:
            dims = " * ".join(self._shape_dims(shapes[0])) or "n"
            return f"{dims}"
        return "unknown"

    @staticmethod
    def _shape_dims(shape: str) -> list[str]:
        text = (shape or "").strip().strip("[]()")
        if not text:
            return []
        return [item.strip() for item in text.split(",") if item.strip()]

    @staticmethod
    def _call_name(node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    @staticmethod
    def _read_text(path: str) -> str:
        if not path:
            return ""
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                return handle.read()
        except Exception:
            return ""

    @staticmethod
    def _find_function_node(source: str, symbol: str) -> ast.AST | None:
        if not source:
            return None
        try:
            tree = ast.parse(source)
        except Exception:
            return None

        fn_name = (symbol or "").split(".")[-1]
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_name:
                return node
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node
        return None


__all__ = ["FLOPCounter", "FLOPEstimate"]
