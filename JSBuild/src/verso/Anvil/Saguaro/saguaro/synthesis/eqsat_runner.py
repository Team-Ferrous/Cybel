from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class RewriteApplication:
    rule: str
    before: str
    after: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(slots=True)
class EqsatResult:
    original: str
    optimized: str
    rewrites: list[RewriteApplication] = field(default_factory=list)
    telemetry: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "optimized": self.optimized,
            "rewrites": [item.as_dict() for item in self.rewrites],
            "telemetry": dict(self.telemetry),
        }


class BoundedEqsatRunner:
    """Apply a bounded, deterministic rewrite catalog to small expressions."""

    def __init__(self, *, node_limit: int = 32) -> None:
        self.node_limit = node_limit

    def optimize_expression(self, expression: str) -> EqsatResult:
        tree = ast.parse(expression, mode="eval")
        node_count = sum(1 for _ in ast.walk(tree))
        if node_count > self.node_limit:
            return EqsatResult(
                original=expression,
                optimized=expression,
                rewrites=[],
                telemetry={
                    "egraph_node_count": node_count,
                    "rewrite_fire_count": 0,
                    "extract_cost_delta": 0,
                    "memory_ceiling_hits": 1,
                },
            )
        rewrites: list[RewriteApplication] = []
        optimized_node = self._rewrite(tree.body, rewrites)
        optimized = ast.unparse(optimized_node)
        return EqsatResult(
            original=expression,
            optimized=optimized,
            rewrites=rewrites,
            telemetry={
                "egraph_node_count": node_count,
                "rewrite_fire_count": len(rewrites),
                "extract_cost_delta": len(expression) - len(optimized),
                "memory_ceiling_hits": 0,
            },
        )

    def _rewrite(self, node: ast.AST, rewrites: list[RewriteApplication]) -> ast.AST:
        if isinstance(node, ast.BinOp):
            node.left = self._rewrite(node.left, rewrites)
            node.right = self._rewrite(node.right, rewrites)
            if isinstance(node.op, ast.Add) and self._is_constant(node.right, 0):
                rewrites.append(
                    RewriteApplication("add_zero_rhs", ast.unparse(node), ast.unparse(node.left))
                )
                return node.left
            if isinstance(node.op, ast.Add) and self._is_constant(node.left, 0):
                rewrites.append(
                    RewriteApplication("add_zero_lhs", ast.unparse(node), ast.unparse(node.right))
                )
                return node.right
            if isinstance(node.op, ast.Mult) and self._is_constant(node.right, 1):
                rewrites.append(
                    RewriteApplication("mul_one_rhs", ast.unparse(node), ast.unparse(node.left))
                )
                return node.left
            if isinstance(node.op, ast.Mult) and self._is_constant(node.left, 1):
                rewrites.append(
                    RewriteApplication("mul_one_lhs", ast.unparse(node), ast.unparse(node.right))
                )
                return node.right
        return node

    @staticmethod
    def _is_constant(node: ast.AST, value: int | float) -> bool:
        return isinstance(node, ast.Constant) and node.value == value

