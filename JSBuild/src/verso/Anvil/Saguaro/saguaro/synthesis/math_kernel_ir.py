from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from typing import Any

from .ast_builder import ASTBuilder, SagParameter


@dataclass(slots=True)
class MathKernelIR:
    expression: str
    variables: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class MathKernelCompiler:
    def parse(self, expression: str) -> MathKernelIR:
        tree = ast.parse(expression, mode="eval")
        variables = sorted(
            {
                node.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Name) and node.id not in {"min", "max", "abs"}
            }
        )
        return MathKernelIR(expression=expression, variables=variables)

    def lower(self, ir: MathKernelIR, *, language: str, function_name: str) -> str:
        builder = ASTBuilder()
        params = [SagParameter(name, "double" if language == "cpp" else "float") for name in ir.variables]
        if language == "cpp":
            node = builder.build_function(
                language="cpp",
                name=function_name,
                return_type="double",
                parameters=params,
                imports_or_includes=["algorithm"],
                body_lines=[f"return {ir.expression};"],
            )
        else:
            node = builder.build_function(
                language="python",
                name=function_name,
                return_type="float",
                parameters=params,
                imports_or_includes=[],
                body_lines=[f"return {ir.expression}"],
            )
        return builder.emit(node)

