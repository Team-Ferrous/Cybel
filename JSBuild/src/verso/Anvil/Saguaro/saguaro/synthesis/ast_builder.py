from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SagParameter:
    name: str
    type_name: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(slots=True)
class SagFunction:
    language: str
    name: str
    return_type: str
    parameters: list[SagParameter]
    body_lines: list[str]
    imports_or_includes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "name": self.name,
            "return_type": self.return_type,
            "parameters": [item.as_dict() for item in self.parameters],
            "body_lines": list(self.body_lines),
            "imports_or_includes": list(self.imports_or_includes),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class SagASTNode:
    kind: str
    language: str
    payload: dict[str, Any]
    children: list["SagASTNode"] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "language": self.language,
            "payload": dict(self.payload),
            "children": [child.as_dict() for child in self.children],
        }


class ASTBuilder:
    """Deterministically construct function-level source artifacts."""

    def build_function(
        self,
        *,
        language: str,
        name: str,
        return_type: str,
        parameters: list[SagParameter],
        body_lines: list[str],
        imports_or_includes: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SagASTNode:
        function = SagFunction(
            language=language,
            name=name,
            return_type=return_type,
            parameters=list(parameters),
            body_lines=[str(line).rstrip() for line in body_lines],
            imports_or_includes=[
                str(item).strip()
                for item in list(imports_or_includes or [])
                if str(item).strip()
            ],
            metadata=dict(metadata or {}),
        )
        return SagASTNode(
            kind="function",
            language=language,
            payload=function.as_dict(),
        )

    def emit(self, node: SagASTNode) -> str:
        if node.kind != "function":
            raise ValueError(f"Unsupported AST node kind: {node.kind}")
        payload = dict(node.payload)
        language = str(payload.get("language") or node.language or "python")
        if language == "python":
            return self._emit_python(payload)
        if language == "cpp":
            return self._emit_cpp(payload)
        raise ValueError(f"Unsupported language: {language}")

    def roundtrip_report(self, node: SagASTNode) -> dict[str, Any]:
        emitted = self.emit(node)
        language = str(node.language or node.payload.get("language") or "python")
        if language == "python":
            ast.parse(emitted)
            return {
                "language": language,
                "syntax_ok": True,
                "emit_roundtrip_fidelity": 1.0,
                "line_count": len(emitted.splitlines()),
            }
        braces_balanced = emitted.count("{") == emitted.count("}")
        parens_balanced = emitted.count("(") == emitted.count(")")
        syntax_ok = braces_balanced and parens_balanced and emitted.rstrip().endswith("}")
        return {
            "language": language,
            "syntax_ok": syntax_ok,
            "emit_roundtrip_fidelity": 1.0 if syntax_ok else 0.0,
            "line_count": len(emitted.splitlines()),
        }

    def _emit_python(self, payload: dict[str, Any]) -> str:
        imports = [
            f"import {item}"
            for item in payload.get("imports_or_includes") or []
            if item
        ]
        params = ", ".join(
            f"{item['name']}: {item['type_name']}"
            for item in payload.get("parameters") or []
        )
        body_lines = [str(line).rstrip() for line in payload.get("body_lines") or []]
        body = "\n".join(f"    {line}" if line else "" for line in (body_lines or ["pass"]))
        function_block = (
            f"def {payload['name']}({params}) -> {payload['return_type']}:\n{body}"
        ).rstrip()
        if imports:
            return "\n".join([*imports, "", function_block]) + "\n"
        return function_block + "\n"

    def _emit_cpp(self, payload: dict[str, Any]) -> str:
        includes = [
            f"#include <{item}>"
            for item in payload.get("imports_or_includes") or []
            if item
        ]
        params = ", ".join(
            f"{item['type_name']} {item['name']}"
            for item in payload.get("parameters") or []
        )
        body_lines = [str(line).rstrip() for line in payload.get("body_lines") or []]
        body = "\n".join(
            f"    {line}" if line else ""
            for line in (body_lines or [f"return {payload['return_type']}{{}};"])
        )
        function_block = (
            f"{payload['return_type']} {payload['name']}({params}) {{\n{body}\n}}"
        ).rstrip()
        if includes:
            return "\n".join([*includes, "", function_block]) + "\n"
        return function_block + "\n"


class Emitter:
    def __init__(self) -> None:
        self._builder = ASTBuilder()

    def emit(self, node: SagASTNode) -> str:
        return self._builder.emit(node)
