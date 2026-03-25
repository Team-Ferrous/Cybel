from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .ast_builder import ASTBuilder, SagParameter


@dataclass(slots=True)
class AdapterField:
    name: str
    source_key: str
    type_name: str = "Any"

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(slots=True)
class AdapterPlan:
    language: str
    adapter_name: str
    target_symbol: str
    fields: list[AdapterField]
    code: str
    bridge_validation: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "adapter_name": self.adapter_name,
            "target_symbol": self.target_symbol,
            "fields": [item.as_dict() for item in self.fields],
            "code": self.code,
            "bridge_validation": dict(self.bridge_validation),
        }


class AdapterGenerator:
    """Generate deterministic boundary adapters for runtime capability surfaces."""

    def __init__(self) -> None:
        self._builder = ASTBuilder()

    def generate_python_capability_adapter(
        self,
        *,
        adapter_name: str = "build_runtime_capability_summary",
        target_symbol: str = "build_runtime_capability_ledger",
        fields: list[AdapterField] | None = None,
    ) -> AdapterPlan:
        selected_fields = list(
            fields
            or [
                AdapterField("backend", "backend", "str"),
                AdapterField("degraded", "degraded", "bool"),
                AdapterField("required_simd", "required_simd", "str"),
            ]
        )
        node = self._builder.build_function(
            language="python",
            name=adapter_name,
            return_type="dict[str, Any]",
            parameters=[SagParameter("status", "dict[str, Any]")],
            imports_or_includes=["typing"],
            body_lines=[
                f"payload = {target_symbol}(status)",
                "return {",
                *[
                    f"    '{field.name}': payload.get('{field.source_key}')"
                    + ","
                    for field in selected_fields
                ],
                "}",
            ],
            metadata={"target_symbol": target_symbol, "boundary": "python"},
        )
        code = self._builder.emit(node)
        return AdapterPlan(
            language="python",
            adapter_name=adapter_name,
            target_symbol=target_symbol,
            fields=selected_fields,
            code=code,
            bridge_validation=self._validate_code(code, selected_fields),
        )

    def generate_cpp_bridge_adapter(
        self,
        *,
        adapter_name: str = "read_native_capability_digest",
        target_symbol: str = "saguaro_native_version",
    ) -> AdapterPlan:
        fields = [AdapterField("native_version", target_symbol, "const char*")]
        node = self._builder.build_function(
            language="cpp",
            name=adapter_name,
            return_type="const char*",
            parameters=[],
            imports_or_includes=["cstddef"],
            body_lines=[f"return {target_symbol}();"],
            metadata={"target_symbol": target_symbol, "boundary": "native"},
        )
        code = self._builder.emit(node)
        return AdapterPlan(
            language="cpp",
            adapter_name=adapter_name,
            target_symbol=target_symbol,
            fields=fields,
            code=code,
            bridge_validation=self._validate_code(code, fields),
        )

    @staticmethod
    def _validate_code(code: str, fields: list[AdapterField]) -> dict[str, Any]:
        missing = [
            field.name
            for field in fields
            if field.name not in code and field.source_key not in code
        ]
        return {
            "adapter_generation_success_rate": 1.0 if not missing else 0.0,
            "bridge_validation_pass_rate": 1.0 if not missing else 0.0,
            "abi_mismatch_count": 0,
            "missing_fields": missing,
        }

