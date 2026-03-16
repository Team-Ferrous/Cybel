"""Typed math IR records used by Saguaro's static math pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class MathComplexity:
    """A lightweight structural complexity summary for one expression."""

    operator_count: int
    symbol_count: int
    function_call_count: int
    max_nesting_depth: int
    token_count: int
    structural_score: int
    band: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LoopFrame:
    """Loop context attached to a math record."""

    loop_kind: str
    loop_variables: list[str] = field(default_factory=list)
    nesting_depth: int = 0
    bounds_hint: str = ""
    reduction: bool = False
    recurrence: bool = False
    reduction_symbol: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AccessSignature:
    """A typed view of one memory access inside a math record."""

    base_symbol: str
    access_kind: str
    index_expression: str
    index_affinity: str
    stride_class: str
    reuse_hint: str
    write_mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LayoutState:
    """A symbol-level layout hint derived from the observed accesses."""

    symbol: str
    layout: str
    contiguous: bool
    alias_risk: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ComplexityReductionHint:
    """A safe algebraic simplification or reformulation suggestion."""

    kind: str
    summary: str
    estimated_cost_delta: float
    confidence: float
    safe: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MathIRRecord:
    """Typed mathematical statement extracted from docs or code."""

    id: str
    file: str
    expression: str
    normalized_expression: str
    line_start: int
    line_end: int
    symbols: list[str]
    language: str
    source_kind: str
    statement_kind: str
    lhs: str = ""
    rhs: str = ""
    complexity: MathComplexity | None = None
    loop_context: LoopFrame | None = None
    access_signatures: list[AccessSignature] = field(default_factory=list)
    layout_states: list[LayoutState] = field(default_factory=list)
    complexity_reduction_hints: list[ComplexityReductionHint] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.complexity is not None:
            payload["complexity"] = self.complexity.to_dict()
        if self.loop_context is not None:
            payload["loop_context"] = self.loop_context.to_dict()
        payload["access_signatures"] = [item.to_dict() for item in self.access_signatures]
        payload["layout_states"] = [item.to_dict() for item in self.layout_states]
        payload["complexity_reduction_hints"] = [
            item.to_dict() for item in self.complexity_reduction_hints
        ]
        return payload
