from __future__ import annotations

from saguaro.synthesis.language_pack_compiler import (
    EmissionTemplate,
    LanguageSupportPack,
    LibraryContract,
    OperatorRule,
)


def build_cpp_pack() -> LanguageSupportPack:
    return LanguageSupportPack(
        language="cpp",
        syntax_families={
            "function": ["function_definition", "parameter_list", "return_statement"],
            "container": ["std::vector", "std::array"],
            "include": ["include_directive"],
            "numeric": ["binary_expression", "unary_expression", "call_expression"],
        },
        operator_rules=[
            OperatorRule(symbol="+", arity=2, result_type="double", commutative=True),
            OperatorRule(symbol="-", arity=2, result_type="double"),
            OperatorRule(symbol="*", arity=2, result_type="double", commutative=True),
            OperatorRule(symbol="/", arity=2, result_type="double"),
            OperatorRule(symbol="std::min", arity=2, result_type="double", commutative=True),
            OperatorRule(symbol="std::max", arity=2, result_type="double", commutative=True),
        ],
        library_contracts=[
            LibraryContract("std", "min", "std::min(double, double) -> double", ["result <= lhs or result <= rhs"]),
            LibraryContract("std", "max", "std::max(double, double) -> double", ["result >= lhs or result >= rhs"]),
            LibraryContract("std", "isfinite", "std::isfinite(double) -> bool", ["returns bool"]),
        ],
        emission_templates=[
            EmissionTemplate(
                "function_wrapper",
                "{return_type} {name}({params}) {{\n    {body}\n}}\n",
            ),
            EmissionTemplate("include", "#include <{header}>"),
        ],
        unsupported_constructs=["template_metaprogramming", "inline_assembly", "ub_pointer_aliasing"],
        metadata={"manual_override_count": 0, "bounded_scope": "function_level"},
    )
