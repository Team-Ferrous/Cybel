from __future__ import annotations

from saguaro.synthesis.language_pack_compiler import (
    EmissionTemplate,
    LanguageSupportPack,
    LibraryContract,
    OperatorRule,
)


def build_python_pack() -> LanguageSupportPack:
    return LanguageSupportPack(
        language="python",
        syntax_families={
            "function": ["function_def", "arguments", "return"],
            "container": ["list", "dict", "tuple"],
            "import": ["import", "from_import"],
            "numeric": ["binop", "compare", "unaryop"],
        },
        operator_rules=[
            OperatorRule(symbol="+", arity=2, result_type="float", commutative=True),
            OperatorRule(symbol="-", arity=2, result_type="float"),
            OperatorRule(symbol="*", arity=2, result_type="float", commutative=True),
            OperatorRule(symbol="/", arity=2, result_type="float"),
            OperatorRule(symbol="min", arity=2, result_type="float", commutative=True),
            OperatorRule(symbol="max", arity=2, result_type="float", commutative=True),
        ],
        library_contracts=[
            LibraryContract("builtins", "min", "min(a: float, b: float) -> float", ["result <= a or result <= b"]),
            LibraryContract("builtins", "max", "max(a: float, b: float) -> float", ["result >= a or result >= b"]),
            LibraryContract("math", "isfinite", "isfinite(value: float) -> bool", ["returns bool"]),
        ],
        emission_templates=[
            EmissionTemplate(
                "function_wrapper",
                "def {name}({params}) -> {return_type}:\n    {body}\n",
            ),
            EmissionTemplate("import", "import {module}"),
        ],
        unsupported_constructs=["metaclass_mutation", "bytecode_codegen", "dynamic_exec"],
        metadata={"manual_override_count": 0, "bounded_scope": "function_level"},
    )
