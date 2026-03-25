from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class OperatorRule:
    symbol: str
    arity: int
    result_type: str
    commutative: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LibraryContract:
    namespace: str
    symbol: str
    signature: str
    postconditions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EmissionTemplate:
    kind: str
    template: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LanguageSupportPack:
    language: str
    syntax_families: dict[str, list[str]]
    operator_rules: list[OperatorRule]
    library_contracts: list[LibraryContract]
    emission_templates: list[EmissionTemplate]
    unsupported_constructs: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "syntax_families": dict(self.syntax_families),
            "operator_rules": [item.as_dict() for item in self.operator_rules],
            "library_contracts": [item.as_dict() for item in self.library_contracts],
            "emission_templates": [item.as_dict() for item in self.emission_templates],
            "unsupported_constructs": list(self.unsupported_constructs),
            "metadata": dict(self.metadata),
        }

    def coverage_report(self, constructs: list[str]) -> dict[str, Any]:
        supported = sorted(
            construct
            for construct in constructs
            if any(construct in family for family in self.syntax_families.values())
        )
        unsupported = sorted(set(constructs) - set(supported))
        total = max(1, len(constructs))
        return {
            "language": self.language,
            "supported_constructs": supported,
            "unsupported_constructs": unsupported,
            "language_pack_coverage_pct": round((len(supported) / total) * 100.0, 1),
            "unsupported_construct_count": len(unsupported),
            "manual_override_count": int(self.metadata.get("manual_override_count", 0)),
        }


class LanguagePackCompiler:
    def compile(self, packs: list[LanguageSupportPack]) -> dict[str, Any]:
        operator_table = self.compile_operator_table(packs)
        contract_index = self.compile_contract_index(packs)
        template_index = self.compile_template_index(packs)
        return {
            "compiled_rule_count": sum(len(items) for items in operator_table.values()),
            "operator_table": operator_table,
            "contract_index": contract_index,
            "template_index": template_index,
            "languages": [pack.as_dict() for pack in packs],
        }

    def compile_operator_table(self, packs: list[LanguageSupportPack]) -> dict[str, list[dict[str, Any]]]:
        return {
            pack.language: [rule.as_dict() for rule in pack.operator_rules]
            for pack in packs
        }

    def compile_contract_index(self, packs: list[LanguageSupportPack]) -> dict[str, dict[str, dict[str, Any]]]:
        index: dict[str, dict[str, dict[str, Any]]] = {}
        for pack in packs:
            pack_index: dict[str, dict[str, Any]] = {}
            for contract in pack.library_contracts:
                pack_index[f"{contract.namespace}.{contract.symbol}"] = contract.as_dict()
            index[pack.language] = pack_index
        return index

    def compile_template_index(self, packs: list[LanguageSupportPack]) -> dict[str, dict[str, str]]:
        return {
            pack.language: {
                template.kind: template.template for template in pack.emission_templates
            }
            for pack in packs
        }
