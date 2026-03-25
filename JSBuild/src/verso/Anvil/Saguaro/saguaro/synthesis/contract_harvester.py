from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class HarvestedContract:
    symbol: str
    source_path: str
    contract_type: str
    clause: str
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContractHarvester:
    """Mine light-weight contracts from code, tests, and docs."""

    def harvest_file(self, path: str | Path) -> list[HarvestedContract]:
        target = Path(path)
        if not target.exists():
            return []
        text = target.read_text(encoding="utf-8", errors="ignore")
        if target.suffix == ".py":
            return self._harvest_python(target, text)
        return self._harvest_text(target, text)

    def harvest_paths(self, paths: list[str | Path]) -> list[HarvestedContract]:
        harvested: list[HarvestedContract] = []
        for path in paths:
            harvested.extend(self.harvest_file(path))
        return harvested

    def build_symbol_store(self, paths: list[str | Path]) -> dict[str, list[dict[str, Any]]]:
        store: dict[str, list[dict[str, Any]]] = {}
        for contract in self.harvest_paths(paths):
            store.setdefault(contract.symbol, []).append(contract.as_dict())
        return store

    def _harvest_python(self, path: Path, text: str) -> list[HarvestedContract]:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return []
        harvested: list[HarvestedContract] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                docstring = ast.get_docstring(node)
                if docstring:
                    harvested.append(
                        HarvestedContract(
                            symbol=node.name,
                            source_path=str(path),
                            contract_type="docstring",
                            clause=docstring.splitlines()[0].strip(),
                            confidence=0.82,
                        )
                    )
            if isinstance(node, ast.Assert):
                clause = ast.unparse(node.test) if hasattr(ast, "unparse") else "assert"
                harvested.append(
                    HarvestedContract(
                        symbol=path.stem,
                        source_path=str(path),
                        contract_type="assertion",
                        clause=clause,
                        confidence=0.9,
                    )
                )
        return harvested

    def _harvest_text(self, path: Path, text: str) -> list[HarvestedContract]:
        harvested: list[HarvestedContract] = []
        for match in re.finditer(r"`([^`]+)`", text):
            harvested.append(
                HarvestedContract(
                    symbol=match.group(1),
                    source_path=str(path),
                    contract_type="doc_reference",
                    clause=match.group(1),
                    confidence=0.65,
                )
            )
        return harvested
