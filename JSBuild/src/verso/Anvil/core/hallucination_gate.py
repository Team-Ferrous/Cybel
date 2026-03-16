import re
from typing import Dict, List, Tuple


class HallucinationGate:
    """Validate synthesized responses against loaded evidence."""

    entity_pattern = re.compile(r"`([A-Z][a-zA-Z0-9_]+(?:\.[a-z_][a-zA-Z0-9_]*)*)`")
    method_pattern = re.compile(r"`([a-z_][a-zA-Z0-9_]*)\(\)`")

    def validate(self, response: str, evidence: Dict) -> Tuple[str, List[str]]:
        known_entities = self._extract_known_entities(evidence)
        violations: List[str] = []

        for match in self.entity_pattern.finditer(response or ""):
            entity = match.group(1)
            base_name = entity.split(".")[0]
            if entity not in known_entities and base_name not in known_entities:
                violations.append(f"Unverified entity: `{entity}`")

        for match in self.method_pattern.finditer(response or ""):
            method = match.group(1)
            if method not in known_entities:
                violations.append(f"Unverified method: `{method}()`")

        if violations:
            warning = [
                "",
                "",
                "> **Grounding Warning**: The following references could not be verified against loaded evidence:",
            ]
            warning.extend(f"> - {violation}" for violation in violations)
            response = (response or "") + "\n".join(warning)

        return response, violations

    def _extract_known_entities(self, evidence: Dict) -> set:
        known = set()

        for skeleton in evidence.get("skeletons", {}).values():
            for match in re.finditer(r"(?:class|def)\s+([A-Za-z_]\w*)", skeleton):
                known.add(match.group(1))

        for content in evidence.get("file_contents", {}).values():
            for match in re.finditer(r"(?:class|def)\s+([A-Za-z_]\w*)", content):
                known.add(match.group(1))

        for name in evidence.get("entities", {}).keys():
            known.add(name)

        for tree_view in evidence.get("tree_views", {}).values():
            for entry in tree_view:
                name = entry.get("name")
                if name:
                    known.add(name)

        return known
