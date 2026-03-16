from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .component_catalog import ComponentCatalog, ComponentDescriptor
from .spec import SagSpec


@dataclass(slots=True)
class AssemblyCandidate:
    component: ComponentDescriptor
    score: float
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["component"] = self.component.as_dict()
        return payload


@dataclass(slots=True)
class AssemblyPlan:
    spec_digest: str
    selected_components: list[AssemblyCandidate]
    compatibility_evidence: list[str]
    reuse_ratio: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "spec_digest": self.spec_digest,
            "selected_components": [item.as_dict() for item in self.selected_components],
            "compatibility_evidence": list(self.compatibility_evidence),
            "reuse_ratio": self.reuse_ratio,
        }


class ComponentAssembler:
    def rank_components(
        self,
        spec: SagSpec,
        catalog: ComponentCatalog,
        *,
        top_k: int = 5,
    ) -> list[AssemblyCandidate]:
        desired_terms = set(spec.metadata.get("objective_tokens") or [])
        desired_terms.update(path.split("/")[-1].split(".")[0] for path in spec.target_files)
        desired_terms.update(spec.outputs.keys())
        candidates: list[AssemblyCandidate] = []
        for component in catalog.by_language(spec.language):
            score = 0.0
            reasons: list[str] = []
            if component.component_type in {"function", "method", "class"}:
                score += 0.2
                reasons.append("bounded_symbol_type")
            overlap = desired_terms.intersection(set(component.terms))
            if overlap:
                score += min(0.6, len(overlap) * 0.15)
                reasons.append("term_overlap")
            if any(target == component.file_path for target in spec.target_files):
                score += 0.3
                reasons.append("target_file_match")
            if component.contracts:
                score += 0.1
                reasons.append("contract_bearing")
            if score > 0.0:
                candidates.append(AssemblyCandidate(component=component, score=round(score, 3), reasons=reasons))
        return sorted(candidates, key=lambda item: (-item.score, item.component.qualified_name))[:top_k]

    def assemble(
        self,
        spec: SagSpec,
        catalog: ComponentCatalog,
        *,
        top_k: int = 3,
    ) -> AssemblyPlan:
        ranked = self.rank_components(spec, catalog, top_k=top_k)
        evidence = [f"{item.component.qualified_name}:{','.join(item.reasons)}" for item in ranked]
        reuse_ratio = round(min(1.0, len(ranked) / max(1, len(spec.target_files))), 3)
        return AssemblyPlan(
            spec_digest=spec.stable_digest(),
            selected_components=ranked,
            compatibility_evidence=evidence,
            reuse_ratio=reuse_ratio,
        )
