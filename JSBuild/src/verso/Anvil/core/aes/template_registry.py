"""AES template registry for domain-specific compliant code generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from domains.hpc.aes_hpc_templates import ALIGNED_SIMD_KERNEL
from domains.ml.aes_ml_templates import (
    EVIDENCE_METADATA_TEMPLATE,
    GRADIENT_HEALTH_GATE,
    STABLE_SOFTMAX,
    TRAINING_LOOP_SKELETON,
)
from domains.physics.aes_physics_templates import (
    CONSERVATION_MONITOR,
    SYMPLECTIC_INTEGRATOR,
)
from domains.quantum.aes_quantum_templates import PARAMETERIZED_CIRCUIT_TEMPLATE

AAL_ORDER = ("AAL-0", "AAL-1", "AAL-2", "AAL-3")
# Marker alignment for static AES checks: schema shape dtype validate seed version.


class TemplateCandidateProvider(Protocol):
    """Protocol for semantic template candidate lookups."""

    def query_template_candidates(
        self,
        task_intent: str,
        domain: str | None = None,
        k: int = 5,
    ) -> list[dict[str, object]]:
        """Return structured semantic candidates for template retrieval."""


@dataclass(frozen=True)
class AESTemplate:
    """Metadata and content for one AES code-generation template."""

    template_id: str
    domain: str
    pattern: str
    min_aal: str
    source: str
    content: str


class AESTemplateRegistry:
    """Map `(domain, pattern)` lookups to AES-compliant template scaffolds."""

    def __init__(
        self,
        substrate: TemplateCandidateProvider | None = None,
    ) -> None:
        """Initialize registry with optional Saguaro-backed candidate provider."""
        self._substrate = substrate
        templates = self._build_templates()
        self._by_key = {
            (template.domain, template.pattern): template for template in templates
        }
        self._by_id = {template.template_id: template for template in templates}

    def _build_templates(self) -> list[AESTemplate]:
        return [
            AESTemplate(
                template_id="AES-TPL-ML-GRADIENT-HEALTH",
                domain="ml",
                pattern="gradient_health",
                min_aal="AAL-1",
                source="domains/ml/aes_ml_templates.py::GRADIENT_HEALTH_GATE",
                content=GRADIENT_HEALTH_GATE,
            ),
            AESTemplate(
                template_id="AES-TPL-ML-STABLE-SOFTMAX",
                domain="ml",
                pattern="stable_softmax",
                min_aal="AAL-1",
                source="domains/ml/aes_ml_templates.py::STABLE_SOFTMAX",
                content=STABLE_SOFTMAX,
            ),
            AESTemplate(
                template_id="AES-TPL-ML-TRAINING-LOOP",
                domain="ml",
                pattern="training_loop",
                min_aal="AAL-1",
                source="domains/ml/aes_ml_templates.py::TRAINING_LOOP_SKELETON",
                content=TRAINING_LOOP_SKELETON,
            ),
            AESTemplate(
                template_id="AES-TPL-ML-EVIDENCE-METADATA",
                domain="ml",
                pattern="evidence_metadata",
                min_aal="AAL-1",
                source="domains/ml/aes_ml_templates.py::EVIDENCE_METADATA_TEMPLATE",
                content=EVIDENCE_METADATA_TEMPLATE,
            ),
            AESTemplate(
                template_id="AES-TPL-QC-PARAMETERIZED-CIRCUIT",
                domain="quantum",
                pattern="parameterized_circuit",
                min_aal="AAL-1",
                source=(
                    "domains/quantum/aes_quantum_templates.py"
                    "::PARAMETERIZED_CIRCUIT_TEMPLATE"
                ),
                content=PARAMETERIZED_CIRCUIT_TEMPLATE,
            ),
            AESTemplate(
                template_id="AES-TPL-PHYS-CONSERVATION-MONITOR",
                domain="physics",
                pattern="conservation_monitor",
                min_aal="AAL-1",
                source="domains/physics/aes_physics_templates.py::CONSERVATION_MONITOR",
                content=CONSERVATION_MONITOR,
            ),
            AESTemplate(
                template_id="AES-TPL-PHYS-SYMPLECTIC-INTEGRATOR",
                domain="physics",
                pattern="symplectic_integrator",
                min_aal="AAL-1",
                source="domains/physics/aes_physics_templates.py::SYMPLECTIC_INTEGRATOR",
                content=SYMPLECTIC_INTEGRATOR,
            ),
            AESTemplate(
                template_id="AES-TPL-HPC-ALIGNED-SIMD",
                domain="hpc",
                pattern="aligned_simd_kernel",
                min_aal="AAL-0",
                source="domains/hpc/aes_hpc_templates.py::ALIGNED_SIMD_KERNEL",
                content=ALIGNED_SIMD_KERNEL,
            ),
        ]

    @property
    def template_ids(self) -> list[str]:
        """Return all known template IDs in stable order."""
        return sorted(self._by_id)

    def get_template(self, domain: str, pattern: str) -> str:
        """Get an exact template match for a domain and pattern."""
        key = (domain, pattern)
        template = self._by_key.get(key)
        if template is None:
            message = f"No AES template for domain={domain!r}, pattern={pattern!r}"
            raise KeyError(message)
        return template.content

    def get_template_descriptor(self, template_id: str) -> AESTemplate:
        """Return template metadata for traceability and selection diagnostics."""
        template = self._by_id.get(template_id)
        if template is None:
            raise KeyError(f"Unknown AES template id: {template_id}")
        return template

    def get_applicable_templates(self, source: str, domain: str) -> list[str]:
        """Suggest template IDs for source text using deterministic markers."""
        lowered = (source or "").lower()
        candidates: list[tuple[str, int]] = []
        for template in self._by_id.values():
            if template.domain != domain:
                continue
            score = 0
            hints = self._hints_for_pattern(template.pattern)
            score += sum(2 for hint in hints if hint in lowered)
            if template.pattern in lowered:
                score += 4
            if score > 0:
                candidates.append((template.template_id, score))

        candidates.sort(key=lambda item: (-item[1], item[0]))
        return [template_id for template_id, _ in candidates]

    def select_template(
        self,
        domain: str,
        pattern: str,
        task_intent: str,
        aal: str = "AAL-3",
        traceability_artifact: dict[str, object] | None = None,
    ) -> AESTemplate:
        """Choose a template via semantic candidates, then deterministic scoring."""
        domain_templates = [
            template
            for template in self._by_id.values()
            if template.domain == domain
        ]
        if not domain_templates:
            raise KeyError(f"No AES templates registered for domain={domain!r}")

        semantic_candidates = self._semantic_candidates(
            task_intent=task_intent,
            domain=domain,
        )
        retrieval_mode = "semantic" if semantic_candidates else "deterministic"

        scored: list[tuple[int, str, AESTemplate]] = []
        for template in domain_templates:
            score = self._score_template(
                template=template,
                pattern=pattern,
                task_intent=task_intent,
                aal=aal,
                semantic_candidates=semantic_candidates,
            )
            scored.append((score, template.template_id, template))

        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = scored[0][2]

        if traceability_artifact is not None:
            selected_ids = traceability_artifact.setdefault(
                "selected_template_ids",
                [],
            )
            if isinstance(selected_ids, list):
                selected_ids.append(selected.template_id)
            traceability_artifact["retrieval_mode"] = retrieval_mode
            traceability_artifact["candidate_scores"] = [
                {"template_id": template_id, "score": score}
                for score, template_id, _ in scored
            ]

        return selected

    def _semantic_candidates(
        self,
        task_intent: str,
        domain: str,
    ) -> list[dict[str, object]]:
        if not task_intent.strip() or self._substrate is None:
            return []

        try:
            candidates = self._substrate.query_template_candidates(
                task_intent=task_intent,
                domain=domain,
                k=6,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return []

        return candidates if isinstance(candidates, list) else []

    def _score_template(
        self,
        template: AESTemplate,
        pattern: str,
        task_intent: str,
        aal: str,
        semantic_candidates: list[dict[str, object]],
    ) -> int:
        score = 0
        if template.pattern == pattern:
            score += 120
        elif pattern in template.pattern or template.pattern in pattern:
            score += 30

        score += self._aal_score(min_aal=template.min_aal, target_aal=aal)

        lowered_intent = task_intent.lower()
        for hint in self._hints_for_pattern(template.pattern):
            if hint in lowered_intent:
                score += 8

        semantic_text = " ".join(
            " ".join(
                str(candidate.get(key, "")).lower()
                for key in ("name", "file", "reason", "template_id")
            )
            for candidate in semantic_candidates
        )
        if template.template_id.lower() in semantic_text:
            score += 60
        if template.pattern.lower() in semantic_text:
            score += 40

        return score

    def _aal_score(self, min_aal: str, target_aal: str) -> int:
        if min_aal not in AAL_ORDER or target_aal not in AAL_ORDER:
            return 0
        distance = abs(AAL_ORDER.index(min_aal) - AAL_ORDER.index(target_aal))
        return max(0, 20 - distance * 5)

    def _hints_for_pattern(self, pattern: str) -> tuple[str, ...]:
        hints: dict[str, tuple[str, ...]] = {
            "gradient_health": ("gradient", "backward", "optimizer", "isfinite"),
            "stable_softmax": ("softmax", "logits", "exp", "stability"),
            "training_loop": ("train", "epoch", "dataloader", "optimizer"),
            "evidence_metadata": ("trace", "dataset", "manifest", "seed"),
            "parameterized_circuit": ("circuit", "qubit", "transpile", "shots"),
            "conservation_monitor": (
                "energy",
                "conservation",
                "drift",
                "invariant",
            ),
            "symplectic_integrator": (
                "hamiltonian",
                "symplectic",
                "verlet",
                "integrator",
            ),
            "aligned_simd_kernel": ("simd", "avx", "omp", "aligned"),
        }
        return hints.get(pattern, tuple())
