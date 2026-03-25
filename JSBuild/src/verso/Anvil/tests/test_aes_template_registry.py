from core.aes import AESTemplateRegistry
from core.aes.checks.hpc_checks import (
    check_alignment_contracts,
    check_explicit_omp_clauses,
    check_scalar_reference_impl,
)
from core.aes.checks.ml_checks import (
    check_data_validation,
    check_gradient_health_gate,
    check_reproducibility_manifest,
    check_stable_numerics,
)
from core.aes.checks.physics_checks import (
    check_conservation_monitors,
    check_symplectic_integrator,
)
from core.aes.checks.quantum_checks import (
    check_no_magic_angles,
    check_noise_model_present,
    check_shot_sufficiency,
    check_transpilation_required,
)
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


class _SemanticStub:
    def query_template_candidates(
        self,
        task_intent: str,
        domain: str | None = None,
        k: int = 5,
    ):
        return [
            {
                "template_id": "AES-TPL-ML-TRAINING-LOOP",
                "name": "AES-TPL-ML-TRAINING-LOOP",
                "file": "domains/ml/aes_ml_templates.py",
                "reason": "training loop scaffold",
            }
        ]


def test_template_registry_exact_lookup():
    registry = AESTemplateRegistry()
    template = registry.get_template("ml", "training_loop")
    assert "optimizer.step()" in template
    assert "evidence_bundle_id" in template


def test_template_registry_applicable_templates_are_ranked():
    registry = AESTemplateRegistry()
    source = "train epoch dataloader optimizer backward stability"
    applicable = registry.get_applicable_templates(source=source, domain="ml")

    assert applicable
    assert applicable[0] in {
        "AES-TPL-ML-TRAINING-LOOP",
        "AES-TPL-ML-GRADIENT-HEALTH",
    }


def test_ml_templates_align_with_phase1_checks():
    ml_source = "\n".join(
        [
            GRADIENT_HEALTH_GATE,
            STABLE_SOFTMAX,
            EVIDENCE_METADATA_TEMPLATE,
            TRAINING_LOOP_SKELETON,
        ]
    )

    assert check_gradient_health_gate(ml_source, "ml_template.py") == []
    assert check_stable_numerics(ml_source, "ml_template.py") == []
    assert check_data_validation(ml_source, "ml_template.py") == []
    assert check_reproducibility_manifest(ml_source, "ml_template.py") == []


def test_quantum_template_aligns_with_phase1_checks():
    source = PARAMETERIZED_CIRCUIT_TEMPLATE

    assert check_no_magic_angles(source, "quantum_template.py") == []
    assert check_transpilation_required(source, "quantum_template.py") == []
    assert check_noise_model_present(source, "quantum_template.py") == []
    assert check_shot_sufficiency(source, "quantum_template.py") == []


def test_physics_templates_align_with_phase1_checks():
    source = "\n".join([CONSERVATION_MONITOR, SYMPLECTIC_INTEGRATOR])

    assert check_conservation_monitors(source, "physics_template.py") == []
    assert check_symplectic_integrator(source, "physics_template.py") == []


def test_hpc_template_aligns_with_phase1_checks():
    source = ALIGNED_SIMD_KERNEL

    assert check_alignment_contracts(source, "hpc_template.h") == []
    assert check_explicit_omp_clauses(source, "hpc_template.h") == []
    assert check_scalar_reference_impl(source, "hpc_template.h") == []


def test_adversarial_negative_snippets_trigger_domain_violations():
    ml_bad = (
        "def train(model, optimizer, x):\n"
        "    loss = model(x)\n"
        "    loss.backward()\n"
        "    optimizer.step()\n"
    )
    quantum_bad = "backend.run(circuit, shots=64)\n"
    physics_bad = (
        "def step_hamiltonian(q, p):\n"
        "    # hamiltonian update with euler\n"
        "    return q + p\n"
    )
    hpc_bad = "__m256 v;\n#pragma omp parallel for\nfor (int i=0; i<n; ++i) {}\n"

    assert check_gradient_health_gate(ml_bad, "ml_bad.py")
    assert check_transpilation_required(quantum_bad, "quantum_bad.py")
    assert check_shot_sufficiency(quantum_bad, "quantum_bad.py")
    assert check_symplectic_integrator(physics_bad, "physics_bad.py")
    assert check_alignment_contracts(hpc_bad, "hpc_bad.h")
    assert check_explicit_omp_clauses(hpc_bad, "hpc_bad.h")


def test_registry_semantic_selection_emits_traceability():
    registry = AESTemplateRegistry(substrate=_SemanticStub())
    traceability_artifact = {}

    selected = registry.select_template(
        domain="ml",
        pattern="training_loop",
        task_intent="generate an aes training loop for gradient checks",
        aal="AAL-1",
        traceability_artifact=traceability_artifact,
    )

    assert selected.template_id == "AES-TPL-ML-TRAINING-LOOP"
    assert traceability_artifact["selected_template_ids"] == [
        "AES-TPL-ML-TRAINING-LOOP"
    ]
    assert traceability_artifact["retrieval_mode"] == "semantic"
    assert traceability_artifact["candidate_scores"]
