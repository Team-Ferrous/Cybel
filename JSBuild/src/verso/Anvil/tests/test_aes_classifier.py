from pathlib import Path

from core.aes import AALClassifier, DomainDetector, AESRuleRegistry


def test_aal_classifier_is_deterministic(tmp_path: Path):
    target = tmp_path / "kernel.cpp"
    target.write_text("#pragma omp parallel\n_mm256_add_ps(a, b);\n", encoding="utf-8")

    classifier = AALClassifier()
    first = classifier.classify_file(str(target))
    second = classifier.classify_file(str(target))

    assert first == "AAL-0"
    assert first == second


def test_aal_changeset_uses_strictest_file(tmp_path: Path):
    doc = tmp_path / "notes.md"
    doc.write_text("documentation only", encoding="utf-8")
    train = tmp_path / "train.py"
    train.write_text("loss.backward()\noptimizer.step()\n", encoding="utf-8")

    classifier = AALClassifier()
    assert classifier.classify_changeset([str(doc), str(train)]) == "AAL-1"


def test_domain_detector_finds_markers(tmp_path: Path):
    target = tmp_path / "trainer.py"
    target.write_text(
        "import torch\nfrom qiskit import QuantumCircuit\n# pragma omp parallel\n",
        encoding="utf-8",
    )

    domains = DomainDetector().detect_domains([str(target)])
    assert {"ml", "quantum", "hpc"}.issubset(domains)


def test_rule_registry_loads_and_resolves_callable():
    registry = AESRuleRegistry()
    registry.load("standards/AES_RULES.json")

    assert registry.get_rules_for_domain("ml")
    assert registry.get_rules_for_engine("semantic")
    assert callable(registry.get_check_function("AES-ML-1"))
