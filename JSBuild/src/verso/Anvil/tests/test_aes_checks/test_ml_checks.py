from core.aes.checks.ml_checks import (
    check_data_validation,
    check_gradient_health_gate,
    check_reproducibility_manifest,
    check_stable_numerics,
)


def test_gradient_health_gate_flags_missing_finite_check() -> None:
    source = """
def train(model, optimizer, loss):
    loss.backward()
    optimizer.step()
"""
    violations = check_gradient_health_gate(source, "trainer.py")
    assert len(violations) == 1
    assert violations[0]["rule_id"] == "AES-ML-1"


def test_gradient_health_gate_passes_with_finite_check() -> None:
    source = """
def train(model, optimizer, loss):
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite")
    optimizer.step()
"""
    assert check_gradient_health_gate(source, "trainer.py") == []


def test_gradient_health_gate_ignores_internal_checker_module() -> None:
    source = """
def check_gradient_health_gate(source: str, filepath: str):
    optimizer.step()
"""
    assert (
        check_gradient_health_gate(source, "core/aes/checks/ml_checks.py") == []
    )


def test_stable_numerics_flags_unbounded_exp_usage() -> None:
    source = """
def logits(x):
    return torch.exp(x) + np.exp(x)
"""
    violations = check_stable_numerics(source, "numerics.py")
    assert len(violations) == 2
    assert {item["rule_id"] for item in violations} == {"AES-ML-2"}


def test_data_validation_flags_dataset_without_schema_checks() -> None:
    source = """
def load():
    dataset = read_csv("samples.csv")
    return dataset
"""
    violations = check_data_validation(source, "data.py")
    assert len(violations) == 1
    assert violations[0]["rule_id"] == "AES-ML-4"


def test_data_validation_passes_with_shape_and_dtype_markers() -> None:
    source = """
def load_and_validate():
    dataset = read_csv("samples.csv")
    validate_schema(dataset)
    assert dataset.shape[1] > 0
    assert dataset.dtype is not None
"""
    assert check_data_validation(source, "data.py") == []


def test_reproducibility_manifest_flags_missing_seed_or_version() -> None:
    source = """
def train_loop(model, optimizer):
    for epoch in range(2):
        optimizer.step()
"""
    violations = check_reproducibility_manifest(source, "train.py")
    assert len(violations) == 1
    assert violations[0]["rule_id"] == "AES-ML-5"


def test_reproducibility_manifest_passes_with_seed_and_version_markers() -> None:
    source = """
def train_loop(model, optimizer):
    seed = 7
    manifest_version = "v1"
    for epoch in range(2):
        optimizer.step()
"""
    assert check_reproducibility_manifest(source, "train.py") == []
