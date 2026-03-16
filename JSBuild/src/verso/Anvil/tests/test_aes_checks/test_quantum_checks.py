from core.aes.checks.quantum_checks import (
    check_no_magic_angles,
    check_noise_model_present,
    check_shot_sufficiency,
    check_transpilation_required,
)


def test_no_magic_angles_flags_literal_gate_parameter() -> None:
    source = """
def build(circuit):
    circuit.rx(3.14159, 0)
"""
    violations = check_no_magic_angles(source, "circuit.py")
    assert len(violations) == 1
    assert violations[0]["rule_id"] == "AES-QC-2"


def test_no_magic_angles_pass_with_symbolic_parameter() -> None:
    source = """
def build(circuit, theta):
    circuit.rx(theta, 0)
"""
    assert check_no_magic_angles(source, "circuit.py") == []


def test_transpilation_required_flags_backend_run_without_transpile() -> None:
    source = """
def execute(circuit, backend):
    return backend.run(circuit)
"""
    violations = check_transpilation_required(source, "run.py")
    assert len(violations) == 1
    assert violations[0]["rule_id"] == "AES-QC-4"


def test_transpilation_required_pass_with_transpile_marker() -> None:
    source = """
def execute(circuit, backend):
    compiled = transpile(circuit, backend)
    return backend.run(compiled)
"""
    assert check_transpilation_required(source, "run.py") == []


def test_noise_model_present_flags_simulator_without_noise() -> None:
    source = """
def run(simulator, circuit):
    return simulator.run(circuit)
"""
    violations = check_noise_model_present(source, "noise.py")
    assert len(violations) == 1
    assert violations[0]["rule_id"] == "AES-QC-3"


def test_shot_sufficiency_flags_low_shot_count() -> None:
    source = """
def run(backend, circuit):
    return backend.run(circuit, shots=64)
"""
    violations = check_shot_sufficiency(source, "shots.py")
    assert len(violations) == 1
    assert violations[0]["rule_id"] == "AES-QC-5"


def test_shot_sufficiency_passes_for_high_shot_count() -> None:
    source = """
def run(backend, circuit):
    return backend.run(circuit, shots=1024)
"""
    assert check_shot_sufficiency(source, "shots.py") == []
