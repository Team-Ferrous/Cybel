from core.aes.checks.physics_checks import (
    check_conservation_monitors,
    check_symplectic_integrator,
)


def test_conservation_monitors_flag_missing_invariant_tracking() -> None:
    source = """
def advance_hamiltonian_state(state):
    energy = state.hamiltonian
    momentum = state.momentum
    return energy + momentum
"""
    violations = check_conservation_monitors(source, "physics.py")
    assert len(violations) == 1
    assert violations[0]["rule_id"] == "AES-PHYS-1"


def test_conservation_monitors_pass_with_drift_monitoring() -> None:
    source = """
def advance_hamiltonian_state(state):
    energy = state.hamiltonian
    conservation_drift = 0.0
    return energy, conservation_drift
"""
    assert check_conservation_monitors(source, "physics.py") == []


def test_symplectic_integrator_flags_rk4_for_hamiltonian_system() -> None:
    source = """
def step(state):
    hamiltonian = state.hamiltonian
    return rk4(state, hamiltonian)
"""
    violations = check_symplectic_integrator(source, "integrator.py")
    assert len(violations) == 1
    assert violations[0]["rule_id"] == "AES-PHYS-3"


def test_symplectic_integrator_pass_with_symplectic_marker() -> None:
    source = """
def step(state):
    hamiltonian = state.hamiltonian
    return symplectic_step(state, hamiltonian)
"""
    assert check_symplectic_integrator(source, "integrator.py") == []
