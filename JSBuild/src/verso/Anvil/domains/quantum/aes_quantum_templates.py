"""AES quantum templates used to scaffold compliant generated code."""

PARAMETERIZED_CIRCUIT_TEMPLATE = '''
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter


def build_parameterized_circuit(
    backend,
    noise_model,
    layer: int,
    evidence_bundle_id: str,
):
    # AES-QC-2: named parameters instead of magic-angle literals
    theta = Parameter(f"theta_layer_{layer}")
    phi = Parameter(f"phi_layer_{layer}")

    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.rx(theta, 0)
    circuit.ry(phi, 1)
    circuit.measure([0, 1], [0, 1])

    # AES-QC-4: transpilation required before backend execution
    transpilation_config = {"optimization_level": 2, "seed_transpiler": 7}
    transpiled = transpile(circuit, backend=backend, **transpilation_config)

    run_metadata = {
        "trace_id": f"trace::{evidence_bundle_id}",
        "evidence_bundle_id": evidence_bundle_id,
        "noise_model": str(noise_model),
        "shots": 4096,
        "shot_confidence_target": 0.99,
        "backend": backend.name(),
        "transpiler_settings": transpilation_config,
    }

    job = backend.run(transpiled, shots=4096, noise_model=noise_model)
    return job, run_metadata
'''
