"""Quantum pack."""

from .base import PackSpec

QUANTUM_PACK = PackSpec(
    name="quantum_pack",
    description="Quantum circuits, state vectors, measurement, and operator semantics.",
    keywords=["quantum", "hamiltonian", "statevector", "measurement", "qubit", "operator"],
    languages=["python", "cpp", "rust"],
    file_hints=["quantum", "hamiltonian", "circuit", "statevector"],
)
