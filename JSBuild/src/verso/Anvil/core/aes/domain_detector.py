import re
from pathlib import Path
from typing import Iterable, Set


class DomainDetector:
    """Deterministic domain detection based on imports and content markers."""

    IMPORT_MARKERS = {
        "ml": ["torch", "tensorflow", "keras", "sklearn", "jax", "optax"],
        "quantum": ["qiskit", "cirq", "pennylane", "braket"],
        "physics": ["scipy.integrate", "sympy.physics", "fenics", "fipy"],
        "hpc": ["immintrin.h", "omp.h", "mpi.h", "cuda_runtime"],
    }

    CONTENT_MARKERS = {
        "ml": [r"optimizer|backward|gradient|loss_fn|train_step"],
        "quantum": [r"quantum_circuit|qubit|entangle|transpile"],
        "physics": [r"conservation|hamiltonian|lagrangian|symplectic"],
        "hpc": [r"simd|avx2|vectorize|parallel_for|thread_pool|#?\s*pragma\s+omp"],
    }

    def __init__(self) -> None:
        self._compiled = {
            domain: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for domain, patterns in self.CONTENT_MARKERS.items()
        }

    def _read_file(self, filepath: str) -> str:
        path = Path(filepath)
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return ""

    def detect_domains(self, files: Iterable[str]) -> Set[str]:
        domains: Set[str] = set()
        for filepath in files:
            content = self._read_file(filepath)
            domains.update(self.detect_from_text(content))
        return domains

    def detect_from_text(self, content: str) -> Set[str]:
        text = content or ""
        lowered = text.lower()
        domains: Set[str] = set()
        for domain, imports in self.IMPORT_MARKERS.items():
            if any(marker.lower() in lowered for marker in imports):
                domains.add(domain)
                continue
            if any(pattern.search(text) for pattern in self._compiled[domain]):
                domains.add(domain)
        return domains

    def detect_from_description(self, description: str) -> Set[str]:
        """Alias used by orchestration code for textual task routing."""
        return self.detect_from_text(description)
