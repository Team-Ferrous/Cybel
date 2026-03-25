"""AES physics templates used to scaffold compliant generated code."""

CONSERVATION_MONITOR = '''
class ConservationMonitor:
    """AES-PHYS-1: Runtime monitor for conservation/invariant drift."""

    def __init__(self, initial_energy: float, tolerance: float = 1e-6):
        self.initial_energy = initial_energy
        self.tolerance = tolerance

    def check(
        self,
        current_energy: float,
        step: int,
        evidence_bundle_id: str,
    ) -> dict[str, float]:
        drift = abs(current_energy - self.initial_energy) / max(
            abs(self.initial_energy),
            1e-12,
        )
        if drift > self.tolerance:
            raise RuntimeError(
                "AES-PHYS-1 violation: energy drift "
                f"{drift:.3e} exceeds {self.tolerance:.3e}"
            )
        return {
            "evidence_bundle_id": evidence_bundle_id,
            "energy_drift": drift,
            "invariant_status": 1.0,
            "conservation_step": float(step),
        }
'''

SYMPLECTIC_INTEGRATOR = '''
def stormer_verlet_step(q, p, dt, grad_potential):
    """AES-PHYS-3: Symplectic step for hamiltonian dynamics."""
    p_half = p - 0.5 * dt * grad_potential(q)
    q_next = q + dt * p_half
    p_next = p_half - 0.5 * dt * grad_potential(q_next)
    return q_next, p_next
'''
