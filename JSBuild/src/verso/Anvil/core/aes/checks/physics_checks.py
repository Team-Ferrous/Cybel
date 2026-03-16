from typing import Any


def _violation(rule_id: str, filepath: str, line: int, message: str) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "filepath": filepath,
        "line": line,
        "message": message,
    }


def check_conservation_monitors(source: str, filepath: str) -> list[dict[str, Any]]:
    lowered = source.lower()
    if any(token in lowered for token in ("hamiltonian", "momentum", "energy")) and not any(
        token in lowered for token in ("conservation", "drift", "invariant")
    ):
        return [
            _violation(
                "AES-PHYS-1",
                filepath,
                1,
                "Physics-critical code lacks conservation/invariant monitoring markers",
            )
        ]
    return []


def check_symplectic_integrator(source: str, filepath: str) -> list[dict[str, Any]]:
    lowered = source.lower()
    if "hamiltonian" in lowered and any(token in lowered for token in ("rk4", "euler")):
        return [
            _violation(
                "AES-PHYS-3",
                filepath,
                1,
                "Hamiltonian system uses non-symplectic integrator marker",
            )
        ]
    return []
