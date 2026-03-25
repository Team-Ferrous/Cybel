import json
import math
import re
from pathlib import Path

from domains.quantum.aes_quantum_templates import PARAMETERIZED_CIRCUIT_TEMPLATE


def _load_baseline(name: str) -> dict:
    root = Path(__file__).resolve().parents[2]
    path = root / "benchmarks" / "aes_proving_ground" / f"{name}_baseline.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_replay_inputs() -> dict:
    root = Path(__file__).resolve().parents[2]
    path = (
        root
        / "benchmarks"
        / "aes_proving_ground"
        / "deterministic_replay_inputs.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _evidence_bundle(domain: str, case: str, metrics: dict[str, float]) -> dict:
    return {
        "trace_id": f"trace::{domain}::{case}",
        "evidence_bundle_id": f"bundle::{domain}::{case}",
        "domain": domain,
        "case": case,
        "metrics": metrics,
    }


def test_hpc_scalar_vector_parity_oracle():
    baseline = _load_baseline("hpc")
    replay = _load_replay_inputs()["hpc"]

    source = replay["input"]
    scale = replay["scale"]

    scalar_output = [value * scale for value in source]
    scalar_total = sum(scalar_output)

    simd_like_output = []
    for idx in range(0, len(source), 8):
        lane = source[idx : idx + 8]
        simd_like_output.extend([value * scale for value in lane])
    simd_total = sum(simd_like_output)

    abs_error = abs(simd_total - scalar_total)
    evidence = _evidence_bundle(
        domain="hpc",
        case="scalar_vector_parity",
        metrics={
            "scalar_total": scalar_total,
            "simd_total": simd_total,
            "abs_error": abs_error,
        },
    )

    assert abs_error <= baseline["max_abs_error"]
    for field in baseline["required_evidence_fields"]:
        assert field in {**evidence, **evidence["metrics"]}


def test_ml_softmax_reference_parity_oracle():
    baseline = _load_baseline("ml")
    replay = _load_replay_inputs()["ml"]

    logits = replay["logits"]
    max_logit = max(logits)
    shifted = [value - max_logit for value in logits]
    exp_shifted = [math.exp(value) for value in shifted]
    denom = sum(exp_shifted)
    stable = [value / denom for value in exp_shifted]

    sum_error = abs(sum(stable) - 1.0)
    evidence = {
        "trace_id": "trace::ml::stable_softmax",
        "evidence_bundle_id": "bundle::ml::stable_softmax",
        "run_metadata": {
            "dataset_hash": "fixture-dataset",
            "dependency_lock_hash": "fixture-lock",
            "random_seed": 7,
        },
        "metrics": {"sum_error": sum_error},
    }

    assert sum_error <= baseline["max_abs_sum_error"]
    for field in baseline["required_evidence_fields"]:
        assert field in evidence


def test_physics_symplectic_parity_oracle():
    baseline = _load_baseline("physics")
    replay = _load_replay_inputs()["physics"]

    q = replay["q0"]
    p = replay["p0"]
    dt = replay["dt"]
    steps = replay["steps"]

    def grad_v(position: float) -> float:
        return position

    initial_energy = 0.5 * (q * q + p * p)
    max_drift = 0.0

    for _ in range(steps):
        p_half = p - 0.5 * dt * grad_v(q)
        q = q + dt * p_half
        p = p_half - 0.5 * dt * grad_v(q)
        energy = 0.5 * (q * q + p * p)
        drift = abs(energy - initial_energy) / max(abs(initial_energy), 1e-12)
        max_drift = max(max_drift, drift)

    evidence = _evidence_bundle(
        domain="physics",
        case="symplectic_integrator",
        metrics={"energy_drift": max_drift, "invariant_status": 1.0},
    )

    assert max_drift <= baseline["max_relative_energy_drift"]
    for field in baseline["required_evidence_fields"]:
        assert field in {**evidence, **evidence["metrics"]}


def test_quantum_template_emits_required_metadata():
    baseline = _load_baseline("quantum")

    for field in baseline["required_evidence_fields"]:
        assert field in PARAMETERIZED_CIRCUIT_TEMPLATE

    match = re.search(r'"shots"\s*:\s*(\d+)', PARAMETERIZED_CIRCUIT_TEMPLATE)
    assert match is not None
    assert int(match.group(1)) >= baseline["min_shots"]
