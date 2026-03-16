from __future__ import annotations

from saguaro.math import MathEngine
from saguaro.omnigraph.store import OmniGraphStore


def test_math_engine_parses_and_maps_equations(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    spec = docs / "spec.md"
    spec.write_text(
        "# Spec\n\n"
        "We define $$E = m c^2$$ for validation.\n",
        encoding="utf-8",
    )

    graph = OmniGraphStore(str(tmp_path))
    graph.build(
        traceability_payload={
            "generation_id": "math-test",
            "requirements": [],
            "records": [
                {
                    "id": "REL-1",
                    "requirement_id": "REQ-1",
                    "artifact_type": "symbol",
                    "artifact_id": "symbol::energy",
                    "relation_type": "supports",
                    "evidence_types": ["lexical"],
                    "confidence": 0.8,
                    "verification_state": "verified",
                    "notes": ["energy mass relation"],
                }
            ],
        }
    )

    engine = MathEngine(str(tmp_path))
    parsed = engine.parse("docs/spec.md")

    assert parsed["count"] == 1
    equation_id = parsed["equations"][0]["id"]
    mapped = engine.map(equation_id)

    assert mapped["status"] == "ok"
    assert mapped["equation"]["expression"] == "E = m c^2"


def test_math_engine_extracts_code_equations_and_complexity(tmp_path):
    kernels = tmp_path / "kernels"
    kernels.mkdir()
    kernel = kernels / "qssm_kernel.h"
    kernel.write_text(
        """
/*
 * S_t = sigma_vqc(x_t) * S_prev + (1 - sigma_vqc(x_t)) * Update(x_t)
 */
inline float StepKernel(float gate, float* state, float inp_val, int idx) {
    state[idx] =
        gate * state[idx] +
        (1.0f - gate) * inp_val;
    sum += input[idx] / scale;
    return x.dot(bracket) + omega.dot(v);
}
""".strip(),
        encoding="utf-8",
    )

    engine = MathEngine(str(tmp_path))
    parsed = engine.parse("kernels/qssm_kernel.h")

    assert parsed["count"] >= 4
    assert parsed["summary"]["by_source_kind"]["code_comment"] >= 1
    assert parsed["summary"]["by_source_kind"]["code_expression"] >= 3
    expressions = {item["expression"]: item for item in parsed["equations"]}
    assert (
        "state[idx] = gate * state[idx] + (1.0f - gate) * inp_val"
        in expressions
    )
    assert "sum += input[idx] / scale" in expressions
    assert "return x.dot(bracket) + omega.dot(v)" in expressions
    assert (
        expressions["state[idx] = gate * state[idx] + (1.0f - gate) * inp_val"][
            "complexity"
        ]["band"]
        in {"medium", "high"}
    )
