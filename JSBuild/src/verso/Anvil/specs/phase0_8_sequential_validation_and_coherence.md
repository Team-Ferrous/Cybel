# Phase 0-8 Sequential Validation and Coherence Spec

## Scope

Execute roadmap phases 0 through 8 in sequence for validation closure, capture measured throughput deltas, and enforce output coherence safety for granitehybrid runtime defaults.

## Repairs

- `PERF-PH0-P0-001`: fused-attention integration evidence closure.
- `PERF-PH1-P0-002`: block unsafe full-graph mode by default on granitehybrid unless explicitly overridden.
- `PERF-PH2-P0/P1/P2`: mmap closure and verification evidence.
- `PERF-PH3..PH8-P1-001`: phase-level test closure plus benchmark evidence and coherence gate status.

## Design Decisions

1. Keep high-performance graph path available behind explicit unsafe override (`ANVIL_NATIVE_GRAPH_UNSAFE_HYBRID=1`).
2. Default runtime prioritizes coherence for granitehybrid by disabling unsafe full-graph mode.
3. Report phase closure sequentially (`0 -> 8`) even when implementation dependencies differ.
4. Mark phases `PARTIAL` when coherence gate is unmet, even if functional tests and throughput deltas are positive.

## Verification Artifacts

- Functional suite:
  - `pytest tests/test_fused_attention_integration.py tests/test_perf_roadmap_phases.py tests/test_native_qsg_engine.py tests/test_mmap_weights.py tests/test_weight_store.py -v`
- Throughput matrix:
  - `benchmarks/native_qsg_benchmark.py` with phase-configured env profiles.
- Governance:
  - `saguaro verify . --engines native,ruff,semantic --format json` (currently blocked by missing TensorFlow-backed embeddings).

## Exit Criteria

- Roadmap has addendums for phases 0-8.
- Repair map and gate dashboard include all phases.
- Benchmarks include speed + coherence evidence.
- Runtime includes coherence safety for granitehybrid graph mode.

## 2026-03-05 Closure Addendum

### Additional Repairs

- Graph interleave coherence fix: deinterleave interleaved quantized rows before passing pointers into C++ graph registration (`core/native/model_graph_wrapper.py`).
- Quantized MoE fused-kernel safety: fallback to mathematically safe composition when interleaved hidden-channel rows are present (`core/native/quantized_matmul_wrapper.py`).
- Expert projection policy: keep expert quant matrices non-interleaved to preserve fused expert kernel channel semantics (`core/native/weight_store.py`).
- Runtime defaults tuned for coherent throughput:
  - graph mode default ON with env opt-out (`ANVIL_NATIVE_GRAPH_MODE`),
  - granitehybrid self-spec default ON in graph+hybrid mode with env override (`ANVIL_ENABLE_SSM_SELF_SPEC`),
  - granitehybrid graph-mode interleave default set to 4 with env override (`ANVIL_QUANT_ROW_INTERLEAVE`).

### Validation Evidence

- Tests:
  - `python3.12 -m pytest tests/test_model_graph_wrapper.py tests/test_model_graph.py tests/test_native_qsg_engine.py tests/test_perf_roadmap_phases.py tests/test_graph_coherence.py -q` → `46 passed`
  - `python3.12 -m pytest tests/test_quantized_matmul.py tests/test_simd_smoke.py tests/test_weight_store.py -q` → `18 passed`
- Coherent benchmark matrix:
  - Granite default (env unset): `7.1549 tok/s` vs forced Python baseline (`GRAPH=0`, `HYBRID=0`, `INTERLEAVE=1`, `SELF_SPEC=0`) `5.3006 tok/s` (**+35.0%**).
  - Qwen default (env unset): `1.0278 tok/s` vs forced Python baseline `0.8431 tok/s` (**+21.9%**).
  - Granite coherent feature deltas:
    - interleave: `7.1510 -> 7.2134 tok/s` (**+0.9%**),
    - self-spec on top of interleave: `7.2134 -> 7.4297 tok/s` (**+3.0%**).

### Compliance Notes

- C++ kernels and threading controls remain SIMD/OpenMP compliant; no fallback masking/clamping was introduced.
- `saguaro verify . --engines native,ruff,semantic --format json` still fails on repository-wide pre-existing violations (`21320`), outside this phase closure scope.
