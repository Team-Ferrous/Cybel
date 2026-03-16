# Phase 9 Implementation Spec

- Phase id: `9`
- Phase name: `Quantization and Numeric Redesign`
- Roadmap section: `## 15. Phase 9: Quantization and Numeric Redesign`
- Acceptance references: `## 15.11 Acceptance criteria for Phase 9`, `## 22.5 Final success definition`, `## 24.1`, `## 24.2`

## In-Scope Files

- `audit/runner/benchmark_suite.py`
- `audit/runner/suite_preflight.py`
- `core/native/native_qsg_engine.py`
- `core/native/runtime_telemetry.py`
- `benchmarks/native_qsg_benchmark.py`
- `tests/audit/test_benchmark_suite.py`
- `tests/test_native_qsg_benchmark.py`
- `tests/test_native_qsg_engine.py`
- `Parallel-First Native QSG Roadmap.md`

## Prerequisite Assumptions

- Existing native low-bit weight execution remains the production path.
- Existing q8 KV cache support in native graph code is the current Phase 9 bridge for KV quantization.
- Full concurrency evidence requires local benchmark execution against installed models and current host hardware.

## Scope Classification

- `IN_SCOPE_IMPLEMENTABLE`
  - Enforce numeric calibration error as a calibration quality gate.
  - Expose quantization and KV-cache quantization profile details in runtime and benchmark outputs.
  - Persist richer quantization profile metadata in audit/preflight artifacts.
- `IN_SCOPE_BUT_BLOCKED_BY_PREREQUISITE`
  - Produce live concurrency benchmark evidence for KV quantization on local production models.
- `OUT_OF_SCOPE_FUTURE_PHASE`
  - New activation quantization kernels, AWQ/SmoothQuant conversion flows, or new KV quantization algorithms beyond existing q8 support.
- `OUT_OF_SCOPE_UNRELATED`
  - Scheduler redesign, speculative decode expansion, non-AR runtime work.

## Implementation Steps

1. Tighten calibration quality gating around `expected_calibration_error`.
2. Surface KV-cache quantization mode from the native runtime status path.
3. Propagate quantization profile metadata into benchmark/audit payloads without breaking existing schema consumers.
4. Add targeted tests for failing and passing quality-gate behavior plus runtime/benchmark metadata coverage.
5. Update the Phase 9 roadmap section with an implementation addendum and remaining blockers.

## Verification Steps

1. Run targeted pytest coverage for:
   - `tests/audit/test_benchmark_suite.py`
   - `tests/test_native_qsg_benchmark.py`
   - `tests/test_native_qsg_engine.py`
2. Run mandatory governance verification:
   - `source venv/bin/activate && saguaro verify . --engines native,ruff,semantic --format json`

## Exit Criteria

- Calibration rejects numerically unstable candidates based on explicit calibration error thresholds.
- Runtime and benchmark payloads expose quantization/KV quantization state needed for Phase 9 evidence capture.
- Targeted tests pass.
- Roadmap addendum records what advanced Phase 9 and what remains blocked on live benchmark evidence.
