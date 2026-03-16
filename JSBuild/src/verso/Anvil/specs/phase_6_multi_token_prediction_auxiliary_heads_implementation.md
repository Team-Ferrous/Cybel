# Phase 6 Implementation Spec

- Phase id: `6`
- Phase name: `Multi-Token Prediction and Auxiliary Heads`
- Roadmap section: `## 21. Phase 6: Multi-Token Prediction and Auxiliary Heads`
- Date: `2026-03-10`

## Scope

This execution is bounded to Phase 6 native head evidence plumbing for Medusa/Hydra proposal paths so native candidate tokens and confidences survive into runtime telemetry and benchmark artifacts.

### In scope

- `core/native/parallel_generation.py`
- `core/native/native_qsg_engine.py`
- `core/native/runtime_telemetry.py`
- `benchmarks/native_qsg_benchmark.py`
- `tests/test_native_qsg_engine.py`
- `tests/test_native_qsg_benchmark.py`
- `Latent_Space_QSG_Roadmap.md`
- `specs/phase_6_multi_token_prediction_auxiliary_heads_implementation.md`

### Out of scope

- Replacing Python-owned draft verification with a native batched verify path
- Proving real throughput wins over AR baseline on representative workloads
- Training-aware MTP auxiliary heads or checkpoint changes
- Broader scheduler/runtime ownership work from earlier phases

## Parallel ownership

- `PRIMARY_OWNED_FILES`
  - `core/native/parallel_generation.py`
  - `core/native/runtime_telemetry.py`
  - `benchmarks/native_qsg_benchmark.py`
  - `tests/test_native_qsg_engine.py`
  - `tests/test_native_qsg_benchmark.py`
  - `specs/phase_6_multi_token_prediction_auxiliary_heads_implementation.md`
- `SHARED_FILES`
  - `core/native/native_qsg_engine.py`
  - `Latent_Space_QSG_Roadmap.md`
- `READ_ONLY_FILES`
  - `prompts/audit_implementation.md`
  - `Latent_Space_QSG_Roadmap.md` (except for the dated Phase 6 addendum)
- `ARTIFACT_NAMESPACE`
  - `specs/phase_6_multi_token_prediction_auxiliary_heads_implementation.md`
  - `audit/runs/phase_6_smoke_20260310/`
  - `.anvil/phase_6/`
  - `benchmarks/phase_6/`

## Current reality and dependency review

### Satisfied prerequisites

- Native Medusa and Hydra proposal kernels exist in `core/native/qsg_parallel_kernels.cpp`.
- The Python planner and engine already route Medusa/Hydra modes and benchmark labels.
- Benchmark and runtime telemetry already carry speculative acceptance counters.

### Blocking prerequisite outside allowed bridge scope

- Head verification still runs through Python `_verify_draft_tokens` and per-token scoring in `core/native/native_qsg_engine.py`.
- `parallel_generation.py` and `qsg_parallel_kernels_wrapper.py` still rely on Python/NumPy marshalling, so the production hot path is not fully native-owned.
- No repository evidence currently proves head-based throughput improvement over the AR baseline on representative workloads.

## Phase classification

- `IN_SCOPE_IMPLEMENTABLE`
  - preserve native Medusa/Hydra confidence outputs across the Python boundary
  - preserve proposal-count and confidence evidence even when Medusa/Hydra verification rejects the proposal bundle
  - attach proposal-count and confidence summaries to generation evidence
  - propagate those fields into runtime telemetry and benchmark reporting
  - add targeted regression coverage for planner fallback behavior, telemetry, and benchmark summaries
- `IN_SCOPE_BUT_BLOCKED_BY_PREREQUISITE`
  - fully native head verification and acceptance policy integration
  - production-path removal of Python/NumPy marshalling from speculative head execution
  - measured throughput evidence sufficient for Phase 6 `PASS`
- `OUT_OF_SCOPE_FUTURE_PHASE`
  - continued-pretraining MTP losses
  - future-checkpoint MTP / GQA / early-exit co-design
  - Phase 12 governance for full quality-gated promotion
- `OUT_OF_SCOPE_UNRELATED`
  - unrelated audit harness drift
  - unrelated KV, sampling, or microarchitecture work

## Implementation steps

1. Introduce a structured native head draft bundle that carries tokens and confidences.
2. Propagate draft proposal counts and confidence summaries into `GenerationEvidence`, including rejected proposal bundles.
3. Surface the new evidence in `NativeGenerationTelemetry` and benchmark result/report output.
4. Add targeted tests covering Medusa/Hydra evidence propagation, rejected-bundle fallback behavior, and report aggregation.
5. Update the roadmap Phase 6 section with a dated implementation addendum describing shipped artifacts and remaining blockers.

## Verification

1. Run targeted Medusa/Hydra tests in `tests/test_native_qsg_engine.py`.
2. Run targeted benchmark report tests in `tests/test_native_qsg_benchmark.py`.
3. Run a smoke profile artifact command for Phase 6 evidence capture and record the outcome under `audit/runs/phase_6_smoke_20260310/`.
4. Run `source venv/bin/activate && saguaro verify . --engines native,ruff,semantic --format json`.

## Exit criteria

- Native Medusa/Hydra proposal probabilities are preserved in structured evidence instead of being discarded immediately.
- Runtime telemetry and benchmark artifacts expose proposal counts and proposal-confidence summaries, including rejected head proposals.
- Regression tests cover the new proposal evidence surfaces.
- Final status must remain below `PASS` unless native verify ownership and real throughput evidence are both present.
