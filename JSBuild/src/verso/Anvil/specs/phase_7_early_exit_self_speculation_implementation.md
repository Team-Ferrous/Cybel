# Phase 7 Implementation Spec

- Phase id: `7`
- Phase name: `Early-Exit and Self-Speculation`
- Roadmap section: `Latent_Space_QSG_Roadmap.md :: ## 22. Phase 7: Early-Exit and Self-Speculation`
- Date: `2026-03-10`

## Scope

This execution is bounded to Phase 7 native early-exit and self-spec runtime correctness, exit-depth policy wiring, telemetry truthfulness, targeted tests, and roadmap evidence for the active native QSG path.

## Ownership

### Primary owned files

- `core/native/native_qsg_engine.py`
- `tests/test_native_qsg_engine.py`
- `specs/phase_7_early_exit_self_speculation_implementation.md`

### Allowed shared files

- `benchmarks/native_qsg_benchmark.py`
- `tests/test_native_qsg_benchmark.py`
- `Latent_Space_QSG_Roadmap.md`

### Read-only dependency files

- `core/native/model_graph.cpp`
- `core/native/model_graph_wrapper.py`
- `core/native/runtime_telemetry.py`

### Artifact namespace

- `specs/phase_7_early_exit_self_speculation_implementation.md`
- `runs/roadmap/phase_7/`
- `.anvil/phase_7/`
- `benchmarks/phase_7/`

## Current reality and dependency review

### Satisfied prerequisites

- `core/native/model_graph.cpp` already exposes execution checkpoint, hidden-state exit capture, verifier-tail continuation, and head projection primitives.
- `core/native/model_graph_wrapper.py` already surfaces `create_execution_checkpoint`, `restore_execution_checkpoint`, `forward_token_id_to_exit`, `continue_from_hidden`, and `forward_head`.
- `core/native/native_qsg_engine.py` already contains a native early-exit and self-spec control path plus telemetry fields for exit policy and drafted token counts.

### Blocking prerequisites outside allowed bridge scope

- The broader speculative request loop still has Python-owned fallback behavior in `core/native/cpu_speculative_decode.py`.
- Phase-level `PASS` remains blocked without representative benchmark evidence showing acceptable quality and a credible performance win after governance checks.
- Fully trained checkpoint support for learned early-exit or LayerSkip-style policies belongs to later roadmap work.

## Phase classification

- `IN_SCOPE_IMPLEMENTABLE`
  - repair broken Phase 7 engine control flow on the active native path
  - validate native early-exit policy selection and verifier-tail reuse
  - validate telemetry and benchmark-facing self-spec fields
  - update spec and roadmap evidence to match actual repository state
- `IN_SCOPE_BUT_BLOCKED_BY_PREREQUISITE`
  - native-only end-to-end speculative ownership across all fallback paths
  - benchmark-backed quality and speed evidence sufficient for `PASS`
- `OUT_OF_SCOPE_FUTURE_PHASE`
  - learned exit-head policies
  - checkpoint-training changes for LayerSkip or related supervision
  - scheduler-wide speculative overlap work beyond the Phase 7 bridge
- `OUT_OF_SCOPE_UNRELATED`
  - unrelated audit harness or benchmark-suite drift

## Implementation steps

1. Fix the malformed native self-spec branch in `NativeQSGEngine._generate_ssd_bridge`.
2. Keep the engine path Phase 7-native: exit-layer selection, verifier-tail continuation, and truthful self-spec telemetry.
3. Add or adjust targeted tests only where Phase 7 behavior is not covered or is currently mis-specified.
4. Capture the strongest available local evidence for Phase 7 and update the roadmap with a dated implementation addendum.

## Verification steps

1. Run `python3 -m py_compile core/native/native_qsg_engine.py`.
2. Run targeted Phase 7 tests in `tests/test_native_qsg_engine.py`.
3. Run targeted benchmark report tests in `tests/test_native_qsg_benchmark.py`.
4. Run the strongest relevant benchmark or telemetry command that can produce Phase 7 evidence locally.
5. Run `source venv/bin/activate && saguaro verify . --engines native,ruff,semantic --format json`.

## Exit criteria

- The native self-spec path imports, executes, and remains the preferred Phase 7 route when native continuation support is available.
- Exit-depth policy selection is bounded and reported truthfully in telemetry.
- Tests cover the native early-exit and verifier-tail path.
- Phase status remains below `PASS` unless benchmark evidence exists and the remaining Python-owned speculative fallback does not invalidate the phase goals.
