# Phase 5 Implementation Spec

- Phase id: `5`
- Phase name: `Speculative Bridge Roadmap`
- Roadmap section: `## 20. Phase 5: Speculative Bridge Roadmap`
- Date: `2026-03-10`

## Ownership

### PRIMARY_OWNED_FILES

- `core/native/parallel_generation.py`
- `core/native/native_qsg_engine.py`
- `core/native/runtime_telemetry.py`
- `tests/test_native_qsg_engine.py`
- `tests/test_native_parallel_generation_engine.py`

### SHARED_FILES

- `core/native/qsg_parallel_kernels.h`
- `benchmarks/native_qsg_benchmark.py`
- `tests/test_native_qsg_benchmark.py`
- `Latent_Space_QSG_Roadmap.md`
- `specs/phase_5_speculative_bridge_implementation.md`

### READ_ONLY_FILES

- `core/native/qsg_parallel_kernels.cpp`
- `core/native/qsg_parallel_kernels_wrapper.py`
- `core/native/cpu_speculative_decode.py`

### ARTIFACT_NAMESPACE

- `specs/phase_5_speculative_bridge_implementation.md`
- `runs/roadmap/phase_5/`
- `.anvil/phase_5/`
- `benchmarks/phase_5/`

## Scope

This execution is bounded to Phase 5 speculative bridge observability and truthful mode attribution on the active native QSG runtime and benchmark surfaces.

### Exact in-scope files

- `core/native/parallel_generation.py`
- `core/native/native_qsg_engine.py`
- `core/native/runtime_telemetry.py`
- `tests/test_native_parallel_generation_engine.py`
- `benchmarks/native_qsg_benchmark.py`
- `tests/test_native_qsg_engine.py`
- `tests/test_native_qsg_benchmark.py`

### Out of scope

- Replacing the Python-owned self-spec executor with a native runtime path
- Native batched verifier ownership for prompt lookup acceptance
- Full SSD branch state management and speculative tree reuse
- Multi-head drafting implementation beyond already-present planner hooks

## Current reality and dependency review

### Satisfied prerequisites

- The planner already contains prompt lookup, replacement, head-based, and SSD bridge selection logic.
- Prompt lookup drafting already calls a native kernel via `qsg_prompt_lookup_draft`.
- The engine and benchmark harness already expose speculative counters and generation-mode metadata.

### Blocking prerequisite outside allowed bridge scope

- `core/native/cpu_speculative_decode.py` still owns self-spec draft/verify loops in Python and NumPy.
- Prompt lookup verification still executes through Python `_verify_draft_tokens` rather than a native verifier batch path.
- Because of those two facts, the roadmap clause `prompt lookup and self-spec are fully native` cannot be satisfied in this session without overlapping unfinished Phase 1/2/3 ownership work.
- `saguaro workset create` exceeded the repository hard limit during this phase session, so workset creation is documented as a tooling limitation rather than a phase blocker.

## Phase classification

- `IN_SCOPE_IMPLEMENTABLE`
  - expose prompt lookup as an explicit speculative mode instead of folding it into `parallel_hybrid`
  - tag runtime telemetry with prompt category and temperature band
  - surface benchmark summaries for speculative acceptance grouped by prompt category and temperature band
  - add regression tests for the new telemetry and report surfaces
- `IN_SCOPE_BUT_BLOCKED_BY_PREREQUISITE`
  - fully native prompt lookup verification
  - fully native self-spec execution
  - benchmark evidence showing one multi-token speculative path outperforms AR on representative workloads
- `OUT_OF_SCOPE_FUTURE_PHASE`
  - SSD speculative-overlap runtime ownership
  - future trained self-spec / LayerSkip checkpoints
  - broader quality-governance gating from Phase 12
- `OUT_OF_SCOPE_UNRELATED`
  - unrelated audit harness or benchmark-suite drift

## Implementation steps

1. Introduce truthful prompt-lookup generation mode and benchmark label surfaces.
2. Add prompt-category and temperature-band tagging to native runtime telemetry.
3. Surface Phase 5 runtime fields on the native scheduler metrics snapshot used by continuous batching telemetry.
4. Extend benchmark result/report output with prompt-category and temperature-band acceptance grouping.
5. Add targeted tests covering mode attribution, acceptance-band reporting, and scheduler metrics propagation.
6. Update the roadmap Phase 5 section with a dated addendum describing the shipped observability bridge and remaining blockers.

## Verification steps

1. Run targeted runtime/speculative tests in `tests/test_native_qsg_engine.py`.
2. Run targeted native scheduler metrics tests in `tests/test_native_parallel_generation_engine.py`.
3. Run targeted benchmark report tests in `tests/test_native_qsg_benchmark.py`.
4. Run `source venv/bin/activate && saguaro verify . --engines native,ruff,semantic --format json`.

## Exit criteria

- Prompt lookup is reported as its own speculative mode and benchmark label.
- Runtime/benchmark artifacts expose prompt-category and temperature-band fields needed for acceptance slicing.
- Native scheduler metrics snapshot exposes speculative acceptance/runtime fields emitted by the underlying runtime status.
- Benchmark report contains an acceptance summary grouped by prompt category and temperature band.
- Tests cover the new fields and grouping.
- Final status must remain below `PASS` unless prompt lookup verification and self-spec execution are actually native.
