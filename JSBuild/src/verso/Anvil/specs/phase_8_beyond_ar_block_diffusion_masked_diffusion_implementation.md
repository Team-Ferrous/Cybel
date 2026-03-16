# Phase 8 Implementation Spec

- Phase id: `8`
- Phase name: `Beyond AR - Block Diffusion and Masked Diffusion`
- Roadmap section: `## 23. Phase 8: Beyond AR - Block Diffusion and Masked Diffusion`
- Date: `2026-03-10`

## Scope

This execution is bounded to the existing block-diffusion prototype path, its benchmark/report truthfulness, and the evidence needed to keep Phase 8 explicitly research-bounded.

## Parallel Ownership

### Primary owned files

- `benchmarks/native_qsg_benchmark.py`
- `tests/test_native_qsg_benchmark.py`
- `specs/phase_8_beyond_ar_block_diffusion_masked_diffusion_implementation.md`
- `runs/roadmap/phase_8/`
- `Latent_Space_QSG_Roadmap.md`

### Shared files

- `core/native/parallel_generation.py`
- `core/native/native_qsg_engine.py`
- `core/native/runtime_telemetry.py`
- `tests/test_native_qsg_engine.py`

### Read-only files

- `core/native/qsg_parallel_kernels_wrapper.py`
- `core/native/qsg_parallel_kernels.cpp`
- `Parallel-First Native QSG Roadmap.md`
- `audit/runs/20260310T172510Z_smoke_0bea3bede3a8874a_36d5bc6/`
- `audit/runs/20260310T173528Z_smoke_0bea3bede3a8874a_36d5bc6/`

### Artifact namespace

- `specs/phase_8_beyond_ar_block_diffusion_masked_diffusion_implementation.md`
- `runs/roadmap/phase_8/`
- `.anvil/phase_8/`
- `benchmarks/phase_8/`

## Subagents

- `explorer`
  - Ownership: inspect Phase 8 runtime, telemetry, and kernel surfaces
  - Output: confirmed existing block-diffusion mode, telemetry plumbing, and masked-generation absence
- `verifier`
  - Ownership: inspect current tests and recommend targeted verification commands
  - Output: confirmed planner, generation, and benchmark report test coverage plus remaining evidentiary gaps
- `benchmarks_operator`
  - Ownership: inspect checked-in artifacts for Phase 8 metric evidence
  - Output: confirmed no checked-in measured `block_diffusion` candidate artifact and no `runs/roadmap/phase_8/` baseline before this session

## Phase Classification

- `IN_SCOPE_IMPLEMENTABLE`
  - keep the non-AR benchmark decision framework truthful by marking Phase 8 recommendations as research-only
  - expose readiness blockers directly in the benchmark markdown/JSON surface
  - add regression tests that prevent future production-readiness overclaims
  - collect verification artifacts for the current repo state
- `IN_SCOPE_BUT_BLOCKED_BY_PREREQUISITE`
  - fully native blockwise verification/recovery ownership
  - masked-generation mode, request state, telemetry, and kernel ABI
  - real benchmark evidence with `block_diffusion` candidate runs and AR/self-spec comparison rows
- `OUT_OF_SCOPE_FUTURE_PHASE`
  - diffusion-native checkpoints and training work
  - production scheduler substrate for non-AR serving
  - broader Phase 12 quality governance
- `OUT_OF_SCOPE_UNRELATED`
  - unrelated audit harness drift
  - unrelated kernel tuning and quantization work

## Prerequisite Assumptions

- The repo already contains a Phase 8 prototype:
  - `GenerationMode.BLOCK_DIFFUSION`
  - planner routing into block diffusion
  - `_generate_block_diffusion`
  - blockwise telemetry fields
  - benchmark `non_ar_decision` reporting
- Production closure is still blocked by earlier substrate phases, consistent with roadmap section `23.7`.
- Workset creation was attempted but rejected by Saguaro with `EXCEEDS HARD LIMIT (170496 > 32000)`, so this session proceeded without a claimed workset.

## Implementation Steps

1. Audit the existing Phase 8 prototype surfaces and confirm what is already implemented.
2. Correct the benchmark/report layer so a passing synthetic `block_diffusion` row cannot be read as a production-ready recommendation.
3. Add explicit `research_only` scope and readiness blockers to the non-AR decision output.
4. Add regression tests for the new readiness fields and markdown columns.
5. Capture verification artifacts under `runs/roadmap/phase_8/`.
6. Update the roadmap with a dated Phase 8 addendum reflecting the bounded outcome.

## Implemented This Session

- `benchmarks/native_qsg_benchmark.py`
  - Added explicit Phase 8 `research_only` scope and static readiness blockers to `non_ar_decision` rows.
  - Propagated `native_hot_path_owned` plus Python/NumPy hot-path maxima into the decision output.
  - Extended the markdown report table so Phase 8 readiness is visible in human-readable artifacts.
- `tests/test_native_qsg_benchmark.py`
  - Added assertions that the non-AR decision surface remains research-only even when synthetic gates pass.
  - Added coverage that a Python-owned block-diffusion hot path is surfaced as a readiness blocker.
- `runs/roadmap/phase_8/`
  - Added `saguaro_verify.json`
  - Added `targeted_pytest.txt`
  - Added `existing_audit_phase8_snapshot.txt`

## Verification

1. `source venv/bin/activate && pytest -q tests/test_native_qsg_benchmark.py::test_build_report_groups_metrics_for_analysis tests/test_native_qsg_benchmark.py::test_markdown_report_includes_summary_and_phase_columns tests/test_native_qsg_benchmark.py::test_build_report_emits_non_ar_decision_framework tests/test_native_qsg_engine.py::test_parallel_planner_selects_block_diffusion_candidate tests/test_native_qsg_engine.py::test_generate_block_diffusion_verifies_tokens_with_native_score`
2. `source venv/bin/activate && pytest -q tests/test_native_qsg_engine.py -k "block_diffusion or parallel_decode"`
3. `source venv/bin/activate && pytest -q tests/test_native_qsg_benchmark.py -k "non_ar or full_qsg or sanctioned_path"`
4. `source venv/bin/activate && saguaro verify . --engines native,ruff,semantic --format json`
5. Existing artifact snapshot:
   - `runs/roadmap/phase_8/existing_audit_phase8_snapshot.txt`

## Exit Criteria

- Benchmark/report surfaces must describe Phase 8 as research-only rather than silently implying production readiness.
- Tests must cover the readiness fields and preserve the existing block-diffusion prototype plumbing.
- `saguaro verify` must run and its result must be recorded.
- Final status cannot be `PASS` unless a checked-in benchmark artifact proves real measured `block_diffusion` candidate wins on representative workloads and masked-generation readiness is no longer absent.
