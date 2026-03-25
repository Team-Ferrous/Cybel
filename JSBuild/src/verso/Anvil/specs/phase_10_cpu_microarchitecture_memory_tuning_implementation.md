# Phase 10 Implementation Spec

- Phase id: `10`
- Phase name: `CPU Microarchitecture and Memory Subsystem Tuning`
- Roadmap section: `Latent_Space_QSG_Roadmap.md -> ## 25. Phase 10: CPU Microarchitecture and Memory Subsystem Tuning`
- Session mode: `single-phase bounded`
- Artifact namespace:
  - `specs/phase_10_cpu_microarchitecture_memory_tuning_implementation.md`
  - `runs/roadmap/phase_10/`
  - `.anvil/phase_10/`
  - `benchmarks/phase_10/`
  - `audit/runs/<suite_run_id>/`

## Scope

`IN_SCOPE_IMPLEMENTABLE`
- Make strict suite PMU collection real in the operator path, not just declared via `collect_hw_counters`.
- Preserve explicit OpenMP placement and host NUMA policy evidence in Phase 10 artifacts.
- Add regression coverage for PMU success, PMU safe degradation, and strict gating when PMU evidence is missing.
- Update roadmap evidence for the Phase 10 operator-path fix and current host measurements.

`IN_SCOPE_BUT_BLOCKED_BY_PREREQUISITE`
- Hardware-specific AVX2 kernel retuning, OpenMP loop restructuring, and measurable IPC gains across the native runtime.
- Full NUMA page-placement evidence for every strict suite attempt, because the current suite contract does not yet require page-map capture.

`OUT_OF_SCOPE_FUTURE_PHASE`
- New native kernel rewrites in `core/native/simd_ops.cpp`.
- Native scheduler redesign, admission control, or broader runtime topology ownership changes.
- Phase 11 model-side changes and Phase 12 quality-governance expansion.

`OUT_OF_SCOPE_UNRELATED`
- COCONUT latent-mode execution changes.
- Speculative decode and non-AR roadmap work beyond shared benchmark plumbing already present in the repo.

## Parallel Ownership

- `PRIMARY_OWNED_FILES`
  - `audit/runner/attempt_executor.py`
  - `tests/audit/test_attempt_executor.py`
  - `tests/test_native_qsg_benchmark.py`
  - `specs/phase_10_cpu_microarchitecture_memory_tuning_implementation.md`
- `SHARED_FILES`
  - `Latent_Space_QSG_Roadmap.md`
- `READ_ONLY_FILES`
  - `benchmarks/native_qsg_benchmark.py`
  - `audit/runner/benchmark_suite.py`
  - `core/native/native_qsg_engine.py`
  - `core/native/CMakeLists.txt`
  - `core/native/simd_ops.cpp`
  - `core/native/numa_allocator.cpp`
  - `core/native/numa_topology.cpp`
  - `audit/profiles/native_qsg_smoke.yaml`
  - `audit/profiles/native_qsg_silver.yaml`
  - `tests/test_perf_roadmap_phases.py`

## Dependencies and Assumptions

- The suite continues to launch `benchmarks/native_qsg_benchmark.py --isolated-child` through `audit/runner/attempt_executor.py`.
- `perf` is available on the target host for strict profiles; preflight already validates this requirement.
- Existing OpenMP placement defaults and NUMA host-policy snapshots remain authoritative for current Phase 10 evidence.
- Saguaro workset creation is currently unreliable here: `saguaro workset create` failed with `EXCEEDS HARD LIMIT (135484 > 32000)`. This is documented as tooling drift, not a phase blocker.

## Implementation Steps

1. Fix the operator-path PMU gap in `audit/runner/attempt_executor.py` so `collect_hw_counters=required` actually wraps the isolated benchmark child with `perf stat`.
2. Merge parsed PMU fields and the generated `perf_stat_artifact` back into the flattened attempt record before gate evaluation.
3. Fail strict attempts when PMU evidence is required but not observed.
4. Add regression tests for:
   - successful strict PMU wrapping
   - missing `perf stat` artifact failure
   - missing/unavailable PMU artifact parsing in the benchmark parser
5. Capture current host evidence from a bounded Phase 10 suite run and update the roadmap addendum.

## Verification Steps

1. `pytest -q tests/test_native_qsg_benchmark.py tests/audit/test_attempt_executor.py tests/audit/test_benchmark_suite.py`
2. `source venv/bin/activate && saguaro verify . --engines native,ruff,semantic --format json`
3. Bounded Phase 10 host evidence:
   - use the suite operator path or equivalent strict attempt path
   - confirm host preflight records `cpu_governor=performance`, `thp=madvise|never`, visible thread budget, and `perf` availability
   - confirm per-attempt PMU evidence is now populated or explicitly fails

## Exit Criteria

- Strict suite attempts no longer silently skip PMU collection when `collect_hw_counters=required`.
- Attempt records surface `perf_stat_artifact` plus parsed PMU fields, or fail with an explicit PMU error.
- Regression tests cover the PMU success path and the safe-degradation path.
- The roadmap Phase 10 section contains a dated implementation addendum with artifact references and remaining blockers.
- Phase status remains `PARTIAL` until a fresh strict suite run proves per-attempt PMU evidence and broader NUMA/page-placement benchmarking on the current host.
