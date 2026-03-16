# Phase 4 Implementation Spec

- Phase id: `4`
- Phase name: `KV Cache Redesign for CPU`
- Roadmap section: `## 19. Phase 4: KV Cache Redesign for CPU`
- Date: `2026-03-10`
- Mode: `Implementation (single-phase bounded)`

## Scope

This execution is bounded to the standalone native KV cache path used by `NativeKVCacheWrapper`, plus Phase 4 evidence updates. The graph runtime remains read-only for this session because it is a shared Phase 2 / Phase 4 dependency surface.

## Ownership

- Primary owned files
  - `core/native/native_kv_cache.cpp`
  - `core/native/native_kv_cache_wrapper.py`
  - `tests/test_perf_roadmap_phases.py`
  - `specs/phase_4_kv_cache_redesign_cpu_implementation.md`
- Allowed shared files
  - `Latent_Space_QSG_Roadmap.md`
- Read-only files
  - `core/native/model_graph.cpp`
  - `core/native/model_graph_wrapper.py`
  - `core/native/paged_kv_cache.py`
  - `tests/test_native_qsg_benchmark.py`

## Artifact namespace

- `specs/phase_4_kv_cache_redesign_cpu_implementation.md`
- `runs/roadmap/phase_4/`
- `.anvil/phase_4/`
- `benchmarks/phase_4/`

## Current reality and dependency review

### Satisfied prerequisites

- The standalone native KV cache already exposes paged allocation, prefix snapshots, copy-on-write, and metrics in `core/native/native_kv_cache.cpp`.
- `NativeKVCacheWrapper` already binds snapshot and metrics ABI entry points.
- Focused Phase 4 tests already exist in `tests/test_perf_roadmap_phases.py`.

### Blocking prerequisite outside allowed bridge scope

- The full graph runtime in `core/native/model_graph.cpp` owns a separate paged KV store and checkpoint path that still deep-copies storage instead of using shared-page branching.
- Stable per-request or per-sequence identity from Phase 2 is still missing from the standalone KV ABI, so Phase 4 cannot honestly close continuous batching, parked-sequence ownership, or cross-request prefix reuse in this session.

## Phase classification

- `IN_SCOPE_IMPLEMENTABLE`
  - verify and harden the standalone native paged-KV path
  - tighten wrapper and test evidence for prefix snapshot/restore, copy-on-write isolation, and fragmentation telemetry
  - capture artifact-backed metrics for the standalone native KV path
  - update roadmap evidence to distinguish shipped Phase 4 work from remaining blockers
- `IN_SCOPE_BUT_BLOCKED_BY_PREREQUISITE`
  - full `model_graph.cpp` paged-KV prefix sharing and copy-on-write
  - runtime-level cross-request prefix cache ownership
  - parked-sequence keep/evict behavior tied to scheduler state
  - cold-page resume path and latent checkpoint linkage
- `OUT_OF_SCOPE_FUTURE_PHASE`
  - speculative tree decode on shared KV branches
  - production parked-state compression and KV quantization redesign
  - architecture-specific page-size autotuning
- `OUT_OF_SCOPE_UNRELATED`
  - scheduler and parallel kernel changes outside the Phase 4 KV surface

## Implementation steps

1. Align the Phase 4 spec with the actual roadmap section, ownership boundaries, and blockers.
2. Strengthen targeted Phase 4 tests around snapshot boundaries, invalid snapshot handling, and metrics expectations.
3. Run focused verification and capture standalone KV metrics artifacts under the Phase 4 namespace.
4. Update the Phase 4 roadmap section with a dated addendum that separates implemented standalone KV work from graph-runtime blockers.

## Verification

1. Run focused Phase 4 pytest coverage for `NativeKVCache` wrapper and flash-attention surfaces.
2. Attempt native rebuild and record unrelated repo-level failures separately from Phase 4 results.
3. Capture artifact-backed standalone KV metrics under `benchmarks/phase_4/`.
4. Run `source venv/bin/activate && saguaro verify . --engines native,ruff,semantic --format json`.

## Exit criteria

- The standalone native KV cache exposes prefix snapshot/restore semantics without deep-copying all pages.
- Mutating a shared prefix page triggers copy-on-write rather than aliasing snapshot state.
- Native metrics report committed pages, shared pages, snapshot count, and fragmentation ratio, with artifact-backed evidence.
- Wrapper reads and tests prove the active KV state is observable across page boundaries and after snapshot/restore.
- Final status cannot be `PASS` unless the graph/runtime-owned KV path also satisfies the Phase 4 roadmap clauses.
