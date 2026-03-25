# Phase 4: KV Cache Redesign for CPU

- `phase_id`: `4`
- `phase_name`: `KV Cache Redesign for CPU`
- `roadmap_section`: `Latent_Space_QSG_Roadmap.md -> 19. Phase 4: KV Cache Redesign for CPU`
- `status`: `implementation_in_progress`

## Scope

- Integrate existing native KV cache residency, sharing, snapshot, and copy-on-write metrics into the live QSG runtime status path.
- Preserve merge-safe boundaries with Phase 2 by avoiding per-sequence scheduler identity changes.
- Capture explicit blocker classification for parked per-sequence KV ownership and resume-safe state restoration.

## Ownership

- `PRIMARY_OWNED_FILES`
- `core/native/native_qsg_engine.py`
- `core/native/parallel_generation.py`
- `tests/test_native_qsg_engine.py`
- `tests/test_native_parallel_generation_engine.py`
- `specs/phase_4_kv_cache_redesign_for_cpu_implementation.md`

- `SHARED_FILES`
- `Latent_Space_QSG_Roadmap.md`

- `READ_ONLY_FILES`
- `core/native/native_kv_cache_wrapper.py`
- `core/native/native_kv_cache.cpp`
- `core/qsg/continuous_engine.py`
- `core/qsg/state_pager.py`
- `tests/test_perf_roadmap_phases.py`

## Scope Classification

- `IN_SCOPE_IMPLEMENTABLE`
- Native KV metrics propagation for fragmentation, page residency, snapshot count, prefix sharing, and copy-on-write evidence.
- Runtime capability/status exposure for the native KV metrics already emitted by the native cache wrapper.

- `IN_SCOPE_BUT_BLOCKED_BY_PREREQUISITE`
- Per-request parked KV ownership across concurrent scheduler identities.
- Resume-safe native KV restore per suspended request.
- Cold-page resume path that is keyed by stable Phase 2 request/sequence identity.

- `OUT_OF_SCOPE_FUTURE_PHASE`
- KV quantization implementation changes beyond existing runtime capability reporting.
- Parking-aware compression/page-in policy.

- `OUT_OF_SCOPE_UNRELATED`
- Phase 3 sampling changes.
- Scheduler admission control redesign.

## Prerequisite Assumptions

- Existing native KV wrapper metrics and prefix snapshot APIs are correct enough to reuse.
- Phase 2 scheduler identity remains the hard blocker for production closure of parked per-sequence KV ownership.

## Implementation Steps

1. Merge native KV wrapper metrics into parallel generation metrics snapshots when a native KV cache handle exists.
2. Persist those metrics into `NativeQSGEngine` scheduler/runtime snapshots.
3. Expose the additional Phase 4 evidence in runtime status surfaces.
4. Add targeted unit coverage for the new metrics path.
5. Update the roadmap Phase 4 section with a bounded implementation addendum if materially advanced.

## Verification

1. Targeted `pytest` for native parallel generation and native QSG engine metrics/status coverage.
2. Mandatory governance verification:
   - `source venv/bin/activate && saguaro verify . --engines native,ruff,semantic --format json`

## Artifact Namespace

- `specs/phase_4_kv_cache_redesign_for_cpu_implementation.md`
- `runs/roadmap/phase_4/`
- `.anvil/phase_4/`
- `benchmarks/phase_4/`

## Exit Criteria

- Native KV page/sharing/COW metrics are visible in runtime-facing status paths.
- Phase 4 blockers are explicit where scheduler identity is still missing.
- Tests cover the new metrics propagation path.
- Verification results are recorded.
