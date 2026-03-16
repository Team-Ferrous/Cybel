# Phase 2 mmap Zero-Copy Weight Access Closure Spec

## Scope

This spec closes Phase 2 (`mmap Zero-Copy Weight Access`) items in `ROADMAP_PERF_OPTIMIZATION.md` for native QSG runtime wiring, memory-copy elimination, and gate evidence.

## Repair Map

- `PERF-PH2-P0-001` mmap weight path is disabled by default and silently falls back to copied `WeightStore`.
  - Severity: P0
  - Root cause: `NativeQSGEngine` uses `ANVIL_NATIVE_USE_MMAP_WEIGHTS="0"` default and catches mmap init errors by downgrading to non-mmap path.
  - Risk: Phase 2 never activates in default runtime and RSS/load-time benefits are not realized or observable.
  - Fix: Enable mmap store by default with explicit opt-out; remove silent mmap->non-mmap fallback from initialization path.
- `PERF-PH2-P1-001` mmap store still duplicates dense float16 tensors via promotion cache.
  - Severity: P1
  - Root cause: inherited `WeightStore.get_weight` promotes float16 tensors into `_f32_cache` for all stores.
  - Risk: unnecessary memory duplication undermines Phase 2 RSS objectives.
  - Fix: make float16 promotion cache configurable and disable it for `MMapWeightStore`.
- `PERF-PH2-P2-001` missing gate-grade tests and roadmap evidence for Phase 2 wiring.
  - Severity: P2
  - Root cause: current Phase 2 tests validate import/accessor existence only.
  - Fix: add guardrail tests for default mmap selection, explicit opt-out, no silent fallback, and float16 no-copy behavior; update roadmap repair map and gate dashboard with measured deltas.

## Design Decisions

1. mmap path is default-on for native QSG (`ANVIL_NATIVE_USE_MMAP_WEIGHTS` opt-out).
2. If mmap is requested but unavailable, initialization fails with explicit error (no silent downgrade).
3. `WeightStore` keeps existing default float16->float32 promotion for non-mmap paths.
4. `MMapWeightStore` disables float16 promotion cache to preserve zero-copy dense tensor access.
5. Verification evidence includes targeted Phase 2 tests and updated roadmap gate entries.

## Out Of Scope

- Phase 3+ kernel or layout optimizations.
- Phase 6 QHPM / critical feature-family wiring checks.
- C++ kernel modifications (Phase 2 closure is Python runtime/store wiring).

## Gate Target (Phase 2)

- Functional: native engine selects mmap weight store by default and only leaves mmap path on explicit opt-out.
- Safety: no silent fallback from mmap request to copied weight path.
- Memory: mmap store avoids float16 promotion duplication.
- Evidence: roadmap Phase 2 addendum + repair map rows + gate dashboard row with test and metric delta.
