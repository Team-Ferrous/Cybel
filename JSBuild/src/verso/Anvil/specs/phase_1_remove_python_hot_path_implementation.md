# Phase 1 Remove Python from the Hot Path Implementation Spec

## Phase

- Phase ID: `1`
- Phase Name: `Remove Python from the Hot Path`
- Roadmap Section: `## 7. Phase 1: Remove Python from the Hot Path`

## Scope Classification

### IN_SCOPE_IMPLEMENTABLE

- Correct Phase 1 ownership telemetry so Python token-time loops are reported as active blockers.
- Mark the continuous batching "native" compatibility shim as not Phase 1 complete until a native serve runtime ABI exists.
- Add regression tests that fail if the runtime claims `executed_cpp_only` while Python still owns token-time orchestration.
- Add a roadmap implementation addendum documenting the current blocker set.

### IN_SCOPE_BUT_BLOCKED_BY_PREREQUISITE

- `NativeServeRuntime` C++ ABI with submit/poll/cancel/metrics/shutdown ownership.
- Native output queue / ring buffer for chunk emission.
- Native scheduler loop that owns decode cadence instead of Python `run_forever`.
- Native streaming path that removes Python per-token iteration from `NativeQSGEngine.generate_stream` and adapter continuous serving.

### OUT_OF_SCOPE_FUTURE_PHASE

- Phase 2 native continuous batching redesign beyond blocker documentation.
- Phase 3 fused native sampler/logits processing work beyond truthful hot-path accounting.
- KV allocator redesign, speculative serving, and quantization changes.

### OUT_OF_SCOPE_UNRELATED

- Unrelated benchmark harness changes outside Phase 1 evidence reporting.
- General repo drift not tied to hot-path ownership or Phase 1 verification.

## Exact In-Scope Files

- `core/native/native_qsg_engine.py`
- `core/native/parallel_generation.py`
- `tests/test_native_qsg_engine.py`
- `tests/test_native_parallel_generation_engine.py`
- `Parallel-First Native QSG Roadmap.md`

## Prerequisite Assumptions

- Existing native graph execution (`forward_token_id` / `forward_token_ids`) remains the underlying decode primitive.
- No native serve runtime object currently exposes the roadmap ABI (`create runtime`, `submit`, `poll`, `cancel`, `metrics`, `shutdown`).
- The current continuous engine path still uses Python iterator ownership and is therefore not eligible for a Phase 1 `PASS`.

## Implementation Steps

1. Update `NativeQSGEngine` generation telemetry so Python token-time loops increment `python_hot_path_calls`.
2. Update `NativeParallelGenerationEngine.metrics_snapshot()` to surface truthful runtime ownership, Phase 1 readiness, and blocker metadata using native scheduler iterations plus runtime telemetry.
3. Add targeted tests covering:
   - autoregressive/native streaming telemetry reports Python hot-path activity
   - continuous batching compatibility shim reports blocked Phase 1 status
4. Add a dated Phase 1 roadmap addendum with implemented artifacts, evidence summary, and remaining blockers.

## Verification Steps

- `pytest tests/test_native_qsg_engine.py tests/test_native_parallel_generation_engine.py`
- `source venv/bin/activate && saguaro verify . --engines native,ruff,semantic --format json`

## Exit Criteria

Phase 1 can only be marked `PASS` when:
- production token-time generation no longer uses Python per-token loops
- continuous batching runtime ownership is native rather than Python `run_forever`
- runtime telemetry and hot-path proof agree with the actual execution path
- verification passes and evidence shows the native ownership shift

For this session, success is narrower:
- runtime telemetry no longer falsely claims full native hot-path ownership
- blockers are explicit, tested, and recorded in the roadmap

## 2026-03-10 Execution Notes (Single-Phase Closure Attempt)

- Added explicit runtime ABI readiness surfacing:
  - `core/native/native_qsg_engine.py`: set `native_runtime_abi_ready` in runtime capabilities from native ABI match state.
- Added Phase 1 ownership regression coverage for ABI-ready path:
  - `tests/test_native_parallel_generation_engine.py`: new case verifies `continuous_runtime_owner` transitions to `native_runtime` when `native_runtime_abi_ready=True` while still reporting Python scheduler ownership blockers.
- No native scheduler/token-loop removal was implemented in this session; `run_forever` and direct stream iteration remain.
