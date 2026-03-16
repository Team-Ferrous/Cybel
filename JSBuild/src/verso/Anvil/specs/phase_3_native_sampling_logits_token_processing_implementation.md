# Phase 3 Implementation Spec: Native Sampling, Logits, and Token Processing

- Phase id: `3`
- Phase name: `## 9. Phase 3: Native Sampling, Logits, and Token Processing`
- Roadmap reference: `Parallel-First Native QSG Roadmap.md` sections `## 9`, `## 22.1`, `## 23.3`, `## 24.2`, `## 24.4`, and `## 26`

## Scope Classification

- `IN_SCOPE_IMPLEMENTABLE`
  - Add a native verification-scoring primitive for drafted token acceptance decisions.
  - Remove Python/NumPy softmax and argmax verification from active native speculative helpers.
  - Update targeted tests and evidence for the covered paths.
- `IN_SCOPE_BUT_BLOCKED_BY_PREREQUISITE`
  - Remove Python per-token autoregressive orchestration from `NativeQSGEngine._generate_autoregressive`.
  - Eliminate Phase 1 `python_hot_path_calls` ownership of the decode loop.
- `OUT_OF_SCOPE_FUTURE_PHASE`
  - Native scheduler ownership changes beyond the verification/sampling surface.
  - Phase 5+ speculative architecture redesign beyond acceptance verification.
  - Phase 6/7/8 model-level draft generation redesign.
- `OUT_OF_SCOPE_UNRELATED`
  - KV allocator redesign.
  - Quantization and CPU microarchitecture tuning not required for verification scoring.

## In-Scope Files

- `core/native/simd_ops.cpp`
- `core/native/simd_ops_wrapper.py`
- `core/native/parallel_generation.py`
- `core/native/native_qsg_engine.py`
- `tests/test_native_qsg_engine.py`
- `tests/test_native_parallel_generation_engine.py`
- `Parallel-First Native QSG Roadmap.md`

## Prerequisite Assumptions

- Native SIMD ops library builds successfully in the current environment.
- Existing native sampler entrypoints remain the authoritative sampling path.
- Python decode-loop ownership is not fixed in this phase and must be reported as a residual blocker.

## Implementation Steps

1. Add a native postprocess-and-score function that returns:
   - greedy token id
   - drafted token probability after temperature/top-k/top-p/min-p filtering
2. Expose that function through `simd_ops_wrapper.py` without Python fallback.
3. Replace `_stable_softmax` / `np.argmax` verification in:
   - `core/native/parallel_generation.py`
   - `core/native/native_qsg_engine.py` speculative verification paths
4. Keep Python logits processors gated as-is; do not broaden Phase 1 ownership changes.
5. Update tests to assert covered verification paths avoid NumPy-specific accounting and preserve acceptance behavior.

## Verification Steps

1. Run targeted unit tests for native QSG engine and parallel generation verification behavior.
2. Run native build/test validation needed by touched C++ bindings.
3. Run `source venv/bin/activate && saguaro verify . --engines native,ruff,semantic --format json`.

## Exit Criteria

- Covered speculative verification paths no longer use Python/NumPy softmax or argmax.
- Native verification scoring is exposed and exercised by tests.
- Remaining Phase 3 blocker is documented precisely as Phase 1 Python decode-loop ownership, not hidden.
