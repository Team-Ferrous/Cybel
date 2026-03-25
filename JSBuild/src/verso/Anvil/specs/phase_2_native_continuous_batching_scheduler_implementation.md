# Phase 2 Implementation Spec

- Phase id: `2`
- Phase name: `Native Continuous Batching and Scheduler Redesign`
- Roadmap section: `## 17. Phase 2: Native Continuous Batching and Scheduler Redesign`
- Date: `2026-03-10`

## Scope

This execution is bounded to the active Phase 2 continuous-batching path used by `OllamaQSGAdapter` when `continuous_batching_enabled=True`.

### In scope

- `core/native/qsg_parallel_kernels.h`
- `core/native/qsg_parallel_kernels.cpp`
- `core/native/qsg_parallel_kernels_wrapper.py`
- `core/native/parallel_generation.py`
- `core/qsg/ollama_adapter.py`
- `tests/test_native_parallel_generation_engine.py`
- `tests/test_qsg_adapter_unification.py`
- `Parallel-First Native QSG Roadmap.md`

### Out of scope

- Full native decode/prefill execution ownership
- Native sampler/logits fusion work from Phase 3
- KV allocator redesign from Phase 4
- Model-architecture changes from later phases

## Current reality and dependency review

### Satisfied prerequisites

- Native scheduler storage and metrics ABI already exist in `qsg_parallel_kernels.*`.
- Continuous batching is already wired through `OllamaQSGAdapter._init_continuous_engine`.
- Native runtime already exposes prompt tokenization and batched prefill metrics on single-request generation paths.

### Blocking prerequisite outside allowed bridge scope

- `NativeParallelGenerationEngine.run_forever` still owns the steady-state iteration in Python.
- The active stream producer remains a Python iterator around `_stream_generate_qsg`.
- Because of that, the roadmap acceptance clause `scheduler loop is fully native` cannot be satisfied in this session without completing a larger hot-path migration that overlaps Phase 1 and later Phase 2 work.

## Phase classification

- `IN_SCOPE_IMPLEMENTABLE`
  - preserve prompt-token metadata on the active native continuous-batching path
  - export native scheduler prefill/decode class accounting
  - export chunked-prefill scheduling counters and distributions through the active path
  - test the adapter and native scheduler façade around those metrics
- `IN_SCOPE_BUT_BLOCKED_BY_PREREQUISITE`
  - fully native scheduler loop
  - measurable TTFT improvement against a prior native-loop baseline
- `OUT_OF_SCOPE_FUTURE_PHASE`
  - native token sampling fusion
  - paged KV / copy-on-write branch state
- `OUT_OF_SCOPE_UNRELATED`
  - AES/governance prompt rewrites

## Implementation steps

1. Extend native scheduler request metadata and metrics to understand prefill-vs-decode state and chunked-prefill estimates.
2. Pass prompt-token metadata into the native continuous-batching path from `OllamaQSGAdapter`.
3. Update `NativeParallelGenerationEngine` to notify the native scheduler about phase transitions and to surface the new metrics.
4. Add targeted tests for native scheduler metrics and adapter request preparation.
5. Update the roadmap Phase 2 section with a dated addendum describing the shipped bridge and the remaining blocker.

## Verification

1. Run targeted tests for `NativeParallelGenerationEngine`.
2. Run targeted tests for the continuous-batching adapter integration.
3. Validate the native build surfaces still compile for the touched C++ bridge.
4. Run `source venv/bin/activate && saguaro verify . --engines native,ruff,semantic --format json`.

## Session Execution Update

- Date: `2026-03-10`
- Status: `PARTIAL`
- PRIMARY_OWNED_FILES:
  - `core/native/qsg_parallel_kernels.cpp`
  - `core/native/qsg_parallel_kernels_wrapper.py`
  - `core/native/parallel_generation.py`
  - `specs/phase_2_native_continuous_batching_scheduler_implementation.md`
  - `tests/test_native_parallel_generation_engine.py`
- SHARED_FILES:
  - `core/qsg/ollama_adapter.py`
  - `tests/test_qsg_adapter_unification.py`
- READ_ONLY_FILES:
  - `Latent_Space_QSG_Roadmap.md`
  - `Parallel-First%20Native%20QSG%20Roadmap.md`
- ARTIFACT_NAMESPACE:
  - `specs/phase_2_native_continuous_batching_scheduler_implementation.md`
  - `runs/roadmap/phase_2/`
  - `.anvil/phase_2/`
- SUBAGENT_PLAN:
  - No subagents were spawned in this single-path closure run.
- Changes made:
  - Added native scheduler latent/suspended state setters and ABI symbols to Python wrapper binding.
  - Wired continuous submit path to propagate latent/suspended request options to the native scheduler.
  - Added request-state marker assertions in scheduler integration tests.
- Verification status:
  - Native bridge wiring is implemented and import-safe.
  - Python metrics path now carries latent/suspended counters through scheduler `metrics()` and native integration snapshots.
  - `saguaro verify` and full performance benchmarks not run in this session.

## Exit criteria

- Native scheduler metrics include class-aware prefill/decode accounting on the active continuous-batching path.
- Continuous requests carry prompt-token metadata into the native scheduler façade.
- Tests cover the new metrics and adapter wiring.
- Roadmap addendum is updated.
- Final status must not be `PASS` unless the fully native scheduler loop clause is actually satisfied.
