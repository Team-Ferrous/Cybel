# Phase 11 Implementation Spec

- Phase id: `11`
- Phase name: `Model-Level Changes for Future Checkpoints`
- Roadmap section: `Latent_Space_QSG_Roadmap.md :: ## 26. Phase 11: Model-Level Changes for Future Checkpoints`
- Date: `2026-03-10`

## Phase execution map

### Goals

- make future checkpoint requirements explicit for CPU-first latent and multi-token serving
- define checkpoint-family direction for `GQA` or `MQA`, `MTP`, `early-exit`, `latent-intercept`, and quantization-friendly layers
- encode latent-specific requirements for recurrence, evidence projection compatibility, and resume robustness
- provide regression coverage so the Phase 11 artifact contract stays testable

### Exact in-scope files

- `specs/phase_11_model_level_future_checkpoints_implementation.md`
- `docs/cpu_first_checkpoint_strategy.md`
- `tests/test_phase_11_checkpoint_strategy.py`
- `Latent_Space_QSG_Roadmap.md`
- `runs/roadmap/phase_11/verification_summary.md`

### Primary-owned files

- `specs/phase_11_model_level_future_checkpoints_implementation.md`
- `docs/cpu_first_checkpoint_strategy.md`
- `tests/test_phase_11_checkpoint_strategy.py`
- `runs/roadmap/phase_11/verification_summary.md`

### Allowed shared files

- `Latent_Space_QSG_Roadmap.md`

### Read-only dependency files

- `prompts/shared_prompt_foundation.md`
- `core/model/model_profile.py`
- `core/native/model_graph.cpp`
- `core/native/model_graph_wrapper.py`
- `core/native/native_qsg_engine.py`
- `core/native/runtime_telemetry.py`
- `core/qsg/config.py`

### Artifact namespace

- `specs/phase_11_model_level_future_checkpoints_implementation.md`
- `runs/roadmap/phase_11/`
- `.anvil/phase_11/`
- `benchmarks/phase_11/`

### Subagents

- `explorer`: map existing runtime surfaces that already constrain Phase 11
- `architecture_researcher`: validate checkpoint-family recommendations against CPU-first runtime realities
- `roadmap_verifier`: confirm roadmap addendum stays surgical and Phase 11-only
- `verifier`: run targeted tests and governance verification

## Current reality and dependency review

### Satisfied prerequisites

- `core/model/model_profile.py` already normalizes `n_kv_heads` and derives `gqa` from checkpoint metadata.
- `core/native/model_graph.cpp` already implements fused `GQA` or `MQA` attention paths and native execution checkpoint capture and restore surfaces.
- `core/native/model_graph_wrapper.py` already exposes `n_kv_heads` wiring and execution checkpoint APIs to Python control-plane code.
- `core/native/native_qsg_engine.py` already infers concrete KV-head layout from weights, discovers draft-head tensors for `MTP`-style auxiliary heads, and captures exit-hidden state for early-exit self-spec paths.
- `core/native/runtime_telemetry.py` already exposes native-hot-path proof fields that future checkpoint families must preserve.
- Phase 11 is design-scoped in the roadmap, so no additional native bridge is required to make the strategy artifact testable.

### Blocking prerequisites outside allowed bridge scope

- Operational `MTP` throughput claims remain blocked on Phase 6 runtime integration.
- Operational early-exit reuse claims remain blocked on Phase 7 runtime evidence and quality gates.
- Low-bit checkpoint acceptance remains blocked on Phase 9 runtime maturity and Phase 12 governance.

## Phase classification

- `IN_SCOPE_IMPLEMENTABLE`
  - publish a Phase 11 strategy artifact aligned to the latent roadmap
  - define explicit future checkpoint requirements for CPU-first latent serving
  - document runtime anchor surfaces so serving teams know what later phases must target
  - add regression tests for the strategy artifact and roadmap addendum
- `IN_SCOPE_BUT_BLOCKED_BY_PREREQUISITE`
  - proving throughput or quality gains from net-new checkpoint families
  - training and validating `MTP`, early-exit, or quantization-aware checkpoints
  - demonstrating latent resume robustness on trained future checkpoints
- `OUT_OF_SCOPE_FUTURE_PHASE`
  - Phase 6 native multi-token proposal flow
  - Phase 7 early-exit runtime reuse
  - Phase 9 low-bit runtime ownership
  - Phase 12 quality and calibration governance
- `OUT_OF_SCOPE_UNRELATED`
  - unrelated native runtime refactors already in progress outside the listed files

## Prerequisite assumptions

- Phase 11 closure is documentation-, evidence-, and regression-driven rather than requiring live model training.
- The strategy is valid only if it names concrete checkpoint families and latent-specific hooks, not generic research directions.
- Existing dirty-worktree changes outside the Phase 11 files remain untouched.

## Blocking risks

- The shared roadmap file is being edited in other phase chats, so the Phase 11 addendum must stay limited to the Phase 11 section.
- Future checkpoint guidance can drift into runtime implementation claims if the spec is not kept design-bounded.
- Repository-level `saguaro verify` failures may include unrelated drift and must be separated from Phase 11 results.

## Implementation steps

1. Align the Phase 11 spec and tests to `Latent_Space_QSG_Roadmap.md`.
2. Extend the checkpoint strategy with latent-specific hooks: latent recurrence, evidence projection compatibility, latent-intercept supervision, and latent resume robustness.
3. Record the current runtime anchor surfaces from `core/native/model_graph.cpp`, `core/native/model_graph_wrapper.py`, `core/native/native_qsg_engine.py`, and `core/native/runtime_telemetry.py`.
4. Add a dated implementation addendum to the Phase 11 section of `Latent_Space_QSG_Roadmap.md`.
5. Capture targeted verification output in `runs/roadmap/phase_11/`.

## Verification steps

1. `pytest -q tests/test_phase_11_checkpoint_strategy.py`
2. `source venv/bin/activate && saguaro verify . --engines native,ruff,semantic --format json`

## Exit criteria

- Phase 11 artifacts reference the latent roadmap, not a missing or unrelated roadmap file.
- The strategy defines at least one primary CPU-first checkpoint family and covers `GQA` or `MQA`, `MTP`, `early-exit`, `latent-intercept`, and quantization-friendly design.
- The strategy explicitly includes latent recurrence support, evidence projection compatibility, and latent resume robustness goals.
- Regression tests fail if the strategy artifact or Phase 11 addendum is removed or weakened.
- Final status may be `PASS` only if targeted tests and `saguaro verify` complete and any unrelated repo drift is separated from phase-local results.
