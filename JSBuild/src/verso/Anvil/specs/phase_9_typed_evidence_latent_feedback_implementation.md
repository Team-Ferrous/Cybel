# Phase 9 Typed Evidence and Latent Feedback Implementation

## Phase
- `PHASE_ID`: 9
- `PHASE_NAME`: Typed Evidence and Latent Feedback
- `ROADMAP_PATH`: `Latent_Space_QSG_Roadmap.md`
- `ROADMAP_SECTION`: `## 9. Typed Evidence and Latent Feedback`

## Source-Of-Truth Notes
- The implementation prompt labels Phase 9 as low-bit inference, but the repository source of truth currently defines Phase 9 as typed evidence and latent feedback.
- An unrelated untracked spec already exists for `specs/phase_9_quantization_numeric_redesign_implementation.md`; this work does not take ownership of that document or its implied scope.
- Saguaro health succeeded, but index coverage is partial and some `saguaro agent` reads timed out. Discovery used Saguaro first, then narrow fallback reads on the exact files Saguaro surfaced.

## Scope Classification
- `IN_SCOPE_IMPLEMENTABLE`
  - Formalize a typed evidence capsule for subprocess-backed tool completions in the audit runner.
  - Persist inspectable evidence artifacts alongside current run outputs.
  - Add replay references tying evidence to checkpoint metadata and the suite event timeline.
  - Add deterministic latent feedback projection data derived from typed evidence.
  - Add targeted schema and unit-test coverage.
- `IN_SCOPE_BUT_BLOCKED_BY_PREREQUISITE`
  - Native C++ `EvidenceCapsuleEncoder` implementation from roadmap 9.5 is not present in the current runtime surfaces.
- `OUT_OF_SCOPE_FUTURE_PHASE`
  - Generic latent packet consumption in serving hot paths.
  - Quantization, low-bit inference, or parked-state memory work.
- `OUT_OF_SCOPE_UNRELATED`
  - Existing native QSG hot-path performance changes already in progress in unrelated files.

## Ownership
- `PRIMARY_OWNED_FILES`
  - `specs/phase_9_typed_evidence_latent_feedback_implementation.md`
  - `audit/evidence_capsule.py`
  - `audit/schemas/evidence_capsule.schema.json`
  - `tests/audit/test_evidence_capsule.py`
- `SHARED_FILES`
  - `audit/runtime_logging.py`
  - `audit/runner/attempt_executor.py`
  - `audit/runner/benchmark_suite.py`
  - `audit/store/suite_layout.py`
  - `audit/schemas/attempt_record.schema.json`
  - `audit/schemas/checkpoint.schema.json`
  - `tests/audit/test_attempt_executor.py`
  - `tests/audit/test_benchmark_suite.py`
  - `Latent_Space_QSG_Roadmap.md`
- `READ_ONLY_FILES`
  - `prompts/shared_prompt_foundation.md`
  - `core/checkpoint.py`
  - `core/native/runtime_telemetry.py`
  - `audit/provenance/capture.py`
  - `tests/audit/test_runtime_logging.py`

## Prerequisite Assumptions
- Audit suite subprocess execution remains the authoritative current control-plane surface for tool completion evidence.
- Existing checkpoint metadata and `events.ndjson` timeline are the replay anchors for this phase.
- Any change to shared audit files must remain surgical because the worktree contains unrelated ongoing roadmap edits.

## Implementation Steps
1. Add a typed evidence capsule model and schema with the roadmap 9.3 minimum fields plus replay and latent projection sections.
2. Extend logged subprocess execution to capture wall time and child resource usage and persist evidence capsules as inspectable artifacts.
3. Thread evidence capsule references into attempt records and checkpoint metadata without changing unrelated benchmark semantics.
4. Add deterministic latent feedback projection generation from typed evidence for audit/control-plane use.
5. Add targeted unit tests for capsule schema validation, subprocess evidence persistence, and replay references.
6. Update the roadmap Phase 9 section with a dated implementation addendum if the phase is materially advanced.

## Verification Steps
- `pytest tests/audit/test_evidence_capsule.py tests/audit/test_attempt_executor.py tests/audit/test_benchmark_suite.py tests/audit/test_runtime_logging.py`
- `source venv/bin/activate && saguaro verify . --engines native,ruff,semantic --format json`
- Generate a phase artifact showing roadmap 9.3 required-field coverage before and after the implementation.

## Artifact Namespace
- `specs/phase_9_typed_evidence_latent_feedback_implementation.md`
- `runs/roadmap/phase_9/`
- `.anvil/phase_9/`
- `benchmarks/phase_9/`

## Exit Criteria
- Evidence capsules are written as inspectable artifacts for subprocess-backed audit tool completions.
- Capsules include the roadmap 9.3 minimum fields and deterministic result hashing.
- Replay references identify checkpoint metadata and suite event timeline artifacts without re-running the model.
- Targeted tests pass and schema validation covers the new capsule format.
- `saguaro verify` is run and any unrelated failures are separated from phase-local results.
