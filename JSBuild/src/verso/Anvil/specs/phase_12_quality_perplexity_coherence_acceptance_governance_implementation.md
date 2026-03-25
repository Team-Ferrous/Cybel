# Phase 12 Implementation Spec

- Phase id: `12`
- Phase name: `Quality, Perplexity, Coherence, and Acceptance Governance`
- Roadmap section: `## 27. Phase 12: Quality, Perplexity, Coherence, and Acceptance Governance`
- Date: `2026-03-10`

## Scope

This execution is bounded to Phase 12 governance inside the existing audit benchmark suite and its persisted artifacts.

### In scope

- `audit/runner/benchmark_suite.py`
- `tests/audit/test_benchmark_suite.py`
- `specs/phase_12_quality_perplexity_coherence_acceptance_governance_implementation.md`
- `Latent_Space_QSG_Roadmap.md`

### Out of scope

- Native speculative runtime implementation
- Native non-AR runtime implementation
- New model checkpoints or training hooks
- Scheduler, sampler, or KV hot-path rewrites from earlier phases

## Current reality and dependency review

### Satisfied prerequisites

- The suite already runs held-out perplexity, confidence calibration, and coherence rubric evaluations.
- The quality lane already records structural validity and degeneration evidence inside coherence records.
- The benchmark harness already emits speculative acceptance counters and mode metadata when such modes are active.
- Calibration contracts already gate regressions in perplexity, confidence, coherence, TTFT, and decode throughput.
- The attempt records already expose drift telemetry, raw/finalized decode surfaces, and checkpoint state that Phase 12 can govern without hot-path changes.

### Blocking prerequisites outside allowed bridge scope

- No benchmark profile currently exercises speculative or non-AR serving as first-class configurations.
- Native speculative and non-AR execution remain roadmap work in Phases 5 through 8, so Phase 12 cannot truthfully claim full first-class mode coverage in this session.
- Repo-level governance is currently red because `source venv/bin/activate && saguaro verify . --engines native,ruff,semantic --format json` fails on unrelated drift in `saguaro/api.py:34` (`F401` unused import).

## Phase classification

- `IN_SCOPE_IMPLEMENTABLE`
  - add explicit latent-to-text faithfulness and evidence-capsule fidelity gates to the existing quality-governance report
  - surface hidden-drift detection and resume-quality status in summary-level acceptance governance
  - add targeted tests for latent projection failures, hidden-drift coverage, and resume-quality coverage/failure
- `IN_SCOPE_BUT_BLOCKED_BY_PREREQUISITE`
  - benchmark speculative modes as first-class configurations
  - benchmark non-AR modes as first-class configurations
- `OUT_OF_SCOPE_FUTURE_PHASE`
  - native acceptance telemetry plumbing needed by speculative and non-AR production paths
  - future checkpoint-level calibration work under quantization and early-exit phases
- `OUT_OF_SCOPE_UNRELATED`
  - unrelated runtime, build, or roadmap drift outside Phase 12 governance

## Implementation steps

1. Extend `benchmark_suite._quality_governance_report()` with a latent projection report backed by existing coherence-record evidence.
2. Extend `benchmark_suite._acceptance_governance_report()` with hidden-drift and resume-quality summaries, without changing runtime ownership.
3. Keep speculative and non-AR mode coverage as blocked-by-prerequisite unless first-class evidence exists.
4. Add targeted tests that prove the new latent projection and resume-quality gates work.
5. Update the Phase 12 roadmap section with a dated implementation addendum and strict status.

## Verification

1. Run `source venv/bin/activate && pytest -q tests/audit/test_benchmark_suite.py`.
2. Run focused governance regression tests for the new latent/resume gates.
3. Attempt a live smoke suite run for real benchmark evidence.
4. Run `source venv/bin/activate && saguaro verify . --engines native,ruff,semantic --format json`.

## Parallel ownership

- PRIMARY_OWNED_FILES:
  - `audit/runner/benchmark_suite.py`
  - `tests/audit/test_benchmark_suite.py`
  - `specs/phase_12_quality_perplexity_coherence_acceptance_governance_implementation.md`
- SHARED_FILES:
  - `Latent_Space_QSG_Roadmap.md`
- READ_ONLY_FILES:
  - `audit/profiles/native_qsg_smoke.yaml`
  - `scripts/run_native_qsg_suite.sh`
  - `audit/schemas/summary.schema.json`
- ARTIFACT_NAMESPACE:
  - `specs/phase_12_quality_perplexity_coherence_acceptance_governance_implementation.md`
  - `audit/runs/phase12_smoke_20260310T181849Z/`
  - `.anvil/phase_12/`
  - `runs/roadmap/phase_12/`

## Subagents

- `verifier`:
  - inspected the benchmark-suite tests, suite profiles, script wrapper, and existing run artifacts
  - confirmed missing latent faithfulness and resume-quality coverage before the patch
  - reported that repo-level `saguaro verify` drift is unrelated to Phase 12
- `explorer`:
  - two tightly scoped explorer attempts were interrupted before delivering usable output
  - no explorer output was used as evidence

## Session execution update

- Date: `2026-03-10`
- Status: `PARTIAL`
- Workset:
  - attempted `saguaro workset create --desc "Phase 12 Quality Governance" ...`
  - failed with `EXCEEDS HARD LIMIT (78098 > 32000)`, so no workset artifact was claimed
- Changes made:
  - added `latent_projection` reporting to the quality governance summary
  - added `latent_to_text_faithfulness` and `evidence_capsule_fidelity` benchmark-family coverage
  - added `hidden_drift` and `resume_quality` reporting to acceptance governance
  - added regression tests for latent control-tag leakage and resume-quality failures
- Verification status:
  - `pytest -q tests/audit/test_benchmark_suite.py` passed: `43 passed`
  - focused governance regressions passed: `4 passed`
  - smoke run `phase12_smoke_20260310T181849Z` reached live native execution and produced decode/TTFT/drift evidence for the canonical lane
  - `saguaro verify` failed on unrelated repo drift in `saguaro/api.py:34`

## Exit criteria

- Quality artifacts include latent-specific governance instead of only generic coherence checks.
- Final summary includes artifact completeness, mode coverage, hidden-drift visibility, and resume-quality classification.
- Missing speculative and non-AR first-class coverage is reported as blocked-by-prerequisite, not hidden.
- Targeted tests pass.
- Final session status remains below `PASS` unless the live suite completes, repo-level `saguaro verify` is green or cleanly phase-exempted, and first-class speculative/non-AR coverage actually exists.
