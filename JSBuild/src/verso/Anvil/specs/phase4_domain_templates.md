# Phase 4 Domain Templates Spec

Status: Draft for implementation
Scope: Phase 4 only (`4.0` to `4.8` in `ROADMAP_AES.md`)

## Objective
Provide AES-compliant domain code templates and deterministic template retrieval so generated code is compliant by construction across ML, quantum, physics, and HPC contexts.

## Non-Goals
- Legacy codebase remediation (Phase 5)
- CI pipeline hard gating and dashboard rollout (Phase 6)
- Enforcement-engine expansion beyond template parity checks (Phase 7)

## Required Outcomes
1. Domain template libraries exist for ML, quantum, physics, and HPC.
2. Templates emit default evidence metadata required by traceability/reporting flows.
3. Template registry supports deterministic lookup by `(domain, pattern)`.
4. Template selection supports Saguaro-first semantic candidate retrieval with deterministic fallback.
5. Domain checks and templates are pattern-aligned (checks pass on canonical templates).
6. Proving-ground/oracle fixtures exist for positive and adversarial cases.

## Design Decisions
- Keep templates as plain string scaffolds (model-consumable and serializable).
- Keep registry in `core/aes/template_registry.py` to colocate with AAL/domain governance primitives.
- Avoid hard dependency on live Saguaro for tests by making semantic retrieval optional.
- Record template selection decisions into a traceability artifact payload when provided.

## Data Contracts
- Template descriptor:
  - `template_id`
  - `domain`
  - `pattern`
  - `min_aal`
  - `source`
  - `content`
- Traceability emission payload (optional input dict):
  - `selected_template_ids: list[str]`
  - `candidate_scores: list[dict[str, object]]`
  - `retrieval_mode: str` (`semantic` or `deterministic`)

## Verification Plan
- Unit tests for registry exact lookup and domain-pattern parity.
- Unit tests for Saguaro-assisted selection with mocked candidate output.
- Oracle parity tests for domain templates and adversarial anti-pattern snippets.
- Sentinel verification run for `native,ruff,semantic,aes` against current repo state.

## Exit Criteria
- Phase 4 roadmap items are updated with status/evidence and no missing mandatory closure artifacts.
- Template registry returns deterministic results for exact lookups.
- Oracle tests pass for all four domains.
