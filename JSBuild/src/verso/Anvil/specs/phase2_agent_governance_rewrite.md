# Phase 2 Agent Governance Rewrite Spec

## Scope

Phase 2 converts prompt governance from free-form guidance into structured, runtime-parseable contract inputs.
This phase is limited to:

- prompt policy rewrites in `prompts/GEMINI.md` and `prompts/shared_prompt_foundation.md`
- dynamic AES-aware prompt assembly in `core/prompts/`
- prompt contract emission with mandatory keys
- master/subagent/verification prompt packaging policy backed by token budgets
- mechanical AES compliance insertion into thinking chains
- deterministic compliance-context repair/retry semantics
- prompt contract linter and freshness guard
- prompt-policy integration in pre-commit, CI, and Sentinel `aes` verification

Out of scope:

- Phase 3 orchestration CRS upgrades
- Phase 4 domain template generation
- Phase 5 repo-wide remediation

## Traceability

- `AES-PH2-2.1`: AAL-native prompt policy and severity matrix must be explicit.
- `AES-PH2-2.2`: mandatory AES rules and anti-patterns must be encoded in shared prompt documents.
- `AES-PH2-2.3`: runtime 8-phase crosswalk must be documented for prompt/runtime alignment.
- `AES-PH2-2.4`: domain-conditional injection markers must map to deterministic detector markers.
- `AES-PH2-2.5`: `standards/AES_CONDENSED.md` must be mandatory baseline prompt payload.
- `AES-PH2-2.6`: prompt contract integrity and freshness must be machine-validated.
- `AES-PH2-2.7`: master/subagent/verification prompt context budgets must be enforced by code.
- `AES-PH2-2.8`: thinking chains must include mechanical compliance reasoning with required IDs.
- `AES-PH2-2.9`: missing compliance fields must trigger deterministic correction/retry, not silent continuation.

## Design Decisions

1. Added `AESPromptBuilder` as a dedicated governance payload builder, separate from legacy template formatting.
2. Prompt contracts are emitted in parseable `<AES_PROMPT_CONTRACT>` blocks with deterministic key/value fields.
3. Budgeting is enforced with `PromptContextBudgetPolicy` and section-level inclusion/drop metadata.
4. Master prompt payload remains condensed-first; domain packs are prioritized for subagent prompts.
5. `EnhancedThinkingSystem.start_chain()` injects a mandatory `COMPLIANCE` block before other thought types.
6. Compliance context now requires `trace_id`, `evidence_bundle_id`, `red_team_required`, and `waiver_ids`.
7. Prompt policy checks are executable (`scripts/validate_prompt_contracts.py`) and integrated into Sentinel `aes` via `AES-PRM-1`.

## Verification Plan

- unit tests for prompt contract keys and subagent domain scoping
- unit tests for mechanical compliance insertion and deterministic compliance repair
- unit tests for prompt-policy Sentinel integration (`AES-PRM-1`)
- script validation run:
  - `python scripts/validate_prompt_contracts.py --repo . --json`
- targeted pytest run for updated phase-2 surfaces
- Sentinel verification run:
  - `saguaro verify . --engines native,ruff,semantic,aes --format json`

## Exit Criteria

Phase 2 is complete when all are true:

- `PromptManager` emits required contract keys (`AAL_CLASSIFICATION`, `APPLICABLE_RULE_IDS`, `REQUIRED_ARTIFACTS`, `BLOCKING_GATES`)
- master/subagent/verification prompt packaging honors explicit token allocation policy
- thinking chains always contain a mechanical compliance block with required compliance fields
- missing/invalid compliance fields are corrected deterministically or fail loudly
- prompt contract linter is wired into pre-commit and CI and callable from Sentinel `aes`
- roadmap Phase 2 repair map and gate decision are updated with test/evidence references
