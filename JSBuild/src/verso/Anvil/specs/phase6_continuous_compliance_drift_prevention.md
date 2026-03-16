# Phase 6 Spec: Continuous Compliance and Drift Prevention

Date: 2026-03-03
Owner: HighNoon Lead Implementation Engineer
Phase: 6 (Continuous Compliance & Drift Prevention)

## Scope

This phase implements roadmap items 6.0-6.8 with phase-bounded changes:

- 6.1 `saguaro verify --aes-report` compliance dashboard surface
- 6.2 Chronicle protocol integration in action workflows
- 6.3 Auto-legislator drift-prevention loop (runtime trigger wiring)
- 6.4 pre-commit integration for Ruff + Saguaro verify
- 6.5 CI hard-gate workflow for verify/deadcode/impact/audit
- 6.6 compliance telemetry dashboard specification
- 6.8 external baseline drift intake process

## Non-goals

- No cross-phase monolith decomposition work (Phase 5 backlog remains out of scope)
- No new fallback/synthetic logic paths
- No weakening of existing AAL governance checkpoints

## Design Decisions

### D1. AES report is derived from verification violations

`verify --aes-report` computes category compliance from `api.verify(...)` outputs:

- Categories: Traceability, Type Safety, Error Handling, Security, Dead Code, Complexity, Documentation
- Category mapping is deterministic from `rule_id` and message heuristics
- Report is available in text or JSON using `--format`
- Exit code behavior remains unchanged: verification failure still exits non-zero

Rationale: keep report generation coupled to enforceable findings and compatible with CI gates.

### D2. Chronicle is wired directly into action execution for high-assurance changes

For AAL-0/AAL-1 action plans containing file-mutating tools:

1. pre-action: chronicle snapshot
2. execute tools and verification loop
3. post-action: chronicle diff
4. persist delta receipt under `aiChangeLog/`

Rationale: enforceable runtime evidence closure in the same flow that mutates code.

### D3. Drift prevention loop triggers legislator drafts in high-assurance cycles

After high-assurance mutation cycles, the runtime invokes legislation drafting and stores the output to `.saguaro.rules.draft`.

Rationale: Phase 6 requires continuous learn loop; this wiring makes it operational without cross-phase rule promotion redesign.

### D4. CI hard gate lives in infrastructure while remaining runnable in GitHub workflows

Canonical workflow definition is added at `infrastructure/ci/aes_compliance.yml` and designed to:

- run `verify/deadcode/impact/audit`
- parse JSON outputs
- fail on unresolved high-severity governance findings

Rationale: satisfy roadmap artifact contract and provide branch-protection-ready policy logic.

## Files Planned

- `saguaro/cli.py`
- `domains/code_intelligence/saguaro_substrate.py`
- `core/unified_chat_loop.py`
- `.pre-commit-config.yaml`
- `infrastructure/ci/aes_compliance.yml`
- `infrastructure/monitoring/aes_dashboard_spec.md`
- `infrastructure/monitoring/aes_baseline_review.md`
- `tests/test_phase6_continuous_compliance.py`
- `ROADMAP_AES.md`

## Verification Plan

- Unit/targeted tests for new report and chronicle helpers
- Ruff on changed Python files
- Command evidence capture:
  - `saguaro verify . --engines native,ruff,semantic,aes --format json`
  - `saguaro verify . --engines native,ruff,semantic,aes --aes-report --format json`
  - `saguaro deadcode --format json`
  - `saguaro impact --path <changed_file>`
  - `saguaro audit --format json`

## Risk Notes

- Existing Saguaro runtime logs are noisy (TensorFlow/native debug); command parsers must tolerate preamble noise.
- Repository has significant pre-existing Sentinel backlog; Phase 6 gates may remain PARTIAL until backlog burn-down.
