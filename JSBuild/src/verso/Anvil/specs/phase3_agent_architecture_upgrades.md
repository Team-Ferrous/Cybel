# Phase 3 Agent Architecture Upgrades Spec

Status: Draft for implementation
Scope: Phase 3 only (`3.0` to `3.9` in `ROADMAP_AES.md`)

## Objective
Make AES governance structural in execution paths (master/subagent orchestration, runtime loops, recovery, and finalization), not advisory prompt-only behavior.

## Non-Goals
- Repository-wide remediation backlog (Phase 5)
- New domain template libraries (Phase 4)
- CI-only continuous enforcement rollout (Phase 6)

## Required Outcomes
1. Master orchestration must classify AAL/domain and enforce quality gate before accepting subagent output.
2. Subagents must run AES-aware prompting and self-verification with blocking behavior for `AAL-0`/`AAL-1`.
3. Runtime loops must enforce checkpoints at post-evidence, pre-action, and pre-finalization.
4. High-AAL finalization must require verify + evidence bundle + review closure.
5. Recovery strategies that mask symptoms (`fallback tool`, `faster model`) must be blocked for high AAL without waiver.
6. Sentinel policy blocking outcomes must terminate/escalate, never degrade to warn-only for high AAL.
7. Red-team and review artifacts must be modeled and validated with deterministic contracts.

## Design Decisions
- Reuse `core.aes.AALClassifier`, `core.aes.DomainDetector`, and `core.aes.rule_registry.AESRuleRegistry` from Phase 0.
- Keep `agents/master.py` callable as adapter; governance behavior delegates to upgraded unified orchestration semantics.
- Use additive governance modules under `core/aes/`:
  - `governance.py` for hardcoded tier policy
  - `red_team_protocol.py` for artifact contract/checks
  - `review_gate.py` for independent-review matrix enforcement
  - `action_escalation.py` for irreversible-action classification and escalation checks
- Keep hook API compatible with `infrastructure.hooks.registry.HookRegistry`; add AES-specific hook implementations in `infrastructure/hooks/builtin.py`.

## Data Contracts
- Compliance context keys required in runtime paths:
  - `trace_id`
  - `evidence_bundle_id`
  - `red_team_required`
  - `waiver_ids`
- Subagent packaging contract:
  - `result`
  - `verification` (`aal`, `violations`, `passed`, `blocking_count`)
  - `compliance` (`trace_id`, `evidence_bundle_id`, `red_team_required`, `waiver_ids`)
  - `artifacts` (red-team and review placeholders when required)

## Gates
- `SubagentQualityGate.evaluate(...)` must execute before synthesis consumption.
- High-AAL finalization gate checks:
  - Sentinel verify pass for required engines
  - evidence bundle present/valid
  - review gate pass
  - red-team artifact contract satisfied when required
- Error recovery restrictions:
  - `SWITCH_TO_FASTER_MODEL` and `FALLBACK_TO_ALTERNATIVE_TOOL` blocked for `AAL-0`/`AAL-1` unless waiver exists.

## Verification Plan
- Unit tests:
  - master/unified orchestration AAL + CRS + gate behavior
  - subagent self-verify behavior by AAL
  - error recovery restrictions with/without waiver
  - governance/review/red-team/action escalation contracts
  - unified/enterprise loop checkpoint blocks
- Existing regression suites:
  - `tests/test_phase2_prompt_governance.py`
  - `tests/test_aes_runtime_contracts.py`
  - `tests/test_unified_chat_loop.py`
  - `tests/test_unified_master_agent.py`
  - `tests/test_master_agent.py`

## Exit Criteria
- Phase 3 roadmap items marked `DONE` only when code + tests + evidence + roadmap update are all present.
- Any unmet item is `PARTIAL`.
