# Phase 7 Spec: AES Testing Infrastructure

## Scope
Phase 7 validates the AES enforcement stack itself under normal and adversarial conditions.

In-scope repairs:
- P0: Bypass attempts must fail with deterministic blocking reason/rule ID.
- P0/P1: AAL-0/1 closure requirements must be mechanically test-enforced (traceability, evidence, review, waiver, red-team).
- P1: Dedicated test coverage for AES check functions and classifier/pipeline behavior.
- P1: AES engine behavior tests for rule loading, universal checks, and domain scoping.
- P1: Agent behavior regressions for compliance contract generation and CRS escalation wiring.
- P1: Chaos governance drill script validating fail-closed behavior.

## Design Decisions
1. Reuse current enforcement APIs (`AALClassifier`, `DomainDetector`, `AESRuleRegistry`, `AESEngine`, `ReviewGate`, `RedTeamProtocol`) rather than introducing new runtime surfaces in Phase 7.
2. Keep behavior tests deterministic by exercising prompt/thinking/governance outputs without external model calls.
3. Use temp-repo fixtures for artifact/waiver/review/red-team tests so failures are isolated and reproducible.
4. Require explicit assertion of blocking metadata (`rule_id`, `closure_level`, or deterministic failure reason) in adversarial tests.

## Repair Mapping
- R7-0: Adversarial verification matrix -> `tests/test_aes_pipeline.py` + `tests/test_saguaro_aes_engine.py`
- R7-1: Check-function unit suites -> `tests/test_aes_checks/*.py`
- R7-2: AAL classification pipeline integration -> `tests/test_aes_pipeline.py`
- R7-3: Agent behavior regression -> `tests/test_aes_agent_behavior.py`
- R7-4: Saguaro AES engine tests -> `tests/test_saguaro_aes_engine.py`
- R7-5: Traceability closure tests -> `tests/test_aes_traceability.py`
- R7-6: Evidence closure tests -> `tests/test_aes_evidence_closure.py`
- R7-7: Waiver governance tests -> `tests/test_aes_waivers.py`
- R7-8: Red-team closure tests -> `tests/test_aes_red_team.py`
- R7-9: Review matrix enforcement tests -> `tests/test_aes_governance_reviews.py`
- R7-10: Chaos governance drill -> `scripts/aes_chaos_drill.py`

## Acceptance Criteria
- All new Phase 7 suites pass locally.
- Each adversarial bypass case fails predictably with test assertions tied to rule ID or blocking reason.
- High-assurance closure tests cover AAL-0/AAL-1 requirements for traceability, evidence, waivers, reviews, and red-team artifacts.
- Roadmap Phase 7 section updated with implementation addendum + repair status table + gate update.

## Verification Commands
- `pytest tests/test_aes_checks tests/test_aes_pipeline.py -v`
- `pytest tests/test_aes_agent_behavior.py tests/test_saguaro_aes_engine.py -v`
- `pytest tests/test_aes_traceability.py tests/test_aes_evidence_closure.py tests/test_aes_waivers.py tests/test_aes_red_team.py tests/test_aes_governance_reviews.py -v`
- `python scripts/aes_chaos_drill.py --repo . --json`

## Risks
- Some repositories may not have `standards/waivers/`; tests must treat missing directory as valid unless strict mode requires waivers.
- Prompt/governance tests must avoid flaky dependencies on live model/network behavior.
