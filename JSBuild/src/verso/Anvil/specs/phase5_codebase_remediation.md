# Phase 5 Codebase Remediation Spec

## Context
Phase 5 focuses on inherited governance debt and remediation automation.
This session is bounded to Phase 5 only.

## Session Scope
Wave coverage for this implementation session:
- Wave A (Safety blockers, complete): eliminate broad exception surface in high-frequency guard paths.
- Wave B (Trace closure, full for tooling): add remediation factory assets for traceability backfill and waiver expiry automation.
- Wave C (Complexity decomposition, complete): decompose remaining C901 hotspots in `core/agent.py` and `core/unified_chat_loop.py`.
- Wave D (Domain debt): out of scope for this session.

## Completion Push Scope (2026-03-03)
- Drive remaining Wave A + Wave C items toward closure with concrete decomposition in:
  - `core/agent.py`
  - `core/unified_chat_loop.py`
- Target measurable reductions for:
  - `C901` hotspots (function complexity),
  - `BLE001` blind exception handlers.
- Preserve runtime behavior; avoid no-op suppressions.

## Final Closure Push (2026-03-03, Pass Attempt)
- Eliminate remaining `C901` findings in:
  - `core/agent.py`
  - `core/unified_chat_loop.py`
- Keep `BLE001` at zero for both files.
- Re-run dead-code scan and capture deterministic triage evidence for phase-target files.
- Require independent implementer/verifier separation for final gate evidence.

## In-Scope Deliverables
1. `standards/remediation/remediation_board.md`
2. `scripts/aes_backfill_traceability.py`
3. `scripts/aes_waiver_expiry_check.py`
4. `scripts/aes_autofix_pipeline.sh`
5. Removal of stale `*.py.bak` artifacts.
6. Targeted low-risk hardening of broad exception handlers in `core/agent.py` and `core/unified_chat_loop.py` where semantics are deterministic and behavior-preserving.
7. Tests for new remediation scripts.
8. `ROADMAP_AES.md` Phase 5 implementation addendum + repair map + gate row update.

## Out Of Scope
- Repository-wide annotation/docstring retrofit.
- Repository-wide dead-code removal beyond deterministic script output.

## Design Decisions
- Use machine-readable JSON outputs for remediation scripts to support CI and dashboards.
- Preserve existing runtime behavior for exception-path hardening.
- Treat missing required closure artifacts as remediation findings, not silent passes.

## Verification Plan
- `pytest -q tests/test_phase5_remediation_scripts.py tests/test_aes_artifacts.py`
- `ruff check scripts/aes_backfill_traceability.py scripts/aes_waiver_expiry_check.py tests/test_phase5_remediation_scripts.py`
- `saguaro verify . --engines native,ruff,semantic,aes --format json` (expected repo backlog may remain; collect evidence)

## Baseline Metrics
- `*.py.bak` files: 3
- `except Exception` occurrences in `core/unified_chat_loop.py` + `core/agent.py`: 55
- Ruff BLE findings across `core/unified_chat_loop.py` + `core/agent.py`: 55

## Implementation Results (2026-03-03)
- `core/agent.py`:
  - Fixed `_current_wall_clock` runtime regression (`datetime` import restored).
  - Delegated `simple_chat` to helper module (`core/agent_loop_helpers.py`) to reduce monolith complexity.
- `core/unified_chat_loop.py`:
  - Prior decomposition pass remains in place with focused helper methods for evidence formatting/search/dedup paths.
- Remediation factory artifacts confirmed operational:
  - `.anvil/artifacts/phase5/traceability_backfill_report.json`
  - `.anvil/artifacts/phase5/waiver_expiry_report.json`

## Final Verification Evidence
- `source venv/bin/activate && ruff check core/agent.py core/unified_chat_loop.py --select C901,BLE001 --statistics`
  - Result: `0 C901`, `0 BLE001`.
- `source venv/bin/activate && python -m py_compile core/agent.py core/unified_chat_loop.py core/agent_tool_helpers.py core/agent_loop_helpers.py scripts/aes_backfill_traceability.py scripts/aes_waiver_expiry_check.py`
  - Result: pass.
- `source venv/bin/activate && pytest -q tests/test_phase5_remediation_scripts.py`
  - Result: `3 passed`.
- `source venv/bin/activate && pytest -q tests/test_unified_chat_loop.py tests/test_coconut_reranking.py tests/test_subagent_quality_gate.py`
  - Result: `8 passed`.
- `source venv/bin/activate && pytest -q tests/test_chat_delegation.py tests/test_chat_enhanced.py`
  - Result: `5 passed`.
- `source venv/bin/activate && saguaro verify core/agent.py --engines native,ruff,semantic --format json`
  - Result: `status=fail`, `count=154` (legacy non-phase-rule backlog remains).
- `source venv/bin/activate && saguaro verify core/unified_chat_loop.py --engines native,ruff,semantic --format json`
  - Result: `status=fail`, `count=556-557` (legacy non-phase-rule backlog remains).
- `source venv/bin/activate && saguaro deadcode`
  - Result: `Found 102 candidates`, `0` in Phase 5 target files.

## Final Metric Snapshot
- `except Exception` in `core/agent.py` + `core/unified_chat_loop.py`: `55 -> 0` (`-100%`).
- Ruff `BLE001` in `core/agent.py` + `core/unified_chat_loop.py`: `43 -> 0` (`-100%`).
- Ruff `C901` in `core/agent.py` + `core/unified_chat_loop.py`: `19 -> 0` (`-100%`).
- Combined `C901+BLE001`: `62 -> 0` (`-100%`).
