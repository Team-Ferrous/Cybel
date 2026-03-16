# HighNoon Audit Finalization Prompt (Phase 7 Red Team)

**How To Use:** Use this after Phases 0-6 are complete and a fresh audit rerun is available.

Read first:
- [Shared Prompt Foundation](shared_prompt_foundation.md)
- [Agent Audit Analysis Workflow](../workflows/AGENT_AUDIT_ANALYSIS_WORKFLOW.md)
- `AUDIT_ANALYSIS_ROADMAP.md`

---

## Session Header

```text
PHASE: 7
PHASE_NAME: Red Team Final Validation and Sign-Off
BUNDLE: artifacts/audit_visuals/<LATEST or explicit timestamp>
MODE: Finalization
```

---

## Role

Act as HighNoon Red Team Lead performing production GO/NO-GO determination.

---

## Required Inputs

- Latest `run_scorecard.json`
- Full `AUDIT_ANALYSIS_ROADMAP.md`
- Baseline Chronicle entry for this cycle
- Verification evidence from implemented fixes

---

## Finalization Procedure

1. Re-read all phase sections in roadmap and validate gate closure consistency.
2. Build KPI delta table: baseline -> latest -> delta -> status.
3. Re-check unresolved P0/P1 items and ownership status.
4. Validate complexity compliance (no new O(n^2)+ critical paths).
5. Validate SIMD/OpenMP posture for changed hot paths.
6. Validate regression budget across Golden Signals.
7. Run FMEA/FTA closure check for critical issues.
8. Run security/compliance quick pass (OWASP/SOC2-oriented checks).
9. Issue GO/NO-GO with explicit blockers.
10. If GO, generate `docs/PRODUCTION_SIGNOFF_YYYY-MM-DD.md`.

---

## Final Output Format

1. `Gate Dashboard Summary`
2. `KPI Delta Table`
3. `Unresolved Risk Register`
4. `Compliance Checks`
5. `Verdict`: GO or NO-GO
6. `Blockers`: required remediation if NO-GO
7. `Sign-Off Artifact`: file path if GO

---

## Automatic NO-GO Conditions

- Any unresolved P0.
- Any unresolved P1 designated release-blocking.
- Any missing verification evidence for critical fixes.
- Any major regression outside approved budget.
- Any critical security/compliance issue not accepted with owner/date.

---

## Handoff Format

```text
FINALIZATION_COMPLETE: true
VERDICT: <GO/NO-GO>
BLOCKERS: <none or list>
NEXT_ACTION: <release or remediation cycle>
```

