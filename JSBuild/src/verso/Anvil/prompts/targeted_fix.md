# HighNoon Targeted Fix Prompt

**How To Use:** Use for one bounded repair item (not full-phase execution).

Read first:
- [Shared Prompt Foundation](shared_prompt_foundation.md)
- [Agent Audit Analysis Workflow](../workflows/AGENT_AUDIT_ANALYSIS_WORKFLOW.md)
- `AUDIT_ANALYSIS_ROADMAP.md`

---

## Session Header

```text
MODE: Targeted Fix
TARGET_REPAIR_ITEM: <roadmap repair id/title>
PHASE_CONTEXT: <0-7>
BUNDLE: artifacts/audit_visuals/<LATEST or explicit timestamp>
```

---

## Role

Act as HighNoon engineer for one repair item with strict traceability.

---

## Targeted Fix Contract

1. Locate the exact repair item in roadmap.
2. Confirm phase context and dependencies.
3. Capture pre-change metrics relevant to this issue.
4. Implement minimal complete fix (no band-aids).
5. Add/update tests for the repair path.
6. Run verification commands.
7. Record post-change metrics and delta.
8. Update roadmap repair item and phase addendum.

---

## Required Output Structure

1. `Repair Scope`
2. `Root Cause Evidence`
3. `Code/Test Changes`
4. `Metrics Before/After`
5. `Residual Risks`
6. `Roadmap Updates`

---

## Constraints

- Stay within one repair item or one tightly-coupled pair.
- No broad refactors unrelated to the root cause.
- No phase migration in this session.
- Keep DAL severity and traceability explicit.

---

## Optional Subagent Split

- Subagent A: implement fix
- Subagent B: independent verification

---

## Fix Target

Describe the exact repair item to execute.

