# HighNoon Audit Analysis Prompt (Single Phase)

**How To Use:** Start a new chat for one phase only. This prompt must diagnose failures and construct implementation-ready fixes for that phase.

Read first:
- [Shared Prompt Foundation](shared_prompt_foundation.md)
- [Agent Audit Analysis Workflow](../workflows/AGENT_AUDIT_ANALYSIS_WORKFLOW.md)

---

## Session Header (Fill Before Running)

```text
PHASE: <0-7>
PHASE_NAME: <from workflow>
BUNDLE: artifacts/audit_visuals/<LATEST or explicit timestamp>
MODE: Analysis + Fix Construction (no broad implementation in this step)
ROADMAP_TARGET: AUDIT_ANALYSIS_ROADMAP.md
```

---

## Role

Act as HighNoon Lead Audit Analyst for the selected phase. Keep scope bounded to that phase.

---

## Mission (Non-Negotiable)

The output is not complete if it only describes problems.
You must construct a concrete, testable repair plan for every failing P0/P1 issue in phase scope.

Specifically:
- identify what is failing
- prove root cause with evidence
- construct exact fix actions (file-level)
- define required tests and gate criteria
- write these into the roadmap repair map

---

## Execution Contract

1. Validate artifact preflight for this phase.
2. Analyze only required artifacts for this phase.
3. Produce evidence-backed findings with severity and DAL.
4. Trace each issue to artifact:key and/or code file:line.
5. Construct implementation-ready fix items for each P0/P1 issue.
6. Append phase findings to `AUDIT_ANALYSIS_ROADMAP.md`.
7. Update Gate Dashboard row for this phase.
8. Append repair entries to roadmap repair map with tests and gate impact.
9. Append verification commands for detected issues.
10. Stop and hand off to implementation.

---

## Forbidden During Analysis

- No lazy fixes or vague “investigate later” placeholders.
- No synthetic bridges or passthrough fixes.
- No fallback-path proposals.
- No clipping/clamping workaround proposals.
- No speculative root cause without explicit evidence.
- No cross-phase analysis in the same session.

---

## Required Output Structure

Use this exact structure:

1. `Phase Status`: PASS / FAIL / PARTIAL / BLOCKED
2. `Artifact Contract`: present/missing table
3. `Evidence Summary`: key metrics and anomalies
4. `Issue Register`: numbered issues with P0-P3 and DAL
5. `Root Cause`: file:line or artifact:key evidence chain
6. `Fix Construction Table`:
   - Issue ID
   - Target file(s)
   - Required code change intent (what must change)
   - Forbidden approaches for this issue
   - Required tests
   - Gate criteria to close
7. `Roadmap Repair Entries Added`: exact items appended
8. `Verification Commands`: reproducible commands
9. `Gate Decision`: why pass/fail/partial/blocked

---

## Roadmap Construction Contract (Mandatory)

For each P0/P1 issue in scope, append a roadmap repair entry that includes:
- failure statement
- root cause evidence
- code surface (`file:line`)
- fix intent (implementation-ready)
- tests required
- acceptance gate
- owner placeholder and phase linkage

If any P0/P1 issue lacks a repair entry, phase output is incomplete.

---

## Architecture Coverage Reminder

Ensure selected phase covers all architecture surfaces mapped in workflow section 11.
Missing surfaces in scope must be called out as phase failure.
For Phases 3/4/6, include explicit wiring verdicts for QHPM and the critical feature-family checklist.

---

## Subagent Contract (Optional But Recommended)

For P0/P1 findings:
- Agent A: primary analysis
- Agent B: independent verification
- Agent C: contradiction search (P0 only)

Every subagent must report:
- scope
- evidence references
- issue list
- fix-construction items
- confidence and uncertainty

---

## Handoff Format

End with:

```text
PHASE_COMPLETE: <phase>
STATUS: <PASS/FAIL/PARTIAL/BLOCKED>
IMPLEMENTATION_READY: <YES/NO>
NEXT_RECOMMENDED_PHASE: <phase>
BLOCKERS: <none or list>
```

---

## Phase Assignment (Canonical 8-Phase Model)

Execute exactly one selected phase from:
- Phase 0: Preflight and Baseline
- Phase 1: Runtime Integrity
- Phase 2: Mathematical Integrity
- Phase 3: Learning Dynamics and Control
- Phase 4: Architecture and Routing Integrity
- Phase 5: Performance and HPC Saturation
- Phase 6: Config, Feature, and Observability Coverage
- Phase 7: Red Team Final Validation and Sign-Off

---

## Operator Phase Selector (Edit This Block)

```text
RUN_PHASE=7
RUN_PHASE_NAME=Red Team Final Validation and Sign-Off
RUN_BUNDLE=dir=artifacts/audit_visuals/2026-03-01-114747
TERMINAL_LOGS=full_logs.txt
RUN_MODE=analysis_fix_construction
```

You are updating the AUDIT_ANALYSIS_ROADMAP.md file with the findings from the audit.

Notes to use when investigating the proper phase that relates:
- Loss does not change at all over 31 steps, leading me to believe the QULS needs serious work. 
- Performance is not great either, we need to investigate the tokens/s throughput and time per step to completion. 
- Fused ops appear to be skipped, they should be utilized and not skipped. 
- Open to refactors and upgrades to our audit folder and setup as well, some things don't really work that well and I'd like to enhance that as well. 