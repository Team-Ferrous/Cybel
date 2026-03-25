# Saguaro System Investigation and Planning Prompt

You are the principal systems engineer for the Anvil codebase. Your task is to perform a full technical investigation of the Saguaro subsystem and produce an executable implementation plan.

## Context
- Repository: Anvil
- Date context: Assume current date is March 2, 2026
- Objective: Analyze Saguaro reliability, performance, integration boundaries, and governance readiness, then produce a prioritized implementation roadmap with acceptance criteria.
- Constraint: Do NOT assume Saguaro health from prior runs; verify all critical claims with direct evidence.

## Required Investigation Workflow
1. Establish baseline and environment
- Identify runtime paths for `saguaro` (system/global) and `./venv/bin/saguaro` (project venv).
- Capture versions, backend/runtime dependencies, and startup differences.
- Confirm whether strict mode and TensorFlow/plugin loading are stable.

2. Health and index integrity
- Run:
  - `saguaro health`
  - `saguaro index --path .` (only if required)
  - `saguaro verify . --engines native,ruff,semantic`
- Identify recurrent failures (ABI mismatch, plugin conflicts, index corruption, memory pressure).
- Quantify failure reproducibility and trigger conditions.

3. Feature-capability mapping
- Inventory implemented vs expected capabilities from AGENTS/GEMINI instructions:
  - query/skeleton/slice/impact/verify
  - chronicle, legislation, workset, refactor, deadcode, entrypoints, build-graph, audit
  - SSAI patch/verify/impact/commit flow
- Mark each as: Available, Partial, Stub, or Broken.

4. Integration audit with Anvil agent loop
- Trace where Saguaro is invoked in:
  - tool registry
  - REPL commands
  - mission/orchestrator flows
  - verification gates
- Verify graceful fallback behavior if Saguaro fails.
- Identify silent-failure paths and missing user-visible diagnostics.

5. Performance and determinism profile
- Measure cold-start and warm-start latencies for key commands.
- Compare venv vs system command behavior.
- Estimate CPU/RAM footprint and peak resource hotspots.
- Document nondeterministic outputs or excessive log noise impacting automation.

6. Security and governance assessment
- Evaluate whether Saguaro invocation paths satisfy policy controls in regulated mode.
- Check auditability of:
  - command invocations
  - verify outputs
  - failure traces
  - mutation operations (patch/commit)
- Identify compliance gaps relative to NASA-STD-8739.8B / DO-178C style evidence expectations.

## Deliverables (required)
Produce all outputs in markdown with clear headings:

1. `SAGUARO_FINDINGS.md`
- Executive summary
- Root-cause findings (ordered by severity)
- Evidence (commands, outputs, file references)
- Clear statement of what is currently safe/unsafe for production use

2. `SAGUARO_CAPABILITY_MATRIX.md`
- Table: Capability | Expected | Actual | Gap | Recommended Fix | Owner | Effort

3. `SAGUARO_PLAN.md`
- 30/60/90 day plan or Wave 1/2/3 plan
- Each task must include:
  - objective
  - implementation steps
  - risk
  - verification method
  - exit criteria

4. `SAGUARO_RUNBOOK.md`
- Standard operating procedures for:
  - startup checks
  - index rebuild
  - verification
  - common failure remediation
  - CI integration

## Quality Bar
- No vague statements. Every claim must include evidence.
- Use absolute dates and explicit versions.
- Separate confirmed facts from inferences.
- Include a “Known Unknowns” section with targeted follow-up experiments.

## Final Output Format
Return:
1. One-paragraph executive readiness verdict.
2. Top 10 prioritized actions.
3. Links/paths to all generated deliverables.
