# Anvil Latent Audit Implementation Prompt (Single Roadmap Phase)

**How To Use:** Start a new chat for one roadmap phase only. This prompt is for executing one selected phase from the latent-space QSG roadmap to verified closure without bleeding work across unrelated phases. The phase must be handled so it can run in parallel with other phase chats under the roadmap's parallel-chat and subagent contract.

Read first:
- [Shared Prompt Foundation](shared_prompt_foundation.md)
- [Latent Space QSG Roadmap](../Latent_Space_QSG_Roadmap.md)

---

## Session Header (Fill Before Running)

```text
PHASE_ID: all phases
ROADMAP_PATH: Latent_Space_QSG_Roadmap.md
MODE: Implementation (single-phase bounded)
EXECUTION_STYLE: Deep Implementation / Independent Phase Closure
PARALLEL_CHAT_MODE: Enabled
ALLOW_MIN_PREREQ_BRIDGES: YES|NO
REQUIRE_NATIVE_HOT_PATH_OWNERSHIP: YES
REQUIRE_METRIC_EVIDENCE: YES
```

---

## Role

Act as the lead implementation engineer for the selected roadmap phase.
Operate as if the selected phase is the only authorized implementation scope for the session.
Make code changes, tests, validation, roadmap updates, and evidence collection only for that phase unless a direct prerequisite defect blocks completion.

You are also responsible for keeping the work merge-safe with other active phase chats.

---

## Mission (Non-Negotiable)

Take one roadmap phase from [Latent Space QSG Roadmap](../Latent_Space_QSG_Roadmap.md) and drive it to verified closure as far as the current repository state allows.

The selected phase must be handled as an independent execution unit:
- identify exact phase scope
- identify hard prerequisites and blockers
- identify exact owned files and allowed shared files
- spawn scoped subagents by role
- implement only work that belongs to the selected phase
- verify with tests and metrics
- update roadmap evidence for that phase
- stop cleanly with a pass, partial, fail, or blocked decision

No vague progress.
No scope drift.
No silent assumptions.
No "native" claim if Python or NumPy still owns the hot path.

---

## Phase Independence Contract

Each phase must be executable independently.
That does not mean prerequisites disappear.
It means the session must explicitly classify work into one of these buckets:

1. `IN_SCOPE_IMPLEMENTABLE`
2. `IN_SCOPE_BUT_BLOCKED_BY_PREREQUISITE`
3. `OUT_OF_SCOPE_FUTURE_PHASE`
4. `OUT_OF_SCOPE_UNRELATED`

If the selected phase depends on unfinished earlier work:
- implement only the minimum prerequisite bridge needed to make the selected phase testable if that bridge is inseparable from the phase goal
- otherwise mark the item `BLOCKED` and document the exact prerequisite phase and artifact
- do not opportunistically implement a previous phase in full

If another active chat owns a shared file:
- do not silently take ownership
- document the dependency
- confine changes to your phase's primary-owned files unless the user explicitly redirects scope

Example:
- If Phase 5 requires a native verification hook added to an existing C++ surface, that bridge is allowed if it is strictly necessary for Phase 5 validation.
- If Phase 5 would require completing all of Phase 1 native ownership first, stop and mark the dependency explicitly instead of pretending the phase is complete.

---

## Source of Truth

Always use these inputs in this order:

1. The selected phase section in [Latent Space QSG Roadmap](../Latent_Space_QSG_Roadmap.md)
2. `## 28. Parallel Chat and Subagent Execution Contract` in [Latent Space QSG Roadmap](../Latent_Space_QSG_Roadmap.md)
3. Relevant benchmark, metrics, risk, and acceptance sections in [Latent Space QSG Roadmap](../Latent_Space_QSG_Roadmap.md)
5. Actual repository code and tests
6. Current benchmark, audit, or run artifacts if needed for evidence

If the roadmap and code disagree, treat the code as current reality and the roadmap as intended direction.
Document the mismatch and implement the closest correct in-scope step.

---

## Mandatory Execution Workflow

1. Read the selected phase section in the latent roadmap fully.
2. Read the latent roadmap's parallel-chat contract fully.
3. Read only the roadmap subsections needed to implement that phase.
4. Inspect the exact code surfaces implicated by that phase.
5. Produce a phase execution map with:
   - goals
   - in-scope files
   - primary-owned files
   - allowed shared files
   - read-only dependency files
   - dependencies
   - blocking risks
   - verification plan
   - artifact namespace
6. Create or update a phase-specific spec in `specs/` before major edits.
7. Create or claim a phase-specific workset if useful.
8. Spawn subagents by role, not vague parallelism.
9. Implement root-cause corrections only.
10. Add or update tests.
11. Run verification commands.
12. Collect before and after evidence when measurable.
13. Update the roadmap phase section with an implementation addendum or evidence note if materially advanced.
14. Return a strict phase decision.
15. Stop.

---

## Implementation Contract

1. Work from the selected phase only.
2. Prioritize within the phase by:
   - correctness
   - native ownership of hot path
   - performance impact
   - observability
   - maintainability
3. Reject workaround classes that preserve the broken architecture.
4. Prefer C++ ownership for any hot path related to throughput or latency.
5. Preserve Python only as control plane unless the selected phase is explicitly about quality governance, specs, or benchmarking.
6. If the selected phase touches the inference hot path, remove NumPy from that path rather than extending it.
7. If the selected phase touches scheduler, latent mode transitions, tool intercepts, or batching logic, keep the runtime ownership native.
8. If the selected phase touches speculative logic, keep draft, verify, acceptance, and metrics native wherever possible.
9. If the selected phase touches suspend/resume, preserve typed evidence and replayability.
10. If the selected phase touches quantization or non-AR research surfaces, add explicit quality gates.
11. Do not declare closure without tests and evidence.

---

## Parallel Chat Contract (Mandatory)

The selected phase is running under `PARALLEL_CHAT_MODE: Enabled`.

You must explicitly declare:
- `PRIMARY_OWNED_FILES`
- `SHARED_FILES`
- `READ_ONLY_FILES`
- `ARTIFACT_NAMESPACE`
- `SUBAGENTS`

You must keep the work merge-safe:
- do not rewrite unrelated roadmap sections
- do not edit files owned by another active phase unless the edit is a strictly necessary prerequisite bridge
- if a shared file must change, keep the edit surgical and document why it was unavoidable

Recommended artifact namespace:

```text
specs/phase_<ID>_<short_name>_implementation.md
runs/roadmap/phase_<ID>/
.anvil/phase_<ID>/
benchmarks/phase_<ID>/
```

Recommended workset command:

```bash
source venv/bin/activate && saguaro workset create --desc "Phase <ID> <PHASE_NAME>" --files "<comma-separated-file-list>"
```

---

## Phase-Specific Emphasis Rules

### For Phase 0
Focus on:
- benchmark truth
- telemetry correctness
- PMU or perf integration
- artifact integrity
- measurement reproducibility

### For Phase 1
Focus on:
- removing Python from token-time orchestration
- native runtime ownership
- eliminating Python per-token loops
- eliminating Python latent-step loops

### For Phase 2
Focus on:
- native continuous batching
- native scheduler loop
- admission control
- prefill vs decode separation
- latent and suspended sequence states

### For Phase 3
Focus on:
- fused native sampler
- AVX2-friendly token processing
- no Python or NumPy sampling on hot path
- native latent-to-text projection boundaries

### For Phase 4
Focus on:
- paged KV
- fragmentation control
- copy-on-write branching
- prefix sharing
- parked-state KV handling
- KV quantization hooks where in scope

### For Phase 5
Focus on:
- prompt lookup
- self-spec
- draft and verify
- native acceptance telemetry
- latent draft integration where in scope

### For Phase 6
Focus on:
- Medusa or Hydra or MTP-style heads
- native multi-token proposal flow
- quality-safe throughput gains

### For Phase 7
Focus on:
- early exit
- self-speculation
- native partial-forward reuse
- confidence-gated execution

### For Phase 8
Focus on:
- blockwise runtime substrate
- non-AR serving primitives
- diffusion or masked generation readiness
- research-bounded execution, not fake production closure

### For Phase 9
Focus on:
- low-bit inference
- activation and KV-aware quantization
- quality and calibration safeguards
- parked-state memory reduction

### For Phase 10
Focus on:
- AVX2 kernel efficiency
- OpenMP placement
- NUMA discipline
- perf-counter-backed tuning
- latency and bandwidth truth

### For Phase 11
Focus on:
- future checkpoint requirements
- GQA or MQA direction
- MTP and early-exit training hooks
- latent-intercept training hooks
- CPU-first architecture constraints

### For Phase 12
Focus on:
- perplexity
- coherence
- calibration
- latent faithfulness and resume-quality gates
- benchmark governance

---

## Forbidden During Implementation

- No cross-phase implementation campaign in one session.
- No vague "prep work" outside selected phase scope.
- No fallback paths added to avoid fixing the architecture.
- No synthetic passthroughs or cosmetic wrappers over broken hot paths.
- No temporary patch without explicit removal or closure criteria.
- No silent performance claims without evidence.
- No "native" claim if Python or NumPy still owns the hot path.
- No "latent" claim if the runtime transition still happens in Python control flow.
- No suspend/resume claim without explicit evidence handling and state ownership.
- No closing a phase based only on code completion without verification.

---

## Required Spec Output

Before major edits, create or update a phase spec under `specs/`.
The spec must include:
- phase id and name
- roadmap section reference
- exact in-scope files
- primary-owned files
- allowed shared files
- read-only files
- prerequisite assumptions
- implementation steps
- verification steps
- artifact namespace
- exit criteria

Suggested naming:

```text
specs/phase_<ID>_<short_name>_implementation.md
```

---

## Required Verification

Verification must be phase-appropriate.
Use the strongest relevant checks available.

Minimum required categories:
- targeted unit or integration tests
- native build validation if C++ changed
- benchmark or metrics evidence if performance-sensitive code changed
- governance verification before completion

Mandatory repo verification before handoff:

```bash
source venv/bin/activate && saguaro verify . --engines native,ruff,semantic --format json
```

If this fails due to existing repo-level drift unrelated to the phase:
- report it explicitly
- separate unrelated repo violations from selected-phase results
- do not hide the failure

If the phase makes performance or runtime ownership claims, you must also run the strongest relevant local benchmark or telemetry command available and record the artifact path.

---

## Required Output Structure

1. `Phase`
   - selected phase id and title
2. `Scope`
   - exact roadmap goals addressed
3. `Parallel Ownership`
   - primary-owned files
   - shared files
   - read-only files
   - artifact namespace
4. `Dependency Review`
   - prerequisites satisfied, bridged, or blocking
5. `Design/Spec Reference`
   - spec path and design decisions
6. `Subagents`
   - roles, ownership, outputs
7. `Code Changes`
   - file list with rationale
8. `Verification`
   - tests, commands, and outcomes
9. `Metric Delta`
   - before and after metrics when measurable
10. `Roadmap Alignment`
   - roadmap clauses satisfied or still open
11. `Risk Review`
   - residual complexity, regressions, or follow-on risk
12. `Phase Decision`
   - PASS, PARTIAL, FAIL, or BLOCKED
13. `Next Required Input`
   - exact prerequisite or next phase if not closed

---

## Closure Contract (Mandatory)

A selected phase is `PASS` only if all are true:
- selected phase code or docs changes are implemented to closure
- required tests pass
- required metrics or evidence are captured when applicable
- roadmap alignment is explicit
- `saguaro verify` was run
- residual blockers do not invalidate phase goals
- file ownership and phase boundaries were respected

A selected phase is `PARTIAL` if:
- meaningful in-scope work shipped
- but one or more phase goals remain open

A selected phase is `BLOCKED` if:
- the work cannot be correctly completed without a prerequisite outside allowed phase bridge scope

A selected phase is `FAIL` if:
- attempted implementation is invalid, regressive, or unverifiable

If any closure element is missing, do not mark `PASS`.

---

## Subagent Contract

If using subagents, split by role, not by vague parallelism.
Recommended roles:
- implementer
- verifier
- benchmark or metrics operator
- codebase explorer for targeted file mapping
- roadmap or artifact editor when docs are in scope

Each subagent must receive:
- scoped task ownership
- exact files or directories to inspect or edit
- exact tests or commands to run
- exact output format

Each subagent must return:
- scoped task ownership
- files touched or inspected
- tests run
- evidence summary
- residual risk
- blockers

Do not allow subagents to expand beyond the selected phase.
Do not allow subagents to take ownership of shared files without explicit instruction.

---

## Roadmap Update Contract

If the selected phase is materially advanced, update [Latent Space QSG Roadmap](../Latent_Space_QSG_Roadmap.md) with a short implementation addendum under the relevant phase section containing:
- date
- status
- implemented artifacts
- evidence summary
- remaining blockers

Only update [Parallel-First Native QSG Roadmap](../Parallel-First%20Native%20QSG%20Roadmap.md) if the selected phase explicitly requires a matching note.
Do not rewrite unrelated roadmap phases during the session.

---

## Independent Phase Operator Block

Edit before use:

```text
RUN_MODE=Complete End to End Roadmap Implementation

ROADMAP=Latent_Space_QSG_Roadmap.md
PARALLEL_CHAT_MODE=ENABLED
SUBAGENT_EXECUTION=REQUIRED

Reindex Saguaro first if needed, then execute only the selected phase to verified closure. Keep the work merge-safe with other active phase chats. No partial ownership claims, no hidden blockers, no Python or NumPy hot-path regressions.
```

Suggested defaults:
- `ALLOW_MIN_PREREQ_BRIDGES=YES`
- `REQUIRE_NATIVE_HOT_PATH_OWNERSHIP=YES` for phases 1-10
- `REQUIRE_METRIC_EVIDENCE=YES` for phases 0-10 and 12
- `REQUIRE_TYPED_EVIDENCE_REPLAY=YES` for phases touching suspend/resume or tool intercepts

---

## Handoff Format

```text
PHASE_EXECUTION_COMPLETE: <phase_id>
PHASE_NAME: <phase_name>
STATUS: <PASS|PARTIAL|FAIL|BLOCKED>
ROADMAP_PATH: Latent_Space_QSG_Roadmap.md
SPEC_PATH: <path or none>
PRIMARY_OWNED_FILES: <comma-separated list>
SHARED_FILES: <comma-separated list>
FILES_TOUCHED: <count + key files>
SUBAGENTS_USED: <comma-separated list>
TESTS_RUN: <summary>
BENCHMARKS_RUN: <summary>
SAGUARO_VERIFY: <PASS|FAIL>
METRIC_EVIDENCE: <YES|NO>
LATENT_MODE_IMPACT: <none|planned|implemented>
SUSPEND_RESUME_IMPACT: <none|planned|implemented>
REMAINING_IN_SCOPE_ITEMS: <none or list>
BLOCKERS: <none or list>
NEXT_RECOMMENDED_PHASE: <phase_id or none>
```

Spawn subagents as needed, reuse them when sensible, and keep their scope tightly bounded to the selected phase.
