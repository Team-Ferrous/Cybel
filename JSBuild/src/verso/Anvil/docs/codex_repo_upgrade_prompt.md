Use this as a reusable suffix at the end of any prompt:

```text
Repository Upgrade Mode: I want you to act like a repo modernization and code intelligence architect. Your job is to find weak areas in this codebase, prove they are weak using repo evidence, and turn that into a technically grounded roadmap for cleanup, hardening, and capability upgrades.

Mission:
- Find dead code, low-value code, weakly wired systems, duplicate logic, stale abstractions, shallow integrations, risky boundaries, brittle flows, poor observability, and underused Saguaro/Anvil capabilities.
- Build a roadmap that improves architectural coherence, removes waste, strengthens wiring, and expands high-leverage capabilities.
- Do not produce generic cleanup advice. Everything must be grounded in this repo.

Mandatory repo exploration workflow:
- Run everything from the repo venv:
  - `source venv/bin/activate`
- Use Saguaro-first exploration wherever possible.
- Prefer these tools before fallback inspection:
  - `saguaro health`
  - `saguaro coverage`
  - `saguaro deadcode`
  - `saguaro unwired --format json`
  - `saguaro impact --path <file>`
  - `saguaro entrypoints`
  - `saguaro build-graph`
  - `saguaro query "<concept>" --k 5`
  - `saguaro agent skeleton <file>`
  - `saguaro agent slice <symbol> --depth 2`
  - `saguaro refactor plan --symbol "<symbol>"`
  - `saguaro audit`
  - `saguaro verify . --engines native,ruff,semantic --format json`
  - `saguaro chronicle diff` when useful for understanding recent drift
- If a Saguaro command fails, briefly document the error, check health/index state, and then use minimal fallback inspection.

What to look for:
- Dead code candidates
- Unwired or partially wired feature islands
- Duplicate logic and DRY violations
- Parallel implementations that should be unified
- Code paths with no meaningful entrypoint or downstream use
- Modules with high conceptual overlap but fragmented ownership
- Places where Saguaro capabilities exist but are not integrated into Anvil agent loops, hooks, planners, governance, memory, or telemetry
- Incorrect or incomplete prompt wiring
- Native/Python boundary drift
- Missing verification coverage
- Weak runtime observability or missing feedback loops
- Systems that look architecturally important but operationally shallow
- Places where roadmap intent appears only partially realized in code
- Risky areas with poor tests, poor discoverability, or high impact radius

How to think:
- Think like a principal engineer doing technical due diligence on a large internal platform.
- Prefer structural weaknesses over stylistic complaints.
- Distinguish between:
  - dead and removable
  - dead and needs wired in
  - alive but underwired
  - alive but redundant
  - alive but strategically underpowered
- Identify not just what is wrong, but what existing primitives suggest the intended architecture already wants to be.
- Use the repo's actual seams to propose upgrades: hooks, prompts, planners, memory, orchestration, telemetry, native ops, verification engines, worksets, audit, and entrypoints.

Deliverable:
1. Repo grounding
   - Summarize the Saguaro analyses and code surfaces inspected
   - Name the most important subsystems, seams, and signals you used

2. Weakness map
   - Identify the main weak areas of the repo
   - Group them by category such as dead code, unwired systems, duplication, wiring gaps, governance gaps, observability gaps, architectural drift, native boundary risk, or cleanup debt
   - For each weakness include concrete evidence from commands, files, symbols, or code paths

3. Opportunity classification
   - For each weak area, classify it as:
     - Remove
     - Consolidate
     - Rewire
     - Harden
     - Extend
   - Explain why

4. Technical roadmap
   - Produce a phased roadmap of the highest-leverage improvements
   - Each roadmap item must include:
     - Title
     - Problem
     - Evidence
     - Exact wiring points in the codebase
     - Existing primitives to reuse
     - New primitive or refactor needed
     - Expected payoff
     - Main risk
     - Smallest credible implementation step

5. Priority ranking
   - Rank the roadmap by leverage and feasibility
   - Call out:
     - quickest cleanup wins
     - highest-risk weak points
     - highest strategic upside
     - best unification opportunities

6. Closing synthesis
   - State what the repo appears to be trying to become
   - Explain which upgrades best align the current code with that trajectory

Style requirements:
- Be direct, technical, and evidence-based.
- Do not give vague "improve code quality" advice.
- Name actual files, modules, classes, functions, hooks, loops, planners, or commands.
- Prefer repo-grounded diagnosis over speculation.
- When you speculate, label it clearly as inference.
- Treat this as architecture analysis, not a product brainstorm.

Important constraints:
- Do not use the markdown roadmap files in this repo as primary input unless I explicitly ask.
- Do not just list issues; synthesize them into an upgrade path.
- Do not stop at identifying problems. Show how they can be wired into the current architecture cleanly.
- If helpful, ask me 3-5 short questions first about cleanup appetite, refactor tolerance, and whether I prefer aggressive consolidation or incremental hardening.
```

If you want a shorter version, use this:

```text
Act as a repo modernization architect. Use Saguaro from `venv` to analyze this codebase for dead code, unwired systems, duplicate logic, weak integrations, incorrect wiring, underused capabilities, governance gaps, and architectural drift. Use commands like `saguaro deadcode`, `saguaro unwired`, `saguaro coverage`, `saguaro audit`, `saguaro entrypoints`, `saguaro build-graph`, `saguaro impact`, `saguaro verify`, `saguaro query`, `saguaro agent skeleton`, and `saguaro agent slice`. Then produce an evidence-based technical roadmap that classifies weak areas as Remove, Consolidate, Rewire, Harden, or Extend, and for each roadmap item include exact wiring points, reused primitives, missing primitives, payoff, risk, and smallest first step. Do not use the markdown roadmap files unless explicitly asked.
```
