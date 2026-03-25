Use this as a reusable suffix when you want Codex to fully implement the roadmap it produces, not just describe it:

```text
Roadmap Execution Mode: Do not stop at analysis, design, or planning. If you generate a roadmap, you are expected to execute it end-to-end inside this repo unless a hard blocker makes that impossible. The goal is complete implementation, tight wiring, verification, and working outcomes, not partial scaffolding.

Mission:
- Analyze the repo, identify the highest-leverage roadmap, and then implement that roadmap completely.
- Carry changes through discovery, design, code changes, wiring, cleanup, verification, tests, and final integration.
- Do not leave behind TODO-shaped architecture, empty abstractions, placeholder code, or dead-end partials.
- Prefer clean end-to-end execution over sprawling unfinished ambition.

Core engineering stance:
- Operate at very high engineering standards: clean architecture, strong performance, minimal waste, and coherent integration.
- Favor durable primitives over shallow wrappers.
- Prefer native implementation where performance, data movement, indexing, orchestration, or compute-heavy logic matter.
- Python should mainly be orchestration, bindings, thin wrappers, configuration surfaces, and user-facing control flow.

Native implementation rules:
- Prefer C++ for core execution paths, native ops, indexing logic, compute kernels, hot-path transforms, and reusable low-level primitives.
- Assume AVX2 is the default CPU optimization target where SIMD is relevant.
- Do not use OpenMP broadly.
- Use OpenMP only when:
  - the work is clearly a hotspot compute path
  - the parallelism is justified by the workload
  - the implementation remains maintainable
  - there is a clear reason not to rely on plain vectorization or better algorithmic structure first
- If you use OpenMP, confine it to hotspot kernels and explain why.
- Do not push substantial logic into Python if it belongs in native code.

Mandatory workflow:
- Run everything from the repo venv:
  - `source venv/bin/activate`
- Use Saguaro-first exploration and validation where possible:
  - `saguaro health`
  - `saguaro query "<concept>" --k 5`
  - `saguaro agent skeleton <file>`
  - `saguaro agent slice <symbol> --depth 2`
  - `saguaro impact --path <file>`
  - `saguaro refactor plan --symbol "<symbol>"`
  - `saguaro verify . --engines native,ruff,semantic --format json`
  - `saguaro chronicle snapshot`
  - `saguaro chronicle diff`
- Use repo-native build/test/benchmark flows as needed.
- Before completion, verify the final state, not just individual edits.
- Treat the roadmap file provided in the prompt as the authoritative implementation contract.
- Resolve and keep track of the roadmap file path as `<roadmap_path>`.
- Before you are allowed to claim the roadmap is fully implemented, you must run:
  - `./venv/bin/saguaro roadmap validate --path <roadmap_path> --format json`
  - `./venv/bin/saguaro roadmap graph --path <roadmap_path> --format json`
- Treat those two commands as hard completion gates, not optional reporting.
- If `validate` or `graph` reports missing work, partial implementation, broken dependencies, incomplete wiring, or roadmap gaps, the task is not done. Continue implementation until those gaps are closed end-to-end, then rerun both commands.

Execution requirements:
- Do the investigation needed to choose the roadmap.
- Use the roadmap file supplied in the prompt as the roadmap contract whenever one is provided.
- Convert the roadmap into implementation phases.
- Execute those phases instead of stopping after the writeup.
- Update all necessary call sites, wrappers, prompts, hooks, configs, tests, docs, and verification paths so the feature is actually integrated.
- Remove dead code and obsolete paths that your implementation supersedes when safe to do so.
- Fix incorrect wiring, stale assumptions, and partial integrations you encounter along the path if they materially block or weaken the result.
- If a roadmap item turns out to be wrong, adapt and continue rather than freezing at the plan.
- Any partials, missing edges, broken wiring, unimplemented dependencies, or validation gaps surfaced by Saguaro roadmap tooling must be fully implemented before completion.

Implementation priorities:
- Correctness first
- Architectural coherence second
- Performance third, but take it seriously
- Simplicity fourth, but do not oversimplify away important capability
- Polish after the system works end-to-end

What good looks like:
- The roadmap is reflected in real code, not just markdown.
- Wiring is complete across native code, Python wrappers, orchestration, prompts, hooks, verification, entrypoints, and tests where relevant.
- New capabilities are reachable through the actual system, not stranded in isolated files.
- Replaced paths are cleaned up.
- Verification passes or failures are explained precisely.
- Performance-sensitive logic lives in the right layer.
- `./venv/bin/saguaro roadmap validate --path <roadmap_path> --format json` passes for the roadmap file supplied in the prompt.
- `./venv/bin/saguaro roadmap graph --path <roadmap_path> --format json` reflects a complete implementation with no meaningful missing nodes or dangling gaps left unresolved.

Decision rules:
- Prefer modifying existing strong architectural seams over creating new disconnected subsystems.
- Prefer consolidation over duplication.
- Prefer deeper primitives over repeated special cases.
- Prefer native shared utilities over Python copies of performance-sensitive logic.
- Avoid speculative abstraction unless the implementation clearly needs it.
- Avoid adding configuration surface area unless it unlocks meaningful control.

Deliverable behavior:
1. Investigation
   - Briefly summarize the relevant repo surfaces inspected
   - State the roadmap you selected and why

2. Execution plan
   - Break the work into implementation phases
   - Then actually execute them

3. Implementation
   - Make the code changes
   - Wire everything through properly
   - Clean up superseded paths

4. Verification
   - Run the relevant validation, tests, and Saguaro verification
   - Run `./venv/bin/saguaro roadmap validate --path <roadmap_path> --format json`
   - Run `./venv/bin/saguaro roadmap graph --path <roadmap_path> --format json`
   - If either command exposes gaps, go back to implementation and close them before finishing
   - Include build or benchmark checks when performance-sensitive code changed

5. Final report
   - Summarize what was implemented
   - Name the exact files and systems affected
   - State the roadmap path used for validation
   - Report the outcome of `roadmap validate` and `roadmap graph`
   - State what remains, if anything, and why

Constraints:
- Do not stop at “here is the roadmap” unless I explicitly asked for planning only.
- Do not leave major roadmap items unimplemented if they are feasible in the current task.
- Do not default to Python for heavy logic that belongs in C++.
- Do not spray OpenMP across the codebase.
- Do not keep obsolete paths alive just to avoid cleanup.
- Do not use the markdown roadmap files in this repo as authority unless I explicitly tell you to implement one of them.
- If I provide a roadmap file in the prompt, do not claim the work is fully implemented until both roadmap commands have been run against that roadmap path and any reported gaps have been closed.

Style:
- Be direct and execution-oriented.
- Favor action over narration.
- Explain key tradeoffs briefly and concretely.
- When you choose native versus Python, say why.
- When you use AVX2 or OpenMP, state where and why.
```

If you want a shorter version, use this:

```text
Do not stop at planning. Analyze the repo and implement the roadmap end-to-end. If I provide a roadmap file in the prompt, treat it as the implementation contract and use it for completion gating. Use Saguaro from `venv` for discovery, impact analysis, refactor planning, and verification. Prefer C++ for core logic, hot paths, indexing, and reusable low-level primitives; use AVX2 where SIMD matters; keep Python mainly as thin wrappers and orchestration. Use OpenMP only in justified hotspot compute kernels, not broadly. Fully wire the result through prompts, hooks, planners, native boundaries, configs, entrypoints, and tests. Before claiming completion, run `./venv/bin/saguaro roadmap validate --path <roadmap_path> --format json` and `./venv/bin/saguaro roadmap graph --path <roadmap_path> --format json`; if they show gaps or partials, continue implementing until those are resolved end-to-end.
```
