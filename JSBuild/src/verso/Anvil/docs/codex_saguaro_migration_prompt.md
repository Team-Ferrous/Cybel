Use this as a reusable prompt template or suffix when you want Codex to analyze an imported local repository and turn it into a Saguaro-native migration roadmap.

Set this placeholder before use:

```text
TARGET_REPO_PATH="<PASTE_IMPORTED_REPO_DIRECTORY_PATH_HERE>"
```

```text
Saguaro Native Migration Mode: Analyze the imported repository at `TARGET_REPO_PATH` as a capability mine, then produce a technically grounded migration roadmap for what Saguaro should build natively to absorb the best features, mechanisms, workflows, and operator advantages from that repo.

Mission:
- Treat the imported repo as a source of transferable primitives, not as something to clone blindly.
- Identify which features, execution models, analysis surfaces, data models, UX loops, and verification behaviors would materially strengthen Saguaro if implemented natively.
- Focus on features that improve:
  - code intelligence depth
  - migration leverage
  - operator usability
  - architectural coherence
  - native performance
  - verification quality
  - agent workflow power
  - practical adoption
- Produce a migration technical roadmap, not a generic comparison or feature wish list.

Primary question:
- If `TARGET_REPO_PATH` contains strong ideas, what should Saguaro absorb, what should it ignore, and what should it reinvent as a stronger native primitive?

Mandatory analysis workflow:
- Run from the Anvil repo virtualenv:
  - `source venv/bin/activate`
- Use Saguaro-first inspection on both Anvil/Saguaro and the imported repo wherever possible:
  - `saguaro health`
  - `saguaro query "<concept from imported repo>" --k 5`
  - `saguaro agent skeleton <file>`
  - `saguaro agent slice <symbol> --depth 2`
  - `saguaro entrypoints`
  - `saguaro build-graph`
  - `saguaro impact --path <file>`
  - `saguaro verify . --engines native,ruff,semantic --format json`
- If the imported repo is not indexed or not discoverable enough, explicitly do the minimum needed to inspect it well:
  - `saguaro index --path "TARGET_REPO_PATH"` when appropriate
  - if Saguaro fails, document the error briefly, check `saguaro health`, then use careful fallback inspection
- Inspect both sides of the migration:
  - the imported repo's entrypoints, execution loops, planners, UX surfaces, memory systems, analysis engines, governance, telemetry, caches, native boundaries, and extension points
  - Saguaro/Anvil's current code intelligence, indexing, SSAI flow, verification, orchestration, roadmap tooling, native layer, and operator surfaces

What to extract from the imported repo:
- Core features users actually benefit from
- Deep primitives hiding underneath surface features
- Strong execution loops or operating models
- Specialized indexing, storage, search, retrieval, or planning mechanisms
- Useful governance, verification, or safety patterns
- Better UX workflows, REPL affordances, onboarding, or observability
- Valuable packaging, install, or deployment patterns
- Performance techniques worth rebuilding natively
- Research-facing or enterprise-facing capabilities with real leverage

What to reject:
- Cosmetic parity
- Vendor-specific glue that does not generalize
- Features that conflict with Saguaro's architectural center of gravity
- Shallow wrappers that should remain integrations instead of native features
- Complexity that does not improve operator leverage or code intelligence quality

How to think:
- Start from first principles:
  - What is Saguaro fundamentally?
  - What should a world-class code intelligence operating system do natively?
  - Which imported-repo capabilities are really hidden primitives that Saguaro lacks?
- Decompose imported features into:
  - user-visible surface
  - underlying mechanism
  - required data model
  - required runtime loop
  - required verification model
  - required UX surface
- Prefer native Saguaro primitives over thin compatibility layers.
- Ask:
  - Which features would most increase Saguaro's usefulness on day one?
  - Which imported mechanisms would deepen Saguaro's moat if rebuilt properly?
  - Which capabilities only look attractive because of packaging, not architecture?
  - Which features belong in Saguaro core versus Anvil orchestration versus optional tooling?
  - What should be rebuilt as native, high-performance substrate instead of copied at the Python layer?

Required comparison model:
- For every serious candidate, classify it as one of:
  - `Adopt Natively`
  - `Adapt and Strengthen`
  - `Integrate Only`
  - `Ignore`
- Explain why.

Mandatory deliverable structure:
1. First-principles framing
   - What Saguaro fundamentally is
   - What the imported repo fundamentally is
   - Where their overlap is real versus superficial
   - What "native migration" should mean in this context

2. Imported repo decomposition
   - Main subsystems, loops, primitives, and user-facing capabilities in `TARGET_REPO_PATH`
   - Which ones appear strategically important
   - Which ones are implementation artifacts rather than transferable value

3. Saguaro grounding summary
   - Which Saguaro/Anvil modules, hooks, commands, native boundaries, and entrypoints you inspected
   - Which existing primitives can absorb imported capabilities
   - Where Saguaro is already stronger
   - Where Saguaro is currently thin

4. Capability crosswalk
   - Map imported-repo capabilities to Saguaro-native target areas
   - For each mapped capability include:
     - Imported feature or primitive
     - Why users value it
     - Underlying mechanism
     - Saguaro equivalent today
     - Gap type: missing primitive, missing wiring, weak UX, weak performance, weak governance, or missing surface
     - Decision: `Adopt Natively`, `Adapt and Strengthen`, `Integrate Only`, or `Ignore`

5. Candidate migration phases
   - Give me 15-25 candidate phases or workstreams
   - Do not make them generic
   - For each phase include:
     - Name
     - Suggested `phase_id`
     - Imported capability being mined
     - Why it matters to Saguaro users
     - Why native implementation is justified or not justified
     - Exact Saguaro/Anvil wiring points
     - Existing primitives to reuse
     - New primitive, subsystem, or contract needed
     - `repo_scope`
     - `owning_specialist_type`
     - `allowed_writes`
     - `telemetry_contract`
     - `required_evidence`
     - `rollback_criteria`
     - `promotion_gate`
     - `success_criteria`
     - Main risk
     - Smallest credible first implementation step
     - Confidence

6. Migration prioritization
   - Rank the strongest phases by:
     - user leverage
     - architectural fit
     - native performance upside
     - implementation complexity
     - strategic moat
   - Call out:
     - quickest high-value wins
     - foundational substrate work
     - high-risk moonshots
     - features that should stay non-native

7. Ordered implementation roadmap
   - Promote the strongest 3-7 candidates into an execution sequence
   - Use canonical phase ids when possible:
     - `intake`
     - `research`
     - `eid`
     - `questionnaire`
     - `feature_map`
     - `roadmap_draft`
     - `development`
     - `analysis_upgrade`
     - `deep_test_audit`
     - `convergence`
   - For each promoted phase include:
     - `phase_id`
     - Phase title
     - Objective
     - Dependencies
     - Repo scope
     - Owning specialist type
     - Allowed writes
     - Telemetry contract
     - Required evidence
     - Rollback criteria
     - Promotion gate
     - Success criteria
     - Deliverables
     - Exact code wiring points
     - Tests
     - Verification commands
     - Exit criteria

8. Implementation contract
   - Write a validator-friendly markdown section
   - Use only normative requirements with `shall`, `must`, or equivalent language
   - Every bullet must reference exact code paths in backticks
   - Every bullet must reference at least one relevant test path in backticks
   - Every bullet must include explicit verification commands when possible
   - Prefer bullets shaped like:
     - `The system shall implement <behavior> through \`path/to/code.py\` and \`path/to/other.py\`, tested by \`tests/test_x.py\`, and verified with \`pytest tests/test_x.py\` and \`./venv/bin/saguaro verify . --engines native,ruff,semantic --format json\`.`

Important constraints:
- Do not answer from imported-repo marketing or README claims alone.
- Do not just copy the imported repo's feature names into Saguaro.
- Do not produce a generic migration spreadsheet.
- Do not optimize for parity; optimize for native leverage.
- Do not rely on roadmap markdown files in this repo as primary ideation input unless I explicitly ask.
- Do not hand-wave implementation. Name actual files, modules, classes, commands, loops, hooks, or subsystems.
- When something is inference rather than verified repo evidence, label it clearly.

Style:
- Write like a research-heavy systems architect doing acquisition-quality technical due diligence.
- Be concrete, comparative, and mechanism-level.
- Favor architectural synthesis over product-language summaries.
- Treat the imported repo as raw ore and Saguaro as the refinery.
```

If you want a shorter version, use this:

```text
Analyze the imported repository at `TARGET_REPO_PATH` as a capability mine and produce a Saguaro-native migration roadmap. Run from `venv`, use Saguaro-first inspection on both the imported repo and Anvil/Saguaro, and identify which features, mechanisms, execution loops, UX surfaces, verification patterns, performance techniques, and data-model primitives should be rebuilt natively in Saguaro. For each serious capability, classify it as `Adopt Natively`, `Adapt and Strengthen`, `Integrate Only`, or `Ignore`, and explain why. Then give me 15-25 candidate migration phases with exact wiring points, reused primitives, missing primitives, owner type, telemetry contract, evidence, rollback criteria, promotion gates, risks, and first steps. Finally, promote the strongest 3-7 items into an ordered implementation roadmap plus an `Implementation Contract` section written in validator-friendly `shall` statements with exact code refs, test refs, and verification commands.
```
