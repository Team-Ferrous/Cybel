# Deterministic Program Synthesis Roadmap for Saguaro and Anvil

> Authoritative roadmap for reducing the LLM to planning, analysis, and spec formation while moving construction, optimization, and verification into deterministic Saguaro machinery.
>
> Validation target: `./venv/bin/saguaro roadmap validate --path ROADMAP_PROGRAM_SYNTHESIS.md --format json`

---

## 0. Evidence Posture

- `[Repo]` means grounded in local inspection of this repository.
- `[External]` means grounded in papers, official docs, repos, blogs, or practitioner discussions.
- `[Inference]` means design synthesis based on the first two classes of evidence.

This document treats `## 8. Implementation Program` and `## 9. Implementation Contract` as the authoritative execution surface. Earlier sections widen the design space but are intentionally non-authoritative narrative.

---

## 1. First-Principles Framing

### 1.1 What the system fundamentally is

[Repo] Saguaro plus Anvil is already two systems fused together:

- A semantic code operating system: index, graph, parsers, verification, impact analysis, worksets, and native compute.
- A multi-agent execution shell: REPL, planning loops, subagents, memory, telemetry, governance, and evidence-producing workflows.

[Repo] The repo is not fundamentally "an LLM app." The live center of gravity is:

- `Saguaro` as a code understanding and verification substrate.
- `domains/task_execution/enhanced_loop.py` as the execution spine.
- `shared_kernel/event_store.py` as the audit and replay spine.
- `core/native/runtime_telemetry.py` and `core/native/native_qsg_engine.py` as the runtime truth surface.
- `core/task_memory.py` and `core/memory/fabric/policies.py` as the memory policy surface.
- `core/prompts/prompt_manager.py` and `core/prompts/system_prompt_builder.py` as the current LLM loading dock.

### 1.2 What the system appears to be trying to become

[Inference] The architecture wants to become a semantic compiler and mission control plane:

- Humans or LLMs state intent.
- The system lowers intent into a typed, auditable, mechanically checkable program specification.
- Retrieval, construction, optimization, and verification happen through deterministic machinery.
- Code only reaches disk if it satisfies syntax, type, contract, semantic, runtime, and governance gates.
- The LLM becomes a spec author, search heuristic, or exception handler, not the entity that writes final source text.

### 1.3 Core constraints and engineering standards

The viable design space is shaped by these constraints:

| Constraint | Why it matters here |
| --- | --- |
| Determinism | The same spec shall produce the same code, proofs, and evidence bundle. |
| Mathematical rigor | The repo explicitly targets math-heavy domains where "close enough" is not acceptable. |
| Auditability | Existing event, checkpoint, and roadmap machinery means every synthesis step should be inspectable. |
| Native performance | The repo already invests in AVX2/OpenMP native paths; synthesis search cannot live entirely in Python. |
| Incremental adoption | The system has a large active architecture; the roadmap must attach to current loops, not demand a rewrite. |
| Verification-first | Existing culture already values `verify`, AES contracts, replay, and governance. |
| Multi-language realism | Python and C++ are both central. Cross-language synthesis must respect FFI, ABI, and lowering boundaries. |
| Elite ops standard | NASA or Formula 1 quality means telemetry-backed decisions, explicit rollback gates, and zero magical black boxes in the hot path. |

### 1.4 What language support actually has to become

[Inference] You do not need "docs ingestion" in the loose RAG sense. You need executable language support packs.

A language support pack is not a prose summary. It is a deterministic bundle containing:

- Concrete syntax grammar and node taxonomy.
- Typed AST and lowered IR mapping rules.
- Operator semantics and effect semantics.
- Undefined-behavior model and runtime preconditions.
- Standard library contracts and common API schemas.
- Rewrite rules and canonical equivalence classes.
- Emission templates and formatting contracts.
- Proof obligations and translation-validation rules.
- Benchmark examples and failure-inducing counterexamples.

[Inference] Plain docs are just one input into building these packs. The substrate has to become "grammar plus semantics plus contracts plus proofs," not "language manuals stuffed into embeddings."

---

## 2. External Research Scan

### 2.1 Primary patterns that matter

| Source | Type | Pattern | Why it matters here |
| --- | --- | --- | --- |
| [R1](https://microsoft.github.io/z3guide/) | Official docs | SMT solving over arithmetic, bit-vectors, arrays, quantifiers | Needed for deterministic proof of math and interface constraints. |
| [R2](https://cvc5.github.io/) | Official docs | Industrial SMT with built-in SyGuS support | Useful for synthesis tasks that should stay close to formal grammars. |
| [R3](https://sygus.org/) | Official benchmark hub | Syntax-guided synthesis problem framing | Strong template for grammar-bounded synthesis requests and benchmark design. |
| [R4](https://www.semgus.org/) | Research community site | Semantics-guided synthesis | Important step beyond grammar-only synthesis; aligns with language support packs. |
| [R5](https://emina.github.io/rosette/) | Official docs | Solver-aided programming via interpreters and symbolic execution | Closest architectural precedent for turning semantics into synthesis and verification. |
| [R6](https://people.csail.mit.edu/asolar/papers/thesis.pdf) | Thesis | Sketching and CEGIS | Strong fit for partial-program plus hole-filling workflows. |
| [R7](https://github.com/egraphs-good/egg) | GitHub repo | Equality saturation and e-graphs | Relevant for deterministic optimization and canonicalization after assembly. |
| [R8](https://arxiv.org/abs/2111.13040) | Paper | Sketch-guided equality saturation | Strong evidence that eqsat scales better when constrained by sketches and costs. |
| [R9](https://github.com/AliveToolkit/alive2) | GitHub repo | Translation validation for compiler transforms | Ideal template for proving equivalence across lowerings and aggressive rewrites. |
| [R10](https://tree-sitter.github.io/tree-sitter/) | Official docs | Incremental parsing, grammar-driven syntax trees, queryable code navigation | Already central to Saguaro; critical for deterministic construction and structural rewrites. |
| [R11](https://tree-sitter.github.io/tree-sitter/4-code-navigation.html) | Official docs | Query language atop syntax trees for navigation and references | Good model for structural component retrieval and patch induction. |
| [R12](https://semgrep.dev/docs/writing-rules/data-flow/taint-mode/overview) | Official docs | Taint and data-flow style semantic rule execution | Useful for effect constraints, forbidden flows, and safety guards during synthesis. |
| [R13](https://semgrep.dev/blog/2022/demystifying-taint-mode/) | Engineering blog | Practitioner framing of taint rule design | Shows how to move from AST matching to semantically meaningful constraints. |
| [R14](https://coccinelle.gitlabpages.inria.fr/website/) | Official docs | Semantic patching for C | Strong precedent for deterministic large-scale structural edits. |
| [R15](https://comby.dev/) | Official docs | Structural search and replace with syntax awareness | Lightweight bridge between current patch workflows and deeper synthesis. |
| [R16](https://dafny.org/) | Official docs | Spec-first verified programming and code generation | Useful model for carrying contracts through to executable targets. |
| [R17](https://kframework.org/) | Official docs | Executable formal semantics for languages | Strong precedent for "language docs become machine semantics" only when compiled into executable definitions. |
| [R18](https://news.ycombinator.com/item?id=20779535) | Practitioner discussion | Two-solver synthesis intuition and deployment lessons | Reinforces the idea that synthesis is not magic; it is verifier-guided search. |
| [R19](https://news.ycombinator.com/item?id=26225298) | Practitioner discussion | Tree-sitter is syntax only; analysis still has to be built on top | Important warning against treating parsing as semantics. |
| [R20](https://news.ycombinator.com/item?id=29775477) | Practitioner discussion | E-graphs are powerful but memory-hungry; cost models matter | Direct warning for this repo's constant-memory ambitions. |
| [R21](https://discourse.llvm.org/t/blog-post-about-basic-translation-validation-for-aarch64-backend/62988) | Engineering forum | Translation validation catches real codegen bugs in practice | Strong evidence that validation should sit in the deployment path, not only in research mode. |
| [R22](https://github.blog/2020-08-04-codegen-semantics-improved-language-support-system/) | Engineering blog | GitHub's language support stack translates tree-sitter outputs into richer semantic forms | Validates the need for a semantic layer above parsing. |
| [R23](https://github.blog/engineering/architecture-optimization/crafting-a-better-faster-code-view/) | Engineering blog | Fast, structured code views need careful performance engineering | Relevant to user-facing deterministic synthesis lanes and structural navigation. |
| [R24](https://cvc5.github.io/docs/cvc5-1.0.0/examples/sygus-grammar.html) | Official docs | Grammar-constrained synthesis examples | Useful for early bounded synthesis experiments. |
| [R25](https://cvc5.github.io/docs/cvc5-1.1.0/examples/sygus-fun.html) | Official docs | Function synthesis examples | Good fit for deterministic math and adapter synthesis pilots. |
| [R26](https://cvc5.github.io/docs/cvc5-1.0.0/examples/sygus-inv.html) | Official docs | Invariant synthesis examples | Relevant to generating loop and boundary guards from specs. |

### 2.2 Adjacent-industry analogies worth stealing

[External] The best analogies are not "other AI agents." They are systems that separate planning from execution and prove state transitions before commitment.

- Formula 1 race engineering: the strategist proposes, but telemetry, simulation, and pit-wall constraints decide. In this repo, `core/native/runtime_telemetry.py` already looks like a pit-wall ledger waiting to govern synthesis choices.
- Digital-twin manufacturing: no part is cut until the manufacturing path is simulated. In this repo, `shared_kernel/event_store.py`, `sandbox`, and replay exports can become code manufacturing twins.
- EDA and hardware synthesis: hardware engineers do not ask a language model to freehand a netlist. They write constraints, libraries, timing budgets, and equivalence checks. That is the right mental model for Saguaro.
- Mission assurance: spacecraft software carries traceability, evidence, and rollback criteria. This repo already has roadmap validation, checkpoints, and safety-case surfaces.

### 2.3 The design rules implied by the research

[Inference] Compressed into operating rules:

1. Grammar alone is not enough; deterministic synthesis requires executable semantics and contracts.
2. Component assembly is the right first beachhead because it reuses verified local code.
3. Equality saturation is powerful, but it must be sketch-bounded, cost-bounded, and telemetry-bounded.
4. Translation validation is strategically better than hoping every rewrite rule is perfect.
5. Semantic patching is not a side tool; it is a bridge between human-maintained patterns and automated construction.
6. The most effective LLM reduction pattern is "LLM writes structured spec, deterministic engine constructs and proves."
7. Benchmarks shall emphasize proof success, equivalence, and operational evidence, not only output similarity.

---

## 3. Repo Grounding Summary

### 3.1 Commands executed

[Repo] The inspection path used the repo venv and Saguaro-first workflow:

- `source venv/bin/activate`
- `./venv/bin/saguaro health`
- `./venv/bin/saguaro entrypoints`
- `./venv/bin/saguaro build-graph`
- `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- `./venv/bin/saguaro query "prompt builder system prompt assembly" --k 8`
- `./venv/bin/saguaro query "orchestration planning loop master agent subagent execution loop" --k 5`
- `./venv/bin/saguaro query "memory telemetry runtime signals hooks lifecycle governance verification" --k 5`
- `./venv/bin/saguaro query "anvil cli repl main entrypoint" --k 8`
- `./venv/bin/saguaro impact --path core/prompts/system_prompt_builder.py`
- `./venv/bin/saguaro impact --path domains/task_execution/enhanced_loop.py`
- `./venv/bin/saguaro agent skeleton ...` on the core files listed below

### 3.2 Saguaro health and build posture

[Repo] `saguaro health` reported:

- Fresh index with `58,640` indexed entities.
- Graph ready with `450,437` nodes and `414,035` edges.
- Graph coverage `88.7%`.
- Native stack ready with AVX2, FMA, OpenMP, and runtime probe success.
- Resident query gateway enabled with p50 about `3294ms`, p95 about `7745ms`, p99 about `12386ms`.

[Repo] `saguaro build-graph` found the relevant native targets:

- `saguaro_core`
- `saguaro_native`
- `anvil_runtime_core`
- `anvil_native_ops`

This matters because deterministic synthesis cannot be designed as a pure Python feature bolted onto the side of this system.

### 3.3 Inspection outcomes by subsystem

#### Prompt assembly

[Repo] `core/prompts/prompt_manager.py` and `core/prompts/system_prompt_builder.py` are compact and high-leverage.

- `PromptManager.get_master_prompt()` assembles identity, cognition, protection, context, tool mandates, and AES contracts.
- `PromptManager.get_system_prompt()` already produces role-specific contract blocks.
- `SystemPromptBuilder.build()` appends AES condensed guidance, visual guidance, connectivity context, AAL classification, domain rules, and required artifacts.

[Inference] This is the natural place to stop asking the LLM for free-form code and start demanding a typed synthesis spec.

#### Orchestration and planning loops

[Repo] `domains/task_execution/enhanced_loop.py` is the main high-impact orchestration surface.

- `_planning_phase()` retrieves prior lessons, generates understanding text, and asks the model to emit a markdown implementation plan.
- That plan is written to `temp_plan.md` and reparsed through `PlanArtifact.load()`.
- `_verification_phase()` aggregates syntax, lint, AES, type, test, and runtime checks through `tools/verify.py`, then may spawn a `SubAgent` to repair failures.

[Repo] `core/loops/phases/synthesis_phase.py` is not code synthesis. It is final answer synthesis with grounding checks against loaded evidence.

[Inference] The current architecture still treats the model output as the source of executable content. The strongest first move is to change the planning output from markdown plan text into a typed synthesis IR.

#### Memory systems

[Repo] `core/task_memory.py` already stores semantic signatures, persists task memories to Saguaro memory, creates fabric memories, and projects them.

[Repo] `core/memory/fabric/policies.py` already chooses between `recent_kv`, `latent_replay`, `semantic_summary`, and `repo_delta_memory` based on cache ratios, queue wait, drift overhead, and delta size.

[Inference] This is stronger than a typical agent memory layer. It should govern synthesis search tiering, not only conversation recall.

#### Telemetry and runtime signals

[Repo] `core/native/runtime_telemetry.py` normalizes runtime capability ledgers with:

- capability digests
- ISA state
- thread configuration
- scheduler state
- affinity
- degraded capability reporting

[Repo] `core/native/native_qsg_engine.py` builds a deep runtime capability structure that already tracks:

- backend selection
- tokenizer backend
- native-vs-Python hot path calls
- full graph and full QSG enablement
- scheduler queue wait
- KV fragmentation
- state pages
- latent packet counts
- evidence capsule counts
- hot-path proof metadata

[Inference] This is exactly the missing signal plane that most synthesis systems never have. It should be promoted into search-policy input.

#### Governance and verification

[Repo] `tools/verify.py` aggregates syntax, lint, AES, types, tests, and QSG runtime verification.

[Repo] `core/unified_chat_loop.py` runs pre-finalize governance checkpoints before synthesis.

[Repo] `tests/test_saguaro_roadmap_validator.py` confirms that the roadmap pipeline prefers `## Implementation Contract` over narrative prose.

[Inference] The repo already has the culture and tooling needed for proof-carrying program construction. The gap is not governance; it is formal synthesis substrate.

#### Event and lifecycle recording

[Repo] `shared_kernel/event_store.py` already provides:

- event emission
- run export
- deterministic replay hashes
- replay tapes for QSG requests
- mission checkpoints
- mission capsules
- safety-case graphs

[Inference] This is the right home for synthesis trace tapes, counterexample packs, and proof capsules.

#### User-facing loop and entrypoints

[Repo] `cli/repl.py` is the user-facing cockpit. It imports:

- `LoopOrchestrator`
- `CampaignRunner`
- `ProjectMemory`
- `EnhancedAgenticLoop`
- `SaguaroSubstrate`
- event store and workset machinery

[Inference] Deterministic synthesis should appear here as a first-class mode or lane, not as a hidden feature toggle.

### 3.4 Strongest existing primitives

[Repo] The most important underexploited primitives already present are:

- `shared_kernel/event_store.py` as deterministic replay substrate.
- `core/native/runtime_telemetry.py` as runtime decision substrate.
- `core/memory/fabric/policies.py` as search-memory policy substrate.
- `Saguaro` parsing, graph, and query surfaces as component discovery substrate.
- AES prompt and runtime contracts as spec discipline substrate.
- roadmap validation as completion discipline substrate.

### 3.5 Where the architecture looks strongest

[Repo]

- Auditability is unusually strong.
- Telemetry depth is unusually strong.
- Native CPU-first performance posture is unusually strong.
- Memory tiering is stronger than expected for an agent repo.
- Query and graph infrastructure are real, not conceptual.

### 3.6 Where the architecture looks thin or underexploited

[Repo]

- Planning still lowers to markdown, not to typed IR.
- Synthesis phase still means "compose a user response."
- Verification is broad but mostly post-hoc and tool-aggregated, not proof-driven.
- There is no AST construction or deterministic emission layer.
- There is no language support pack abstraction.
- There is no translation-validation lane across lowerings.
- There is no synthesis benchmark corpus tied to roadmap promotion.

### 3.7 Errors and fallbacks encountered

[Repo]

- `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json` did not finish within a `30s` timeout and only yielded partial evidence up to `ArchitectureAnalyzer`.
- Several `saguaro agent slice` requests did not return useful payloads within roughly `20s`.
- Fallback used only after documenting the issue, and only for exact line-bounded reads of the affected symbols.

---

## 4. Hidden Assumptions Limiting the Design Space

1. The planner must emit markdown because humans read markdown well.
2. The best use of Saguaro is retrieval plus verification, not construction.
3. Language support can be approximated by prose docs and examples.
4. It is acceptable for the model to perform the final lowering from intent to source code.
5. Math-heavy tasks are fundamentally an LLM weakness rather than a signal to move the problem into solver-backed search.
6. Telemetry belongs to runtime optimization only and should not steer synthesis.
7. Event logs are for observability only and not for deterministic replay and counterexample reuse.
8. Verification happens after code exists, rather than guiding code construction.
9. Cross-language synthesis requires a universal magical IR, instead of explicit boundary packs and translation validators.
10. Equality saturation must be global to be useful, even though local bounded eqsat is strategically enough for a first system.

---

## 5. Candidate Implementation Phases

> Mix target: about 70% practical, 30% moonshot.

### 5.1 SagSpec IR Boundary `[Practical]`

- Suggested `phase_id`: `eid`
- Core insight: replace markdown planning artifacts with a typed synthesis specification.
- External inspiration or analogy: Rosette specifications, Sketch holes, Dafny contracts `[R5][R6][R16]`
- Why it fits Saguaro and Anvil specifically: `core/agents/planner.py`, `core/prompts/prompt_manager.py`, and `domains/task_execution/enhanced_loop.py` already centralize prompt and plan assembly.
- Exact wiring points: `core/agents/planner.py`, `core/agents/planner_agent.py`, `core/prompts/prompt_manager.py`, `domains/task_execution/enhanced_loop.py`, `core/loops/phases/synthesis_phase.py`
- Existing primitives it can reuse: AES prompt contracts, `PlanArtifact`, roadmap phase packets, prompt-context plumbing.
- New primitive, data flow, or subsystem needed: `Saguaro/saguaro/synthesis/spec.py` with `SagSpec`, `SpecConstraint`, `SpecEvidenceRef`, and `SpecLowerer`.
- `repo_scope`: `Saguaro/saguaro/synthesis/**`, `core/agents/**`, `core/prompts/**`
- `owning_specialist_type`: `formal_methods_architect`
- `allowed_writes`: `Saguaro/saguaro/synthesis/**`, `core/agents/planner.py`, `core/prompts/prompt_manager.py`, `tests/test_sagspec.py`
- `telemetry_contract`: `spec_parse_latency_ms`, `spec_validation_rate`, `constraint_count`, `missing_field_count`
- `required_evidence`: hand-authored and model-authored specs for math, adapters, semantic patches, and small feature additions
- `rollback_criteria`: spec invalidation rate above `5%` on curated tasks
- `promotion_gate`: planner emits validator-clean specs for at least `20` benchmark tasks
- `success_criteria`: planning output is no longer source text or markdown-first
- Why this creates value: it changes the authority boundary
- Why this creates moat: proprietary spec plus proof loop is harder to imitate than prompt engineering
- Main risk or failure mode: overdesigned schema that users and models cannot fill
- Smallest credible first experiment: lower one simple function request into `SagSpec` and reject incomplete specs deterministically
- Confidence level: `0.95`

### 5.2 Language Support Packs `[Practical]`

- Suggested `phase_id`: `research`
- Core insight: codify each supported language as a deterministic pack, not a doc bundle.
- External inspiration or analogy: K executable semantics, SemGuS, GitHub Semantic language support `[R4][R17][R22]`
- Why it fits Saguaro and Anvil specifically: Saguaro already parses Python and C++; the missing layer is semantics and emission.
- Exact wiring points: `Saguaro/saguaro/parsing/parser.py`, `Saguaro/saguaro/analysis/code_graph.py`, `core/native/native_qsg_engine.py`
- Existing primitives it can reuse: tree-sitter grammars, `CodeEntity`, graph builders, bridge synthesizer.
- New primitive, data flow, or subsystem needed: `Saguaro/saguaro/language_packs/{python,cpp}.py` with grammar maps, operator semantics, contracts, emitter rules, UB tables.
- `repo_scope`: `Saguaro/saguaro/language_packs/**`, `Saguaro/saguaro/parsing/**`, `Saguaro/saguaro/analysis/**`
- `owning_specialist_type`: `language_systems_engineer`
- `allowed_writes`: `Saguaro/saguaro/language_packs/**`, `tests/test_language_packs.py`
- `telemetry_contract`: `language_pack_coverage_pct`, `unsupported_construct_count`, `contract_stub_hit_rate`
- `required_evidence`: coverage reports over repo symbols for Python and C++
- `rollback_criteria`: supported-node coverage below `85%` for initial target languages
- `promotion_gate`: language packs resolve syntax, type families, and emitter rules for top repo constructs
- `success_criteria`: docs are no longer the synthesis substrate; packs are
- Why this creates value: it answers the "what does language support look like?" question precisely
- Why this creates moat: executable language packs compound over time
- Main risk or failure mode: semantics pack becomes too broad too early
- Smallest credible first experiment: Python and C++ packs limited to function-level synthesis with numeric operators, containers, and imports/includes
- Confidence level: `0.92`

### 5.3 Planner-to-Spec Lowering and Spec Linting `[Practical]`

- Suggested `phase_id`: `eid`
- Core insight: force every planning request through a deterministic spec linter before synthesis begins.
- External inspiration or analogy: compiler front-end diagnostics, mission checklist discipline
- Why it fits Saguaro and Anvil specifically: `EnhancedAgenticLoop._planning_phase()` already stages understanding before execution.
- Exact wiring points: `domains/task_execution/enhanced_loop.py`, `core/agents/planner_agent.py`, `core/prompts/system_prompt_builder.py`
- Existing primitives it can reuse: plan artifacts, AAL context, required artifacts logic, task sections.
- New primitive, data flow, or subsystem needed: `SpecLintResult`, `SpecCompletenessScore`, `spec_missing_evidence` diagnostics.
- `repo_scope`: `core/agents/**`, `domains/task_execution/**`, `Saguaro/saguaro/synthesis/**`
- `owning_specialist_type`: `planning_systems_engineer`
- `allowed_writes`: `core/agents/planner_agent.py`, `domains/task_execution/enhanced_loop.py`, `tests/test_sagspec_lowering.py`
- `telemetry_contract`: `spec_lint_failure_rate`, `missing_precondition_count`, `clarification_required_rate`
- `required_evidence`: mis-specified prompts and expected deterministic failures
- `rollback_criteria`: false-block rate above `10%`
- `promotion_gate`: lint blocks ambiguous requests reliably while letting bounded tasks pass
- `success_criteria`: ambiguous requests fail early and explicitly
- Why this creates value: prevents garbage specs from poisoning later stages
- Why this creates moat: disciplined front-end quality compounds across the whole stack
- Main risk or failure mode: user friction if linter is too strict
- Smallest credible first experiment: lint only missing input/output types, target paths, and verification expectations
- Confidence level: `0.90`

### 5.4 SagAST Constructor and Emitter `[Practical]`

- Suggested `phase_id`: `development`
- Core insight: construct source through syntax-safe AST assembly instead of free-form string generation.
- External inspiration or analogy: tree-sitter construction, compiler AST emitters `[R10][R11]`
- Why it fits Saguaro and Anvil specifically: the repo already has parsing and graph extraction, but no deterministic reverse path.
- Exact wiring points: `Saguaro/saguaro/parsing/parser.py`, new `Saguaro/saguaro/synthesis/ast_builder.py`
- Existing primitives it can reuse: parser node taxonomy, `CodeEntity`, tree-sitter language definitions.
- New primitive, data flow, or subsystem needed: `SagASTNode`, `ASTBuilder`, `Emitter`, round-trip serializer.
- `repo_scope`: `Saguaro/saguaro/synthesis/**`, `Saguaro/saguaro/parsing/**`
- `owning_specialist_type`: `compiler_engineer`
- `allowed_writes`: `Saguaro/saguaro/synthesis/ast_builder.py`, `tests/test_ast_builder.py`
- `telemetry_contract`: `ast_build_latency_ms`, `emit_roundtrip_fidelity`, `syntax_reject_count`
- `required_evidence`: parse-build-emit-parse round-trip suite over repo snippets
- `rollback_criteria`: round-trip fidelity below `99%`
- `promotion_gate`: zero syntax failures on curated emission benchmarks
- `success_criteria`: function-level and adapter-level code can be built without string concatenation
- Why this creates value: removes a major nondeterministic failure mode
- Why this creates moat: grammar-bounded construction is a deeper primitive than prompt-cooked code text
- Main risk or failure mode: lossless emission and formatting become harder than expected
- Smallest credible first experiment: emit valid Python and C++ function wrappers from typed signatures and block bodies
- Confidence level: `0.88`

### 5.5 Component Retriever and Assembler `[Practical]`

- Suggested `phase_id`: `feature_map`
- Core insight: the first reliable synthesizer should mostly assemble indexed local components.
- External inspiration or analogy: component-based synthesis `[R18]`
- Why it fits Saguaro and Anvil specifically: Saguaro already stores semantically indexed code entities and graph relationships.
- Exact wiring points: `Saguaro/saguaro/query/**`, `Saguaro/saguaro/indexing/engine.py`, `Saguaro/saguaro/analysis/bridge_synthesizer.py`
- Existing primitives it can reuse: query pipeline, code graph, bridge confidence, impact analysis.
- New primitive, data flow, or subsystem needed: `ComponentCandidate`, `AssemblyPlan`, `CompatibilityScorer`, `AssemblyGraph`.
- `repo_scope`: `Saguaro/saguaro/query/**`, `Saguaro/saguaro/synthesis/**`
- `owning_specialist_type`: `synthesis_engineer`
- `allowed_writes`: `Saguaro/saguaro/synthesis/assembler.py`, `tests/test_component_assembler.py`
- `telemetry_contract`: `component_recall_at_k`, `assembly_success_rate`, `assembly_reuse_ratio`
- `required_evidence`: successful synthesis of wrappers, adapters, validators, and math helpers using local components
- `rollback_criteria`: recall at `k=10` below `0.75` on curated tasks
- `promotion_gate`: component assembly handles at least `60%` of phase-one benchmark tasks
- `success_criteria`: most early synthesis uses retrieval plus composition, not invention
- Why this creates value: immediate leverage from the existing repo
- Why this creates moat: index plus graph plus compatibility plus proofs is hard to clone
- Main risk or failure mode: missing or weak type contracts on retrieved components
- Smallest credible first experiment: synthesize Python adapters around existing runtime telemetry and memory APIs
- Confidence level: `0.93`

### 5.6 Contract Harvester from Code, Tests, and Docs `[Practical]`

- Suggested `phase_id`: `feature_map`
- Core insight: extract latent preconditions, postconditions, and examples from existing repo assets.
- External inspiration or analogy: contract mining, test-to-spec extraction, Semgrep-style semantic rules `[R12][R13]`
- Why it fits Saguaro and Anvil specifically: the repo already has dense tests, contracts, and runtime invariants.
- Exact wiring points: `tools/verify.py`, `tests/**`, `Saguaro/saguaro/requirements/**`, `core/qsg/runtime_contracts.py`
- Existing primitives it can reuse: roadmap validation, runtime contract checks, AES rules, test suites.
- New primitive, data flow, or subsystem needed: `ContractHarvester`, `ExampleCorpus`, `SymbolContractStore`.
- `repo_scope`: `Saguaro/saguaro/synthesis/**`, `Saguaro/saguaro/requirements/**`, `core/**`
- `owning_specialist_type`: `spec_mining_engineer`
- `allowed_writes`: `Saguaro/saguaro/synthesis/contract_harvester.py`, `tests/test_contract_harvester.py`
- `telemetry_contract`: `harvested_contract_count`, `symbol_contract_coverage`, `example_reuse_rate`
- `required_evidence`: harvested contracts for runtime telemetry, event store, and memory policies
- `rollback_criteria`: contract precision too low for promotion decisions
- `promotion_gate`: extracted contracts materially improve component compatibility scoring
- `success_criteria`: synthesis no longer depends only on hand-authored specs
- Why this creates value: converts repo history into machine-usable semantics
- Why this creates moat: the repo's own contracts become reusable synthesis capital
- Main risk or failure mode: noisy inference from weak tests or prose
- Smallest credible first experiment: extract contracts from `tests/test_qsg_runtime_contracts.py` and `tests/test_aes_runtime_contracts.py`
- Confidence level: `0.86`

### 5.7 Solver Bridge `[Practical]`

- Suggested `phase_id`: `analysis_upgrade`
- Core insight: attach Z3 or cvc5 to `SagSpec` and `SagAST` so correctness is checked symbolically.
- External inspiration or analogy: Z3, cvc5, Rosette, Dafny `[R1][R2][R5][R16]`
- Why it fits Saguaro and Anvil specifically: the repo already distinguishes math-heavy, strict, and contract-sensitive work.
- Exact wiring points: `Saguaro/saguaro/verification/**`, `Saguaro/saguaro/analysis/dfg_builder.py`, `core/agents/domain/science/formal_methods_subagent.py`
- Existing primitives it can reuse: DFG typing, runtime invariants, verification lanes.
- New primitive, data flow, or subsystem needed: `Saguaro/saguaro/synthesis/solver.py`, SMT sort lowering, counterexample model export.
- `repo_scope`: `Saguaro/saguaro/synthesis/**`, `Saguaro/saguaro/verification/**`
- `owning_specialist_type`: `verification_engineer`
- `allowed_writes`: `Saguaro/saguaro/synthesis/solver.py`, `tests/test_solver.py`
- `telemetry_contract`: `solver_sat_time_ms`, `solver_timeout_rate`, `counterexample_count`, `proof_coverage_pct`
- `required_evidence`: solver-backed proofs for numeric helper functions and interface guards
- `rollback_criteria`: timeout rate above `10%` on bounded tasks
- `promotion_gate`: zero false positives on curated solver benchmarks
- `success_criteria`: deterministic proof becomes a construction-stage gate
- Why this creates value: directly attacks the math weakness of LLMs
- Why this creates moat: proof-backed synthesis is categorically stronger than text generation
- Main risk or failure mode: poorly lifted semantics causing unsound or useless proofs
- Smallest credible first experiment: verify normalization, bounds, and guard insertion for a numeric helper function
- Confidence level: `0.90`

### 5.8 Translation Validation Lane `[Practical]`

- Suggested `phase_id`: `analysis_upgrade`
- Core insight: validate that lowerings and optimizations preserve behavior rather than trusting every transform.
- External inspiration or analogy: Alive2 and LLVM translation validation `[R9][R21]`
- Why it fits Saguaro and Anvil specifically: the repo has meaningful Python/C++ boundaries and native hot paths.
- Exact wiring points: `core/native/native_qsg_engine.py`, `core/native/runtime_telemetry.py`, `Saguaro/saguaro/analysis/bridge_synthesizer.py`
- Existing primitives it can reuse: bridge confidence, native capability ledgers, event replay tapes.
- New primitive, data flow, or subsystem needed: `translation_validator.py`, IR comparator, proof artifact format.
- `repo_scope`: `Saguaro/saguaro/synthesis/**`, `core/native/**`
- `owning_specialist_type`: `lowering_blacksmith`
- `allowed_writes`: `Saguaro/saguaro/synthesis/translation_validator.py`, `tests/test_translation_validation.py`
- `telemetry_contract`: `translation_validation_pass_rate`, `ir_mismatch_count`, `undefined_behavior_block_count`
- `required_evidence`: validation cases for generated wrappers, numeric kernels, and FFI adapters
- `rollback_criteria`: validation false negatives or false positives on seeded faults
- `promotion_gate`: translated code paths ship only when equivalence witnesses exist
- `success_criteria`: optimization and lowering are no longer trust-based
- Why this creates value: catches subtle bugs that normal tests miss
- Why this creates moat: high-assurance generation pipelines rarely ship real equivalence checks
- Main risk or failure mode: IR generation for dynamic-language surfaces becomes messy
- Smallest credible first experiment: translation-validate a narrow C++ helper lowering and a Python wrapper around it
- Confidence level: `0.82`

### 5.9 Semantic Patch Induction `[Practical]`

- Suggested `phase_id`: `development`
- Core insight: infer reusable structural rewrites from repeated repo edits and codify them as deterministic transformations.
- External inspiration or analogy: Coccinelle, Comby `[R14][R15]`
- Why it fits Saguaro and Anvil specifically: the repo already values governance, rules, and repetitive architectural enforcement.
- Exact wiring points: `Saguaro/saguaro/agents/scribe.py`, `Saguaro/saguaro/verification/**`, `core/campaign/roadmap_validator.py`
- Existing primitives it can reuse: semantic queries, graph edges, code entity metadata.
- New primitive, data flow, or subsystem needed: `semantic_patch_rule.py`, `patch_inductor.py`, rule witness format.
- `repo_scope`: `Saguaro/saguaro/synthesis/**`, `Saguaro/saguaro/verification/**`
- `owning_specialist_type`: `refactoring_engineer`
- `allowed_writes`: `Saguaro/saguaro/synthesis/patch_inductor.py`, `tests/test_semantic_patch_induction.py`
- `telemetry_contract`: `rule_match_precision`, `rule_replay_success_rate`, `unsafe_rewrite_block_count`
- `required_evidence`: induced rules from recurring prompt, contract, or telemetry patterns
- `rollback_criteria`: false rewrite rate above `1%`
- `promotion_gate`: rules can replay cleanly in sandbox before disk writes
- `success_criteria`: common transformations stop depending on model text generation
- Why this creates value: converts repo evolution into deterministic editing power
- Why this creates moat: pattern capital accumulates
- Main risk or failure mode: overgeneralized rewrites
- Smallest credible first experiment: infer and replay a structural update pattern in contract-bearing prompt code
- Confidence level: `0.81`

### 5.10 Effect, Taint, and Resource Constraint Engine `[Practical]`

- Suggested `phase_id`: `analysis_upgrade`
- Core insight: synthesis should reason about forbidden flows, mutations, file writes, and runtime budgets as first-class constraints.
- External inspiration or analogy: Semgrep taint mode `[R12][R13]`
- Why it fits Saguaro and Anvil specifically: the repo already has governance gates and runtime budgets.
- Exact wiring points: `Saguaro/saguaro/analysis/dfg_builder.py`, `tools/verify.py`, `core/memory/fabric/policies.py`
- Existing primitives it can reuse: DFGs, AES verification, memory tier telemetry, runtime budgets.
- New primitive, data flow, or subsystem needed: `EffectModel`, `ForbiddenFlow`, `BudgetConstraint`.
- `repo_scope`: `Saguaro/saguaro/analysis/**`, `Saguaro/saguaro/synthesis/**`, `tools/**`
- `owning_specialist_type`: `security_formal_methods_engineer`
- `allowed_writes`: `Saguaro/saguaro/synthesis/effects.py`, `tests/test_synthesis_effects.py`
- `telemetry_contract`: `forbidden_flow_block_count`, `resource_budget_reject_count`, `effect_inference_coverage`
- `required_evidence`: blocked syntheses that would violate file-write, subprocess, or data-flow constraints
- `rollback_criteria`: effect model blocks valid safe programs excessively
- `promotion_gate`: effect constraints catch seeded unsafe constructions
- `success_criteria`: safety and ops budgets are construction-time constraints
- Why this creates value: fewer unsafe proposals reach later verification
- Why this creates moat: effect-aware synthesis is materially harder than prompt-only codegen
- Main risk or failure mode: weak effect inference for dynamic surfaces
- Smallest credible first experiment: forbid generation that bypasses the sanctioned verification path
- Confidence level: `0.79`

### 5.11 FFI and Native Boundary Synthesis `[Practical]`

- Suggested `phase_id`: `development`
- Core insight: deterministic synthesis is especially tractable for wrapper code and boundary glue.
- External inspiration or analogy: codegen for bindings, ABI enforcement
- Why it fits Saguaro and Anvil specifically: this repo has substantial Python/C++ interaction and runtime capability surfaces.
- Exact wiring points: `core/native/native_qsg_engine.py`, `core/native/runtime_telemetry.py`, `Saguaro/saguaro/analysis/bridge_synthesizer.py`
- Existing primitives it can reuse: bridge synthesizer, native build graph, capability ledgers.
- New primitive, data flow, or subsystem needed: `ffi_pack.py`, `adapter_generator.py`, ABI proof metadata.
- `repo_scope`: `Saguaro/saguaro/synthesis/**`, `core/native/**`
- `owning_specialist_type`: `abi_cartographer`
- `allowed_writes`: `Saguaro/saguaro/synthesis/adapter_generator.py`, `tests/test_ffi_synthesis.py`
- `telemetry_contract`: `adapter_generation_success_rate`, `abi_mismatch_count`, `bridge_validation_pass_rate`
- `required_evidence`: generated wrappers for existing native operations
- `rollback_criteria`: synthesized adapters produce runtime ABI mismatches
- `promotion_gate`: wrappers pass runtime contract and smoke tests
- `success_criteria`: a narrow but valuable synthesis lane exists early
- Why this creates value: boundary code is tedious, structured, and high-volume
- Why this creates moat: repo-specific native boundary knowledge is durable
- Main risk or failure mode: hidden ABI assumptions not modeled in the first version
- Smallest credible first experiment: synthesize a wrapper for a simple native capability report surface
- Confidence level: `0.84`

### 5.12 Synthesis Replay Tape and Counterexample Store `[Practical]`

- Suggested `phase_id`: `deep_test_audit`
- Core insight: every synthesis attempt should export a deterministic trace, counterexample set, and proof story.
- External inspiration or analogy: flight data recorders, digital twins, mission assurance
- Why it fits Saguaro and Anvil specifically: `shared_kernel/event_store.py` already supports deterministic hashes, replay exports, mission capsules, and safety cases.
- Exact wiring points: `shared_kernel/event_store.py`, `domains/task_execution/enhanced_loop.py`, `cli/repl.py`
- Existing primitives it can reuse: replay export, checkpoints, safety-case generation.
- New primitive, data flow, or subsystem needed: `synthesis_replay.py`, `counterexample_pack.py`, `proof_capsule.json`.
- `repo_scope`: `shared_kernel/**`, `Saguaro/saguaro/synthesis/**`, `domains/task_execution/**`
- `owning_specialist_type`: `observability_contract_engineer`
- `allowed_writes`: `shared_kernel/event_store.py`, `Saguaro/saguaro/synthesis/replay_tape.py`, `tests/test_synthesis_replay_tape.py`
- `telemetry_contract`: `replay_export_count`, `counterexample_reuse_rate`, `proof_capsule_count`
- `required_evidence`: replayable traces for successful and failed synthesis runs
- `rollback_criteria`: replay cannot deterministically reconstruct decision sequence
- `promotion_gate`: counterexamples re-enter search and materially improve subsequent attempts
- `success_criteria`: synthesis becomes inspectable and learnable
- Why this creates value: debugging and trust both improve
- Why this creates moat: repo-specific synthesis memory becomes cumulative
- Main risk or failure mode: trace volume or event schema sprawl
- Smallest credible first experiment: export one deterministic synthesis tape through the existing event store
- Confidence level: `0.94`

### 5.13 Runtime-Aware Synthesis Policy `[Practical]`

- Suggested `phase_id`: `development`
- Core insight: use native runtime capability and pressure signals to choose search strategy, component preference, and optimization aggression.
- External inspiration or analogy: Formula 1 pit-wall strategy under live telemetry
- Why it fits Saguaro and Anvil specifically: `build_runtime_capability_ledger()` and `_build_runtime_capabilities()` already emit rich decision inputs.
- Exact wiring points: `core/native/runtime_telemetry.py`, `core/native/native_qsg_engine.py`, `core/memory/fabric/policies.py`
- Existing primitives it can reuse: queue wait, drift overhead, cache hit ratios, degraded capabilities.
- New primitive, data flow, or subsystem needed: `SynthesisPolicyEngine`, `StrategyDecision`, telemetry-to-search mapping.
- `repo_scope`: `core/native/**`, `core/memory/fabric/**`, `Saguaro/saguaro/synthesis/**`
- `owning_specialist_type`: `runtime_strategy_engineer`
- `allowed_writes`: `Saguaro/saguaro/synthesis/policy.py`, `tests/test_synthesis_policy_engine.py`
- `telemetry_contract`: `strategy_switch_count`, `telemetry_guided_win_rate`, `degraded_mode_usage`
- `required_evidence`: cases where policy shifts from assembler-only to solver-plus-search or from eqsat-on to eqsat-off
- `rollback_criteria`: unstable or oscillating policy decisions
- `promotion_gate`: policy outperforms fixed heuristics on bounded benchmarks
- `success_criteria`: synthesis path selection is telemetry-backed
- Why this creates value: better latency and predictability
- Why this creates moat: few synthesis systems have this depth of runtime self-knowledge
- Main risk or failure mode: overfitting policy to local machine behavior
- Smallest credible first experiment: use queue wait and degraded capabilities to disable expensive search modes
- Confidence level: `0.88`

### 5.14 Memory-Tiered Synthesis Cache `[Practical]`

- Suggested `phase_id`: `development`
- Core insight: cache specs, subproofs, counterexamples, and component assemblies across multiple memory tiers.
- External inspiration or analogy: compilation caches, retrieval-augmented proof search
- Why it fits Saguaro and Anvil specifically: the repo already has explicit tier decisions in `MemoryTierPolicy`.
- Exact wiring points: `core/task_memory.py`, `core/memory/fabric/policies.py`, `shared_kernel/event_store.py`
- Existing primitives it can reuse: latent replay, semantic summary, repo delta memory, task memories.
- New primitive, data flow, or subsystem needed: `SynthesisMemoryRecord`, `SubproofCache`, `CounterexampleIndex`.
- `repo_scope`: `core/task_memory.py`, `core/memory/fabric/**`, `Saguaro/saguaro/synthesis/**`
- `owning_specialist_type`: `memory_systems_engineer`
- `allowed_writes`: `Saguaro/saguaro/synthesis/cache.py`, `tests/test_synthesis_cache.py`
- `telemetry_contract`: `cache_hit_ratio`, `counterexample_cache_hits`, `proof_reuse_hits`
- `required_evidence`: repeated tasks showing lower latency and fewer failed attempts
- `rollback_criteria`: stale cache causes incorrect synthesis acceptance
- `promotion_gate`: cached traces produce strictly better bounded performance
- `success_criteria`: prior synthesis evidence is reusable without model involvement
- Why this creates value: lower cost and faster convergence
- Why this creates moat: the system learns deterministically from itself
- Main risk or failure mode: invalidation complexity
- Smallest credible first experiment: cache normalized specs and counterexamples keyed by capability digest plus repo delta
- Confidence level: `0.85`

### 5.15 Benchmark and Competition Harness `[Practical]`

- Suggested `phase_id`: `deep_test_audit`
- Core insight: build a repo-local synthesis benchmark harness modeled on SyGuS-style tasks and replayable proofs.
- External inspiration or analogy: SyGuS competition and solver benchmark culture `[R3][R24][R25][R26]`
- Why it fits Saguaro and Anvil specifically: the repo already has benchmark and audit machinery.
- Exact wiring points: `benchmarks/**`, `audit/**`, `tests/test_perf_roadmap_phases.py`
- Existing primitives it can reuse: benchmark suites, roadmap validation, runtime audit flows.
- New primitive, data flow, or subsystem needed: `benchmarks/synthesis_suite.py`, benchmark manifests, proof-score reports.
- `repo_scope`: `benchmarks/**`, `audit/**`, `Saguaro/saguaro/synthesis/**`
- `owning_specialist_type`: `benchmark_engineer`
- `allowed_writes`: `benchmarks/synthesis_suite.py`, `tests/test_synthesis_benchmarks.py`
- `telemetry_contract`: `proof_pass_rate`, `median_synthesis_latency_ms`, `repair_iterations_per_task`
- `required_evidence`: benchmark corpus covering math, adapters, patches, and contractful edits
- `rollback_criteria`: no stable benchmark signal for promotions
- `promotion_gate`: roadmap promotions require benchmark deltas, not narrative claims
- `success_criteria`: deterministic synthesis progress becomes measurable
- Why this creates value: keeps the roadmap honest
- Why this creates moat: internal benchmark realism drives architecture quality
- Main risk or failure mode: benchmarks overfit to narrow tasks
- Smallest credible first experiment: `10` repo-shaped synthesis tasks with proof witnesses
- Confidence level: `0.93`

### 5.16 REPL Deterministic Synthesis Lane `[Practical]`

- Suggested `phase_id`: `convergence`
- Core insight: expose synthesis as a visible, user-controlled operating mode.
- External inspiration or analogy: compiler front-end modes, mission control operating states
- Why it fits Saguaro and Anvil specifically: `cli/repl.py` already centralizes mode, commands, and startup posture.
- Exact wiring points: `cli/repl.py`, `cli/commands/**`, `core/unified_chat_loop.py`
- Existing primitives it can reuse: command registration, policy profiles, worksets, loop orchestration.
- New primitive, data flow, or subsystem needed: `deterministic_synthesis` mode, synthesis status panel, proof summary rendering.
- `repo_scope`: `cli/**`, `core/unified_chat_loop.py`, `Saguaro/saguaro/synthesis/**`
- `owning_specialist_type`: `platform_dx_architect`
- `allowed_writes`: `cli/repl.py`, `cli/commands/features.py`, `tests/test_deterministic_synthesis_lane.py`
- `telemetry_contract`: `synthesis_mode_invocations`, `user_abort_rate`, `proof_render_latency_ms`
- `required_evidence`: interactive runs with clear proof and rollback output
- `rollback_criteria`: user surface hides critical uncertainty or proof failures
- `promotion_gate`: a user can distinguish planning-only from deterministic synthesis runs
- `success_criteria`: the mode is explicit and operationally understandable
- Why this creates value: trust rises when the mode is visible
- Why this creates moat: operational clarity becomes part of the product
- Main risk or failure mode: confusing UX if too much internal machinery leaks
- Smallest credible first experiment: add a visible synthesis-lane badge and summary artifact rendering
- Confidence level: `0.87`

### 5.17 Governance-Coupled Synthesis Promotion `[Practical]`

- Suggested `phase_id`: `convergence`
- Core insight: deterministic synthesis outputs should only promote when proof, replay, and test evidence all exist.
- External inspiration or analogy: mission readiness review
- Why it fits Saguaro and Anvil specifically: pre-finalize governance checkpoints already exist in `core/unified_chat_loop.py`.
- Exact wiring points: `core/unified_chat_loop.py`, `domains/task_execution/enhanced_loop.py`, `core/campaign/roadmap_validator.py`
- Existing primitives it can reuse: finalize checkpoints, roadmap validator, event checkpoints.
- New primitive, data flow, or subsystem needed: `SynthesisPromotionGate`, `proof_missing` blockers, replay freshness checks.
- `repo_scope`: `core/unified_chat_loop.py`, `core/campaign/**`, `shared_kernel/**`
- `owning_specialist_type`: `fitness_function_engineer`
- `allowed_writes`: `core/unified_chat_loop.py`, `core/campaign/roadmap_validator.py`, `tests/test_synthesis_promotion_gate.py`
- `telemetry_contract`: `promotion_block_count`, `missing_proof_block_count`, `missing_replay_block_count`
- `required_evidence`: blocked promotions for seeded missing-proof and missing-test cases
- `rollback_criteria`: promotions are either too lax or too hard to achieve
- `promotion_gate`: only proof-carrying syntheses can clear the gate
- `success_criteria`: roadmap completion and synthesis completion align
- Why this creates value: prevents "green narrative, red reality"
- Why this creates moat: governance becomes machine-enforced
- Main risk or failure mode: bureaucratic drag if gates are not staged carefully
- Smallest credible first experiment: treat deterministic synthesis as a separate completion class in roadmap validation
- Confidence level: `0.89`

### 5.18 Proof-Carrying Patch Capsules `[Moonshot]`

- Suggested `phase_id`: `development`
- Core insight: every patch carries a compact proof capsule, replay tape, and capability digest.
- External inspiration or analogy: proof-carrying code, mission capsule exports
- Why it fits Saguaro and Anvil specifically: mission capsules and safety cases already exist in `shared_kernel/event_store.py`.
- Exact wiring points: `shared_kernel/event_store.py`, `Saguaro/saguaro/synthesis/**`, `core/campaign/roadmap_validator.py`
- Existing primitives it can reuse: replay hashes, mission capsules, safety-case nodes.
- New primitive, data flow, or subsystem needed: `proof_capsule.json`, `patch_capsule_validator`, witness graph extraction.
- `repo_scope`: `shared_kernel/**`, `Saguaro/saguaro/synthesis/**`, `core/campaign/**`
- `owning_specialist_type`: `translation_validation_prosecutor`
- `allowed_writes`: `shared_kernel/event_store.py`, `Saguaro/saguaro/synthesis/proof_capsule.py`, `tests/test_proof_capsules.py`
- `telemetry_contract`: `proof_capsule_emission_rate`, `witness_graph_size`, `capsule_replay_success_rate`
- `required_evidence`: capsules for at least math, adapter, and patch tasks
- `rollback_criteria`: capsules are too large or too weak to be useful
- `promotion_gate`: patch capsules can be validated offline without the model
- `success_criteria`: each patch is a transportable proof object
- Why this creates value: handoff and audit become drastically easier
- Why this creates moat: proof-carrying developer automation is rare
- Main risk or failure mode: witness size explosion
- Smallest credible first experiment: capsule only the spec, selected components, verification commands, and counterexamples
- Confidence level: `0.74`

### 5.19 Deterministic Math Kernel DSL Lane `[Moonshot]`

- Suggested `phase_id`: `development`
- Core insight: stop trying to synthesize general-purpose math code first; synthesize a bounded DSL for math kernels and lower it safely.
- External inspiration or analogy: hardware description languages, tensor DSLs, EDA flows
- Why it fits Saguaro and Anvil specifically: math-heavy domains are a stated target and a known LLM weakness.
- Exact wiring points: `core/agents/domain/science/scientific_computing_subagent.py`, `core/native/**`, `Saguaro/saguaro/synthesis/**`
- Existing primitives it can reuse: native capability ledgers, solver bridge, runtime telemetry.
- New primitive, data flow, or subsystem needed: `math_kernel_ir.py`, DSL parser, lowering passes, equivalence checker.
- `repo_scope`: `Saguaro/saguaro/synthesis/**`, `core/native/**`, `core/agents/domain/science/**`
- `owning_specialist_type`: `scientific_computing_synthesist`
- `allowed_writes`: `Saguaro/saguaro/synthesis/math_kernel_ir.py`, `tests/test_math_kernel_synthesis.py`
- `telemetry_contract`: `kernel_lowering_latency_ms`, `solver_proof_success_rate`, `vectorization_selection_rate`
- `required_evidence`: proof-backed synthesis of normalization, dot products, clamping, reductions, and adapters
- `rollback_criteria`: DSL is too narrow to matter or too broad to verify
- `promotion_gate`: bounded math tasks pass with proofs and reproducible lowerings
- `success_criteria`: math moves almost entirely out of model text generation
- Why this creates value: solves the hardest pain point first
- Why this creates moat: domain DSL plus proof plus runtime tuning is durable
- Main risk or failure mode: users resist a DSL if it feels like a separate language
- Smallest credible first experiment: lower a tiny numeric expression DSL to verified Python and C++ helpers
- Confidence level: `0.77`

### 5.20 Bounded Equality Saturation Optimizer `[Moonshot]`

- Suggested `phase_id`: `analysis_upgrade`
- Core insight: use local bounded eqsat after assembly, not global eqsat before assembly.
- External inspiration or analogy: `egg`, sketch-guided eqsat, colored e-graphs `[R7][R8]`
- Why it fits Saguaro and Anvil specifically: the repo already emits cost and capability signals needed to bound search.
- Exact wiring points: `Saguaro/saguaro/synthesis/**`, `core/native/runtime_telemetry.py`, `core/memory/fabric/policies.py`
- Existing primitives it can reuse: policy engine, counterexamples, proof capsules.
- New primitive, data flow, or subsystem needed: `eqsat_runner.py`, rewrite catalog, extraction cost model.
- `repo_scope`: `Saguaro/saguaro/synthesis/**`, `core/native/**`, `core/memory/fabric/**`
- `owning_specialist_type`: `optimizer_engineer`
- `allowed_writes`: `Saguaro/saguaro/synthesis/eqsat_runner.py`, `tests/test_eqsat_optimizer.py`
- `telemetry_contract`: `egraph_node_count`, `rewrite_fire_count`, `extract_cost_delta`, `memory_ceiling_hits`
- `required_evidence`: bounded optimization wins on numeric kernels or local expression rewrites
- `rollback_criteria`: e-graph growth violates memory or latency budgets
- `promotion_gate`: optimizer improves benchmarks without breaking proofs
- `success_criteria`: post-assembly canonicalization is deterministic and effective
- Why this creates value: extracts better local code from equivalent forms
- Why this creates moat: repo-specific rewrite catalogs compound
- Main risk or failure mode: memory explosion
- Smallest credible first experiment: eqsat only on expression trees smaller than a fixed node threshold
- Confidence level: `0.71`

### 5.21 Portfolio Search with Bounded MCTS Fallback `[Moonshot]`

- Suggested `phase_id`: `analysis_upgrade`
- Core insight: when direct assembly fails, run bounded search over AST growth decisions, guided by contracts and telemetry rather than token probabilities.
- External inspiration or analogy: planner search, bounded MCTS, solver-guided search
- Why it fits Saguaro and Anvil specifically: the repo already has specialist routing, runtime caps, and event replay.
- Exact wiring points: `core/orchestrator/**`, `domains/task_execution/enhanced_loop.py`, `Saguaro/saguaro/synthesis/**`
- Existing primitives it can reuse: subagents, route selection, memory tiers, event logging.
- New primitive, data flow, or subsystem needed: `portfolio_search.py`, `search_budget.py`, `candidate_merit_score`.
- `repo_scope`: `Saguaro/saguaro/synthesis/**`, `domains/task_execution/**`, `core/orchestrator/**`
- `owning_specialist_type`: `search_engineer`
- `allowed_writes`: `Saguaro/saguaro/synthesis/portfolio_search.py`, `tests/test_portfolio_search.py`
- `telemetry_contract`: `search_branch_count`, `search_budget_exhaustions`, `candidate_accept_rate`
- `required_evidence`: bounded fallback solves tasks that assembler-only cannot
- `rollback_criteria`: search degenerates into slow probabilistic guessing
- `promotion_gate`: fallback beats "ask the model to try again" on a curated hard set
- `success_criteria`: search is deterministic-enough, bounded, and proof-checked
- Why this creates value: provides a safety valve without returning to free-form generation
- Why this creates moat: proof-aware portfolio search tied to repo telemetry is uncommon
- Main risk or failure mode: hidden nondeterminism or state explosion
- Smallest credible first experiment: bounded top-down AST search over a tiny grammar with solver-backed pruning
- Confidence level: `0.68`

### 5.22 Holographic Program Lattice `[Moonshot]`

- Suggested `phase_id`: `research`
- Core insight: extend holographic embeddings from retrieval-only into compositional fragment representation and search steering.
- External inspiration or analogy: vector symbolic architectures, hyperdimensional computing
- Why it fits Saguaro and Anvil specifically: Saguaro already uses holographic vector operations in native code.
- Exact wiring points: `Saguaro/saguaro/native/ops/**`, `Saguaro/saguaro/indexing/**`, `Saguaro/saguaro/synthesis/**`
- Existing primitives it can reuse: holographic bundling, native embedding ops, graph/index state.
- New primitive, data flow, or subsystem needed: `program_lattice.py`, fragment bind/bundle operators, lattice-to-AST extraction hooks.
- `repo_scope`: `Saguaro/saguaro/native/ops/**`, `Saguaro/saguaro/synthesis/**`
- `owning_specialist_type`: `research_librarian_subagent`
- `allowed_writes`: `Saguaro/saguaro/synthesis/program_lattice.py`, `tests/test_program_lattice.py`
- `telemetry_contract`: `fragment_collision_rate`, `lattice_recall_at_k`, `composition_hint_accuracy`
- `required_evidence`: fragment composition improves component search or bounded search
- `rollback_criteria`: signal is too weak or too opaque to justify complexity
- `promotion_gate`: lattice hints improve real assembly or search outcomes
- `success_criteria`: embeddings become more than a retrieval sidecar
- Why this creates value: potentially collapses search breadth with compositional hints
- Why this creates moat: unusual combination of semantic indexing and deterministic synthesis
- Main risk or failure mode: elegant science experiment with little product leverage
- Smallest credible first experiment: use bundled fragment vectors only to rerank component candidates
- Confidence level: `0.56`

### 5.23 Executable Semantics Compiler `[Moonshot]`

- Suggested `phase_id`: `research`
- Core insight: compile grammar, contracts, and library semantics into machine-executable language packs rather than hand-maintaining them forever.
- External inspiration or analogy: K framework, Rosette interpreters, SemGuS `[R4][R5][R17]`
- Why it fits Saguaro and Anvil specifically: the repo already has parsers, queries, contracts, tests, and standards; it needs a mechanism to fuse them.
- Exact wiring points: `Saguaro/saguaro/parsing/**`, `Saguaro/saguaro/requirements/**`, `Saguaro/saguaro/synthesis/**`
- Existing primitives it can reuse: tree-sitter grammars, requirements extractor, test corpora, contracts.
- New primitive, data flow, or subsystem needed: `language_pack_compiler.py`, semantics stubs, contract merger, rule emitter.
- `repo_scope`: `Saguaro/saguaro/**`
- `owning_specialist_type`: `language_semantics_architect`
- `allowed_writes`: `Saguaro/saguaro/synthesis/language_pack_compiler.py`, `tests/test_language_pack_compiler.py`
- `telemetry_contract`: `compiled_rule_count`, `stub_resolution_rate`, `manual_override_count`
- `required_evidence`: generated packs that are correct enough to support bounded synthesis
- `rollback_criteria`: compiler emits misleading semantics without strong evidence
- `promotion_gate`: compiled packs beat purely hand-authored packs on coverage-to-effort ratio
- `success_criteria`: language support scale stops being purely manual
- Why this creates value: unlocks more languages without linear headcount
- Why this creates moat: automated semantics fusion is difficult and compounding
- Main risk or failure mode: false confidence from inferred semantics
- Smallest credible first experiment: compile operator tables and stdlib stubs, not full semantics
- Confidence level: `0.59`

### 5.24 Hardware-Aware Kernel Synthesis `[Moonshot]`

- Suggested `phase_id`: `convergence`
- Core insight: let deterministic synthesis choose code variants based on actual runtime ISA, affinity, and pressure data.
- External inspiration or analogy: F1 setup sheets, autotuners, codegen for architecture targets
- Why it fits Saguaro and Anvil specifically: the repo already tracks AMX, AVX2, queue wait, affinity, and degraded capabilities.
- Exact wiring points: `core/native/native_qsg_engine.py`, `core/native/runtime_telemetry.py`, `Saguaro/saguaro/synthesis/**`
- Existing primitives it can reuse: capability digests, hot-path proof metadata, backend selection reason fields.
- New primitive, data flow, or subsystem needed: `variant_selector.py`, kernel family catalog, hardware-conditioned proof witnesses.
- `repo_scope`: `core/native/**`, `Saguaro/saguaro/synthesis/**`, `benchmarks/**`
- `owning_specialist_type`: `pmu_pathologist`
- `allowed_writes`: `Saguaro/saguaro/synthesis/variant_selector.py`, `tests/test_hardware_aware_synthesis.py`
- `telemetry_contract`: `variant_selection_count`, `hardware_specific_speedup`, `unsafe_variant_block_count`
- `required_evidence`: variant choice justified by capability ledgers and benchmark deltas
- `rollback_criteria`: variant choice destabilizes correctness or portability
- `promotion_gate`: hardware-aware choice yields measurable wins without proof regressions
- `success_criteria`: synthesis exploits the repo's unusually deep runtime self-knowledge
- Why this creates value: turns telemetry into performance leverage
- Why this creates moat: hardware-aware deterministic synthesis is a rare combination
- Main risk or failure mode: premature optimization before the formal substrate is mature
- Smallest credible first experiment: choose among two equivalent numeric helper implementations based on AVX2 and degraded-mode state
- Confidence level: `0.65`

---

## 6. Critical Pressure Test

### 6.1 Elegant ideas that are likely wrong if attempted too early

- Full holographic program lattices before the repo has a stable synthesis IR.
- Global equality saturation across arbitrary Python and C++ programs.
- Automatic compilation of whole language manuals into executable semantics.
- A single universal cross-language IR for all future supported languages.

### 6.2 Ugly ideas that are strategically powerful

- Start with wrappers, adapters, semantic patches, and math helpers rather than grand general code synthesis.
- Use a small deterministic math DSL lane instead of claiming universal synthesis immediately.
- Enforce translation validation and proof capsules even if the initial developer UX is blunt.
- Force the planner to emit incomplete-spec errors instead of letting ambiguous requests drift into execution.

### 6.3 Ideas likely to fail because the repo is still missing a key primitive

- Any serious synthesis workflow that assumes a working AST constructor today.
- Any equivalence-preserving optimization flow without translation validation.
- Any multi-language expansion without language support packs.
- Any proof-carrying patch story without a normalized synthesis replay schema.

### 6.4 Pressure-tested view of the current repo

[Inference] The repo is already strong enough to support:

- a spec boundary
- a component assembler
- a solver bridge
- a replay and proof surface
- a deterministic synthesis lane in the REPL

[Inference] The repo is not yet strong enough to support:

- fully automatic general-purpose synthesis across open-ended tasks
- semantics-complete language support compiled from docs alone
- unconstrained optimizer search

---

## 7. Synthesis

### 7.1 Strongest ideas overall

1. `SagSpec IR Boundary` because it changes the authority boundary immediately.
2. `Language Support Packs` because it correctly answers what language support must become.
3. `Component Retriever and Assembler` because it is the fastest route to useful deterministic synthesis.
4. `Solver Bridge plus Translation Validation` because math and interface correctness are the core pain points.
5. `Synthesis Replay Tape and Counterexample Store` because learning and trust need the same substrate.

### 7.2 Best balance of novelty and plausibility

- `Language Support Packs`
- `Component Retriever and Assembler`
- `Solver Bridge`
- `Runtime-Aware Synthesis Policy`
- `Proof-Carrying Patch Capsules`

### 7.3 Most feasible now

`SagSpec IR Boundary` is the most feasible now because the relevant code paths are already centralized and compact:

- `core/prompts/prompt_manager.py`
- `core/prompts/system_prompt_builder.py`
- `core/agents/planner.py`
- `domains/task_execution/enhanced_loop.py`

### 7.4 Biggest long-term moat bet

`Language Support Packs` plus `Executable Semantics Compiler` is the biggest moat bet. That is the difference between "we prompt better" and "we own a real semantic compiler substrate."

### 7.5 Cleanest unifier with the current codebase

`Synthesis Replay Tape and Counterexample Store` is the cleanest unifier because it ties together:

- event store
- governance
- roadmap validation
- memory tiers
- benchmark harnesses
- solver counterexamples

### 7.6 What should be prototyped first

Prototype order by conviction:

1. `SagSpec IR Boundary`
2. `Language Support Packs`
3. `Component Retriever and Assembler`
4. `Solver Bridge`
5. `Synthesis Replay Tape and Counterexample Store`

---

## 8. Implementation Program

### 8.1 Phase `research` - Language Support Packs and Formal Substrate

- `phase_id`: `research`
- Phase title: `Language Support Packs and Formal Substrate`
- Objective: define deterministic language support packs for Python and C++ that turn syntax, contracts, and standard-library behavior into machine-usable synthesis substrate.
- Dependencies: none
- Repo scope: `Saguaro/saguaro/parsing/**`, `Saguaro/saguaro/analysis/**`, `Saguaro/saguaro/language_packs/**`, `docs/**`
- Owning specialist type: `language_semantics_architect`
- Allowed writes: `Saguaro/saguaro/language_packs/**`, `Saguaro/saguaro/parsing/**`, `tests/test_language_packs.py`, `tests/test_language_pack_compiler.py`
- Telemetry contract: `language_pack_coverage_pct`, `unsupported_construct_count`, `manual_override_count`
- Required evidence: coverage map for top Python and C++ constructs, emitted operator and contract tables, explicit unsupported set
- Rollback criteria: pack coverage below `85%` on target constructs or semantic ambiguity too high for safe synthesis
- Promotion gate: Python and C++ packs can drive bounded function-level synthesis tasks without prose docs in the loop
- Success criteria: language support is represented as deterministic packs, not free-form prompt context
- Exact wiring points: `Saguaro/saguaro/parsing/parser.py`, `Saguaro/saguaro/analysis/code_graph.py`, `Saguaro/saguaro/analysis/bridge_synthesizer.py`
- Deliverables:
  - `Saguaro/saguaro/language_packs/python_pack.py`
  - `Saguaro/saguaro/language_packs/cpp_pack.py`
  - `Saguaro/saguaro/synthesis/language_pack_compiler.py`
- Tests:
  - `tests/test_language_packs.py`
  - `tests/test_language_pack_compiler.py`
- Verification commands:
  - `pytest tests/test_language_packs.py tests/test_language_pack_compiler.py`
  - `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: language packs expose syntax families, operator semantics, library contracts, emission hooks, and unsupported constructs explicitly

### 8.2 Phase `eid` - SagSpec and Planner Lowering

- `phase_id`: `eid`
- Phase title: `SagSpec and Planner Lowering`
- Objective: replace markdown-first planning outputs with a typed synthesis specification and linting stage.
- Dependencies: `research`
- Repo scope: `core/agents/**`, `core/prompts/**`, `domains/task_execution/**`, `Saguaro/saguaro/synthesis/**`
- Owning specialist type: `formal_methods_architect`
- Allowed writes: `Saguaro/saguaro/synthesis/spec.py`, `core/agents/planner.py`, `core/agents/planner_agent.py`, `core/prompts/prompt_manager.py`, `domains/task_execution/enhanced_loop.py`, `tests/test_sagspec.py`, `tests/test_sagspec_lowering.py`
- Telemetry contract: `spec_validation_rate`, `spec_lint_failure_rate`, `clarification_required_rate`
- Required evidence: valid specs for bounded math, adapter, and semantic patch tasks plus deterministic failures for ambiguous tasks
- Rollback criteria: spec authoring becomes materially slower than current planning for bounded tasks
- Promotion gate: planner and loop can emit and consume validator-clean specs for at least `20` benchmark tasks
- Success criteria: plans are typed artifacts, not markdown that the system reparses
- Exact wiring points: `core/agents/planner.py`, `core/agents/planner_agent.py`, `core/prompts/system_prompt_builder.py`, `domains/task_execution/enhanced_loop.py`
- Deliverables:
  - `Saguaro/saguaro/synthesis/spec.py`
  - `Saguaro/saguaro/synthesis/spec_lint.py`
  - planner integration in `core/agents/planner.py`
- Tests:
  - `tests/test_sagspec.py`
  - `tests/test_sagspec_lowering.py`
  - `tests/test_strict_prompt_contracts.py`
- Verification commands:
  - `pytest tests/test_sagspec.py tests/test_sagspec_lowering.py tests/test_strict_prompt_contracts.py`
  - `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: synthesis begins only from a typed and linted spec

### 8.3 Phase `feature_map` - Contract and Component Inventory

- `phase_id`: `feature_map`
- Phase title: `Contract and Component Inventory`
- Objective: build a component and contract map that lets Saguaro assemble from local verified parts before attempting invention.
- Dependencies: `research`, `eid`
- Repo scope: `Saguaro/saguaro/query/**`, `Saguaro/saguaro/indexing/**`, `Saguaro/saguaro/requirements/**`, `Saguaro/saguaro/synthesis/**`
- Owning specialist type: `synthesis_engineer`
- Allowed writes: `Saguaro/saguaro/synthesis/assembler.py`, `Saguaro/saguaro/synthesis/contract_harvester.py`, `tests/test_component_assembler.py`, `tests/test_contract_harvester.py`
- Telemetry contract: `component_recall_at_k`, `symbol_contract_coverage`, `assembly_reuse_ratio`
- Required evidence: component recall report, contract coverage report, assembly success traces for bounded tasks
- Rollback criteria: contracts are too sparse or too noisy to support compatibility ranking
- Promotion gate: assembler solves a majority of bounded benchmark tasks through retrieval plus composition
- Success criteria: local components become the dominant first-choice synthesis substrate
- Exact wiring points: `Saguaro/saguaro/indexing/engine.py`, `Saguaro/saguaro/query/pipeline.py`, `Saguaro/saguaro/analysis/bridge_synthesizer.py`
- Deliverables:
  - `Saguaro/saguaro/synthesis/assembler.py`
  - `Saguaro/saguaro/synthesis/contract_harvester.py`
  - `Saguaro/saguaro/synthesis/component_catalog.py`
- Tests:
  - `tests/test_component_assembler.py`
  - `tests/test_contract_harvester.py`
  - `tests/test_saguaro_interface.py`
- Verification commands:
  - `pytest tests/test_component_assembler.py tests/test_contract_harvester.py tests/test_saguaro_interface.py`
  - `./venv/bin/saguaro impact --path domains/task_execution/enhanced_loop.py`
  - `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: assembly plans can cite retrieved components, contracts, and compatibility evidence

### 8.4 Phase `development` - SagAST, Adapter Generation, and Patch Induction

- `phase_id`: `development`
- Phase title: `SagAST, Adapter Generation, and Patch Induction`
- Objective: construct deterministic code artifacts through AST building, wrapper generation, and semantic patch replay.
- Dependencies: `research`, `eid`, `feature_map`
- Repo scope: `Saguaro/saguaro/synthesis/**`, `core/native/**`, `shared_kernel/**`
- Owning specialist type: `compiler_engineer`
- Allowed writes: `Saguaro/saguaro/synthesis/ast_builder.py`, `Saguaro/saguaro/synthesis/adapter_generator.py`, `Saguaro/saguaro/synthesis/patch_inductor.py`, `tests/test_ast_builder.py`, `tests/test_ffi_synthesis.py`, `tests/test_semantic_patch_induction.py`
- Telemetry contract: `emit_roundtrip_fidelity`, `adapter_generation_success_rate`, `rule_replay_success_rate`
- Required evidence: round-trip emission suite, successful adapter synthesis traces, semantic rewrite replay traces
- Rollback criteria: emitted code is unstable or wrapper generation creates unsafe ABI mismatches
- Promotion gate: generated artifacts pass syntax, runtime, and semantic replay checks in sandbox
- Success criteria: the system can deterministically build and emit bounded source artifacts
- Exact wiring points: `Saguaro/saguaro/parsing/parser.py`, `Saguaro/saguaro/analysis/bridge_synthesizer.py`, `core/native/native_qsg_engine.py`
- Deliverables:
  - `Saguaro/saguaro/synthesis/ast_builder.py`
  - `Saguaro/saguaro/synthesis/adapter_generator.py`
  - `Saguaro/saguaro/synthesis/patch_inductor.py`
- Tests:
  - `tests/test_ast_builder.py`
  - `tests/test_ffi_synthesis.py`
  - `tests/test_semantic_patch_induction.py`
- Verification commands:
  - `pytest tests/test_ast_builder.py tests/test_ffi_synthesis.py tests/test_semantic_patch_induction.py`
  - `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: bounded source artifacts can be built without the LLM writing final code text

### 8.5 Phase `analysis_upgrade` - Solver Bridge, Translation Validation, and Bounded Eqsat

- `phase_id`: `analysis_upgrade`
- Phase title: `Solver Bridge, Translation Validation, and Bounded Eqsat`
- Objective: move correctness and optimization from heuristic post-hoc checks into proof-carrying construction and equivalence validation.
- Dependencies: `research`, `eid`, `feature_map`, `development`
- Repo scope: `Saguaro/saguaro/synthesis/**`, `Saguaro/saguaro/verification/**`, `core/native/**`, `core/memory/fabric/**`
- Owning specialist type: `verification_engineer`
- Allowed writes: `Saguaro/saguaro/synthesis/solver.py`, `Saguaro/saguaro/synthesis/translation_validator.py`, `Saguaro/saguaro/synthesis/eqsat_runner.py`, `tests/test_solver.py`, `tests/test_translation_validation.py`, `tests/test_eqsat_optimizer.py`
- Telemetry contract: `solver_timeout_rate`, `translation_validation_pass_rate`, `egraph_node_count`, `extract_cost_delta`
- Required evidence: solver proofs, seeded failing transforms, bounded optimization wins, counterexample exports
- Rollback criteria: solver and validation overhead destroy operational viability for bounded tasks
- Promotion gate: optimizations only promote when proofs and validation witnesses exist
- Success criteria: correctness is enforced during synthesis, not only after it
- Exact wiring points: `Saguaro/saguaro/verification/contracts.py`, `core/native/runtime_telemetry.py`, `core/native/native_qsg_engine.py`, `core/memory/fabric/policies.py`
- Deliverables:
  - `Saguaro/saguaro/synthesis/solver.py`
  - `Saguaro/saguaro/synthesis/translation_validator.py`
  - `Saguaro/saguaro/synthesis/eqsat_runner.py`
- Tests:
  - `tests/test_solver.py`
  - `tests/test_translation_validation.py`
  - `tests/test_eqsat_optimizer.py`
  - `tests/test_qsg_runtime_contracts.py`
- Verification commands:
  - `pytest tests/test_solver.py tests/test_translation_validation.py tests/test_eqsat_optimizer.py tests/test_qsg_runtime_contracts.py`
  - `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: proof and equivalence evidence are first-class blocking requirements

### 8.6 Phase `deep_test_audit` - Replay Tape, Counterexamples, and Benchmarks

- `phase_id`: `deep_test_audit`
- Phase title: `Replay Tape, Counterexamples, and Benchmarks`
- Objective: make synthesis inspectable, replayable, and benchmark-governed.
- Dependencies: `eid`, `feature_map`, `development`, `analysis_upgrade`
- Repo scope: `shared_kernel/**`, `benchmarks/**`, `audit/**`, `Saguaro/saguaro/synthesis/**`
- Owning specialist type: `observability_contract_engineer`
- Allowed writes: `shared_kernel/event_store.py`, `benchmarks/synthesis_suite.py`, `Saguaro/saguaro/synthesis/replay_tape.py`, `tests/test_synthesis_replay_tape.py`, `tests/test_synthesis_benchmarks.py`
- Telemetry contract: `replay_export_count`, `counterexample_reuse_rate`, `proof_pass_rate`, `benchmark_runtime_ms`
- Required evidence: replay tapes, mission capsules, proof capsules, and benchmark scorecards
- Rollback criteria: replay cannot reconstruct decision state or benchmark suite produces unstable signals
- Promotion gate: roadmap promotions require benchmark and replay evidence
- Success criteria: the system can explain and replay why a synthesis passed or failed
- Exact wiring points: `shared_kernel/event_store.py`, `tests/test_saguaro_roadmap_validator.py`, `tests/test_campaign_roadmap_phase_pack.py`
- Deliverables:
  - `Saguaro/saguaro/synthesis/replay_tape.py`
  - `benchmarks/synthesis_suite.py`
  - proof capsule support in `shared_kernel/event_store.py`
- Tests:
  - `tests/test_synthesis_replay_tape.py`
  - `tests/test_synthesis_benchmarks.py`
  - `tests/test_saguaro_roadmap_validator.py`
- Verification commands:
  - `pytest tests/test_synthesis_replay_tape.py tests/test_synthesis_benchmarks.py tests/test_saguaro_roadmap_validator.py`
  - `./venv/bin/saguaro roadmap validate --path ROADMAP_PROGRAM_SYNTHESIS.md --format json`
  - `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: every promoted synthesis capability has replay and benchmark evidence

### 8.7 Phase `convergence` - REPL Lane and Governance-Coupled Operation

- `phase_id`: `convergence`
- Phase title: `REPL Lane and Governance-Coupled Operation`
- Objective: expose deterministic synthesis as a first-class operating mode and bind promotion to proof, replay, and verification evidence.
- Dependencies: `deep_test_audit`
- Repo scope: `cli/**`, `core/unified_chat_loop.py`, `domains/task_execution/**`, `core/campaign/**`
- Owning specialist type: `platform_dx_architect`
- Allowed writes: `cli/repl.py`, `core/unified_chat_loop.py`, `domains/task_execution/enhanced_loop.py`, `core/campaign/roadmap_validator.py`, `tests/test_deterministic_synthesis_lane.py`, `tests/test_synthesis_promotion_gate.py`
- Telemetry contract: `synthesis_mode_invocations`, `promotion_block_count`, `proof_missing_block_count`
- Required evidence: user-facing lane demos, blocked promotion cases, successful promoted runs with complete evidence bundles
- Rollback criteria: the lane confuses users or hides failure causes
- Promotion gate: deterministic synthesis outputs only clear the finish line when proof, replay, tests, and roadmap evidence are present
- Success criteria: the feature is visible, trustworthy, and operationally governed
- Exact wiring points: `cli/repl.py`, `core/unified_chat_loop.py`, `domains/task_execution/enhanced_loop.py`, `core/campaign/roadmap_validator.py`
- Deliverables:
  - REPL synthesis lane in `cli/repl.py`
  - promotion logic in `core/unified_chat_loop.py`
  - roadmap-aware gating in `core/campaign/roadmap_validator.py`
- Tests:
  - `tests/test_deterministic_synthesis_lane.py`
  - `tests/test_synthesis_promotion_gate.py`
  - `tests/test_entrypoints.py`
- Verification commands:
  - `pytest tests/test_deterministic_synthesis_lane.py tests/test_synthesis_promotion_gate.py tests/test_entrypoints.py`
  - `./venv/bin/saguaro roadmap validate --path ROADMAP_PROGRAM_SYNTHESIS.md --format json`
  - `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: users can deliberately run deterministic synthesis and see exactly why promotion did or did not occur

---

## 9. Implementation Contract

- The system shall implement typed synthesis specifications through `Saguaro/saguaro/synthesis/spec.py`, `core/agents/planner.py`, and `domains/task_execution/enhanced_loop.py`, tested by `tests/test_sagspec.py` and `tests/test_sagspec_lowering.py`, and verified with `pytest tests/test_sagspec.py tests/test_sagspec_lowering.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall implement deterministic language support packs through `Saguaro/saguaro/language_packs/python_pack.py`, `Saguaro/saguaro/language_packs/cpp_pack.py`, and `Saguaro/saguaro/synthesis/language_pack_compiler.py`, tested by `tests/test_language_packs.py` and `tests/test_language_pack_compiler.py`, and verified with `pytest tests/test_language_packs.py tests/test_language_pack_compiler.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall implement component and contract assembly through `Saguaro/saguaro/synthesis/assembler.py`, `Saguaro/saguaro/synthesis/contract_harvester.py`, and `Saguaro/saguaro/indexing/engine.py`, tested by `tests/test_component_assembler.py`, `tests/test_contract_harvester.py`, and `tests/test_saguaro_interface.py`, and verified with `pytest tests/test_component_assembler.py tests/test_contract_harvester.py tests/test_saguaro_interface.py` and `./venv/bin/saguaro impact --path domains/task_execution/enhanced_loop.py`.
- The system shall implement syntax-safe AST construction and emission through `Saguaro/saguaro/synthesis/ast_builder.py` and `Saguaro/saguaro/parsing/parser.py`, tested by `tests/test_ast_builder.py`, and verified with `pytest tests/test_ast_builder.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall implement deterministic adapter generation across the Python and native boundary through `Saguaro/saguaro/synthesis/adapter_generator.py`, `core/native/native_qsg_engine.py`, and `core/native/runtime_telemetry.py`, tested by `tests/test_ffi_synthesis.py` and `tests/test_qsg_runtime_contracts.py`, and verified with `pytest tests/test_ffi_synthesis.py tests/test_qsg_runtime_contracts.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall implement solver-backed proof checking through `Saguaro/saguaro/synthesis/solver.py`, `Saguaro/saguaro/verification/contracts.py`, and `Saguaro/saguaro/analysis/dfg_builder.py`, tested by `tests/test_solver.py` and `tests/test_qsg_runtime_contracts.py`, and verified with `pytest tests/test_solver.py tests/test_qsg_runtime_contracts.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall implement translation validation through `Saguaro/saguaro/synthesis/translation_validator.py`, `core/native/native_qsg_engine.py`, and `Saguaro/saguaro/analysis/bridge_synthesizer.py`, tested by `tests/test_translation_validation.py`, and verified with `pytest tests/test_translation_validation.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall implement bounded equality-saturation optimization through `Saguaro/saguaro/synthesis/eqsat_runner.py`, `core/native/runtime_telemetry.py`, and `core/memory/fabric/policies.py`, tested by `tests/test_eqsat_optimizer.py`, and verified with `pytest tests/test_eqsat_optimizer.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall implement synthesis replay tapes and proof capsules through `Saguaro/saguaro/synthesis/replay_tape.py`, `shared_kernel/event_store.py`, and `domains/task_execution/enhanced_loop.py`, tested by `tests/test_synthesis_replay_tape.py` and `tests/test_campaign_timeline_and_verification_lane.py`, and verified with `pytest tests/test_synthesis_replay_tape.py tests/test_campaign_timeline_and_verification_lane.py` and `./venv/bin/saguaro roadmap validate --path ROADMAP_PROGRAM_SYNTHESIS.md --format json`.
- The system shall implement synthesis benchmark governance through `benchmarks/synthesis_suite.py`, `core/campaign/roadmap_validator.py`, and `audit/runner/benchmark_suite.py`, tested by `tests/test_synthesis_benchmarks.py`, `tests/test_saguaro_roadmap_validator.py`, and `tests/test_perf_roadmap_phases.py`, and verified with `pytest tests/test_synthesis_benchmarks.py tests/test_saguaro_roadmap_validator.py tests/test_perf_roadmap_phases.py` and `./venv/bin/saguaro roadmap validate --path ROADMAP_PROGRAM_SYNTHESIS.md --format json`.
- The system shall expose a deterministic synthesis operating lane through `cli/repl.py`, `core/unified_chat_loop.py`, and `domains/task_execution/enhanced_loop.py`, tested by `tests/test_deterministic_synthesis_lane.py`, `tests/test_synthesis_promotion_gate.py`, and `tests/test_entrypoints.py`, and verified with `pytest tests/test_deterministic_synthesis_lane.py tests/test_synthesis_promotion_gate.py tests/test_entrypoints.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.

---

## 10. References

- `[R1]` Z3 Guide. https://microsoft.github.io/z3guide/
- `[R2]` cvc5 project and SMT plus SyGuS support. https://cvc5.github.io/
- `[R3]` Syntax-Guided Synthesis competition. https://sygus.org/
- `[R4]` Semantics-Guided Synthesis community site. https://www.semgus.org/
- `[R5]` Rosette solver-aided programming. https://emina.github.io/rosette/
- `[R6]` Solar-Lezama, *Program Synthesis by Sketching*. https://people.csail.mit.edu/asolar/papers/thesis.pdf
- `[R7]` `egg` equality saturation library. https://github.com/egraphs-good/egg
- `[R8]` *Sketch-Guided Equality Saturation*. https://arxiv.org/abs/2111.13040
- `[R9]` Alive2 translation validation. https://github.com/AliveToolkit/alive2
- `[R10]` Tree-sitter book. https://tree-sitter.github.io/tree-sitter/
- `[R11]` Tree-sitter code navigation. https://tree-sitter.github.io/tree-sitter/4-code-navigation.html
- `[R12]` Semgrep taint mode overview. https://semgrep.dev/docs/writing-rules/data-flow/taint-mode/overview
- `[R13]` Semgrep engineering blog on taint mode. https://semgrep.dev/blog/2022/demystifying-taint-mode/
- `[R14]` Coccinelle semantic patches. https://coccinelle.gitlabpages.inria.fr/website/
- `[R15]` Comby structural search and replace. https://comby.dev/
- `[R16]` Dafny. https://dafny.org/
- `[R17]` K framework. https://kframework.org/
- `[R18]` Hacker News program synthesis discussion and two-solver intuition. https://news.ycombinator.com/item?id=20779535
- `[R19]` Hacker News discussion on Tree-sitter limits and need for analysis above parsing. https://news.ycombinator.com/item?id=26225298
- `[R20]` Hacker News discussion on equality saturation tradeoffs. https://news.ycombinator.com/item?id=29775477
- `[R21]` LLVM discussion on translation validation in practice. https://discourse.llvm.org/t/blog-post-about-basic-translation-validation-for-aarch64-backend/62988
- `[R22]` GitHub engineering on Semantic language support. https://github.blog/2020-08-04-codegen-semantics-improved-language-support-system/
- `[R23]` GitHub engineering on structured high-performance code views. https://github.blog/engineering/architecture-optimization/crafting-a-better-faster-code-view/
- `[R24]` cvc5 SyGuS grammar examples. https://cvc5.github.io/docs/cvc5-1.0.0/examples/sygus-grammar.html
- `[R25]` cvc5 SyGuS function examples. https://cvc5.github.io/docs/cvc5-1.1.0/examples/sygus-fun.html
- `[R26]` cvc5 SyGuS invariant examples. https://cvc5.github.io/docs/cvc5-1.0.0/examples/sygus-inv.html

---

## 11. Bottom Line

[Inference] Yes, it is believable that Saguaro could eventually write significant classes of code without an LLM doing the final writing. The credible path is not "bigger embeddings" and not "better prompt engineering." The credible path is:

1. formal spec boundary
2. executable language support packs
3. component assembly first
4. AST-safe construction
5. solver-backed proof
6. translation validation
7. replayable evidence and governance

[Inference] If that path is followed, the LLM becomes what it should have been in a high-assurance system all along: a planning and interpretation layer, not the entity trusted to freehand the final source of truth.
