# Saguaro Deep Analysis, Comparative Repo Intelligence, and Native Migration Roadmap

> Generated: 2026-03-13  
> Workspace: `/home/mike/Documents/Github/Anvil`  
> Standard: NASA and Formula 1 grade systems engineering  
> Intent: upgrade Saguaro from a strong single-repo semantic index into a multi-corpus comparative analysis system that can ingest external repositories, focalize on subtrees, reason about function internals, surface disparate relations, and feed evidence directly into Anvil's R&D loop.

---

## 1. First-Principles Framing

### What the system fundamentally is

Saguaro is not just a code search tool. It is a repository operating substrate that turns source trees into a machine-usable world model.

At its core, the existing system is already composed of four layers:

- `[Repo]` A corpus ingestion layer that discovers files, parses languages, embeds entities, and stores an index in `.saguaro/`, primarily through `Saguaro/saguaro/api.py`, `Saguaro/saguaro/indexing/engine.py`, `Saguaro/saguaro/parsing/parser.py`, and `Saguaro/saguaro/storage/*`.
- `[Repo]` A graph layer that already knows more than lexical search: CFG, DFG, impact, liveness, bridge synthesis, topology, and runtime symbol resolution exist in `Saguaro/saguaro/analysis/*`, `Saguaro/saguaro/services/platform.py`, and `Saguaro/saguaro/parsing/runtime_symbols.py`.
- `[Repo]` A workflow layer that already has campaign, research, EID, repo acquisition, repo caching, and governance primitives in `core/campaign/*`, `core/research/*`, and `core/dare/*`.
- `[Repo]` A native acceleration layer, with live C++ targets and native bindings in `Saguaro/src/ops/*`, `Saguaro/saguaro/indexing/native_indexer_bindings.py`, `Saguaro/saguaro/storage/native_vector_store.py`, and `Saguaro/CMakeLists.txt`.

The system is therefore best understood as a code intelligence operating system with an under-realized comparative analysis capability.

### What the system appears to be trying to become

The architecture is already leaning toward a larger role:

- `[Repo]` `core/campaign/control_plane.py` already acquires external repositories, registers roles, builds analysis packs, stores repo dossiers, and feeds them into `run_research()` and `run_eid()`.
- `[Repo]` `core/campaign/repo_cache.py` and `core/campaign/repo_registry.py` already treat external repositories as controlled assets, not just loose folders.
- `[Repo]` `Saguaro/saguaro/query/gateway.py` already introduces a resident runtime and bounded admission control, which is the correct shape for multi-corpus routing.
- `[Repo]` `Saguaro/saguaro/api.py` already accepts `path=` in `index()`, which means subtree focus exists in embryo, but not yet as a first-class corpus.

From first principles, the system wants to become a comparative repository intelligence engine that can:

- isolate multiple corpora without contaminating the primary index,
- run deep analysis on a subtree or a whole external repository,
- compare foreign repos against the indexed target repo,
- determine what should be copied, rewritten natively, wrapped, or ignored,
- convert those findings into evidence-backed R&D and implementation workstreams.

### Core constraints that actually matter

These constraints are real; everything else is secondary.

- Index isolation is mandatory. The main target index cannot be polluted by `repo_analysis/**`.
- Corpus identity must be explicit. Repo, subtree, snapshot, build profile, and language coverage must become addressable state.
- Function-level fidelity matters more than file-level breadth. Without trustworthy symbol, CFG, DFG, and relation identity, disparate-relation analysis is noise.
- Comparative analysis must be evidence-producing, not opinion-producing.
- The operator model must support three modes: target-only, one external corpus, and fleet mode across `repo_analysis/`.
- Compute-heavy relation scoring, graph alignment, and similarity kernels should land in native code where latency, RAM, or throughput pressure justifies it.
- EID integration must consume structured comparative evidence rather than shallow markdown dossiers.
- Verification and rollback must remain first-class; cleverness without hard gates is not acceptable.

### Engineering standard for the design space

- `[Inference]` The correct benchmark is not "more features." It is "can this behave like a pit wall plus flight rules for repository modernization?"
- `[Inference]` Formula 1 analogy: live telemetry, lap-by-lap deltas, controlled experiments, and bounded intervention windows.
- `[Inference]` NASA analogy: explicit mission phase gates, evidence closure, anomaly logging, and rollback criteria before promotion.
- `[Inference]` The design target is a system that predicts, compares, simulates, and governs repository upgrades with the same rigor it currently applies to query and verification.

---

## 2. External Research Scan

### High-signal patterns from external systems and literature

1. `[External]` Sourcegraph's precise code navigation and SCIP ecosystem treat index precision and repository identity as first-class operational concerns, with language-specific indexers feeding a common graph format. That matters here because Saguaro needs isolated, language-aware corpora rather than one monolithic store.  
   Source: Sourcegraph precise code navigation docs, `https://sourcegraph.com/docs/code-search/code-navigation/precise_code_navigation`  
   Source: Sourcegraph indexers docs, `https://sourcegraph.com/docs/code-intelligence/references/indexers`

2. `[External]` `scip-clang` shows the correct posture for C and C++ precision: reuse the build system, capture real compile flags, and generate symbol identity from the true build graph instead of from heuristic parsing alone. That maps directly to Saguaro's native-heavy C++ ambitions.  
   Source: `https://github.com/sourcegraph/scip-clang`

3. `[External]` Kythe's `VName` schema makes `corpus`, `root`, `path`, `language`, and `signature` explicit identity fields. This is the cleanest external model for the missing Saguaro primitive: corpus-qualified symbol identity.  
   Source: `https://kythe.io/docs/schema/`

4. `[External]` Tree-sitter's main contribution is not just broad language support. It is incremental, resilient parsing under partial code quality and local edits. That is exactly what subtree-scoped and ephemeral corpus indexing need in practice.  
   Source: `https://tree-sitter.github.io/tree-sitter/`

5. `[External]` CodeQL's database model is the cleanest practical precedent for isolated analysis assets. It creates separate databases per source tree, supports build-aware extraction, and now supports multi-repository variant analysis at scale. That is directly analogous to the user's requested temporary repo indexes and cross-repo analysis.  
   Source: CodeQL database creation docs, `https://docs.github.com/code-security/codeql-cli/creating-codeql-databases`  
   Source: CodeQL multi-repository variant analysis docs, `https://docs.github.com/en/code-security/concepts/code-scanning/multi-repository-variant-analysis`

6. `[External]` CodeQL's compiled-language guidance is especially relevant because it formalizes build modes. Some languages can tolerate `none`; others need autobuild or manual build tracing. Saguaro needs the same operational distinction if it wants trustworthy cross-language depth.  
   Source: `https://docs.github.com/en/code-security/concepts/code-scanning/codeql/about-codeql-code-scanning-for-compiled-languages`

7. `[External]` Joern and the broader Code Property Graph work matter because they fuse AST, CFG, and DFG into one queryable representation. Saguaro already has pieces of that model; the gap is not conceptual possibility but cross-corpus unification and operator surfaces.  
   Source: Joern repo, `https://github.com/joernio/joern`  
   Source: Code Property Graph specification, `https://cpg.joern.io/`

8. `[External]` Semgrep's interfile analysis documentation is a useful cautionary operational reference. Deeper cross-file reasoning is valuable, but it costs RAM and wall-clock time. Saguaro should therefore make relation depth a budgeted mode, not a default everywhere.  
   Source: `https://semgrep.dev/docs/semgrep-code/semgrep-pro-engine-intro`

9. `[External]` OpenRewrite proves that large-scale modernization succeeds when migrations are encoded as explicit recipes rather than left as free-form model improvisation. The important lesson here is recipe-backed transformation planning, not Java-specific tooling.  
   Source: OpenRewrite recipes docs, `https://docs.openrewrite.org/concepts-and-explanations/recipes`  
   Source: Recipe catalog, `https://docs.openrewrite.org/recipes`

10. `[External]` Uber's Piranha and Polyglot Piranha work show that industrial migration systems win when they automate repetitive cleanup with explicit semantics and auditability. Saguaro should learn from the copy-vs-rewrite decision discipline and recipe generation model.  
    Source: Uber Piranha blog, `https://www.uber.com/en-NG/blog/piranha/`  
    Source: Lightweight Polyglot Code Transformation Language, `https://danieltrt.github.io/papers/pldi24.pdf`

11. `[External]` Meta's Getafix is relevant because it learns structured fixes from historical patterns and proposes them for human review instead of silently mutating code. The right Saguaro parallel is a native rewrite recommender and migration recipe synthesizer, not an unchecked auto-porter.  
    Source: `https://engineering.fb.com/2018/11/06/developer-tools/getafix-how-facebook-tools-learn-to-fix-bugs-automatically/`

12. `[External]` RepoHyper matters because it demonstrates repository-level retrieval based on a semantic graph with search-expand-refine rather than flat chunk retrieval. That is directly applicable to comparative analysis and report generation.  
    Source: `https://arxiv.org/abs/2403.06095`

13. `[External]` RepoMod-Bench matters because it demonstrates how quickly repository-level modernization performance collapses as repository size rises. That is an argument for mission-phase decomposition, stronger evidence gates, and implementation-agnostic verification.  
    Source: `https://arxiv.org/abs/2602.22518`

14. `[External]` FreshBrew matters because it treats migration quality as semantics preservation under hidden tests and high coverage, not just patch plausibility. That is exactly how Saguaro should evaluate comparative repo port recommendations.  
    Source: `https://arxiv.org/abs/2510.04852`

15. `[External]` TRANSAGENT is useful less for translation novelty and more for its staged fault-localize/fix/align structure. Saguaro's EID loop should borrow this decomposition for native rewrite recommendations: localize, align, rewrite, verify.  
    Source: `https://arxiv.org/abs/2409.19894`

### Practitioner and operational takeaways

- `[External]` CodeQL MRVA shows that multi-repo analysis is tractable when controller state, analysis assets, and result routing are explicit. Saguaro needs a controller corpus, not ad hoc folder scans.
- `[External]` Semgrep's interfile posture is a reminder that deep analysis must be budget-aware, preferably operator-tunable by mode: fast scan, deep scan, or mission scan.
- `[External]` OpenRewrite and Piranha show that migration systems become durable when they encode recipes, evidence, and audit trails.
- `[External]` RepoMod-Bench and FreshBrew show that repository modernization is still hard at realistic scale. That argues against magical end-to-end autonomy and for explicit phase gates.

### Cross-industry analogies that actually reveal mechanism

- `[Inference]` Pit wall model: every external corpus is a car entering the garage. You do not bolt it straight onto the race car. You inspect, instrument, compare, and only then decide whether to transplant a component, adapt it, or reject it.
- `[Inference]` Digital twin model: Saguaro should produce a repo twin for every external repository or scoped subtree, then run relation alignment and migration scoring against the target repo.
- `[Inference]` Flight rules model: the comparative pipeline needs explicit go/no-go criteria for copy, wrap, rewrite, and reject. These should not remain implicit in prompt text.

### Why the external scan changes the design space

- `[Synthesis]` The missing center of gravity is not "better search." It is a corpus-aware comparative intelligence runtime.
- `[Synthesis]` The repo already contains many of the right primitives: registry, cache, query gateway, bridge synthesis, graph services, roadmap validator, and EID control flow.
- `[Synthesis]` What is missing is identity, depth, and orchestration between them.

---

## 3. Repo Grounding Summary

### Commands executed from the repo virtual environment

All repo analysis was run from the repo venv with `source venv/bin/activate`.

Commands executed:

- `./venv/bin/saguaro health`
- `./venv/bin/saguaro entrypoints`
- `./venv/bin/saguaro build-graph`
- `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- `./venv/bin/saguaro query "temporary repo indexing and workspace scope" --k 5`
- `./venv/bin/saguaro query "campaign EID R&D loop repository analysis" --k 5`
- `./venv/bin/saguaro query "disparate relations cross repo similarity analysis" --k 5`
- `./venv/bin/saguaro query "prompt assembly orchestration planning hooks memory telemetry" --k 5`
- `./venv/bin/saguaro agent skeleton core/campaign/control_plane.py`
- `./venv/bin/saguaro agent skeleton core/campaign/base_campaign.py`
- `./venv/bin/saguaro agent skeleton core/research/eid_master.py`
- `./venv/bin/saguaro agent skeleton core/research/repo_acquisition.py`
- `./venv/bin/saguaro agent skeleton Saguaro/saguaro/query/gateway.py`
- `./venv/bin/saguaro agent skeleton Saguaro/saguaro/analysis/report.py`
- `./venv/bin/saguaro roadmap validate --path saguaro_deep_analysis_roadmap.md --format json`

### Relevant live findings

- `[Repo]` `./venv/bin/saguaro health` currently fails in this workspace because `Saguaro/saguaro/fastpath.py` references `collect_native_compute_report`, which is undefined. The observable failure is a `NameError`. This is a real health-path regression, not a hypothetical concern.
- `[Repo]` `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json` passes, with graph and parser coverage already high enough to justify deeper expansion rather than foundational replacement.
- `[Repo]` `./venv/bin/saguaro entrypoints` reports 42 entry points, including `Saguaro/saguaro/cli.py`, `Saguaro/saguaro/dni/server.py`, `Saguaro/saguaro/mcp/server.py`, `main.py`, `anvil.py`, and `cli/repl.py`.
- `[Repo]` `./venv/bin/saguaro build-graph` reports live native targets such as `cmake:saguaro_core`, `cmake:saguaro_native`, `cmake:anvil_native_ops`, and `cmake:anvil_runtime_core`.
- `[Repo]` A symbol slice request for `SaguaroAPI.index` mis-resolved into a symbol under `repo_analysis/` rather than the intended API symbol, which is a clean symptom of missing corpus-qualified symbol identity.

### Code paths inspected and why they matter

#### Comparative repo intake and lifecycle

- `[Repo]` `core/campaign/repo_cache.py`
  - Already snapshots or clones external repositories into controlled cache locations.
  - Strong primitive for immutable analysis corpora.

- `[Repo]` `core/campaign/repo_registry.py`
  - Already tracks repo roles such as target and analysis roles.
  - Strong primitive for corpus registration, write policy, and role separation.

- `[Repo]` `core/research/repo_acquisition.py`
  - Already acquires local and remote repos and immediately builds analysis packs.
  - Good intake shell, shallow analysis core.

- `[Repo]` `core/research/analysis_pack.py`
  - Builds normalized repo analysis packs, but by static file cataloging, token indexing, lightweight symbol extraction, and optional per-file impact calls.
  - This is currently too shallow for native migration intelligence or full comparative reports.

- `[Repo]` `core/dare/repo_ingestion.py`
  - Already performs single and cross-repo ingestion.
  - `_cross_repo_analysis()` currently only intersects tech stacks and aggregates language counts, then writes markdown into the knowledge base.
  - This is the clearest underpowered comparative analysis path in the repo.

#### Campaign, research, and EID wiring

- `[Repo]` `core/campaign/control_plane.py`
  - Central orchestrator with `acquire_repos()`, `run_research()`, `run_eid()`, `build_feature_map()`, and `build_roadmap()`.
  - Exact place where comparative repo intelligence must be formalized.

- `[Repo]` `core/research/eid_master.py`
  - `EIDMasterLoop._repo_dossier_actions()` currently converts repo dossiers into a trivial "evaluate top candidate path first" action.
  - This is far below the intended ambition.

- `[Repo]` `core/unified_chat_loop.py`
  - Existing repo-analysis UX still loads a small set of candidate files and text blobs.
  - Useful fallback, but not the final operator surface for comparative intelligence.

#### Saguaro API, query, and state

- `[Repo]` `Saguaro/saguaro/api.py`
  - `index(path=...)` already accepts subpaths.
  - `workspace()` manages workspace synchronization through `StateLedger`.
  - `query()` already supports scopes such as `local`, `workspace`, `peer`, and `global`, but this is retrieval scope, not isolated corpus scope.
  - The system has scope semantics without corpus semantics.

- `[Repo]` `Saguaro/saguaro/query/gateway.py`
  - Already implements a resident `QueryGateway`, a `SessionGovernor`, admission control, state reporting, and runtime prewarming.
  - This is the natural home for federated multi-corpus routing.

- `[Repo]` `Saguaro/saguaro/state/ledger.py`
  - Existing ledger substrate is a strong place to attach corpus session manifests, TTLs, provenance, and temporary index lifecycle state.

#### Parsing and graph depth

- `[Repo]` `Saguaro/saguaro/parsing/parser.py`
  - Broad language detection already exists.
  - Python gets AST handling; many other languages depend on tree-sitter or lighter parsing flows.
  - Depth is uneven; breadth is already good.

- `[Repo]` `Saguaro/saguaro/parsing/runtime_symbols.py`
  - Runtime symbol manifest and resolution exist.
  - There is no explicit corpus-qualified identity layer.

- `[Repo]` `Saguaro/saguaro/agents/perception.py`
  - Symbol resolution and slice generation exist.
  - Current ambiguity handling is insufficient for cross-repo symbol collisions.

- `[Repo]` `Saguaro/saguaro/analysis/code_graph.py`
  - Powerful graph substrate already exists.
  - It is a promising base for comparative relation overlays.

- `[Repo]` `Saguaro/saguaro/analysis/icfg_builder.py`
  - Interprocedural graph builder exists.
  - This is one of the right places to deepen cross-function and cross-corpus semantics.

- `[Repo]` `Saguaro/saguaro/analysis/bridge_synthesizer.py`
  - The most underexploited primitive.
  - It already thinks in relation edges across boundaries. That should be generalized to cross-corpus disparate relations.

- `[Repo]` `Saguaro/saguaro/services/platform.py`
  - Rebuilds bridge edges and maintains graph metrics.
  - Likely central to adding omnigraph-style cross-corpus relation layers.

#### Governance and exclusions

- `[Repo]` `Saguaro/saguaro/query/corpus_rules.py`
  - Explicitly excludes `repo_analysis/**` from the regular corpus.
  - Correct default for the primary index.

- `[Repo]` `standards/scan_exclusion_policy.yaml`
  - Also excludes `repo_analysis/**`.
  - Confirms the user's stated operational intent is already encoded in the repo.

- `[Repo]` `core/campaign/phase_packet.py`, `core/campaign/roadmap_validator.py`, and `Saguaro/saguaro/roadmap/validator.py`
  - Roadmap tooling already expects explicit phase packet fields and implementation-contract traceability.
  - This means the roadmap can be written as an executable governance artifact, not a speculative memo.

### Existing primitives that are strongest

- `[Repo]` Repo acquisition and isolation: `core/campaign/repo_cache.py`, `core/campaign/repo_registry.py`
- `[Repo]` Orchestration spine: `core/campaign/control_plane.py`
- `[Repo]` Query runtime shell: `Saguaro/saguaro/query/gateway.py`
- `[Repo]` Multi-language parse breadth: `Saguaro/saguaro/parsing/parser.py`
- `[Repo]` Relation substrate: `Saguaro/saguaro/analysis/code_graph.py`, `Saguaro/saguaro/analysis/icfg_builder.py`, `Saguaro/saguaro/analysis/bridge_synthesizer.py`
- `[Repo]` Native compute boundary: `Saguaro/src/ops/native_indexer_api.cc`, `Saguaro/saguaro/indexing/native_indexer_bindings.py`, `Saguaro/saguaro/storage/native_vector_store.py`
- `[Repo]` Governance and validation: `Saguaro/saguaro/roadmap/validator.py`, `core/campaign/phase_packet.py`

### Where the architecture is strongest

- `[Repo]` The repo already thinks in phases, evidence, and campaign control loops.
- `[Repo]` The repo already has a real native boundary rather than a notional one.
- `[Repo]` The repo already distinguishes target and analysis repositories in campaign logic.
- `[Repo]` The repo already carries graph, impact, and bridge concepts far beyond basic semantic search.

### Where the architecture looks thin or underexploited

- `[Repo]` Comparative repo analysis is currently dossier-shaped rather than graph-shaped.
- `[Repo]` Multi-corpus identity does not exist as a first-class abstraction.
- `[Repo]` Subtree focus exists only as a path parameter, not as a managed analysis corpus.
- `[Repo]` Symbol resolution is not robust under repo collisions.
- `[Repo]` Function-depth understanding is uneven across languages.
- `[Repo]` The R&D loop consumes simplistic repo dossier reuse suggestions instead of evidence-backed migration plans.
- `[Repo]` The health path has a live fastpath regression, which is unacceptable for the system that is supposed to govern the rest of the stack.

### Bottom line

- `[Synthesis]` Saguaro already has enough architecture to support the requested future.
- `[Synthesis]` The gap is not "invent an entirely new system."
- `[Synthesis]` The gap is "promote corpus identity, deepen program semantics, and wire comparative evidence into the campaign and EID spine."

---

## 4. Hidden Assumptions

1. `[Inference]` One repository equals one index, so subtrees are just filters rather than first-class corpora.
2. `[Inference]` Excluding `repo_analysis/**` from the main index is enough, when what is actually needed is a quarantined comparative corpus lifecycle.
3. `[Inference]` Broad language detection implies deep semantic understanding. It does not.
4. `[Inference]` File-level reuse candidates are a sufficient basis for R&D prioritization. They are not.
5. `[Inference]` Symbol names are globally resolvable without corpus qualification.
6. `[Inference]` Copying and rewriting are just implementation choices rather than different governance paths with different evidence requirements.
7. `[Inference]` Comparative repo analysis is mostly about identifying reusable code, when in practice it should identify reusable mechanisms, invariants, and architecture patterns.
8. `[Inference]` A markdown dossier is an acceptable terminal artifact for comparative analysis. It is not; it should be a structured evidence graph plus a human report.
9. `[Inference]` Deep graph analysis should run monolithically, when the correct posture is budgeted and mode-selectable.
10. `[Inference]` The EID loop only needs hypotheses and lightweight repo notes, rather than comparative relation evidence, migration recipes, and rewrite arbitrations.

---

## 5. Candidate Implementation Phases

The set below mixes roughly 70 percent practical phases and 30 percent moonshot phases. Every candidate is phrased as an executable workstream, not an idea card.

### Candidate 01. Corpus Session Ledger

Suggested `phase_id`: `intake`  
Class: practical  
Core insight: temporary external repos and focused subtrees must become first-class corpus sessions with explicit identity, provenance, TTL, and quarantine flags.  
External inspiration or analogy: `[External]` Kythe corpus identity, CodeQL database assets.  
Why it fits Saguaro and Anvil specifically: `Saguaro/saguaro/api.py` and `Saguaro/saguaro/state/ledger.py` already manage workspace state; they need corpus identity, not replacement.  
Exact places in this codebase where it could wire in: `Saguaro/saguaro/api.py`, `Saguaro/saguaro/state/ledger.py`, `domains/code_intelligence/saguaro_substrate.py`, `core/campaign/repo_registry.py`.  
Existing primitives it can reuse: `StateLedger`, repo roles, workspace metadata, query scope plumbing.  
New primitive, data flow, or subsystem needed: a corpus session manifest keyed by corpus id, repo root, snapshot digest, subtree root, build profile, TTL, and trust level.  
`repo_scope`: `["primary:Anvil", "subtree:Saguaro", "external:repo_analysis/*"]`  
`owning_specialist_type`: `["saguaro-state-architecture"]`  
`allowed_writes`: `["Saguaro/saguaro/api.py", "Saguaro/saguaro/state/ledger.py", "domains/code_intelligence/saguaro_substrate.py", "core/campaign/repo_registry.py"]`  
`telemetry_contract`: minimum metrics `corpus_session_count`, `corpus_session_boot_ms`, `corpus_session_gc_ms`, `quarantine_breach_count`.  
`required_evidence`: corpus session manifest, session lifecycle events, CLI and API transcript.  
`rollback_criteria`: any session mutates the live primary corpus manifest or breaks existing workspace sync semantics.  
`promotion_gate`: target repo, one subtree, and one external repo can coexist with unique corpus ids and isolated index roots.  
`success_criteria`: corpus id appears in query, slice, report, and EID payloads without ambiguity.  
Why this creates value: it makes every later comparative feature governable.  
Why this creates moat: corpus identity is the base primitive competitors usually bolt on late and painfully.  
Main risk or failure mode: overloading the existing workspace ledger with too many unrelated concerns.  
Smallest credible first experiment: create isolated sessions for `.` and `Saguaro/`, then verify separate manifests and query results.  
Confidence level: high.

### Candidate 02. Scoped Subtree Indexing and Focus Cones

Suggested `phase_id`: `intake`  
Class: practical  
Core insight: subtree analysis should not be a weak filter over the main corpus; it should create a focal corpus with explicit include roots, inherited exclusions, and upstream/downstream relation cones.  
External inspiration or analogy: `[External]` Tree-sitter incremental parsing; CodeQL `--source-root`; F1 practice session focused telemetry.  
Why it fits Saguaro and Anvil specifically: `Saguaro/saguaro/api.py:index(path=...)` already accepts a path; `Saguaro/saguaro/utils/file_utils.py` and `Saguaro/saguaro/query/corpus_rules.py` already own discovery.  
Exact places in this codebase where it could wire in: `Saguaro/saguaro/api.py`, `Saguaro/saguaro/utils/file_utils.py`, `Saguaro/saguaro/query/corpus_rules.py`, `Saguaro/saguaro/indexing/engine.py`.  
Existing primitives it can reuse: path resolution, manifest building, exclusion policy, incremental indexing.  
New primitive, data flow, or subsystem needed: focus cone descriptors capturing include roots plus dependency spillover edges into adjacent files.  
`repo_scope`: `["subtree:Saguaro", "subtree:core/research", "external:repo_analysis/*"]`  
`owning_specialist_type`: `["saguaro-indexing"]`  
`allowed_writes`: `["Saguaro/saguaro/api.py", "Saguaro/saguaro/utils/file_utils.py", "Saguaro/saguaro/query/corpus_rules.py", "Saguaro/saguaro/indexing/engine.py"]`  
`telemetry_contract`: minimum metrics `focus_corpus_file_count`, `focus_cone_expansion_edges`, `scoped_index_wall_ms`, `scoped_query_hit_rate`.  
`required_evidence`: manifest diff, cone-expansion artifact, scoped query traces.  
`rollback_criteria`: scoped indexing duplicates primary storage or silently pulls excluded roots.  
`promotion_gate`: focal indexing on `Saguaro/` yields deterministic manifests and queryable symbol sets.  
`success_criteria`: operator can analyze `Saguaro/` without touching unrelated `core/*` content unless relation cone explicitly requires it.  
Why this creates value: it enables precise upgrades to Saguaro inside Anvil.  
Why this creates moat: focused comparative analysis is more useful than whole-repo brute force.  
Main risk or failure mode: relation cones over-expand and destroy the point of scope control.  
Smallest credible first experiment: scoped corpus for `Saguaro/saguaro/analysis/` with optional spillover to parser and query modules.  
Confidence level: high.

### Candidate 03. Corpus-Qualified Symbol Identity

Suggested `phase_id`: `research`  
Class: practical  
Core insight: the system needs a stable symbol key shaped like `{corpus, root, path, language, symbol signature}` to end cross-repo collisions and enable trustworthy slices, relations, and migration maps.  
External inspiration or analogy: `[External]` Kythe `VName`; SCIP symbol identity.  
Why it fits Saguaro and Anvil specifically: the observed `SaguaroAPI.index` mis-resolution is a direct symptom of missing qualified identity.  
Exact places in this codebase where it could wire in: `Saguaro/saguaro/parsing/runtime_symbols.py`, `Saguaro/saguaro/agents/perception.py`, `Saguaro/saguaro/api.py`, `Saguaro/saguaro/services/platform.py`.  
Existing primitives it can reuse: runtime symbol manifests, slices, platform graph.  
New primitive, data flow, or subsystem needed: canonical corpus-qualified symbol ids and disambiguation rules.  
`repo_scope`: `["primary:Anvil", "external:repo_analysis/*"]`  
`owning_specialist_type`: `["saguaro-symbolics"]`  
`allowed_writes`: `["Saguaro/saguaro/parsing/runtime_symbols.py", "Saguaro/saguaro/agents/perception.py", "Saguaro/saguaro/api.py", "Saguaro/saguaro/services/platform.py"]`  
`telemetry_contract`: minimum metrics `symbol_collision_count`, `symbol_disambiguation_success_rate`, `slice_resolution_retry_count`.  
`required_evidence`: symbol identity schema, ambiguity corpus, before and after slice transcripts.  
`rollback_criteria`: existing single-corpus symbol resolution regresses or API compatibility breaks.  
`promotion_gate`: identical symbol names in multiple corpora resolve correctly without manual path hints.  
`success_criteria`: slice, graph, query, and report APIs all surface corpus-qualified ids.  
Why this creates value: it makes comparative analysis safe enough to trust.  
Why this creates moat: accurate cross-corpus symbol identity is hard and central.  
Main risk or failure mode: too many downstream consumers assume symbol ids are path-only.  
Smallest credible first experiment: qualify symbol ids for slice resolution only, then widen to graph edges and reports.  
Confidence level: high.

### Candidate 04. Build Fingerprint and Toolchain Capture

Suggested `phase_id`: `research`  
Class: practical  
Core insight: deep understanding of C, C++, Rust, Java, Go, and mixed native repos requires capturing build fingerprints rather than inferring semantics from source alone.  
External inspiration or analogy: `[External]` `scip-clang`; CodeQL compiled-language build modes.  
Why it fits Saguaro and Anvil specifically: the repo is itself native-heavy, with live CMake targets, native indexer bindings, and build graph tooling.  
Exact places in this codebase where it could wire in: `Saguaro/saguaro/build_system/ingestor.py`, `Saguaro/saguaro/api.py`, `core/research/repo_acquisition.py`, `core/campaign/repo_cache.py`.  
Existing primitives it can reuse: build graph, repo acquisition, target cache, native runtime hooks.  
New primitive, data flow, or subsystem needed: build fingerprint records containing compiler, flags, include roots, generated sources, and language confidence.  
`repo_scope`: `["primary:Anvil", "external:repo_analysis/*"]`  
`owning_specialist_type`: `["build-intelligence"]`  
`allowed_writes`: `["Saguaro/saguaro/build_system/ingestor.py", "Saguaro/saguaro/api.py", "core/research/repo_acquisition.py", "core/campaign/repo_cache.py"]`  
`telemetry_contract`: minimum metrics `build_profile_detected`, `build_capture_ms`, `compiled_language_precision_rate`.  
`required_evidence`: build fingerprint json, corpus manifest linkage, detection coverage report.  
`rollback_criteria`: build capture blocks analysis of lightweight interpreted repos.  
`promotion_gate`: build metadata is captured for at least CMake, Cargo, npm, and Python-based repos without operator hand-editing.  
`success_criteria`: corpus reports explicitly distinguish heuristic parse vs build-backed parse.  
Why this creates value: it makes native rewrite proposals believable.  
Why this creates moat: build-aware repo intelligence is where shallow semantic tools break down.  
Main risk or failure mode: build ingestion becomes too invasive or assumes builds must run.  
Smallest credible first experiment: passive build fingerprint extraction without executing builds.  
Confidence level: high.

### Candidate 05. Language Pack Negotiation and Capability Matrix

Suggested `phase_id`: `research`  
Class: practical  
Core insight: Saguaro needs to expose what it can understand deeply, shallowly, or not at all per corpus, instead of pretending all supported extensions are equal.  
External inspiration or analogy: `[External]` Semgrep language support tiers; CodeQL extractor availability.  
Why it fits Saguaro and Anvil specifically: `Saguaro/saguaro/parsing/parser.py` already detects many languages, but depth is uneven.  
Exact places in this codebase where it could wire in: `Saguaro/saguaro/parsing/parser.py`, `Saguaro/saguaro/api.py`, `Saguaro/saguaro/health.py`, `Saguaro/saguaro/analysis/report.py`.  
Existing primitives it can reuse: language detection, health reporting, report generation.  
New primitive, data flow, or subsystem needed: parser capability matrix with modes such as lexical, ast, cfg, dfg, symbol, and build-backed.  
`repo_scope`: `["primary:Anvil", "external:repo_analysis/*"]`  
`owning_specialist_type`: `["parser-governance"]`  
`allowed_writes`: `["Saguaro/saguaro/parsing/parser.py", "Saguaro/saguaro/api.py", "Saguaro/saguaro/health.py", "Saguaro/saguaro/analysis/report.py"]`  
`telemetry_contract`: minimum metrics `language_capability_coverage`, `deep_parse_coverage_percent`, `shallow_parse_fallback_count`.  
`required_evidence`: per-corpus capability matrix, operator-visible warnings, report excerpts.  
`rollback_criteria`: capability reporting becomes noisy or blocks existing workflows.  
`promotion_gate`: every comparative report distinguishes high-confidence and low-confidence language regions.  
`success_criteria`: EID and report compilers stop treating unsupported semantics as ground truth.  
Why this creates value: it prevents false certainty.  
Why this creates moat: trustworthy self-awareness beats inflated claims.  
Main risk or failure mode: capability taxonomy becomes bureaucratic instead of operationally useful.  
Smallest credible first experiment: emit capability matrix into report output only.  
Confidence level: high.

### Candidate 06. Corpus Lifecycle TTL and Quarantine

Suggested `phase_id`: `development`  
Class: practical  
Core insight: temporary corpora need explicit creation, retention, eviction, and quarantine semantics so that foreign repos can be analyzed aggressively without contaminating the target workspace.  
External inspiration or analogy: `[External]` ephemeral environment patterns; CodeQL database assets; F1 parc ferme rules.  
Why it fits Saguaro and Anvil specifically: the repo already excludes `repo_analysis/**`; the next step is controlled temporary corpora, not relaxed exclusion.  
Exact places in this codebase where it could wire in: `Saguaro/saguaro/query/corpus_rules.py`, `standards/scan_exclusion_policy.yaml`, `Saguaro/saguaro/state/ledger.py`, `core/campaign/repo_registry.py`.  
Existing primitives it can reuse: exclusion policies, repo roles, state ledger.  
New primitive, data flow, or subsystem needed: TTL scheduler, quarantine flagging, and cleanup hooks.  
`repo_scope`: `["external:repo_analysis/*"]`  
`owning_specialist_type`: `["corpus-operations"]`  
`allowed_writes`: `["Saguaro/saguaro/query/corpus_rules.py", "standards/scan_exclusion_policy.yaml", "Saguaro/saguaro/state/ledger.py", "core/campaign/repo_registry.py"]`  
`telemetry_contract`: minimum metrics `quarantined_corpus_count`, `expired_corpus_gc_ms`, `residual_artifact_count`.  
`required_evidence`: quarantine manifest, cleanup logs, zero-pollution verification.  
`rollback_criteria`: primary repo indexing behavior changes unexpectedly.  
`promotion_gate`: expired external corpora are removed without leaving retrievable artifacts in the main corpus.  
`success_criteria`: target repo queries remain stable after repeated external corpus ingestion cycles.  
Why this creates value: it preserves trust in the main index.  
Why this creates moat: safe comparative analysis is operationally harder than raw analysis.  
Main risk or failure mode: cleanup races or stale references in query gateway state.  
Smallest credible first experiment: TTL cleanup for one external corpus family under `repo_analysis/`.  
Confidence level: high.

### Candidate 07. Function-Interior Semantic Lift

Suggested `phase_id`: `analysis_upgrade`  
Class: practical  
Core insight: Saguaro must explicitly model function internals, not just declaration-level symbols, if it wants to support disparate relations and native rewrite planning.  
External inspiration or analogy: `[External]` Joern CPG; RepoHyper graph expansion.  
Why it fits Saguaro and Anvil specifically: parser, CFG, DFG, and ICFG machinery already exist but are not consistently elevated into comparative analysis.  
Exact places in this codebase where it could wire in: `Saguaro/saguaro/parsing/parser.py`, `Saguaro/saguaro/analysis/icfg_builder.py`, `Saguaro/saguaro/analysis/code_graph.py`, `Saguaro/saguaro/services/platform.py`.  
Existing primitives it can reuse: CFG, DFG, runtime symbols, graph store, shape propagation.  
New primitive, data flow, or subsystem needed: normalized intra-function relation summaries and operator-selectable deep slices.  
`repo_scope`: `["primary:Anvil", "subtree:Saguaro", "external:repo_analysis/*"]`  
`owning_specialist_type`: `["program-semantics"]`  
`allowed_writes`: `["Saguaro/saguaro/parsing/parser.py", "Saguaro/saguaro/analysis/icfg_builder.py", "Saguaro/saguaro/analysis/code_graph.py", "Saguaro/saguaro/services/platform.py"]`  
`telemetry_contract`: minimum metrics `function_body_graph_count`, `deep_slice_latency_ms`, `dfg_edge_density`, `cfg_edge_density`.  
`required_evidence`: deep slice examples, graph stats delta, comparative function report.  
`rollback_criteria`: graph size or query latency grows beyond controlled budgets without improved evidence.  
`promotion_gate`: at least one comparative analysis can explain not just what a function is named, but how it works and where semantics differ.  
`success_criteria`: reports can identify internal mechanism equivalence between target and foreign functions.  
Why this creates value: it answers the user's direct question about whether Saguaro can truly understand code.  
Why this creates moat: deep semantic mechanism mapping is a qualitatively different class of capability.  
Main risk or failure mode: graph explosion without usable summarization.  
Smallest credible first experiment: deep slices only for top-ranked candidate functions in a comparative report.  
Confidence level: high.

### Candidate 08. Cross-Language Code Property Graph Layer

Suggested `phase_id`: `analysis_upgrade`  
Class: practical  
Core insight: Saguaro should explicitly fuse AST, CFG, DFG, call, import, FFI, and bridge edges into one comparable graph layer per corpus, then support overlays across corpora.  
External inspiration or analogy: `[External]` Joern CPG, LLVM plus CPG integrations.  
Why it fits Saguaro and Anvil specifically: Saguaro already has most component graphs; the missing step is a unified comparative graph contract.  
Exact places in this codebase where it could wire in: `Saguaro/saguaro/analysis/code_graph.py`, `Saguaro/saguaro/services/platform.py`, `Saguaro/saguaro/analysis/bridge_synthesizer.py`, `Saguaro/saguaro/analysis/liveness.py`.  
Existing primitives it can reuse: platform graph stats, bridge edges, liveness, runtime symbols.  
New primitive, data flow, or subsystem needed: corpus-scoped CPG export and comparative overlay construction.  
`repo_scope`: `["primary:Anvil", "external:repo_analysis/*"]`  
`owning_specialist_type`: `["graph-architecture"]`  
`allowed_writes`: `["Saguaro/saguaro/analysis/code_graph.py", "Saguaro/saguaro/services/platform.py", "Saguaro/saguaro/analysis/bridge_synthesizer.py", "Saguaro/saguaro/analysis/liveness.py"]`  
`telemetry_contract`: minimum metrics `cpg_node_count`, `cpg_edge_count`, `cross_corpus_overlay_edges`, `graph_build_ms`.  
`required_evidence`: graph schema document, overlay stats, graph-backed comparative report.  
`rollback_criteria`: graph construction duplicates existing structures with no net utility.  
`promotion_gate`: one end-to-end comparative report must be explainable via overlayed graph evidence.  
`success_criteria`: operator can request "show me analogous mechanisms" and receive graph-backed results.  
Why this creates value: it gives the rest of the pipeline one semantic substrate.  
Why this creates moat: unified graph reasoning across corpora is hard to retrofit.  
Main risk or failure mode: excessive graph duplication between platform and analysis layers.  
Smallest credible first experiment: export a minimal comparative graph containing symbol, calls, imports, and FFI edges only.  
Confidence level: medium-high.

### Candidate 09. Disparate Relations Omnigraph

Suggested `phase_id`: `analysis_upgrade`  
Class: practical  
Core insight: comparative analysis should discover not just direct clones, but mechanism-level analogs with partial similarity across language, layering, performance posture, and operational role.  
External inspiration or analogy: `[External]` RepoHyper semantic graphs; bridge synthesis; heterogeneous graph retrieval.  
Why it fits Saguaro and Anvil specifically: the repo already talks about "disparate relations" in requirements and already has `bridge_synthesizer` and `reality/store.py` relation machinery.  
Exact places in this codebase where it could wire in: `Saguaro/saguaro/analysis/bridge_synthesizer.py`, `Saguaro/saguaro/services/platform.py`, `Saguaro/saguaro/reality/store.py`, `core/dare/repo_ingestion.py`.  
Existing primitives it can reuse: bridge scoring, omni-relations, graph metrics, repo ingestion.  
New primitive, data flow, or subsystem needed: cross-corpus relation classes such as `analogous_mechanism`, `portable_pattern`, `rewrite_candidate`, `copy_candidate`, `ffi_analogue`, and `performance_upgrade_path`.  
`repo_scope`: `["primary:Anvil", "external:repo_analysis/*"]`  
`owning_specialist_type`: `["comparative-intelligence"]`  
`allowed_writes`: `["Saguaro/saguaro/analysis/bridge_synthesizer.py", "Saguaro/saguaro/services/platform.py", "Saguaro/saguaro/reality/store.py", "core/dare/repo_ingestion.py"]`  
`telemetry_contract`: minimum metrics `disparate_relation_count`, `relation_precision_audit_score`, `false_bridge_rate`.  
`required_evidence`: ranked relation ledger, analyst review set, contrasting true and false positives.  
`rollback_criteria`: relation output becomes recommendation theater with poor precision.  
`promotion_gate`: top-N relations must survive manual review on at least one target and two external corpora.  
`success_criteria`: comparative reports can point to mechanism-aligned upgrade opportunities even when file names and languages differ.  
Why this creates value: it enables real upgrade discovery instead of file scavenging.  
Why this creates moat: disparate relation inference is much harder than clone detection.  
Main risk or failure mode: relation hallucination driven by weak embeddings or shallow symbols.  
Smallest credible first experiment: generate relations only among top-100 graph-central functions and validate manually.  
Confidence level: medium-high.

### Candidate 10. Native Similarity Kernels for Relation Scoring

Suggested `phase_id`: `development`  
Class: moonshot  
Core insight: relation scoring, structural alignment, and candidate pruning will eventually bottleneck in Python; this should become a native kernel family with bounded memory layouts.  
External inspiration or analogy: `[External]` native graph engines, CPG scale discipline, high-frequency telemetry pipelines.  
Why it fits Saguaro and Anvil specifically: the repo already has `Saguaro/src/ops/native_indexer_api.cc` and native vector store infrastructure.  
Exact places in this codebase where it could wire in: `Saguaro/src/ops/native_indexer_api.cc`, `Saguaro/saguaro/indexing/native_indexer_bindings.py`, `Saguaro/saguaro/storage/native_vector_store.py`, `Saguaro/saguaro/services/platform.py`.  
Existing primitives it can reuse: native indexer bindings, mmap vector store, query gateway.  
New primitive, data flow, or subsystem needed: native relation scoring kernels over corpus-qualified symbol and graph feature vectors.  
`repo_scope`: `["primary:Anvil", "external:repo_analysis/*"]`  
`owning_specialist_type`: `["native-graph-runtime"]`  
`allowed_writes`: `["Saguaro/src/ops/native_indexer_api.cc", "Saguaro/saguaro/indexing/native_indexer_bindings.py", "Saguaro/saguaro/storage/native_vector_store.py", "Saguaro/saguaro/services/platform.py"]`  
`telemetry_contract`: minimum metrics `native_relation_score_ms`, `native_relation_ram_mb`, `python_fallback_rate`.  
`required_evidence`: benchmark traces, CPU profile, equivalence checks against Python scorer.  
`rollback_criteria`: native path is faster but semantically divergent, or operationally fragile.  
`promotion_gate`: native scorer beats Python baseline on latency and stays within semantic tolerance thresholds.  
`success_criteria`: comparative scans across several corpora become tractable without runaway latency.  
Why this creates value: it keeps the ambitious feature set operational.  
Why this creates moat: native comparative graph kernels are hard to copy.  
Main risk or failure mode: premature native complexity before semantics are stable.  
Smallest credible first experiment: native top-K candidate pruning for one relation type.  
Confidence level: medium.

### Candidate 11. Migration Recipe Synthesizer

Suggested `phase_id`: `analysis_upgrade`  
Class: practical  
Core insight: the system should produce explicit migration recipes, not merely ranked candidate files, whenever it identifies a strong external mechanism worth porting or rewriting.  
External inspiration or analogy: `[External]` OpenRewrite recipes, Piranha cleanup rules, Getafix structured fixes.  
Why it fits Saguaro and Anvil specifically: campaign and EID infrastructure already think in tasks, phases, and evidence.  
Exact places in this codebase where it could wire in: `core/dare/repo_ingestion.py`, `core/research/analysis_pack.py`, `core/research/eid_master.py`, `Saguaro/saguaro/analysis/report.py`.  
Existing primitives it can reuse: dossiers, report generator, EID packets, knowledge base.  
New primitive, data flow, or subsystem needed: recipe documents containing source mechanism, target insertion points, rewrite posture, invariants, and verification requirements.  
`repo_scope`: `["primary:Anvil", "external:repo_analysis/*"]`  
`owning_specialist_type`: `["migration-architect"]`  
`allowed_writes`: `["core/dare/repo_ingestion.py", "core/research/analysis_pack.py", "core/research/eid_master.py", "Saguaro/saguaro/analysis/report.py"]`  
`telemetry_contract`: minimum metrics `recipe_count`, `recipe_acceptance_rate`, `recipe_verification_pass_rate`.  
`required_evidence`: recipe packets, sample migration plan, review outcomes.  
`rollback_criteria`: recipes become vague prose or duplicate report content.  
`promotion_gate`: one promoted recipe must be actionable enough to enter a development phase.  
`success_criteria`: comparative analysis yields structured next-step artifacts.  
Why this creates value: it turns insight into execution.  
Why this creates moat: explicit migration recipes create accumulated institutional memory.  
Main risk or failure mode: recipe explosion with poor prioritization.  
Smallest credible first experiment: emit recipe skeletons only for high-confidence comparative hits.  
Confidence level: high.

### Candidate 12. Copy-vs-Rewrite Arbiter

Suggested `phase_id`: `eid`  
Class: practical  
Core insight: Saguaro should never treat "port it" as one mode. It needs explicit arbitration between direct copy, wrapper integration, transliteration, native rewrite, or pattern-only adoption.  
External inspiration or analogy: `[External]` Piranha and OpenRewrite recipe discipline; Meta-style reviewable fix proposals.  
Why it fits Saguaro and Anvil specifically: the user explicitly wants native versions when appropriate and direct copy only when it actually makes sense.  
Exact places in this codebase where it could wire in: `core/research/eid_master.py`, `core/campaign/control_plane.py`, `core/dare/repo_ingestion.py`, `core/research/analysis_pack.py`.  
Existing primitives it can reuse: EID ranking, hypotheses, repo dossiers, specialist packets.  
New primitive, data flow, or subsystem needed: arbitration matrix combining fit, performance, risk, maintenance cost, build compatibility, and native opportunity.  
`repo_scope`: `["primary:Anvil", "external:repo_analysis/*"]`  
`owning_specialist_type`: `["eid-migration-arbiter"]`  
`allowed_writes`: `["core/research/eid_master.py", "core/campaign/control_plane.py", "core/dare/repo_ingestion.py", "core/research/analysis_pack.py"]`  
`telemetry_contract`: minimum metrics `arbiter_decision_count`, `native_rewrite_fraction`, `copy_reversal_rate`.  
`required_evidence`: arbitration tables, rationale traces, later outcome comparisons.  
`rollback_criteria`: decision classes prove unstable or unauditable.  
`promotion_gate`: EID actions must include arbitration posture and evidence, not just candidate path.  
`success_criteria`: downstream development phases know whether they are copying, adapting, or rewriting before work begins.  
Why this creates value: it aligns the system with the user's stated engineering philosophy.  
Why this creates moat: most tools stop at "similarity found"; few arbitrate implementation posture well.  
Main risk or failure mode: arbiter scoring becomes overconfident with thin evidence.  
Smallest credible first experiment: apply the arbiter to top five comparative candidates from one external repo.  
Confidence level: high.

### Candidate 13. Repo Twin Report Compiler

Suggested `phase_id`: `convergence`  
Class: practical  
Core insight: every comparative analysis should yield a reproducible repo twin report that explains what the foreign repo is, how it works, what parts resemble the target, and what should be ported, rewritten, or rejected.  
External inspiration or analogy: `[Inference]` digital twin operations; `[External]` RepoHyper report-style repository context expansion.  
Why it fits Saguaro and Anvil specifically: `Saguaro/saguaro/analysis/report.py` already exists, but it focuses on dead code, coverage, entry points, and verification, not comparative repo intelligence.  
Exact places in this codebase where it could wire in: `Saguaro/saguaro/analysis/report.py`, `core/research/analysis_pack.py`, `core/dare/repo_ingestion.py`, `core/campaign/control_plane.py`.  
Existing primitives it can reuse: report generator, repo dossiers, campaign artifacts.  
New primitive, data flow, or subsystem needed: comparative report sections for topology, semantics, analogous mechanisms, migration recipes, and port ledger.  
`repo_scope`: `["primary:Anvil", "external:repo_analysis/*", "subtree:Saguaro"]`  
`owning_specialist_type`: `["comparative-reporting"]`  
`allowed_writes`: `["Saguaro/saguaro/analysis/report.py", "core/research/analysis_pack.py", "core/dare/repo_ingestion.py", "core/campaign/control_plane.py"]`  
`telemetry_contract`: minimum metrics `report_compile_ms`, `report_section_coverage`, `report_evidence_density`.  
`required_evidence`: generated report, source evidence bundles, operator acceptance review.  
`rollback_criteria`: report compiler becomes verbose without increasing decision quality.  
`promotion_gate`: one report must be strong enough to drive an implementation phase without manual repo spelunking.  
`success_criteria`: user can ask a repo-comparison question and get a full report instead of a loose note set.  
Why this creates value: it creates the direct user-facing payoff.  
Why this creates moat: high-trust report compilation from code evidence is hard.  
Main risk or failure mode: reports become boilerplate rather than mechanism-level documents.  
Smallest credible first experiment: comparative report between `Saguaro/` subtree and one external analysis repo.  
Confidence level: high.

### Candidate 14. Evidence-Backed Port Ledger

Suggested `phase_id`: `convergence`  
Class: practical  
Core insight: port candidates should accumulate in a ledger with status, evidence, arbitration result, target insertion points, and verification obligations.  
External inspiration or analogy: `[Inference]` engineering change orders and flight rule logs.  
Why it fits Saguaro and Anvil specifically: campaign and roadmap tooling already care about evidence closure; comparative repo work needs the same discipline.  
Exact places in this codebase where it could wire in: `core/campaign/control_plane.py`, `core/research/eid_master.py`, `Saguaro/saguaro/reality/store.py`, `Saguaro/saguaro/roadmap/validator.py`.  
Existing primitives it can reuse: phase packets, omni-relations, roadmap validation.  
New primitive, data flow, or subsystem needed: port ledger records with statuses `candidate`, `promoted`, `implemented`, `rejected`, `superseded`.  
`repo_scope`: `["primary:Anvil", "external:repo_analysis/*"]`  
`owning_specialist_type`: `["evidence-governance"]`  
`allowed_writes`: `["core/campaign/control_plane.py", "core/research/eid_master.py", "Saguaro/saguaro/reality/store.py", "Saguaro/saguaro/roadmap/validator.py"]`  
`telemetry_contract`: minimum metrics `port_candidate_count`, `port_promotion_rate`, `evidence_closure_rate`.  
`required_evidence`: ledger snapshot, linked report ids, linked verification refs.  
`rollback_criteria`: ledger duplicates roadmap data without improving control.  
`promotion_gate`: promoted items must have complete provenance to source corpus and target insertion point.  
`success_criteria`: no port recommendation exists only in free text.  
Why this creates value: it prevents comparative intelligence from evaporating between sessions.  
Why this creates moat: durable institutional memory compounds over time.  
Main risk or failure mode: too much clerical overhead.  
Smallest credible first experiment: ledger entries only for top 10 comparative candidates per report.  
Confidence level: medium-high.

### Candidate 15. Batch `repo_analysis/` Fleet Sweep

Suggested `phase_id`: `research`  
Class: practical  
Core insight: the operator should be able to ask for one repo, one folder, or all repos under `repo_analysis/` and receive bounded comparative analysis rather than one-off manual orchestration.  
External inspiration or analogy: `[External]` CodeQL multi-repository variant analysis controller model.  
Why it fits Saguaro and Anvil specifically: `repo_analysis/**` is already a known excluded zone and `core/campaign/repo_cache.py` already manages acquisition.  
Exact places in this codebase where it could wire in: `core/research/repo_acquisition.py`, `core/campaign/control_plane.py`, `Saguaro/saguaro/api.py`, `domains/code_intelligence/saguaro_substrate.py`.  
Existing primitives it can reuse: repo cache, acquisition, query API, control plane.  
New primitive, data flow, or subsystem needed: fleet controller that instantiates one corpus session per local repo and aggregates results.  
`repo_scope`: `["external:repo_analysis/*"]`  
`owning_specialist_type`: `["fleet-analysis"]`  
`allowed_writes`: `["core/research/repo_acquisition.py", "core/campaign/control_plane.py", "Saguaro/saguaro/api.py", "domains/code_intelligence/saguaro_substrate.py"]`  
`telemetry_contract`: minimum metrics `fleet_repo_count`, `fleet_scan_wall_ms`, `fleet_failure_count`, `aggregate_candidate_count`.  
`required_evidence`: fleet session manifest, per-repo summaries, top-level aggregate report.  
`rollback_criteria`: fleet mode starves single-repo workflows.  
`promotion_gate`: system can scan at least three local analysis repos and aggregate findings by category.  
`success_criteria`: user can query one repo or all local repos from the same surface.  
Why this creates value: it turns `repo_analysis/` into a real research asset.  
Why this creates moat: multi-repo local comparative workflows are operationally rare.  
Main risk or failure mode: uncontrolled compute and storage growth.  
Smallest credible first experiment: batch scan only metadata, capability matrix, and top comparative candidates.  
Confidence level: high.

### Candidate 16. Federated Query Gateway

Suggested `phase_id`: `development`  
Class: practical  
Core insight: multi-corpus querying should route through one resident gateway that knows which corpora are active, which are temporary, and how to merge or isolate results.  
External inspiration or analogy: `[External]` MRVA controller routing, service mesh ingress.  
Why it fits Saguaro and Anvil specifically: `Saguaro/saguaro/query/gateway.py` already has admission control and a resident server.  
Exact places in this codebase where it could wire in: `Saguaro/saguaro/query/gateway.py`, `Saguaro/saguaro/api.py`, `Saguaro/saguaro/state/ledger.py`, `domains/code_intelligence/saguaro_substrate.py`.  
Existing primitives it can reuse: `QueryGateway`, `SessionGovernor`, state reporting, prewarm hooks.  
New primitive, data flow, or subsystem needed: corpus routing policy, merge strategy, and per-corpus scoring normalization.  
`repo_scope`: `["primary:Anvil", "subtree:Saguaro", "external:repo_analysis/*"]`  
`owning_specialist_type`: `["query-federation"]`  
`allowed_writes`: `["Saguaro/saguaro/query/gateway.py", "Saguaro/saguaro/api.py", "Saguaro/saguaro/state/ledger.py", "domains/code_intelligence/saguaro_substrate.py"]`  
`telemetry_contract`: minimum metrics `federated_query_ms`, `corpus_fanout_count`, `gateway_queue_depth`, `merged_result_dedup_rate`.  
`required_evidence`: gateway state dumps, query traces, merged and isolated result samples.  
`rollback_criteria`: local single-corpus query latency regresses materially.  
`promotion_gate`: gateway can answer local, subtree, one-external, and all-external query modes.  
`success_criteria`: corpus routing is explicit in responses and deterministic under repeat runs.  
Why this creates value: it gives the operator one coherent surface.  
Why this creates moat: runtime federation is the bridge from architecture to usability.  
Main risk or failure mode: score normalization across corpora becomes misleading.  
Smallest credible first experiment: gateway fanout across primary and one temporary corpus only.  
Confidence level: high.

### Candidate 17. Symbol Ambiguity Resolver

Suggested `phase_id`: `analysis_upgrade`  
Class: practical  
Core insight: when symbol lookup is ambiguous, the system should present a ranked disambiguation set with corpus, path, language, and local relation context, rather than returning the wrong target.  
External inspiration or analogy: `[External]` precise code navigation disambiguation; search engine query reformulation.  
Why it fits Saguaro and Anvil specifically: the observed `SaguaroAPI.index` mis-resolution demonstrates the operational need immediately.  
Exact places in this codebase where it could wire in: `Saguaro/saguaro/agents/perception.py`, `Saguaro/saguaro/parsing/runtime_symbols.py`, `Saguaro/saguaro/api.py`, `core/unified_chat_loop.py`.  
Existing primitives it can reuse: slices, runtime symbol reports, query explanations, chat loop rendering.  
New primitive, data flow, or subsystem needed: ambiguity bundles and disambiguation ranking logic.  
`repo_scope`: `["primary:Anvil", "external:repo_analysis/*"]`  
`owning_specialist_type`: `["symbol-resolution"]`  
`allowed_writes`: `["Saguaro/saguaro/agents/perception.py", "Saguaro/saguaro/parsing/runtime_symbols.py", "Saguaro/saguaro/api.py", "core/unified_chat_loop.py"]`  
`telemetry_contract`: minimum metrics `ambiguous_lookup_count`, `wrong_target_rate`, `disambiguation_acceptance_rate`.  
`required_evidence`: ambiguous symbol corpus, before and after examples, UI traces.  
`rollback_criteria`: added ambiguity logic slows ordinary precise lookups disproportionately.  
`promotion_gate`: wrong-target slice reproduction disappears in comparative test fixtures.  
`success_criteria`: no silent cross-corpus symbol confusion.  
Why this creates value: it raises trust immediately.  
Why this creates moat: trustworthy ambiguity handling is a strong multiplier for every other feature.  
Main risk or failure mode: excessive prompting for disambiguation.  
Smallest credible first experiment: trigger disambiguation only when multiple corpora contain the same symbol signature.  
Confidence level: high.

### Candidate 18. EID Comparative Frontier Planner

Suggested `phase_id`: `eid`  
Class: practical  
Core insight: the EID loop should treat external repos as a frontier map of mechanisms and experiments, not just a ranked pile of reuse candidates.  
External inspiration or analogy: `[External]` staged translation agents; pit wall experiment planning.  
Why it fits Saguaro and Anvil specifically: `core/research/eid_master.py` already has ranking, scheduler, design service, simulator planner, and memory projection.  
Exact places in this codebase where it could wire in: `core/research/eid_master.py`, `core/campaign/control_plane.py`, `core/memory/fabric/jobs.py`, `core/qsg/latent_bridge.py`.  
Existing primitives it can reuse: hypothesis ranking, scheduler, latent bridge, memory fabric.  
New primitive, data flow, or subsystem needed: comparative frontier packets containing external mechanism, target gap, arbitration posture, and recommended experiment tracks.  
`repo_scope`: `["primary:Anvil", "external:repo_analysis/*", "subtree:Saguaro"]`  
`owning_specialist_type`: `["eid-comparative-planner"]`  
`allowed_writes`: `["core/research/eid_master.py", "core/campaign/control_plane.py", "core/memory/fabric/jobs.py", "core/qsg/latent_bridge.py"]`  
`telemetry_contract`: minimum metrics `comparative_frontier_count`, `eid_repo_evidence_rate`, `experiment_track_acceptance_rate`.  
`required_evidence`: frontier packet artifacts, EID packet deltas, experiment recommendations.  
`rollback_criteria`: EID output becomes noisier without more actionable experiments.  
`promotion_gate`: EID must emit at least one comparative experiment plan with explicit source mechanism and target insertion scope.  
`success_criteria`: repo intelligence directly changes R&D recommendations.  
Why this creates value: it wires comparative analysis into the system's strategic loop.  
Why this creates moat: this is where analysis becomes long-term innovation velocity.  
Main risk or failure mode: EID overload from too many external signals.  
Smallest credible first experiment: feed only top three arbitration-approved candidates into EID.  
Confidence level: high.

### Candidate 19. Prompt and Loop Wiring for Comparative R&D

Suggested `phase_id`: `eid`  
Class: practical  
Core insight: campaign and chat loops should know how to request comparative scans, scoped corpus work, and report compilation without burying those moves in ad hoc prompt text.  
External inspiration or analogy: `[External]` tool-using agents with controller APIs; explicit function pools.  
Why it fits Saguaro and Anvil specifically: the user wants this wired into the EID campaign setup, not left as a manual operator skill.  
Exact places in this codebase where it could wire in: `core/campaign/control_plane.py`, `core/unified_chat_loop.py`, `domains/code_intelligence/saguaro_substrate.py`, `prompts/prompts.md`.  
Existing primitives it can reuse: control plane loop stages, substrate tool wrappers, specialist prompts.  
New primitive, data flow, or subsystem needed: comparative-analysis intents and tool call envelopes.  
`repo_scope`: `["primary:Anvil", "external:repo_analysis/*"]`  
`owning_specialist_type`: `["loop-integration"]`  
`allowed_writes`: `["core/campaign/control_plane.py", "core/unified_chat_loop.py", "domains/code_intelligence/saguaro_substrate.py", "prompts/prompts.md"]`  
`telemetry_contract`: minimum metrics `comparative_tool_call_count`, `comparative_loop_success_rate`, `report_request_to_report_ms`.  
`required_evidence`: prompt trace, loop transcripts, tool routing trace.  
`rollback_criteria`: baseline conversational repo workflows regress.  
`promotion_gate`: campaign setup can request comparative analysis without manual shell orchestration.  
`success_criteria`: EID loop natively calls the new comparative surfaces.  
Why this creates value: it converts capability into actual use.  
Why this creates moat: many systems build tools they fail to integrate into the main loop.  
Main risk or failure mode: prompt surfaces overfit a narrow use case.  
Smallest credible first experiment: add a single comparative-analysis tool path into the campaign research step.  
Confidence level: high.

### Candidate 20. Operator Surface for Comparative Analysis

Suggested `phase_id`: `convergence`  
Class: practical  
Core insight: the operator should be able to say "index this repo temporarily", "compare that repo to `Saguaro/`", or "scan all repos in `repo_analysis/`" from one coherent CLI and REPL surface.  
External inspiration or analogy: `[External]` CodeQL MRVA controller UX, precise navigation UX.  
Why it fits Saguaro and Anvil specifically: `Saguaro/saguaro/cli.py`, `cli/repl.py`, and `core/unified_chat_loop.py` already form the operator boundary.  
Exact places in this codebase where it could wire in: `Saguaro/saguaro/cli.py`, `cli/repl.py`, `core/unified_chat_loop.py`, `domains/code_intelligence/saguaro_substrate.py`.  
Existing primitives it can reuse: CLI parser, substrate wrappers, chat loop tool routing.  
New primitive, data flow, or subsystem needed: corpus session commands and comparative report commands.  
`repo_scope`: `["primary:Anvil", "subtree:Saguaro", "external:repo_analysis/*"]`  
`owning_specialist_type`: `["operator-experience"]`  
`allowed_writes`: `["Saguaro/saguaro/cli.py", "cli/repl.py", "core/unified_chat_loop.py", "domains/code_intelligence/saguaro_substrate.py"]`  
`telemetry_contract`: minimum metrics `comparative_cli_invocations`, `comparative_repl_invocations`, `operator_error_rate`.  
`required_evidence`: CLI help text, REPL transcript, operator acceptance runbook.  
`rollback_criteria`: CLI complexity damages discoverability or breaks existing commands.  
`promotion_gate`: one documented operator flow exists for single repo, subtree, and fleet analysis.  
`success_criteria`: comparative intelligence is operationally accessible, not just architecturally possible.  
Why this creates value: it brings the feature into daily use.  
Why this creates moat: operational polish separates a system from a demo.  
Main risk or failure mode: too many knobs with poor defaults.  
Smallest credible first experiment: add a hidden or experimental `corpus` and `compare` CLI path.  
Confidence level: high.

### Candidate 21. Native Partial Clone and Lazy Materialization

Suggested `phase_id`: `development`  
Class: moonshot  
Core insight: large fleet analysis should not eagerly materialize every external corpus at full depth; it should lazy-load relation-rich regions and keep cold corpora in a compressed native form.  
External inspiration or analogy: `[Inference]` memory hierarchy design, telemetry buffering, database paging.  
Why it fits Saguaro and Anvil specifically: mmap vector storage and native indexer APIs already exist.  
Exact places in this codebase where it could wire in: `Saguaro/saguaro/storage/native_vector_store.py`, `Saguaro/saguaro/api.py`, `Saguaro/src/ops/native_indexer_api.cc`, `Saguaro/saguaro/query/gateway.py`.  
Existing primitives it can reuse: native vector store, gateway prewarm, shared index lock.  
New primitive, data flow, or subsystem needed: lazy corpus materialization policy and cold-corpus relation summaries.  
`repo_scope`: `["external:repo_analysis/*"]`  
`owning_specialist_type`: `["native-storage-runtime"]`  
`allowed_writes`: `["Saguaro/saguaro/storage/native_vector_store.py", "Saguaro/saguaro/api.py", "Saguaro/src/ops/native_indexer_api.cc", "Saguaro/saguaro/query/gateway.py"]`  
`telemetry_contract`: minimum metrics `lazy_materialization_ms`, `cold_corpus_hit_rate`, `resident_ram_mb`.  
`required_evidence`: memory profile, cache hit traces, corpus paging logs.  
`rollback_criteria`: lazy loading creates correctness drift or unpredictable latency spikes.  
`promotion_gate`: fleet scan memory footprint falls materially without harming top-K candidate quality.  
`success_criteria`: multi-repo analysis scales to larger local corpora sets.  
Why this creates value: it extends operational headroom.  
Why this creates moat: real local multi-corpus scale requires disciplined memory architecture.  
Main risk or failure mode: complexity before proven need.  
Smallest credible first experiment: lazy-load only deep graph artifacts while keeping lightweight manifests hot.  
Confidence level: medium.

### Candidate 22. Learned Relation Embedding Layer

Suggested `phase_id`: `analysis_upgrade`  
Class: moonshot  
Core insight: after explicit graph features are stable, Saguaro could learn a cross-corpus relation embedding that ranks analogous mechanisms beyond surface graph overlap.  
External inspiration or analogy: `[External]` repository-level graph retrieval and link prediction from RepoHyper-style pipelines.  
Why it fits Saguaro and Anvil specifically: the repo already contains native embedding primitives and query-time hybrid ranking infrastructure.  
Exact places in this codebase where it could wire in: `Saguaro/saguaro/services/platform.py`, `Saguaro/saguaro/api.py`, `Saguaro/src/ops/hyperdimensional_embedding_op.cc`, `Saguaro/src/ops/quantum_embedding_op.cc`.  
Existing primitives it can reuse: embeddings, platform graph, hybrid query planning.  
New primitive, data flow, or subsystem needed: relation embedding features over corpus-qualified graph neighborhoods.  
`repo_scope`: `["primary:Anvil", "external:repo_analysis/*"]`  
`owning_specialist_type`: `["graph-ml-runtime"]`  
`allowed_writes`: `["Saguaro/saguaro/services/platform.py", "Saguaro/saguaro/api.py", "Saguaro/src/ops/hyperdimensional_embedding_op.cc", "Saguaro/src/ops/quantum_embedding_op.cc"]`  
`telemetry_contract`: minimum metrics `relation_model_precision_at_k`, `learned_ranker_gain`, `training_data_drift`.  
`required_evidence`: offline benchmark, ablation report, misfire review set.  
`rollback_criteria`: learned ranking obscures explainability or underperforms explicit signals.  
`promotion_gate`: learned ranker beats explicit baseline on manually reviewed comparative datasets.  
`success_criteria`: higher-quality analogous mechanism discovery on difficult cross-language cases.  
Why this creates value: it pushes beyond deterministic pattern matching.  
Why this creates moat: learned cross-corpus mechanism ranking is advanced and hard to commoditize.  
Main risk or failure mode: weak labels and explainability gaps.  
Smallest credible first experiment: rerank only candidates already approved by explicit relation filters.  
Confidence level: medium-low.

### Candidate 23. Integration Wind Tunnel

Suggested `phase_id`: `deep_test_audit`  
Class: moonshot  
Core insight: before promoting a port or native rewrite, Saguaro should simulate integration impact in a bounded "wind tunnel" mode that predicts likely blast radius, dependency conflicts, and benchmark implications.  
External inspiration or analogy: `[Inference]` aerodynamics wind tunnel and software digital twin.  
Why it fits Saguaro and Anvil specifically: impact analysis, build graph, and campaign verification lanes already exist.  
Exact places in this codebase where it could wire in: `Saguaro/saguaro/analysis/impact.py`, `Saguaro/saguaro/build_system/ingestor.py`, `core/campaign/control_plane.py`, `core/research/eid_master.py`.  
Existing primitives it can reuse: impact analysis, build graph, verification lanes, simulator planner.  
New primitive, data flow, or subsystem needed: comparative integration simulation envelope with predicted dependency and test impact.  
`repo_scope`: `["primary:Anvil", "subtree:Saguaro", "external:repo_analysis/*"]`  
`owning_specialist_type`: `["integration-simulation"]`  
`allowed_writes`: `["Saguaro/saguaro/analysis/impact.py", "Saguaro/saguaro/build_system/ingestor.py", "core/campaign/control_plane.py", "core/research/eid_master.py"]`  
`telemetry_contract`: minimum metrics `predicted_blast_radius`, `simulation_wall_ms`, `simulation_vs_actual_delta`.  
`required_evidence`: simulation artifact, predicted vs actual diff, benchmark traces.  
`rollback_criteria`: simulation consistently fails to correlate with real impact.  
`promotion_gate`: simulation meaningfully predicts test or dependency impact on at least one implemented migration.  
`success_criteria`: promoted work arrives with an impact forecast, not just hope.  
Why this creates value: it lowers the cost of ambitious comparative adoption.  
Why this creates moat: predictive integration analysis is strategically powerful.  
Main risk or failure mode: false confidence from weak impact models.  
Smallest credible first experiment: predict touched modules and tests only.  
Confidence level: medium-low.

### Candidate 24. Self-Healing Comparative Workbench

Suggested `phase_id`: `convergence`  
Class: moonshot  
Core insight: the long-term endpoint is a workbench where comparative analysis produces a report, recipes, arbitration, candidate patches, and verification plans in one controlled loop with human promotion gates.  
External inspiration or analogy: `[External]` Getafix proposal loop, staged translation agents, controller-based modernization systems.  
Why it fits Saguaro and Anvil specifically: the campaign system, roadmap validator, memory fabric, and native/runtime stack already support pieces of this end-state.  
Exact places in this codebase where it could wire in: `core/campaign/control_plane.py`, `core/research/eid_master.py`, `Saguaro/saguaro/analysis/report.py`, `Saguaro/saguaro/roadmap/validator.py`.  
Existing primitives it can reuse: control plane, EID, report compiler, roadmap contracts.  
New primitive, data flow, or subsystem needed: closed-loop comparative mission controller.  
`repo_scope`: `["primary:Anvil", "external:repo_analysis/*", "subtree:Saguaro"]`  
`owning_specialist_type`: `["comparative-mission-control"]`  
`allowed_writes`: `["core/campaign/control_plane.py", "core/research/eid_master.py", "Saguaro/saguaro/analysis/report.py", "Saguaro/saguaro/roadmap/validator.py"]`  
`telemetry_contract`: minimum metrics `comparative_mission_count`, `promotion_gate_pass_rate`, `autonomous_step_reversal_rate`.  
`required_evidence`: end-to-end mission transcript, promotion packet, rollback packet.  
`rollback_criteria`: the loop starts skipping evidence and human review gates.  
`promotion_gate`: one end-to-end comparative mission completes with explicit gate review between analysis and implementation.  
`success_criteria`: comparative repo intelligence becomes a repeatable operational mode.  
Why this creates value: it turns the roadmap into a living operating capability.  
Why this creates moat: tightly coupled analysis-to-roadmap-to-implementation loops are difficult to replicate well.  
Main risk or failure mode: too much automation before the core semantics are trustworthy.  
Smallest credible first experiment: mission controller that stops at report and recipe generation only.  
Confidence level: medium-low.

---

## 6. Critical Pressure Test

### Elegant but likely wrong

1. `[Inference]` Jumping straight to a learned relation embedding layer before corpus identity and explicit graph relations are stable. That is attractive, but premature.
2. `[Inference]` Treating build-aware extraction as mandatory for all repos from day one. That would slow the system and punish lightweight repos.
3. `[Inference]` Building a giant generic multi-repo knowledge graph before proving high-value comparative workflows on one subtree and one external repo.
4. `[Inference]` Attempting full autonomous native rewrites directly from comparative matches. The repo is not yet ready for that trust level.

### Ugly but strategically powerful

1. `[Repo+Inference]` Corpus-qualified symbol ids are not glamorous, but they are foundational and immediately justified by a real mis-resolution observed during analysis.
2. `[Repo+Inference]` A parser capability matrix may feel bureaucratic, but it is how the system stops overstating what it can understand.
3. `[Repo+Inference]` Quarantine and TTL rules for temporary corpora are operational plumbing, but without them the main index will lose credibility.
4. `[Repo+Inference]` Copy-vs-rewrite arbitration is less glamorous than "AI-powered migration," but it is the difference between architecture and cargo-cult porting.

### Likely to fail because the repo is missing a key primitive

1. `[Repo+Inference]` A fully trustworthy disparate-relations engine will fail if corpus-qualified identity is not implemented first.
2. `[Repo+Inference]` Deep comparative analysis across native repos will underperform until build fingerprint capture exists.
3. `[Repo+Inference]` EID integration will remain shallow unless repo ingestion stops producing static markdown-like summaries and starts producing structured comparative evidence.
4. `[Repo+Inference]` Native relation kernels will be wasted effort if the semantic feature contract is unstable.

### Can Saguaro understand code fully today?

Short answer: no, not uniformly.

- `[Repo]` It already understands a lot more than lexical search. It has AST extraction, CFG/DFG surfaces, runtime symbol resolution, impact analysis, and bridge edges.
- `[Repo]` It already supports broad language detection across many languages.
- `[Repo]` But understanding depth is uneven by language and by build context.
- `[Repo]` It does not yet have a robust corpus-qualified identity model.
- `[Repo]` Comparative mechanism-level understanding is therefore not yet strong enough to be the backbone of "disparate relations" across repos.

The correct answer is therefore:

- breadth: already strong,
- single-corpus semantics: promising,
- full function-interior comparative understanding across heterogeneous repos: not yet.

That gap is exactly what this roadmap closes.

---

## 7. Synthesis

### Strongest ideas overall

1. Corpus Session Ledger
2. Corpus-Qualified Symbol Identity
3. Function-Interior Semantic Lift
4. Disparate Relations Omnigraph
5. Copy-vs-Rewrite Arbiter

### Best balance of novelty and plausibility

1. Disparate Relations Omnigraph
2. Repo Twin Report Compiler
3. EID Comparative Frontier Planner
4. Federated Query Gateway

### Most feasible now

Corpus Session Ledger plus Scoped Subtree Indexing.

Reason:

- `[Repo]` `index(path=...)` already exists.
- `[Repo]` state and registry infrastructure already exist.
- `[Repo]` the operational need is explicit and immediate.

### Biggest long-term moat bet

Disparate Relations Omnigraph backed by corpus-qualified identity and native similarity kernels.

Reason:

- `[Synthesis]` This is where Saguaro stops being "a good local code intelligence tool" and becomes "a comparative mechanism discovery system."

### Cleanest unification with the current codebase

The promoted program should unify around one architectural center:

- corpus identity,
- comparative graph evidence,
- EID consumption,
- report and roadmap promotion.

That center of gravity fits existing code better than bolting on a separate repo-analysis subsystem.

### What should be prototyped first

Prototype order by conviction:

1. `intake`: corpus sessions plus scoped subtree indexing.
2. `analysis_upgrade`: corpus-qualified symbol identity.
3. `analysis_upgrade`: relation overlay for one relation class only.
4. `eid`: copy-vs-rewrite arbitration on top of that narrow relation set.
5. `convergence`: comparative report compiler.

### Ranking by conviction

High conviction:

- Corpus Session Ledger
- Scoped Subtree Indexing
- Corpus-Qualified Symbol Identity
- Copy-vs-Rewrite Arbiter
- EID Comparative Frontier Planner

Medium-high conviction:

- Function-Interior Semantic Lift
- Disparate Relations Omnigraph
- Repo Twin Report Compiler
- Federated Query Gateway

Medium conviction:

- Build Fingerprint and Toolchain Capture
- Evidence-Backed Port Ledger
- Batch `repo_analysis/` Fleet Sweep
- Native Similarity Kernels

Lower initial conviction, but strategically interesting:

- Native Partial Clone and Lazy Materialization
- Learned Relation Embedding Layer
- Integration Wind Tunnel
- Self-Healing Comparative Workbench

---

## 8. Implementation Program

This section promotes the strongest candidates into an ordered roadmap using the canonical phase ids already used by the repo.

### Phase `intake` - External Corpus Identity and Scoped Indexing

`phase_id`: `intake`  
Phase title: External Corpus Identity and Scoped Indexing  
Objective: make external repos and subtrees first-class corpora with isolated manifests, explicit identity, scope cones, and quarantine-aware lifecycle semantics.  
Dependencies: none.  
Repo scope: `["primary:Anvil", "subtree:Saguaro", "external:repo_analysis/*"]`  
Owning specialist type: `["saguaro-state-architecture", "saguaro-indexing"]`  
Allowed writes: `["Saguaro/saguaro/api.py", "Saguaro/saguaro/state/ledger.py", "Saguaro/saguaro/utils/file_utils.py", "Saguaro/saguaro/query/corpus_rules.py", "standards/scan_exclusion_policy.yaml", "domains/code_intelligence/saguaro_substrate.py"]`  
Telemetry contract: emit `corpus_session_count`, `corpus_session_boot_ms`, `focus_corpus_file_count`, `focus_cone_expansion_edges`, `quarantined_corpus_count`, `expired_corpus_gc_ms`.  
Required evidence: corpus session manifests, scoped manifest diffs, quarantine logs, isolated query traces.  
Rollback criteria: any temporary corpus mutates primary corpus state; scoped indexing silently includes excluded roots; workspace sync semantics regress.  
Promotion gate: operator can create isolated corpora for `.`, `Saguaro/`, and one external repo under `repo_analysis/`, then query each separately.  
Success criteria: corpus ids, scope roots, and trust levels are explicit in session state and query responses.  
Exact wiring points: `Saguaro/saguaro/api.py`, `Saguaro/saguaro/state/ledger.py`, `Saguaro/saguaro/utils/file_utils.py`, `Saguaro/saguaro/query/corpus_rules.py`, `domains/code_intelligence/saguaro_substrate.py`.  
Deliverables: corpus session manifest, scope cone policy, TTL and quarantine rules, operator-facing session commands.  
Tests: `tests/test_saguaro_interface.py`, `tests/test_state_ledger.py`, `tests/test_saguaro_corpus_manifest.py`, `tests/test_saguaro_query_accuracy.py`, `tests/test_saguaro_incremental_identity.py`.  
Verification commands:

- `pytest tests/test_saguaro_interface.py tests/test_state_ledger.py tests/test_saguaro_corpus_manifest.py tests/test_saguaro_query_accuracy.py tests/test_saguaro_incremental_identity.py`
- `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- `./venv/bin/saguaro build-graph`

Exit criteria: subtree and external corpus analysis become normal, isolated operations rather than ad hoc path filtering.

### Phase `research` - Build-Aware Comparative Corpus Acquisition

`phase_id`: `research`  
Phase title: Build-Aware Comparative Corpus Acquisition  
Objective: turn repo acquisition into a build-aware, language-aware, capability-aware comparative intake pipeline for local and remote analysis repos.  
Dependencies: `intake`.  
Repo scope: `["primary:Anvil", "external:repo_analysis/*"]`  
Owning specialist type: `["build-intelligence", "parser-governance", "fleet-analysis"]`  
Allowed writes: `["core/research/repo_acquisition.py", "core/research/analysis_pack.py", "core/campaign/repo_cache.py", "Saguaro/saguaro/build_system/ingestor.py", "Saguaro/saguaro/parsing/parser.py", "Saguaro/saguaro/analysis/report.py"]`  
Telemetry contract: emit `build_profile_detected`, `compiled_language_precision_rate`, `language_capability_coverage`, `fleet_repo_count`, `fleet_scan_wall_ms`.  
Required evidence: build fingerprint records, parser capability matrices, fleet scan summaries, comparative repo dossiers with structured evidence.  
Rollback criteria: intake becomes too slow for ordinary repos; build capture blocks lightweight scans; reports misrepresent unsupported languages as deep coverage.  
Promotion gate: system can ingest one compiled external repo, one interpreted repo, and one mixed-language repo with explicit capability output.  
Success criteria: every comparative corpus has a build profile, capability matrix, and structured dossier instead of a shallow token scan.  
Exact wiring points: `core/research/repo_acquisition.py`, `core/research/analysis_pack.py`, `core/campaign/repo_cache.py`, `Saguaro/saguaro/build_system/ingestor.py`, `Saguaro/saguaro/parsing/parser.py`, `Saguaro/saguaro/analysis/report.py`.  
Deliverables: build fingerprint capture, parser capability matrix, enhanced analysis pack schema, fleet sweep intake.  
Tests: `tests/test_saguaro_build_ingestor.py`, `tests/test_saguaro_parser_languages.py`, `tests/test_analysis_flows.py`, `tests/test_dare_web_research.py`, `tests/test_repo_campaign_subagent.py`.  
Verification commands:

- `pytest tests/test_saguaro_build_ingestor.py tests/test_saguaro_parser_languages.py tests/test_analysis_flows.py tests/test_dare_web_research.py tests/test_repo_campaign_subagent.py`
- `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- `./venv/bin/saguaro entrypoints`

Exit criteria: the system knows what it can understand deeply, what it understands shallowly, and what build assumptions shape that understanding.

### Phase `analysis_upgrade` - Comparative Program Semantics and Disparate Relations

`phase_id`: `analysis_upgrade`  
Phase title: Comparative Program Semantics and Disparate Relations  
Objective: give Saguaro corpus-qualified symbol identity, deeper function-interior semantics, and a comparative relation overlay able to identify analogous mechanisms across repos and subtrees.  
Dependencies: `intake`, `research`.  
Repo scope: `["primary:Anvil", "subtree:Saguaro", "external:repo_analysis/*"]`  
Owning specialist type: `["saguaro-symbolics", "program-semantics", "graph-architecture", "comparative-intelligence"]`  
Allowed writes: `["Saguaro/saguaro/parsing/runtime_symbols.py", "Saguaro/saguaro/agents/perception.py", "Saguaro/saguaro/parsing/parser.py", "Saguaro/saguaro/analysis/icfg_builder.py", "Saguaro/saguaro/analysis/code_graph.py", "Saguaro/saguaro/analysis/bridge_synthesizer.py", "Saguaro/saguaro/services/platform.py", "Saguaro/saguaro/reality/store.py", "core/dare/repo_ingestion.py"]`  
Telemetry contract: emit `symbol_collision_count`, `function_body_graph_count`, `deep_slice_latency_ms`, `cross_corpus_overlay_edges`, `disparate_relation_count`, `relation_precision_audit_score`.  
Required evidence: qualified symbol traces, deep slice examples, relation review sets, comparative graph artifacts, false-positive audit samples.  
Rollback criteria: graph and relation growth overwhelms runtime budgets or precision remains poor under manual review.  
Promotion gate: one comparative report can correctly identify analogous mechanisms between a target subtree and an external repo with explicit graph evidence.  
Success criteria: Saguaro can answer mechanism-level questions, not just file-level similarity questions, across corpora.  
Exact wiring points: `Saguaro/saguaro/parsing/runtime_symbols.py`, `Saguaro/saguaro/agents/perception.py`, `Saguaro/saguaro/parsing/parser.py`, `Saguaro/saguaro/analysis/icfg_builder.py`, `Saguaro/saguaro/analysis/code_graph.py`, `Saguaro/saguaro/analysis/bridge_synthesizer.py`, `Saguaro/saguaro/services/platform.py`, `Saguaro/saguaro/reality/store.py`, `core/dare/repo_ingestion.py`.  
Deliverables: qualified symbol ids, ambiguity resolver, deep slices, comparative overlay relations, initial disparate-relation engine.  
Tests: `tests/test_saguaro_graph_resolution.py`, `tests/test_runtime_symbol_federation.py`, `tests/test_saguaro_code_graph.py`, `tests/test_saguaro_omnigraph.py`, `tests/test_saguaro_cfg_builder.py`, `tests/test_saguaro_dfg_enhancements.py`.  
Verification commands:

- `pytest tests/test_saguaro_graph_resolution.py tests/test_runtime_symbol_federation.py tests/test_saguaro_code_graph.py tests/test_saguaro_omnigraph.py tests/test_saguaro_cfg_builder.py tests/test_saguaro_dfg_enhancements.py`
- `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- `./venv/bin/saguaro impact --path core/dare/repo_ingestion.py`

Exit criteria: comparative semantics are trustworthy enough to drive promotion decisions.

### Phase `eid` - Comparative Frontier and Rewrite Arbitration

`phase_id`: `eid`  
Phase title: Comparative Frontier and Rewrite Arbitration  
Objective: feed comparative evidence into the EID loop as structured frontier packets and arbitration-backed migration options rather than shallow repo dossiers.  
Dependencies: `analysis_upgrade`.  
Repo scope: `["primary:Anvil", "subtree:Saguaro", "external:repo_analysis/*"]`  
Owning specialist type: `["eid-comparative-planner", "eid-migration-arbiter", "loop-integration"]`  
Allowed writes: `["core/research/eid_master.py", "core/campaign/control_plane.py", "core/research/analysis_pack.py", "core/dare/repo_ingestion.py", "core/memory/fabric/jobs.py", "core/qsg/latent_bridge.py", "core/unified_chat_loop.py", "domains/code_intelligence/saguaro_substrate.py"]`  
Telemetry contract: emit `comparative_frontier_count`, `arbiter_decision_count`, `native_rewrite_fraction`, `eid_repo_evidence_rate`, `comparative_tool_call_count`.  
Required evidence: frontier packets, arbitration matrices, EID output deltas, comparative tool traces, promoted experiment designs.  
Rollback criteria: EID becomes noisier or loses actionability; arbitration posture lacks evidence or reverses frequently.  
Promotion gate: EID must emit at least one experiment or implementation track with a clear copy, wrap, rewrite, or reject posture backed by comparative evidence.  
Success criteria: external repo intelligence changes the system's R&D behavior in a traceable way.  
Exact wiring points: `core/research/eid_master.py`, `core/campaign/control_plane.py`, `core/research/analysis_pack.py`, `core/dare/repo_ingestion.py`, `core/memory/fabric/jobs.py`, `core/qsg/latent_bridge.py`, `core/unified_chat_loop.py`, `domains/code_intelligence/saguaro_substrate.py`.  
Deliverables: frontier packets, arbitration matrix, EID comparative planner, loop/tool routing integration.  
Tests: `tests/test_campaign_eid_exchange.py`, `tests/test_campaign_eid_portfolio_scheduler.py`, `tests/test_campaign_research_frontier.py`, `tests/test_unified_chat_loop.py`, `tests/test_campaign_control_kernel.py`.  
Verification commands:

- `pytest tests/test_campaign_eid_exchange.py tests/test_campaign_eid_portfolio_scheduler.py tests/test_campaign_research_frontier.py tests/test_unified_chat_loop.py tests/test_campaign_control_kernel.py`
- `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- `./venv/bin/saguaro impact --path core/research/eid_master.py`

Exit criteria: repo comparison becomes a native input to the R&D loop.

### Phase `development` - Federated Runtime and Native Comparative Acceleration

`phase_id`: `development`  
Phase title: Federated Runtime and Native Comparative Acceleration  
Objective: provide one runtime surface for local, subtree, and external corpora, and move compute-heavy comparative scoring onto the native boundary where justified.  
Dependencies: `intake`, `analysis_upgrade`.  
Repo scope: `["primary:Anvil", "subtree:Saguaro", "external:repo_analysis/*"]`  
Owning specialist type: `["query-federation", "native-graph-runtime", "operator-experience"]`  
Allowed writes: `["Saguaro/saguaro/query/gateway.py", "Saguaro/saguaro/api.py", "Saguaro/saguaro/state/ledger.py", "Saguaro/src/ops/native_indexer_api.cc", "Saguaro/saguaro/indexing/native_indexer_bindings.py", "Saguaro/saguaro/storage/native_vector_store.py", "Saguaro/saguaro/cli.py", "cli/repl.py", "domains/code_intelligence/saguaro_substrate.py"]`  
Telemetry contract: emit `federated_query_ms`, `corpus_fanout_count`, `native_relation_score_ms`, `native_relation_ram_mb`, `comparative_cli_invocations`, `gateway_queue_depth`.  
Required evidence: federated query traces, native benchmark results, CLI and REPL transcripts, fallback parity checks.  
Rollback criteria: ordinary single-corpus performance regresses; native path diverges semantically from Python reference path; operator surface becomes confusing.  
Promotion gate: gateway can service primary, subtree, single external, and fleet queries with explicit corpus routing, while native scorer beats Python on at least one hot path.  
Success criteria: comparative analysis is performant enough to become routine.  
Exact wiring points: `Saguaro/saguaro/query/gateway.py`, `Saguaro/saguaro/api.py`, `Saguaro/saguaro/state/ledger.py`, `Saguaro/src/ops/native_indexer_api.cc`, `Saguaro/saguaro/indexing/native_indexer_bindings.py`, `Saguaro/saguaro/storage/native_vector_store.py`, `Saguaro/saguaro/cli.py`, `cli/repl.py`, `domains/code_intelligence/saguaro_substrate.py`.  
Deliverables: federated gateway, corpus-aware query routes, native relation scoring kernels, comparative CLI and REPL surfaces.  
Tests: `tests/test_saguaro_query_gateway.py`, `tests/test_saguaro_query_broker.py`, `tests/test_saguaro_interface.py`, `tests/test_saguaro_native_vector_store.py`, `tests/test_native_compute_contract.py`, `tests/test_interactive_cli.py`.  
Verification commands:

- `pytest tests/test_saguaro_query_gateway.py tests/test_saguaro_query_broker.py tests/test_saguaro_interface.py tests/test_saguaro_native_vector_store.py tests/test_native_compute_contract.py tests/test_interactive_cli.py`
- `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- `./venv/bin/saguaro build-graph`

Exit criteria: comparative analysis no longer depends on one-off shell choreography or impractical Python-only hot paths.

### Phase `deep_test_audit` - Precision, Budget, and Reliability Gates

`phase_id`: `deep_test_audit`  
Phase title: Precision, Budget, and Reliability Gates  
Objective: harden the comparative stack with accuracy audits, latency and memory budgets, health-path reliability, and anti-pollution verification.  
Dependencies: `development`.  
Repo scope: `["primary:Anvil", "external:repo_analysis/*", "subtree:Saguaro"]`  
Owning specialist type: `["comparative-assurance"]`  
Allowed writes: `["Saguaro/saguaro/fastpath.py", "Saguaro/saguaro/health.py", "Saguaro/saguaro/api.py", "Saguaro/saguaro/roadmap/validator.py", "core/campaign/roadmap_validator.py"]`  
Telemetry contract: emit `health_path_pass_rate`, `relation_precision_audit_score`, `federated_query_p95_ms`, `comparative_ram_mb`, `primary_corpus_pollution_count`.  
Required evidence: passing health command, audited relation review sets, latency and RAM budgets, roadmap validation output, zero-pollution proof after temporary corpus cleanup.  
Rollback criteria: comparative features remain interesting but unverifiable, or health and reliability remain flaky.  
Promotion gate: `saguaro health` passes again, relation precision is measured, and temporary corpus cleanup is proven not to pollute the main index.  
Success criteria: the comparative stack is safe enough to become default infrastructure for campaign R&D.  
Exact wiring points: `Saguaro/saguaro/fastpath.py`, `Saguaro/saguaro/health.py`, `Saguaro/saguaro/api.py`, `Saguaro/saguaro/roadmap/validator.py`, `core/campaign/roadmap_validator.py`.  
Deliverables: fixed health path, comparative benchmark suite, precision audits, budget gates, validator-friendly evidence.  
Tests: `Saguaro/tests/test_health_dashboard.py`, `tests/test_saguaro_platform_foundation.py`, `tests/test_saguaro_query_latency.py`, `tests/test_perf_regression.py`, `tests/test_saguaro_roadmap_validator.py`.  
Verification commands:

- `pytest Saguaro/tests/test_health_dashboard.py tests/test_saguaro_platform_foundation.py tests/test_saguaro_query_latency.py tests/test_perf_regression.py tests/test_saguaro_roadmap_validator.py`
- `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- `./venv/bin/saguaro health`

Exit criteria: the comparative subsystem meets reliability standards instead of living as an experimental side path.

### Phase `convergence` - Comparative Report Compiler and Port Ledger

`phase_id`: `convergence`  
Phase title: Comparative Report Compiler and Port Ledger  
Objective: compile comparative findings into durable repo twin reports, migration recipes, and an evidence-backed port ledger that can flow into roadmap and implementation work.  
Dependencies: `eid`, `deep_test_audit`.  
Repo scope: `["primary:Anvil", "subtree:Saguaro", "external:repo_analysis/*"]`  
Owning specialist type: `["comparative-reporting", "evidence-governance", "comparative-mission-control"]`  
Allowed writes: `["Saguaro/saguaro/analysis/report.py", "core/campaign/control_plane.py", "core/research/eid_master.py", "Saguaro/saguaro/reality/store.py", "Saguaro/saguaro/roadmap/validator.py", "core/campaign/phase_packet.py"]`  
Telemetry contract: emit `report_compile_ms`, `report_evidence_density`, `port_candidate_count`, `port_promotion_rate`, `evidence_closure_rate`.  
Required evidence: repo twin reports, recipe packets, ledger snapshots, roadmap validation traces.  
Rollback criteria: convergence artifacts are verbose but not operationally useful, or they duplicate existing campaign artifacts without improving traceability.  
Promotion gate: one complete comparative mission must yield a report, recipe set, ledger entries, and a promotable roadmap packet.  
Success criteria: comparative repo work exits as a governed implementation pipeline rather than a dead-end research note.  
Exact wiring points: `Saguaro/saguaro/analysis/report.py`, `core/campaign/control_plane.py`, `core/research/eid_master.py`, `Saguaro/saguaro/reality/store.py`, `Saguaro/saguaro/roadmap/validator.py`, `core/campaign/phase_packet.py`.  
Deliverables: repo twin report compiler, migration recipes, port ledger, roadmap-aligned promotion packets.  
Tests: `tests/test_campaign_roadmap_phase_pack.py`, `tests/test_saguaro_roadmap_validator.py`, `tests/test_saguaro_traceability.py`, `tests/test_campaign_state_store_phase_artifacts.py`, `tests/test_campaign_artifact_registry.py`.  
Verification commands:

- `pytest tests/test_campaign_roadmap_phase_pack.py tests/test_saguaro_roadmap_validator.py tests/test_saguaro_traceability.py tests/test_campaign_state_store_phase_artifacts.py tests/test_campaign_artifact_registry.py`
- `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- `./venv/bin/saguaro roadmap validate --path saguaro_deep_analysis_roadmap.md --format json`

Exit criteria: comparative analysis outputs are promotion-ready roadmap inputs, not orphaned artifacts.

---

## 9. Implementation Contract

- The system shall implement isolated corpus session identity through `Saguaro/saguaro/api.py`, `Saguaro/saguaro/state/ledger.py`, `domains/code_intelligence/saguaro_substrate.py`, and `core/campaign/repo_registry.py`, tested by `tests/test_saguaro_interface.py`, `tests/test_state_ledger.py`, and `tests/test_saguaro_incremental_identity.py`, and verified with `pytest tests/test_saguaro_interface.py tests/test_state_ledger.py tests/test_saguaro_incremental_identity.py`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`, and `./venv/bin/saguaro build-graph`.

- The system shall implement subtree-scoped corpus indexing and focus-cone manifest control through `Saguaro/saguaro/api.py`, `Saguaro/saguaro/utils/file_utils.py`, `Saguaro/saguaro/query/corpus_rules.py`, and `Saguaro/saguaro/indexing/engine.py`, tested by `tests/test_saguaro_corpus_manifest.py`, `tests/test_saguaro_query_accuracy.py`, and `tests/test_saguaro_interface.py`, and verified with `pytest tests/test_saguaro_corpus_manifest.py tests/test_saguaro_query_accuracy.py tests/test_saguaro_interface.py`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`, and `./venv/bin/saguaro impact --path Saguaro/saguaro/api.py`.

- The system shall preserve primary-index isolation for temporary external corpora through `Saguaro/saguaro/query/corpus_rules.py`, `standards/scan_exclusion_policy.yaml`, `Saguaro/saguaro/state/ledger.py`, and `core/campaign/repo_registry.py`, tested by `tests/test_saguaro_query_accuracy.py`, `tests/test_governance_trust_zones.py`, and `tests/test_state_ledger.py`, and verified with `pytest tests/test_saguaro_query_accuracy.py tests/test_governance_trust_zones.py tests/test_state_ledger.py`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`, and `./venv/bin/saguaro health`.

- The system shall implement corpus-qualified symbol identity and ambiguity-safe slicing through `Saguaro/saguaro/parsing/runtime_symbols.py`, `Saguaro/saguaro/agents/perception.py`, `Saguaro/saguaro/api.py`, and `core/unified_chat_loop.py`, tested by `tests/test_saguaro_graph_resolution.py`, `tests/test_runtime_symbol_federation.py`, and `tests/test_unified_chat_loop.py`, and verified with `pytest tests/test_saguaro_graph_resolution.py tests/test_runtime_symbol_federation.py tests/test_unified_chat_loop.py`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`, and `./venv/bin/saguaro impact --path Saguaro/saguaro/agents/perception.py`.

- The system shall expose parser and build-confidence capability matrices for each comparative corpus through `Saguaro/saguaro/parsing/parser.py`, `Saguaro/saguaro/build_system/ingestor.py`, `Saguaro/saguaro/health.py`, and `Saguaro/saguaro/analysis/report.py`, tested by `tests/test_saguaro_parser_languages.py`, `tests/test_saguaro_build_ingestor.py`, and `Saguaro/tests/test_health_dashboard.py`, and verified with `pytest tests/test_saguaro_parser_languages.py tests/test_saguaro_build_ingestor.py Saguaro/tests/test_health_dashboard.py`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`, and `./venv/bin/saguaro entrypoints`.

- The system shall upgrade repository acquisition and dossier generation into a structured comparative intake pipeline through `core/research/repo_acquisition.py`, `core/research/analysis_pack.py`, `core/campaign/repo_cache.py`, and `core/dare/repo_ingestion.py`, tested by `tests/test_analysis_flows.py`, `tests/test_dare_web_research.py`, and `tests/test_repo_campaign_subagent.py`, and verified with `pytest tests/test_analysis_flows.py tests/test_dare_web_research.py tests/test_repo_campaign_subagent.py`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`, and `./venv/bin/saguaro impact --path core/dare/repo_ingestion.py`.

- The system shall implement function-interior comparative semantics through `Saguaro/saguaro/parsing/parser.py`, `Saguaro/saguaro/analysis/icfg_builder.py`, `Saguaro/saguaro/analysis/code_graph.py`, and `Saguaro/saguaro/services/platform.py`, tested by `tests/test_saguaro_cfg_builder.py`, `tests/test_saguaro_dfg_enhancements.py`, and `tests/test_saguaro_code_graph.py`, and verified with `pytest tests/test_saguaro_cfg_builder.py tests/test_saguaro_dfg_enhancements.py tests/test_saguaro_code_graph.py`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`, and `./venv/bin/saguaro impact --path Saguaro/saguaro/analysis/code_graph.py`.

- The system shall implement a comparative disparate-relations overlay through `Saguaro/saguaro/analysis/bridge_synthesizer.py`, `Saguaro/saguaro/services/platform.py`, `Saguaro/saguaro/reality/store.py`, and `core/dare/repo_ingestion.py`, tested by `tests/test_saguaro_omnigraph.py`, `tests/test_saguaro_code_graph.py`, and `tests/test_saguaro_traceability.py`, and verified with `pytest tests/test_saguaro_omnigraph.py tests/test_saguaro_code_graph.py tests/test_saguaro_traceability.py`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`, and `./venv/bin/saguaro impact --path Saguaro/saguaro/services/platform.py`.

- The system shall implement copy-vs-rewrite arbitration and comparative frontier planning through `core/research/eid_master.py`, `core/campaign/control_plane.py`, `core/research/analysis_pack.py`, and `core/dare/repo_ingestion.py`, tested by `tests/test_campaign_eid_exchange.py`, `tests/test_campaign_eid_portfolio_scheduler.py`, and `tests/test_campaign_research_frontier.py`, and verified with `pytest tests/test_campaign_eid_exchange.py tests/test_campaign_eid_portfolio_scheduler.py tests/test_campaign_research_frontier.py`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`, and `./venv/bin/saguaro impact --path core/research/eid_master.py`.

- The system shall route comparative analysis through campaign and chat loops using explicit tool surfaces in `core/campaign/control_plane.py`, `core/unified_chat_loop.py`, `domains/code_intelligence/saguaro_substrate.py`, and `Saguaro/saguaro/cli.py`, tested by `tests/test_campaign_control_kernel.py`, `tests/test_unified_chat_loop.py`, and `tests/test_interactive_cli.py`, and verified with `pytest tests/test_campaign_control_kernel.py tests/test_unified_chat_loop.py tests/test_interactive_cli.py`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`, and `./venv/bin/saguaro entrypoints`.

- The system shall provide a federated corpus-aware query runtime through `Saguaro/saguaro/query/gateway.py`, `Saguaro/saguaro/api.py`, `Saguaro/saguaro/state/ledger.py`, and `domains/code_intelligence/saguaro_substrate.py`, tested by `tests/test_saguaro_query_gateway.py`, `tests/test_saguaro_query_broker.py`, and `tests/test_saguaro_interface.py`, and verified with `pytest tests/test_saguaro_query_gateway.py tests/test_saguaro_query_broker.py tests/test_saguaro_interface.py`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`, and `./venv/bin/saguaro build-graph`.

- The system shall accelerate compute-heavy comparative scoring through `Saguaro/src/ops/native_indexer_api.cc`, `Saguaro/saguaro/indexing/native_indexer_bindings.py`, `Saguaro/saguaro/storage/native_vector_store.py`, and `Saguaro/saguaro/services/platform.py`, tested by `tests/test_native_compute_contract.py`, `tests/test_saguaro_native_vector_store.py`, and `tests/test_perf_regression.py`, and verified with `pytest tests/test_native_compute_contract.py tests/test_saguaro_native_vector_store.py tests/test_perf_regression.py`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`, and `./venv/bin/saguaro build-graph`.

- The system shall restore health-path correctness for comparative runtime governance through `Saguaro/saguaro/fastpath.py`, `Saguaro/saguaro/health.py`, and `Saguaro/saguaro/api.py`, tested by `Saguaro/tests/test_health_dashboard.py`, `tests/test_saguaro_platform_foundation.py`, and `tests/test_saguaro_query_latency.py`, and verified with `pytest Saguaro/tests/test_health_dashboard.py tests/test_saguaro_platform_foundation.py tests/test_saguaro_query_latency.py`, `./venv/bin/saguaro health`, and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.

- The system shall compile comparative repo twin reports and an evidence-backed port ledger through `Saguaro/saguaro/analysis/report.py`, `core/campaign/control_plane.py`, `core/research/eid_master.py`, `Saguaro/saguaro/reality/store.py`, `Saguaro/saguaro/roadmap/validator.py`, and `core/campaign/phase_packet.py`, tested by `tests/test_campaign_roadmap_phase_pack.py`, `tests/test_saguaro_roadmap_validator.py`, `tests/test_saguaro_traceability.py`, and `tests/test_campaign_artifact_registry.py`, and verified with `pytest tests/test_campaign_roadmap_phase_pack.py tests/test_saguaro_roadmap_validator.py tests/test_saguaro_traceability.py tests/test_campaign_artifact_registry.py`, `./venv/bin/saguaro roadmap validate --path saguaro_deep_analysis_roadmap.md --format json`, and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
