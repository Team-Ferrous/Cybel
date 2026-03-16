# Engineering Inventive Designer (R&D) Loop: Inventive Research Roadmap

## 1. First-Principles Framing

### 1.1 What the system fundamentally is

Anvil is not fundamentally a chat application.

It is a local-first engineering control system that:

- ingests ambiguous objectives,
- converts them into campaign state,
- accumulates machine-readable evidence,
- spawns bounded specialist work,
- records telemetry and artifacts,
- promotes or rejects work through explicit gates, and
- attempts to close the loop with verification rather than prose alone.

[Repo] That is already visible in `core/campaign/control_plane.py`, `core/campaign/base_campaign.py`, `core/campaign/roadmap_compiler.py`, `core/campaign/gate_engine.py`, `core/campaign/transition_policy.py`, `core/campaign/state_store.py`, `core/research/eid_master.py`, `core/research/experiment_runner.py`, `shared_kernel/event_store.py`, and `domains/verification/verification_lane.py`.

[Repo] The most important existing fact is that Anvil already has the skeleton of an engineering operating system:

- a persistent campaign ledger in `core/campaign/state_store.py`,
- artifact families and approval states in `core/campaign/artifact_registry.py`,
- a deterministic transition policy in `core/campaign/transition_policy.py`,
- a typed roadmap compiler in `core/campaign/roadmap_compiler.py`,
- memory projection and latent capture in `core/memory/fabric/*` and `core/qsg/latent_bridge.py`,
- a replay-oriented event log in `shared_kernel/event_store.py`,
- experiment execution with telemetry contracts in `core/research/experiment_runner.py`,
- and a validator surface in `core/campaign/roadmap_validator.py` plus `saguaro roadmap validate`.

### 1.2 What the system appears to be trying to become

Anvil appears to be trying to become a campaign-centric engineering autonomy stack where:

- research is not free-form browsing but evidence production,
- EID is not brainstorming but bounded invention portfolio management,
- roadmap generation is not a markdown dump but a typed promotion artifact,
- development is not an open-ended coding loop but a series of verifiable experiment lanes,
- and closure is not "done" but an evidence-backed proof that the right work was executed and survived verification.

[Inference] The strongest latent ambition in the repo is not "more agents." It is a stronger center of gravity where campaign state, memory, telemetry, and verification become one continuous state machine instead of adjacent modules.

### 1.3 Core constraints that should shape the design space

If this is engineered to NASA / Formula 1 standards, the governing constraints are:

1. Bounded autonomy.
   The loop must have explicit cost, blast-radius, and stopping semantics.

2. Verifiable traceability.
   The system must emit enough typed evidence that an operator can reconstruct why a promotion happened.

3. Local-model realism.
   The repo runs local Ollama weights through a QSG pipeline, so the design must assume scarce reasoning budget, non-frontier model reliability, and a strong need for staged compression and verification.

4. Multi-fidelity execution.
   Not every idea deserves a full run. Strong systems downshift between cheap simulation, medium-fidelity dry run, and high-fidelity execution.

5. Graph truth over string intuition.
   Campaigns need higher-confidence structure than heuristics on topic strings and filenames.

6. Deterministic governance.
   Human override, replay, risk limits, and promotion gates must be first-class, not ad hoc.

7. Hardware-aware performance.
   The system already carries native telemetry and CPU-first runtime ambitions; EID should exploit that instead of behaving like a cloud-agent template.

8. Evidence closure.
   Every promoted workstream should tighten the evidence graph, not just produce more text.

### 1.4 Design stance

[Inference] The correct mental model is:

- `research` should function like active sensing,
- `eid` should function like a portfolio optimizer and experiment designer,
- `roadmap_draft` should function like a contract compiler,
- `development` should function like a bounded execution market,
- `convergence` should function like a safety-case assembler.

That means the upgrade target is not "add more features to EID."

It is "turn campaign + EID into a multi-fidelity engineering control loop with evidence economics, counterfactual simulation, and verifier-coupled promotion."

## 2. External Research Scan

### 2.1 Most relevant external patterns

[External] **Safety-bounded autonomy from aerospace software discipline.**
Gerard Holzmann's "Power of 10 Rules for Developing Safety Critical Code" argues for explicit structural limits, simple control flow, and analyzable behavior rather than post hoc heroics. This matters here because Anvil's campaign and EID loops are effectively high-level control loops with real cost and state consequences, yet current EID logic is still mostly heuristic and open-ended. Source: [R1](https://spinroot.com/p10/).

[External] **Durable execution and interrupts as first-class orchestration primitives.**
LangGraph's durable execution and interrupt model treats pause/resume, checkpointing, and human intervention as part of the runtime contract rather than a failure case. This matters because Anvil already has campaign workspaces, event logs, and state transitions; it lacks a stronger interruptible execution ABI around EID and speculative lanes. Sources: [R2](https://docs.langchain.com/oss/python/langgraph/durable-execution), [R3](https://docs.langchain.com/oss/python/langgraph/interrupts).

[External] **Open-source software agents are converging on eventful, sandboxed, benchmarked execution.**
OpenHands and SWE-agent both treat software work as an instrumented execution problem rather than prompt-only reasoning. This matters because Anvil already has stronger local control-plane primitives than either, but its EID branch does not yet exploit them deeply. Sources: [R4](https://github.com/All-Hands-AI/OpenHands), [R5](https://github.com/SWE-agent/SWE-agent).

[External] **Graph-based retrieval is becoming the practical answer to long-context drift.**
Microsoft's GraphRAG formalizes retrieval around entities, relationships, and community summaries rather than flat chunks. This matters because Anvil already has Memory Fabric, repo dossiers, topic clusters, and a state ledger, but EID still mostly ranks with keyword bonuses and static evidence lists. Source: [R6](https://github.com/microsoft/graphrag).

[External] **Translation validation beats "trust the optimizer."**
Alive2 shows a powerful pattern: do aggressive transformations, but require a checker that proves the transformed artifact preserves intended behavior. This matters because Anvil needs the same idea for prompt-to-task lowering, roadmap promotion, and speculative branch adoption. Source: [R7](https://github.com/AliveToolkit/alive2).

[External] **Driver-in-the-loop simulation is about rejecting bad trajectories before they hit the expensive path.**
rFpro's real-time driver-in-the-loop positioning reflects F1's larger operating model: simulate repeatedly, commit selectively, and let telemetry choose what graduates to real execution. This matters because EID already has `SimulatorPlanner`, `ExperimentDesignService`, and lane execution, but no true preflight lap. Source: [R8](https://rfpro.com/real-time-driver-in-the-loop/).

[External] **Engineering design loops increasingly use multi-fidelity active learning, not one-pass brainstorming.**
Recent active-learning and multi-fidelity surrogate papers in engineering design optimize which expensive evaluations to run next based on uncertainty and expected information gain. This matters because `CampaignControlPlane.run_research()` already calculates `yield_history` and `impact_deltas`; it just does not turn them into a formal acquisition strategy. Sources: [R9](https://arxiv.org/abs/2202.06902), [R10](https://arxiv.org/abs/2211.10360).

[External] **Local-model operations demand explicit routing and serving policies.**
Ollama's API surface and Olla's router demonstrate that self-hosted model stacks need explicit mediation across model size, latency, concurrency, and failure behavior. This matters because the user constraint is local Ollama through QSG, and EID should explicitly choose between cheap shadow models and expensive execution models. Sources: [R11](https://github.com/ollama/ollama/blob/main/docs/api.md), [R12](https://github.com/rmusser01/olla).

[External] **Practitioner incident threads around `llama.cpp` point to real local-serving footguns.**
Prompt caching, slot reuse, and speculative decoding are powerful but can quietly deform determinism and measurement if not instrumented. This matters because Anvil already cares about determinism, replay, and prompt protocol hashes; campaign telemetry should treat local serving behavior as part of the experiment contract. Sources: [R13](https://github.com/ggml-org/llama.cpp/issues/8947), [R14](https://github.com/ggml-org/llama.cpp/issues/3479).

[External] **Agent benchmarks are being challenged on realism and reward hacking.**
Recent work argues that black-box agent benchmarks overestimate real capability, and other work catalogs reward hacks where models satisfy the benchmark but not the operator's intention. This matters because Anvil's roadmap pipeline should not promote work from artifact presence alone; it needs stronger witness criteria and anti-cheating telemetry. Sources: [R15](https://arxiv.org/abs/2503.06745), [R16](https://arxiv.org/abs/2508.17511).

[External] **Multimodal and performance-focused software benchmarks expose a blind spot in current coding agents.**
SWE-bench Multimodal and PerfBench show that performance bugs, systems behavior, and non-textual evidence remain hard. This matters because Anvil is natively interested in CPU/QSG/runtime behavior; EID should lean into that gap rather than copying text-only agent patterns. Sources: [R17](https://arxiv.org/abs/2505.21542), [R18](https://arxiv.org/abs/2509.24091).

[External] **Prompt theft and prompt reconstruction are no longer hypothetical.**
Prompt Stealing Attacks show that prompt contracts and hidden instructions are operational surfaces, not magical secrets. This matters because Anvil already has AES prompt contracts and prompt hashes; campaign and EID artifacts should treat prompt protocol identity as a governance-relevant signal. Source: [R19](https://arxiv.org/abs/2402.12959).

[External] **NASA's digital twin push is about executable state alignment, not dashboards.**
NASA framing around digital twins emphasizes a stateful virtual counterpart used for prediction, risk, and operations. This matters because `RepoTwinBuilder` exists in the repo, but it is still closer to a snapshot artifact than a predictive twin that can steer EID. Source: [R20](https://www.nasa.gov/directorates/stmd/space-tech-research-grants/digital-twins-for-spaceflight-systems/).

### 2.2 Broad takeaways for Anvil

[Inference] The external pattern is consistent:

- strong systems reduce free-form exploration with bounded, interruptible execution,
- run cheap simulation before expensive execution,
- move more reasoning into typed state and explicit contracts,
- use graph structure and translation validation to keep aggressive automation honest,
- and treat telemetry as a control input, not a postmortem artifact.

[Inference] That combination maps unusually well onto this repo because Anvil already has:

- campaign state,
- artifact promotion,
- local memory projection,
- runtime event logs,
- verification lanes,
- and native telemetry surfaces.

The gap is not missing modules.

The gap is missing mechanism strength.

### 2.3 Reference Index

- [R1] Gerard J. Holzmann, "The Power of 10 Rules for Developing Safety Critical Code." https://spinroot.com/p10/
- [R2] LangGraph durable execution docs. https://docs.langchain.com/oss/python/langgraph/durable-execution
- [R3] LangGraph interrupts docs. https://docs.langchain.com/oss/python/langgraph/interrupts
- [R4] OpenHands GitHub repository. https://github.com/All-Hands-AI/OpenHands
- [R5] SWE-agent GitHub repository. https://github.com/SWE-agent/SWE-agent
- [R6] Microsoft GraphRAG GitHub repository. https://github.com/microsoft/graphrag
- [R7] Alive2 GitHub repository. https://github.com/AliveToolkit/alive2
- [R8] rFpro real-time driver-in-the-loop. https://rfpro.com/real-time-driver-in-the-loop/
- [R9] "A Multi-Fidelity Active Learning Method for Global Design Optimization Problems with Noisy Evaluations." https://arxiv.org/abs/2202.06902
- [R10] "Data efficient surrogate modeling for engineering design: Ensemble-free batch mode deep active learning for regression." https://arxiv.org/abs/2211.10360
- [R11] Ollama API docs. https://github.com/ollama/ollama/blob/main/docs/api.md
- [R12] Olla GitHub repository. https://github.com/rmusser01/olla
- [R13] `llama.cpp` practitioner discussion on prompt cache behavior. https://github.com/ggml-org/llama.cpp/issues/8947
- [R14] `llama.cpp` practitioner discussion on speculative decoding. https://github.com/ggml-org/llama.cpp/issues/3479
- [R15] "Beyond Black-box Benchmarking: Overestimation in Existing LLM Agent Benchmarks by Over 100%." https://arxiv.org/abs/2503.06745
- [R16] "School of Reward Hacks." https://arxiv.org/abs/2508.17511
- [R17] "SWE-bench Multimodal: Do AI Systems Generalize to Visual Software Domains?" https://arxiv.org/abs/2505.21542
- [R18] "PerfBench: A Benchmark for Performance Issues in Software Engineering." https://arxiv.org/abs/2509.24091
- [R19] "Prompt Stealing Attacks Against Text-to-Image Generation Models." https://arxiv.org/abs/2402.12959
- [R20] NASA digital twins for spaceflight systems. https://www.nasa.gov/directorates/stmd/space-tech-research-grants/digital-twins-for-spaceflight-systems/

## 3. Repo Grounding Summary

### 3.1 Commands run and signals observed

[Repo] Repo analysis was run from the repo virtual environment with:

- `source venv/bin/activate && saguaro health`
- `source venv/bin/activate && saguaro entrypoints`
- `source venv/bin/activate && saguaro build-graph`
- `source venv/bin/activate && saguaro query "campaign orchestration planning loop engineering inventive designer telemetry memory governance hooks" --k 10`
- `source venv/bin/activate && saguaro query "Engineering Inventive Designer EID campaign inventive research designer loop" --k 10`
- `source venv/bin/activate && saguaro agent skeleton ...`
- `source venv/bin/activate && saguaro impact --path core/campaign/control_plane.py`
- `source venv/bin/activate && saguaro impact --path core/research/eid_master.py`
- `source venv/bin/activate && saguaro verify . --engines native,ruff,semantic --format json`

[Repo] Saguaro findings that materially shape the roadmap:

- index freshness is acceptable and execution target is `cpu`,
- graph coverage is low at 8.2 percent with missing `cfg`, `dfg`, `call`, and `ffi_bridge` edge classes,
- `impact` currently returns zero meaningful downstream impact for `core/campaign/control_plane.py` and `core/research/eid_master.py`,
- `verify` currently passes with zero violations,
- build topology already exposes the Python control plane and native runtime targets,
- and entrypoint coverage is high, which means the operational surfaces are visible even if the call graph is weak.

### 3.2 Code paths inspected

[Repo] Campaign orchestration and state:

- `core/campaign/control_plane.py`
- `core/campaign/base_campaign.py`
- `core/campaign/transition_policy.py`
- `core/campaign/gate_engine.py`
- `core/campaign/loop_scheduler.py`
- `core/campaign/state_store.py`
- `core/campaign/workspace.py`
- `core/campaign/retrieval_policy.py`
- `core/campaign/risk_radar.py`
- `core/campaign/roadmap_compiler.py`
- `core/campaign/roadmap_validator.py`
- `core/campaign/phase_packet.py`
- `core/campaign/telemetry.py`
- `core/campaign/telemetry_contracts.py`

[Repo] EID and research loop:

- `core/research/eid_master.py`
- `core/research/hypothesis_lab.py`
- `core/research/hypothesis_ranker.py`
- `core/research/eid_scheduler.py`
- `core/research/experiment_design.py`
- `core/research/simulator_planner.py`
- `core/research/experiment_runner.py`
- `core/research/crawler.py`
- `core/research/store.py`
- `core/research/browser_runtime.py`

[Repo] Runtime, prompts, memory, and verification:

- `shared_kernel/event_store.py`
- `core/prompts/prompt_manager.py`
- `core/agent.py`
- `core/agent_tool_helpers.py`
- `core/native/runtime_telemetry.py`
- `core/qsg/runtime_contracts.py`
- `core/qsg/latent_bridge.py`
- `domains/verification/verification_lane.py`

### 3.3 Existing primitives that matter most

[Repo] `CampaignControlPlane.run_research()` already does more than a typical research loop:

- imports repo analysis packs,
- stores normalized external research,
- records claims and chunks,
- clusters topics,
- calculates stop conditions and stop proof,
- mirrors repo dossiers into memory,
- emits research artifacts,
- and records convergence checkpoints.

[Repo] `CampaignControlPlane.run_eid()` already does more than a typical ideation loop:

- loads campaign objective and research claims,
- generates hypotheses through `HypothesisLab`,
- passes repo dossiers into `EIDMasterLoop`,
- executes experiment lanes,
- emits a design review whitepaper,
- records a decision record,
- stores summary artifacts,
- and advances campaign state.

[Repo] `ExperimentRunner` is a stronger primitive than the current EID logic implies:

- it already captures experiments, runs, results, telemetry contracts, promotion policy, CPU scorecards, worktrees, memory projection, and latent packages.

[Repo] `RoadmapCompiler` is already a typed graph compiler, not a markdown formatter:

- it compiles features, questions, promotable hypotheses, experiment lanes, repo dossiers, and research clusters into `RoadmapItem`s,
- emits a task graph,
- renders phase packs,
- and validates phase packets.

[Repo] `EventStore` already behaves like a flight recorder:

- append-only events,
- event links,
- mission checkpoints,
- replay export with deterministic hash,
- and a dedicated QSG replay event stream.

### 3.4 Where the current architecture is strongest

[Repo] Strongest areas:

- explicit campaign state transitions,
- workspace artifactization,
- memory-fabric integration,
- experiment execution surfaces,
- typed roadmap items,
- and a coherent notion of artifact families and approval states.

[Inference] This is enough substrate to support a genuinely differentiated engineering loop.

### 3.5 Where the current architecture is underexploited or thin

[Repo] Thin areas:

- `HypothesisLab.generate()` is currently template-driven and only produces a few canned hypothesis shapes.
- `HypothesisRanker.rank()` is keyword- and bonus-based.
- `EIDScheduler.build_proposals()` creates deterministic packets, but their novelty and kill criteria are shallow.
- `ExperimentDesignService._commands_for()` mostly emits toy shell probes.
- `SimulatorPlanner.plan()` exists but is not yet the front door to a real shadow-execution regime.
- `RetrievalPolicyEngine.decide()` only chooses `saguaro` versus `fallback`; it does not optimize across internal evidence sources.
- `RoadmapValidator` checks only a minimal phase-packet schema.
- `saguaro impact` cannot yet guide roadmap blast-radius because graph confidence is too low.
- prompt assembly exists, but prompt protocol identity is not yet treated as a campaign-level invariant.

### 3.6 Architectural center of gravity the repo wants

[Inference] The repo wants a stronger middle layer where these things become one system:

- evidence economics,
- simulation fidelity selection,
- telemetry contracts,
- promotion gates,
- replay identity,
- and graph-backed blast-radius estimates.

[Inference] The correct upgrade is therefore not "improve EID prompts."

It is "move more of EID from heuristic text generation into typed experiment and promotion mechanics."

## 4. Hidden Assumptions

1. The EID loop assumes a small set of static hypothesis templates is enough to explore the design space.
2. The system assumes claim count and stop-proof heuristics are acceptable substitutes for explicit information gain.
3. The campaign assumes roadmap risk can remain mostly post-compilation instead of driving earlier branch pruning.
4. The repo assumes prompt assembly is separate from campaign governance instead of part of the same execution identity.
5. The current design assumes simulation planning can be advisory instead of being a gate before expensive lane execution.
6. The system assumes experiment lanes are single-shot tasks instead of members of a portfolio competing for limited budget.
7. The roadmap pipeline assumes artifact presence is a reasonable proxy for implementation progress.
8. The architecture assumes graph confidence can be low without materially harming promotion quality.
9. The control plane assumes repo twins are summaries, not predictive operating models.
10. The local-model stack is treated as an execution substrate, not a schedulable portfolio with explicit routing policy.

## 5. Candidate Implementation Phases

### 5.1 Research Frontier Economics Engine

- Suggested `phase_id`: `research`
- Type: Practical
- Core insight: replace raw frontier progression with a utility function that ranks repo analysis packs, browser sources, claim clusters, and unresolved questions by expected information gain per unit cost.
- External inspiration or analogy: multi-fidelity active learning and design optimization [R9][R10].
- Why it fits Saguaro and Anvil specifically: `CampaignControlPlane.run_research()` already computes `yield_history`, `impact_deltas`, stop conditions, and coverage details, which is most of an acquisition loop.
- Exact places in this codebase where it could wire in: `core/campaign/control_plane.py`, `core/research/crawler.py`, `core/research/store.py`, `core/campaign/coverage_engine.py`.
- Existing primitives it can reuse: `ResearchCrawler.stop_conditions()`, `CoverageEngine.stop_proof()`, `ResearchStore.record_claim()`, `CampaignStateStore.record_convergence_checkpoint()`.
- New primitive, data flow, or subsystem needed: `ResearchUtilityModel` that emits `expected_information_gain`, `expected_runtime_cost`, `expected_token_cost`, and `frontier_priority` for each source candidate.
- `repo_scope`: `["core/campaign", "core/research"]`
- `owning_specialist_type`: `ResearchLoopSubagent`
- `allowed_writes`: `["core/campaign/control_plane.py", "core/research/crawler.py", "core/research/store.py"]`
- `telemetry_contract`: `{"minimum": ["frontier_priority", "expected_information_gain", "expected_runtime_cost", "remaining_frontier", "stop_allowed"]}`
- `required_evidence`: `["research_claims", "topic_clusters", "analysis_packs", "coverage_stop_proof", "R9", "R10"]`
- `rollback_criteria`: `["claim yield drops below current heuristic baseline", "frontier starvation increases", "stop_allowed becomes less stable"]`
- `promotion_gate`: `{"median_claim_quality_delta": "> 0", "research_wall_time_regression": "<= 10%", "stop_proof_precision": "improves"}`
- `success_criteria`: `["higher useful-claim density", "fewer low-value browser fetches", "more stable stop proofs"]`
- Why this creates value: it turns research from FIFO accumulation into budgeted evidence search.
- Why this creates moat: it couples repo structure, external evidence, and local cost accounting into one acquisition policy.
- Main risk or failure mode: the utility model could overfit to cheap evidence and under-sample genuinely novel sources.
- Smallest credible first experiment: add a frontier scorer that reranks a synthetic mixed source list and compare `research_digest` quality against the current ordering.
- Confidence level: `0.86`

### 5.2 Evidence Exchange Hypothesis Scoring

- Suggested `phase_id`: `eid`
- Type: Practical
- Core insight: rank hypotheses by evidence coverage, counterexample pressure, runtime cost, and expected leverage, not by keyword bonuses alone.
- External inspiration or analogy: portfolio ranking in software agents plus graph retrieval [R4][R6][R15].
- Why it fits Saguaro and Anvil specifically: `HypothesisLab`, `HypothesisRanker`, and Memory Fabric already produce linkable evidence; they just do not behave like an exchange with explicit pricing.
- Exact places in this codebase where it could wire in: `core/research/hypothesis_lab.py`, `core/research/hypothesis_ranker.py`, `core/research/store.py`, `core/campaign/control_plane.py`.
- Existing primitives it can reuse: `supporting_claim_ids`, hypothesis memory aliases, topic clusters, repo dossiers, and `MemoryEdge` support links.
- New primitive, data flow, or subsystem needed: `HypothesisExchangeBook` containing long thesis score, short thesis score, counterexample debt, execution cost estimate, and portfolio weight.
- `repo_scope`: `["core/research", "core/campaign"]`
- `owning_specialist_type`: `EIDMasterSubagent`
- `allowed_writes`: `["core/research/hypothesis_lab.py", "core/research/hypothesis_ranker.py", "core/campaign/control_plane.py"]`
- `telemetry_contract`: `{"minimum": ["evidence_coverage_score", "counterexample_debt", "execution_cost_estimate", "innovation_score", "promotable"]}`
- `required_evidence`: `["research_claims", "repo_dossiers", "topic_clusters", "memory_edges", "R6", "R15"]`
- `rollback_criteria`: `["rankings become less predictive of successful lanes", "portfolio collapses to one subsystem", "counterexample debt is ignored"]`
- `promotion_gate`: `{"rank_correlation_with_lane_success": "positive", "portfolio_diversity_index": ">= current baseline"}`
- `success_criteria`: `["better hit rate on promotable hypotheses", "fewer false-positive moonshots", "clearer score breakdowns"]`
- Why this creates value: it upgrades EID from list sorting to allocation under uncertainty.
- Why this creates moat: evidence-priced invention is harder to copy than prompt tweaks.
- Main risk or failure mode: apparent rigor could still mask poor priors if counterexamples are weak.
- Smallest credible first experiment: score existing hypotheses against historical lane verdicts and compare against the current heuristic ranker.
- Confidence level: `0.84`

### 5.3 Counterexample Questionnaire Engine

- Suggested `phase_id`: `questionnaire`
- Type: Practical
- Core insight: architecture questions should be selected by expected branch-pruning power, not template order.
- External inspiration or analogy: counterexample-guided refinement and active query selection [R7][R9].
- Why it fits Saguaro and Anvil specifically: unresolved blockers already stop roadmap promotion; the missing step is generating the highest-value blocker first.
- Exact places in this codebase where it could wire in: `core/campaign/questionnaire.py`, `core/campaign/control_plane.py`, `core/campaign/transition_policy.py`.
- Existing primitives it can reuse: `blocking_level`, `linked_roadmap_items`, current question persistence, roadmap promotion gates.
- New primitive, data flow, or subsystem needed: `QuestionValueEstimator` that scores each candidate question by branch elimination, blast-radius reduction, and verification impact.
- `repo_scope`: `["core/campaign"]`
- `owning_specialist_type`: `ArchitectureAdjudicatorSubagent`
- `allowed_writes`: `["core/campaign/questionnaire.py", "core/campaign/control_plane.py"]`
- `telemetry_contract`: `{"minimum": ["question_value_score", "linked_item_count", "blocking_questions", "branch_pruning_estimate"]}`
- `required_evidence`: `["questions", "roadmap_items", "risk_summary", "R7", "R9"]`
- `rollback_criteria`: `["question churn increases", "blocking questions remain unresolved for longer", "operator satisfaction drops materially"]`
- `promotion_gate`: `{"mean_blocker_resolution_time": "decreases", "linked_item_coverage": "increases"}`
- `success_criteria`: `["fewer low-value questions", "earlier architecture certainty", "cleaner roadmap promotion"]`
- Why this creates value: it reduces operator fatigue and collapses ambiguity earlier.
- Why this creates moat: it turns questionnaire generation into a decision-theory primitive rather than a UX list.
- Main risk or failure mode: estimation could be wrong when the real blocker is social or organizational rather than technical.
- Smallest credible first experiment: re-score current questionnaire items and compare their value estimate against actual linked roadmap outcomes.
- Confidence level: `0.78`

### 5.4 Retrieval Budget Router

- Suggested `phase_id`: `research`
- Type: Practical
- Core insight: retrieval routing should optimize across Saguaro query, repo dossier summaries, Memory Fabric, prior artifacts, and fallback browsing under a single cost and confidence policy.
- External inspiration or analogy: GraphRAG and local model routers [R6][R11][R12].
- Why it fits Saguaro and Anvil specifically: `RetrievalPolicyEngine.decide()` already records route decisions, but only as `saguaro` versus `fallback`.
- Exact places in this codebase where it could wire in: `core/campaign/retrieval_policy.py`, `core/campaign/control_plane.py`, `core/memory/fabric/retrieval_planner.py`.
- Existing primitives it can reuse: repo dossier brief rendering, Memory Retrieval Planner, campaign telemetry, event store.
- New primitive, data flow, or subsystem needed: `RetrievalBudgetRouter` with route classes `saguaro_slice`, `artifact_cache`, `memory_fabric`, `repo_dossier`, `browser_fetch`, and `manual_escalation`.
- `repo_scope`: `["core/campaign", "core/memory/fabric"]`
- `owning_specialist_type`: `RepoAnalystSubagent`
- `allowed_writes`: `["core/campaign/retrieval_policy.py", "core/campaign/control_plane.py", "core/memory/fabric/retrieval_planner.py"]`
- `telemetry_contract`: `{"minimum": ["route", "reason", "evidence_quality", "latency_ms", "token_savings_estimate"]}`
- `required_evidence`: `["retrieval_policy telemetry", "repo_dossier_brief", "memory reads", "R6", "R11", "R12"]`
- `rollback_criteria`: `["fallback usage spikes without quality gain", "latency increases materially", "evidence quality drops"]`
- `promotion_gate`: `{"saguaro_first_compliance": "preserved", "median_retrieval_latency": "improves or holds"}`
- `success_criteria`: `["better route selection", "more reuse of internal evidence", "less redundant browsing"]`
- Why this creates value: it lowers local-model context pressure and accelerates campaign work.
- Why this creates moat: it unifies repo-native retrieval with external research policy.
- Main risk or failure mode: the router may become opaque if it optimizes too many variables without good explanations.
- Smallest credible first experiment: emit route-class telemetry for current research and development replay flows without changing behavior.
- Confidence level: `0.88`

### 5.5 Prompt Contract Fingerprinting

- Suggested `phase_id`: `analysis_upgrade`
- Type: Practical
- Core insight: prompt protocol identity should be persisted and diffed like any other execution-critical artifact.
- External inspiration or analogy: prompt-stealing threat models and translation validation [R7][R19].
- Why it fits Saguaro and Anvil specifically: Anvil already has AES prompt contracts, prompt hashes in replay state, and strong governance language.
- Exact places in this codebase where it could wire in: `core/prompts/prompt_manager.py`, `core/prompts/aes_prompt_builder.py`, `core/campaign/control_plane.py`, `shared_kernel/event_store.py`.
- Existing primitives it can reuse: prompt contract rendering, `prompt_protocol_hash`, replay event emission, campaign artifacts.
- New primitive, data flow, or subsystem needed: `PromptProtocolIdentity` capturing hash, salient fields, authorized deltas, and downstream experiment bindings.
- `repo_scope`: `["core/prompts", "core/campaign", "shared_kernel"]`
- `owning_specialist_type`: `DeterminismComplianceSubagent`
- `allowed_writes`: `["core/prompts/prompt_manager.py", "core/campaign/control_plane.py", "shared_kernel/event_store.py"]`
- `telemetry_contract`: `{"minimum": ["prompt_protocol_hash", "prompt_contract_delta_count", "replay_binding_id"]}`
- `required_evidence`: `["prompt contracts", "replay tapes", "R7", "R19"]`
- `rollback_criteria`: `["prompt identity creates excessive false positives", "operator iteration speed degrades materially"]`
- `promotion_gate`: `{"prompt_identity_present_in_all_replays": true, "unauthorized_prompt_drift": 0}`
- `success_criteria`: `["replay tapes preserve prompt identity", "experiment outputs are explainable against prompt protocol", "drift is auditable"]`
- Why this creates value: it makes local-model experimentation safer and more reproducible.
- Why this creates moat: most agent frameworks still treat prompts as ephemeral text.
- Main risk or failure mode: over-instrumentation could make prompt iteration awkward if there is no notion of authorized drift.
- Smallest credible first experiment: persist prompt protocol identity into development replay artifacts and compare hashes across repeated runs.
- Confidence level: `0.74`

### 5.6 Experiment Packet ABI v2

- Suggested `phase_id`: `development`
- Type: Practical
- Core insight: experiment lanes need a stronger ABI that binds objective, evidence basis, telemetry contract, rollback semantics, and replay identity into one typed packet.
- External inspiration or analogy: durable execution contracts and software agent task packets [R2][R4][R5].
- Why it fits Saguaro and Anvil specifically: `ExperimentDesignService` already emits structured lane payloads, and `ExperimentRunner` already consumes command specs and telemetry contracts.
- Exact places in this codebase where it could wire in: `core/research/experiment_design.py`, `core/research/experiment_runner.py`, `core/campaign/control_plane.py`, `core/campaign/lane_interfaces.py`.
- Existing primitives it can reuse: lane metadata, telemetry contracts, promotion policy, baseline manager, worktree manager.
- New primitive, data flow, or subsystem needed: `ExperimentPacketV2` with fields for `evidence_refs`, `counterfactual_refs`, `budget_caps`, `prompt_protocol_identity`, and `verification_tier`.
- `repo_scope`: `["core/research", "core/campaign"]`
- `owning_specialist_type`: `ExperimentLane`
- `allowed_writes`: `["core/research/experiment_design.py", "core/research/experiment_runner.py", "core/campaign/control_plane.py"]`
- `telemetry_contract`: `{"minimum": ["experiment_packet_version", "verification_tier", "budget_cap_seconds", "telemetry_contract_satisfied"]}`
- `required_evidence`: `["experimental_tracks", "experiment_results", "lane telemetry", "R2", "R4", "R5"]`
- `rollback_criteria`: `["packet version mismatch", "telemetry contract gaps", "rollback criteria omitted"]`
- `promotion_gate`: `{"all_lanes_emit_packet_v2": true, "contract_satisfaction_rate": ">= baseline"}`
- `success_criteria`: `["cleaner experiment semantics", "lower ambiguity in lane execution", "stronger replay and verification"]`
- Why this creates value: it hardens the handoff from ideation to execution.
- Why this creates moat: packet ABI strength becomes an execution discipline, not a markdown convention.
- Main risk or failure mode: changing the packet schema could create migration churn in speculative branches and tooling.
- Smallest credible first experiment: version the current lane packet and require `experiment_packet_version` in telemetry without changing commands yet.
- Confidence level: `0.9`

### 5.7 Native Telemetry Bridge

- Suggested `phase_id`: `development`
- Type: Practical
- Core insight: EID and campaign lanes should consume native runtime telemetry as first-class evidence instead of generic pass/fail metrics.
- External inspiration or analogy: PerfBench and local-serving incident analysis [R13][R14][R18].
- Why it fits Saguaro and Anvil specifically: `core/native/runtime_telemetry.py` already exposes TTFT, cache, backend-path, and QSG-specific metrics, but campaign telemetry contracts do not demand them.
- Exact places in this codebase where it could wire in: `core/campaign/telemetry_contracts.py`, `core/research/experiment_runner.py`, `core/native/runtime_telemetry.py`, `core/qsg/runtime_contracts.py`.
- Existing primitives it can reuse: `NativeGenerationTelemetry`, `TelemetryContractRegistry`, CPU scorecards, promotion policy.
- New primitive, data flow, or subsystem needed: `NativeTelemetryAdapter` that maps runtime telemetry into lane summaries and experiment scorecards.
- `repo_scope`: `["core/campaign", "core/research", "core/native", "core/qsg"]`
- `owning_specialist_type`: `ObservabilityContractEngineerSubagent`
- `allowed_writes`: `["core/campaign/telemetry_contracts.py", "core/research/experiment_runner.py", "core/native/runtime_telemetry.py", "core/qsg/runtime_contracts.py"]`
- `telemetry_contract`: `{"minimum": ["ttft_ms", "prefill_ms", "decode_ms", "prompt_cache_hit_ratio", "strict_path_stable", "measurement_valid"]}`
- `required_evidence`: `["runtime telemetry payloads", "experiment results", "R13", "R14", "R18"]`
- `rollback_criteria`: `["measurement_valid false positives spike", "runtime overhead > acceptable bound", "telemetry mapping loses fidelity"]`
- `promotion_gate`: `{"native_metrics_present_in_lane_results": true, "measurement_valid_rate": ">= baseline"}`
- `success_criteria`: `["hardware-fit recommendations become evidence-backed", "performance regressions are visible inside campaign flow"]`
- Why this creates value: it makes local-model and native-runtime behavior part of the invention loop.
- Why this creates moat: few agent systems can tie code-generation campaigns directly to low-level runtime evidence.
- Main risk or failure mode: metric overload could make EID overreact to noisy runtime variance.
- Smallest credible first experiment: inject `NativeGenerationTelemetry.as_dict()` payloads into one controlled experiment lane and verify contract evaluation.
- Confidence level: `0.91`

### 5.8 Repo Twin Risk Closure

- Suggested `phase_id`: `analysis_upgrade`
- Type: Practical
- Core insight: roadmap and EID promotion should depend on a predictive repo twin with graph-confidence, volatility, and blast-radius signals, not only artifact presence.
- External inspiration or analogy: NASA digital twins and risk-informed operations [R20].
- Why it fits Saguaro and Anvil specifically: `RepoTwinBuilder` and `RoadmapRiskRadar` already exist, but the twin is still mostly a snapshot and event excerpt.
- Exact places in this codebase where it could wire in: `core/campaign/risk_radar.py`, `core/campaign/control_plane.py`, `shared_kernel/event_store.py`, `saguaro/state/ledger.py`.
- Existing primitives it can reuse: repo twin capture, roadmap risk summary, event excerpts, state ledger snapshots.
- New primitive, data flow, or subsystem needed: `PredictiveRepoTwin` that stores graph confidence, volatility, impacted subsystems, missing edge classes, and recommended promotion restrictions.
- `repo_scope`: `["core/campaign", "shared_kernel", "saguaro/state"]`
- `owning_specialist_type`: `RepoAnalystSubagent`
- `allowed_writes`: `["core/campaign/risk_radar.py", "core/campaign/control_plane.py", "shared_kernel/event_store.py", "saguaro/state/ledger.py"]`
- `telemetry_contract`: `{"minimum": ["graph_confidence_score", "volatility_index", "predicted_blast_radius", "promotion_restriction_count"]}`
- `required_evidence`: `["repo_twin snapshots", "roadmap_risk", "state ledger", "R20"]`
- `rollback_criteria`: `["risk scores drift without explaining outcomes", "promotion latency grows too much", "false blocks increase"]`
- `promotion_gate`: `{"graph_confidence_score": "improves", "blocked_high_risk_promotions": "increase only where justified"}`
- `success_criteria`: `["risk radar becomes predictive", "roadmap adoption is safer", "EID reuse suggestions become more grounded"]`
- Why this creates value: it hardens promotion decisions and increases trust.
- Why this creates moat: campaign planning becomes state-aware rather than artifact-count aware.
- Main risk or failure mode: twin fidelity could remain too low if graph confidence is not upgraded.
- Smallest credible first experiment: persist graph-confidence and edge-class deficits into `repo_twin_roadmap_promoted` artifacts and surface them in the dashboard.
- Confidence level: `0.8`

### 5.9 Flight Recorder Unification

- Suggested `phase_id`: `convergence`
- Type: Practical
- Core insight: campaign events, experiment lanes, QSG replay, and verification results should collapse into one mission replay tape with deterministic identity.
- External inspiration or analogy: flight recorder plus durable execution [R2][R8].
- Why it fits Saguaro and Anvil specifically: `EventStore.export_run()` and `record_qsg_replay_event()` already expose deterministic replay metadata.
- Exact places in this codebase where it could wire in: `shared_kernel/event_store.py`, `core/campaign/control_plane.py`, `core/research/experiment_runner.py`, `domains/verification/verification_lane.py`.
- Existing primitives it can reuse: event links, checkpoints, replay hashes, verification-lane telemetry, experiment memories.
- New primitive, data flow, or subsystem needed: `MissionReplayTape` that references campaign state transitions, experiment results, prompt protocol identity, and verification witnesses in one bundle.
- `repo_scope`: `["shared_kernel", "core/campaign", "core/research", "domains/verification"]`
- `owning_specialist_type`: `DeterminismComplianceSubagent`
- `allowed_writes`: `["shared_kernel/event_store.py", "core/campaign/control_plane.py", "core/research/experiment_runner.py", "domains/verification/verification_lane.py"]`
- `telemetry_contract`: `{"minimum": ["replay_hash", "event_count", "checkpoint_count", "verification_lane_count"]}`
- `required_evidence`: `["event logs", "mission checkpoints", "experiment results", "verification records", "R2", "R8"]`
- `rollback_criteria`: `["replay export size becomes unmanageable", "hash instability appears", "critical links are missing"]`
- `promotion_gate`: `{"deterministic_hash_stable_on_replay": true, "all_promoted_lanes_have_replay_tape": true}`
- `success_criteria`: `["campaigns become inspectable without model rerun", "closure proofs have stronger evidence"]`
- Why this creates value: it makes failure analysis and handoff much cheaper.
- Why this creates moat: replay-quality evidence systems are rare in local software agents.
- Main risk or failure mode: event capture could become too verbose without compression tiers.
- Smallest credible first experiment: emit a replay tape for one full campaign path from `RESEARCH` through `EID_LAB`.
- Confidence level: `0.87`

### 5.10 Contract-First Roadmap Promotion

- Suggested `phase_id`: `roadmap_draft`
- Type: Practical
- Core insight: roadmap promotion should be driven by executable contract coverage and witness strength, not only phase-packet completeness.
- External inspiration or analogy: translation validation and benchmark anti-cheating [R7][R15][R16].
- Why it fits Saguaro and Anvil specifically: the repo already has two validators, one shallow in `core/campaign/roadmap_validator.py` and one stronger in `saguaro/roadmap/validator.py`.
- Exact places in this codebase where it could wire in: `core/campaign/roadmap_validator.py`, `core/campaign/roadmap_compiler.py`, `core/campaign/control_plane.py`, `saguaro/roadmap/*`.
- Existing primitives it can reuse: phase packets, `exit_gate`, `telemetry_contract`, `saguaro roadmap validate`.
- New primitive, data flow, or subsystem needed: `RoadmapWitnessPolicy` that requires a minimum set of code refs, test refs, graph refs, and verification refs before promotion.
- `repo_scope`: `["core/campaign", "saguaro/roadmap", "saguaro/requirements", "saguaro/validation"]`
- `owning_specialist_type`: `ArchitectureAdjudicatorSubagent`
- `allowed_writes`: `["core/campaign/roadmap_validator.py", "core/campaign/roadmap_compiler.py", "core/campaign/control_plane.py"]`
- `telemetry_contract`: `{"minimum": ["validation_error_count", "witness_count", "missing_test_refs", "missing_verification_refs"]}`
- `required_evidence`: `["roadmap draft", "phase packets", "requirement graph", "R7", "R15", "R16"]`
- `rollback_criteria`: `["validator noise overwhelms signal", "promotion stalls on irrelevant requirements"]`
- `promotion_gate`: `{"blocking_witness_gaps": 0, "phase_packet_schema_compliance": true}`
- `success_criteria`: `["fewer hollow promotions", "more traceable roadmap execution", "better alignment with doc validator"]`
- Why this creates value: it strengthens the roadmap as an execution contract.
- Why this creates moat: the roadmap becomes an active control artifact, not a static planning note.
- Main risk or failure mode: contract strictness may initially expose many latent gaps and feel punitive.
- Smallest credible first experiment: enrich `RoadmapValidator` output with witness classes while leaving current pass/fail behavior unchanged.
- Confidence level: `0.83`

### 5.11 EID-to-Legislation Autocodifier

- Suggested `phase_id`: `convergence`
- Type: Practical
- Core insight: repeated EID findings and lane failures should draft governance rules automatically so successful invention becomes institutional memory.
- External inspiration or analogy: post-incident CAPA loops and operating manuals.
- Why it fits Saguaro and Anvil specifically: the repo already exposes `RuleProposalEngine` and `saguaro legislation --draft`.
- Exact places in this codebase where it could wire in: `core/campaign/control_plane.py`, `core/governance/rule_proposal_engine.py`, `core/research/eid_master.py`.
- Existing primitives it can reuse: rule proposals, audit findings, experiment lane verdicts, whitepaper findings.
- New primitive, data flow, or subsystem needed: `EIDGovernanceEmitter` that converts high-confidence hypothesis outcomes and recurrent failure patterns into draft governance rules.
- `repo_scope`: `["core/campaign", "core/governance", "core/research"]`
- `owning_specialist_type`: `ADRStewardSubagent`
- `allowed_writes`: `["core/campaign/control_plane.py", "core/governance/rule_proposal_engine.py", "core/research/eid_master.py"]`
- `telemetry_contract`: `{"minimum": ["draft_rule_count", "rule_reuse_trigger_count", "lane_failure_pattern_count"]}`
- `required_evidence`: `["eid_summary", "audit_findings", "experiment_results", "rule proposals"]`
- `rollback_criteria`: `["rule spam increases", "draft quality is consistently poor", "operators ignore proposals"]`
- `promotion_gate`: `{"accepted_rule_rate": "positive", "repeated_failure_patterns": "decrease over time"}`
- `success_criteria`: `["successful invention becomes reusable governance", "fewer repeated avoidable failures"]`
- Why this creates value: it compounds learning across campaigns.
- Why this creates moat: campaign outcomes become governance assets.
- Main risk or failure mode: prematurely codified rules could freeze experimentation too early.
- Smallest credible first experiment: draft legislation candidates only for repeated telemetry-contract failures and repeated replay failures.
- Confidence level: `0.72`

### 5.12 Shadow Lap Simulator

- Suggested `phase_id`: `deep_test_audit`
- Type: Moonshot
- Core insight: run a cheap shadow model and tool-outcome predictor to simulate the first few steps of an experiment lane before real execution.
- External inspiration or analogy: driver-in-the-loop simulation and preflight laps [R2][R8].
- Why it fits Saguaro and Anvil specifically: `SimulatorPlanner` already exists, `ExperimentDesignService` already produces bounded tracks, and local-model constraints make preflight triage economically attractive.
- Exact places in this codebase where it could wire in: `core/research/simulator_planner.py`, `core/research/eid_master.py`, `core/research/experiment_runner.py`, `core/campaign/control_plane.py`.
- Existing primitives it can reuse: simulator plans, experiment packets, event store, telemetry contracts, rollout artifacts.
- New primitive, data flow, or subsystem needed: `ShadowLapRuntime` that predicts likely command success, missing artifacts, replayability, and telemetry sufficiency before real lane launch.
- `repo_scope`: `["core/research", "core/campaign"]`
- `owning_specialist_type`: `CounterfactualStrategistSubagent`
- `allowed_writes`: `["core/research/simulator_planner.py", "core/research/eid_master.py", "core/research/experiment_runner.py", "core/campaign/control_plane.py"]`
- `telemetry_contract`: `{"minimum": ["shadow_success_probability", "predicted_missing_artifacts", "predicted_contract_gaps", "shadow_runtime_ms"]}`
- `required_evidence`: `["lane packets", "historical experiment results", "event store traces", "R2", "R8"]`
- `rollback_criteria`: `["false negative rate on viable lanes is high", "shadow simulation cost approaches real execution"]`
- `promotion_gate`: `{"preflight_rejection_precision": "positive", "shadow_cost": "<< real lane cost"}`
- `success_criteria`: `["bad lanes are killed early", "good lanes proceed with higher confidence", "operator trust increases"]`
- Why this creates value: it saves local compute and shortens time-to-rejection.
- Why this creates moat: it operationalizes multi-fidelity invention rather than brute-forcing every idea.
- Main risk or failure mode: the shadow model could confidently reject unconventional but correct work.
- Smallest credible first experiment: predict telemetry-contract success on historical lane runs using only packet metadata and prior outcomes.
- Confidence level: `0.68`

### 5.13 Portfolio Auction Scheduler

- Suggested `phase_id`: `eid`
- Type: Moonshot
- Core insight: EID should allocate limited execution budget through an auction where hypotheses compete on expected leverage, diversity contribution, and cost.
- External inspiration or analogy: portfolio optimization, exchange matching, and race-strategy allocation.
- Why it fits Saguaro and Anvil specifically: the current EID loop already produces hypotheses, proposals, experiments, and limited `max_tracks`; it just hard-caps them instead of pricing them.
- Exact places in this codebase where it could wire in: `core/research/eid_master.py`, `core/research/eid_scheduler.py`, `core/research/hypothesis_ranker.py`, `core/campaign/control_plane.py`.
- Existing primitives it can reuse: innovation scores, required experiments, `max_tracks`, lane metadata.
- New primitive, data flow, or subsystem needed: `EIDBudgetAuction` with bid components for novelty, certainty, diversity, runtime budget, and native telemetry upside.
- `repo_scope`: `["core/research", "core/campaign"]`
- `owning_specialist_type`: `EIDMasterSubagent`
- `allowed_writes`: `["core/research/eid_master.py", "core/research/eid_scheduler.py", "core/research/hypothesis_ranker.py", "core/campaign/control_plane.py"]`
- `telemetry_contract`: `{"minimum": ["portfolio_budget", "accepted_bid_count", "diversity_score", "expected_value_density"]}`
- `required_evidence`: `["innovation_hypotheses", "experimental_tracks", "runtime budget", "R4", "R5", "R9"]`
- `rollback_criteria`: `["portfolio becomes unstable", "all budget flows to similar ideas", "operator comprehensibility collapses"]`
- `promotion_gate`: `{"accepted_bid_quality": "improves", "portfolio_diversity": "improves"}`
- `success_criteria`: `["better allocation of scarce local compute", "less arbitrary track selection"]`
- Why this creates value: it upgrades EID from ranking to resource allocation.
- Why this creates moat: budget-aware invention scheduling is a deeper primitive than idea listing.
- Main risk or failure mode: overformalized bidding could hide brittle assumptions behind fake precision.
- Smallest credible first experiment: simulate budget allocation offline on current hypothesis sets and compare with simple top-k.
- Confidence level: `0.67`

### 5.14 Hybrid MPC Transition Governor

- Suggested `phase_id`: `convergence`
- Type: Moonshot
- Core insight: transition policy should move from fixed state branching to a model-predictive governor that trades evidence sufficiency, cost, operator latency, and risk.
- External inspiration or analogy: model predictive control in aerospace and industrial control loops.
- Why it fits Saguaro and Anvil specifically: `CampaignTransitionPolicy` is already centralized and deterministic, which makes it a clean insertion point for a stronger decision policy.
- Exact places in this codebase where it could wire in: `core/campaign/transition_policy.py`, `core/campaign/control_plane.py`, `core/campaign/gate_engine.py`, `core/campaign/risk_radar.py`.
- Existing primitives it can reuse: current state, snapshot, gate rules, risk summary, artifact map.
- New primitive, data flow, or subsystem needed: `TransitionCostModel` plus an MPC-style solver that simulates a short horizon of phase transitions.
- `repo_scope`: `["core/campaign"]`
- `owning_specialist_type`: `SystemsArchitectSubagent`
- `allowed_writes`: `["core/campaign/transition_policy.py", "core/campaign/control_plane.py", "core/campaign/gate_engine.py"]`
- `telemetry_contract`: `{"minimum": ["transition_score", "predicted_operator_delay", "predicted_risk", "selected_action"]}`
- `required_evidence`: `["campaign snapshots", "artifact approvals", "risk radar", "gate decisions"]`
- `rollback_criteria`: `["transition decisions become less predictable", "operator surprises increase", "state deadlocks appear"]`
- `promotion_gate`: `{"decision_explainability": "preserved", "avoidable_transition_churn": "decreases"}`
- `success_criteria`: `["smarter phase progression", "less unnecessary looping", "better timing of development promotion"]`
- Why this creates value: it turns campaign flow into controlled optimization.
- Why this creates moat: phase progression becomes adaptively optimal rather than static.
- Main risk or failure mode: a poor cost model could make the control plane feel arbitrary.
- Smallest credible first experiment: compute MPC recommendations in shadow mode while keeping current transition decisions authoritative.
- Confidence level: `0.58`

### 5.15 Counterfactual Repo Twin Branching

- Suggested `phase_id`: `analysis_upgrade`
- Type: Moonshot
- Core insight: repo twins should support "what if we promote this lane?" branch scoring before code is touched.
- External inspiration or analogy: digital twins and counterfactual planning [R20].
- Why it fits Saguaro and Anvil specifically: `RepoTwinBuilder`, event excerpts, and roadmap risk already exist; branch speculation exists too, but without a predictive twin.
- Exact places in this codebase where it could wire in: `core/campaign/risk_radar.py`, `core/campaign/control_plane.py`, `core/campaign/continuous_lane.py`, `shared_kernel/event_store.py`.
- Existing primitives it can reuse: speculative roadmap items, repo twins, event links, roadmap risk summaries.
- New primitive, data flow, or subsystem needed: `CounterfactualTwinBranch` storing predicted changed surfaces, likely failing contracts, and delta authority conflicts.
- `repo_scope`: `["core/campaign", "shared_kernel"]`
- `owning_specialist_type`: `CounterfactualStrategistSubagent`
- `allowed_writes`: `["core/campaign/risk_radar.py", "core/campaign/control_plane.py", "shared_kernel/event_store.py"]`
- `telemetry_contract`: `{"minimum": ["predicted_changed_surface", "predicted_failing_contracts", "counterfactual_confidence"]}`
- `required_evidence`: `["repo twin", "speculative branch artifacts", "verification lane results", "R20"]`
- `rollback_criteria`: `["predictions are poorly calibrated", "branch creation overhead outweighs benefit"]`
- `promotion_gate`: `{"counterfactual_precision": "positive", "speculative_lane_failures": "decrease"}`
- `success_criteria`: `["better speculation lane triage", "less wasted branch work", "earlier hazard discovery"]`
- Why this creates value: it reduces costly speculative execution.
- Why this creates moat: it makes the roadmap behave like an executable trade study.
- Main risk or failure mode: the twin may stay too shallow if graph confidence is not improved first.
- Smallest credible first experiment: compare predicted failing contract families against actual verification-lane results on speculative branches.
- Confidence level: `0.61`

### 5.16 Bayesian Design-of-Experiments Lab

- Suggested `phase_id`: `eid`
- Type: Moonshot
- Core insight: EID should choose the next experiments using Bayesian or surrogate-guided design-of-experiments rather than static required-experiment lists.
- External inspiration or analogy: active learning and surrogate optimization [R9][R10].
- Why it fits Saguaro and Anvil specifically: `ExperimentDesignService` is currently a deterministic translator from proposal to lane; it could become a DOE engine.
- Exact places in this codebase where it could wire in: `core/research/experiment_design.py`, `core/research/eid_master.py`, `core/research/experiment_runner.py`.
- Existing primitives it can reuse: lane metadata, telemetry contracts, result summaries, promotion policy.
- New primitive, data flow, or subsystem needed: `ExperimentSurrogateModel` that estimates expected value, uncertainty, and interaction effects across lane types and hardware regimes.
- `repo_scope`: `["core/research"]`
- `owning_specialist_type`: `HypothesisLabSubagent`
- `allowed_writes`: `["core/research/experiment_design.py", "core/research/eid_master.py", "core/research/experiment_runner.py"]`
- `telemetry_contract`: `{"minimum": ["surrogate_mean", "surrogate_uncertainty", "expected_improvement", "actual_outcome"]}`
- `required_evidence`: `["historical experiment results", "native telemetry", "R9", "R10"]`
- `rollback_criteria`: `["DOE overhead exceeds benefit", "model calibration is poor", "operator trust falls"]`
- `promotion_gate`: `{"expected_improvement_calibration": "positive", "wasted_experiment_fraction": "decreases"}`
- `success_criteria`: `["fewer redundant lanes", "better sequencing of experiments", "clearer exploration versus exploitation"]`
- Why this creates value: it makes experimentation more sample-efficient under local compute constraints.
- Why this creates moat: invention becomes a learnable scientific process rather than a template engine.
- Main risk or failure mode: sparse historical data may not support useful surrogates yet.
- Smallest credible first experiment: fit a simple surrogate over historical experiment metadata and lane outcomes before using it for control.
- Confidence level: `0.63`

### 5.17 Campaign FDIR

- Suggested `phase_id`: `development`
- Type: Moonshot
- Core insight: campaigns need fault detection, isolation, and recovery logic for autonomy failures, not just exceptions and retries.
- External inspiration or analogy: spacecraft FDIR and safety engineering.
- Why it fits Saguaro and Anvil specifically: campaigns already have state transitions, event logs, verification lanes, and completion proofs, which is enough substrate for explicit fault classes.
- Exact places in this codebase where it could wire in: `core/campaign/control_plane.py`, `core/campaign/transition_policy.py`, `core/campaign/daemon.py`, `shared_kernel/event_store.py`.
- Existing primitives it can reuse: state machine, checkpoints, event store, replay export, audit findings.
- New primitive, data flow, or subsystem needed: `CampaignFaultManager` with fault classes like `evidence_starvation`, `verification_instability`, `prompt_protocol_drift`, and `native_runtime_invalid`.
- `repo_scope`: `["core/campaign", "shared_kernel"]`
- `owning_specialist_type`: `DeterminismComplianceSubagent`
- `allowed_writes`: `["core/campaign/control_plane.py", "core/campaign/transition_policy.py", "core/campaign/daemon.py", "shared_kernel/event_store.py"]`
- `telemetry_contract`: `{"minimum": ["fault_class", "fault_severity", "recovery_action", "recovery_success"]}`
- `required_evidence`: `["event timelines", "verification lane failures", "runtime telemetry", "audit findings"]`
- `rollback_criteria`: `["false alarm rate too high", "recovery actions mask real failures", "state loops emerge"]`
- `promotion_gate`: `{"recovery_success_rate": "positive", "silent failures": 0}`
- `success_criteria`: `["campaigns fail louder and recover cleaner", "operators get structured failure semantics"]`
- Why this creates value: it turns campaign instability into a controllable runtime phenomenon.
- Why this creates moat: explicit autonomy fault management is rare in coding agents.
- Main risk or failure mode: fault taxonomy could become too broad before enough data exists.
- Smallest credible first experiment: detect and isolate repeated telemetry-contract failures as one explicit fault class.
- Confidence level: `0.7`

### 5.18 Prompt-to-Task Translation Validation

- Suggested `phase_id`: `analysis_upgrade`
- Type: Moonshot
- Core insight: when prompts or EID proposals are lowered into task packets, the system should check whether the lowered packet preserves the intended constraints.
- External inspiration or analogy: Alive2-style translation validation [R7].
- Why it fits Saguaro and Anvil specifically: there is already a typed boundary between prompts, proposals, packets, and roadmap items.
- Exact places in this codebase where it could wire in: `core/prompts/prompt_manager.py`, `core/research/eid_scheduler.py`, `core/research/experiment_design.py`, `core/campaign/roadmap_compiler.py`.
- Existing primitives it can reuse: prompt contracts, proposal fields, packet metadata, phase packets.
- New primitive, data flow, or subsystem needed: `ConstraintPreservationChecker` that compares source intent fields against lowered packet fields and emits counterexamples.
- `repo_scope`: `["core/prompts", "core/research", "core/campaign"]`
- `owning_specialist_type`: `TranslationValidationProsecutorSubagent`
- `allowed_writes`: `["core/prompts/prompt_manager.py", "core/research/eid_scheduler.py", "core/research/experiment_design.py", "core/campaign/roadmap_compiler.py"]`
- `telemetry_contract`: `{"minimum": ["source_constraint_count", "preserved_constraint_count", "counterexample_count"]}`
- `required_evidence`: `["prompt contracts", "eid proposals", "experiment packets", "roadmap items", "R7"]`
- `rollback_criteria`: `["checker noise is high", "packet generation latency becomes unacceptable"]`
- `promotion_gate`: `{"preserved_constraint_ratio": "high", "counterexample_count": "falls over time"}`
- `success_criteria`: `["lowering errors become visible", "operator intent survives packet generation"]`
- Why this creates value: it attacks silent semantic drift.
- Why this creates moat: it applies a formal compiler pattern to agent execution lowering.
- Main risk or failure mode: intent comparison may be underspecified and devolve into string matching.
- Smallest credible first experiment: compare required metrics and allowed writes between EID proposals and generated packets, and fail on mismatch.
- Confidence level: `0.66`

### 5.19 Mission Capsule Distillation

- Suggested `phase_id`: `convergence`
- Type: Moonshot
- Core insight: every completed campaign should distill into a compact mission capsule that can seed future research, EID, roadmap, and development loops without replaying the entire artifact set.
- External inspiration or analogy: latent state compression and mission data products.
- Why it fits Saguaro and Anvil specifically: Memory Fabric and `QSGLatentBridge` already create latent packages; the missing step is campaign-level distillation and reuse policy.
- Exact places in this codebase where it could wire in: `core/qsg/latent_bridge.py`, `core/campaign/control_plane.py`, `core/memory/fabric/store.py`, `shared_kernel/event_store.py`.
- Existing primitives it can reuse: latent capture, memory projection, replay tapes, experiment results, repo dossiers.
- New primitive, data flow, or subsystem needed: `MissionCapsule` artifact with `invariants`, `failed_paths`, `winning_tracks`, `prompt_protocol_identity`, and `hardware_fit_digest`.
- `repo_scope`: `["core/qsg", "core/campaign", "core/memory/fabric", "shared_kernel"]`
- `owning_specialist_type`: `ImplementationEngineerSubagent`
- `allowed_writes`: `["core/qsg/latent_bridge.py", "core/campaign/control_plane.py", "core/memory/fabric/store.py", "shared_kernel/event_store.py"]`
- `telemetry_contract`: `{"minimum": ["capsule_id", "compression_ratio", "winning_track_count", "replay_hash_bound"]}`
- `required_evidence`: `["latent packages", "replay tapes", "eid summaries", "roadmap final"]`
- `rollback_criteria`: `["capsules become lossy to the point of misleading reuse", "compression ratio is poor"]`
- `promotion_gate`: `{"capsule_reuse_success_rate": "positive", "capsule_reconstruction_fidelity": "acceptable"}`
- `success_criteria`: `["future campaigns start from mission capsules", "context load drops", "successful invariants persist"]`
- Why this creates value: it reduces repeated rediscovery.
- Why this creates moat: accumulated campaign wisdom becomes a reusable substrate.
- Main risk or failure mode: compression could erase precisely the edge cases that matter later.
- Smallest credible first experiment: distill one completed campaign into a capsule and use it to seed a similar objective in shadow mode.
- Confidence level: `0.65`

### 5.20 Semantic Wind Tunnel

- Suggested `phase_id`: `deep_test_audit`
- Type: Moonshot
- Core insight: patch candidates should be stress-tested against semantic, graph, and telemetry perturbations before they are allowed into a promoted branch.
- External inspiration or analogy: wind-tunnel testing and software stress harnesses.
- Why it fits Saguaro and Anvil specifically: Anvil already has speculative lanes, verification lanes, and Saguaro semantic verification; it lacks an orchestrated pre-promotion stress harness.
- Exact places in this codebase where it could wire in: `core/campaign/control_plane.py`, `domains/verification/verification_lane.py`, `saguaro/sentinel/engines/semantic.py`, `core/research/experiment_runner.py`.
- Existing primitives it can reuse: changed-file verification, counterexamples, semantic verify, experiment lane packets.
- New primitive, data flow, or subsystem needed: `SemanticWindTunnel` that mutates retrieval context, prompt identity, graph confidence assumptions, and telemetry noise to test stability.
- `repo_scope`: `["core/campaign", "domains/verification", "saguaro/sentinel", "core/research"]`
- `owning_specialist_type`: `TestAuditSubagent`
- `allowed_writes`: `["core/campaign/control_plane.py", "domains/verification/verification_lane.py", "core/research/experiment_runner.py"]`
- `telemetry_contract`: `{"minimum": ["stress_case_count", "semantic_stability_score", "counterexample_count", "promotion_blocked"]}`
- `required_evidence`: `["verification results", "experiment packets", "semantic engine output"]`
- `rollback_criteria`: `["wind tunnel cost dominates development", "false failures are high"]`
- `promotion_gate`: `{"semantic_stability_score": "above threshold", "critical_counterexamples": 0}`
- `success_criteria`: `["more robust promoted changes", "fewer post-promotion surprises"]`
- Why this creates value: it catches brittle wins before they become roadmap reality.
- Why this creates moat: it treats semantic stability as an engineered property.
- Main risk or failure mode: stress design may lag behind real failure modes.
- Smallest credible first experiment: run semantic perturbation against one speculative roadmap item and compare stability with baseline verification.
- Confidence level: `0.62`

### 5.21 Multi-Fidelity Local-Model Stack

- Suggested `phase_id`: `intake`
- Type: Moonshot
- Core insight: the campaign stack should explicitly allocate objectives across tiny shadow models, medium analysis models, and heavy execution models, all within the local Ollama/QSG regime.
- External inspiration or analogy: Olla routing, local-serving policy, and speculative execution [R11][R12][R14].
- Why it fits Saguaro and Anvil specifically: the user constraint is local-only models, and EID is exactly where model stratification can save the most compute.
- Exact places in this codebase where it could wire in: `core/ollama_client.py`, `core/model_manager.py`, `core/research/eid_master.py`, `core/campaign/control_plane.py`, `core/qsg/ollama_adapter.py`.
- Existing primitives it can reuse: prompt profiles, model-family compression guidance, runtime telemetry, QSG adapter surfaces.
- New primitive, data flow, or subsystem needed: `ModelFidelityPolicy` with roles like `shadow_preflight`, `analysis`, `execution`, and `adjudication`.
- `repo_scope`: `["core/ollama_client.py", "core/model_manager.py", "core/research", "core/qsg"]`
- `owning_specialist_type`: `PlatformDXArchitectSubagent`
- `allowed_writes`: `["core/ollama_client.py", "core/model_manager.py", "core/research/eid_master.py", "core/qsg/ollama_adapter.py"]`
- `telemetry_contract`: `{"minimum": ["selected_model_role", "selected_model_name", "routing_reason", "latency_ms", "token_budget_estimate"]}`
- `required_evidence`: `["runtime telemetry", "Ollama API behavior", "R11", "R12", "R14"]`
- `rollback_criteria`: `["routing instability", "quality collapses on shadow paths", "operator debugging becomes harder"]`
- `promotion_gate`: `{"cost_per_successful_lane": "decreases", "quality_regression": "bounded"}`
- `success_criteria`: `["cheaper preflight", "faster EID iteration", "more deliberate model usage"]`
- Why this creates value: it makes the local-model constraint an architectural advantage instead of a handicap.
- Why this creates moat: model routing becomes coupled to campaign state and evidence, not generic latency rules.
- Main risk or failure mode: routing logic could mask weak models behind optimistic policy.
- Smallest credible first experiment: route `SimulatorPlanner` and shadow-lap work to a lighter model while keeping `ExperimentRunner` unchanged.
- Confidence level: `0.79`

### 5.22 Hardware-Fit Inverse Design

- Suggested `phase_id`: `eid`
- Type: Moonshot
- Core insight: EID should propose architecture changes not only from software semantics but from inverse reasoning over native telemetry and CPU/QSG bottlenecks.
- External inspiration or analogy: performance engineering and inverse design [R18].
- Why it fits Saguaro and Anvil specifically: the repo already includes CPU-first telemetry, QSG runtime contracts, and strong native surfaces.
- Exact places in this codebase where it could wire in: `core/research/eid_master.py`, `core/research/hypothesis_ranker.py`, `core/native/runtime_telemetry.py`, `core/campaign/cpu_scorecard.py`.
- Existing primitives it can reuse: hardware-fit recommendations, CPU scorecards, native telemetry payloads, runtime contract data.
- New primitive, data flow, or subsystem needed: `HardwareInverseDesignAdvisor` that reasons from telemetry bottlenecks toward candidate algorithmic or scheduling interventions.
- `repo_scope`: `["core/research", "core/native", "core/campaign"]`
- `owning_specialist_type`: `BenchmarkEngineerSubagent`
- `allowed_writes`: `["core/research/eid_master.py", "core/research/hypothesis_ranker.py", "core/native/runtime_telemetry.py", "core/campaign/cpu_scorecard.py"]`
- `telemetry_contract`: `{"minimum": ["bottleneck_class", "predicted_speedup", "measurement_valid", "hardware_fit_confidence"]}`
- `required_evidence`: `["runtime telemetry", "cpu scorecards", "PerfBench-style cases", "R18"]`
- `rollback_criteria`: `["predicted speedups are not realized", "telemetry noise dominates recommendations"]`
- `promotion_gate`: `{"telemetry_to_hypothesis_precision": "positive", "native-regression-rate": "decreases"}`
- `success_criteria`: `["more believable performance moonshots", "tighter link between runtime evidence and EID output"]`
- Why this creates value: it steers invention toward the parts of the stack where the repo is already differentiated.
- Why this creates moat: most idea engines are blind to the native runtime they are shaping.
- Main risk or failure mode: inverse reasoning from noisy telemetry can be misleading without tight measurement discipline.
- Smallest credible first experiment: generate EID recommendations from known native telemetry fixtures and score whether they target the right bottleneck class.
- Confidence level: `0.71`

### 5.23 Evidence Shock Absorber

- Suggested `phase_id`: `research`
- Type: Moonshot
- Core insight: large external evidence bursts should be buffered and normalized into a bounded "shock absorber" before they are allowed to perturb the EID portfolio.
- External inspiration or analogy: suspension systems and rate limiters in control loops.
- Why it fits Saguaro and Anvil specifically: `run_research()` can ingest external sources quickly, but current downstream hypothesis generation has little damping.
- Exact places in this codebase where it could wire in: `core/campaign/control_plane.py`, `core/research/store.py`, `core/research/clusterer.py`, `core/research/hypothesis_lab.py`.
- Existing primitives it can reuse: research chunks, topic clusters, stop proof, claim confidence.
- New primitive, data flow, or subsystem needed: `EvidenceShockAbsorber` that controls how rapidly new evidence can reweight hypotheses, features, or roadmap items.
- `repo_scope`: `["core/campaign", "core/research"]`
- `owning_specialist_type`: `FailureEconomistSubagent`
- `allowed_writes`: `["core/campaign/control_plane.py", "core/research/store.py", "core/research/clusterer.py", "core/research/hypothesis_lab.py"]`
- `telemetry_contract`: `{"minimum": ["evidence_influx_rate", "portfolio_reweight_delta", "shock_absorber_state"]}`
- `required_evidence`: `["research claims", "clusters", "hypothesis sets", "frontier telemetry"]`
- `rollback_criteria`: `["evidence latency becomes too high", "novel signals are excessively damped"]`
- `promotion_gate`: `{"portfolio_instability": "decreases", "novelty_capture": "preserved"}`
- `success_criteria`: `["less thrashing after large research batches", "more stable EID prioritization"]`
- Why this creates value: it makes the loop less twitchy and more governable.
- Why this creates moat: it imports real control-theory damping into agent research flow.
- Main risk or failure mode: damping may also hide fast-breaking important evidence.
- Smallest credible first experiment: simulate large mixed evidence bursts and compare hypothesis rank volatility with and without damping.
- Confidence level: `0.6`

### 5.24 Safety Case Compiler

- Suggested `phase_id`: `convergence`
- Type: Moonshot
- Core insight: convergence should compile a structured safety case that links claims, evidence, tests, telemetry, replay tapes, and risk acceptance into one machine-inspectable closure artifact.
- External inspiration or analogy: aerospace safety cases and assurance cases.
- Why it fits Saguaro and Anvil specifically: the repo already has the pieces of an assurance graph but not the compiler that assembles them.
- Exact places in this codebase where it could wire in: `core/campaign/completion_engine.py`, `core/campaign/control_plane.py`, `core/campaign/audit_engine.py`, `shared_kernel/event_store.py`, `saguaro/traceability/*`.
- Existing primitives it can reuse: completion checks, audit findings, roadmap final, event logs, verification-lane results, traceability artifacts.
- New primitive, data flow, or subsystem needed: `SafetyCaseCompiler` that emits claims, strategies, evidence nodes, unresolved risks, and operator signoffs in one graph artifact.
- `repo_scope`: `["core/campaign", "shared_kernel", "saguaro/traceability", "saguaro/validation"]`
- `owning_specialist_type`: `ArchitectureAdjudicatorSubagent`
- `allowed_writes`: `["core/campaign/completion_engine.py", "core/campaign/control_plane.py", "core/campaign/audit_engine.py", "shared_kernel/event_store.py"]`
- `telemetry_contract`: `{"minimum": ["safety_case_node_count", "unresolved_risk_count", "evidence_link_density", "closure_status"]}`
- `required_evidence`: `["roadmap_final", "audit findings", "verification lane records", "replay tapes", "repo twin"]`
- `rollback_criteria`: `["safety case becomes pure paperwork", "evidence graphs are too sparse to justify the artifact"]`
- `promotion_gate`: `{"unresolved_risk_count": "explicit and bounded", "closure_status": "provable"}`
- `success_criteria`: `["convergence emits a real assurance object", "operators can inspect closure rationale without rerunning the model"]`
- Why this creates value: it converts closure from narrative to evidence structure.
- Why this creates moat: it gives enterprise users something stronger than a finished diff and green tests.
- Main risk or failure mode: if evidence density is weak, the compiler may only expose the repo's current lack of closure discipline.
- Smallest credible first experiment: build a skeletal safety case graph from one campaign's final roadmap, audit, and verification artifacts.
- Confidence level: `0.69`

## 6. Critical Pressure Test

### 6.1 Ideas that are elegant but likely wrong

- `Hybrid MPC Transition Governor` is elegant, but it risks pretending we have a calibrated cost model before we do.
- `Counterfactual Repo Twin Branching` is compelling, but if graph confidence remains near the current 8.2 percent edge coverage, the twin will hallucinate authority.
- `Bayesian Design-of-Experiments Lab` is scientifically attractive, but sparse experimental history may make the surrogate too weak for control.

### 6.2 Ideas that are ugly but strategically powerful

- `Native Telemetry Bridge` is not glamorous, but it is one of the cleanest ways to make EID respect the repo's CPU/QSG reality.
- `Contract-First Roadmap Promotion` is bureaucratic on the surface, but it directly counters hollow artifact promotion and benchmark gaming.
- `Flight Recorder Unification` sounds operational rather than inventive, but it is one of the best enablers of trust, replay, and later automation.

### 6.3 Ideas likely to fail because the repo is missing a key primitive

- Any serious digital-twin or counterfactual-branch program will struggle until Saguaro graph confidence materially improves.
- Any portfolio auction or DOE engine will be weak until historical lane outcome data is richer and more normalized.
- Any prompt-to-task translation validator will be shallow until prompt constraints are expressed in a more typed way than current free-form contract text.

### 6.4 Non-obvious failure modes across the whole program

- Local shadow models may reject exactly the unconventional tracks that create moat.
- Stronger telemetry contracts may improve rigor but accidentally optimize EID toward measurable, boring work.
- Formalization can drift into theater if the validators do not check the right witnesses.
- Richer replay and safety-case artifacts can become too heavy if mission capsules and compression tiers are not built in parallel.

### 6.5 The hard architectural truth

[Inference] The biggest current weakness is not insufficient creativity.

It is insufficient mechanism for:

- evidence pricing,
- counterexample handling,
- graph-truth,
- and replay-grade closure.

Until those improve, the repo's ambition will outrun its control authority.

## 7. Synthesis

### 7.1 Strongest ideas overall

1. `Experiment Packet ABI v2`
2. `Native Telemetry Bridge`
3. `Research Frontier Economics Engine`
4. `Evidence Exchange Hypothesis Scoring`
5. `Flight Recorder Unification`

### 7.2 Best novelty-plausibility balance

- `Research Frontier Economics Engine` has the best balance.
  It is new enough to matter and already has visible insertion points in `run_research()`.

- `Evidence Exchange Hypothesis Scoring` is second.
  It changes EID's character without requiring a brand-new runtime.

- `Native Telemetry Bridge` is third.
  It is highly plausible and makes the loop meaningfully more differentiated.

### 7.3 Most feasible now

`Experiment Packet ABI v2` is the most feasible now because:

- the packet boundary already exists,
- the experiment runner already expects structured command specs,
- telemetry contracts already exist,
- and the roadmap compiler already consumes lane metadata.

### 7.4 Biggest long-term moat bet

`Counterfactual Repo Twin Branching` is the biggest long-term moat bet.

If Anvil can make speculative campaign planning depend on a predictive repo twin with meaningful confidence estimates, it becomes much harder to imitate with prompt wrappers.

### 7.5 Cleanest unifier with the current codebase

`Flight Recorder Unification` most cleanly unifies the current codebase because it directly ties together:

- `shared_kernel/event_store.py`,
- `core/campaign/control_plane.py`,
- `core/research/experiment_runner.py`,
- `domains/verification/verification_lane.py`,
- and the roadmap / completion path.

### 7.6 Which idea should be prototyped first

Prototype `Research Frontier Economics Engine` first.

Reason:

- it upgrades the earliest evidence-producing phase,
- it feeds better inputs to EID,
- it is measurable,
- it is reversible,
- and it does not depend on solving graph-confidence or prompt-identity first.

## 8. Implementation Program

### 8.1 Phase `research`: Frontier Economics and Retrieval Budgeting

- Phase title: Research Frontier Economics
- Objective: Convert `run_research()` from heuristic accumulation into a cost-aware evidence acquisition loop that explicitly chooses what to ingest next and why.
- Dependencies: `intake`
- Repo scope: `["core/campaign/control_plane.py", "core/research/crawler.py", "core/research/store.py", "core/campaign/retrieval_policy.py", "core/campaign/coverage_engine.py"]`
- Owning specialist type: `ResearchLoopSubagent`
- Allowed writes: `["core/campaign/control_plane.py", "core/research/crawler.py", "core/research/store.py", "core/campaign/retrieval_policy.py"]`
- Telemetry contract: `{"minimum": ["frontier_priority", "expected_information_gain", "expected_runtime_cost", "route", "remaining_frontier", "stop_allowed"]}`
- Required evidence: `["research_claims", "topic_clusters", "analysis_packs", "repo_dossier_brief", "stop_proof"]`
- Rollback criteria: `["claim usefulness drops", "frontier latency regresses materially", "fallback browsing rises without justification"]`
- Promotion gate: `{"claim_quality_improves": true, "redundant_browser_fetches_decrease": true, "stop_proof_stability_improves": true}`
- Success criteria: `["research sources are explicitly prioritized", "internal evidence is reused before external fetch", "stop conditions become better calibrated"]`
- Exact wiring points: `CampaignControlPlane.run_research`, `ResearchCrawler.enqueue`, `ResearchCrawler.stop_conditions`, `ResearchStore.record_claim`, `RetrievalPolicyEngine.decide`.
- Deliverables: `ResearchUtilityModel`, route-class telemetry, frontier priority persistence, dossier-aware retrieval routing, research digest additions.
- Tests: `tests/test_campaign_control_kernel.py`, `tests/test_campaign_research_frontier.py`, `tests/test_campaign_retrieval_budget_router.py`
- Verification commands: `pytest tests/test_campaign_control_kernel.py tests/test_campaign_research_frontier.py tests/test_campaign_retrieval_budget_router.py`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: research digests include utility scores and route explanations, and the new path outperforms the current heuristic on at least one replayable fixture.

### 8.2 Phase `eid`: Evidence Exchange and Portfolio EID

- Phase title: Evidence-Priced EID
- Objective: Upgrade EID from canned hypotheses and keyword ranking into a portfolio system that prices evidence coverage, counterexample debt, diversity, and expected execution cost.
- Dependencies: `research`
- Repo scope: `["core/research/hypothesis_lab.py", "core/research/hypothesis_ranker.py", "core/research/eid_scheduler.py", "core/research/eid_master.py", "core/campaign/control_plane.py"]`
- Owning specialist type: `EIDMasterSubagent`
- Allowed writes: `["core/research/hypothesis_lab.py", "core/research/hypothesis_ranker.py", "core/research/eid_scheduler.py", "core/research/eid_master.py", "core/campaign/control_plane.py"]`
- Telemetry contract: `{"minimum": ["evidence_coverage_score", "counterexample_debt", "portfolio_weight", "execution_cost_estimate", "innovation_score", "accepted_bid_count"]}`
- Required evidence: `["research claims", "topic clusters", "repo dossiers", "hypothesis memory edges", "lane outcome history"]`
- Rollback criteria: `["portfolio collapses to narrow ideas", "ranking no longer predicts lane quality", "counterexample debt is ignored"]`
- Promotion gate: `{"rank_to_outcome_correlation_improves": true, "portfolio_diversity_index_improves": true}`
- Success criteria: `["promotable hypotheses are better calibrated", "EID output explains why tracks are funded", "moonshots and practical bets are intentionally mixed"]`
- Exact wiring points: `HypothesisLab.generate`, `HypothesisRanker.rank`, `EIDScheduler.build_proposals`, `EIDMasterLoop.run`, `CampaignControlPlane.run_eid`.
- Deliverables: `HypothesisExchangeBook`, counterexample-debt scoring, budget-aware proposal selection, evidence-priced whitepaper outline.
- Tests: `tests/test_campaign_eid_exchange.py`, `tests/test_campaign_eid_portfolio_scheduler.py`, `tests/test_latent_package_capture.py`
- Verification commands: `pytest tests/test_campaign_eid_exchange.py tests/test_campaign_eid_portfolio_scheduler.py tests/test_latent_package_capture.py`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: EID summaries contain priced hypotheses and portfolio rationale, and historical replays show improved hit rate on successful experimental tracks.

### 8.3 Phase `questionnaire`: Counterexample-Guided Architecture Questions

- Phase title: Branch-Pruning Questionnaire
- Objective: Make the questionnaire phase ask the smallest number of questions that most reduce roadmap branching uncertainty and risk.
- Dependencies: `research`
- Repo scope: `["core/campaign/questionnaire.py", "core/campaign/control_plane.py", "core/campaign/transition_policy.py", "core/campaign/risk_radar.py"]`
- Owning specialist type: `ArchitectureAdjudicatorSubagent`
- Allowed writes: `["core/campaign/questionnaire.py", "core/campaign/control_plane.py", "core/campaign/transition_policy.py"]`
- Telemetry contract: `{"minimum": ["question_value_score", "branch_pruning_estimate", "blocking_questions", "linked_item_count"]}`
- Required evidence: `["questions", "risk summary", "linked roadmap items", "feature dependencies"]`
- Rollback criteria: `["question count increases without resolution gain", "blocking questions linger longer", "operators cannot interpret scores"]`
- Promotion gate: `{"mean_blocker_resolution_time_decreases": true, "linked_item_coverage_improves": true}`
- Success criteria: `["blocking questions become more targeted", "feature-map promotion encounters fewer surprises", "operator review load decreases"]`
- Exact wiring points: `QuestionnaireBuilder.build`, `QuestionnaireBuilder.pending_blockers`, `CampaignTransitionPolicy.decide`, `CampaignControlPlane.build_questionnaire`.
- Deliverables: `QuestionValueEstimator`, question ranking metadata, blocker-to-risk linkage.
- Tests: `tests/test_campaign_questionnaire_counterexamples.py`, `tests/test_campaign_control_kernel.py`, `tests/test_campaign_roadmap_phase_pack.py`
- Verification commands: `pytest tests/test_campaign_questionnaire_counterexamples.py tests/test_campaign_control_kernel.py tests/test_campaign_roadmap_phase_pack.py`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: questionnaire artifacts include branch-pruning rationale and promotion blockers are measurably better resolved.

### 8.4 Phase `analysis_upgrade`: Graph Confidence, Prompt Identity, and Predictive Repo Twins

- Phase title: Control-Plane Truth Upgrade
- Objective: Raise the authority of promotion decisions by attaching graph confidence, prompt protocol identity, and predictive repo-twin signals to roadmap and replay artifacts.
- Dependencies: `eid`, `questionnaire`
- Repo scope: `["core/campaign/risk_radar.py", "core/campaign/control_plane.py", "core/prompts/prompt_manager.py", "shared_kernel/event_store.py", "saguaro/state/ledger.py", "core/campaign/roadmap_validator.py"]`
- Owning specialist type: `DeterminismComplianceSubagent`
- Allowed writes: `["core/campaign/risk_radar.py", "core/campaign/control_plane.py", "core/prompts/prompt_manager.py", "shared_kernel/event_store.py", "core/campaign/roadmap_validator.py"]`
- Telemetry contract: `{"minimum": ["graph_confidence_score", "prompt_protocol_hash", "predicted_blast_radius", "witness_count", "missing_verification_refs"]}`
- Required evidence: `["repo_twin artifacts", "roadmap draft", "replay tapes", "graph confidence report", "prompt contracts"]`
- Rollback criteria: `["signal noise blocks normal promotion", "graph metrics remain useless", "prompt identity causes workflow friction without value"]`
- Promotion gate: `{"graph_confidence_visible_in_roadmap": true, "prompt_identity_bound_to_replay": true, "roadmap_witness_gaps_reduced": true}`
- Success criteria: `["promotion decisions are better explained", "semantic drift is easier to locate", "roadmap validation gains stronger evidence classes"]`
- Exact wiring points: `RoadmapRiskRadar.analyze`, `RepoTwinBuilder.capture`, `PromptManager.build_prompt_contract`, `EventStore.emit`, `RoadmapCompiler.validate`.
- Deliverables: predictive repo-twin payload, prompt protocol identity artifact, witness-aware roadmap validation summary.
- Tests: `tests/test_saguaro_roadmap_validator.py`, `tests/test_event_store.py`, `tests/test_campaign_speculation_risk_and_governance.py`, `tests/test_phase2_prompt_governance.py`
- Verification commands: `pytest tests/test_saguaro_roadmap_validator.py tests/test_event_store.py tests/test_campaign_speculation_risk_and_governance.py tests/test_phase2_prompt_governance.py`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: roadmap drafts include graph-confidence and witness summaries, and replay artifacts bind prompt identity and repo-twin context.

### 8.5 Phase `roadmap_draft`: Contract-First Roadmap Compiler

- Phase title: Roadmap as Executable Contract
- Objective: Strengthen roadmap drafting so promoted items and phase packets carry explicit witnesses, stronger validation, and execution-ready contract semantics.
- Dependencies: `analysis_upgrade`
- Repo scope: `["core/campaign/roadmap_compiler.py", "core/campaign/phase_packet.py", "core/campaign/roadmap_validator.py", "core/campaign/control_plane.py", "saguaro/roadmap", "saguaro/requirements", "saguaro/validation"]`
- Owning specialist type: `ArchitectureAdjudicatorSubagent`
- Allowed writes: `["core/campaign/roadmap_compiler.py", "core/campaign/phase_packet.py", "core/campaign/roadmap_validator.py", "core/campaign/control_plane.py"]`
- Telemetry contract: `{"minimum": ["validation_error_count", "phase_artifact_count", "witness_count", "blocking_witness_gaps"]}`
- Required evidence: `["feature map", "questionnaire approval", "eid_summary", "research_digest", "repo_twin", "verification lane outputs"]`
- Rollback criteria: `["roadmap compiler becomes too strict to be usable", "phase-pack generation breaks downstream consumers"]`
- Promotion gate: `{"blocking_witness_gaps": 0, "phase_packet_schema_compliance": true, "roadmap_risk_summary_present": true}`
- Success criteria: `["roadmap draft is more machine-actionable", "phase packets are richer without losing determinism", "final promotion is less hollow"]`
- Exact wiring points: `RoadmapCompiler.compile`, `RoadmapCompiler.render_phase_pack`, `PhasePacketBuilder.build`, `RoadmapValidator.validate`, `CampaignControlPlane.build_roadmap`.
- Deliverables: witness-aware roadmap items, upgraded phase packets, contract-strengthened draft artifact.
- Tests: `tests/test_campaign_roadmap_phase_pack.py`, `tests/test_saguaro_roadmap_validator.py`, `tests/test_campaign_control_kernel.py`
- Verification commands: `pytest tests/test_campaign_roadmap_phase_pack.py tests/test_saguaro_roadmap_validator.py tests/test_campaign_control_kernel.py`, `./venv/bin/saguaro roadmap validate --path inventive_research_roadmap.md --format json`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: `roadmap_draft` artifacts preserve required witnesses, phase packets still compile, and promotion blockers are explicit.

### 8.6 Phase `development`: Packet ABI v2 and Native-Telemetry Experiment Lanes

- Phase title: Measured Experiment Lanes
- Objective: Make all EID-driven development lanes execute through a richer packet ABI and report native/QSG telemetry as part of their contractual outcome.
- Dependencies: `roadmap_draft`
- Repo scope: `["core/research/experiment_design.py", "core/research/experiment_runner.py", "core/campaign/telemetry_contracts.py", "core/native/runtime_telemetry.py", "core/qsg/runtime_contracts.py", "domains/verification/verification_lane.py"]`
- Owning specialist type: `ExperimentLane`
- Allowed writes: `["core/research/experiment_design.py", "core/research/experiment_runner.py", "core/campaign/telemetry_contracts.py", "core/native/runtime_telemetry.py", "core/qsg/runtime_contracts.py", "domains/verification/verification_lane.py"]`
- Telemetry contract: `{"minimum": ["experiment_packet_version", "ttft_ms", "prefill_ms", "decode_ms", "prompt_cache_hit_ratio", "measurement_valid", "telemetry_contract_satisfied"]}`
- Required evidence: `["lane packets", "experiment results", "runtime telemetry", "verification lane records"]`
- Rollback criteria: `["packet migrations break lanes", "runtime overhead is too high", "measurement_valid becomes noisy"]`
- Promotion gate: `{"all_lanes_emit_packet_v2": true, "native_metrics_visible_in_verdicts": true, "verification_feedback_bound_to_read_id": true}`
- Success criteria: `["lanes are more replayable", "hardware-fit claims are measurable", "verification outcomes feed memory feedback"]`
- Exact wiring points: `ExperimentDesignService.design`, `ExperimentRunner.run`, `TelemetryContractRegistry.build`, `TelemetryContractRegistry.evaluate`, `VerificationLane.run`.
- Deliverables: packet ABI v2, native telemetry adapter, verification-bound lane verdicts.
- Tests: `tests/test_runtime_telemetry.py`, `tests/test_campaign_timeline_and_verification_lane.py`, `tests/test_development_replay.py`, `tests/test_qsg_runtime_contracts.py`
- Verification commands: `pytest tests/test_runtime_telemetry.py tests/test_campaign_timeline_and_verification_lane.py tests/test_development_replay.py tests/test_qsg_runtime_contracts.py`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: experiment lanes carry packet version and native metrics, and promotion decisions can depend on them.

### 8.7 Phase `deep_test_audit`: Shadow Laps and Semantic Wind Tunnel

- Phase title: Multi-Fidelity Preflight
- Objective: Add two pre-promotion guards: a cheap shadow lap that predicts likely lane viability and a semantic wind tunnel that stress-tests candidate changes before adoption.
- Dependencies: `development`
- Repo scope: `["core/research/simulator_planner.py", "core/research/eid_master.py", "core/research/experiment_runner.py", "core/campaign/control_plane.py", "domains/verification/verification_lane.py", "saguaro/sentinel/engines/semantic.py"]`
- Owning specialist type: `TestAuditSubagent`
- Allowed writes: `["core/research/simulator_planner.py", "core/research/eid_master.py", "core/research/experiment_runner.py", "core/campaign/control_plane.py", "domains/verification/verification_lane.py"]`
- Telemetry contract: `{"minimum": ["shadow_success_probability", "predicted_contract_gaps", "stress_case_count", "semantic_stability_score"]}`
- Required evidence: `["historical lane outcomes", "verification counterexamples", "semantic verify outputs", "runtime telemetry"]`
- Rollback criteria: `["false rejection rate is too high", "preflight cost is too close to real execution", "semantic stress cases are poorly calibrated"]`
- Promotion gate: `{"shadow_rejection_precision": "positive", "critical_counterexamples_detected_pre_adoption": true}`
- Success criteria: `["expensive bad lanes are killed sooner", "promoted changes are less brittle", "speculative work is better triaged"]`
- Exact wiring points: `SimulatorPlanner.plan`, `EIDMasterLoop.run`, `ExperimentRunner.run_lane`, `CampaignControlPlane.run_speculative_roadmap_item`, `VerificationLane.run`.
- Deliverables: shadow-lap runtime, semantic wind tunnel harness, pre-adoption risk score.
- Tests: `tests/test_campaign_shadow_laps.py`, `tests/test_campaign_speculation_risk_and_governance.py`, `tests/test_campaign_timeline_and_verification_lane.py`
- Verification commands: `pytest tests/test_campaign_shadow_laps.py tests/test_campaign_speculation_risk_and_governance.py tests/test_campaign_timeline_and_verification_lane.py`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: at least one historical failure mode is caught by preflight without running a full lane.

### 8.8 Phase `convergence`: Replay Tapes, Mission Capsules, and Safety Case

- Phase title: Closure as Evidence
- Objective: End campaigns with a replay tape, mission capsule, and machine-inspectable safety case rather than only a completion narrative.
- Dependencies: `deep_test_audit`
- Repo scope: `["shared_kernel/event_store.py", "core/campaign/completion_engine.py", "core/campaign/control_plane.py", "core/campaign/audit_engine.py", "core/qsg/latent_bridge.py", "core/memory/fabric/store.py"]`
- Owning specialist type: `DeterminismComplianceSubagent`
- Allowed writes: `["shared_kernel/event_store.py", "core/campaign/completion_engine.py", "core/campaign/control_plane.py", "core/campaign/audit_engine.py", "core/qsg/latent_bridge.py", "core/memory/fabric/store.py"]`
- Telemetry contract: `{"minimum": ["replay_hash", "capsule_id", "safety_case_node_count", "unresolved_risk_count", "closure_status"]}`
- Required evidence: `["event logs", "mission checkpoints", "roadmap_final", "verification lane outputs", "audit findings", "latent packages"]`
- Rollback criteria: `["closure artifacts are too weak to justify their complexity", "replay hash instability appears", "capsules lose critical edge cases"]`
- Promotion gate: `{"closure_status": "provable", "all_promoted_lanes_have_replay_tape": true, "unresolved_risk_count": "explicitly bounded"}`
- Success criteria: `["completed campaigns are replayable", "future campaigns can ingest mission capsules", "closure arguments are inspectable"]`
- Exact wiring points: `EventStore.export_run`, `EventStore.record_qsg_replay_event`, `CampaignControlPlane.build_completion_proof`, `CompletionEngine`, `QSGLatentBridge.capture_summary_package`.
- Deliverables: unified replay tape, mission capsule artifact, safety-case graph.
- Tests: `tests/test_event_store.py`, `tests/test_latent_package_capture.py`, `tests/test_campaign_closure_safety_case.py`, `tests/test_saguaro_roadmap_validator.py`
- Verification commands: `pytest tests/test_event_store.py tests/test_latent_package_capture.py tests/test_campaign_closure_safety_case.py tests/test_saguaro_roadmap_validator.py`, `./venv/bin/saguaro roadmap validate --path inventive_research_roadmap.md --format json`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: one campaign can be audited from intake to closure through replay tape and safety-case graph without rerunning the model.

## 9. Implementation Contract

- The system shall implement research frontier utility scoring through `core/campaign/control_plane.py`, `core/research/crawler.py`, `core/research/store.py`, and `core/campaign/retrieval_policy.py`, tested by `tests/test_campaign_control_kernel.py`, `tests/test_campaign_research_frontier.py`, and `tests/test_campaign_retrieval_budget_router.py`, and verified with `pytest tests/test_campaign_control_kernel.py tests/test_campaign_research_frontier.py tests/test_campaign_retrieval_budget_router.py`, `./venv/bin/saguaro roadmap validate --path inventive_research_roadmap.md --format json`, and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall implement evidence-priced hypothesis ranking and portfolio allocation through `core/research/hypothesis_lab.py`, `core/research/hypothesis_ranker.py`, `core/research/eid_scheduler.py`, `core/research/eid_master.py`, and `core/campaign/control_plane.py`, tested by `tests/test_campaign_eid_exchange.py`, `tests/test_campaign_eid_portfolio_scheduler.py`, and `tests/test_latent_package_capture.py`, and verified with `pytest tests/test_campaign_eid_exchange.py tests/test_campaign_eid_portfolio_scheduler.py tests/test_latent_package_capture.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall implement branch-pruning questionnaire scoring through `core/campaign/questionnaire.py`, `core/campaign/control_plane.py`, and `core/campaign/transition_policy.py`, tested by `tests/test_campaign_questionnaire_counterexamples.py`, `tests/test_campaign_control_kernel.py`, and `tests/test_campaign_roadmap_phase_pack.py`, and verified with `pytest tests/test_campaign_questionnaire_counterexamples.py tests/test_campaign_control_kernel.py tests/test_campaign_roadmap_phase_pack.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall bind graph-confidence, prompt-protocol identity, and predictive repo-twin signals into roadmap and replay artifacts through `core/campaign/risk_radar.py`, `core/campaign/control_plane.py`, `core/prompts/prompt_manager.py`, `shared_kernel/event_store.py`, and `core/campaign/roadmap_validator.py`, tested by `tests/test_saguaro_roadmap_validator.py`, `tests/test_event_store.py`, `tests/test_campaign_speculation_risk_and_governance.py`, and `tests/test_phase2_prompt_governance.py`, and verified with `pytest tests/test_saguaro_roadmap_validator.py tests/test_event_store.py tests/test_campaign_speculation_risk_and_governance.py tests/test_phase2_prompt_governance.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall compile roadmap drafts as witness-bearing execution contracts through `core/campaign/roadmap_compiler.py`, `core/campaign/phase_packet.py`, `core/campaign/roadmap_validator.py`, and `core/campaign/control_plane.py`, tested by `tests/test_campaign_roadmap_phase_pack.py`, `tests/test_saguaro_roadmap_validator.py`, and `tests/test_campaign_control_kernel.py`, and verified with `pytest tests/test_campaign_roadmap_phase_pack.py tests/test_saguaro_roadmap_validator.py tests/test_campaign_control_kernel.py`, `./venv/bin/saguaro roadmap validate --path inventive_research_roadmap.md --format json`, and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall execute EID-driven development through packet ABI v2 and native-runtime telemetry integration in `core/research/experiment_design.py`, `core/research/experiment_runner.py`, `core/campaign/telemetry_contracts.py`, `core/native/runtime_telemetry.py`, `core/qsg/runtime_contracts.py`, and `domains/verification/verification_lane.py`, tested by `tests/test_runtime_telemetry.py`, `tests/test_campaign_timeline_and_verification_lane.py`, `tests/test_development_replay.py`, and `tests/test_qsg_runtime_contracts.py`, and verified with `pytest tests/test_runtime_telemetry.py tests/test_campaign_timeline_and_verification_lane.py tests/test_development_replay.py tests/test_qsg_runtime_contracts.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall implement shadow-lap preflight and semantic wind-tunnel gating through `core/research/simulator_planner.py`, `core/research/eid_master.py`, `core/research/experiment_runner.py`, `core/campaign/control_plane.py`, `domains/verification/verification_lane.py`, and `saguaro/sentinel/engines/semantic.py`, tested by `tests/test_campaign_shadow_laps.py`, `tests/test_campaign_speculation_risk_and_governance.py`, and `tests/test_campaign_timeline_and_verification_lane.py`, and verified with `pytest tests/test_campaign_shadow_laps.py tests/test_campaign_speculation_risk_and_governance.py tests/test_campaign_timeline_and_verification_lane.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall emit unified replay tapes, mission capsules, and safety-case closure artifacts through `shared_kernel/event_store.py`, `core/campaign/completion_engine.py`, `core/campaign/control_plane.py`, `core/campaign/audit_engine.py`, `core/qsg/latent_bridge.py`, and `core/memory/fabric/store.py`, tested by `tests/test_event_store.py`, `tests/test_latent_package_capture.py`, `tests/test_campaign_closure_safety_case.py`, and `tests/test_saguaro_roadmap_validator.py`, and verified with `pytest tests/test_event_store.py tests/test_latent_package_capture.py tests/test_campaign_closure_safety_case.py tests/test_saguaro_roadmap_validator.py`, `./venv/bin/saguaro roadmap validate --path inventive_research_roadmap.md --format json`, and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
