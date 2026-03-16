# Anvil Roadmap: Silver Smoke Readiness, Benchmark Truth Fabric, and Reusable Audit Intelligence

Generated: 2026-03-11

Research basis:
- Repo grounding was performed from `venv` with `saguaro entrypoints`, `saguaro build-graph`, targeted `saguaro query` runs, `saguaro agent skeleton`, `saguaro agent slice`, and `saguaro impact`.
- `saguaro health` did not return structured output within the initial timeout window and emitted only TensorFlow/native startup noise, so health confidence in this document is inferred from successful `query`, `impact`, `entrypoints`, `build-graph`, and `skeleton` calls rather than a clean `health` report.
- The roadmap intentionally does not use the other markdown roadmap files in this repository as ideation sources.
- This document is written to be saveable as markdown and later checkable with `./venv/bin/saguaro roadmap validate --path anvil_roadmap.md --format json`.

## 1. First-Principles Framing

- Repo-grounded observation: Anvil is fundamentally a CPU-first, native-assisted, multi-agent engineering runtime with evidence, telemetry, and benchmark surfaces already distributed across `audit/runner/benchmark_suite.py`, `audit/runner/native_benchmark_runner.py`, `core/unified_chat_loop.py`, `core/native/native_qsg_engine.py`, `core/native/parallel_generation.py`, `core/telemetry/black_box.py`, and `shared_kernel/event_store.py`.
- Repo-grounded observation: Saguaro is fundamentally a code intelligence and verification substrate with graph, query, eval, metrics, evidence, and app surfaces already present in `saguaro/services/platform.py`, `saguaro/analysis/code_graph.py`, `saguaro/query/benchmark.py`, `saguaro/sentinel/engines/graph.py`, and `saguaro/cli.py`.
- Repo-grounded observation: The benchmark stack is already much more than a smoke harness. `audit/runner/benchmark_suite.py` contains preflight, calibration, ablations, continuous scheduler runs, kernel microbenchmarks, quality evals, memory replay hooks, artifact persistence, certification reporting, and governance summaries.
- Repo-grounded observation: The native runtime is already performance-aware. `core/native/native_qsg_engine.py` includes autotune profile loading, automatic thread selection, batch-thread selection, ubatch sizing, perf probing, runtime capability synthesis, telemetry annotation, and parallel generation construction.
- Repo-grounded observation: `core/native/parallel_generation.py` already models scheduling, sequence lifecycle, speculative verification, latent packets, tool evidence, suspend checkpoints, and metrics snapshots.
- Repo-grounded observation: The prompt and orchestration center is `core/unified_chat_loop.py`, where system prompt assembly, context compression guidance, evidence builders, governance, runtime gate runners, and message bus registration are already co-located.
- Repo-grounded observation: Latent and memory-state packaging is real, not aspirational. `core/memory/latent_memory.py` and `core/memory/fabric/store.py` already support latent packaging, tensor persistence, memory edges, embeddings, and feedback.
- Repo-grounded observation: Evidence capture is real, not aspirational. `core/telemetry/black_box.py` and `shared_kernel/event_store.py` already support run events, tool plans, verification records, performance snapshots, replay tapes, checkpoints, mission capsules, and safety-case export.
- Repo-grounded observation: Benchmark profile configuration already exists in `audit/profiles/native_qsg_silver.yaml` with strict host requirements, required tuning contracts, forced parallel decode, and explicit quality/perf lanes.
- Inference: The system is trying to become an evidence-bearing engineering platform where planning, runtime, performance, quality, governance, and refactoring intelligence operate on shared truth instead of isolated logs and JSON files.
- Inference: The strongest version of Anvil is not "a better chat agent." It is a mission-control-grade engineering runtime that can preflight, predict, tune, schedule, benchmark, verify, explain, and refuse unsafe or under-proven executions.
- Inference: The strongest version of Saguaro is not "semantic search for code." It is a reusable audit intelligence substrate that can ingest code, runtime, benchmark, and incident evidence from this repo or any other repo.
- Constraint: CPU topology, cache locality, NUMA placement, SIMD path selection, affinity policy, and memory bandwidth can dominate TTFT and decode TPS more than model count alone.
- Constraint: TTFT, steady-state decode throughput, continuous batching efficiency, memory pressure, coherence, perplexity, confidence calibration, and stability must be treated as a coupled system.
- Constraint: Silver readiness cannot be inferred from one lane. It must be proven across preflight, tuning, scheduler behavior, kernel hotspots, quality surfaces, and operator evidence completeness.
- Constraint: Saguaro itself consumes resources. Benchmark and runtime policy must reason about concurrency limits across Anvil agents, subagents, and Saguaro analysis jobs, not just model instances.
- Constraint: Cross-repo reuse matters. If the benchmark intelligence substrate only works inside this repository, it is strategically weak.
- Constraint: Native/Python boundary visibility is currently uneven. Python graph and service surfaces are clearer than native C++ causality surfaces, which means the roadmap must improve native attribution rather than pretend it is already solved.
- Constraint: Governance is present but thinly enforced in some paths. `core/aes/governance.py` is meaningful, while `infrastructure/hooks/base.py` and `core/campaign/telemetry.py` are still minimal relative to the repo's ambition.
- Engineering standard: No hidden fallback shall be considered acceptable at silver level.
- Engineering standard: No tuning decision shall be accepted without persisted host and model evidence.
- Engineering standard: No performance claim shall be accepted without latency, throughput, and quality coupling.
- Engineering standard: No graph insight shall remain postmortem-only if it can be converted into preflight prevention.
- Engineering standard: No operator-facing conclusion shall depend on a human manually reconciling disconnected artifacts.
- Engineering standard: No phase promotion shall happen without explicit rollback criteria and required evidence.
- Current design center of gravity: The benchmark harness, native runtime, event store, black box recorder, and Saguaro platform services already want to form a single truth fabric.
- Current architectural gap: Those pieces still behave like adjacent systems more than one coherent operating model.
- Strategic conclusion: The first job is not to add more benchmark features blindly. The first job is to establish a stronger architectural center of gravity around benchmark truth, runtime admission, resource governance, and reusable audit intelligence.

## 2. External Research Scan

- External research: [Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention"](https://arxiv.org/abs/2309.06180) and the [vLLM paged attention design docs](https://docs.vllm.ai/en/latest/design/paged_attention.html) matter because they treat memory layout and scheduler policy as the core serving problem, not secondary optimization.
- Why it matters here: `core/native/native_qsg_engine.py`, `core/native/parallel_generation.py`, and `audit/runner/benchmark_suite.py` already expose scheduling and runtime capability surfaces, but memory truth is not yet fused into the same promotion contract.
- External research: [Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models"](https://www.usenix.org/conference/osdi22/presentation/yu) matters because iteration-level scheduling changes the fairness and latency model from request batching to token-step scheduling.
- Why it matters here: `audit/profiles/native_qsg_silver.yaml` already includes `continuous_scheduler` lanes, but the current program still needs a sharper separation between scheduler truth, runtime truth, and adapter overhead.
- External research: [Agrawal et al., "Taming Throughput-Latency Trade-off in LLM Inference with Sarathi-Serve"](https://arxiv.org/abs/2403.02310) and the [Sarathi-Serve repository](https://github.com/microsoft/sarathi-serve) matter because chunked prefills and stall-free scheduling make TTFT and throughput a schedulable control problem, not a binary tradeoff.
- Why it matters here: the silver profile already declares `continuous_concurrency` and multiple scheduler policies, but it does not yet persist a capacity envelope that includes Saguaro resource contention and memory headroom.
- External research: [Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://proceedings.mlr.press/v235/cai24b.html) matters because speculative acceleration becomes measurable only when acceptance, rejection, and verification cost are first-class metrics.
- Why it matters here: `audit/schemas/attempt_record.schema.json` and `core/native/parallel_generation.py` already expose speculative surfaces, but the certification logic still needs deeper acceptance economics and rollback triggers.
- External research: [TensorRT-LLM speculative decoding docs](https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html) matter because they operationalize how draft models, Medusa-style heads, and verification trees become deployment surfaces instead of research toys.
- Why it matters here: `core/native/parallel_generation.py` already has multiple draft planners and verification paths. The missing move is to bind these to benchmark certification and runtime admission rather than leaving them as local capabilities.
- External research: [Hugging Face perplexity guidance](https://huggingface.co/docs/transformers/en/perplexity) matters because it explicitly warns that perplexity can be misleading when context assumptions and tokenization details are not controlled.
- Why it matters here: `audit/eval/native_logits.py` already computes perplexity and confidence, so the roadmap must force evaluation-context evidence into benchmark artifacts and promotion gates.
- External research: [HELM: Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) and the [HELM repository](https://github.com/stanford-crfm/helm) matter because they frame model evaluation as multidimensional transparency, not a single leaderboard number.
- Why it matters here: your stated target is not just raw speed. It is speed plus coherence, stability, confidence, and proveable quality. HELM-style thinking is a better fit than narrow throughput-only benchmarking.
- External research: [SelfCheckGPT](https://arxiv.org/abs/2303.08896) matters because output stability and internal agreement can be measured even without a perfect external oracle.
- Why it matters here: the silver suite already has a coherence rubric corpus in `benchmarks/corpora/coherence_rubric.jsonl`, but it does not yet appear to include a first-class consistency or self-disagreement lane.
- External research: [OpenTelemetry signals](https://opentelemetry.io/docs/concepts/signals/), [OpenTelemetry traces](https://opentelemetry.io/docs/concepts/signals/traces/), and [OpenTelemetry T-shaped signals guidance](https://opentelemetry.io/docs/specs/semconv/how-to-write-conventions/t-shaped-signals/) matter because they separate broad operator signals from deep specialist signals.
- Why it matters here: `core/campaign/telemetry.py` is too thin for the repo's ambition, while `core/telemetry/black_box.py` and `shared_kernel/event_store.py` already support richer traces. The missing layer is a signal contract, not another log sink.
- External research: [Parca](https://www.parca.dev/) and [Grafana Pyroscope](https://grafana.com/docs/pyroscope/latest/) matter because continuous profiling turns hotspot detection, memory leaks, and regression comparisons into always-on capabilities rather than lab-only sessions.
- Why it matters here: `audit/runtime_logging.py`, `core/native/runtime_telemetry.py`, and `benchmarks/native_kernel_microbench.py` already create performance artifacts. The roadmap should make those comparable over time and queryable by cause, version, and topology.
- External research: [`perf stat` on Linux](https://man7.org/linux/man-pages/man1/perf-stat.1.html) matters because interval and grouped PMU counters let engineers reason about microarchitectural behavior instead of just end-to-end wall clock.
- Why it matters here: the silver profile already requires perf, and `audit/runner/suite_preflight.py` already probes it. The benchmark program should promote PMU evidence from a binary prerequisite into stage-local attribution.
- External research: [Coz causal profiling](https://github.com/plasma-umass/coz) and ["Coz: Finding Code that Counts with Causal Profiling"](https://cacm.acm.org/research/coz/) matter because they quantify which optimizations actually matter instead of just showing where time is spent.
- Why it matters here: the repo already has hotspot and kernel summaries, but it still needs a stronger "what would actually move TTFT or decode TPS" mechanism when choosing optimization work.
- External research: [Intel's Optimization Reference Manual](https://www.intel.com/content/www/us/en/developer/articles/technical/intel64-and-ia-32-architectures-optimization-reference-manual.html) matters because AVX2 gains are bounded by memory layout, branch behavior, load/store pressure, and thread placement.
- Why it matters here: your instinct to optimize for AVX2 is sound, but silver should certify ISA paths in combination with cache and scheduling evidence, not as a compile-time badge.
- External research: [`llama.cpp`](https://github.com/ggml-org/llama.cpp) matters because it operationalized CPU-first inference, prompt evaluation profiling, and perplexity tooling as reusable primitives rather than one-off benchmarks.
- Why it matters here: Anvil's CPU-first design can benefit from the discipline of making low-level measurement and host-specific tuning operator-visible by default.
- External research: [Open MCT](https://nasa.github.io/openmct/) matters because mission control software treats streaming telemetry, history, anomalies, procedures, and operator context as one integrated system.
- Why it matters here: `core/telemetry/black_box.py`, `shared_kernel/event_store.py`, `audit/runtime_logging.py`, and `saguaro/services/platform.py` already have enough ingredients to justify a mission-control-grade audit view.
- External synthesis: The common thread across these sources is not "add more benchmark charts." It is to unify scheduling, memory, quality, profiling, and evidence into one operator and certification model.
- External synthesis: That unification is exactly where Anvil and Saguaro have the most durable moat potential because the repo already contains the separate primitives needed to do it.

## 3. Repo Grounding Summary

### 3.1 Commands Executed

- `source venv/bin/activate && saguaro entrypoints`
- `source venv/bin/activate && saguaro build-graph`
- `source venv/bin/activate && timeout 20s saguaro health`
- `source venv/bin/activate && saguaro query "benchmark audit kernel hotspots telemetry perplexity coherence stability continuous batching avx2" --k 8`
- `source venv/bin/activate && saguaro query "prompt assembly orchestration planning loops hooks lifecycle triggers memory telemetry governance" --k 8`
- `source venv/bin/activate && saguaro query "graph ffi math parse cpu scan analyze benchmark results debugging code errors relationships" --k 8`
- `source venv/bin/activate && saguaro agent skeleton audit/runner/benchmark_suite.py`
- `source venv/bin/activate && saguaro agent skeleton audit/runner/native_benchmark_runner.py`
- `source venv/bin/activate && saguaro agent slice UnifiedChatLoop._get_master_system_prompt --depth 1`
- `source venv/bin/activate && saguaro impact --path audit/runner/benchmark_suite.py`
- `source venv/bin/activate && saguaro agent skeleton core/native/native_qsg_engine.py`
- `source venv/bin/activate && saguaro agent skeleton core/native/parallel_generation.py`
- `source venv/bin/activate && saguaro agent skeleton core/native/runtime_telemetry.py`
- `source venv/bin/activate && saguaro agent skeleton core/memory/latent_memory.py`
- `source venv/bin/activate && saguaro agent skeleton core/memory/fabric/store.py`
- `source venv/bin/activate && saguaro agent skeleton core/telemetry/black_box.py`
- `source venv/bin/activate && saguaro agent skeleton shared_kernel/event_store.py`
- `source venv/bin/activate && saguaro agent skeleton saguaro/services/platform.py`
- `source venv/bin/activate && saguaro agent skeleton saguaro/analysis/code_graph.py`
- `source venv/bin/activate && saguaro agent skeleton audit/runner/suite_preflight.py`
- `source venv/bin/activate && saguaro agent skeleton audit/runner/suite_certification.py`
- `source venv/bin/activate && saguaro agent skeleton audit/runner/suite_profiles.py`

### 3.2 Surfaces Inspected

- Entry points found by Saguaro include `audit/runner/benchmark_suite.py`, `audit/runner/native_benchmark_runner.py`, `cli/repl.py`, `main.py`, `anvil.py`, `benchmarks/native_kernel_microbench.py`, and `saguaro/cli.py`.
- The build graph exposed native targets for `anvil_runtime_core`, `anvil_native_ops`, backend libraries, `saguaro_core`, `saguaro_native`, and the Python package target.
- `audit/runner/benchmark_suite.py` is the central benchmark orchestrator and contains calibration, continuous scheduling, kernel microbench, quality evaluation, memory replay, summary generation, certification state, and artifact persistence.
- `audit/runner/native_benchmark_runner.py` is the lower-level benchmark executor and summary builder with stage hotspot computation, continuous surface execution, attempt planning, checkpointing, and baseline storage.
- `audit/runner/suite_preflight.py` already probes Saguaro, perf, CPU governor, host contract, tuning contract, memory information, and runtime metadata before a suite run begins.
- `audit/runner/suite_certification.py` already contains host contract paths, tuning contract paths, validation routines, runtime tuning bootstrap, assessment, and affinity repair utilities.
- `audit/runner/suite_profiles.py` already models the benchmark profile contract through `BenchmarkSuiteSpec`, `ScenarioPack`, `KernelMicrobenchSpec`, `QualityEvalSpec`, and `MemoryReplaySpec`.
- `audit/profiles/native_qsg_silver.yaml` currently specifies `max_parallel_models: 1`, `require_saguaro: true`, `require_perf: true`, `tuning_contract_policy: required`, `force_parallel_decode: true`, `forbid_autoregressive_fallback: true`, plus quality and kernel lanes.
- `benchmarks/native_kernel_microbench.py` already maps named kernels to stages and source buckets, runs warmups and iterations, and produces recoverable-gain style summaries.
- `audit/eval/native_logits.py` already computes perplexity, entropy, confidence, and expected calibration error using `NativeQSGEngine`.
- `core/native/native_qsg_engine.py` already loads autotune profiles, infers thread counts, configures affinity, probes perf availability, builds runtime capabilities, annotates telemetry, and constructs parallel generation engines.
- `core/native/parallel_generation.py` already models sequence lifecycle, verification, draft planners, speculative execution plans, tool evidence, latent packets, suspend checkpoints, and scheduler metrics snapshots.
- `core/native/runtime_telemetry.py` already knows how to normalize runtime metrics into dictionaries with latency percentiles and mode-aware labels.
- `core/unified_chat_loop.py` already assembles the master prompt through `PromptManager`, adds context compression guidance, initializes Saguaro integration, registers a message bus, instantiates governance engines, and binds a black box recorder.
- `core/telemetry/black_box.py` already records events, tool plans, tool results, verification outcomes, performance snapshots, message segments, and replay context.
- `shared_kernel/event_store.py` already persists events, checkpoints, replay tapes, mission capsules, and safety-case exports.
- `core/memory/latent_memory.py` and `core/memory/fabric/store.py` already provide latent state packaging, edge storage, feedback, embeddings, and latent package retrieval.
- `core/aes/governance.py` already provides action and repo-action checks, but the governance layer is still thinner than the benchmark and runtime layers.
- `core/campaign/telemetry.py` exists, but it is currently minimal compared with the rest of the evidence and runtime stack.
- `infrastructure/hooks/base.py` defines only the base hook abstraction, which means lifecycle extensibility is not yet on the same maturity level as the benchmark and native runtime.
- `saguaro/services/platform.py` already exposes `GraphService`, `QueryService`, `EvidenceService`, `ResearchService`, `MetricsService`, `VerifyService`, `EvalService`, and `AppService`.
- `saguaro/analysis/code_graph.py` already composes `BridgeSynthesizer`, `CallGraphBuilder`, `CFGBuilder`, `DFGBuilder`, and `FFIScanner`.
- `saguaro/sentinel/engines/graph.py` already provides a graph-based sentinel engine, but its surface is still small compared with the potential of the surrounding graph stack.

### 3.3 Existing Primitives That Matter Most

- Benchmark profile contract primitive: `audit/runner/suite_profiles.py`
- Preflight and environment audit primitive: `audit/runner/suite_preflight.py`
- Host and tuning certification primitive: `audit/runner/suite_certification.py`
- Attempt and summary artifact primitive: `audit/runner/benchmark_suite.py`
- Kernel hotspot and baseline summary primitive: `audit/runner/native_benchmark_runner.py`
- Native runtime autotune primitive: `core/native/native_qsg_engine.py`
- Continuous scheduling and speculative verification primitive: `core/native/parallel_generation.py`
- Runtime metrics normalization primitive: `core/native/runtime_telemetry.py`
- Quality measurement primitive: `audit/eval/native_logits.py`
- Evidence and replay primitive: `core/telemetry/black_box.py`
- Event and safety-case storage primitive: `shared_kernel/event_store.py`
- Prompt and orchestration primitive: `core/unified_chat_loop.py`
- Latent packaging primitive: `core/memory/latent_memory.py`
- Memory fabric primitive: `core/memory/fabric/store.py`
- Graph build and query primitive: `saguaro/analysis/code_graph.py` and `saguaro/services/platform.py`
- Benchmark scoring primitive for Saguaro query quality: `saguaro/query/benchmark.py`

### 3.4 Where the Architecture Is Strongest

- Strongest area: benchmark orchestration depth
- Why: the benchmark suite already reasons about preflight, contracts, runtime metadata, kernel lanes, quality lanes, and certification outputs.
- Strongest area: native runtime introspection
- Why: the runtime already has host probing, thread heuristics, capability synthesis, speculative planning, and detailed telemetry hooks.
- Strongest area: evidence persistence
- Why: the black box recorder and event store already provide a non-trivial base for replayable execution evidence.
- Strongest area: reusable Saguaro service shell
- Why: graph, query, metrics, research, verify, and app services are already co-located inside one platform layer.

### 3.5 Where the Architecture Looks Underexploited or Thin

- Underexploited area: unified benchmark truth graph
- Why: benchmark artifacts, runtime telemetry, event store records, and Saguaro graph data are still adjacent rather than fused.
- Underexploited area: runtime admission and backpressure
- Why: silver currently constrains `max_parallel_models`, but there is no explicit repo-wide capacity governor for model plus Saguaro plus agent concurrency.
- Underexploited area: signal semantics
- Why: `core/campaign/telemetry.py` is not yet the operator-grade signal contract the rest of the stack deserves.
- Underexploited area: graph-assisted preflight prevention
- Why: `FFIScanner`, `CodeGraph`, and `GraphService` exist, but they are not yet being used as a first-class "debug before run" gate for benchmark and runtime changes.
- Underexploited area: cross-repo reuse
- Why: the audit stack is powerful, but its artifact semantics are still Anvil-centric instead of explicitly portable.
- Thin area: hook and lifecycle framework
- Why: `infrastructure/hooks/base.py` is minimal compared with the rest of the system.
- Thin area: native cause attribution
- Why: kernel summaries exist, but "which optimization would actually move silver" is not yet represented with causal or counterfactual evidence.

## 4. Hidden Assumptions

- Assumption 1: maximizing AVX2 usage is automatically the dominant path to silver readiness.
- Assumption 2: a per-model tuning contract is sufficient without a joint memory-and-analysis capacity envelope.
- Assumption 3: `max_parallel_models: 1` is enough to protect correctness without modeling Saguaro concurrency, event logging pressure, and memory-fabric usage.
- Assumption 4: benchmark JSON artifacts can remain human-correlated instead of machine-related.
- Assumption 5: perplexity plus a rubric corpus is sufficient to certify coherence and stability.
- Assumption 6: preflight is enough even if runtime conditions drift after launch.
- Assumption 7: graph intelligence is mostly a post-run forensic tool rather than a pre-run error prevention tool.
- Assumption 8: Anvil and Saguaro can make resource decisions independently.
- Assumption 9: throughput optimization work should be prioritized from hotspots alone instead of hotspot plus counterfactual impact.
- Assumption 10: the benchmark program should stay repo-specific instead of becoming a reusable substrate for any codebase.

## 5. Candidate Implementation Phases

### Candidate 01. Benchmark Truth Fabric

- Class: Practical
- Suggested `phase_id`: `research`
- Core insight: unify benchmark attempts, runtime metrics, PMU counters, quality evidence, and governance outcomes into one typed evidence graph.
- External inspiration or analogy: vLLM's memory-first serving discipline plus OpenTelemetry signal design plus NASA mission telemetry integration.
- Why it fits Saguaro and Anvil specifically: Anvil already emits rich artifacts while Saguaro already has `MetricsService`, `EvidenceService`, and `GraphService`.
- Exact places in this codebase where it could wire in: `audit/runner/benchmark_suite.py`, `audit/runner/native_benchmark_runner.py`, `audit/store/suite_layout.py`, `audit/schemas/summary.schema.json`, `core/telemetry/black_box.py`, `shared_kernel/event_store.py`, `saguaro/services/platform.py`.
- Existing primitives it can reuse: suite artifacts, black box events, event store export, Saguaro metrics runs, Saguaro evidence bundles.
- New primitive, data flow, or subsystem needed: a normalized benchmark evidence bundle schema and relationship graph.
- `repo_scope`: `audit/**`, `core/telemetry/**`, `shared_kernel/**`, `saguaro/services/**`, `saguaro/query/**`
- `owning_specialist_type`: `benchmark_assurance_specialist`
- `allowed_writes`: `audit/**`, `core/telemetry/**`, `shared_kernel/**`, `saguaro/services/**`, `tests/**`
- `telemetry_contract`: emit run, lane, artifact, counter, quality, and gate spans with stable IDs that can be joined across files and replay tapes.
- `required_evidence`: bundle schema version, artifact index, join IDs, timing coverage, counter coverage, quality coverage, and gate coverage.
- `rollback_criteria`: revert if bundle creation adds significant runtime overhead or breaks existing suite artifact compatibility.
- `promotion_gate`: all silver lanes persist into one coherent bundle and old reports remain readable.
- `success_criteria`: operators can answer "what happened, why, and where it changed" from one bundle instead of manual file correlation.
- Why this creates value: it upgrades the benchmark from artifact generation to machine-usable operational truth.
- Why this creates moat: most agent systems have logging; few have benchmark, runtime, and code graph evidence fused together.
- Main risk or failure mode: schema bloat without enough query ergonomics.
- Smallest credible first experiment: persist one joinable bundle for `kernel_microbench`, `quality_eval`, and `continuous_scheduler` lanes.
- Confidence level: high

### Candidate 02. Silver Admission Freeze

- Class: Practical
- Suggested `phase_id`: `intake`
- Core insight: silver should fail closed unless host contract, tuning contract, runtime capability envelope, and artifact completeness checks all pass together.
- External inspiration or analogy: aerospace launch commit criteria and F1 parc ferme discipline.
- Why it fits Saguaro and Anvil specifically: `suite_preflight.py` and `suite_certification.py` already implement much of the contract logic.
- Exact places in this codebase where it could wire in: `audit/runner/suite_preflight.py`, `audit/runner/suite_certification.py`, `audit/profiles/native_qsg_silver.yaml`, `audit/runner/benchmark_suite.py`, `core/aes/governance.py`.
- Existing primitives it can reuse: `run_preflight`, `validate_host_contract`, `validate_tuning_contract`, `bootstrap_runtime_tuning`, `ensure_runtime_affinity`.
- New primitive, data flow, or subsystem needed: a silver admission manifest with explicit required evidence and failure taxonomy.
- `repo_scope`: `audit/runner/**`, `audit/profiles/**`, `core/aes/**`, `tests/audit/**`
- `owning_specialist_type`: `runtime_governance_specialist`
- `allowed_writes`: `audit/runner/**`, `audit/profiles/**`, `core/aes/**`, `tests/**`
- `telemetry_contract`: emit admission decision, failed prerequisites, waived prerequisites, and rollback recommendation.
- `required_evidence`: host contract digest, tuning contract digest, harness hash, perf status, Saguaro status, memory envelope, and feature flag digest.
- `rollback_criteria`: revert if valid silver runs are blocked by non-deterministic admission checks.
- `promotion_gate`: silver cannot proceed on incomplete evidence and provides actionable remediation messages.
- `success_criteria`: silver readiness becomes an enforceable state, not a human judgment call.
- Why this creates value: it prevents false confidence before expensive or misleading runs.
- Why this creates moat: it turns performance work into a contract-governed operation.
- Main risk or failure mode: over-strict gating on early iterations.
- Smallest credible first experiment: add a silver-only manifest that must be persisted before the first measured lane begins.
- Confidence level: high

### Candidate 03. Runtime Tuning Boot Kernel

- Class: Practical
- Suggested `phase_id`: `eid`
- Core insight: supported models should cold-start into an explicit tuning lifecycle that computes, validates, persists, and reuses certified thread, ubatch, and affinity choices.
- External inspiration or analogy: firmware calibration tables and engine control maps.
- Why it fits Saguaro and Anvil specifically: `bootstrap_runtime_tuning` already exists in `suite_certification.py`, and `NativeQSGEngine` already loads autotune profiles.
- Exact places in this codebase where it could wire in: `audit/runner/suite_certification.py`, `core/native/native_qsg_engine.py`, `audit/contracts/tuning/**`, `audit/runner/benchmark_suite.py`, `cli/repl.py`.
- Existing primitives it can reuse: autotune profile loading, tuning contract validation, `assess_runtime_tuning`, thread heuristics, affinity repair.
- New primitive, data flow, or subsystem needed: boot-time runtime tuning handshake shared by benchmark runs and interactive Anvil startup.
- `repo_scope`: `audit/runner/**`, `core/native/**`, `cli/**`, `tests/**`
- `owning_specialist_type`: `runtime_performance_specialist`
- `allowed_writes`: `audit/runner/**`, `core/native/**`, `cli/**`, `tests/**`, `audit/contracts/**`
- `telemetry_contract`: emit tune-needed, tune-running, tune-certified, and tune-rejected events with host and model digests.
- `required_evidence`: initial runtime capability snapshot, tuning candidate table, final contract, and post-tuning sanity probe.
- `rollback_criteria`: revert if tune-on-start significantly degrades startup without reducing later run instability.
- `promotion_gate`: startup can determine whether tuning is missing, stale, or reusable before normal operation continues.
- `success_criteria`: every supported model starts with a known contract state and no silent "best guess" runtime.
- Why this creates value: it removes hidden variability from the first serious benchmark or real run.
- Why this creates moat: it binds host truth and model truth into reusable startup mechanics.
- Main risk or failure mode: startup time inflation and user frustration if calibration is too heavy.
- Smallest credible first experiment: add a lightweight tune status probe at Anvil start and a deferred calibration prompt for missing contracts.
- Confidence level: high

### Candidate 04. Continuous Batch Governor

- Class: Practical
- Suggested `phase_id`: `development`
- Core insight: continuous batching should be governed by measured latency-quality-memory envelopes rather than fixed concurrency lists.
- External inspiration or analogy: Sarathi-Serve stall-free scheduling and F1 race strategy adapting to current tire state, not static plans.
- Why it fits Saguaro and Anvil specifically: `parallel_generation.py`, `runtime_telemetry.py`, and silver profile concurrency lists already expose the control levers.
- Exact places in this codebase where it could wire in: `core/native/parallel_generation.py`, `core/native/native_qsg_engine.py`, `core/native/runtime_telemetry.py`, `audit/runner/benchmark_suite.py`, `audit/profiles/native_qsg_silver.yaml`.
- Existing primitives it can reuse: scheduler metrics snapshots, benchmark labels, runtime capability snapshots, continuous scheduler lane.
- New primitive, data flow, or subsystem needed: a governor policy that adjusts concurrency and queueing based on observed TTFT, TPS, and memory headroom.
- `repo_scope`: `core/native/**`, `audit/runner/**`, `audit/profiles/**`, `tests/**`
- `owning_specialist_type`: `scheduler_systems_specialist`
- `allowed_writes`: `core/native/**`, `audit/runner/**`, `audit/profiles/**`, `tests/**`
- `telemetry_contract`: emit queue depth, admission delay, TTFT percentiles, decode percentiles, rejection events, and governor decisions.
- `required_evidence`: before-and-after scheduler traces, memory envelope deltas, throughput curves, and quality deltas.
- `rollback_criteria`: revert if adaptive scheduling improves throughput but destabilizes TTFT or quality.
- `promotion_gate`: governor performs at least as well as the static policy on silver and better on stressed continuous lanes.
- `success_criteria`: batch control becomes data-driven and safety-bounded.
- Why this creates value: it directly targets your TTFT and continuous batching concerns.
- Why this creates moat: it moves from "supports continuous batching" to "governs continuous batching with evidence."
- Main risk or failure mode: oscillatory policy or overfitting to benchmark prompts.
- Smallest credible first experiment: adaptive cap selection for the `continuous_concurrency` silver lane without changing decode algorithms.
- Confidence level: medium-high

### Candidate 05. Memory Pressure Flight Controller

- Class: Practical
- Suggested `phase_id`: `development`
- Core insight: memory, Saguaro analysis jobs, model instances, and agent work should share one explicit admission budget.
- External inspiration or analogy: spacecraft power budgeting and mission resource arbitration.
- Why it fits Saguaro and Anvil specifically: `max_parallel_models: 1` already hints at resource control, while `MemoryFabricStore`, `LatentMemory`, and Saguaro services consume additional footprint.
- Exact places in this codebase where it could wire in: `core/native/native_qsg_engine.py`, `core/native/parallel_generation.py`, `core/memory/fabric/store.py`, `core/memory/latent_memory.py`, `core/orchestrator/scheduler.py`, `saguaro/services/platform.py`, `audit/runner/suite_preflight.py`.
- Existing primitives it can reuse: runtime capability snapshots, latent package storage, orchestrator task queue, preflight memory inspection.
- New primitive, data flow, or subsystem needed: a shared capacity envelope and wait queue for models, Saguaro analysis, and subagent jobs.
- `repo_scope`: `core/native/**`, `core/memory/**`, `core/orchestrator/**`, `saguaro/services/**`, `audit/runner/**`, `tests/**`
- `owning_specialist_type`: `memory_systems_specialist`
- `allowed_writes`: `core/native/**`, `core/memory/**`, `core/orchestrator/**`, `saguaro/services/**`, `audit/runner/**`, `tests/**`
- `telemetry_contract`: emit capacity reservations, queue delays, memory pressure transitions, denied admissions, and wake-up events.
- `required_evidence`: memory watermark traces, reservation logs, queue timing, and benchmark impact comparisons.
- `rollback_criteria`: revert if the controller adds friction without improving stability or preserving memory headroom.
- `promotion_gate`: the system can prove why work was admitted, delayed, or denied.
- `success_criteria`: concurrency limits become explicit, explainable, and cross-subsystem.
- Why this creates value: it addresses the exact concern that Saguaro and agents must sometimes wait for accuracy-preserving resource availability.
- Why this creates moat: it makes Anvil a governed engineering runtime rather than a best-effort concurrent toy.
- Main risk or failure mode: simplistic budgeting that blocks too much work.
- Smallest credible first experiment: add a shared reservation gate around silver benchmarking and Saguaro graph-heavy commands.
- Confidence level: high

### Candidate 06. Quality Stability Observatory

- Class: Practical
- Suggested `phase_id`: `analysis_upgrade`
- Core insight: silver should treat coherence, consistency, confidence calibration, and perplexity drift as an observatory, not a single report file.
- External inspiration or analogy: HELM-style multi-axis evaluation plus SelfCheckGPT-style stability checks.
- Why it fits Saguaro and Anvil specifically: `native_logits.py` already computes perplexity and calibration signals, and the silver profile already names a coherence rubric corpus.
- Exact places in this codebase where it could wire in: `audit/eval/native_logits.py`, `audit/runner/benchmark_suite.py`, `benchmarks/corpora/coherence_rubric.jsonl`, `audit/schemas/summary.schema.json`, `saguaro/services/platform.py`.
- Existing primitives it can reuse: perplexity scoring, confidence evaluation, quality record generation, quality gates.
- New primitive, data flow, or subsystem needed: repeated-sampling stability analysis and longitudinal quality drift views.
- `repo_scope`: `audit/eval/**`, `audit/runner/**`, `benchmarks/corpora/**`, `saguaro/services/**`, `tests/**`
- `owning_specialist_type`: `evaluation_science_specialist`
- `allowed_writes`: `audit/eval/**`, `audit/runner/**`, `benchmarks/corpora/**`, `saguaro/services/**`, `tests/**`
- `telemetry_contract`: emit per-model quality envelope, sample stability, calibration error, coherence failures, and drift classification.
- `required_evidence`: repeated sample sets, rubric scores, calibration histograms, perplexity context metadata, and gate outcomes.
- `rollback_criteria`: revert if added evaluation cost yields noisy or non-actionable quality signals.
- `promotion_gate`: quality surfaces show both static score and stability behavior.
- `success_criteria`: silver can detect when a faster runtime became less stable, less coherent, or poorly calibrated.
- Why this creates value: it closes the exact truth gap you called out around proving or disproving hypotheses.
- Why this creates moat: it joins systems optimization and model reliability in one program.
- Main risk or failure mode: quality signals become too expensive or too prompt-sensitive.
- Smallest credible first experiment: repeated sampling on a narrow stability subset for silver runs.
- Confidence level: high

### Candidate 07. Confidence Calibration Lane

- Class: Practical
- Suggested `phase_id`: `analysis_upgrade`
- Core insight: confidence should be benchmarked as a calibration problem, not just a model score export.
- External inspiration or analogy: reliability diagrams in supervised learning and mission sensor calibration loops.
- Why it fits Saguaro and Anvil specifically: `evaluate_confidence` already computes expected calibration error in `audit/eval/native_logits.py`.
- Exact places in this codebase where it could wire in: `audit/eval/native_logits.py`, `audit/runner/benchmark_suite.py`, `audit/schemas/summary.schema.json`, `tests/audit/test_benchmark_suite.py`.
- Existing primitives it can reuse: expected calibration error, percentile logic, confidence corpus.
- New primitive, data flow, or subsystem needed: promotion thresholds and failure taxonomies specific to calibration drift.
- `repo_scope`: `audit/eval/**`, `audit/runner/**`, `audit/schemas/**`, `tests/**`
- `owning_specialist_type`: `evaluation_science_specialist`
- `allowed_writes`: `audit/eval/**`, `audit/runner/**`, `audit/schemas/**`, `tests/**`
- `telemetry_contract`: emit calibration bins, ECE, confidence skew, and failure class.
- `required_evidence`: calibration curve artifact, corpus digest, and comparison against prior certified runs.
- `rollback_criteria`: revert if calibration checks duplicate other quality surfaces without adding signal.
- `promotion_gate`: silver must expose calibration status explicitly rather than burying it in summary JSON.
- `success_criteria`: confidence is promotable as a first-class quality dimension.
- Why this creates value: it directly aligns with your request to measure confidence and prove or disprove assumptions.
- Why this creates moat: it upgrades agent/runtime evaluation from "works" to "knows how much to trust itself."
- Main risk or failure mode: threshold selection may become arbitrary.
- Smallest credible first experiment: add an ECE gate and one failure report artifact per model.
- Confidence level: high

### Candidate 08. Cross-Repo Audit Bundle Format

- Class: Practical
- Suggested `phase_id`: `feature_map`
- Core insight: benchmark audit intelligence becomes reusable only if the artifact contract is portable across repositories.
- External inspiration or analogy: black box flight recorder formats and reproducible benchmark harness packaging.
- Why it fits Saguaro and Anvil specifically: Saguaro already has `ResearchService`, `MetricsService`, and `EvidenceService`, while Anvil already produces rich benchmark artifacts.
- Exact places in this codebase where it could wire in: `audit/schemas/**`, `audit/runner/benchmark_suite.py`, `saguaro/services/platform.py`, `saguaro/query/benchmark.py`, `tests/fixtures/saguaro_accuracy/anvil_query_benchmark.json`.
- Existing primitives it can reuse: benchmark schemas, Saguaro benchmark scoring, evidence bundles.
- New primitive, data flow, or subsystem needed: a canonical import/export contract with stable IDs, provenance, and artifact role labels.
- `repo_scope`: `audit/schemas/**`, `audit/runner/**`, `saguaro/services/**`, `saguaro/query/**`, `tests/**`
- `owning_specialist_type`: `platform_interop_specialist`
- `allowed_writes`: `audit/schemas/**`, `audit/runner/**`, `saguaro/services/**`, `saguaro/query/**`, `tests/**`
- `telemetry_contract`: emit imported bundle metadata, schema compatibility state, and provenance trace.
- `required_evidence`: schema version, required artifact set, provenance digest, and import validation report.
- `rollback_criteria`: revert if portability requirements break local benchmark ergonomics.
- `promotion_gate`: Saguaro can ingest one Anvil run and one external run through the same interface.
- `success_criteria`: audit intelligence becomes a reusable product instead of a repo-local script bundle.
- Why this creates value: it makes your "from here or any repository" requirement concrete.
- Why this creates moat: it creates a portable truth layer others can build on.
- Main risk or failure mode: over-generalized schema that loses important local nuance.
- Smallest credible first experiment: export one silver run into a self-contained portable evidence bundle.
- Confidence level: high

### Candidate 09. Saguaro Benchmark Ingestor

- Class: Practical
- Suggested `phase_id`: `feature_map`
- Core insight: Saguaro should ingest benchmark audit bundles and expose graph, metrics, and query views over them.
- External inspiration or analogy: Open MCT telemetry plugins and observability backends ingesting heterogeneous signal types.
- Why it fits Saguaro and Anvil specifically: `platform.py` already contains `ResearchService`, `MetricsService`, `EvalService`, and `AppService`.
- Exact places in this codebase where it could wire in: `saguaro/services/platform.py`, `saguaro/query/benchmark.py`, `saguaro/analysis/code_graph.py`, `audit/runner/benchmark_suite.py`, `core/telemetry/black_box.py`.
- Existing primitives it can reuse: metrics writing, evidence bundles, graph service, query benchmark scoring.
- New primitive, data flow, or subsystem needed: an audit ingestor that maps benchmark evidence into queryable Saguaro entities and edges.
- `repo_scope`: `saguaro/services/**`, `saguaro/query/**`, `saguaro/analysis/**`, `audit/**`, `tests/**`
- `owning_specialist_type`: `graph_intelligence_specialist`
- `allowed_writes`: `saguaro/services/**`, `saguaro/query/**`, `saguaro/analysis/**`, `audit/**`, `tests/**`
- `telemetry_contract`: emit ingest counts, unresolved references, derived relationships, and query readiness state.
- `required_evidence`: imported nodes, imported edges, unresolved artifacts, and benchmark query accuracy regression results.
- `rollback_criteria`: revert if ingest adds storage complexity without improving query usefulness.
- `promotion_gate`: benchmark runs become queryable through Saguaro APIs and app surfaces.
- `success_criteria`: engineers can ask Saguaro about benchmark failures, hotspot shifts, and run comparisons directly.
- Why this creates value: it transforms benchmark data from static artifacts into analyzable intelligence.
- Why this creates moat: it creates a generalized benchmark-debugging substrate few repos have.
- Main risk or failure mode: noisy relationship inference if the artifact model is weak.
- Smallest credible first experiment: ingest lane summary, kernel summary, and quality summary into `MetricsService` plus a thin query layer.
- Confidence level: high

### Candidate 10. Graph FFI Preflight Sentinel

- Class: Practical
- Suggested `phase_id`: `research`
- Core insight: graph and FFI analysis should block or warn on likely native-boundary regressions before benchmark execution.
- External inspiration or analogy: preflight structural checks in avionics integration and static link-time interface validation.
- Why it fits Saguaro and Anvil specifically: `CodeGraph` already composes `FFIScanner`, and a graph sentinel engine already exists.
- Exact places in this codebase where it could wire in: `saguaro/analysis/code_graph.py`, `saguaro/sentinel/engines/graph.py`, `saguaro/services/platform.py`, `core/native/**`, `audit/runner/suite_preflight.py`.
- Existing primitives it can reuse: `FFIScanner`, `GraphService`, `CodeGraphEngine`, preflight remediation reporting.
- New primitive, data flow, or subsystem needed: preflight rules that reason about Python-to-native symbol drift and graph breakage risk.
- `repo_scope`: `saguaro/analysis/**`, `saguaro/sentinel/**`, `saguaro/services/**`, `audit/runner/**`, `tests/**`
- `owning_specialist_type`: `graph_intelligence_specialist`
- `allowed_writes`: `saguaro/analysis/**`, `saguaro/sentinel/**`, `saguaro/services/**`, `audit/runner/**`, `tests/**`
- `telemetry_contract`: emit detected boundary risks, unresolved symbols, changed bridge edges, and severity classification.
- `required_evidence`: graph diff, FFI scan result, rule report, and preflight decision trace.
- `rollback_criteria`: revert if false positives overwhelm usability.
- `promotion_gate`: the system can catch at least one real native-boundary issue before runtime.
- `success_criteria`: graph intelligence becomes a prevention tool, not only a forensic tool.
- Why this creates value: it directly matches your request for graph-FFI-assisted debugging before hitting runtime errors.
- Why this creates moat: it ties semantic code intelligence to operational risk prevention.
- Main risk or failure mode: insufficient native symbol fidelity.
- Smallest credible first experiment: add a preflight report for changed files touching `core/native/**` and related wrappers.
- Confidence level: high

### Candidate 11. Causal Hotspot Prioritizer

- Class: Moonshot
- Suggested `phase_id`: `analysis_upgrade`
- Core insight: hotspot reports should be upgraded with counterfactual impact estimates so engineers know which optimization is worth doing.
- External inspiration or analogy: Coz causal profiling and race-engineering what-if analysis.
- Why it fits Saguaro and Anvil specifically: benchmark summaries already compute hotspots, but prioritization is still mostly descriptive.
- Exact places in this codebase where it could wire in: `benchmarks/native_kernel_microbench.py`, `audit/runner/native_benchmark_runner.py`, `audit/runner/benchmark_suite.py`, `core/native/runtime_telemetry.py`, `saguaro/services/platform.py`.
- Existing primitives it can reuse: kernel stage map, recoverable gain estimates, runtime metrics snapshots, Saguaro metrics storage.
- New primitive, data flow, or subsystem needed: virtual-speedup style experiments and causal ranking output.
- `repo_scope`: `benchmarks/**`, `audit/runner/**`, `core/native/**`, `saguaro/services/**`, `tests/**`
- `owning_specialist_type`: `performance_science_specialist`
- `allowed_writes`: `benchmarks/**`, `audit/runner/**`, `core/native/**`, `saguaro/services/**`, `tests/**`
- `telemetry_contract`: emit observed hotspot, simulated speedup assumption, predicted TTFT delta, and predicted TPS delta.
- `required_evidence`: experimental method description, confidence interval, and agreement with real follow-up measurements.
- `rollback_criteria`: revert if counterfactual estimates are unstable or misleading.
- `promotion_gate`: prioritizer correctly predicts at least one optimization order on a controlled benchmark set.
- `success_criteria`: optimization planning moves from "hottest code" to "highest leverage code."
- Why this creates value: it reduces wasted kernel optimization effort.
- Why this creates moat: it adds a stronger mechanism than ordinary profiling.
- Main risk or failure mode: native workloads may be too complex for cheap causal approximation.
- Smallest credible first experiment: inject artificial delay into one kernel class and compare predicted vs observed TTFT/TPS change.
- Confidence level: medium

### Candidate 12. Scheduler-Memory Digital Twin

- Class: Moonshot
- Suggested `phase_id`: `roadmap_draft`
- Core insight: build a predictive model of queueing, memory pressure, and scheduling outcomes before admitting real work.
- External inspiration or analogy: digital twins in aerospace and motorsport telemetry strategy.
- Why it fits Saguaro and Anvil specifically: the repo already has runtime telemetry, scheduler metrics, and event history, but no predictive envelope model.
- Exact places in this codebase where it could wire in: `core/native/parallel_generation.py`, `core/native/runtime_telemetry.py`, `shared_kernel/event_store.py`, `saguaro/services/platform.py`, `audit/runner/benchmark_suite.py`.
- Existing primitives it can reuse: sequence ledgers, scheduler metrics snapshots, run exports, metrics runs.
- New primitive, data flow, or subsystem needed: a digital twin model trained or parameterized from recorded silver and deepdive runs.
- `repo_scope`: `core/native/**`, `shared_kernel/**`, `saguaro/services/**`, `audit/**`, `tests/**`
- `owning_specialist_type`: `simulation_systems_specialist`
- `allowed_writes`: `core/native/**`, `shared_kernel/**`, `saguaro/services/**`, `audit/**`, `tests/**`
- `telemetry_contract`: emit predicted queue metrics, predicted headroom, predicted TTFT/TPS, and prediction error after real execution.
- `required_evidence`: prediction vs outcome traces, calibration report, and admission decisions informed by the twin.
- `rollback_criteria`: revert if the twin consumes engineering energy without improving decisions.
- `promotion_gate`: the twin predicts enough to avoid at least one harmful admission or misconfigured run.
- `success_criteria`: runtime planning can test hypotheses before paying the full cost of real execution.
- Why this creates value: it extends your "skip assumptions and ground in truth" requirement into predictive experimentation.
- Why this creates moat: it turns historical runs into an active decision surface.
- Main risk or failure mode: model complexity outruns available training data.
- Smallest credible first experiment: predict memory pressure and TTFT from recent continuous scheduler traces only.
- Confidence level: medium-low

### Candidate 13. Counterfactual Kernel Lab

- Class: Moonshot
- Suggested `phase_id`: `analysis_upgrade`
- Core insight: benchmark harnesses should support targeted synthetic perturbations that test whether a kernel or scheduler assumption is actually causal.
- External inspiration or analogy: fault injection and causal profiling combined.
- Why it fits Saguaro and Anvil specifically: microbench and runtime telemetry already exist; what is missing is a disciplined perturbation lab.
- Exact places in this codebase where it could wire in: `benchmarks/native_kernel_microbench.py`, `core/native/qsg_parallel_kernels.cpp`, `core/native/cpu_math_runtime.cpp`, `audit/runner/benchmark_suite.py`, `saguaro/services/platform.py`.
- Existing primitives it can reuse: kernel maps, run loops, runtime telemetry, benchmark summaries.
- New primitive, data flow, or subsystem needed: a perturbation API for artificial delays, throttles, or feature toggles per kernel class.
- `repo_scope`: `benchmarks/**`, `core/native/**`, `audit/runner/**`, `saguaro/services/**`, `tests/**`
- `owning_specialist_type`: `performance_science_specialist`
- `allowed_writes`: `benchmarks/**`, `core/native/**`, `audit/runner/**`, `saguaro/services/**`, `tests/**`
- `telemetry_contract`: emit perturbation type, target stage, amplitude, observed change, and confidence in causal attribution.
- `required_evidence`: perturbation manifest, results table, and predicted-vs-observed deltas.
- `rollback_criteria`: revert if perturbations contaminate production-like benchmarks or complicate maintenance.
- `promotion_gate`: perturbation results improve optimization prioritization accuracy.
- `success_criteria`: engineers can experimentally prove which low-level changes matter.
- Why this creates value: it makes kernel tuning more scientific.
- Why this creates moat: it is an uncommon mechanism in agent/runtime repos.
- Main risk or failure mode: perturbation machinery may be invasive.
- Smallest credible first experiment: synthetic delay insertion in one benchmark-only path for one kernel family.
- Confidence level: medium-low

### Candidate 14. Mission Control Wall

- Class: Practical
- Suggested `phase_id`: `feature_map`
- Core insight: benchmark, runtime, and governance evidence should be observable through one mission-control-grade operator surface.
- External inspiration or analogy: Open MCT and motorsport telemetry walls.
- Why it fits Saguaro and Anvil specifically: `AppService`, `BlackBoxRecorder`, `EventStore`, and runtime artifacts already exist.
- Exact places in this codebase where it could wire in: `saguaro/services/platform.py`, `core/telemetry/black_box.py`, `shared_kernel/event_store.py`, `audit/runtime_logging.py`, `core/native/runtime_telemetry.py`.
- Existing primitives it can reuse: app dashboard, campaign summary, event exports, black box artifacts, runtime telemetry dictionaries.
- New primitive, data flow, or subsystem needed: a mission-control view that joins operator signals, deep signals, and promotion gates.
- `repo_scope`: `saguaro/services/**`, `core/telemetry/**`, `shared_kernel/**`, `audit/**`, `tests/**`
- `owning_specialist_type`: `operator_experience_specialist`
- `allowed_writes`: `saguaro/services/**`, `core/telemetry/**`, `shared_kernel/**`, `audit/**`, `tests/**`
- `telemetry_contract`: expose broad signals for status and deep signals for diagnosis with shared trace IDs.
- `required_evidence`: dashboard payload schema, traceability from widget to artifact, and operator drill-down flows.
- `rollback_criteria`: revert if dashboard work outruns the truth model underneath it.
- `promotion_gate`: an operator can trace one failed silver run from admission to hotspot to rollback cause in one surface.
- `success_criteria`: operational comprehension time drops dramatically.
- Why this creates value: it removes manual correlation friction.
- Why this creates moat: it makes the benchmark/audit system usable at mission-control standards.
- Main risk or failure mode: UI polish can distract from core data contracts.
- Smallest credible first experiment: add one joined view in `AppService` for silver admission, runtime, and quality outcome.
- Confidence level: medium-high

### Candidate 15. Resource Reservation and Wait Queue

- Class: Practical
- Suggested `phase_id`: `development`
- Core insight: when Saguaro-heavy analysis or model-heavy runtime work would compromise accuracy, the system should queue instead of degrade silently.
- External inspiration or analogy: air-traffic slot assignment and compute cluster queueing.
- Why it fits Saguaro and Anvil specifically: `AgentOrchestrator`, `TaskQueue`, `NativeParallelGenerationEngine`, and Saguaro services already provide multiple workload sources.
- Exact places in this codebase where it could wire in: `core/orchestrator/scheduler.py`, `core/native/parallel_generation.py`, `core/native/native_qsg_engine.py`, `saguaro/services/platform.py`, `shared_kernel/event_store.py`.
- Existing primitives it can reuse: task queue, scheduler metrics, event logging, runtime capability snapshots.
- New primitive, data flow, or subsystem needed: a shared reservation broker and explicit wait-state semantics.
- `repo_scope`: `core/orchestrator/**`, `core/native/**`, `saguaro/services/**`, `shared_kernel/**`, `tests/**`
- `owning_specialist_type`: `scheduler_systems_specialist`
- `allowed_writes`: `core/orchestrator/**`, `core/native/**`, `saguaro/services/**`, `shared_kernel/**`, `tests/**`
- `telemetry_contract`: emit queue insertions, queue exits, denied admissions, and starvation risk.
- `required_evidence`: queue timing histograms, reservation decisions, and accuracy-preservation evidence under load.
- `rollback_criteria`: revert if queueing increases latency without measurable stability or quality benefit.
- `promotion_gate`: the system prefers explicit waiting to uncontrolled contention.
- `success_criteria`: concurrency behavior becomes auditable.
- Why this creates value: it directly addresses your desire to trade some speed for higher accuracy and stability.
- Why this creates moat: it treats agent and analysis concurrency as a first-class operating problem.
- Main risk or failure mode: over-serialization of useful work.
- Smallest credible first experiment: queue Saguaro graph builds behind active benchmark runs on constrained hosts.
- Confidence level: high

### Candidate 16. Native Capability Contract Ledger

- Class: Practical
- Suggested `phase_id`: `research`
- Core insight: native runtime capabilities should be versioned, certified, and diffed like any other contract.
- External inspiration or analogy: compatibility matrices and certified configuration ledgers.
- Why it fits Saguaro and Anvil specifically: `NativeQSGEngine._build_runtime_capabilities` already synthesizes rich capability data, but it is not yet a promoted contract.
- Exact places in this codebase where it could wire in: `core/native/native_qsg_engine.py`, `core/native/runtime_telemetry.py`, `audit/runner/suite_certification.py`, `audit/runner/suite_preflight.py`, `audit/contracts/**`.
- Existing primitives it can reuse: runtime capability banner, tuning contracts, host contracts, perf probes.
- New primitive, data flow, or subsystem needed: a capability ledger that records ISA path, selected kernel mode, perf status, affinity state, and verified feature availability.
- `repo_scope`: `core/native/**`, `audit/runner/**`, `audit/contracts/**`, `tests/**`
- `owning_specialist_type`: `runtime_governance_specialist`
- `allowed_writes`: `core/native/**`, `audit/runner/**`, `audit/contracts/**`, `tests/**`
- `telemetry_contract`: emit capability digest, change reason, degradation reason, and certification state.
- `required_evidence`: capability snapshot, prior snapshot comparison, and gate decision.
- `rollback_criteria`: revert if the ledger becomes redundant with other contracts and adds maintenance burden.
- `promotion_gate`: runtime capability changes become visible and explainable before silver promotion.
- `success_criteria`: operators can distinguish performance drift caused by model/code changes from capability drift caused by environment changes.
- Why this creates value: it reduces silent host and build drift.
- Why this creates moat: it makes native runtime state a governed artifact.
- Main risk or failure mode: ledger scope becomes too broad.
- Smallest credible first experiment: persist runtime capability digest beside tuning contracts for silver runs.
- Confidence level: high

### Candidate 17. Prompt-to-Performance Correlator

- Class: Moonshot
- Suggested `phase_id`: `feature_map`
- Core insight: prompt contract shape, compression guidance, and context pressure should be correlated with performance and stability outcomes.
- External inspiration or analogy: feature attribution systems and prompt-aware serving optimization.
- Why it fits Saguaro and Anvil specifically: `UnifiedChatLoop._get_master_system_prompt` already assembles prompt context and compression guidance, while silver captures quality and performance outcomes.
- Exact places in this codebase where it could wire in: `core/unified_chat_loop.py`, `core/model/chat_templates.py`, `audit/runner/benchmark_suite.py`, `core/telemetry/black_box.py`, `saguaro/services/platform.py`.
- Existing primitives it can reuse: prompt contract hashes, memory snapshot hashes, quality summaries, event exports.
- New primitive, data flow, or subsystem needed: prompt-shape fingerprints joined to benchmark outcomes.
- `repo_scope`: `core/unified_chat_loop.py`, `core/model/**`, `audit/runner/**`, `core/telemetry/**`, `saguaro/services/**`, `tests/**`
- `owning_specialist_type`: `prompt_systems_specialist`
- `allowed_writes`: `core/unified_chat_loop.py`, `core/model/**`, `audit/runner/**`, `core/telemetry/**`, `saguaro/services/**`, `tests/**`
- `telemetry_contract`: emit prompt fingerprint, compression mode, context pressure level, and resulting TTFT/quality envelope.
- `required_evidence`: prompt fingerprint table, run comparison report, and confounder notes.
- `rollback_criteria`: revert if correlations are noisy and not actionable.
- `promotion_gate`: at least one meaningful prompt/runtime relationship is repeatedly observable.
- `success_criteria`: prompt assembly becomes a measurable performance surface.
- Why this creates value: it ties the user-facing orchestration layer back to runtime truth.
- Why this creates moat: it unifies prompt engineering and systems engineering.
- Main risk or failure mode: spurious correlation from small datasets.
- Smallest credible first experiment: fingerprint silver canonical prompts and compare across three models and two compression settings.
- Confidence level: medium

### Candidate 18. Fault Tree Synthesizer

- Class: Moonshot
- Suggested `phase_id`: `analysis_upgrade`
- Core insight: failed or degraded benchmark runs should produce a machine-generated fault tree linking symptoms, likely causes, and disproving evidence.
- External inspiration or analogy: aerospace fault tree analysis and safety-case construction.
- Why it fits Saguaro and Anvil specifically: `EventStore._safety_case`, black box events, graph services, and benchmark summaries already provide many inputs.
- Exact places in this codebase where it could wire in: `shared_kernel/event_store.py`, `core/telemetry/black_box.py`, `audit/runner/benchmark_suite.py`, `saguaro/services/platform.py`, `saguaro/analysis/code_graph.py`.
- Existing primitives it can reuse: safety-case export, event links, artifact summaries, code graph relationships.
- New primitive, data flow, or subsystem needed: fault-tree synthesis rules with causal confidence.
- `repo_scope`: `shared_kernel/**`, `core/telemetry/**`, `audit/**`, `saguaro/services/**`, `saguaro/analysis/**`, `tests/**`
- `owning_specialist_type`: `failure_analysis_specialist`
- `allowed_writes`: `shared_kernel/**`, `core/telemetry/**`, `audit/**`, `saguaro/services/**`, `saguaro/analysis/**`, `tests/**`
- `telemetry_contract`: emit symptom nodes, cause hypotheses, disproving evidence, and residual uncertainty.
- `required_evidence`: linked artifacts, decision path, and confidence annotation.
- `rollback_criteria`: revert if generated trees are too speculative or noisy.
- `promotion_gate`: fault trees help explain at least one real failed run more quickly than current ad hoc inspection.
- `success_criteria`: post-run triage becomes mechanized and traceable.
- Why this creates value: it speeds debugging and improves governance evidence.
- Why this creates moat: it adds systems-failure reasoning on top of ordinary logging.
- Main risk or failure mode: low precision if evidence links are weak.
- Smallest credible first experiment: synthesize trees only for explicit silver preflight failures first.
- Confidence level: medium

### Candidate 19. Disparate Relationship Analyzer

- Class: Moonshot
- Suggested `phase_id`: `analysis_upgrade`
- Core insight: Saguaro should discover non-obvious relationships across benchmark outputs, graph changes, FFI edges, scheduler states, and quality regressions.
- External inspiration or analogy: graph intelligence, root-cause mining, and motorsport telemetry correlation.
- Why it fits Saguaro and Anvil specifically: the repo already has `GraphService`, `QueryService`, `MetricsService`, `CodeGraph`, and benchmark artifacts.
- Exact places in this codebase where it could wire in: `saguaro/services/platform.py`, `saguaro/analysis/code_graph.py`, `saguaro/query/benchmark.py`, `audit/runner/benchmark_suite.py`, `core/native/runtime_telemetry.py`.
- Existing primitives it can reuse: graph rows, metrics runs, benchmark scoring, code graph relationships.
- New primitive, data flow, or subsystem needed: relationship mining over code, runtime, and benchmark entities with explainable ranking.
- `repo_scope`: `saguaro/services/**`, `saguaro/analysis/**`, `saguaro/query/**`, `audit/**`, `core/native/**`, `tests/**`
- `owning_specialist_type`: `graph_intelligence_specialist`
- `allowed_writes`: `saguaro/services/**`, `saguaro/analysis/**`, `saguaro/query/**`, `audit/**`, `core/native/**`, `tests/**`
- `telemetry_contract`: emit relationship candidates, supporting evidence, confidence, and false-discovery notes.
- `required_evidence`: ranked relationship report, evidence links, and validation sample review.
- `rollback_criteria`: revert if output is too speculative to trust operationally.
- `promotion_gate`: analyzer surfaces at least one novel, validated relationship on historical runs.
- `success_criteria`: Saguaro can help answer "what changed elsewhere that explains this benchmark shift?"
- Why this creates value: it operationalizes the exact disparate-relationship idea you described.
- Why this creates moat: it combines code intelligence and benchmark intelligence in a reusable way.
- Main risk or failure mode: false patterns and operator overload.
- Smallest credible first experiment: correlate changed files, changed capability digests, and kernel hotspot deltas across recent runs.
- Confidence level: medium

### Candidate 20. Repair-Before-Run Gate

- Class: Practical
- Suggested `phase_id`: `development`
- Core insight: if graph, contract, or capability checks identify a likely correctable problem, the system should attempt targeted remediation before expensive benchmark execution.
- External inspiration or analogy: maintenance crews correcting launch constraints before window open.
- Why it fits Saguaro and Anvil specifically: `suite_preflight.py` already emits remediations, and Saguaro already provides verify/fix surfaces.
- Exact places in this codebase where it could wire in: `audit/runner/suite_preflight.py`, `audit/runner/suite_certification.py`, `saguaro/services/platform.py`, `saguaro/sentinel/**`, `core/aes/governance.py`.
- Existing primitives it can reuse: remediation payloads, contract bootstrap, Saguaro verify/fix, governance checks.
- New primitive, data flow, or subsystem needed: a preflight remediation execution contract with audit logging.
- `repo_scope`: `audit/runner/**`, `saguaro/services/**`, `saguaro/sentinel/**`, `core/aes/**`, `tests/**`
- `owning_specialist_type`: `runtime_governance_specialist`
- `allowed_writes`: `audit/runner/**`, `saguaro/services/**`, `saguaro/sentinel/**`, `core/aes/**`, `tests/**`
- `telemetry_contract`: emit proposed remediation, accepted remediation, rejected remediation, and post-repair verification result.
- `required_evidence`: before/after state, command trace, and verification record.
- `rollback_criteria`: revert if automated remediation becomes too invasive or unpredictable.
- `promotion_gate`: low-risk repairable failures are corrected and reverified automatically.
- `success_criteria`: avoidable benchmark failures are reduced.
- Why this creates value: it reduces wasted benchmark cycles.
- Why this creates moat: it operationalizes tool-assisted correction as part of runtime admission.
- Main risk or failure mode: unintended side effects from automated repair.
- Smallest credible first experiment: auto-bootstrap missing host or tuning contract under explicit silver policy.
- Confidence level: medium-high

### Candidate 21. Cross-Repo Differential Benchmark Judge

- Class: Moonshot
- Suggested `phase_id`: `deep_test_audit`
- Core insight: Saguaro should compare benchmark bundles across repositories or branches and explain semantic, runtime, and quality differences.
- External inspiration or analogy: regression triage across fleets and multi-program telemetry comparison.
- Why it fits Saguaro and Anvil specifically: Saguaro already has metrics and graph services, while the benchmark bundle proposal creates a portable import unit.
- Exact places in this codebase where it could wire in: `saguaro/services/platform.py`, `saguaro/query/benchmark.py`, `saguaro/analysis/code_graph.py`, `audit/schemas/**`, `audit/runner/benchmark_suite.py`.
- Existing primitives it can reuse: benchmark scoring, code graph, evidence bundles, metrics runs.
- New primitive, data flow, or subsystem needed: diff logic that joins benchmark deltas to code and capability deltas.
- `repo_scope`: `saguaro/services/**`, `saguaro/query/**`, `saguaro/analysis/**`, `audit/**`, `tests/**`
- `owning_specialist_type`: `platform_interop_specialist`
- `allowed_writes`: `saguaro/services/**`, `saguaro/query/**`, `saguaro/analysis/**`, `audit/**`, `tests/**`
- `telemetry_contract`: emit baseline bundle, candidate bundle, classified deltas, and confidence of explanation.
- `required_evidence`: import success, diff report, and validation examples.
- `rollback_criteria`: revert if the judge is too brittle across repos with weak schema adherence.
- `promotion_gate`: one Anvil run and one external run can be compared with a coherent explanation.
- `success_criteria`: performance/debugging intelligence becomes genuinely portable.
- Why this creates value: it matches your goal of tooling reusable across repositories and builds.
- Why this creates moat: portable, explainable benchmark diffing is rare.
- Main risk or failure mode: schema portability gaps.
- Smallest credible first experiment: compare two Anvil silver runs produced from different commits using the same host.
- Confidence level: medium

### Candidate 22. Model-Hardware Knowledge Bank

- Class: Moonshot
- Suggested `phase_id`: `feature_map`
- Core insight: observed tuning winners, capability snapshots, quality envelopes, and hotspot signatures should accumulate into a reusable knowledge base by model family and host fingerprint.
- External inspiration or analogy: race engineering setup books and fleet maintenance knowledge systems.
- Why it fits Saguaro and Anvil specifically: host fingerprints and tuning contracts already exist, and Saguaro already supports research and metrics persistence.
- Exact places in this codebase where it could wire in: `audit/runner/suite_certification.py`, `audit/contracts/**`, `core/native/native_qsg_engine.py`, `saguaro/services/platform.py`, `shared_kernel/event_store.py`.
- Existing primitives it can reuse: host fingerprints, tuning contracts, metrics runs, research entries, run exports.
- New primitive, data flow, or subsystem needed: a knowledge bank keyed by model digest, host fingerprint, and capability digest.
- `repo_scope`: `audit/contracts/**`, `audit/runner/**`, `core/native/**`, `saguaro/services/**`, `shared_kernel/**`, `tests/**`
- `owning_specialist_type`: `knowledge_systems_specialist`
- `allowed_writes`: `audit/contracts/**`, `audit/runner/**`, `core/native/**`, `saguaro/services/**`, `shared_kernel/**`, `tests/**`
- `telemetry_contract`: emit knowledge insertions, superseded entries, and confidence source.
- `required_evidence`: source run bundle, certification status, and compatibility checks.
- `rollback_criteria`: revert if the bank accumulates stale or low-quality entries that are hard to prune.
- `promotion_gate`: prior certified runs measurably reduce tuning and diagnosis time on repeated hosts.
- `success_criteria`: experience compounds instead of resetting every run.
- Why this creates value: it cuts repeated rediscovery cost.
- Why this creates moat: it turns local optimization work into a growing asset.
- Main risk or failure mode: stale knowledge reused in the wrong environment.
- Smallest credible first experiment: persist certified silver winners by model and host digest with supersession logic.
- Confidence level: medium-high

### Candidate 23. Semantic Performance Legislation

- Class: Moonshot
- Suggested `phase_id`: `convergence`
- Core insight: repeated benchmark and runtime lessons should be codified into Saguaro verification rules automatically where patterns are strong enough.
- External inspiration or analogy: legislative codification of repeated incident learnings.
- Why it fits Saguaro and Anvil specifically: the repo already has governance JSON, Saguaro legislation tooling, and verify engines.
- Exact places in this codebase where it could wire in: `standards/**`, `core/aes/governance.py`, `saguaro/sentinel/**`, `saguaro/services/platform.py`, `.saguaro.rules.draft`.
- Existing primitives it can reuse: governance engine, standards, verify service, legislation draft flow.
- New primitive, data flow, or subsystem needed: performance-and-audit-specific rule drafting from repeated failures or drift patterns.
- `repo_scope`: `standards/**`, `core/aes/**`, `saguaro/sentinel/**`, `saguaro/services/**`, `tests/**`
- `owning_specialist_type`: `governance_systems_specialist`
- `allowed_writes`: `standards/**`, `core/aes/**`, `saguaro/sentinel/**`, `saguaro/services/**`, `tests/**`
- `telemetry_contract`: emit proposed rules, accepted rules, rejected rules, and evidence basis.
- `required_evidence`: repeated failure pattern, supporting runs, and false-positive review.
- `rollback_criteria`: revert if legislation generates brittle or overfit rules.
- `promotion_gate`: at least one repeated benchmark lesson is converted into a stable, useful verification rule.
- `success_criteria`: learning from incidents becomes mechanized.
- Why this creates value: it prevents the same mistake class from recurring.
- Why this creates moat: it turns experience into governance, not just memory.
- Main risk or failure mode: premature codification.
- Smallest credible first experiment: draft rules for missing contract evidence or unsafe fallback patterns.
- Confidence level: medium

### Candidate 24. Silver Smoke Convergence Gate

- Class: Practical
- Suggested `phase_id`: `deep_test_audit`
- Core insight: silver must converge benchmark truth, runtime truth, graph truth, and governance truth into one promotion decision.
- External inspiration or analogy: integrated readiness review boards.
- Why it fits Saguaro and Anvil specifically: all ingredients already exist, but they are not yet fused into a single promotion gate.
- Exact places in this codebase where it could wire in: `audit/runner/benchmark_suite.py`, `audit/runner/suite_certification.py`, `audit/profiles/native_qsg_silver.yaml`, `saguaro/services/platform.py`, `core/aes/governance.py`, `shared_kernel/event_store.py`.
- Existing primitives it can reuse: summary payloads, certification states, tuning contracts, Saguaro metrics and verify services, event safety case.
- New primitive, data flow, or subsystem needed: a convergence manifest declaring pass, fail, hold, and rollback reasons across all major truth planes.
- `repo_scope`: `audit/runner/**`, `audit/profiles/**`, `saguaro/services/**`, `core/aes/**`, `shared_kernel/**`, `tests/**`
- `owning_specialist_type`: `benchmark_assurance_specialist`
- `allowed_writes`: `audit/runner/**`, `audit/profiles/**`, `saguaro/services/**`, `core/aes/**`, `shared_kernel/**`, `tests/**`
- `telemetry_contract`: emit convergence verdict, unmet gates, residual risk, and promotion recommendation.
- `required_evidence`: admission manifest, bundle completeness report, quality observatory report, hotspot prioritizer report, and safety-case export.
- `rollback_criteria`: revert if convergence becomes another opaque report instead of a concrete decision contract.
- `promotion_gate`: silver promotion becomes machine-checkable and evidence-complete.
- `success_criteria`: "silver ready" becomes a validated state with durable operator meaning.
- Why this creates value: it stages the benchmark audit properly before the silver smoke screen.
- Why this creates moat: it closes the loop from code to runtime to benchmark to governance.
- Main risk or failure mode: premature final gate before enough deep signals exist.
- Smallest credible first experiment: require an explicit convergence document for silver runs while allowing informational-only soft failures.
- Confidence level: high

## 6. Critical Pressure Test

- Elegant but likely wrong: the full scheduler-memory digital twin is elegant, but it may be premature until the truth fabric and reservation mechanics exist.
- Elegant but likely wrong: prompt-to-performance correlation can become seductive but noisy if prompt fingerprints are not joined to enough controlled runs.
- Elegant but likely wrong: semantic performance legislation is powerful only after repeated patterns are truly stable; doing it too early creates brittle rules.
- Ugly but strategically powerful: silver admission freeze is less glamorous than new optimizers, but it is the most direct way to prevent misleading benchmark claims.
- Ugly but strategically powerful: resource reservation and wait queues can make the system feel slower in some moments, but they directly support the user's requirement to preserve accuracy and stability over opportunistic concurrency.
- Ugly but strategically powerful: capability ledgers and boot-time tuning are operational plumbing, but they are exactly what high-standard systems rely on to avoid drift.
- Likely to fail because the repo is missing a key primitive: graph-FFI preflight will stall if native symbol and bridge fidelity are too shallow.
- Likely to fail because the repo is missing a key primitive: causal hotspot prioritization needs cleaner stage-local timing and counter joins than the current artifacts may yet provide.
- Likely to fail because the repo is missing a key primitive: cross-repo differential judging depends on a portable bundle schema first.
- Likely to fail because the repo is missing a key primitive: the mission control wall will underdeliver if broad and deep signals are not normalized first.
- Highest leverage correction to the current architecture: create one shared evidence model for benchmark, runtime, and graph relationships before adding more isolated features.
- Highest leverage correction to current thinking: treat Saguaro as a workload competing for capacity and as an analysis engine consuming that capacity, not as a free background capability.

## 7. Synthesis

- Strongest overall idea: Benchmark Truth Fabric
- Why: it unifies the benchmark harness, black box recorder, event store, and Saguaro platform into one center of gravity.
- Strongest novelty-plus-plausibility balance: Graph FFI Preflight Sentinel
- Why: the underlying graph and FFI primitives already exist, and the value of preventing failures before runtime is unusually high.
- Most feasible now: Silver Admission Freeze
- Why: the contract and preflight machinery is already present in `suite_preflight.py`, `suite_certification.py`, and the silver profile.
- Biggest long-term moat bet: Saguaro Benchmark Ingestor plus Disparate Relationship Analyzer
- Why: that pair converts this repo's benchmark output into a reusable cross-repo intelligence system.
- Most cleanly unifying idea: Memory Pressure Flight Controller plus Resource Reservation and Wait Queue
- Why: it resolves the current split between "benchmark the models" and "ignore what Saguaro and agents are consuming."
- Best first prototype: Runtime Tuning Boot Kernel
- Why: it has the clearest existing surfaces, immediately improves silver rigor, and supports later capacity and convergence work.
- Conviction ranking:
- `1.` Benchmark Truth Fabric
- `2.` Silver Admission Freeze
- `3.` Runtime Tuning Boot Kernel
- `4.` Memory Pressure Flight Controller
- `5.` Graph FFI Preflight Sentinel
- `6.` Saguaro Benchmark Ingestor
- `7.` Quality Stability Observatory
- `8.` Resource Reservation and Wait Queue
- `9.` Continuous Batch Governor
- `10.` Causal Hotspot Prioritizer

## 8. Implementation Program

### Phase 1

- `phase_id`: `intake`
- Phase title: Silver Baseline Freeze
- Objective: establish a hard, evidence-bearing silver admission state that prevents misleading runs.
- Dependencies: none
- Repo scope: `audit/runner/suite_preflight.py`, `audit/runner/suite_certification.py`, `audit/profiles/native_qsg_silver.yaml`, `audit/runner/benchmark_suite.py`, `core/aes/governance.py`
- Owning specialist type: `runtime_governance_specialist`
- Allowed writes: `audit/runner/**`, `audit/profiles/**`, `core/aes/**`, `tests/**`
- Telemetry contract: every silver attempt shall emit an admission manifest with host, tuning, capability, perf, Saguaro, and fallback status.
- Required evidence: admission manifest JSON, failed prerequisite list, remediation guidance, and governance decision trace.
- Rollback criteria: revert if admission checks are non-deterministic or reject already-certified configurations without clear evidence.
- Promotion gate: no silver run can start measured work without a persisted admission manifest.
- Success criteria: silver becomes a machine-checkable state with explicit pass, fail, or hold.
- Exact wiring points: `audit/runner/suite_preflight.py`, `audit/runner/suite_certification.py`, `audit/runner/benchmark_suite.py`, `audit/profiles/native_qsg_silver.yaml`, `core/aes/governance.py`
- Deliverables: silver admission manifest, explicit failure taxonomy, remediation hooks, updated profile contract.
- Tests: `tests/audit/test_benchmark_suite.py`, `tests/audit/test_suite_certification.py`, `tests/test_native_capability_governance.py`
- Verification commands: `pytest tests/audit/test_benchmark_suite.py tests/audit/test_suite_certification.py tests/test_native_capability_governance.py`
- Verification commands: `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: silver runs fail closed on missing or stale contracts and report actionable reasons.

### Phase 2

- `phase_id`: `research`
- Phase title: Benchmark Truth Fabric
- Objective: create a normalized evidence bundle that joins benchmark artifacts, runtime telemetry, event traces, and governance decisions.
- Dependencies: `intake`
- Repo scope: `audit/runner/benchmark_suite.py`, `audit/runner/native_benchmark_runner.py`, `audit/store/suite_layout.py`, `audit/schemas/**`, `core/telemetry/black_box.py`, `shared_kernel/event_store.py`, `saguaro/services/platform.py`
- Owning specialist type: `benchmark_assurance_specialist`
- Allowed writes: `audit/**`, `core/telemetry/**`, `shared_kernel/**`, `saguaro/services/**`, `tests/**`
- Telemetry contract: every lane, artifact, and decision shall share stable trace and bundle IDs.
- Required evidence: bundle schema, artifact index, join coverage report, run export example, and compatibility report.
- Rollback criteria: revert if bundle overhead is excessive or breaks current consumers.
- Promotion gate: benchmark truth becomes queryable and replayable as one bundle.
- Success criteria: a silver run can be understood from one evidence object without manual cross-file stitching.
- Exact wiring points: `audit/runner/benchmark_suite.py`, `audit/runner/native_benchmark_runner.py`, `audit/schemas/summary.schema.json`, `core/telemetry/black_box.py`, `shared_kernel/event_store.py`, `saguaro/services/platform.py`
- Deliverables: portable evidence bundle, stable join IDs, bundle writer, bundle reader.
- Tests: `tests/audit/test_benchmark_suite.py`, `tests/test_black_box_recorder.py`, `tests/test_saguaro_platform_foundation.py`
- Verification commands: `pytest tests/audit/test_benchmark_suite.py tests/test_black_box_recorder.py tests/test_saguaro_platform_foundation.py`
- Verification commands: `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: bundle import and replay work on at least one silver run end to end.

### Phase 3

- `phase_id`: `eid`
- Phase title: Runtime Tuning and Capacity Envelope
- Objective: make tuning, capability certification, and resource reservation explicit at startup and before heavy work admission.
- Dependencies: `intake`
- Repo scope: `audit/runner/suite_certification.py`, `core/native/native_qsg_engine.py`, `core/orchestrator/scheduler.py`, `core/memory/fabric/store.py`, `core/memory/latent_memory.py`, `saguaro/services/platform.py`
- Owning specialist type: `runtime_performance_specialist`
- Allowed writes: `audit/runner/**`, `core/native/**`, `core/orchestrator/**`, `core/memory/**`, `saguaro/services/**`, `tests/**`
- Telemetry contract: every tune, reservation, wait, denial, and capability change shall emit a traceable event.
- Required evidence: tuning status table, capability ledger, reservation log, queue timing, and memory watermark report.
- Rollback criteria: revert if startup becomes materially worse without reducing instability or contention.
- Promotion gate: the system can explain why a model or analysis job is ready, waiting, or denied.
- Success criteria: startup and scheduling no longer rely on silent heuristics alone.
- Exact wiring points: `audit/runner/suite_certification.py`, `core/native/native_qsg_engine.py`, `core/native/parallel_generation.py`, `core/orchestrator/scheduler.py`, `core/memory/fabric/store.py`, `saguaro/services/platform.py`
- Deliverables: tune-on-start handshake, capability ledger, reservation broker, wait-state semantics.
- Tests: `tests/test_native_qsg_engine.py`, `tests/test_native_parallel_generation_engine.py`, `tests/test_memory_governance.py`, `tests/test_qsg_continuous_engine.py`
- Verification commands: `pytest tests/test_native_qsg_engine.py tests/test_native_parallel_generation_engine.py tests/test_memory_governance.py tests/test_qsg_continuous_engine.py`
- Verification commands: `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: supported models and heavy Saguaro jobs enter an explicit certified or queued state.

### Phase 4

- `phase_id`: `feature_map`
- Phase title: Saguaro Benchmark Intelligence Plane
- Objective: ingest benchmark bundles into Saguaro and expose graph, metric, and operator views over them.
- Dependencies: `research`
- Repo scope: `saguaro/services/platform.py`, `saguaro/query/benchmark.py`, `saguaro/analysis/code_graph.py`, `saguaro/sentinel/engines/graph.py`, `audit/schemas/**`, `audit/runner/benchmark_suite.py`
- Owning specialist type: `graph_intelligence_specialist`
- Allowed writes: `saguaro/services/**`, `saguaro/query/**`, `saguaro/analysis/**`, `saguaro/sentinel/**`, `audit/**`, `tests/**`
- Telemetry contract: ingest, relationship creation, unresolved references, and query response confidence shall be recorded.
- Required evidence: imported bundle stats, relationship graph, benchmark query examples, and graph-preflight reports.
- Rollback criteria: revert if ingest quality is poor or false relationships dominate.
- Promotion gate: Saguaro can answer questions about a benchmark run and preflight a likely native-boundary issue.
- Success criteria: benchmark audit becomes a first-class Saguaro workload and intelligence source.
- Exact wiring points: `saguaro/services/platform.py`, `saguaro/query/benchmark.py`, `saguaro/analysis/code_graph.py`, `saguaro/sentinel/engines/graph.py`, `audit/runner/benchmark_suite.py`, `audit/schemas/summary.schema.json`
- Deliverables: audit ingestor, benchmark query adapters, graph-FFI preflight report, app-service views.
- Tests: `tests/test_saguaro_platform_foundation.py`, `tests/test_saguaro_code_graph.py`, `tests/test_saguaro_graph_resolution.py`, `tests/test_saguaro_query_accuracy.py`
- Verification commands: `pytest tests/test_saguaro_platform_foundation.py tests/test_saguaro_code_graph.py tests/test_saguaro_graph_resolution.py tests/test_saguaro_query_accuracy.py`
- Verification commands: `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: Saguaro can ingest and query one silver run and flag one preflight boundary concern.

### Phase 5

- `phase_id`: `development`
- Phase title: Native Batch Governor and Reservation Queue
- Objective: govern continuous batching, model admission, and analysis contention through measured latency-quality-memory envelopes.
- Dependencies: `eid`, `feature_map`
- Repo scope: `core/native/parallel_generation.py`, `core/native/native_qsg_engine.py`, `core/native/runtime_telemetry.py`, `core/orchestrator/scheduler.py`, `audit/runner/benchmark_suite.py`, `audit/profiles/native_qsg_silver.yaml`
- Owning specialist type: `scheduler_systems_specialist`
- Allowed writes: `core/native/**`, `core/orchestrator/**`, `audit/runner/**`, `audit/profiles/**`, `tests/**`
- Telemetry contract: emit admission delays, governor decisions, TTFT/TPS percentiles, memory headroom, and starvation risk.
- Required evidence: static-policy baseline, adaptive-policy run set, memory comparisons, and quality comparisons.
- Rollback criteria: revert if adaptive control introduces instability or unacceptable latency spikes.
- Promotion gate: governed scheduling outperforms or matches the static policy while preserving quality and memory headroom.
- Success criteria: continuous batching becomes a controlled, auditable system rather than a fixed configuration list.
- Exact wiring points: `core/native/parallel_generation.py`, `core/native/native_qsg_engine.py`, `core/native/runtime_telemetry.py`, `core/orchestrator/scheduler.py`, `audit/runner/benchmark_suite.py`, `audit/profiles/native_qsg_silver.yaml`
- Deliverables: adaptive governor, shared reservation queue, capacity envelope policy, benchmark comparison artifacts.
- Tests: `tests/test_native_parallel_generation_engine.py`, `tests/test_qsg_continuous_engine.py`, `tests/test_runtime_telemetry.py`, `tests/test_continuous_qsg_benchmark.py`
- Verification commands: `pytest tests/test_native_parallel_generation_engine.py tests/test_qsg_continuous_engine.py tests/test_runtime_telemetry.py tests/test_continuous_qsg_benchmark.py`
- Verification commands: `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: the system can justify its current concurrency and queueing behavior with evidence.

### Phase 6

- `phase_id`: `analysis_upgrade`
- Phase title: Quality Stability and Causal Hotspot Observatory
- Objective: add repeated-sampling stability analysis, calibration gates, and counterfactual hotspot prioritization to the benchmark program.
- Dependencies: `research`, `development`
- Repo scope: `audit/eval/native_logits.py`, `audit/runner/benchmark_suite.py`, `benchmarks/native_kernel_microbench.py`, `core/native/runtime_telemetry.py`, `saguaro/services/platform.py`
- Owning specialist type: `evaluation_science_specialist`
- Allowed writes: `audit/eval/**`, `audit/runner/**`, `benchmarks/**`, `core/native/**`, `saguaro/services/**`, `tests/**`
- Telemetry contract: emit stability samples, coherence drift, calibration errors, hotspot counterfactuals, and residual uncertainty.
- Required evidence: repeated-sample reports, calibration curves, hotspot ranking reports, and validation examples.
- Rollback criteria: revert if analysis cost outweighs diagnostic value or signals are too noisy.
- Promotion gate: silver can show not only what got faster, but whether it stayed stable and which optimization matters most.
- Success criteria: performance tuning and quality assurance operate under one observatory.
- Exact wiring points: `audit/eval/native_logits.py`, `audit/runner/benchmark_suite.py`, `benchmarks/native_kernel_microbench.py`, `core/native/runtime_telemetry.py`, `saguaro/services/platform.py`
- Deliverables: stability lane, calibration lane, hotspot prioritizer, observatory summaries.
- Tests: `tests/audit/test_benchmark_suite.py`, `tests/test_runtime_telemetry.py`, `tests/test_phase8_kernels.py`, `tests/test_saguaro_query_accuracy.py`
- Verification commands: `pytest tests/audit/test_benchmark_suite.py tests/test_runtime_telemetry.py tests/test_phase8_kernels.py tests/test_saguaro_query_accuracy.py`
- Verification commands: `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: silver reports can prove whether a change improved the right things for the right reasons.

### Phase 7

- `phase_id`: `deep_test_audit`
- Phase title: Silver Smoke Certification and Convergence
- Objective: converge admission, benchmark truth, runtime truth, graph truth, and governance truth into one final silver promotion decision.
- Dependencies: `intake`, `research`, `eid`, `feature_map`, `development`, `analysis_upgrade`
- Repo scope: `audit/runner/benchmark_suite.py`, `audit/runner/suite_certification.py`, `audit/profiles/native_qsg_silver.yaml`, `saguaro/services/platform.py`, `shared_kernel/event_store.py`, `core/aes/governance.py`
- Owning specialist type: `benchmark_assurance_specialist`
- Allowed writes: `audit/**`, `saguaro/services/**`, `shared_kernel/**`, `core/aes/**`, `tests/**`
- Telemetry contract: emit one convergence verdict with unmet gates, residual risk, rollback recommendation, and safety-case linkage.
- Required evidence: admission manifest, evidence bundle completeness report, capacity report, quality observatory report, graph-preflight report, and safety-case export.
- Rollback criteria: revert if convergence is opaque, non-actionable, or routinely bypassed.
- Promotion gate: silver readiness becomes fully evidence-complete and machine-auditable.
- Success criteria: the silver smoke screen is staged properly and cannot overstate readiness.
- Exact wiring points: `audit/runner/benchmark_suite.py`, `audit/runner/suite_certification.py`, `audit/profiles/native_qsg_silver.yaml`, `saguaro/services/platform.py`, `shared_kernel/event_store.py`, `core/aes/governance.py`
- Deliverables: silver convergence manifest, final certification report, rollback reasons, and roadmap-validation-compatible phase evidence.
- Tests: `tests/audit/test_benchmark_suite.py`, `tests/test_phase6_continuous_compliance.py`, `tests/test_aes_governance_reviews.py`, `tests/test_campaign_speculation_risk_and_governance.py`
- Verification commands: `pytest tests/audit/test_benchmark_suite.py tests/test_phase6_continuous_compliance.py tests/test_aes_governance_reviews.py tests/test_campaign_speculation_risk_and_governance.py`
- Verification commands: `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Verification commands: `./venv/bin/saguaro roadmap validate --path anvil_roadmap.md --format json`
- Exit criteria: silver promotion is coupled to concrete evidence and a clear residual-risk story.

## 9. Implementation Contract

- The system shall persist a unified benchmark evidence bundle through `audit/runner/benchmark_suite.py`, `audit/runner/native_benchmark_runner.py`, `audit/store/suite_layout.py`, `audit/schemas/summary.schema.json`, `core/telemetry/black_box.py`, and `shared_kernel/event_store.py`, tested by `tests/audit/test_benchmark_suite.py`, `tests/test_black_box_recorder.py`, and `tests/test_saguaro_platform_foundation.py`, and verified with `pytest tests/audit/test_benchmark_suite.py tests/test_black_box_recorder.py tests/test_saguaro_platform_foundation.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall fail silver admission on missing or stale host, tuning, perf, Saguaro, or fallback evidence through `audit/runner/suite_preflight.py`, `audit/runner/suite_certification.py`, `audit/profiles/native_qsg_silver.yaml`, and `audit/runner/benchmark_suite.py`, tested by `tests/audit/test_benchmark_suite.py`, `tests/audit/test_suite_certification.py`, and `tests/test_native_capability_governance.py`, and verified with `pytest tests/audit/test_benchmark_suite.py tests/audit/test_suite_certification.py tests/test_native_capability_governance.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall implement boot-time tuning state detection and certified runtime tuning reuse through `audit/runner/suite_certification.py`, `core/native/native_qsg_engine.py`, and `cli/repl.py`, tested by `tests/audit/test_suite_certification.py` and `tests/test_native_qsg_engine.py`, and verified with `pytest tests/audit/test_suite_certification.py tests/test_native_qsg_engine.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall persist a native capability ledger that records ISA path, selected kernel mode, perf availability, affinity state, and capability digest through `core/native/native_qsg_engine.py`, `core/native/runtime_telemetry.py`, `audit/runner/suite_preflight.py`, and `audit/runner/suite_certification.py`, tested by `tests/test_runtime_telemetry.py`, `tests/test_native_qsg_engine.py`, and `tests/test_native_capability_governance.py`, and verified with `pytest tests/test_runtime_telemetry.py tests/test_native_qsg_engine.py tests/test_native_capability_governance.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall implement a shared reservation and wait-state broker for model execution, Saguaro-heavy analysis, and subagent work through `core/native/parallel_generation.py`, `core/native/native_qsg_engine.py`, `core/orchestrator/scheduler.py`, `core/memory/fabric/store.py`, and `saguaro/services/platform.py`, tested by `tests/test_native_parallel_generation_engine.py`, `tests/test_qsg_continuous_engine.py`, and `tests/test_memory_governance.py`, and verified with `pytest tests/test_native_parallel_generation_engine.py tests/test_qsg_continuous_engine.py tests/test_memory_governance.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall govern continuous batching with measured TTFT, decode throughput, memory headroom, and queueing evidence through `core/native/parallel_generation.py`, `core/native/runtime_telemetry.py`, `core/native/native_qsg_engine.py`, `audit/runner/benchmark_suite.py`, and `audit/profiles/native_qsg_silver.yaml`, tested by `tests/test_native_parallel_generation_engine.py`, `tests/test_qsg_continuous_engine.py`, `tests/test_runtime_telemetry.py`, and `tests/test_continuous_qsg_benchmark.py`, and verified with `pytest tests/test_native_parallel_generation_engine.py tests/test_qsg_continuous_engine.py tests/test_runtime_telemetry.py tests/test_continuous_qsg_benchmark.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall expose repeated-sampling stability, coherence drift, perplexity context metadata, and calibration error as first-class benchmark outputs through `audit/eval/native_logits.py`, `audit/runner/benchmark_suite.py`, `audit/schemas/summary.schema.json`, and `benchmarks/corpora/coherence_rubric.jsonl`, tested by `tests/audit/test_benchmark_suite.py` and `tests/test_runtime_telemetry.py`, and verified with `pytest tests/audit/test_benchmark_suite.py tests/test_runtime_telemetry.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall ingest benchmark evidence bundles into Saguaro metrics, evidence, and query surfaces through `saguaro/services/platform.py`, `saguaro/query/benchmark.py`, `saguaro/analysis/code_graph.py`, and `audit/runner/benchmark_suite.py`, tested by `tests/test_saguaro_platform_foundation.py`, `tests/test_saguaro_query_accuracy.py`, and `tests/test_saguaro_code_graph.py`, and verified with `pytest tests/test_saguaro_platform_foundation.py tests/test_saguaro_query_accuracy.py tests/test_saguaro_code_graph.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall implement graph- and FFI-aware preflight diagnostics for native-boundary risk through `saguaro/analysis/code_graph.py`, `saguaro/sentinel/engines/graph.py`, `saguaro/services/platform.py`, and `audit/runner/suite_preflight.py`, tested by `tests/test_saguaro_code_graph.py`, `tests/test_saguaro_graph_resolution.py`, and `tests/test_native_split_build_graph.py`, and verified with `pytest tests/test_saguaro_code_graph.py tests/test_saguaro_graph_resolution.py tests/test_native_split_build_graph.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall export broad operator signals and deep specialist signals with shared trace identifiers through `core/campaign/telemetry.py`, `core/telemetry/black_box.py`, `shared_kernel/event_store.py`, `core/native/runtime_telemetry.py`, and `saguaro/services/platform.py`, tested by `tests/test_runtime_telemetry.py`, `tests/test_black_box_recorder.py`, and `tests/test_campaign_control_kernel.py`, and verified with `pytest tests/test_runtime_telemetry.py tests/test_black_box_recorder.py tests/test_campaign_control_kernel.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall generate a convergence verdict for silver promotion that joins admission status, evidence completeness, runtime capacity, graph diagnostics, quality observatory outputs, and governance review through `audit/runner/benchmark_suite.py`, `audit/runner/suite_certification.py`, `core/aes/governance.py`, `shared_kernel/event_store.py`, and `saguaro/services/platform.py`, tested by `tests/audit/test_benchmark_suite.py`, `tests/test_phase6_continuous_compliance.py`, `tests/test_aes_governance_reviews.py`, and `tests/test_campaign_speculation_risk_and_governance.py`, and verified with `pytest tests/audit/test_benchmark_suite.py tests/test_phase6_continuous_compliance.py tests/test_aes_governance_reviews.py tests/test_campaign_speculation_risk_and_governance.py`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`, and `./venv/bin/saguaro roadmap validate --path anvil_roadmap.md --format json`.
- The system shall retain roadmap validator traceability by keeping promoted phase deliverables, tests, and verification references aligned with `anvil_roadmap.md`, `audit/runner/benchmark_suite.py`, `saguaro/services/platform.py`, and `audit/runner/suite_certification.py`, tested by `tests/audit/test_benchmark_suite.py` and `tests/test_saguaro_platform_foundation.py`, and verified with `pytest tests/audit/test_benchmark_suite.py tests/test_saguaro_platform_foundation.py` and `./venv/bin/saguaro roadmap validate --path anvil_roadmap.md --format json`.
