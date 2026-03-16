# Saguaro CPU Scan And Math Parse Verification + CPU Optimization Roadmap

## 1. First-Principles Framing
At its core, this subsystem is not a benchmark runner. It is a static-to-runtime translation layer that tries to turn source-level mathematical structure into CPU execution advice. In repo terms, `saguaro/math/engine.py` extracts equations and complexity, `saguaro/math/pipeline.py` enriches them with loop/access/provenance structure, `saguaro/cpu/model.py` converts that IR into hotspot triage, and `core/native/cpu_math_runtime.cpp` provides the native scoring kernel that makes the CPU scan more than a Python heuristic.

What it appears to be trying to become is a repo-coupled CPU performance copilot for Anvil's native inference stack: a system that can inspect `core/native`, predict which kernels matter on real CPUs, explain why they matter, and steer engineering decisions for the exact production model families the runtime supports today: `granite4:tiny-h`, `qwen3.5:4b`, and `qwen3.5:9b` through `core/model/chat_templates.py`, `core/model/model_profile.py`, `core/qsg/ollama_adapter.py`, `core/ollama_client.py`, `core/native/native_ops.py`, and `core/native/CMakeLists.txt`.

The engineering standard that matters is harsher than "passes tests":
- The math parser shall be semantically precise enough that a hotspot really corresponds to executed math, not scaffolding.
- The CPU scan shall be runtime-correlated enough that a priority score predicts actual optimization yield.
- The evidence chain shall close from source text to build target to backend module to runtime telemetry.
- The system shall respect the asymmetry between the three supported models: Granite tiny is hybrid-Mamba/MoE shaped and latency sensitive in different ways than Qwen 3.5 dense/hybrid variants.
- The standard is NASA/F1, not notebook-grade ML: bounded claims, explicit error budgets, replayable evidence, and clear rollback rules.

Current first-principles verdict: the subsystem appears operational, but not yet "completely accurate" in the epistemic sense. Live commands and tests show it works. The remaining problem is proof quality.

## 2. External Research Scan
### 2.1 Inference acceleration and CPU serving
- [Williams, Waterman, Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures"](https://cacm.acm.org/research/roofline-an-insightful-visual-performance-model-for-multicore-architectures/). Why it matters: `saguaro/cpu/roofline.py` already encodes a miniature roofline model. The repo should mature that into a runtime-backed hierarchical roofline instead of a one-shot heuristic.
- [Intel Extension for PyTorch LLM optimization on Intel CPUs](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/llm/int4_weight_only_quantization.html). Why it matters: official Intel guidance keeps converging on low-bit weight-only quantization, AMX/AVX512-aware kernels, and workload-specific dispatch. That aligns directly with `core/native/amx_kernels.cpp`, `core/native/quantized_matmul.cpp`, and the split backend model in `core/native/CMakeLists.txt`.
- [ONNX Runtime GenAI model optimization for CPU](https://onnxruntime.ai/docs/genai/howto/optimize-model.html). Why it matters: the mainstream CPU stack is moving toward graph-level operator fusion plus quantization plus provider-specific dispatch. Anvil already has pieces of this in native kernels and runtime telemetry but lacks a closed optimization loop.
- [Glinthawk repository](https://github.com/microsoft/glinthawk). Why it matters: Glinthawk treats inference as a control problem over memory hierarchy, routing, and throughput, not just kernel speed. That is the right analogy for extending `core/native/runtime_telemetry.py` into a CPU control plane.
- [KTransformers repository](https://github.com/kvcache-ai/ktransformers). Why it matters: heterogeneous serving, aggressive KV/cache strategy, and architecture-aware offload prove that practical moonshots often come from bold systems plumbing rather than one more kernel micro-optimization.

### 2.2 Speculative decoding and dynamic work allocation
- [Xia et al., "Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding"](https://arxiv.org/abs/2401.07851). Why it matters: Anvil already has strict speculative hooks in `core/qsg/ollama_adapter.py` and telemetry fields for speculative execution in `core/native/runtime_telemetry.py`; the missing piece is model-specific policy derived from hotspot and complexity evidence.
- [Santilli et al., "SPIN: Fast and Scalable LLM Inference with Speculative Parallelism"](https://arxiv.org/abs/2405.19325). Why it matters: SPIN's framing is useful for `qwen3.5` optimization because accepted-length economics matter more than raw draft speed.
- [Liu et al., "HeteroSpec"](https://arxiv.org/abs/2505.13254). Why it matters: context-aware speculative depth is a strong analogy for context-aware CPU complexity budgeting inside `core/native/qsg_parallel_kernels.cpp` and `core/native/parallel_generation.py`.
- [vLLM hybrid KV cache manager design](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/). Why it matters: hybrid attention + Mamba cache management is still awkward even in leading serving systems. Granite optimization in Anvil should assume hybrid cache policy is a first-class problem, not an implementation detail.
- [vLLM Mamba/Qwen3Next docs showing prefix-caching constraints](https://docs.vllm.ai/en/stable/api/vllm/model_executor/models/mamba.html), [Qwen3Next MTP notes](https://docs.vllm.ai/en/stable/api/vllm/model_executor/models/qwen3_next_mtp.html). Why it matters: external systems still disable or special-case prefix caching for Mamba and Qwen-next style paths. That is a warning against blindly porting transformer-centric optimizations to Granite/Qwen hybrids.

### 2.3 Parser correctness and corpus discipline
- [Tree-sitter introduction](https://tree-sitter.github.io/tree-sitter/index.html). Why it matters: tree-sitter's value proposition is robust parsing under syntax error and incremental updates. Saguaro should borrow the testing discipline and resilience model even if it does not replace its parser stack wholesale.
- [Tree-sitter parse CLI](https://tree-sitter.github.io/tree-sitter/cli/parse.html). Why it matters: parser correctness should be measured on corpora with parse-error accounting, not only unit tests.
- [Tree-sitter grammar testing guidance](https://tree-sitter.github.io/tree-sitter/creating-parsers/5-writing-tests.html). Why it matters: Saguaro's math parser needs a corpus-style golden suite for edge constructs, not just a handful of tactical tests.
- [tree-sitter-markdown inaccuracies discussion](https://github.com/tree-sitter-grammars/tree-sitter-markdown/issues/10). Why it matters: mixed markdown/code/math surfaces are inherently messy. That supports a differential-parser strategy for `saguaro/math/engine.py` rather than assuming one extraction path is authoritative.

### 2.4 Model-specific external grounding
- [IBM Granite docs](https://www.ibm.com/granite/docs/models/granite/). Why it matters: Granite is positioned as an enterprise-oriented model family, which strengthens the case for proof-carrying optimization rather than benchmark theater.
- [Granite-4.0-Tiny-Preview model card](https://huggingface.co/ibm-granite/granite-4.0-tiny-preview). Why it matters: IBM describes this model as a 7B fine-grained hybrid MoE instruct model for long-context work. Inference implication: optimization should assume hybrid behavior and active-parameter asymmetry.
- [Granite-4.0-H-Tiny model card](https://huggingface.co/ibm-granite/granite-4.0-h-tiny). Why it matters: IBM explicitly documents a Mamba2 + attention + MoE architecture and 128K context family behavior. Inference: Granite-family gains will come from hybrid-cache and memory-motion control as much as from raw GEMM speed. This is external inspiration; the repo itself currently targets `granite4:tiny-h` through native backends.
- [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B). Why it matters: Qwen exposes long context and multi-token prediction adjacent design choices; low-latency draft/verify policy is therefore more relevant than naive dense-kernel speed alone.
- [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B). Why it matters: the 9B class is large enough that memory traffic and KV behavior dominate many CPU scenarios; the runtime should treat it as a different operating regime from 4B.

### 2.5 Synthesis from external work
Inference from sources: the strongest external pattern is not "add one more optimization." It is "close the loop between structural analysis, runtime evidence, and model-family policy." That is exactly where this repo is promising and currently underpowered.

## 3. Repo Grounding Summary
### 3.1 Commands and evidence executed
- `source venv/bin/activate && ./venv/bin/saguaro health`
- `source venv/bin/activate && ./venv/bin/saguaro entrypoints`
- `source venv/bin/activate && ./venv/bin/saguaro build-graph`
- `source venv/bin/activate && ./venv/bin/saguaro query "cpu scan math parse runtime telemetry verification" --k 8`
- `source venv/bin/activate && ./venv/bin/saguaro query "prompt assembly orchestration planning loop unified chat memory telemetry governance native boundary" --k 10`
- `source venv/bin/activate && ./venv/bin/saguaro impact --path core/native/cpu_math_runtime.cpp`
- `source venv/bin/activate && ./venv/bin/saguaro agent skeleton ...` on `core/native/cpu_math_runtime.cpp`, `saguaro/math/engine.py`, `saguaro/math/ir.py`, `saguaro/math/pipeline.py`, `saguaro/math/languages.py`, `saguaro/cpu/report.py`, `saguaro/cpu/topology.py`, `saguaro/cpu/roofline.py`, `saguaro/cpu/vectorization.py`, `core/unified_chat_loop.py`, `core/prompts/prompt_manager.py`, `core/memory/fabric/store.py`, `core/native/runtime_telemetry.py`, `domains/verification/auto_verifier.py`, `saguaro/roadmap/validator.py`, and key tests.
- `source venv/bin/activate && ./venv/bin/saguaro math parse --path core/native/cpu_math_runtime.cpp --format json`
- `source venv/bin/activate && ./venv/bin/saguaro cpu scan --path core/native --arch x86_64-avx2 --format json`
- `source venv/bin/activate && pytest tests/test_saguaro_math.py tests/test_saguaro_math_precision.py tests/test_saguaro_math_pipeline.py tests/test_saguaro_cpu_scan.py tests/test_saguaro_cpu_topology.py tests/test_saguaro_cpu_twin.py tests/test_saguaro_roadmap_validator.py -q`

### 3.2 Live findings
- `saguaro health` reported repo-wide parser success coverage of 92.2%, AST coverage 88.2%, graph coverage 8.2%, missing graph edge classes `cfg`, `dfg`, `call`, `ffi_bridge`, and runtime symbol coverage 0.0. Repo state is healthy enough to analyze, but not healthy enough to claim closed-form end-to-end reasoning.
- The report file says verify passed and repo-wide math parse completed successfully, but its confidence posture is guarded because `parser_coverage_percent=10.0`. Inference: there is a metric mismatch or stale/narrow verifier metric, and that mismatch itself is roadmap-worthy.
- `saguaro math parse --path core/native/cpu_math_runtime.cpp --format json` returned sane records for real assignments in `core/native/cpu_math_runtime.cpp`; this is strong evidence that the false positives described in the report were materially reduced.
- `saguaro cpu scan --path core/native --arch x86_64-avx2 --format json` returned `status=ok`, `files_scanned=82`, `math_record_count=4615`, `hotspot_count=20`, and native-engine reports for hotspots in `core/native/qsg_parallel_kernels.cpp`. This is operational proof that the CPU scan is wired through the native runtime.
- Focused tests passed: `25 passed in 2.20s`.

### 3.3 Real modules inspected
- Math parse core: `saguaro/math/engine.py`, `saguaro/math/pipeline.py`, `saguaro/math/ir.py`, `saguaro/math/languages.py`
- CPU scan core: `saguaro/cpu/model.py`, `saguaro/cpu/topology.py`, `saguaro/cpu/roofline.py`, `saguaro/cpu/vectorization.py`, `saguaro/cpu/cache.py`, `saguaro/cpu/register_pressure.py`, `saguaro/cpu/report.py`
- Native scoring boundary: `core/native/cpu_math_runtime.cpp`, `core/native/cpu_math_runtime.h`, `core/native/runtime_telemetry.py`, `core/native/CMakeLists.txt`
- Model/runtime surfaces: `core/qsg/ollama_adapter.py`, `core/ollama_client.py`, `core/model/chat_templates.py`, `core/model/model_profile.py`, `core/native/native_ops.py`
- Governance and loops: `core/unified_chat_loop.py`, `core/prompts/prompt_manager.py`, `domains/verification/auto_verifier.py`, `saguaro/roadmap/validator.py`
- Tests already present: `tests/test_saguaro_math.py`, `tests/test_saguaro_math_precision.py`, `tests/test_saguaro_math_pipeline.py`, `tests/test_saguaro_cpu_scan.py`, `tests/test_saguaro_cpu_topology.py`, `tests/test_saguaro_cpu_twin.py`, `tests/test_qsg_adapter_unification.py`, `tests/test_runtime_telemetry.py`, `tests/test_native_ops_split.py`, `tests/test_saguaro_roadmap_validator.py`

### 3.4 Strongest existing primitives
- `MathEngine._extract_code_equations` plus `MathPipeline.enrich_records` already produce a useful IR with loop/access/provenance structure.
- `CPUScanner.scan` already gates on `source_kind == code_expression` and `provenance.execution_domain == native_kernel`, which is the right architectural center of gravity.
- `core/native/cpu_math_runtime.cpp` provides a native scoring ABI instead of duplicating heuristics in Python.
- `core/native/runtime_telemetry.py` is far richer than the current CPU scan consumes; it already tracks TTFT, decode timing, cache behavior, topology, backend selection, and speculative fields.
- The model runtime is explicitly pinned to the three supported model variants through strict prompt contracts and backend selection logic.

### 3.5 Underexploited or thin areas
- The parser/scan path does not yet close to runtime witnesses. `CPUScanner` uses static IR plus the native surrogate, but no real perf-counter or telemetry feedback loop.
- `saguaro impact --path core/native/cpu_math_runtime.cpp` fell back to `analysis_mode=heuristic_scan`, with no impacted tests or interfaces. This is a warning that change impact on the native boundary is under-modeled.
- `saguaro query` results on CPU/math concepts were low quality. Discovery fidelity is weaker than the system ambition.
- The health graph lacks `cfg`, `dfg`, `call`, and `ffi_bridge` edges, while runtime symbol coverage is zero. That weakens any claim of "completely accurate end to end" even if the parser itself behaves.
- There is no explicit model-family performance atlas that translates hotspot classes into `granite4:tiny-h` vs `qwen3.5:4b` vs `qwen3.5:9b` policy.

## 4. Hidden Assumptions
1. The math parser is treated as correct if it avoids obvious false positives. That is weaker than proving precision/recall on a gold corpus.
2. `native_kernel` provenance is assumed sufficient for execution relevance. It is not, because build-target closure and runtime path closure are still thin.
3. A native CPU scoring kernel is assumed more truthful than Python heuristics. It is faster and cleaner, but still a surrogate until correlated with runtime evidence.
4. One roofline model is assumed to generalize across Granite hybrid MoE paths and Qwen dense/hybrid paths. It will not.
5. Model support is treated as a backend-selection problem. It is actually a backend-selection plus context-budget plus cache-policy plus draft-policy problem.
6. Parser coverage, verifier coverage, and health coverage are assumed to mean the same thing. The repo already disproves that.
7. Hotspot ranking is assumed to map to optimization ROI. That needs a witness bridge to benchmarks and telemetry.
8. Prefix caching and speculative decode are assumed to be universally beneficial. External evidence says hybrid/Mamba paths often disable or constrain them.
9. The repo assumes missing graph edges are an analysis quality issue, not a product issue. For this subsystem they are a product issue because they block proof closure.
10. The CPU scan is assumed to be model-agnostic. In this repo, it should become model-aware because the runtime is explicitly model-pinned.

## 5. Candidate Implementation Phases
Practical candidates: 10. Moonshot candidates: 12.

### C01. Gold Corpus For Math Parse Truth
- Name: Gold-Corpus Equation Tribunal
- Suggested `phase_id`: `research`
- Core insight: treat parser accuracy like a compiler front-end, with a versioned corpus of positive and negative cases.
- External inspiration or analogy: tree-sitter corpus tests.
- Why it fits Saguaro and Anvil specifically: `saguaro/math/engine.py` is regex-heavy and already shows edge-case churn.
- Exact wiring points: `saguaro/math/engine.py`, `saguaro/math/pipeline.py`, `tests/test_saguaro_math.py`, `tests/test_saguaro_math_precision.py`
- Existing primitives it can reuse: current tests and `MathIRRecord` schema.
- New primitive, data flow, or subsystem needed: `.anvil/validation/math_truth_corpus/` with expected IR snapshots.
- `repo_scope`: `saguaro/math`, `tests`, `.anvil/validation`
- `owning_specialist_type`: `ParserVerificationArchitect`
- `allowed_writes`: `saguaro/math/*.py`, `tests/test_saguaro_math*.py`, `.anvil/validation/math_truth_corpus/*`
- `telemetry_contract`: emit corpus precision, recall, false-positive class, false-negative class.
- `required_evidence`: corpus manifest, diff report, failing examples.
- `rollback_criteria`: corpus becomes flaky across platforms.
- `promotion_gate`: precision >= 0.995 and no severity-A false positives on native kernels.
- `success_criteria`: parser claims are backed by a stable corpus.
- Why this creates value: closes the proof gap fast.
- Why this creates moat: accumulated edge corpus becomes hard to replicate.
- Main risk or failure mode: corpus becomes too synthetic.
- Smallest credible first experiment: add 50 repo-derived edge cases from `core/native` and `core/simd`.
- Confidence level: high.

### C02. Differential Parser Tribunal
- Name: Multi-Extractor Crosscheck
- Suggested `phase_id`: `deep_test_audit`
- Core insight: one parser extracts, a second parser disagrees, and disagreements become work items.
- External inspiration or analogy: N-version avionics and differential compiler testing.
- Why it fits Saguaro and Anvil specifically: low-confidence verifier metrics and markdown/code mix justify a second opinion.
- Exact wiring points: `saguaro/math/engine.py`, `saguaro/parsing/parser.py`, `saguaro/parsing/markdown.py`, `tests/test_saguaro_parser_languages.py`
- Existing primitives it can reuse: `SAGUAROParser`, language policy, current math IR.
- New primitive, data flow, or subsystem needed: `DifferentialParseWitness` artifact.
- `repo_scope`: `saguaro/math`, `saguaro/parsing`, `tests`
- `owning_specialist_type`: `SemanticReliabilityEngineer`
- `allowed_writes`: `saguaro/math/*.py`, `saguaro/parsing/*.py`, `tests/test_saguaro_parser_languages.py`
- `telemetry_contract`: emit disagreement rate by language and file class.
- `required_evidence`: disagreement ledger, adjudication notes.
- `rollback_criteria`: disagreement noise overwhelms signal.
- `promotion_gate`: disagreement rate on native sources < 0.5%.
- `success_criteria`: parser disagreements become bounded and explainable.
- Why this creates value: catches silent drift.
- Why this creates moat: builds a self-auditing parser surface.
- Main risk or failure mode: second parser is not materially independent.
- Smallest credible first experiment: compare current extractor against tree-sitter-backed structural slices on C++ test corpus.
- Confidence level: medium-high.

### C03. FFI Symbol Closure
- Name: Native Boundary Recovery Lane
- Suggested `phase_id`: `analysis_upgrade`
- Core insight: if runtime symbol coverage is 0, end-to-end proof is structurally broken.
- External inspiration or analogy: post-silicon traceability and hardware boundary bring-up.
- Why it fits Saguaro and Anvil specifically: health already reports zero runtime symbol coverage.
- Exact wiring points: `domains/verification/auto_verifier.py`, `saguaro/parsing/runtime_symbols.py`, `core/native/native_ops.py`, `core/native/CMakeLists.txt`
- Existing primitives it can reuse: `AutoVerifier.recover_runtime_symbols`.
- New primitive, data flow, or subsystem needed: symbol-to-build-target witness map.
- `repo_scope`: `domains/verification`, `saguaro/parsing`, `core/native`
- `owning_specialist_type`: `NativeInterfaceAuditor`
- `allowed_writes`: `domains/verification/*.py`, `saguaro/parsing/*.py`, `tests/test_auto_verifier_qsg_runtime.py`
- `telemetry_contract`: runtime symbol recovery rate, unresolved symbol classes.
- `required_evidence`: symbol manifest, recovered edges, unresolved list.
- `rollback_criteria`: recovery introduces false edges.
- `promotion_gate`: runtime symbol coverage > 80% on owned native targets.
- `success_criteria`: native hotspot claims can be traced to loadable runtime symbols.
- Why this creates value: converts heuristics into evidence.
- Why this creates moat: deep repo/runtime traceability.
- Main risk or failure mode: build metadata is inconsistent.
- Smallest credible first experiment: recover symbols for `cpu_math_runtime`, `qsg_parallel_kernels`, and backend modules.
- Confidence level: high.

### C04. Build Provenance Closure
- Name: Source-To-Target Completion
- Suggested `phase_id`: `feature_map`
- Core insight: math records need stable ownership to build targets and backend modules.
- External inspiration or analogy: SBOM plus timing closure.
- Why it fits Saguaro and Anvil specifically: many current records show empty `build_targets`.
- Exact wiring points: `saguaro/math/pipeline.py`, `core/native/CMakeLists.txt`, `tests/test_saguaro_math_pipeline.py`
- Existing primitives it can reuse: `_load_build_targets`, `_build_targets_for_file`.
- New primitive, data flow, or subsystem needed: backend-module provenance tag.
- `repo_scope`: `saguaro/math`, `core/native`, `tests`
- `owning_specialist_type`: `BuildGraphEngineer`
- `allowed_writes`: `saguaro/math/pipeline.py`, `tests/test_saguaro_math_pipeline.py`
- `telemetry_contract`: percent of native records with non-empty build target and backend binding.
- `required_evidence`: provenance coverage report.
- `rollback_criteria`: provenance becomes ambiguous or stale.
- `promotion_gate`: > 95% native records mapped to owned targets.
- `success_criteria`: every promoted hotspot can name its binary owner.
- Why this creates value: makes optimization accountable.
- Why this creates moat: repo-native provenance graph.
- Main risk or failure mode: split wrappers obscure ownership.
- Smallest credible first experiment: map all records under `core/native/qsg_parallel_kernels.cpp` and `core/native/cpu_math_runtime.cpp`.
- Confidence level: high.

### C05. Runtime Witness Bridge
- Name: Static-To-Runtime Correlator
- Suggested `phase_id`: `analysis_upgrade`
- Core insight: hotspot priority shall be judged by correlation to TTFT/TPOT/perf counters, not by surrogate score alone.
- External inspiration or analogy: F1 simulator correlation loops.
- Why it fits Saguaro and Anvil specifically: `core/native/runtime_telemetry.py` is rich enough to support this now.
- Exact wiring points: `saguaro/cpu/model.py`, `core/native/runtime_telemetry.py`, `benchmarks/native_qsg_benchmark.py`, `tests/test_runtime_telemetry.py`
- Existing primitives it can reuse: `benchmark_priority`, telemetry `hot_path_proof`, benchmark suite.
- New primitive, data flow, or subsystem needed: `HotspotWitnessRecord` joining scan IDs to runtime measurements.
- `repo_scope`: `saguaro/cpu`, `core/native`, `benchmarks`, `tests`
- `owning_specialist_type`: `PerformanceCorrelationEngineer`
- `allowed_writes`: `saguaro/cpu/*.py`, `core/native/runtime_telemetry.py`, `benchmarks/*.py`, `tests/test_runtime_telemetry.py`
- `telemetry_contract`: rank correlation between hotspot score and measured impact.
- `required_evidence`: correlation report by model and architecture.
- `rollback_criteria`: witness collection perturbs runtime heavily.
- `promotion_gate`: top-10 hotspot list predicts measured top-10 within 80% overlap.
- `success_criteria`: CPU scan becomes decision-grade.
- Why this creates value: stops wasted optimization effort.
- Why this creates moat: self-calibrating analysis engine.
- Main risk or failure mode: mapping from hotspot to timing bucket is too coarse.
- Smallest credible first experiment: correlate `qsg_parallel_kernels.cpp` sampling hotspots to decode metrics on `qwen3.5:4b`.
- Confidence level: high.

### C06. CPU Scan Truth Dashboard
- Name: Verification Cockpit
- Suggested `phase_id`: `roadmap_draft`
- Core insight: engineers need one artifact that says working, broken, or uncertain.
- External inspiration or analogy: mission control green/yellow/red boards.
- Why it fits Saguaro and Anvil specifically: signals are spread across report files, tests, health, and benchmarks.
- Exact wiring points: `.anvil/validation/`, `saguaro/roadmap/validator.py`, `audit/runner/benchmark_suite.py`
- Existing primitives it can reuse: roadmap validator, benchmark suite, current validation artifacts.
- New primitive, data flow, or subsystem needed: `cpu_math_truth_dashboard.json`.
- `repo_scope`: `audit`, `saguaro/roadmap`, `.anvil/validation`
- `owning_specialist_type`: `ValidationSystemsArchitect`
- `allowed_writes`: `audit/runner/*.py`, `saguaro/roadmap/*.py`, `.anvil/validation/*`
- `telemetry_contract`: status vector by parser truth, scan truth, runtime truth.
- `required_evidence`: one-page machine-readable dashboard.
- `rollback_criteria`: dashboard hides uncertainty.
- `promotion_gate`: dashboard exposes every blocking red/yellow source.
- `success_criteria`: no human has to infer system status from scattered files.
- Why this creates value: operational clarity.
- Why this creates moat: strong engineering governance.
- Main risk or failure mode: becomes decorative.
- Smallest credible first experiment: compile current health, tests, parse, scan, and benchmark artifacts into one JSON.
- Confidence level: high.

### C07. Architecture Alias And Topology Canonicalizer
- Name: CPU Identity Sanitizer
- Suggested `phase_id`: `intake`
- Core insight: scan outputs are only comparable if architecture identities normalize cleanly.
- External inspiration or analogy: sensor fusion canonicalization.
- Why it fits Saguaro and Anvil specifically: `saguaro/cpu/topology.py` is intentionally lightweight.
- Exact wiring points: `saguaro/cpu/topology.py`, `tests/test_saguaro_cpu_topology.py`, `core/native/thread_config.cpp`
- Existing primitives it can reuse: `normalize_arch_name`, architecture packs.
- New primitive, data flow, or subsystem needed: host capability snapshot with AMX/AVX512/NUMA fields.
- `repo_scope`: `saguaro/cpu`, `core/native`, `tests`
- `owning_specialist_type`: `CpuTopologyEngineer`
- `allowed_writes`: `saguaro/cpu/topology.py`, `tests/test_saguaro_cpu_topology.py`, `core/native/thread_config.cpp`
- `telemetry_contract`: canonical arch id, capability flags, mismatch count.
- `required_evidence`: normalized topology manifest.
- `rollback_criteria`: canonicalization hides meaningful differences.
- `promotion_gate`: same host yields same pack across CLI and runtime.
- `success_criteria`: topology is a trustworthy join key.
- Why this creates value: removes false benchmark variance.
- Why this creates moat: better host intelligence.
- Main risk or failure mode: over-normalization.
- Smallest credible first experiment: align `saguaro cpu scan` arch packs with runtime telemetry topology fields.
- Confidence level: medium-high.

### C08. Native/Python Parity Sentinel
- Name: Surrogate Parity Guard
- Suggested `phase_id`: `deep_test_audit`
- Core insight: every native score shall have a fallback parity window so drift is visible.
- External inspiration or analogy: redundant flight computers.
- Why it fits Saguaro and Anvil specifically: CPU scan mixes `native` and `python_fallback` engines today.
- Exact wiring points: `saguaro/cpu/model.py`, `saguaro/cpu/cache.py`, `saguaro/cpu/register_pressure.py`, `tests/test_saguaro_cpu_twin.py`
- Existing primitives it can reuse: existing fallback implementations.
- New primitive, data flow, or subsystem needed: parity diff thresholds by metric.
- `repo_scope`: `saguaro/cpu`, `tests`
- `owning_specialist_type`: `NumericalParityEngineer`
- `allowed_writes`: `saguaro/cpu/*.py`, `tests/test_saguaro_cpu_twin.py`
- `telemetry_contract`: native-fallback delta per hotspot.
- `required_evidence`: parity report across sampled records.
- `rollback_criteria`: fallback stops being interpretable.
- `promotion_gate`: parity deltas bounded for all first-order metrics.
- `success_criteria`: native runtime changes cannot silently skew results.
- Why this creates value: safer native evolution.
- Why this creates moat: reliable hybrid analysis stack.
- Main risk or failure mode: false alarms on legitimately improved scoring.
- Smallest credible first experiment: diff cache/vectorization/register results on 100 corpus records.
- Confidence level: high.

### C09. Model-Family Runtime Atlas
- Name: Three-Model CPU Atlas
- Suggested `phase_id`: `feature_map`
- Core insight: the same hotspot class has different economics on Granite tiny, Qwen 4B, and Qwen 9B.
- External inspiration or analogy: separate car setup sheets for different tracks.
- Why it fits Saguaro and Anvil specifically: the runtime is explicitly pinned to exactly these models.
- Exact wiring points: `core/model/chat_templates.py`, `core/model/model_profile.py`, `core/qsg/ollama_adapter.py`, `benchmarks/native_qsg_benchmark.py`, `tests/test_qsg_adapter_unification.py`
- Existing primitives it can reuse: `ModelProfile`, strict prompt contracts, benchmark suite.
- New primitive, data flow, or subsystem needed: model-family hotspot budget table.
- `repo_scope`: `core/model`, `core/qsg`, `benchmarks`, `tests`
- `owning_specialist_type`: `InferenceProfileArchitect`
- `allowed_writes`: `core/model/*.py`, `core/qsg/*.py`, `benchmarks/*.py`, `tests/test_qsg_adapter_unification.py`
- `telemetry_contract`: TTFT/TPOT/memory budget by model and context band.
- `required_evidence`: atlas report for all three models.
- `rollback_criteria`: model atlas diverges from runtime measurements.
- `promotion_gate`: every optimization proposal names target model regime.
- `success_criteria`: optimization becomes model-specific by default.
- Why this creates value: avoids one-size-fits-none tuning.
- Why this creates moat: operational knowledge tied to real runtime.
- Main risk or failure mode: benchmark matrix becomes too expensive.
- Smallest credible first experiment: 3 models x 3 context bands x 2 prompt shapes.
- Confidence level: high.

### C10. Auto-Benchmark Workcells
- Name: Scheduled Hotspot Bench Harness
- Suggested `phase_id`: `development`
- Core insight: top hotspots shall automatically spawn microbench work items.
- External inspiration or analogy: dyno cell scheduling.
- Why it fits Saguaro and Anvil specifically: `benchmark_priority` and `schedule_twin` already exist.
- Exact wiring points: `saguaro/cpu/model.py`, `benchmarks/native_qsg_benchmark.py`, `audit/runner/benchmark_suite.py`
- Existing primitives it can reuse: `benchmark_priority`, benchmark suite.
- New primitive, data flow, or subsystem needed: generated benchmark capsules per hotspot.
- `repo_scope`: `saguaro/cpu`, `benchmarks`, `audit`
- `owning_specialist_type`: `BenchmarkAutomationEngineer`
- `allowed_writes`: `saguaro/cpu/model.py`, `benchmarks/*.py`, `audit/runner/*.py`
- `telemetry_contract`: benchmark queue depth, completed capsules, rejected capsules.
- `required_evidence`: benchmark capsule manifests and results.
- `rollback_criteria`: harness cost exceeds its value.
- `promotion_gate`: auto-generated capsules reproduce at least one known hotspot.
- `success_criteria`: hotspot ranking produces executable next steps.
- Why this creates value: shortens optimizer feedback loops.
- Why this creates moat: automation around performance science.
- Main risk or failure mode: microbenchmarks misrepresent end-to-end behavior.
- Smallest credible first experiment: spawn capsules for top 3 `qsg_parallel_kernels.cpp` records.
- Confidence level: medium-high.

### C11. Granite Hybrid Cache Governor
- Name: Hybrid KV And State Controller
- Suggested `phase_id`: `development`
- Core insight: Granite tiny will win or lose on hybrid cache/state motion more than on generic transformer tricks.
- External inspiration or analogy: hybrid power unit energy management.
- Why it fits Saguaro and Anvil specifically: repo has Granite-specific backend selection but not Granite-specific cache governance.
- Exact wiring points: `core/qsg/ollama_adapter.py`, `core/native/paged_kv_cache.py`, `core/native/native_kv_cache.cpp`, `core/native/runtime_telemetry.py`
- Existing primitives it can reuse: model profiles, KV cache telemetry.
- New primitive, data flow, or subsystem needed: hybrid cache mode controller keyed by Granite family.
- `repo_scope`: `core/qsg`, `core/native`, `tests`
- `owning_specialist_type`: `HybridRuntimeEngineer`
- `allowed_writes`: `core/qsg/*.py`, `core/native/*kv_cache*`, `tests/test_runtime_telemetry.py`, `tests/test_qsg_adapter_unification.py`
- `telemetry_contract`: hybrid cache mode, TTFT delta, cache hit delta, fragmentation delta.
- `required_evidence`: Granite-only benchmark before/after.
- `rollback_criteria`: Granite latency regresses on short prompts.
- `promotion_gate`: Granite TTFT and TPOT both improve on at least one real prompt class.
- `success_criteria`: Granite behavior is no longer treated as Qwen-lite.
- Why this creates value: direct support for the stated model set.
- Why this creates moat: architecture-specific serving policy.
- Main risk or failure mode: no hybrid control surface exposed low enough in stack.
- Smallest credible first experiment: separate Granite cache path for prompt-lookup vs decode-heavy prompts.
- Confidence level: medium.

### C12. Qwen Vocabulary Shortlist Drafting
- Name: Qwen Dynamic Shortlist Speculation
- Suggested `phase_id`: `development`
- Core insight: Qwen benefits from context-aware shortlist drafting more than from static draft heuristics.
- External inspiration or analogy: DynaSpec and HeteroSpec.
- Why it fits Saguaro and Anvil specifically: `core/qsg/ollama_adapter.py` already holds strict speculative flags and profile knobs.
- Exact wiring points: `core/qsg/ollama_adapter.py`, `core/native/qsg_parallel_kernels.cpp`, `core/native/runtime_telemetry.py`
- Existing primitives it can reuse: `spec_num_candidates`, acceptance thresholds, telemetry fields.
- New primitive, data flow, or subsystem needed: Qwen-only shortlist controller.
- `repo_scope`: `core/qsg`, `core/native`, `tests`
- `owning_specialist_type`: `SpeculativeDecodingEngineer`
- `allowed_writes`: `core/qsg/ollama_adapter.py`, `core/native/qsg_parallel_kernels.cpp`, `tests/test_qsg_adapter_unification.py`, `tests/test_runtime_telemetry.py`
- `telemetry_contract`: accepted length, rejected tokens, shortlist size, shortlist miss rate.
- `required_evidence`: 4B and 9B compare against baseline.
- `rollback_criteria`: acceptance collapses or rare-token quality regresses.
- `promotion_gate`: net throughput gain without correctness regressions.
- `success_criteria`: Qwen paths become policy-optimized rather than static.
- Why this creates value: better CPU economics for both Qwen sizes.
- Why this creates moat: model-specific speculative control.
- Main risk or failure mode: shortlist policy becomes task-fragile.
- Smallest credible first experiment: dynamic shortlist only in low-entropy contexts on `qwen3.5:4b`.
- Confidence level: medium.

### C13. Context-Band Budget Controller
- Name: Prompt-Length Flight Envelope
- Suggested `phase_id`: `questionnaire`
- Core insight: optimizations should activate by context band, not globally.
- External inspiration or analogy: flight envelope protection.
- Why it fits Saguaro and Anvil specifically: runtime already knows TTFT, decode split, and model family.
- Exact wiring points: `core/ollama_client.py`, `core/qsg/ollama_adapter.py`, `core/native/runtime_telemetry.py`
- Existing primitives it can reuse: native context caps and telemetry.
- New primitive, data flow, or subsystem needed: policy table by prompt class and token band.
- `repo_scope`: `core/ollama_client.py`, `core/qsg`, `core/native`
- `owning_specialist_type`: `RuntimePolicyEngineer`
- `allowed_writes`: `core/ollama_client.py`, `core/qsg/ollama_adapter.py`, `tests/test_qsg_adapter_unification.py`
- `telemetry_contract`: active envelope id, policy choice, outcome delta.
- `required_evidence`: policy decision trace.
- `rollback_criteria`: policy thrashes between bands.
- `promotion_gate`: policy choice stable and beneficial.
- `success_criteria`: features activate only where they pay off.
- Why this creates value: reduces accidental regressions.
- Why this creates moat: intelligent runtime policy.
- Main risk or failure mode: too many bands, not enough data.
- Smallest credible first experiment: short, medium, long prompt bands for all three models.
- Confidence level: high.

### C14. Runtime Roofline Assimilation
- Name: Hierarchical Roofline Twin
- Suggested `phase_id`: `analysis_upgrade`
- Core insight: static roofline shall be corrected by runtime counters and telemetry.
- External inspiration or analogy: runtime-specific roofline guidance.
- Why it fits Saguaro and Anvil specifically: `saguaro/cpu/roofline.py` is intentionally simple and ready to be upgraded.
- Exact wiring points: `saguaro/cpu/roofline.py`, `core/native/runtime_telemetry.py`, `benchmarks/native_qsg_benchmark.py`
- Existing primitives it can reuse: roofline estimate, telemetry timing fields.
- New primitive, data flow, or subsystem needed: hierarchical roofline report across L1/L2/L3/memory tiers.
- `repo_scope`: `saguaro/cpu`, `core/native`, `benchmarks`
- `owning_specialist_type`: `PerformanceModelEngineer`
- `allowed_writes`: `saguaro/cpu/roofline.py`, `core/native/runtime_telemetry.py`, `tests/test_saguaro_cpu_twin.py`
- `telemetry_contract`: observed intensity, predicted intensity, bound agreement.
- `required_evidence`: roofline correlation plot.
- `rollback_criteria`: model adds noise without predictive power.
- `promotion_gate`: bound agreement > 80% on benchmark corpus.
- `success_criteria`: roofline becomes useful for prioritization.
- Why this creates value: smarter hotspot triage.
- Why this creates moat: repo-specific performance science.
- Main risk or failure mode: missing perf counters on user machines.
- Smallest credible first experiment: infer hierarchical bounds from telemetry-only before perf events.
- Confidence level: medium-high.

### C15. Schedule Compiler
- Name: Hotspot-To-Schedule Compiler
- Suggested `phase_id`: `eid`
- Core insight: `schedule_twin` should become a code-generation-adjacent planning primitive.
- External inspiration or analogy: Halide/TVM schedule search, F1 setup sheets.
- Why it fits Saguaro and Anvil specifically: current schedule output is shallow but structurally promising.
- Exact wiring points: `saguaro/cpu/schedule_twin.py`, `saguaro/cpu/model.py`, `core/native/qsg_parallel_kernels.cpp`
- Existing primitives it can reuse: schedule candidates and blocker masks.
- New primitive, data flow, or subsystem needed: schedule IR and transform recipes.
- `repo_scope`: `saguaro/cpu`, `core/native`, `tests`
- `owning_specialist_type`: `KernelPlanningArchitect`
- `allowed_writes`: `saguaro/cpu/*.py`, `tests/test_saguaro_cpu_twin.py`
- `telemetry_contract`: proposed schedule, accepted schedule, measured delta.
- `required_evidence`: schedule recommendation ledger.
- `rollback_criteria`: compiler emits generic nonsense.
- `promotion_gate`: schedule recipes explain at least 3 high-value hotspots.
- `success_criteria`: schedule output becomes actionable.
- Why this creates value: bridges analysis to engineering action.
- Why this creates moat: higher-level CPU optimization primitive.
- Main risk or failure mode: insufficient structural richness in current IR.
- Smallest credible first experiment: compile `vectorize`, `cache_block`, `tree_reduce`, `scalar_stabilize` into explicit recipes.
- Confidence level: medium.

### C16. Arithmetic Complexity Reducer
- Name: Algebraic Hotspot Simplifier
- Suggested `phase_id`: `roadmap_draft`
- Core insight: some hotspots should be algebraically simplified or approximated before kernel tuning.
- External inspiration or analogy: symbolic algebra passes in compilers.
- Why it fits Saguaro and Anvil specifically: CPU scan currently ranks complexity but does not propose reductions.
- Exact wiring points: `saguaro/math/engine.py`, `saguaro/cpu/model.py`, `core/native/fast_math.h`
- Existing primitives it can reuse: structural score, operator count, function call count.
- New primitive, data flow, or subsystem needed: `ComplexityReductionHint` IR.
- `repo_scope`: `saguaro/math`, `saguaro/cpu`, `core/native`
- `owning_specialist_type`: `NumericalMethodsArchitect`
- `allowed_writes`: `saguaro/math/*.py`, `saguaro/cpu/*.py`, `tests/test_saguaro_math_access.py`
- `telemetry_contract`: reduction hint type, estimated cost delta, applied/not-applied.
- `required_evidence`: candidate simplification report.
- `rollback_criteria`: hints are too aggressive or numerically unsafe.
- `promotion_gate`: hints identify at least one safe reduction class with measurable benefit.
- `success_criteria`: hotspot analysis can recommend lower-complexity formulations.
- Why this creates value: some wins are algorithmic, not microarchitectural.
- Why this creates moat: deeper mechanism-level optimization.
- Main risk or failure mode: unsafe numerics.
- Smallest credible first experiment: identify repeated `std::max`, clamp, scalar bookkeeping patterns in `cpu_math_runtime.cpp` and similar utility hotspots.
- Confidence level: medium.

### C17. Counterexample Forge
- Name: Adversarial CPU/Math Corpus Generator
- Suggested `phase_id`: `research`
- Core insight: the parser and CPU twin should be stress-tested by generated edge code, not only repo history.
- External inspiration or analogy: fuzzing plus hardware validation pattern generation.
- Why it fits Saguaro and Anvil specifically: current tests are strong but still tactical.
- Exact wiring points: `saguaro/math/engine.py`, `saguaro/cpu/model.py`, `tests/test_saguaro_math_languages.py`, `tests/test_saguaro_cpu_scan.py`
- Existing primitives it can reuse: language policies and current test harnesses.
- New primitive, data flow, or subsystem needed: adversarial corpus generator with metamorphic mutations.
- `repo_scope`: `saguaro/math`, `saguaro/cpu`, `tests`
- `owning_specialist_type`: `AdversarialReliabilityEngineer`
- `allowed_writes`: `saguaro/math/*.py`, `saguaro/cpu/*.py`, `tests/test_saguaro_*`
- `telemetry_contract`: generated case count, unique failure class count.
- `required_evidence`: minimized counterexamples.
- `rollback_criteria`: generated cases are unrealistic noise.
- `promotion_gate`: at least 5 unique high-value failure classes or proof of robustness.
- `success_criteria`: parser/scan robustness increases materially.
- Why this creates value: closes unknown unknowns.
- Why this creates moat: difficult reliability corpus.
- Main risk or failure mode: generation quality too low.
- Smallest credible first experiment: mutate lambda/default-arg/build-log/config-literal edge cases.
- Confidence level: high.

### C18. NUMA And L3 Flight Controller
- Name: Topology-Aware Runtime Governor
- Suggested `phase_id`: `development`
- Core insight: hotspot optimization on CPU is incomplete without topology control.
- External inspiration or analogy: race-engine torque vectoring.
- Why it fits Saguaro and Anvil specifically: telemetry already records NUMA and affinity fields.
- Exact wiring points: `core/native/thread_config.cpp`, `core/native/numa_topology.cpp`, `core/native/runtime_telemetry.py`, `core/native/parallel_generation.py`
- Existing primitives it can reuse: topology and telemetry fields.
- New primitive, data flow, or subsystem needed: runtime governor selecting affinity and memory policy.
- `repo_scope`: `core/native`, `tests`, `benchmarks`
- `owning_specialist_type`: `CpuControlSystemsEngineer`
- `allowed_writes`: `core/native/*.cpp`, `core/native/runtime_telemetry.py`, `tests/test_runtime_control_policy.py`
- `telemetry_contract`: affinity mode, migrations, L3 domain activity, throughput delta.
- `required_evidence`: before/after topology benchmark.
- `rollback_criteria`: instability or machine-specific regressions.
- `promotion_gate`: reduced migrations and improved latency stability.
- `success_criteria`: runtime exploits machine topology deliberately.
- Why this creates value: real CPU wins on long-running decode.
- Why this creates moat: systems-grade serving behavior.
- Main risk or failure mode: portability burden.
- Smallest credible first experiment: pin orchestrator and workers separately and compare jitter.
- Confidence level: medium.

### C19. Weight Layout Superpacker
- Name: Model-Family Weight Repacker
- Suggested `phase_id`: `development`
- Core insight: reduce CPU cost by repacking GGUF-loaded weights into backend-specific contiguous layouts once.
- External inspiration or analogy: pit-lane setup for the known track.
- Why it fits Saguaro and Anvil specifically: split backends and quantized kernels already exist.
- Exact wiring points: `core/model/gguf_loader.py`, `core/native/interleave_weights.cpp`, `core/native/weight_store.py`, `core/native/quantized_matmul.cpp`
- Existing primitives it can reuse: native weight store and interleave kernels.
- New primitive, data flow, or subsystem needed: persisted repacked artifacts keyed by backend and ISA.
- `repo_scope`: `core/model`, `core/native`, `tests`
- `owning_specialist_type`: `WeightLayoutEngineer`
- `allowed_writes`: `core/model/gguf_loader.py`, `core/native/*weight*`, `tests/test_native_qsg_engine.py`
- `telemetry_contract`: repack hit rate, load-time delta, decode delta.
- `required_evidence`: load and runtime benchmarks.
- `rollback_criteria`: load time balloons or artifacts fragment disk.
- `promotion_gate`: net wall-clock improvement on repeated sessions.
- `success_criteria`: repeated runs stop paying the same layout penalty.
- Why this creates value: practical latency and throughput gain.
- Why this creates moat: runtime-aware model preparation.
- Main risk or failure mode: artifact invalidation complexity.
- Smallest credible first experiment: repack LM head and most frequent QKV layouts for `qwen35` backend.
- Confidence level: medium-high.

### C20. Precision Elasticity Plane
- Name: Risk-Bounded Precision Switching
- Suggested `phase_id`: `convergence`
- Core insight: CPU optimization should trade precision and speed dynamically under explicit risk budgets.
- External inspiration or analogy: fly-by-wire law switching.
- Why it fits Saguaro and Anvil specifically: repo already contains quantized kernels and telemetry; missing piece is policy.
- Exact wiring points: `core/native/quantized_matmul.cpp`, `core/native/amx_kernels.cpp`, `core/native/runtime_telemetry.py`, `core/qsg/runtime_contracts.py`
- Existing primitives it can reuse: backend capability registry, quantized kernels.
- New primitive, data flow, or subsystem needed: precision governor with rollback thresholds.
- `repo_scope`: `core/native`, `core/qsg`, `tests`
- `owning_specialist_type`: `NumericalControlEngineer`
- `allowed_writes`: `core/native/*.cpp`, `core/qsg/runtime_contracts.py`, `tests/test_qsg_runtime_contracts.py`
- `telemetry_contract`: active precision mode, quality deltas, rollback events.
- `required_evidence`: quality/speed frontier report.
- `rollback_criteria`: semantic drift or acceptance-rate collapse.
- `promotion_gate`: bounded quality loss under declared task classes.
- `success_criteria`: precision becomes controllable, not fixed.
- Why this creates value: more speed on constrained CPUs.
- Why this creates moat: explicit risk-bounded optimization.
- Main risk or failure mode: hidden quality regressions.
- Smallest credible first experiment: switch only selected utility kernels or LM-head paths under validated prompts.
- Confidence level: medium.

### C21. Proof-Carrying Hotspot Packets
- Name: Hotspot Evidence Capsules
- Suggested `phase_id`: `convergence`
- Core insight: every promoted hotspot shall carry its source, provenance, runtime witness, benchmark result, and rollback rule.
- External inspiration or analogy: airworthiness paperwork for each modification.
- Why it fits Saguaro and Anvil specifically: roadmap validator and traceability stack already exist.
- Exact wiring points: `saguaro/roadmap/validator.py`, `saguaro/validation/engine.py`, `standards/traceability/TRACEABILITY.jsonl`
- Existing primitives it can reuse: roadmap validator, traceability service.
- New primitive, data flow, or subsystem needed: hotspot capsule schema.
- `repo_scope`: `saguaro/roadmap`, `saguaro/validation`, `standards/traceability`
- `owning_specialist_type`: `TraceabilityArchitect`
- `allowed_writes`: `saguaro/roadmap/*.py`, `saguaro/validation/*.py`, `standards/traceability/*`
- `telemetry_contract`: capsule completeness, stale capsule count.
- `required_evidence`: capsule manifest and completion graph.
- `rollback_criteria`: capsule overhead slows work disproportionately.
- `promotion_gate`: all promoted optimizations have full capsules.
- `success_criteria`: optimization history becomes auditable.
- Why this creates value: cleaner operations and safer rollbacks.
- Why this creates moat: deep institutional memory.
- Main risk or failure mode: engineers bypass the process.
- Smallest credible first experiment: generate capsules for top 5 native hotspots.
- Confidence level: high.

### C22. Self-Healing Validation Gate
- Name: CPU Math Roadmap Auto-Governor
- Suggested `phase_id`: `convergence`
- Core insight: the roadmap should refuse promotion if proof is stale or contradictory.
- External inspiration or analogy: launch commit criteria.
- Why it fits Saguaro and Anvil specifically: current roadmap validator can already classify missing and partial evidence.
- Exact wiring points: `saguaro/roadmap/validator.py`, `domains/verification/auto_verifier.py`, `.anvil/validation/*`
- Existing primitives it can reuse: roadmap validator, auto verifier.
- New primitive, data flow, or subsystem needed: CPU/math-specific gates.
- `repo_scope`: `saguaro/roadmap`, `domains/verification`, `.anvil/validation`
- `owning_specialist_type`: `GovernanceSystemsEngineer`
- `allowed_writes`: `saguaro/roadmap/*.py`, `domains/verification/*.py`, `tests/test_saguaro_roadmap_validator.py`
- `telemetry_contract`: stale evidence count, contradictory evidence count, blocked promotion count.
- `required_evidence`: gate report tied to roadmap path.
- `rollback_criteria`: gate blocks obviously safe work.
- `promotion_gate`: false-block rate is acceptable and documented.
- `success_criteria`: roadmap status is machine-checkable.
- Why this creates value: prevents cargo-cult optimization.
- Why this creates moat: strong engineering governance loop.
- Main risk or failure mode: governance friction exceeds benefit.
- Smallest credible first experiment: block optimization phases when parser truth gate is yellow/red.
- Confidence level: high.

## 6. Critical Pressure Test
- Elegant but likely wrong: `C15` Schedule Compiler is attractive, but the current IR may not yet be rich enough to recommend nontrivial transforms safely.
- Elegant but likely wrong: `C20` Precision Elasticity Plane can easily become a complicated way to hide quality regressions behind latency wins.
- Ugly but strategically powerful: `C03` FFI Symbol Closure is not glamorous, but without it the repo cannot honestly claim end-to-end proof.
- Ugly but strategically powerful: `C05` Runtime Witness Bridge will feel bureaucratic until it prevents the team from optimizing the wrong hotspot for a week.
- Likely to fail because the repo is missing a key primitive: `C11` Granite Hybrid Cache Governor needs lower-level state visibility than the current Granite path may expose.
- Likely to fail because the repo is missing a key primitive: `C12` Qwen Dynamic Shortlist Speculation needs a better context-difficulty signal than the runtime currently records.
- Most robust near-term bets: `C01`, `C03`, `C04`, `C05`, `C09`.
- Biggest hidden trap: treating the current passing tests as proof of complete accuracy. They are proof of competence, not proof of closure.

## 7. Synthesis
- Strongest overall ideas: `C01` Gold-Corpus Equation Tribunal, `C03` FFI Symbol Closure, `C05` Static-To-Runtime Correlator, `C09` Three-Model CPU Atlas, `C19` Weight Layout Superpacker.
- Best novelty/plausibility balance: `C05` Runtime Witness Bridge and `C09` Model-Family Runtime Atlas.
- Most feasible now: `C01` because the parser and tests already exist and the report already names the failure classes.
- Biggest long-term moat bet: `C21` Proof-Carrying Hotspot Packets, because it turns local performance work into persistent systems knowledge.
- Cleanest unifier with the current codebase: `C05` Runtime Witness Bridge, because it joins `saguaro/cpu`, `core/native/runtime_telemetry.py`, and existing benchmark infrastructure.
- Best first prototype: `C01` + `C05` in sequence. First prove parser truth, then correlate hotspot truth.
- Overall judgment: CPU scan and math parse are working operationally enough to continue, but they are not yet proven to NASA/F1 evidence standards. The roadmap should therefore start with proof closure and only then spend the serious optimization budget.

## 8. Implementation Program

### Phase 1
- `phase_id`: `research`
- Phase title: Truth Pack For CPU Math Verification
- Objective: Prove or falsify the claim that `saguaro math parse` and `saguaro cpu scan` are end-to-end accurate on owned native sources.
- Dependencies: none
- Repo scope: `saguaro/math`, `saguaro/cpu`, `.anvil/validation`, `tests`
- Owning specialist type: `ParserVerificationArchitect`
- Allowed writes: `saguaro/math/*.py`, `saguaro/cpu/*.py`, `.anvil/validation/math_truth_corpus/*`, `tests/test_saguaro_math*.py`, `tests/test_saguaro_cpu*.py`
- Telemetry contract: corpus precision, corpus recall, false-positive taxonomy, false-negative taxonomy, native-vs-fallback parity deltas.
- Required evidence: gold corpus manifest, parser diff report, CPU scan truth table, parity report.
- Rollback criteria: corpus proves parser precision below 0.995 on native code or severity-A false positives remain.
- Promotion gate: all severity-A parser/scan failure classes are eliminated or explicitly waived with evidence.
- Success criteria: the team can state exactly what classes are covered, not merely that tests pass.
- Exact wiring points: `saguaro/math/engine.py`, `saguaro/math/pipeline.py`, `saguaro/cpu/model.py`, `tests/test_saguaro_math.py`, `tests/test_saguaro_math_precision.py`, `tests/test_saguaro_cpu_scan.py`, `tests/test_saguaro_cpu_twin.py`
- Deliverables: gold corpus, truth report, parity thresholds, updated tests.
- Tests: `tests/test_saguaro_math.py`, `tests/test_saguaro_math_precision.py`, `tests/test_saguaro_cpu_scan.py`, `tests/test_saguaro_cpu_twin.py`
- Verification commands: `pytest tests/test_saguaro_math.py tests/test_saguaro_math_precision.py tests/test_saguaro_cpu_scan.py tests/test_saguaro_cpu_twin.py -q`; `./venv/bin/saguaro math parse --path core/native --format json`; `./venv/bin/saguaro cpu scan --path core/native --arch x86_64-avx2 --format json`; `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: either promote to runtime correlation or enter Phase 2 remediation.

### Phase 2
- `phase_id`: `deep_test_audit`
- Phase title: Differential Tribunal And Repair Lane
- Objective: If Phase 1 exposes accuracy holes, fix them once more and close them with differential evidence.
- Dependencies: `research`
- Repo scope: `saguaro/math`, `saguaro/parsing`, `domains/verification`, `tests`
- Owning specialist type: `SemanticReliabilityEngineer`
- Allowed writes: `saguaro/math/*.py`, `saguaro/parsing/*.py`, `domains/verification/*.py`, `tests/test_saguaro_parser_languages.py`, `tests/test_auto_verifier_qsg_runtime.py`
- Telemetry contract: disagreement rate by language, recovered symbol coverage, unresolved edge classes.
- Required evidence: differential disagreement ledger, recovered symbol manifest, updated health delta.
- Rollback criteria: disagreement rate stays high or repair raises new false positives.
- Promotion gate: runtime symbol coverage materially improves and disagreement rate on native sources falls below threshold.
- Success criteria: the proof gap is reduced enough that optimization work is not premature.
- Exact wiring points: `saguaro/parsing/parser.py`, `saguaro/parsing/runtime_symbols.py`, `domains/verification/auto_verifier.py`, `saguaro/math/engine.py`
- Deliverables: differential extractor, symbol recovery improvements, verifier alignment fix.
- Tests: `tests/test_saguaro_parser_languages.py`, `tests/test_auto_verifier_qsg_runtime.py`, `tests/test_saguaro_roadmap_validator.py`
- Verification commands: `pytest tests/test_saguaro_parser_languages.py tests/test_auto_verifier_qsg_runtime.py tests/test_saguaro_roadmap_validator.py -q`; `./venv/bin/saguaro health`; `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: promotion only if Phase 1 defects are closed or bounded.

### Phase 3
- `phase_id`: `feature_map`
- Phase title: Three-Model CPU Atlas
- Objective: Build the model-family optimization map for `granite4:tiny-h`, `qwen3.5:4b`, and `qwen3.5:9b`.
- Dependencies: `research`; `deep_test_audit`
- Repo scope: `core/model`, `core/qsg`, `benchmarks`, `tests`
- Owning specialist type: `InferenceProfileArchitect`
- Allowed writes: `core/model/*.py`, `core/qsg/*.py`, `benchmarks/*.py`, `tests/test_qsg_adapter_unification.py`, `tests/test_runtime_telemetry.py`
- Telemetry contract: TTFT, TPOT, prompt-band classification, backend selection, active optimization policy, model-family deltas.
- Required evidence: atlas report across 3 models x 3 context bands x representative prompt classes.
- Rollback criteria: atlas data contradicts runtime behavior or cannot be reproduced.
- Promotion gate: every subsequent optimization names target model family and prompt regime.
- Success criteria: optimization stops being generic and becomes family-specific.
- Exact wiring points: `core/model/chat_templates.py`, `core/model/model_profile.py`, `core/qsg/ollama_adapter.py`, `core/ollama_client.py`, `benchmarks/native_qsg_benchmark.py`
- Deliverables: model atlas, prompt-band policy table, benchmark matrix.
- Tests: `tests/test_qsg_adapter_unification.py`, `tests/test_runtime_telemetry.py`, `tests/test_native_qsg_benchmark.py`
- Verification commands: `pytest tests/test_qsg_adapter_unification.py tests/test_runtime_telemetry.py tests/test_native_qsg_benchmark.py -q`; `./venv/bin/saguaro build-graph`; `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: promoted hotspots are labeled by model-family value.

### Phase 4
- `phase_id`: `analysis_upgrade`
- Phase title: Runtime Witness Closure
- Objective: Correlate static hotspot ranking with real runtime evidence and close the source-to-runtime chain.
- Dependencies: `feature_map`
- Repo scope: `saguaro/cpu`, `core/native`, `benchmarks`, `audit`, `tests`
- Owning specialist type: `PerformanceCorrelationEngineer`
- Allowed writes: `saguaro/cpu/*.py`, `core/native/runtime_telemetry.py`, `benchmarks/*.py`, `audit/runner/*.py`, `tests/test_runtime_telemetry.py`, `tests/test_saguaro_cpu_twin.py`
- Telemetry contract: hotspot-to-runtime correlation, bound agreement, witness completeness, stale witness count.
- Required evidence: correlation report, witness manifests, hotspot capsule prototypes.
- Rollback criteria: witness path adds significant benchmark overhead or poor correlation.
- Promotion gate: hotspot ranking predicts measured payoff with acceptable accuracy.
- Success criteria: CPU scan becomes an optimization allocator rather than an advisory guesser.
- Exact wiring points: `saguaro/cpu/model.py`, `saguaro/cpu/roofline.py`, `core/native/runtime_telemetry.py`, `benchmarks/native_qsg_benchmark.py`, `audit/runner/benchmark_suite.py`
- Deliverables: witness bridge, hierarchical roofline report, rank-correlation report.
- Tests: `tests/test_saguaro_cpu_twin.py`, `tests/test_runtime_telemetry.py`, `tests/test_native_qsg_benchmark.py`
- Verification commands: `pytest tests/test_saguaro_cpu_twin.py tests/test_runtime_telemetry.py tests/test_native_qsg_benchmark.py -q`; `./venv/bin/saguaro cpu scan --path core/native --arch x86_64-avx2 --format json`; `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: optimization lane opens only after correlation is credible.

### Phase 5
- `phase_id`: `development`
- Phase title: Hotspot Optimization Program
- Objective: Reduce CPU complexity and memory pressure at the highest-value hotspots while preserving correctness for the three supported models.
- Dependencies: `analysis_upgrade`
- Repo scope: `core/native`, `core/qsg`, `core/model`, `benchmarks`, `tests`
- Owning specialist type: `NativeRuntimeOptimizationLead`
- Allowed writes: `core/native/*.cpp`, `core/native/*.h`, `core/qsg/*.py`, `core/model/*.py`, `benchmarks/*.py`, `tests/test_runtime_telemetry.py`, `tests/test_qsg_adapter_unification.py`, `tests/test_native_qsg_engine.py`
- Telemetry contract: TTFT delta, TPOT delta, accepted-length delta, cache-fragmentation delta, topology-jitter delta, quality delta.
- Required evidence: per-hotspot before/after capsules for Granite tiny, Qwen 4B, Qwen 9B.
- Rollback criteria: any optimization lacks capsule evidence or regresses quality/stability outside declared budget.
- Promotion gate: each optimization shows a model-specific win and no unbounded regressions.
- Success criteria: the program produces measured, attributable gains rather than anecdotal ones.
- Exact wiring points: `core/native/qsg_parallel_kernels.cpp`, `core/native/quantized_matmul.cpp`, `core/native/amx_kernels.cpp`, `core/native/interleave_weights.cpp`, `core/native/native_kv_cache.cpp`, `core/qsg/ollama_adapter.py`, `core/model/model_profile.py`
- Deliverables: hotspot fixes, model-family policy updates, benchmark deltas, rollback docs.
- Tests: `tests/test_qsg_adapter_unification.py`, `tests/test_runtime_telemetry.py`, `tests/test_native_qsg_engine.py`, `tests/test_native_parallel_generation_engine.py`
- Verification commands: `pytest tests/test_qsg_adapter_unification.py tests/test_runtime_telemetry.py tests/test_native_qsg_engine.py tests/test_native_parallel_generation_engine.py -q`; `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: measured gains are stable enough to codify.

### Phase 6
- `phase_id`: `convergence`
- Phase title: Proof-Carrying Optimization Governance
- Objective: Make every future CPU/math optimization machine-auditable and roadmap-governed.
- Dependencies: `development`
- Repo scope: `saguaro/roadmap`, `saguaro/validation`, `standards/traceability`, `.anvil/validation`
- Owning specialist type: `TraceabilityArchitect`
- Allowed writes: `saguaro/roadmap/*.py`, `saguaro/validation/*.py`, `standards/traceability/*`, `.anvil/validation/*`
- Telemetry contract: capsule completeness, stale evidence count, blocked promotion count.
- Required evidence: completion graph, hotspot capsules, roadmap validation report.
- Rollback criteria: governance cost exceeds operational value.
- Promotion gate: all promoted hotspots and model policies have capsules and validation state.
- Success criteria: the subsystem can be operated and improved with confidence rather than folklore.
- Exact wiring points: `saguaro/roadmap/validator.py`, `saguaro/validation/engine.py`, `standards/traceability/TRACEABILITY.jsonl`
- Deliverables: hotspot capsule schema, validation rules, roadmap gates.
- Tests: `tests/test_saguaro_roadmap_validator.py`, `tests/test_saguaro_validate_docs.py`, `tests/test_saguaro_traceability.py`
- Verification commands: `pytest tests/test_saguaro_roadmap_validator.py tests/test_saguaro_validate_docs.py tests/test_saguaro_traceability.py -q`; `./venv/bin/saguaro roadmap validate --path inventive_research_roadmap.md --format json`; `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: CPU/math roadmap status is machine-checkable and evidence-complete.

## 9. Implementation Contract
- The system shall establish a gold-corpus verification lane for `saguaro/math/engine.py` and `saguaro/math/pipeline.py`, tested by `tests/test_saguaro_math.py`, `tests/test_saguaro_math_precision.py`, and `tests/test_saguaro_math_pipeline.py`, and verified with `pytest tests/test_saguaro_math.py tests/test_saguaro_math_precision.py tests/test_saguaro_math_pipeline.py -q`, `./venv/bin/saguaro math parse --path core/native --format json`, and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall bound native-versus-fallback CPU scan drift through `saguaro/cpu/model.py`, `saguaro/cpu/cache.py`, `saguaro/cpu/register_pressure.py`, `saguaro/cpu/roofline.py`, and `saguaro/cpu/vectorization.py`, tested by `tests/test_saguaro_cpu_scan.py`, `tests/test_saguaro_cpu_topology.py`, and `tests/test_saguaro_cpu_twin.py`, and verified with `pytest tests/test_saguaro_cpu_scan.py tests/test_saguaro_cpu_topology.py tests/test_saguaro_cpu_twin.py -q`, `./venv/bin/saguaro cpu scan --path core/native --arch x86_64-avx2 --format json`, and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall recover native runtime symbol and build-target closure through `domains/verification/auto_verifier.py`, `saguaro/parsing/runtime_symbols.py`, `saguaro/math/pipeline.py`, and `core/native/CMakeLists.txt`, tested by `tests/test_auto_verifier_qsg_runtime.py` and `tests/test_saguaro_math_pipeline.py`, and verified with `pytest tests/test_auto_verifier_qsg_runtime.py tests/test_saguaro_math_pipeline.py -q`, `./venv/bin/saguaro health`, and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall create a three-model runtime atlas for `granite4:tiny-h`, `qwen3.5:4b`, and `qwen3.5:9b` through `core/model/chat_templates.py`, `core/model/model_profile.py`, `core/qsg/ollama_adapter.py`, and `core/ollama_client.py`, tested by `tests/test_qsg_adapter_unification.py` and `tests/test_runtime_telemetry.py`, and verified with `pytest tests/test_qsg_adapter_unification.py tests/test_runtime_telemetry.py -q`, `./venv/bin/saguaro build-graph`, and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall correlate static hotspot priority with real runtime evidence through `saguaro/cpu/model.py`, `saguaro/cpu/roofline.py`, `core/native/runtime_telemetry.py`, `benchmarks/native_qsg_benchmark.py`, and `audit/runner/benchmark_suite.py`, tested by `tests/test_saguaro_cpu_twin.py`, `tests/test_runtime_telemetry.py`, and `tests/test_native_qsg_benchmark.py`, and verified with `pytest tests/test_saguaro_cpu_twin.py tests/test_runtime_telemetry.py tests/test_native_qsg_benchmark.py -q`, `./venv/bin/saguaro cpu scan --path core/native --arch x86_64-avx2 --format json`, and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall optimize the highest-value native hotspots through `core/native/qsg_parallel_kernels.cpp`, `core/native/quantized_matmul.cpp`, `core/native/amx_kernels.cpp`, `core/native/interleave_weights.cpp`, `core/native/native_kv_cache.cpp`, and `core/qsg/ollama_adapter.py`, tested by `tests/test_native_qsg_engine.py`, `tests/test_native_parallel_generation_engine.py`, `tests/test_qsg_adapter_unification.py`, and `tests/test_runtime_telemetry.py`, and verified with `pytest tests/test_native_qsg_engine.py tests/test_native_parallel_generation_engine.py tests/test_qsg_adapter_unification.py tests/test_runtime_telemetry.py -q` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall encode proof-carrying optimization governance through `saguaro/roadmap/validator.py`, `saguaro/validation/engine.py`, and `standards/traceability/TRACEABILITY.jsonl`, tested by `tests/test_saguaro_roadmap_validator.py`, `tests/test_saguaro_validate_docs.py`, and `tests/test_saguaro_traceability.py`, and verified with `pytest tests/test_saguaro_roadmap_validator.py tests/test_saguaro_validate_docs.py tests/test_saguaro_traceability.py -q`, `./venv/bin/saguaro roadmap validate --path inventive_research_roadmap.md --format json`, and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
