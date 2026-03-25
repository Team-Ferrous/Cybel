# Saguaro CPU + Math Architecture Roadmap

## 1. First-Principles Framing

- Repo-grounded observation: Saguaro is already two systems at once. `saguaro/parsing/parser.py` is the structural front-end, and `saguaro/math/engine.py` is a lightweight mathematical extractor layered beside it rather than inside it.
- Repo-grounded observation: Anvil is trying to become an agentic engineering runtime, not a document search tool. For math and kernel work, that means the agent needs a static performance and complexity model before it reaches for a live benchmark.
- Repo-grounded observation: `./venv/bin/saguaro health` reports 92.2% structural coverage but only 8.2% graph coverage, with missing `cfg`, `dfg`, `call`, and `ffi_bridge` edges. That is the exact deficit that blocks trustworthy math-to-pipeline reasoning.
- Repo-grounded observation: `./venv/bin/saguaro math parse --path saguaro --format json` produced 18,435 "equations" across 622 files. The top hit was a parser configuration dictionary in `saguaro/parsing/parser.py` with structural score 9312. That is not acceptable precision for pre-benchmark engineering.
- Repo-grounded observation: `saguaro/math/engine.py` only scans Markdown, Python, and C/C++ family files, while `saguaro/parsing/parser.py` already detects dozens of languages and `tests/test_saguaro_parser_languages.py` exercises them.
- Repo-grounded observation: `core/native/runtime_telemetry.py` already carries calibration-grade CPU signals such as `native_isa_baseline`, `l3_domain_count`, `l3_domain_ids_active`, `perf_event_access`, CPU masks, `autotune_profile_id`, and `hot_path_proof`.
- Repo-grounded observation: `core/simd/common/perf_utils.h` already exposes reusable locality primitives: `PrefetchT0`, `PrefetchT1`, `PrefetchT2`, `PrefetchNTA`, `PrefetchW`, `IsCacheLineAligned`, `IsSimdAligned`, `RoundUpToCacheLine`, `PrefetchMatrixRow`, and `PrefetchStreaming`.
- Inference: the right target is not "math extraction that finds formulas." The right target is a repo-scale static observability layer that can answer four questions before runtime:
  1. Where is the real math?
  2. What data movement shape does it imply?
  3. What CPU hazards does that shape suggest on AVX2, AVX-512, and NEON class machines?
  4. Which hotspots deserve expensive benchmark time?
- Engineering standard: NASA and Formula 1 discipline means precision before breadth, explicit evidence trails, calibration against runtime telemetry, deterministic failure modes, and no silent blind spots by language or architecture.

## 2. External Research Scan

- [R1] LLVM MLIR Affine dialect: <https://mlir.llvm.org/docs/Dialects/Affine/>. Why it matters here: it treats loop bounds, memory access maps, and affine transformations as first-class objects. Saguaro does not need full MLIR, but it should steal the idea of a typed intermediate form for loops, indices, and accesses rather than staying regex-only.
- [R2] Halide: <https://halide-lang.org/>. Why it matters here: Halide's core insight is algorithm/schedule separation. Saguaro should extract the "algorithm" from code and infer candidate "schedules" statically, even when the user wrote raw C++, Python, or tensor code.
- [R3] Ansor, OSDI 2020: <https://www.usenix.org/conference/osdi20/presentation/zheng>. Why it matters here: the important lesson is not auto-tuning alone; it is the search-space formulation. Saguaro needs a counterfactual schedule space for static pre-benchmark proposals.
- [R4] Triton programming guide: <https://triton-lang.org/main/programming-guide/chapter-1/introduction.html>. Why it matters here: Triton frames performance around blocked tensor programs mapped to hardware. Saguaro should emit blocked access signatures and vector-width-aware recommendations, not generic "optimize loop" advice.
- [R5] LLVM auto-vectorization docs: <https://llvm.org/docs/Vectorizers.html>. Why it matters here: legality and profitability are separate. Saguaro should report both. Missing alignment or alias information is a different class of problem than a loop that is legal but not worthwhile to vectorize.
- [R6] `llvm-mca`: <https://llvm.org/docs/CommandGuide/llvm-mca.html>. Why it matters here: it models throughput, latency, and resource pressure statically from machine code. Saguaro can use the same mental model one layer higher, at source and IR level, to predict CPU bottlenecks before compilation and benchmarking.
- [R7] OSACA: <https://github.com/RRZE-HPC/OSACA>. Why it matters here: OSACA treats port pressure and throughput as analyzable artifacts. Saguaro should borrow its "architecture-specific analytic report" posture for AVX2/AVX-512/NEON-aware advisory output.
- [R8] `pycachesim`: <https://github.com/RRZE-HPC/pycachesim>. Why it matters here: explicit cache hierarchy simulation is feasible and reusable. Saguaro does not need cycle-accurate simulation, but it should build a symbolic cache-line and reuse-distance estimator that can later be calibrated against measured runs.
- [R9] CANAL, cache timing analysis for LLVM: <https://arxiv.org/abs/1805.05131>. Why it matters here: static cache reasoning can be pushed through compiler infrastructure for timing and side-channel analysis. That validates the basic premise that locality analysis does not have to wait for live benchmarking.
- [R10] Intel Intrinsics Guide: <https://software.intel.com/sites/landingpage/IntrinsicsGuide/>. Why it matters here: AVX2 and AVX-512 legality, lane width, alignment, and instruction families are concrete and enumerable. Saguaro should encode these as architecture packs, not prose.
- [R11] Arm intrinsics reference: <https://developer.arm.com/architectures/instruction-sets/intrinsics/>. Why it matters here: NEON is not "AVX but smaller." The advisor must reason in architecture-specific terms rather than generic SIMD optimism.
- [R12] Time-Based Roofline model: <https://www.cs.utexas.edu/~flame/pubs/TOPC_Storage.pdf>. Why it matters here: the useful idea is not the chart but the decomposition of compute-bound versus bandwidth-bound behavior using operational intensity. Saguaro should estimate operational intensity statically from access signatures and arithmetic density.
- [R13] Polygeist: <https://github.com/llvm/Polygeist>. Why it matters here: lowering C/C++ into MLIR-style IR is a practical pattern for preserving loop and access semantics. Saguaro should mirror that direction conceptually inside its parser-to-math pipeline.
- [R14] Practitioner signal on tree-sitter at scale: <https://www.reddit.com/r/ProgrammingLanguages/comments/12g6uou/tree_sitter_is_amazing_for_incremental_parsing/>. Why it matters here: incremental parsing is powerful, but practitioners regularly hit grammar and large-file edge cases. Because Saguaro already leans on tree-sitter, the roadmap must include parser-quality telemetry and fallback logic rather than assuming AST coverage means semantic correctness.

## 3. Repo Grounding Summary

### 3.1 Inspected code paths

- Repo-grounded observation: `saguaro/parsing/parser.py` via `saguaro agent skeleton` and `saguaro agent slice`:
  - `SAGUAROParser.parse_file`
  - `SAGUAROParser._detect_language`
  - `SAGUAROParser._build_dependency_payload`
- Repo-grounded observation: `saguaro/math/engine.py` via targeted reads:
  - `MathEngine.parse`
  - `_extract_code_equations`
  - `_score_expression`
  - `_iter_code_statements`
  - `_looks_like_math_statement`
- Repo-grounded observation: user-facing seams:
  - `saguaro/api.py`
  - `saguaro/cli.py`
  - `tools/saguaro_tools.py`
  - `domains/code_intelligence/saguaro_substrate.py`
- Repo-grounded observation: roadmap and traceability seams:
  - `core/campaign/roadmap_compiler.py`
  - `core/campaign/phase_packet.py`
  - `core/campaign/roadmap_validator.py`
  - `saguaro/roadmap/validator.py`
  - `saguaro/requirements/traceability.py`
- Repo-grounded observation: CPU and kernel seams:
  - `core/simd/common/perf_utils.h`
  - `core/native/runtime_telemetry.py`
  - `core/native/CMakeLists.txt`
  - `benchmarks/simd_benchmark.py`
  - `audit/runner/benchmark_suite.py`
  - `audit/runner/native_benchmark_runner.py`

### 3.2 Strong existing primitives

- Repo-grounded observation: parser language detection is already broad. `tests/test_saguaro_parser_languages.py` proves structural detection for Zig, Nim, HDL, Fortran, Pascal, Ada, COBOL, Solidity, R, Julia, Clojure, Elixir, Erlang, Haskell, OCaml, Assembly, MATLAB, Scala, Jinja, LaTeX, HCL, QML, Emacs Lisp, and WAT.
- Repo-grounded observation: the math subsystem already exists and is wired through `saguaro/api.py:1196-1202` and `saguaro/cli.py:1714-1725`, so this roadmap should harden an existing feature rather than inventing a parallel one.
- Repo-grounded observation: the roadmap validator already prefers `Implementation Contract` sections and already knows how to trace roadmap code refs, test refs, graph refs, and verification refs.
- Repo-grounded observation: runtime telemetry already contains calibration hooks that most static analyzers wish they had.
- Repo-grounded observation: the agent substrate and tool facade already expose query, slice, impact, verify, health, and report. New math/CPU functionality can route through those same surfaces instead of adding an unrelated UX island.

### 3.3 Underexploited or thin areas

- Repo-grounded observation: `saguaro/math/engine.py` is disconnected from `saguaro/parsing/parser.py`. The parser knows many languages; the math engine ignores almost all of them.
- Repo-grounded observation: the math extractor is currently too permissive. It flags:
  - preprocessor guards in `core/simd/common/perf_utils.h` as mathematical records,
  - fixture strings in `tests/test_saguaro_parser_languages.py` as high-complexity equations,
  - parser backend dictionaries in `saguaro/parsing/parser.py` as the most complex "math" in the repo.
- Repo-grounded observation: graph confidence is weak where this roadmap most needs it. `saguaro health` reports zero `cfg`, `dfg`, `call`, and `ffi_bridge` coverage.
- Repo-grounded observation: `./venv/bin/saguaro query "math engine complexity scoring equation mapping" --k 6 --json` surfaced governance JSON instead of `saguaro/math/engine.py`. Query ranking is not yet math-aware.
- Repo-grounded observation: `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json` passes cleanly, which means current governance does not encode math-precision or CPU-analysis blind spots.

### 3.4 Health and verification results

- `./venv/bin/saguaro health`
  - Structural coverage: 92.2%
  - AST coverage: 88.2%
  - Graph coverage: 8.2%
  - Missing graph edge classes: `cfg`, `dfg`, `call`, `ffi_bridge`
- `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
  - Status: `pass`
  - Violations: `0`
  - Inference: verification is healthy for current governance, but it is not testing the math/CPU property class that this roadmap needs.
- `./venv/bin/saguaro math parse --path core/simd/common/perf_utils.h --format json`
  - 21 records
  - False positives from `#if`/`#elif`
  - High-complexity highlight landed on control flow inside `SafeByteSize`, not on data-movement semantics
- `./venv/bin/saguaro math parse --path tests/test_saguaro_parser_languages.py --format json`
  - 42 records
  - Most are fixture strings and test scaffolding, not mathematical intent

## 4. Hidden Assumptions

1. Math extraction equals formula extraction.
2. Cross-language support at the parser layer automatically implies cross-language support at the math layer.
3. Big-O is an adequate proxy for CPU performance in numerically heavy kernels.
4. SIMD advice can be architecture-agnostic.
5. FFI boundaries are bookkeeping rather than performance structure.
6. Benchmark time is the first place to discover hotspots rather than the last place to confirm them.
7. False positives in static math detection are mostly harmless. They are not; they destroy ranking quality and trust.
8. Query relevance will improve automatically once the index is larger. It usually gets worse without domain priors.
9. A static CPU advisor should wait for full CFG and DFG coverage. It should start earlier with explicit confidence bands and then improve as graph coverage grows.
10. Health and governance should validate only repo hygiene. For Saguaro, they should also validate observability fidelity.

## 5. Candidate Implementation Phases

Legend:

- `[P]` practical
- `[M]` moonshot
- "External inspiration" points at research or tooling above
- "Why it fits" is repo-grounded

### 5.1 Math IR V2 Precision Gate [P]

- Suggested `phase_id`: `analysis_upgrade`
- Core insight: replace regex-classified equations with typed math statements anchored to parser-derived statement roles.
- External inspiration or analogy: [R1], [R13]
- Why it fits Saguaro and Anvil specifically: `saguaro/math/engine.py` already owns cache generation, while `saguaro/parsing/parser.py` already knows far more syntax than the math engine uses.
- Exact wiring points: `saguaro/math/engine.py`, `saguaro/parsing/parser.py`, `saguaro/api.py`, `saguaro/cli.py`
- Existing primitives it can reuse: `MathEngine.parse`, `MathEngine._iter_code_statements`, `SAGUAROParser.parse_file`, `.saguaro/math/cache.json`
- New primitive, data flow, or subsystem needed: `MathIRRecord` with `statement_role`, `operation_family`, `data_type_hints`, `loop_context`, and `false_positive_reason`

```yaml
phase_id: analysis_upgrade
objective: Replace permissive equation scraping with typed mathematical statement extraction and explicit false-positive suppression.
repo_scope: [saguaro/math, saguaro/parsing, saguaro/api.py, saguaro/cli.py]
owning_specialist_type: MathSystemsArchitect
allowed_writes: [saguaro/math/engine.py, saguaro/math/ir.py, saguaro/parsing/parser.py, saguaro/api.py, saguaro/cli.py]
telemetry_contract:
  minimum: [files_scanned, records_emitted, false_positive_rate, unsupported_language_count, parse_seconds]
required_evidence: [saguaro/math/engine.py, saguaro/parsing/parser.py, tests/test_saguaro_math.py, tests/test_saguaro_parser_languages.py]
rollback_criteria: [math_precision_regresses, math_record_count_explodes_without_recall_gain]
promotion_gate: {false_positive_rate_delta: "<=-60%", recall_on_curated_fixtures: ">=0.95"}
success_criteria: [non_math_assignments_suppressed, parser_config_literals_not_ranked_as_math, cache_schema_versioned]
dependencies: []
```

- Why this creates value: trustable math results become possible.
- Why this creates moat: high-precision math extraction across mixed-source repos is hard and sticky.
- Main risk or failure mode: over-pruning real math in scientific code.
- Smallest credible first experiment: harden `tests/test_saguaro_math.py` with explicit negative fixtures from `tests/test_saguaro_parser_languages.py`.
- Confidence level: high

### 5.2 Language Coverage Parity for Math [P]

- Suggested `phase_id`: `research`
- Core insight: math coverage must inherit parser coverage, not lag it by years.
- External inspiration or analogy: [R13], [R14]
- Why it fits Saguaro and Anvil specifically: the parser already detects more languages than the math engine scans.
- Exact wiring points: `saguaro/parsing/parser.py`, `saguaro/math/engine.py`, `tests/test_saguaro_parser_languages.py`
- Existing primitives it can reuse: `_detect_language`, lightweight structural parsing, parser language tests
- New primitive, data flow, or subsystem needed: language-family math packs with per-language statement-start and operator rules

```yaml
phase_id: research
objective: Extend math extraction to every language family that Saguaro can already detect structurally, starting with Fortran, MATLAB, Julia, R, Rust, Go, Java, Scala, Solidity, and HDL.
repo_scope: [saguaro/math, saguaro/parsing, tests]
owning_specialist_type: MultilingualStaticAnalysisLead
allowed_writes: [saguaro/math/engine.py, saguaro/math/languages.py, tests/test_saguaro_math_languages.py, tests/test_saguaro_parser_languages.py]
telemetry_contract:
  minimum: [language_family_coverage, unsupported_language_count, records_by_language, precision_by_language]
required_evidence: [saguaro/parsing/parser.py, tests/test_saguaro_parser_languages.py]
rollback_criteria: [language_addition_without_precision_budget, fallback_to_unknown_for_supported_syntax]
promotion_gate: {target_language_families: ">=10", per_language_precision: ">=0.85"}
success_criteria: [math_engine_extension_list_matches_parser_coverage_policy, unsupported_languages_reported_explicitly]
dependencies: [analysis_upgrade]
```

- Why this creates value: scientific repos stop being second-class citizens.
- Why this creates moat: cross-language numerical reasoning is rarer than generic AST parsing.
- Main risk or failure mode: too many language-specific heuristics with no unifying IR.
- Smallest credible first experiment: add Fortran, Julia, and MATLAB first because they are math-dense and parser-detected already.
- Confidence level: high

### 5.3 Loop and Recurrence Lifter [M]

- Suggested `phase_id`: `analysis_upgrade`
- Core insight: equations without loop context are almost useless for performance reasoning.
- External inspiration or analogy: [R1], [R9]
- Why it fits Saguaro and Anvil specifically: graph coverage is weak, so an explicit loop/recurrence lifter is the shortest path to useful structure.
- Exact wiring points: `saguaro/math/engine.py`, `saguaro/parsing/parser.py`, `saguaro/omnigraph/store.py`
- Existing primitives it can reuse: parser structural entities, dependency payloads, math record cache
- New primitive, data flow, or subsystem needed: `LoopFrame` and `RecurrenceFrame` attached to each `MathIRRecord`

```yaml
phase_id: analysis_upgrade
objective: Lift loops, recurrences, and reduction patterns into explicit math metadata so complexity and locality analysis attach to execution structure rather than isolated statements.
repo_scope: [saguaro/math, saguaro/parsing, saguaro/omnigraph]
owning_specialist_type: ProgramAnalysisResearchLead
allowed_writes: [saguaro/math/engine.py, saguaro/math/ir.py, saguaro/omnigraph/store.py, tests/test_saguaro_math_loops.py]
telemetry_contract:
  minimum: [loop_frame_count, reduction_count, recurrence_count, loop_nesting_depth_histogram]
required_evidence: [saguaro/math/engine.py, saguaro/parsing/parser.py]
rollback_criteria: [loop_frames_unstable_across_reparse, reductions_misclassified_as_assignments]
promotion_gate: {reduction_fixture_recall: ">=0.9", nested_loop_precision: ">=0.85"}
success_criteria: [loop_context_attached_to_hot_math_records, recurrence_patterns_searchable]
dependencies: [analysis_upgrade]
```

- Why this creates value: it converts raw math into performance-relevant math.
- Why this creates moat: recurrence-aware static analysis is much harder to copy than regex extraction.
- Main risk or failure mode: ambiguous loop ownership in macros and generated code.
- Smallest credible first experiment: extract reductions and recurrence updates from `saguaro/native/ops/*.h`.
- Confidence level: medium

### 5.4 Tensor Access Signature Emitter [P]

- Suggested `phase_id`: `feature_map`
- Core insight: locality and roofline work need access signatures, not only expressions.
- External inspiration or analogy: [R2], [R12]
- Why it fits Saguaro and Anvil specifically: the kernel corpus is rich in array-style and tensor-style updates; access signatures can be emitted without full compiler lowering.
- Exact wiring points: `saguaro/math/engine.py`, `saguaro/math/pipeline.py`, `core/simd/common/perf_utils.h`
- Existing primitives it can reuse: loop frames, symbol extraction, complexity scoring
- New primitive, data flow, or subsystem needed: `AccessSignature` with `base_symbol`, `index_affinity`, `stride_class`, `reuse_hint`, `write_mode`

```yaml
phase_id: feature_map
objective: Emit typed access signatures for arrays, tensors, and views so downstream CPU modeling can reason about stride, reuse, and alias-sensitive writes.
repo_scope: [saguaro/math, saguaro/cpu, core/simd]
owning_specialist_type: PerformanceModelEngineer
allowed_writes: [saguaro/math/pipeline.py, saguaro/math/engine.py, saguaro/cpu/model.py, tests/test_saguaro_math_access.py]
telemetry_contract:
  minimum: [access_signature_count, contiguous_ratio, strided_ratio, indirect_ratio]
required_evidence: [core/simd/common/perf_utils.h, saguaro/math/engine.py]
rollback_criteria: [access_signature_noise_exceeds_budget, stride_classes_unstable]
promotion_gate: {access_signature_precision: ">=0.85", array_hotspot_coverage: ">=0.8"}
success_criteria: [top_math_records_emit_access_signatures, tensor_reads_and_writes_distinguished]
dependencies: [analysis_upgrade]
```

- Why this creates value: it unlocks CPU analysis without waiting for machine code.
- Why this creates moat: access-shape extraction is the substrate for many future features.
- Main risk or failure mode: pointer aliasing ambiguity.
- Smallest credible first experiment: classify contiguous versus strided accesses in `core/simd/common/perf_utils.h` and selected `saguaro/native/ops/*.h`.
- Confidence level: high

### 5.5 FFI Math Provenance Bridge [P]

- Suggested `phase_id`: `feature_map`
- Core insight: agentic benchmark preflight is weak unless math records know which Python call sites land in which native kernels.
- External inspiration or analogy: flight-test instrumentation chains; [R6]
- Why it fits Saguaro and Anvil specifically: `saguaro health` already reports 100% FFI boundary coverage, but graph edges are absent where they matter most.
- Exact wiring points: `saguaro/math/pipeline.py`, `saguaro/omnigraph/store.py`, `core/native/CMakeLists.txt`, `saguaro/health.py`
- Existing primitives it can reuse: build graph, FFI boundary coverage metrics, omni-graph store
- New primitive, data flow, or subsystem needed: `MathProvenanceEdge` linking Python/API node -> FFI boundary -> native kernel -> build target

```yaml
phase_id: feature_map
objective: Attach each high-value math record to the function, module, FFI boundary, and build target that actually executes it.
repo_scope: [saguaro/math, saguaro/omnigraph, saguaro/health.py, core/native/CMakeLists.txt]
owning_specialist_type: FFIObservabilityLead
allowed_writes: [saguaro/math/pipeline.py, saguaro/omnigraph/store.py, saguaro/health.py, tests/test_saguaro_math_pipeline.py]
telemetry_contract:
  minimum: [ffi_math_edge_count, unresolved_kernel_binding_count, build_target_binding_count]
required_evidence: [core/native/CMakeLists.txt, saguaro/health.py, saguaro/omnigraph/store.py]
rollback_criteria: [ffi_bindings_incorrect_or_opaque, build_target_mappings_regress]
promotion_gate: {ffi_binding_coverage: ">=0.8", unresolved_kernel_binding_count: "<=5% of hot records"}
success_criteria: [math_records_trace_to_kernels_and_targets, agent_can_request_pipeline_provenance]
dependencies: [feature_map]
```

- Why this creates value: it turns "math found" into "math located in the pipeline."
- Why this creates moat: cross-boundary provenance is hard and operationally valuable.
- Main risk or failure mode: ambiguous bindings in generated wrappers.
- Smallest credible first experiment: trace selected `saguaro/native/ops/*.h` math records to owning targets from `core/native/CMakeLists.txt`.
- Confidence level: medium-high

### 5.6 CPU Architecture Pack Catalog [M]

- Suggested `phase_id`: `research`
- Core insight: CPU advice must be architecture-pack driven, not one-size-fits-all.
- External inspiration or analogy: [R7], [R10], [R11]
- Why it fits Saguaro and Anvil specifically: runtime telemetry already carries the calibration fields needed to ground pack selection.
- Exact wiring points: `saguaro/cpu/topology.py`, `core/native/runtime_telemetry.py`, `saguaro/api.py`, `saguaro/cli.py`
- Existing primitives it can reuse: telemetry fields for ISA and L3 domains, `hot_path_proof`
- New primitive, data flow, or subsystem needed: architecture packs for `x86_64-avx2`, `x86_64-avx512`, `arm64-neon`, later Sapphire Rapids and Graviton subclasses

```yaml
phase_id: research
objective: Build explicit CPU architecture packs that parameterize vector width, preferred alignment, cache assumptions, prefetch policy, and warning thresholds.
repo_scope: [saguaro/cpu, core/native/runtime_telemetry.py, saguaro/api.py, saguaro/cli.py]
owning_specialist_type: CPUMicroarchitectureLead
allowed_writes: [saguaro/cpu/topology.py, saguaro/api.py, saguaro/cli.py, tests/test_saguaro_cpu_topology.py, tests/test_runtime_telemetry.py]
telemetry_contract:
  minimum: [arch_pack_selected, pack_source, isa_match_state, l3_domain_count, perf_event_access]
required_evidence: [core/native/runtime_telemetry.py, tests/test_runtime_telemetry.py]
rollback_criteria: [architecture_pack_misidentification, advice_not_parameterized_by_arch]
promotion_gate: {supported_arch_packs: ">=3", telemetry_pack_match_rate: ">=0.9"}
success_criteria: [cpu_reports_are_arch_specific, static_advice_differs_between_avx2_avx512_neon_when_it_should]
dependencies: [analysis_upgrade]
```

- Why this creates value: same code can be optimal on one ISA and wrong on another.
- Why this creates moat: architecture-aware static advice is much more defensible than generic "SIMD detected."
- Main risk or failure mode: false certainty when runtime topology differs from pack assumptions.
- Smallest credible first experiment: start with three packs only and compare against telemetry from existing runs.
- Confidence level: medium

### 5.7 Cache-Line Risk Estimator [P]

- Suggested `phase_id`: `development`
- Core insight: not all locality analysis needs runtime. A large share is visible from access shape, element width, and block geometry.
- External inspiration or analogy: [R8], [R12]
- Why it fits Saguaro and Anvil specifically: `core/simd/common/perf_utils.h` already codifies cache-line size and prefetch patterns.
- Exact wiring points: `saguaro/cpu/model.py`, `saguaro/cpu/report.py`, `core/simd/common/perf_utils.h`
- Existing primitives it can reuse: `AccessSignature`, cache-line constants, architecture packs
- New primitive, data flow, or subsystem needed: symbolic reuse-distance and cache-line touch estimator

```yaml
phase_id: development
objective: Estimate cache-line pressure, contiguous reuse, and likely L1/L2/L3 stress from static access signatures and architecture packs.
repo_scope: [saguaro/cpu, core/simd/common/perf_utils.h]
owning_specialist_type: CacheModelEngineer
allowed_writes: [saguaro/cpu/model.py, saguaro/cpu/report.py, tests/test_saguaro_cpu_scan.py]
telemetry_contract:
  minimum: [estimated_cache_lines_touched, reuse_distance_class, l1_risk, l2_risk, l3_risk]
required_evidence: [core/simd/common/perf_utils.h, tests/test_saguaro_cpu_scan.py]
rollback_criteria: [risk_scores_non_monotonic, reports_ignore_arch_pack]
promotion_gate: {known_fixture_rank_order_correct: true, hotspot_overlap_with_benchmarks: ">=0.7"}
success_criteria: [cache_risk_report_generated_per_hot_kernel, report_confidence_explicit]
dependencies: [feature_map, research]
```

- Why this creates value: it prunes expensive benchmark search space.
- Why this creates moat: symbolic locality estimation across repo code is uncommon in agent tooling.
- Main risk or failure mode: indirect accesses dominate and the estimator overstates confidence.
- Smallest credible first experiment: emit cache-line estimates for `PrefetchMatrixRow` and contiguous copy kernels.
- Confidence level: high

### 5.8 SIMD Legality Predictor [P]

- Suggested `phase_id`: `development`
- Core insight: the first question is whether vectorization is legal and architecturally sensible, not whether we can imagine a speedup.
- External inspiration or analogy: [R5], [R10], [R11]
- Why it fits Saguaro and Anvil specifically: the repo already exposes explicit SIMD alignment helpers and multiple ISA-specific kernels.
- Exact wiring points: `saguaro/cpu/vectorization.py`, `core/simd/common/perf_utils.h`, `saguaro/query/corpus_rules.py`
- Existing primitives it can reuse: `IsSimdAligned`, architecture packs, access signatures
- New primitive, data flow, or subsystem needed: legality reasons such as alignment unknown, trip count too small, indirect gather, alias hazard, unsupported intrinsic family

```yaml
phase_id: development
objective: Predict SIMD legality and likely profitability for hot loops, with architecture-specific reasons for AVX2, AVX-512, and NEON.
repo_scope: [saguaro/cpu, core/simd/common/perf_utils.h, saguaro/query]
owning_specialist_type: VectorizationArchitect
allowed_writes: [saguaro/cpu/vectorization.py, saguaro/query/corpus_rules.py, tests/test_saguaro_cpu_scan.py]
telemetry_contract:
  minimum: [simd_legal_count, simd_blocked_count, legality_reasons, arch_specific_warnings]
required_evidence: [core/simd/common/perf_utils.h, tests/test_saguaro_cpu_scan.py]
rollback_criteria: [vectorization_blockers_too_generic, architecture_specificity_missing]
promotion_gate: {blocked_loop_explanations_present: ">=90% of blocked cases"}
success_criteria: [reports_explain_why_not_just_what, isa_specific_legality_output_present]
dependencies: [feature_map, research]
```

- Why this creates value: it gives the agent rewriteable reasons.
- Why this creates moat: explanation quality is where trust comes from.
- Main risk or failure mode: no compiler feedback loop to calibrate legality heuristics.
- Smallest credible first experiment: flag loops that lack alignment evidence and compare against explicit aligned helpers in `core/simd/common/perf_utils.h`.
- Confidence level: high

### 5.9 Prefetch Opportunity Classifier [M]

- Suggested `phase_id`: `development`
- Core insight: Saguaro should not only detect existing prefetch usage; it should detect where prefetch is conspicuously absent or wrong for the access pattern.
- External inspiration or analogy: [R7], [R10], [R11]
- Why it fits Saguaro and Anvil specifically: the repo already has canonical prefetch primitives in one reusable header.
- Exact wiring points: `saguaro/cpu/prefetch.py`, `core/simd/common/perf_utils.h`, `saguaro/cpu/report.py`
- Existing primitives it can reuse: `PrefetchT0`, `PrefetchT1`, `PrefetchNTA`, access signatures, architecture packs
- New primitive, data flow, or subsystem needed: prefetch opportunity classes `temporal`, `streaming`, `write-intent`, `none`

```yaml
phase_id: development
objective: Emit architecture-aware prefetch recommendations and anti-recommendations from static access patterns.
repo_scope: [saguaro/cpu, core/simd/common/perf_utils.h]
owning_specialist_type: MemoryHierarchyEngineer
allowed_writes: [saguaro/cpu/prefetch.py, saguaro/cpu/report.py, tests/test_saguaro_cpu_scan.py]
telemetry_contract:
  minimum: [prefetch_opportunity_count, prefetch_misuse_count, streaming_hint_count]
required_evidence: [core/simd/common/perf_utils.h]
rollback_criteria: [prefetch_recommendations_not_arch_specific, opportunity_classifier_conflicts_with_cache_model]
promotion_gate: {prefetch_fixture_precision: ">=0.8"}
success_criteria: [reports_distinguish_temporal_vs_streaming, reuse_hints_attached_to_hotspots]
dependencies: [development]
```

- Why this creates value: it turns an existing utility header into an actionable rule system.
- Why this creates moat: few repo tools reason about prefetch intent statically.
- Main risk or failure mode: recommending prefetch where hardware prefetchers already dominate.
- Smallest credible first experiment: classify `PrefetchMatrixRow` as temporal and `PrefetchStreaming` as non-temporal, then test on analogous kernel patterns.
- Confidence level: medium

### 5.10 Static Roofline Estimator [M]

- Suggested `phase_id`: `eid`
- Core insight: before benchmarking, the system should estimate whether a kernel is probably memory-bound or compute-bound.
- External inspiration or analogy: [R12], [R6]
- Why it fits Saguaro and Anvil specifically: arithmetic density plus access signatures plus CPU packs are enough for a first operational-intensity estimate.
- Exact wiring points: `saguaro/cpu/roofline.py`, `saguaro/cpu/model.py`, `core/native/runtime_telemetry.py`
- Existing primitives it can reuse: complexity scores, access signatures, architecture packs, runtime telemetry calibration
- New primitive, data flow, or subsystem needed: static operational intensity estimator and confidence interval

```yaml
phase_id: eid
objective: Predict a rough roofline position for hot kernels before runtime, then compare that estimate with later benchmark evidence.
repo_scope: [saguaro/cpu, core/native/runtime_telemetry.py]
owning_specialist_type: PerformanceResearchLead
allowed_writes: [saguaro/cpu/roofline.py, saguaro/cpu/report.py, core/native/runtime_telemetry.py, tests/test_saguaro_cpu_twin.py]
telemetry_contract:
  minimum: [operational_intensity_estimate, predicted_bound_class, roofline_confidence, calibration_error]
required_evidence: [core/native/runtime_telemetry.py, benchmarks/simd_benchmark.py]
rollback_criteria: [predicted_bound_class_uncorrelated_with_measurement, confidence_unbounded]
promotion_gate: {predicted_bound_class_accuracy: ">=0.7 on calibration suite"}
success_criteria: [roofline_class_present_in_cpu_report, measured_vs_predicted_delta_stored]
dependencies: [development]
```

- Why this creates value: it helps decide where benchmarking effort belongs.
- Why this creates moat: coupling static and measured roofline classification into an agent workflow is unusual.
- Main risk or failure mode: arithmetic intensity estimated from source alone can be badly wrong on fused kernels.
- Smallest credible first experiment: classify a handful of SIMD benchmark kernels as likely compute-bound or memory-bound and compare against measured throughput curves.
- Confidence level: medium

### 5.11 Register Pressure Surrogate [M]

- Suggested `phase_id`: `eid`
- Core insight: source-level pressure heuristics can still warn about probable spills even without assembly.
- External inspiration or analogy: [R6], [R7]
- Why it fits Saguaro and Anvil specifically: many kernels have wide live ranges, manual vector math, and nested temporaries.
- Exact wiring points: `saguaro/cpu/register_pressure.py`, `saguaro/math/engine.py`, `saguaro/cpu/report.py`
- Existing primitives it can reuse: symbol counts, nesting depth, loop frames, SIMD legality
- New primitive, data flow, or subsystem needed: live-range proxy score and spill-risk class

```yaml
phase_id: eid
objective: Estimate register pressure from source-level live-range proxies and surface probable spill zones in hotspot reports.
repo_scope: [saguaro/cpu, saguaro/math]
owning_specialist_type: MicroarchitectureResearchEngineer
allowed_writes: [saguaro/cpu/register_pressure.py, saguaro/cpu/report.py, tests/test_saguaro_cpu_twin.py]
telemetry_contract:
  minimum: [register_pressure_score, spill_risk_class, temporary_fanout]
required_evidence: [saguaro/math/engine.py]
rollback_criteria: [surrogate_unstable_or_uninterpretable]
promotion_gate: {spill_risk_precision: ">=0.65 on curated kernels"}
success_criteria: [hotspot_report_marks_spill_risk_zones, confidence_band_present]
dependencies: [analysis_upgrade]
```

- Why this creates value: it gives a source-level hint before assembly inspection.
- Why this creates moat: it is a rare feature in repo intelligence systems.
- Main risk or failure mode: false confidence on compiler-optimized code.
- Smallest credible first experiment: rank kernels by temporary fanout and compare against later assembly or perf counter evidence.
- Confidence level: medium-low

### 5.12 Counterfactual Schedule Twin [M]

- Suggested `phase_id`: `eid`
- Core insight: Saguaro should produce not just diagnoses but candidate schedules the benchmark system can validate.
- External inspiration or analogy: [R2], [R3], [R4]
- Why it fits Saguaro and Anvil specifically: the repo already contains rich native kernels plus benchmark infrastructure that can close the loop.
- Exact wiring points: `saguaro/cpu/schedule_twin.py`, `saguaro/cpu/roofline.py`, `audit/runner/benchmark_suite.py`
- Existing primitives it can reuse: access signatures, roofline estimator, runtime telemetry
- New primitive, data flow, or subsystem needed: counterfactual schedule objects such as `tile`, `fuse`, `vector_width`, `prefetch_policy`, `layout_transform`

```yaml
phase_id: eid
objective: Generate a constrained space of schedule alternatives for hot kernels and rank them before benchmark execution.
repo_scope: [saguaro/cpu, audit/runner]
owning_specialist_type: ScheduleSearchArchitect
allowed_writes: [saguaro/cpu/schedule_twin.py, audit/runner/benchmark_suite.py, tests/test_saguaro_cpu_twin.py, tests/audit/test_benchmark_suite.py]
telemetry_contract:
  minimum: [counterfactual_count, top_schedule_score, benchmark_confirmation_rate]
required_evidence: [audit/runner/benchmark_suite.py, core/native/runtime_telemetry.py]
rollback_criteria: [schedule_space_explodes, recommended_schedules_never_win]
promotion_gate: {top3_schedule_contains_measured_winner: ">=0.6 on pilot set"}
success_criteria: [counterfactuals_ranked_and_exported, benchmark_preflight_consumes_top_schedules]
dependencies: [eid]
```

- Why this creates value: it turns Saguaro into a wind tunnel before track time.
- Why this creates moat: counterfactual schedule generation tied to repo evidence is a strong architectural center.
- Main risk or failure mode: search space too large or too fanciful.
- Smallest credible first experiment: support only tile size, vector width, and prefetch policy on a narrow pilot kernel set.
- Confidence level: medium

### 5.13 Vectorization Miss Explainer [M]

- Suggested `phase_id`: `development`
- Core insight: a blocked-vectorization report should read like a principal compiler engineer, not a generic linter.
- External inspiration or analogy: [R5], [R6]
- Why it fits Saguaro and Anvil specifically: the agent needs rewritable, localized reasons to act autonomously.
- Exact wiring points: `saguaro/cpu/vectorization.py`, `saguaro/cli.py`, `saguaro/api.py`, `tools/saguaro_tools.py`
- Existing primitives it can reuse: legality reasons, access signatures, architecture packs
- New primitive, data flow, or subsystem needed: localized miss reports attached to symbols and line ranges

```yaml
phase_id: development
objective: Explain why a loop or kernel is not vectorized in terms precise enough for an agentic rewrite plan.
repo_scope: [saguaro/cpu, saguaro/api.py, saguaro/cli.py, tools/saguaro_tools.py]
owning_specialist_type: CompilerDiagnosticsLead
allowed_writes: [saguaro/cpu/vectorization.py, saguaro/api.py, saguaro/cli.py, tools/saguaro_tools.py, tests/test_saguaro_interface.py]
telemetry_contract:
  minimum: [vectorization_miss_reports, localized_reason_count, rewriteable_reason_ratio]
required_evidence: [tools/saguaro_tools.py, domains/code_intelligence/saguaro_substrate.py]
rollback_criteria: [diagnostics_too_generic_for_agent_use]
promotion_gate: {rewriteable_reason_ratio: ">=0.8"}
success_criteria: [agent_surface_exposes_symbol_level_vectorization_reasons]
dependencies: [development]
```

- Why this creates value: it makes the output operational rather than descriptive.
- Why this creates moat: explanations tuned for agent action are more valuable than raw scores.
- Main risk or failure mode: diagnostic verbosity without decision value.
- Smallest credible first experiment: emit reasons like `alignment_unknown`, `indirect_gather`, `trip_count_small`, `alias_hazard`.
- Confidence level: medium-high

### 5.14 Sparse and Irregular Access Detector [M]

- Suggested `phase_id`: `research`
- Core insight: the static model must know when not to trust contiguous assumptions.
- External inspiration or analogy: [R9], [R12]
- Why it fits Saguaro and Anvil specifically: several kernels and routing structures use graph-like, mask-like, or irregular access patterns.
- Exact wiring points: `saguaro/cpu/model.py`, `saguaro/math/pipeline.py`, `core/simd/hnsw_moe_router_op.h`
- Existing primitives it can reuse: access signatures, loop frames
- New primitive, data flow, or subsystem needed: irregularity score based on indirect indices, masks, gathers, and branch-coupled accesses

```yaml
phase_id: research
objective: Detect when static locality assumptions are unsafe because access is sparse, masked, graph-like, or indirect.
repo_scope: [saguaro/cpu, saguaro/math, core/simd]
owning_specialist_type: SparseKernelResearchLead
allowed_writes: [saguaro/cpu/model.py, saguaro/math/pipeline.py, tests/test_saguaro_cpu_scan.py]
telemetry_contract:
  minimum: [irregular_access_count, indirect_index_count, low_confidence_hotspot_count]
required_evidence: [core/simd/hnsw_moe_router_op.h, core/simd/dijkstra_grammar_pruner_op.h]
rollback_criteria: [irregularity_score_not_explainable]
promotion_gate: {indirect_access_detection_precision: ">=0.8"}
success_criteria: [reports_downgrade_confidence_for_irregular_patterns]
dependencies: [feature_map]
```

- Why this creates value: it protects the system from elegant but wrong locality claims.
- Why this creates moat: confidence-aware static modeling is more trustworthy.
- Main risk or failure mode: too many conservative downgrades.
- Smallest credible first experiment: mark mask-heavy and graph-heavy kernels as low-confidence locality candidates.
- Confidence level: medium

### 5.15 Data Layout Propagation Graph [M]

- Suggested `phase_id`: `feature_map`
- Core insight: math records should track layout changes such as transpose, pack, reshape, view, flatten, and quantized layout shifts.
- External inspiration or analogy: [R1], [R2]
- Why it fits Saguaro and Anvil specifically: many kernels are fast or slow because of layout, not because of individual formulas.
- Exact wiring points: `saguaro/math/pipeline.py`, `saguaro/omnigraph/store.py`, `core/native/runtime_telemetry.py`
- Existing primitives it can reuse: omni-graph, access signatures, runtime telemetry
- New primitive, data flow, or subsystem needed: layout propagation edges and layout state snapshots

```yaml
phase_id: feature_map
objective: Track data-layout transformations through the math pipeline so locality reasoning stays attached to the actual memory format.
repo_scope: [saguaro/math, saguaro/omnigraph, core/native/runtime_telemetry.py]
owning_specialist_type: DataLayoutArchitect
allowed_writes: [saguaro/math/pipeline.py, saguaro/omnigraph/store.py, tests/test_saguaro_math_pipeline.py]
telemetry_contract:
  minimum: [layout_transition_count, unresolved_layout_state_count]
required_evidence: [saguaro/omnigraph/store.py, core/native/runtime_telemetry.py]
rollback_criteria: [layout_states_not_traceable_or_not_actionable]
promotion_gate: {layout_state_resolution_rate: ">=0.75"}
success_criteria: [layout_changes_visible_in_pipeline_report]
dependencies: [feature_map]
```

- Why this creates value: layout is where many real wins live.
- Why this creates moat: layout-aware static lineage is hard to reconstruct after the fact.
- Main risk or failure mode: symbolic layout states become too abstract.
- Smallest credible first experiment: track only contiguous, transposed, blocked, and quantized layouts first.
- Confidence level: medium

### 5.16 Assembly and Intrinsic Linker [M]

- Suggested `phase_id`: `development`
- Core insight: for the hottest kernels, source-only analysis should optionally descend to intrinsic and assembly signatures.
- External inspiration or analogy: [R6], [R7], [R10], [R11]
- Why it fits Saguaro and Anvil specifically: the repo already contains explicit intrinsics and native kernels.
- Exact wiring points: `saguaro/cpu/intrinsics.py`, `core/native/CMakeLists.txt`, `benchmarks/native_kernel_microbench.py`
- Existing primitives it can reuse: build graph, architecture packs
- New primitive, data flow, or subsystem needed: symbol-to-intrinsic family mapper and optional assembly evidence attachment

```yaml
phase_id: development
objective: Attach intrinsic-family and optional assembly signatures to the hottest kernels so CPU advice can be validated at a lower level when needed.
repo_scope: [saguaro/cpu, core/native/CMakeLists.txt, benchmarks]
owning_specialist_type: NativePerformanceEngineer
allowed_writes: [saguaro/cpu/intrinsics.py, tests/test_saguaro_cpu_scan.py, tests/test_performance_optimizations.py]
telemetry_contract:
  minimum: [intrinsic_family_count, assembly_evidence_count, unresolved_hot_symbol_count]
required_evidence: [core/native/CMakeLists.txt, benchmarks/native_kernel_microbench.py]
rollback_criteria: [assembly_linking_fragile_or_unmaintainable]
promotion_gate: {hot_symbol_intrinsic_resolution_rate: ">=0.7"}
success_criteria: [hottest_kernels_show_intrinsic_or_lowering_evidence]
dependencies: [development]
```

- Why this creates value: it allows selective descent only where static ambiguity remains.
- Why this creates moat: source-to-intrinsic linking is high-value and nontrivial.
- Main risk or failure mode: build-specific assembly paths are brittle.
- Smallest credible first experiment: support intrinsic family mapping even before full assembly capture.
- Confidence level: medium-low

### 5.17 CPU Hotspot Report and Query Surface [P]

- Suggested `phase_id`: `development`
- Core insight: the agent needs one report and one query surface, not five sidecars.
- External inspiration or analogy: [R6], [R7]
- Why it fits Saguaro and Anvil specifically: current query ranking does not surface math internals well, and tool surfaces already exist for query/slice/report.
- Exact wiring points: `saguaro/api.py`, `saguaro/cli.py`, `tools/saguaro_tools.py`, `domains/code_intelligence/saguaro_substrate.py`, `saguaro/services/platform.py`
- Existing primitives it can reuse: CLI/API patterns, substrate command routing, report surfaces
- New primitive, data flow, or subsystem needed: `saguaro cpu scan` and `saguaro cpu report` commands plus query priors for `math`, `kernel`, `simd`, `cache`

```yaml
phase_id: development
objective: Expose the new math and CPU analysis through the same API, CLI, tool, and substrate surfaces the agent already uses.
repo_scope: [saguaro/api.py, saguaro/cli.py, tools/saguaro_tools.py, domains/code_intelligence/saguaro_substrate.py, saguaro/services/platform.py]
owning_specialist_type: AgentSurfaceArchitect
allowed_writes: [saguaro/api.py, saguaro/cli.py, tools/saguaro_tools.py, domains/code_intelligence/saguaro_substrate.py, saguaro/services/platform.py, tests/test_saguaro_interface.py]
telemetry_contract:
  minimum: [cpu_scan_calls, math_parse_calls, report_latency_ms, query_hit_rate_for_hot_math]
required_evidence: [tools/saguaro_tools.py, domains/code_intelligence/saguaro_substrate.py, tests/test_saguaro_interface.py]
rollback_criteria: [new_analysis_not_reachable_from_agent_surfaces]
promotion_gate: {agent_surface_end_to_end_pass: true, report_latency_ms: "<=1500"}
success_criteria: [agent_can_request_math_and_cpu_reports_without_fallback_file_reads]
dependencies: [development]
```

- Why this creates value: it makes the work operational for the agent.
- Why this creates moat: unified agent exposure keeps the architecture coherent.
- Main risk or failure mode: CLI proliferation without ranking improvements.
- Smallest credible first experiment: add `cpu scan` to CLI/API and substrate before query-ranking changes.
- Confidence level: high

### 5.18 Benchmark Preflight Gate [P]

- Suggested `phase_id`: `deep_test_audit`
- Core insight: benchmarking should become confirmation, not discovery.
- External inspiration or analogy: wind-tunnel before track, simulation before flight
- Why it fits Saguaro and Anvil specifically: the repo already contains benchmark runners and native microbench infrastructure.
- Exact wiring points: `audit/runner/benchmark_suite.py`, `audit/runner/native_benchmark_runner.py`, `benchmarks/simd_benchmark.py`, `saguaro/cpu/report.py`
- Existing primitives it can reuse: benchmark suite orchestration, runtime telemetry, proposed CPU report
- New primitive, data flow, or subsystem needed: benchmark preflight contract that requires static hotspot report, confidence, and counterfactual candidates

```yaml
phase_id: deep_test_audit
objective: Require a static math and CPU preflight report before expensive benchmark lanes are executed.
repo_scope: [audit/runner, benchmarks, saguaro/cpu]
owning_specialist_type: BenchmarkControlPlaneArchitect
allowed_writes: [audit/runner/benchmark_suite.py, audit/runner/native_benchmark_runner.py, saguaro/cpu/report.py, tests/audit/test_benchmark_suite.py]
telemetry_contract:
  minimum: [preflight_present, predicted_hotspot_count, benchmark_targets_selected, static_to_measured_overlap]
required_evidence: [audit/runner/benchmark_suite.py, audit/runner/native_benchmark_runner.py, benchmarks/simd_benchmark.py]
rollback_criteria: [benchmark_runner_blocked_by_low_value_preflight_noise]
promotion_gate: {preflight_required_for_native_benchmarks: true}
success_criteria: [benchmark_runs_reference_static_report_and_counterfactuals]
dependencies: [development, eid]
```

- Why this creates value: it saves time and focuses compute budgets.
- Why this creates moat: it unifies analysis and measurement into one operating model.
- Main risk or failure mode: preflight becomes bureaucratic instead of discriminative.
- Smallest credible first experiment: require preflight only for SIMD and native kernel microbenchmarks first.
- Confidence level: high

### 5.19 Telemetry Calibration Loop [M]

- Suggested `phase_id`: `convergence`
- Core insight: the static model should learn from measured misses rather than staying frozen.
- External inspiration or analogy: F1 aero model correlation, NASA simulation-to-flight residual analysis
- Why it fits Saguaro and Anvil specifically: `core/native/runtime_telemetry.py` already records the runtime truth we need.
- Exact wiring points: `core/native/runtime_telemetry.py`, `saguaro/cpu/roofline.py`, `saguaro/cpu/report.py`, `audit/runner/benchmark_suite.py`
- Existing primitives it can reuse: runtime telemetry, benchmark suite, static report
- New primitive, data flow, or subsystem needed: calibration residual store and drift-aware confidence update

```yaml
phase_id: convergence
objective: Feed measured benchmark and runtime telemetry back into the static math and CPU model so confidence reflects real-world correlation.
repo_scope: [core/native/runtime_telemetry.py, saguaro/cpu, audit/runner]
owning_specialist_type: PerformanceTwinLead
allowed_writes: [core/native/runtime_telemetry.py, saguaro/cpu/roofline.py, saguaro/cpu/report.py, audit/runner/benchmark_suite.py, tests/test_runtime_telemetry.py]
telemetry_contract:
  minimum: [prediction_error, calibration_generation, confidence_adjustment, residual_store_size]
required_evidence: [core/native/runtime_telemetry.py, tests/test_runtime_telemetry.py]
rollback_criteria: [calibration_destabilizes_reports, confidence_not_monotonic_with_error]
promotion_gate: {prediction_error_trend: "improving across pilot suite"}
success_criteria: [reports_show_calibrated_confidence_and_last_residual]
dependencies: [deep_test_audit]
```

- Why this creates value: the model gets better instead of merely larger.
- Why this creates moat: calibration loops turn a feature into infrastructure.
- Main risk or failure mode: overfitting to current benchmarks.
- Smallest credible first experiment: calibrate only bound-class prediction and hotspot rank overlap first.
- Confidence level: medium

### 5.20 Governance and Health Coverage Hardening [P]

- Suggested `phase_id`: `convergence`
- Core insight: unsupported math languages, blind CPU reports, and low-confidence pipeline traces must be visible in health and governance.
- External inspiration or analogy: aerospace confidence envelopes
- Why it fits Saguaro and Anvil specifically: health and verify already exist, but they are not measuring this property class.
- Exact wiring points: `saguaro/health.py`, `saguaro/sentinel/engines/semantic.py`, `saguaro/roadmap/validator.py`, `saguaro/requirements/traceability.py`
- Existing primitives it can reuse: health coverage vectors, validator worklists, traceability refs
- New primitive, data flow, or subsystem needed: math/cpu coverage vector and missing-capability findings

```yaml
phase_id: convergence
objective: Make math fidelity, language blind spots, CPU-model confidence, and benchmark-preflight coverage first-class health and verification outcomes.
repo_scope: [saguaro/health.py, saguaro/sentinel/engines/semantic.py, saguaro/roadmap/validator.py, saguaro/requirements/traceability.py]
owning_specialist_type: GovernanceRuntimeLead
allowed_writes: [saguaro/health.py, saguaro/sentinel/engines/semantic.py, saguaro/roadmap/validator.py, saguaro/requirements/traceability.py, tests/test_saguaro_roadmap_validator.py]
telemetry_contract:
  minimum: [math_coverage_percent, cpu_model_coverage_percent, unsupported_language_count, low_confidence_hotspot_count]
required_evidence: [saguaro/health.py, saguaro/sentinel/engines/semantic.py, tests/test_saguaro_roadmap_validator.py]
rollback_criteria: [health_reports_hide_blind_spots, verify_remains_silent_on_missing_math_coverage]
promotion_gate: {health_includes_math_and_cpu_vectors: true, verify_emits_missing_coverage_findings: true}
success_criteria: [blind_spots_visible_in_health_and_verify, roadmap_validator_extracts_contract_refs_cleanly]
dependencies: [convergence]
```

- Why this creates value: it makes the system honest.
- Why this creates moat: trustworthy engineering systems expose their uncertainty.
- Main risk or failure mode: warning fatigue.
- Smallest credible first experiment: add a `math_cpu` block to `saguaro health` before adding hard verify failures.
- Confidence level: high

## 6. Critical Pressure Test

- Elegant but likely wrong: exact cycle prediction from source alone. Without assembly, scheduler state, and compiler lowering, this is fantasy.
- Elegant but likely wrong: assuming `structural_score` can become a benchmark proxy with only better coefficients. The present problem is representation quality, not coefficient tuning.
- Ugly but strategically powerful: hard negative filters for obvious non-math patterns such as `super().__init__`, logger setup, parser dictionaries, path joins, fixture strings, and preprocessor guards.
- Ugly but strategically powerful: explicit architecture packs with hand-authored heuristics for AVX2, AVX-512, and NEON rather than a generic "SIMD optimization" score.
- Likely to fail unless a missing primitive lands first: counterfactual schedule search without access signatures and loop frames.
- Likely to fail unless a missing primitive lands first: CPU hotspot ranking without FFI provenance and query-surface integration.
- Likely to fail unless a missing primitive lands first: confidence-aware benchmark preflight without calibration residuals from `core/native/runtime_telemetry.py`.

## 7. Synthesis

- Strongest overall ideas by conviction:
  1. Math IR V2 Precision Gate
  2. Tensor Access Signature Emitter
  3. FFI Math Provenance Bridge
  4. CPU Hotspot Report and Query Surface
  5. Benchmark Preflight Gate
- Best novelty/plausibility balance: FFI Math Provenance Bridge plus CPU Hotspot Report. It is ambitious enough to matter and grounded enough to ship.
- Most feasible now: Math IR V2 Precision Gate. The failure is already measurable and the code seams already exist.
- Biggest long-term moat bet: Counterfactual Schedule Twin. If it correlates well enough to narrow benchmark search, Saguaro becomes a pre-benchmark architecture oracle rather than a passive indexer.
- Cleanest unification with the current codebase: Tensor Access Signature Emitter. It naturally links `saguaro/math`, `saguaro/parsing`, `core/simd`, and the future CPU advisor without inventing a parallel substrate.
- Prototype first: Math IR V2 Precision Gate, then Tensor Access Signature Emitter, then CPU Hotspot Report.

## 8. Implementation Program

### 8.1 `analysis_upgrade` - Math IR V2 and Precision Recovery

- Phase title: Math IR V2
- Objective: Replace permissive equation scraping with parser-grounded mathematical statement extraction, typed records, and negative filtering.
- Dependencies: none
- Repo scope: `saguaro/math/engine.py`, `saguaro/math/ir.py`, `saguaro/parsing/parser.py`, `saguaro/api.py`, `saguaro/cli.py`
- Owning specialist type: `MathSystemsArchitect`
- Allowed writes: `saguaro/math/engine.py`, `saguaro/math/ir.py`, `saguaro/parsing/parser.py`, `saguaro/api.py`, `saguaro/cli.py`, `tests/test_saguaro_math.py`, `tests/test_saguaro_math_precision.py`
- Telemetry contract: `files_scanned`, `records_emitted`, `false_positive_rate`, `records_by_language`, `parse_seconds`
- Required evidence: live `saguaro math parse` output on `saguaro/` and curated negative fixtures
- Rollback criteria: false-positive suppression collapses recall on real math kernels
- Promotion gate: at least 60% false-positive reduction on current repo while preserving at least 95% recall on curated math fixtures
- Success criteria: parser dictionaries, path joins, test fixture strings, and preprocessor guards no longer dominate top math records
- Exact wiring points:
  - `MathEngine.parse` and `_extract_code_equations` in `saguaro/math/engine.py`
  - `SAGUAROParser.parse_file` and `_detect_language` in `saguaro/parsing/parser.py`
  - `SaguaroAPI.math_parse` in `saguaro/api.py`
  - `math parse` CLI dispatch in `saguaro/cli.py`
- Deliverables:
  - versioned `MathIRRecord`
  - negative filter bank
  - per-language support matrix
  - precision telemetry in `.saguaro/math/cache.json`
- Tests: `tests/test_saguaro_math.py`, `tests/test_saguaro_math_precision.py`, `tests/test_saguaro_parser_languages.py`
- Verification commands:
  - `pytest tests/test_saguaro_math.py tests/test_saguaro_math_precision.py tests/test_saguaro_parser_languages.py`
  - `./venv/bin/saguaro math parse --path saguaro --format json`
  - `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: the math engine is precise enough that the top 20 repo-wide records are mostly real numeric work, not incidental scaffolding.

### 8.2 `research` - Cross-Language Math Coverage and CPU Architecture Packs

- Phase title: Language and ISA Coverage
- Objective: bring math extraction up to parser coverage parity and formalize architecture packs for AVX2, AVX-512, and NEON.
- Dependencies: `analysis_upgrade`
- Repo scope: `saguaro/math/engine.py`, `saguaro/math/languages.py`, `saguaro/cpu/topology.py`, `core/native/runtime_telemetry.py`
- Owning specialist type: `MultilingualStaticAnalysisLead`
- Allowed writes: `saguaro/math/engine.py`, `saguaro/math/languages.py`, `saguaro/cpu/topology.py`, `core/native/runtime_telemetry.py`, `tests/test_saguaro_math_languages.py`, `tests/test_saguaro_cpu_topology.py`
- Telemetry contract: `supported_language_families`, `precision_by_language`, `unsupported_language_count`, `arch_pack_selected`, `isa_match_state`
- Required evidence: math fixtures from scientific languages and runtime telemetry pack matches
- Rollback criteria: language breadth increases while precision or architecture specificity collapses
- Promotion gate: at least 10 language families supported and three CPU architecture packs validated against runtime telemetry
- Success criteria: math extraction and CPU analysis are both explicit about support, unsupported space, and architecture assumptions
- Exact wiring points:
  - language-family policy in `saguaro/math/languages.py`
  - pack selection and parameterization in `saguaro/cpu/topology.py`
  - runtime pack calibration using `core/native/runtime_telemetry.py`
- Deliverables:
  - language-family math packs
  - architecture pack catalog
  - unsupported-language reporting
- Tests: `tests/test_saguaro_math_languages.py`, `tests/test_saguaro_parser_languages.py`, `tests/test_saguaro_cpu_topology.py`, `tests/test_runtime_telemetry.py`
- Verification commands:
  - `pytest tests/test_saguaro_math_languages.py tests/test_saguaro_parser_languages.py tests/test_saguaro_cpu_topology.py tests/test_runtime_telemetry.py`
  - `./venv/bin/saguaro math parse --path tests/test_saguaro_parser_languages.py --format json`
- Exit criteria: coverage is explicit and architecture-aware rather than implied.

### 8.3 `feature_map` - Access Signatures, Layout State, and FFI Provenance

- Phase title: Pipeline Truth Map
- Objective: attach each important math record to loop context, access signatures, layout state, FFI boundary, and build target.
- Dependencies: `analysis_upgrade`, `research`
- Repo scope: `saguaro/math/pipeline.py`, `saguaro/omnigraph/store.py`, `saguaro/health.py`, `core/native/CMakeLists.txt`
- Owning specialist type: `FFIObservabilityLead`
- Allowed writes: `saguaro/math/pipeline.py`, `saguaro/omnigraph/store.py`, `saguaro/health.py`, `tests/test_saguaro_math_pipeline.py`
- Telemetry contract: `access_signature_count`, `layout_transition_count`, `ffi_math_edge_count`, `unresolved_kernel_binding_count`
- Required evidence: pipeline reports on selected native kernels and build targets
- Rollback criteria: provenance graph becomes noisy or unresolved for hot symbols
- Promotion gate: at least 80% of hot math records bind to a kernel or build target
- Success criteria: a queryable pipeline truth exists for math-to-kernel-to-target tracing
- Exact wiring points:
  - `MathProvenanceEdge` generation in `saguaro/math/pipeline.py`
  - persistence in `saguaro/omnigraph/store.py`
  - visibility in `saguaro/health.py`
  - build target linkage from `core/native/CMakeLists.txt`
- Deliverables:
  - access signature emitter
  - layout propagation graph
  - FFI/build-target provenance map
- Tests: `tests/test_saguaro_math_pipeline.py`, `tests/test_saguaro_graph_resolution.py`
- Verification commands:
  - `pytest tests/test_saguaro_math_pipeline.py tests/test_saguaro_graph_resolution.py`
  - `./venv/bin/saguaro health`
- Exit criteria: the agent can trace real math into concrete kernels and targets with explicit confidence.

### 8.4 `eid` - Static Roofline and Counterfactual Schedule Twin

- Phase title: Performance Twin
- Objective: estimate bound class and generate constrained schedule alternatives before running live benchmarks.
- Dependencies: `feature_map`
- Repo scope: `saguaro/cpu/roofline.py`, `saguaro/cpu/register_pressure.py`, `saguaro/cpu/schedule_twin.py`, `core/native/runtime_telemetry.py`
- Owning specialist type: `PerformanceResearchLead`
- Allowed writes: `saguaro/cpu/roofline.py`, `saguaro/cpu/register_pressure.py`, `saguaro/cpu/schedule_twin.py`, `core/native/runtime_telemetry.py`, `tests/test_saguaro_cpu_twin.py`
- Telemetry contract: `predicted_bound_class`, `roofline_confidence`, `register_pressure_score`, `counterfactual_count`, `prediction_error`
- Required evidence: measured-versus-predicted comparisons on a pilot kernel suite
- Rollback criteria: model correlation is too weak to rank benchmark effort
- Promotion gate: top-3 counterfactual schedules contain the measured winner on at least 60% of pilot cases
- Success criteria: benchmark preflight receives actionable ranked alternatives, not just diagnostics
- Exact wiring points:
  - operational-intensity estimation in `saguaro/cpu/roofline.py`
  - spill-risk surrogate in `saguaro/cpu/register_pressure.py`
  - candidate schedule generation in `saguaro/cpu/schedule_twin.py`
  - residual capture in `core/native/runtime_telemetry.py`
- Deliverables:
  - static roofline estimator
  - register-pressure surrogate
  - constrained counterfactual schedule generator
- Tests: `tests/test_saguaro_cpu_twin.py`, `tests/test_runtime_telemetry.py`
- Verification commands:
  - `pytest tests/test_saguaro_cpu_twin.py tests/test_runtime_telemetry.py`
- Exit criteria: the twin is useful enough to narrow the benchmark search space.

### 8.5 `development` - Agent-Facing CPU Scan and Explainable Hotspot Reports

- Phase title: CPU Advisor Surface
- Objective: expose static CPU hotspot reports, SIMD legality, prefetch guidance, and vectorization miss explanations through API, CLI, substrate, and tool layers.
- Dependencies: `feature_map`, `eid`
- Repo scope: `saguaro/cpu/model.py`, `saguaro/cpu/vectorization.py`, `saguaro/cpu/prefetch.py`, `saguaro/cpu/report.py`, `saguaro/api.py`, `saguaro/cli.py`, `tools/saguaro_tools.py`, `domains/code_intelligence/saguaro_substrate.py`, `saguaro/services/platform.py`
- Owning specialist type: `AgentSurfaceArchitect`
- Allowed writes: `saguaro/cpu/model.py`, `saguaro/cpu/vectorization.py`, `saguaro/cpu/prefetch.py`, `saguaro/cpu/report.py`, `saguaro/api.py`, `saguaro/cli.py`, `tools/saguaro_tools.py`, `domains/code_intelligence/saguaro_substrate.py`, `saguaro/services/platform.py`, `tests/test_saguaro_cpu_scan.py`, `tests/test_saguaro_interface.py`
- Telemetry contract: `cpu_scan_calls`, `report_latency_ms`, `simd_blocked_count`, `prefetch_opportunity_count`, `query_hit_rate_for_hot_math`
- Required evidence: end-to-end agent access to CPU reports and improved query relevance
- Rollback criteria: reports are inaccessible to agents or too vague for rewrites
- Promotion gate: agent can request `math parse`, `cpu scan`, and hotspot report without fallback filesystem spelunking
- Success criteria: hotspot reports are explainable, ISA-specific, and surfaced through the existing agent substrate
- Exact wiring points:
  - API surface in `saguaro/api.py`
  - CLI surface in `saguaro/cli.py`
  - tool facade in `tools/saguaro_tools.py`
  - substrate routing in `domains/code_intelligence/saguaro_substrate.py`
  - ranking priors in `saguaro/services/platform.py`
- Deliverables:
  - `saguaro cpu scan`
  - `saguaro cpu report`
  - vectorization miss explanations
  - prefetch opportunity diagnostics
- Tests: `tests/test_saguaro_cpu_scan.py`, `tests/test_saguaro_interface.py`
- Verification commands:
  - `pytest tests/test_saguaro_cpu_scan.py tests/test_saguaro_interface.py`
  - `./venv/bin/saguaro cpu scan --path core/simd/common/perf_utils.h --arch x86_64-avx2 --format json`
- Exit criteria: the agent can see and act on CPU hotspot reports through native Saguaro surfaces.

### 8.6 `deep_test_audit` - Benchmark Preflight and Correlation

- Phase title: Benchmark Preflight
- Objective: make static math and CPU analysis a prerequisite for the expensive benchmark lanes that matter.
- Dependencies: `development`
- Repo scope: `audit/runner/benchmark_suite.py`, `audit/runner/native_benchmark_runner.py`, `benchmarks/simd_benchmark.py`, `core/native/runtime_telemetry.py`
- Owning specialist type: `BenchmarkControlPlaneArchitect`
- Allowed writes: `audit/runner/benchmark_suite.py`, `audit/runner/native_benchmark_runner.py`, `benchmarks/simd_benchmark.py`, `core/native/runtime_telemetry.py`, `tests/audit/test_benchmark_suite.py`
- Telemetry contract: `preflight_present`, `predicted_hotspot_count`, `static_to_measured_overlap`, `counterfactual_confirmation_rate`
- Required evidence: benchmark manifests that reference static hotspot reports and measured residuals
- Rollback criteria: preflight blocks productive runs without reducing search space
- Promotion gate: native benchmarks refuse expensive runs lacking preflight on targeted lanes
- Success criteria: benchmark time is spent validating the right candidates, not randomly searching
- Exact wiring points:
  - preflight hook in `audit/runner/benchmark_suite.py`
  - native-runner ingestion in `audit/runner/native_benchmark_runner.py`
  - pilot suites in `benchmarks/simd_benchmark.py`
  - residual capture in `core/native/runtime_telemetry.py`
- Deliverables:
  - benchmark preflight contract
  - correlation report between predictions and measured hotspots
  - benchmark target prioritization
- Tests: `tests/audit/test_benchmark_suite.py`, `tests/test_runtime_telemetry.py`
- Verification commands:
  - `pytest tests/audit/test_benchmark_suite.py tests/test_runtime_telemetry.py`
  - `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: benchmark audit uses Saguaro analysis as a real prerequisite rather than optional commentary.

### 8.7 `convergence` - Health, Governance, and Traceability Hardening

- Phase title: Confidence Surface
- Objective: make blind spots, unsupported languages, low-confidence hotspot claims, and roadmap traceability visible in `health`, `verify`, and roadmap validation.
- Dependencies: `deep_test_audit`
- Repo scope: `saguaro/health.py`, `saguaro/sentinel/engines/semantic.py`, `saguaro/roadmap/validator.py`, `saguaro/requirements/traceability.py`
- Owning specialist type: `GovernanceRuntimeLead`
- Allowed writes: `saguaro/health.py`, `saguaro/sentinel/engines/semantic.py`, `saguaro/roadmap/validator.py`, `saguaro/requirements/traceability.py`, `tests/test_saguaro_roadmap_validator.py`, `tests/test_saguaro_interface.py`
- Telemetry contract: `math_coverage_percent`, `cpu_model_coverage_percent`, `unsupported_language_count`, `low_confidence_hotspot_count`, `verification_ref_count`
- Required evidence: health JSON, verify findings, roadmap validation graph
- Rollback criteria: blind spots remain silent or the validator cannot trace implementation contracts cleanly
- Promotion gate: health and verify both expose math/cpu coverage and missing-capability findings
- Success criteria: Saguaro can state what it knows, what it thinks, and what it does not know
- Exact wiring points:
  - coverage vector updates in `saguaro/health.py`
  - semantic findings in `saguaro/sentinel/engines/semantic.py`
  - roadmap completion graph in `saguaro/roadmap/validator.py`
  - traceability ref extraction in `saguaro/requirements/traceability.py`
- Deliverables:
  - math/cpu health block
  - verify findings for unsupported math/CPU coverage
  - roadmap completion graph that can trace this roadmap contract
- Tests: `tests/test_saguaro_roadmap_validator.py`, `tests/test_saguaro_interface.py`
- Verification commands:
  - `pytest tests/test_saguaro_roadmap_validator.py tests/test_saguaro_interface.py`
  - `./venv/bin/saguaro health`
  - `./venv/bin/saguaro roadmap validate --path Saguaro_CPU_Math_Roadmap.md --format json`
- Exit criteria: the system exposes confidence and blind spots as first-class outputs rather than hidden assumptions.

## 9. Implementation Contract

- The system shall replace permissive math scraping in `saguaro/math/engine.py` with typed `MathIRRecord` extraction in `saguaro/math/ir.py`, grounded by `saguaro/parsing/parser.py`, tested by `tests/test_saguaro_math.py`, `tests/test_saguaro_math_precision.py`, and `tests/test_saguaro_parser_languages.py`, and verified with `pytest tests/test_saguaro_math.py tests/test_saguaro_math_precision.py tests/test_saguaro_parser_languages.py`, `./venv/bin/saguaro math parse --path saguaro --format json`, and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall align math-language coverage in `saguaro/math/engine.py` and `saguaro/math/languages.py` with the parser coverage policy in `saguaro/parsing/parser.py`, tested by `tests/test_saguaro_math_languages.py` and `tests/test_saguaro_parser_languages.py`, and verified with `pytest tests/test_saguaro_math_languages.py tests/test_saguaro_parser_languages.py` and `./venv/bin/saguaro math parse --path tests/test_saguaro_parser_languages.py --format json`.
- The system shall emit loop context, access signatures, and layout states through `saguaro/math/engine.py` and `saguaro/math/pipeline.py`, persist graph-facing provenance in `saguaro/omnigraph/store.py`, tested by `tests/test_saguaro_math_access.py` and `tests/test_saguaro_math_pipeline.py`, and verified with `pytest tests/test_saguaro_math_access.py tests/test_saguaro_math_pipeline.py` and `./venv/bin/saguaro health`.
- The system shall bind hot math records to kernels and build targets through `saguaro/math/pipeline.py`, `saguaro/omnigraph/store.py`, and `core/native/CMakeLists.txt`, tested by `tests/test_saguaro_math_pipeline.py` and `tests/test_saguaro_graph_resolution.py`, and verified with `pytest tests/test_saguaro_math_pipeline.py tests/test_saguaro_graph_resolution.py` and `./venv/bin/saguaro impact --path core/native/runtime_telemetry.py`.
- The system shall implement architecture packs and CPU-scan entrypoints through `saguaro/cpu/topology.py`, `saguaro/cpu/model.py`, `saguaro/api.py`, and `saguaro/cli.py`, tested by `tests/test_saguaro_cpu_topology.py`, `tests/test_saguaro_cpu_scan.py`, and `tests/test_saguaro_interface.py`, and verified with `pytest tests/test_saguaro_cpu_topology.py tests/test_saguaro_cpu_scan.py tests/test_saguaro_interface.py` and `./venv/bin/saguaro cpu scan --path core/simd/common/perf_utils.h --arch x86_64-avx2 --format json`.
- The system shall generate SIMD-legality, prefetch, and vectorization-miss reports through `saguaro/cpu/vectorization.py`, `saguaro/cpu/prefetch.py`, `saguaro/cpu/report.py`, `tools/saguaro_tools.py`, and `domains/code_intelligence/saguaro_substrate.py`, tested by `tests/test_saguaro_cpu_scan.py` and `tests/test_saguaro_interface.py`, and verified with `pytest tests/test_saguaro_cpu_scan.py tests/test_saguaro_interface.py` and `./venv/bin/saguaro cpu scan --path core/simd/common/perf_utils.h --arch arm64-neon --format json`.
- The system shall implement a calibrated static performance twin through `saguaro/cpu/roofline.py`, `saguaro/cpu/register_pressure.py`, `saguaro/cpu/schedule_twin.py`, and `core/native/runtime_telemetry.py`, tested by `tests/test_saguaro_cpu_twin.py` and `tests/test_runtime_telemetry.py`, and verified with `pytest tests/test_saguaro_cpu_twin.py tests/test_runtime_telemetry.py`.
- The system shall require benchmark preflight integration through `audit/runner/benchmark_suite.py`, `audit/runner/native_benchmark_runner.py`, `benchmarks/simd_benchmark.py`, and `core/native/runtime_telemetry.py`, tested by `tests/audit/test_benchmark_suite.py` and `tests/test_runtime_telemetry.py`, and verified with `pytest tests/audit/test_benchmark_suite.py tests/test_runtime_telemetry.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall expose math and CPU coverage, blind spots, and roadmap-traceable verification through `saguaro/health.py`, `saguaro/sentinel/engines/semantic.py`, `saguaro/roadmap/validator.py`, and `saguaro/requirements/traceability.py`, tested by `tests/test_saguaro_roadmap_validator.py` and `tests/test_saguaro_interface.py`, and verified with `pytest tests/test_saguaro_roadmap_validator.py tests/test_saguaro_interface.py`, `./venv/bin/saguaro health`, and `./venv/bin/saguaro roadmap validate --path Saguaro_CPU_Math_Roadmap.md --format json`.
