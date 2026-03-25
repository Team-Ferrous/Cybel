# QSG Latent Memory Roadmap

Status: expanded architecture roadmap
Date: 2026-03-11
Repo: `/home/mike/Documents/Github/Anvil`
Primary focus: QSG adjacencies, COCONUT, Jacobi, Grover, latent memory, inference control plane, end-user memory productization
Authoring mode: repo-grounded plus external research

## 1. First-Principles Framing

### What the system fundamentally is

Repo-grounded observation:
Anvil is not fundamentally a chatbot.
It is a repo-coupled inference control system with a multi-agent UI wrapped around it.
Its real substrate is the interaction between:
`core/unified_chat_loop.py`,
`core/native/native_qsg_engine.py`,
`core/qsg/continuous_engine.py`,
`core/qsg/runtime_contracts.py`,
`core/qsg/latent_bridge.py`,
and the ALMF storage surfaces under `core/memory/fabric/`.

Repo-grounded observation:
The token stream is only one manifestation of state.
The codebase already contains an alternative state path:
continuous thought vectors in `core/reasoning/coconut.py`,
execution capsules and latent packet ABIs in `core/qsg/runtime_contracts.py`,
request-state paging in `core/qsg/state_pager.py`,
and latent capture/replay in `core/qsg/continuous_engine.py` plus `core/qsg/latent_bridge.py`.

Repo-grounded observation:
The system already behaves like an operating system in miniature.
`UnifiedChatLoop` is the control plane.
`NativeQSGEngine` and `QSGInferenceEngine` are the execution plant.
`QSGLatentBridge` plus `MemoryFabricStore` are the persistence and replay layer.
`BlackBoxRecorder` and `NativeGenerationTelemetry` are the flight recorder.
Saguaro is the environment model and repository sensor stack.

### What the system appears to be trying to become

Inference:
The codebase appears to be trying to become a CPU-first, verifiable, memory-native reasoning runtime for code and research work.
The aspiration is not just "better prompts" or "bigger context."
The aspiration is a system that can:
store internal reasoning state compactly,
replay it safely,
route between memory tiers,
adapt runtime policy using telemetry,
and collapse only the minimum useful portion of latent state back into tokens.

Inference:
If engineered to elite standards, Anvil becomes a software digital twin of its own inference process.
It should know:
what it is likely to need next,
which memory representation is cheapest,
which reasoning branch is weak,
which latent assets are stale,
which replay is compatible,
and which artifacts are safe to keep outside the main chat context.

### Core constraints and engineering standards

- CPU-first execution is a hard constraint, not an implementation detail.
- Determinism and replayability matter because the repo already contains governance, telemetry, and roadmap validation infrastructure.
- Context tokens are the scarce budget, but raw vector persistence is not automatically safe or useful; meaning has to survive compression.
- Memory features must improve end-user utility, not just internal elegance.
- Safety must be enforced at the ABI and compatibility level, not just with best-effort retrieval.
- The system must degrade gracefully when a latent package is missing, incompatible, stale, or low-value.
- Any upgrade should reuse the existing center of gravity:
  `UnifiedChatLoop`,
  `NativeQSGEngine`,
  `QSGInferenceEngine`,
  `QSGLatentBridge`,
  `MemoryFabricStore`,
  and `core/qsg/runtime_contracts.py`.
- NASA / Formula 1 standards imply:
  telemetry before folklore,
  replayable failures,
  bounded rollback criteria,
  explicit interface contracts,
  and testable promotion gates.

## 2. External Research Scan

### Research themes that materially expand the design space

Repo-grounded observation:
The current repo already has the primitives for latent capture, replay, telemetry, and controller policies.
External research matters here because it shows what those primitives can become if pushed harder.

### Continuous latent reasoning

- External inspiration:
  COCONUT shows that the last hidden state can be fed back as a continuous thought instead of collapsing every intermediate reasoning step into text.
  That matters because `ContinuousThoughtBlock` already exposes amplitudes and session records rather than only textual summaries.
  Source: [R1](https://arxiv.org/abs/2412.06769)
- External inspiration:
  The COCONUT paper also reports emergent breadth-first search behavior in latent space.
  That maps directly onto the repo’s current `num_paths`, amplitude tracking, and candidate-path scoring in `core/reasoning/coconut.py`.
  Source: [R1](https://arxiv.org/abs/2412.06769)

### Parallel decoding and branch verification

- External inspiration:
  Lookahead Decoding shows that exact parallel decoding can be achieved without an auxiliary draft model by verifying multiple future tokens together.
  That matters because `JacobiRefiner` is still mostly a smoothing placeholder while the runtime already contains frontier and verification language in `SpeculativeFrontierPolicy`.
  Source: [R2](https://arxiv.org/abs/2402.02057)
- External inspiration:
  Medusa shows that multiple decode heads plus a tree verifier can move speculation from a single draft path to a path family.
  That is a better analogy for Anvil’s current COCONUT multi-path reasoning than plain top-k sampling.
  Sources: [R3](https://arxiv.org/abs/2401.10774), [R4](https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html)
- External inspiration:
  EAGLE argues that feature-level speculative prediction can be simpler than token-level speculation.
  That is directly relevant to using latent packets as the object under verification instead of only token strings.
  Source: [R5](https://arxiv.org/abs/2401.15077)

### Memory paging, KV reuse, and bounded-memory inference

- External inspiration:
  PagedAttention reframes KV cache management as virtual memory with paging and copy-on-write.
  That maps tightly to `QSGStatePager`, which already has allocation, import/export, compaction, and clone-COW semantics.
  Source: [R6](https://arxiv.org/abs/2309.06180)
- External inspiration:
  H2O shows that not all cached tokens matter equally; heavy hitters deserve explicit retention policy.
  That strongly suggests Anvil should not treat all latent vectors as equal-value replay assets.
  Source: [R7](https://github.com/FMInference/H2O)
- External inspiration:
  SnapKV and related KV compression work show that targeted retention of salient prompt features can preserve quality while shrinking memory.
  This argues for typed latent saves instead of a single averaged vector in `LatentMemory`.
  Source: [R8](https://arxiv.org/abs/2404.14469)
- External inspiration:
  Compressed Context Memory shows a path to continually compress growing KV context into bounded memory while preserving interactive quality.
  That is directly relevant to chat sessions and long research missions inside `UnifiedChatLoop`.
  Sources: [R9](https://arxiv.org/abs/2312.03414), [R10](https://github.com/snu-mllab/Context-Memory)
- External inspiration:
  Infini-attention demonstrates bounded-memory streaming with long-term compressive memory integrated into the attention block.
  This suggests that latent memory should be thought of as a first-class streaming structure, not just an after-action artifact.
  Source: [R11](https://arxiv.org/abs/2404.07143)
- External inspiration:
  Recurrent Memory Transformer and Memorizing Transformers both argue for explicit memory tokens or external non-differentiable memory at inference time.
  That matters because Anvil already has latent package, multivector, and HD bundle storage surfaces but uses them conservatively.
  Sources: [R12](https://arxiv.org/abs/2207.06881), [R13](https://arxiv.org/abs/2203.08913)

### Test-time memorization and adaptive memory modules

- External inspiration:
  Learning to Learn at Test Time reframes hidden state as an updatable model rather than a fixed vector.
  That matters because the repo already records telemetry and controller state per run, which could become memory-update supervision.
  Source: [R14](https://arxiv.org/abs/2407.04620)
- External inspiration:
  Titans and ATLAS push beyond fixed-memory recurrence by learning what is worth memorizing and optimizing memory using current and past tokens.
  That is a strong mechanism-level analogue for a write controller over `LatentMemory` and `MemoryFabricStore`.
  Sources: [R15](https://research.google/pubs/titans-learning-to-memorize-at-test-time/), [R16](https://arxiv.org/abs/2505.23735)

### Agent memory products and context assembly

- External inspiration:
  Letta’s memory blocks show a pragmatic split between always-visible core memory and larger archival memory.
  This matters because end users need predictable memory behavior, not opaque retrieval magic.
  Sources: [R17](https://docs.letta.com/guides/core-concepts/memory/memory-blocks), [R18](https://docs.letta.com/guides/agents/architectures/memgpt)
- External inspiration:
  Mem0’s direct import and multimodal memory extraction show that memory systems get more useful when they support explicit developer-controlled writes alongside inferred writes.
  This is relevant to user-facing latent save policies and curated memory pinning.
  Sources: [R19](https://docs.mem0.ai/platform/features/direct-import), [R20](https://docs.mem0.ai/open-source/features/multimodal-support)
- External inspiration:
  Zep and Graphiti show that temporal knowledge graphs can materially reduce token use while improving recall and dynamic fact handling.
  This matters because repo delta, user preferences, and tool outcomes all evolve over time and should not be flattened into a single vector.
  Sources: [R21](https://www.getzep.com/product/agent-memory/), [R22](https://www.getzep.com/product/open-source), [R23](https://help.getzep.com/graphiti/graphiti/overview)

### Prompt caching and non-prefix reuse

- External inspiration:
  OpenAI, Azure OpenAI, and Gemini all emphasize that prompt caching depends on stable repeated prefixes and retention policy.
  That matters because end-user chat sessions frequently violate exact-prefix assumptions.
  Latent memory is strategically valuable precisely because it can preserve useful state even when prefix caching misses.
  Sources: [R24](https://platform.openai.com/docs/guides/prompt-caching), [R25](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/prompt-caching), [R26](https://ai.google.dev/gemini-api/docs/caching/)
- External inspiration:
  KVLink, KVShare, CacheBlend, and LMCache show that non-prefix reuse is now a real engineering frontier rather than a toy idea.
  This is especially relevant to multi-turn repository analysis, where chunks recur in different orders and with different wrapper prompts.
  Sources: [R27](https://arxiv.org/abs/2502.16002), [R28](https://arxiv.org/abs/2503.16525), [R29](https://arxiv.org/abs/2405.16444), [R30](https://github.com/LMCache/LMCache)

### Cross-industry analogies that are mechanically useful

- External inspiration:
  NASA JSTAR digital twins emphasize "fail virtually, succeed actually," replayable software environments, and early fault injection.
  That is an almost perfect analogue for replayable latent-package validation before promotion into user-visible behavior.
  Sources: [R31](https://www.nasa.gov/jstar-digital-twins/), [R32](https://www.nasa.gov/ivv-services/)
- External inspiration:
  NASA ISHM and ISAAC work emphasize integrated system health, adaptive telemetry rates, and autonomous caretaking.
  That maps onto runtime-control policies that should change coconut depth, memory write rate, and replay aggressiveness as the session condition changes.
  Sources: [R33](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/), [R34](https://www.nasa.gov/integrated-system-for-autonomous-and-adaptive-caretaking-isaac/)
- External inspiration:
  Formula 1 uses dense real-time telemetry plus counterfactual strategy simulation rather than pure post-race analysis.
  That suggests Anvil should treat replay tapes and performance twins as active control inputs, not archival reports.
  Source: [R35](https://aws.amazon.com/sports/f1/)

### Why this research matters here

Inference:
The external material converges on five mechanisms that are underexploited in this repo:

- typed memory instead of generic memory
- branch verification instead of branch smoothing
- surprise-weighted writes instead of write-everything behavior
- non-prefix reuse instead of only prefix reuse
- telemetry-driven runtime policy instead of static heuristics

Inference:
The practical opportunity is to harden the current ABI and controller surfaces.
The moonshot opportunity is to make latent memory a real operating primitive that sits between token generation, retrieval, prompt assembly, and replay.

## 3. Repo Grounding Summary

### Saguaro workflow and fallback notes

Repo-grounded observation:
The requested repo analysis was run from the repo virtual environment using `source venv/bin/activate`.

Repo-grounded observation:
The following Saguaro commands completed successfully:

- `./venv/bin/saguaro entrypoints`
- `./venv/bin/saguaro build-graph`
- `./venv/bin/saguaro agent skeleton core/qsg/continuous_engine.py`
- `./venv/bin/saguaro agent skeleton core/memory/latent_memory.py`
- `./venv/bin/saguaro agent skeleton core/reasoning/coconut.py`
- `./venv/bin/saguaro agent skeleton core/qsg/grover.py`
- `./venv/bin/saguaro agent skeleton core/qsg/jacobi_refiner.py`
- `./venv/bin/saguaro agent skeleton core/qsg/runtime_contracts.py`
- `./venv/bin/saguaro agent skeleton core/qsg/state_pager.py`
- `./venv/bin/saguaro agent skeleton core/qsg/latent_bridge.py`
- `./venv/bin/saguaro agent slice QSGInferenceEngine.capture_latent_state --depth 2`
- `./venv/bin/saguaro agent slice QSGInferenceEngine.restore_latent_state --depth 2`
- `./venv/bin/saguaro impact --path core/qsg/latent_bridge.py`

Repo-grounded observation:
The following Saguaro commands did not return useful output within a 20 second timeout after TensorFlow startup noise:

- `./venv/bin/saguaro health`
- `./venv/bin/saguaro query "COCONUT" --k 5`
- `./venv/bin/saguaro query "Grovers" --k 5`
- `./venv/bin/saguaro query "Jacobi" --k 5`
- `./venv/bin/saguaro query "latent memory" --k 5`
- `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`

Repo-grounded observation:
Because the timeout behavior was documented, fallback inspection used targeted `rg`, `nl`, and `sed` only after successful Saguaro skeleton and slice passes.

### Code paths inspected

Repo-grounded observation:
Prompt assembly and user-facing loop:

- `core/unified_chat_loop.py`
- `core/prompts/system_prompt_builder.py`
- `core/orchestrator/loop_orchestrator.py`
- `core/agents/planner.py`

Repo-grounded observation:
QSG runtime and native boundary:

- `core/native/native_qsg_engine.py`
- `core/native/runtime_telemetry.py`
- `core/qsg/continuous_engine.py`
- `core/qsg/state_pager.py`
- `core/qsg/runtime_contracts.py`
- `core/qsg/ollama_adapter.py`
- `core/qsg/generator.py`

Repo-grounded observation:
COCONUT, Grover, Jacobi:

- `core/reasoning/coconut.py`
- `core/qsg/grover.py`
- `core/qsg/jacobi_refiner.py`

Repo-grounded observation:
Latent memory and persistence:

- `core/memory/latent_memory.py`
- `core/qsg/latent_bridge.py`
- `core/memory/fabric/models.py`
- `core/memory/fabric/store.py`
- `core/memory/fabric/retrieval_planner.py`

Repo-grounded observation:
Telemetry, governance, hooks:

- `core/telemetry/black_box.py`
- `infrastructure/hooks/registry.py`

Repo-grounded observation:
Tests and benchmarks inspected:

- `tests/test_latent_package_capture.py`
- `tests/test_qsg_continuous_engine.py`
- `tests/test_qsg_runtime_contracts.py`
- `tests/test_coconut_session_record.py`
- `tests/test_qsg_pipeline.py`
- `tests/test_coconut_integration.py`
- `tests/test_subagent_latent_pipeline.py`
- `tests/test_runtime_telemetry.py`

### Existing primitives that matter most

Repo-grounded observation:
`QSGInferenceEngine.capture_latent_state()` already emits:
execution capsule,
latent packet,
delta watermark,
capability digest,
hidden dimension,
generated token count,
phase state,
and the exported request tensor.
That is stronger than a generic "save embedding" function.

Repo-grounded observation:
`QSGInferenceEngine.restore_latent_state()` already rehydrates a request into the scheduler and emits a `latent_restored` event chunk.
That means replay is already a first-class runtime action, not an afterthought.

Repo-grounded observation:
`QSGLatentBridge.replay()` already evaluates compatibility,
computes a memory-tier decision,
writes replay events,
and returns a mission replay descriptor.
This is the right center of gravity for future latent-memory policy.

Repo-grounded observation:
`MemoryFabricStore` already stores:
dense embeddings,
multivectors,
HD bundles,
latent packages,
read records,
feedback records,
and raw tensor blobs.
The storage substrate is broader than the current product surface.

Repo-grounded observation:
`UnifiedChatLoop._run_coconut_phase()` already blends reinjected latent vectors into the current context embedding with a fixed 80/20 mix before deep thinking.
The system is already using latent state to shape future inference.

Repo-grounded observation:
`UnifiedChatLoop._append_subagent_latent_payload()` already merges subagent latent states,
normalizes them,
stores them in evidence,
and writes a merged vector into `self.agent.latent_memory`.
The multi-agent feature stack is already latent-aware.

Repo-grounded observation:
`NativeQSGEngine.get_runtime_status()` already computes frontier, drift, and memory tier controller state plus a performance twin.
The controller vocabulary exists.
The closed-loop behavior is what is thin.

### Where the architecture looks strongest

- The ABI mindset is already present in `core/qsg/runtime_contracts.py`.
- The continuous engine is more mature than the public roadmap implied.
- Replay and compatibility logic are already wired through `QSGLatentBridge`.
- Telemetry coverage in `NativeGenerationTelemetry` is unusually rich for an early-stage system.
- The repo already contains direct tests for capture, replay, runtime invariants, telemetry, and coconut session records.

### Where the architecture looks underexploited or thin

Repo-grounded observation:
`LatentMemory` still behaves like a simple rolling average of recent vectors.
`get_merged_vector()` averages element-wise and `build_package()` turns that into a single-row tensor.
This is structurally too weak for the ambitions implied elsewhere in the repo.

Repo-grounded observation:
`GroverAmplifier.amplify_with_resonance()` is still mostly heuristic.
It scores token resonance by checking if token strings appear inside relevant file names or snippets.
That is not a serious oracle relative to the available latent and repo-delta machinery.

Repo-grounded observation:
`JacobiRefiner` is still a placeholder smoother rather than a verifier-backed frontier refiner.
Its current behavior is useful as a stub but not as an architectural destination.

Repo-grounded observation:
`MemoryRetrievalPlanner` is still mostly dense plus rerank retrieval.
It is not yet exploiting:
latent package compatibility,
delta-watermark-aware retrieval,
or typed user-memory surfaces.

Repo-grounded observation:
`NativeQSGEngine` exposes a `memory_tier` controller state but there is little evidence that this controller materially changes what gets written, retained, or rehydrated at the user-memory feature level.

Repo-grounded observation:
The end-user value surface is still thin.
The stack contains powerful latent artifacts, but users mostly get hidden reinjection rather than an inspectable, editable, productized memory model.

## 4. Hidden Assumptions

- The best latent artifact is a single merged float vector.
- Latent memory is mainly useful for internal context enrichment rather than user-visible continuity.
- Continuous thought should end at reranking or token guidance instead of becoming a typed replay asset.
- Grover’s oracle should be lexical or file-name-driven rather than vector- and invariant-driven.
- Jacobi should smooth distributions rather than manage a verified frontier.
- Memory tiering is mostly about read decisions, not write admission, expiry, or replay composition.
- User memory must either be raw text or hidden vectors, with no middle layer of typed, inspectable memory blocks.
- Repo delta is mostly a compatibility guard, not a retrieval and summarization primitive.
- Telemetry is for reporting rather than for active runtime-control changes.
- Replay safety is an ABI problem only, rather than ABI plus semantics plus user trust.

## 5. Candidate Implementation Phases

### Candidate 01: Typed Latent Capsule ABI v3

- Name: Typed Latent Capsule ABI v3
- Suggested `phase_id`: `research`
- Core insight: Replace the single generic latent packet with a typed capsule carrying branch latents, user memory latents, repo delta latents, controller state, and provenance segments instead of one opaque tensor row.
- External inspiration or analogy: COCONUT continuous thought plus NASA software digital twins and PagedAttention-style page metadata [R1][R6][R31].
- Why it fits Saguaro and Anvil specifically: `ExecutionCapsule`, `LatentPacketABI`, and `MissionReplayDescriptor` already exist, so ABI evolution is structurally aligned with current code.
- Exact places in this codebase where it could wire in: `core/qsg/runtime_contracts.py`, `core/qsg/continuous_engine.py`, `core/qsg/latent_bridge.py`, `core/memory/latent_memory.py`, `tests/test_qsg_runtime_contracts.py`, `tests/test_qsg_continuous_engine.py`, `tests/test_latent_package_capture.py`.
- Existing primitives it can reuse: `ExecutionCapsule`, `LatentPacketABI`, `DeltaWatermark`, `RepoDeltaMemoryRecord`, `LatentPackageRecord`.
- New primitive, data flow, or subsystem needed: `TypedLatentSegment`, `CapsuleSegmentIndex`, and compatibility scoring across segment kinds.
- `repo_scope`: `core/qsg/runtime_contracts.py`, `core/qsg/continuous_engine.py`, `core/qsg/latent_bridge.py`, `core/memory/fabric/models.py`, `tests/test_qsg_runtime_contracts.py`, `tests/test_latent_package_capture.py`.
- `owning_specialist_type`: `runtime_architect`
- `allowed_writes`: ABI dataclasses, replay logic, latent package schemas, focused tests.
- `telemetry_contract`: Emit `latent_segment_count`, `latent_segment_kinds`, `latent_capsule_bytes`, `latent_capsule_compatibility_score`.
- `required_evidence`: Capsule round-trip tests, compatibility mismatch tests, replay descriptor coverage, storage migration receipts.
- `rollback_criteria`: Any ABI change that breaks `tests/test_qsg_runtime_contracts.py` or causes false-positive "compatible" replays on mismatched hidden dims.
- `promotion_gate`: Successful replay of mixed segment capsules with explicit mismatch reporting and no silent degradation.
- `success_criteria`: The runtime can capture, persist, inspect, and selectively replay typed latent segments instead of flattening everything into one vector.
- Why this creates value: It turns latent memory from an implementation trick into a controllable artifact.
- Why this creates moat: Typed replayable latent ABI is harder to replicate than a simple vector store.
- Main risk or failure mode: ABI sprawl without real downstream users.
- Smallest credible first experiment: Add `segment_kind="branch_state"` and `segment_kind="repo_delta"` to the existing packet and validate replay compatibility.
- Confidence level: 0.82

### Candidate 02: Multi-Resolution Latent Package Pyramid

- Name: Multi-Resolution Latent Package Pyramid
- Suggested `phase_id`: `feature_map`
- Core insight: Persist each save as three levels at minimum: micro branch residue, meso task summary, and macro session identity, rather than one tensor plus one summary string.
- External inspiration or analogy: Infini-attention, Recurrent Memory Transformer, and OS cache hierarchies [R11][R12].
- Why it fits Saguaro and Anvil specifically: `MemoryFabricStore` already supports multiple storage forms and `LatentPackageRecord` already carries compatibility metadata.
- Exact places in this codebase where it could wire in: `core/memory/latent_memory.py`, `core/memory/fabric/store.py`, `core/memory/fabric/models.py`, `core/qsg/latent_bridge.py`, `tests/test_memory_state_store.py`, `tests/test_memory_snapshot_restore.py`.
- Existing primitives it can reuse: latent package rows, safetensors storage, supporting memory IDs, expiry timestamps.
- New primitive, data flow, or subsystem needed: `LatentResolutionLevel` and save-time bundle composition.
- `repo_scope`: `core/memory/latent_memory.py`, `core/memory/fabric/store.py`, `core/memory/fabric/models.py`, `core/qsg/latent_bridge.py`, `tests/test_memory_state_store.py`, `tests/test_memory_snapshot_restore.py`.
- `owning_specialist_type`: `memory_systems_architect`
- `allowed_writes`: latent package builders, store schema extensions, snapshot tests.
- `telemetry_contract`: Emit `latent_resolution_level_hits`, `latent_resolution_storage_bytes`, `latent_resolution_restore_mode`.
- `required_evidence`: Retrieval from each tier, replay latency comparisons, no-regression storage integrity tests.
- `rollback_criteria`: Storage bloat exceeds target limits or replay logic becomes ambiguous.
- `promotion_gate`: Demonstrated ability to restore only the required resolution for a given task class.
- `success_criteria`: End users pay fewer tokens because the system can inject the smallest faithful memory artifact.
- Why this creates value: It reduces overinjection and increases replay precision.
- Why this creates moat: Multi-resolution memory engineering is harder than "summarize and embed."
- Main risk or failure mode: Too many tiers without a reliable selector.
- Smallest credible first experiment: Add `micro`, `summary`, and `session` tiers to `LatentMemory.build_package()` and store them under one memory ID.
- Confidence level: 0.78

### Candidate 03: Grover Resonance Oracle Field

- Name: Grover Resonance Oracle Field
- Suggested `phase_id`: `development`
- Core insight: Replace lexical token-name matching with a true oracle field built from latent vectors, repo delta memory, and invariant embeddings.
- External inspiration or analogy: Grover as signal amplifier, H2O heavy-hitter retention, and semantic cache alignment [R7][R28].
- Why it fits Saguaro and Anvil specifically: `GroverAmplifier` already accepts a semantic engine and `UnifiedChatLoop` already produces reinjected latent vectors.
- Exact places in this codebase where it could wire in: `core/qsg/grover.py`, `core/qsg/ollama_adapter.py`, `core/qsg/generator.py`, `core/unified_chat_loop.py`, `tests/test_qsg_components.py`, `tests/test_qsg_adapter_unification.py`, `tests/unit/test_coconut_high_fidelity.py`.
- Existing primitives it can reuse: `semantic_engine`, `LatentMemory.get_merged_vector()`, subagent latent merged signals, `RepoDeltaMemoryRecord`.
- New primitive, data flow, or subsystem needed: `ResonanceOracleField` with token-embedding lookup and invariant projection.
- `repo_scope`: `core/qsg/grover.py`, `core/qsg/ollama_adapter.py`, `core/qsg/generator.py`, `core/unified_chat_loop.py`, `tests/test_qsg_components.py`, `tests/test_qsg_adapter_unification.py`.
- `owning_specialist_type`: `inference_algorithm_architect`
- `allowed_writes`: Grover oracle logic, adapter wiring, inference tests.
- `telemetry_contract`: Emit `grover_oracle_source_count`, `grover_resonance_mean`, `grover_resonance_topk_overlap`, `grover_latent_bias_alpha`.
- `required_evidence`: Ambiguous generation tests where latent-aligned tokens become more probable without destabilizing outputs.
- `rollback_criteria`: Probability sharpening harms exact-match or semantic fidelity benchmarks.
- `promotion_gate`: Oracle field outperforms current lexical heuristic on controlled code-generation prompts.
- `success_criteria`: Grover acts as a real control surface over latent intent and repo invariants.
- Why this creates value: It makes latent memory operational at decode time.
- Why this creates moat: Most agent stacks never push memory into the decode oracle itself.
- Main risk or failure mode: Oracle overfits stale or noisy latent state.
- Smallest credible first experiment: Pass `subagent_latent_merged` from `UnifiedChatLoop` into `GroverAmplifier.amplify_with_resonance()` and compare token rank shifts.
- Confidence level: 0.84

### Candidate 04: Jacobi Frontier Verifier

- Name: Jacobi Frontier Verifier
- Suggested `phase_id`: `development`
- Core insight: Upgrade Jacobi from local smoothing to frontier management that verifies branch families against coherence, policy, and replay cost.
- External inspiration or analogy: Lookahead Decoding, Medusa tree verification, and F1 alternative strategy simulation [R2][R3][R35].
- Why it fits Saguaro and Anvil specifically: `SpeculativeFrontierPolicy` already exists and `NativeQSGEngine` already records proposed and accepted parallel tokens.
- Exact places in this codebase where it could wire in: `core/qsg/jacobi_refiner.py`, `core/qsg/runtime_contracts.py`, `core/native/native_qsg_engine.py`, `tests/test_qsg_components.py`, `tests/test_native_qsg_engine.py`, `tests/test_qsg_pipeline.py`.
- Existing primitives it can reuse: `SpeculativeFrontierPolicy`, runtime telemetry accept/reject counts, coconut path counts, `QSGStatePager`.
- New primitive, data flow, or subsystem needed: `JacobiFrontierState`, `BranchCoherenceMatrix`, and branch-cost-aware verification.
- `repo_scope`: `core/qsg/jacobi_refiner.py`, `core/qsg/runtime_contracts.py`, `core/native/native_qsg_engine.py`, `tests/test_qsg_components.py`, `tests/test_native_qsg_engine.py`.
- `owning_specialist_type`: `speculative_decode_architect`
- `allowed_writes`: Jacobi logic, frontier policy wiring, runtime metrics, tests.
- `telemetry_contract`: Emit `jacobi_frontier_width`, `jacobi_branch_survival_rate`, `jacobi_verify_cost_ms`, `jacobi_branch_entropy`.
- `required_evidence`: Controlled tests showing fewer bad branch acceptances and higher acceptance ratios on structured prompts.
- `rollback_criteria`: Verification cost overwhelms latency gains or frontier width oscillates.
- `promotion_gate`: Improved branch acceptance on code prompts without TTFT regression beyond agreed threshold.
- `success_criteria`: Jacobi becomes a branch verifier instead of a placeholder denoiser.
- Why this creates value: It makes speculative parallelism coherent instead of opportunistic.
- Why this creates moat: Verification-aware frontier control tied to latent memory is unusual.
- Main risk or failure mode: Too much controller complexity for little win on small prompts.
- Smallest credible first experiment: Introduce a branch score matrix derived from neighboring token agreement plus coconut amplitude priors.
- Confidence level: 0.76

### Candidate 05: COCONUT Branch Reservoir

- Name: COCONUT Branch Reservoir
- Suggested `phase_id`: `analysis_upgrade`
- Core insight: Persist not just the winning refined embedding but the high-value losing branches, their amplitudes, and termination reasons for later reuse.
- External inspiration or analogy: COCONUT BFS behavior, Titans surprise-based memorization, and racing telemetry of near-miss lines [R1][R15][R35].
- Why it fits Saguaro and Anvil specifically: `ContinuousThoughtBlock` already surfaces amplitudes and adaptive metrics, and the repo already records session records in tests.
- Exact places in this codebase where it could wire in: `core/reasoning/coconut.py`, `core/unified_chat_loop.py`, `core/memory/latent_memory.py`, `tests/test_coconut_session_record.py`, `tests/test_coconut_accuracy.py`, `tests/test_adaptive_coconut.py`.
- Existing primitives it can reuse: `last_session_record`, adaptive exploration metrics, `latent_memory.add_thought()`.
- New primitive, data flow, or subsystem needed: `BranchReservoirEntry` and write-admission policy based on amplitude, novelty, and downstream payoff.
- `repo_scope`: `core/reasoning/coconut.py`, `core/unified_chat_loop.py`, `core/memory/latent_memory.py`, `tests/test_coconut_session_record.py`, `tests/test_coconut_accuracy.py`.
- `owning_specialist_type`: `reasoning_systems_architect`
- `allowed_writes`: coconut session record format, latent-memory schema, reinjection policy.
- `telemetry_contract`: Emit `coconut_branch_reservoir_size`, `coconut_branch_novelty`, `coconut_branch_reuse_hits`, `coconut_branch_evictions`.
- `required_evidence`: Session-record tests, replay reuse tests, and evidence that saved branches improve later branch selection.
- `rollback_criteria`: Reservoir increases storage without measurable reuse or destabilizes reasoning.
- `promotion_gate`: Reused reservoir branches improve either file reranking or prompt compression quality on repeated tasks.
- `success_criteria`: COCONUT transitions from ephemeral search to reusable branch memory.
- Why this creates value: It captures expensive reasoning work rather than discarding it.
- Why this creates moat: Persistent branch memory is a deeper primitive than conversation summarization.
- Main risk or failure mode: Branches may be too task-specific to generalize.
- Smallest credible first experiment: Save top-2 non-winning branches with amplitude and termination metadata for repeated repo-analysis prompts.
- Confidence level: 0.81

### Candidate 06: Surprise-Weighted Latent Write Controller

- Name: Surprise-Weighted Latent Write Controller
- Suggested `phase_id`: `development`
- Core insight: Memory writes should depend on novelty, surprise, branch utility, and replay likelihood instead of appending every vector-shaped artifact.
- External inspiration or analogy: Titans surprise updates and H2O heavy-hitter retention [R7][R15].
- Why it fits Saguaro and Anvil specifically: `NativeQSGEngine` already exposes performance twin and controller state; the missing piece is using that to govern writes.
- Exact places in this codebase where it could wire in: `core/memory/latent_memory.py`, `core/qsg/latent_bridge.py`, `core/unified_chat_loop.py`, `core/qsg/runtime_contracts.py`, `tests/test_memory_governance.py`, `tests/test_subagent_latent_pipeline.py`.
- Existing primitives it can reuse: `controller_state`, adaptive complexity, subagent latent signals, `MemoryTierPolicy`.
- New primitive, data flow, or subsystem needed: `LatentWriteDecisionRecord`.
- `repo_scope`: `core/memory/latent_memory.py`, `core/qsg/latent_bridge.py`, `core/unified_chat_loop.py`, `core/qsg/runtime_contracts.py`, `tests/test_memory_governance.py`.
- `owning_specialist_type`: `memory_policy_architect`
- `allowed_writes`: write-policy logic, latent package admission rules, tests.
- `telemetry_contract`: Emit `latent_write_admit_rate`, `latent_write_reasons`, `latent_write_surprise_score`, `latent_write_expected_reuse`.
- `required_evidence`: Admission control tests and reduced latent storage growth with equal or better replay hit rate.
- `rollback_criteria`: Important memories are dropped or user continuity degrades.
- `promotion_gate`: Storage growth is reduced without worse recall on repeated tasks.
- `success_criteria`: Memory writes become selective and economically meaningful.
- Why this creates value: It reduces memory junk.
- Why this creates moat: Good memory systems are mostly about what not to save.
- Main risk or failure mode: Surprise metrics can mis-rank mundane but essential information.
- Smallest credible first experiment: Add a write gate that favors high-novelty subagent latent payloads and execution capsules with strong replay compatibility.
- Confidence level: 0.79

### Candidate 07: Repo Delta Latent Compiler

- Name: Repo Delta Latent Compiler
- Suggested `phase_id`: `research`
- Core insight: Convert repo changes into typed latent assets that can bias retrieval, replay, and decode-time resonance without re-reading whole files.
- External inspiration or analogy: Graphiti temporal graphs and software digital twin delta ledgers [R23][R31].
- Why it fits Saguaro and Anvil specifically: `RepoDeltaMemoryRecord` already exists and is attached to compatibility metadata.
- Exact places in this codebase where it could wire in: `core/qsg/runtime_contracts.py`, `core/qsg/latent_bridge.py`, `core/memory/fabric/models.py`, `core/memory/fabric/retrieval_planner.py`, `tests/test_qsg_runtime_contracts.py`, `tests/test_memory_research_ingest.py`.
- Existing primitives it can reuse: `DeltaWatermark`, `RepoDeltaMemoryRecord`, `supporting_memory_ids`.
- New primitive, data flow, or subsystem needed: `RepoDeltaLatentArtifact` plus delta summarization projector.
- `repo_scope`: `core/qsg/runtime_contracts.py`, `core/qsg/latent_bridge.py`, `core/memory/fabric/models.py`, `core/memory/fabric/retrieval_planner.py`, `tests/test_memory_research_ingest.py`.
- `owning_specialist_type`: `repo_memory_architect`
- `allowed_writes`: repo-delta models, retrieval planner logic, package composition.
- `telemetry_contract`: Emit `repo_delta_latent_count`, `repo_delta_changed_path_count`, `repo_delta_replay_block_rate`.
- `required_evidence`: Repeated code-review tasks show lower file-read volume with stable answer quality.
- `rollback_criteria`: Delta latents drift from actual repo state or cause stale retrieval.
- `promotion_gate`: Delta-aware retrieval consistently beats naive recent-file heuristics.
- `success_criteria`: Recent repo evolution becomes queryable without tokenizing entire changed files every time.
- Why this creates value: It compresses the "what changed" problem.
- Why this creates moat: Tightly coupling source-control delta semantics to latent replay is rare.
- Main risk or failure mode: Overcompression hides exact code facts when the user needs literal text.
- Smallest credible first experiment: Project changed file lists and semantic impact hints into a dedicated latent package supporting the main request capsule.
- Confidence level: 0.73

### Candidate 08: Non-Prefix Latent/KV Stitcher

- Name: Non-Prefix Latent/KV Stitcher
- Suggested `phase_id`: `research`
- Core insight: Reuse saved latent and KV fragments even when the reused content is not at the prompt prefix and appears in different order.
- External inspiration or analogy: CacheBlend, KVLink, KVShare, LMCache [R27][R28][R29][R30].
- Why it fits Saguaro and Anvil specifically: Repo-analysis prompts and multi-agent evidence assembly constantly reorder common chunks.
- Exact places in this codebase where it could wire in: `core/qsg/latent_bridge.py`, `core/native/native_qsg_engine.py`, `core/qsg/continuous_engine.py`, `core/memory/fabric/store.py`, `tests/test_qsg_integration.py`, `tests/test_native_qsg_engine.py`.
- Existing primitives it can reuse: latent package compatibility metadata, `QSGStatePager`, prompt cache telemetry, controller state.
- New primitive, data flow, or subsystem needed: `LatentStitchPlan` and selective recompute patching.
- `repo_scope`: `core/qsg/latent_bridge.py`, `core/native/native_qsg_engine.py`, `core/qsg/continuous_engine.py`, `core/memory/fabric/store.py`, `tests/test_native_qsg_engine.py`.
- `owning_specialist_type`: `cache_reuse_architect`
- `allowed_writes`: runtime status, stitch planner, replay loader, tests.
- `telemetry_contract`: Emit `non_prefix_reuse_attempts`, `non_prefix_reuse_hits`, `stitch_patch_tokens`, `stitch_quality_delta`.
- `required_evidence`: TTFT improvement on repeated repo-analysis workloads with reordered context blocks.
- `rollback_criteria`: Quality regressions or compatibility complexity exceeds payoff.
- `promotion_gate`: Controlled workloads show measurable TTFT reduction without unacceptable semantic drift.
- `success_criteria`: Anvil reuses work across semantically similar but structurally reordered tasks.
- Why this creates value: It attacks the real prompt-reuse pattern seen in agent systems.
- Why this creates moat: Non-prefix state reuse is still frontier work.
- Main risk or failure mode: Patch logic becomes too model-specific for CPU-first portability.
- Smallest credible first experiment: Stitch latent package segments for reused repo documents and recompute only the join boundary.
- Confidence level: 0.62

### Candidate 09: Memory Tier Compiler

- Name: Memory Tier Compiler
- Suggested `phase_id`: `feature_map`
- Core insight: Turn memory tier selection into a compiler pass that decides save format, read format, expiry, visibility, and fallback chain per artifact class.
- External inspiration or analogy: Letta core versus archival memory, OS storage classes, and NASA ISHM mode management [R17][R18][R33].
- Why it fits Saguaro and Anvil specifically: `MemoryTierPolicy` already exists but mostly as a selector for replay, not a compiler for artifact lifecycle.
- Exact places in this codebase where it could wire in: `core/qsg/runtime_contracts.py`, `core/memory/fabric/store.py`, `core/qsg/latent_bridge.py`, `core/unified_chat_loop.py`, `tests/test_qsg_runtime_contracts.py`, `tests/test_memory_backend_profiles.py`.
- Existing primitives it can reuse: `MemoryTierPolicy`, retention policy, latent package metadata, feedback/read records.
- New primitive, data flow, or subsystem needed: `CompiledMemoryPlan`.
- `repo_scope`: `core/qsg/runtime_contracts.py`, `core/memory/fabric/store.py`, `core/qsg/latent_bridge.py`, `core/unified_chat_loop.py`, `tests/test_memory_backend_profiles.py`.
- `owning_specialist_type`: `memory_compiler_architect`
- `allowed_writes`: controller policy, store routing, loop integration, tests.
- `telemetry_contract`: Emit `memory_plan_kind`, `memory_plan_selected_tier`, `memory_plan_expiry_s`, `memory_plan_visibility`.
- `required_evidence`: Artifact-type matrix with passing tests for each tier path.
- `rollback_criteria`: Increased routing complexity without better token economy or replay quality.
- `promotion_gate`: Every major artifact class has an explicit compiled memory plan and fallback path.
- `success_criteria`: The system knows where each memory object belongs before writing it.
- Why this creates value: It unifies fragmented memory behavior.
- Why this creates moat: Lifecycle compilation is a stronger center of gravity than ad hoc memory writes.
- Main risk or failure mode: Bureaucratic overdesign.
- Smallest credible first experiment: Compile separate plans for `subagent_latent`, `execution_capsule`, and `user_preference`.
- Confidence level: 0.86

### Candidate 10: Subagent Latent Treaty

- Name: Subagent Latent Treaty
- Suggested `phase_id`: `development`
- Core insight: Standardize what every subagent may emit as latent payload, how it is normalized, and how it is merged into master state.
- External inspiration or analogy: shared-memory multi-agent systems and SRMT-style shared recurrent memory [R36](https://arxiv.org/abs/2501.13200).
- Why it fits Saguaro and Anvil specifically: `UnifiedChatLoop` already extracts and merges subagent latent payloads, but the contract is informal.
- Exact places in this codebase where it could wire in: `core/unified_chat_loop.py`, `core/agents/subagent.py`, `core/memory/latent_memory.py`, `tests/test_subagent_latent_pipeline.py`, `tests/test_subagent_quality_gate.py`.
- Existing primitives it can reuse: `_extract_subagent_latent_payload()`, `_append_subagent_latent_payload()`, `latent_memory.add_thought()`.
- New primitive, data flow, or subsystem needed: `SubagentLatentContract` with versioning and confidence fields.
- `repo_scope`: `core/unified_chat_loop.py`, `core/agents/subagent.py`, `core/memory/latent_memory.py`, `tests/test_subagent_latent_pipeline.py`.
- `owning_specialist_type`: `multi_agent_architect`
- `allowed_writes`: subagent response schema, master merge logic, tests.
- `telemetry_contract`: Emit `subagent_latent_signal_count`, `subagent_latent_merge_confidence`, `subagent_latent_contract_version`.
- `required_evidence`: Heterogeneous subagents can emit compatible latent payloads without ad hoc parsing branches.
- `rollback_criteria`: Contract becomes too strict and breaks existing specialists.
- `promotion_gate`: Research, specialist, and repo-analysis subagents all pass treaty conformance tests.
- `success_criteria`: Multi-agent latent exchange becomes predictable and evolvable.
- Why this creates value: It reduces brittle glue logic.
- Why this creates moat: Strong inter-agent latent contracts are rare in agent platforms.
- Main risk or failure mode: Contract rigidity suppresses useful specialist variation.
- Smallest credible first experiment: Add `contract_version`, `confidence`, and `artifact_kind` to current payload extraction.
- Confidence level: 0.85

### Candidate 11: User Memory Blocks over Latent Substrates

- Name: User Memory Blocks over Latent Substrates
- Suggested `phase_id`: `convergence`
- Core insight: Give users editable memory blocks backed by latent packages rather than forcing a choice between raw text summaries and hidden vector state.
- External inspiration or analogy: Letta memory blocks and Zep automated context assembly [R17][R21].
- Why it fits Saguaro and Anvil specifically: The repo already has memory fabrics, but the visible product surface is still thin.
- Exact places in this codebase where it could wire in: `core/unified_chat_loop.py`, `cli/commands/memory.py`, `core/memory/fabric/store.py`, `core/prompts/system_prompt_builder.py`, `tests/test_memory_conversation_ingest.py`, `tests/test_memory_governance.py`.
- Existing primitives it can reuse: `MemoryFabricStore.create_memory()`, latent package IDs, summary text, retention classes.
- New primitive, data flow, or subsystem needed: `PinnedMemoryBlock`, `EditableMemoryProjection`.
- `repo_scope`: `cli/commands/memory.py`, `core/unified_chat_loop.py`, `core/memory/fabric/store.py`, `core/prompts/system_prompt_builder.py`, `tests/test_memory_conversation_ingest.py`.
- `owning_specialist_type`: `memory_product_architect`
- `allowed_writes`: CLI surfaces, memory schemas, prompt injection rules, tests.
- `telemetry_contract`: Emit `user_memory_block_reads`, `user_memory_block_writes`, `user_memory_block_token_cost`.
- `required_evidence`: End-to-end tests showing user-pinned memories change future behavior predictably.
- `rollback_criteria`: User-visible memory becomes misleading or silently stale.
- `promotion_gate`: User can inspect, pin, unpin, and delete high-value memories with deterministic effects.
- `success_criteria`: End users perceive continuity and control without paying the full token cost every turn.
- Why this creates value: It productizes the memory stack.
- Why this creates moat: Most systems either hide memory or expose crude note-taking.
- Main risk or failure mode: User mental model diverges from actual latent behavior.
- Smallest credible first experiment: Add a "pinned latent memory block" command that stores summary text plus typed latent package reference.
- Confidence level: 0.77

### Candidate 12: Temporal Knowledge Graph Memory Plane

- Name: Temporal Knowledge Graph Memory Plane
- Suggested `phase_id`: `research`
- Core insight: Some memory should be represented as facts with validity intervals and relationships, not as raw text or dense vectors alone.
- External inspiration or analogy: Zep/Graphiti temporal knowledge graphs [R22][R23].
- Why it fits Saguaro and Anvil specifically: repo delta, user preference change, and tool outcomes are naturally temporal and relational.
- Exact places in this codebase where it could wire in: `core/memory/fabric/models.py`, `core/memory/fabric/store.py`, `core/memory/fabric/retrieval_planner.py`, `core/qsg/latent_bridge.py`, `tests/test_memory_temporal_hierarchy.py`, `tests/test_memory_embeddings.py`.
- Existing primitives it can reuse: `MemoryObject`, `MemoryEdge`, `RepoDeltaMemoryRecord`, read and feedback records.
- New primitive, data flow, or subsystem needed: temporal fact nodes and relation validity windows.
- `repo_scope`: `core/memory/fabric/models.py`, `core/memory/fabric/store.py`, `core/memory/fabric/retrieval_planner.py`, `tests/test_memory_temporal_hierarchy.py`.
- `owning_specialist_type`: `graph_memory_architect`
- `allowed_writes`: fabric models, store schema, retrieval logic, tests.
- `telemetry_contract`: Emit `temporal_fact_count`, `temporal_conflict_resolutions`, `graph_memory_query_hits`.
- `required_evidence`: Correct handling of changed user preferences and changed repo conditions across time.
- `rollback_criteria`: Graph maintenance overhead dominates benefit.
- `promotion_gate`: Temporal graph lookups improve dynamic-state questions over baseline dense retrieval.
- `success_criteria`: The system can remember both what was true and what is true now.
- Why this creates value: It fixes stale-memory contradictions.
- Why this creates moat: Temporal reasoning is a deeper capability than flat retrieval.
- Main risk or failure mode: Overcomplication for small deployments.
- Smallest credible first experiment: Store preference changes and repo delta changes as temporal edges and use them in retrieval planner reranking.
- Confidence level: 0.68

### Candidate 13: Persona and Preference Latent Anchors

- Name: Persona and Preference Latent Anchors
- Suggested `phase_id`: `development`
- Core insight: Persist user style, tool preferences, trust boundaries, and response-shape preferences as compact anchors that steer prompt assembly and decoding.
- External inspiration or analogy: Letta core memory blocks and Mem0 direct imports [R17][R19].
- Why it fits Saguaro and Anvil specifically: `SystemPromptBuilder` and `UnifiedChatLoop` already assemble structured prompt context and can consume stable preference artifacts.
- Exact places in this codebase where it could wire in: `core/prompts/system_prompt_builder.py`, `core/unified_chat_loop.py`, `core/memory/fabric/store.py`, `tests/test_aes_prompt_builder.py`, `tests/test_granite_prompt_contract.py`.
- Existing primitives it can reuse: summary text, memory objects, prompt-manager sections.
- New primitive, data flow, or subsystem needed: `PreferenceAnchorRecord`.
- `repo_scope`: `core/prompts/system_prompt_builder.py`, `core/unified_chat_loop.py`, `core/memory/fabric/store.py`, `tests/test_aes_prompt_builder.py`.
- `owning_specialist_type`: `prompt_memory_architect`
- `allowed_writes`: prompt builder integration, memory objects, tests.
- `telemetry_contract`: Emit `preference_anchor_hits`, `preference_anchor_conflicts`, `preference_anchor_token_savings`.
- `required_evidence`: Stable adherence to user style with lower prompt footprint.
- `rollback_criteria`: Preference anchors overconstrain responses or become sticky after user changes.
- `promotion_gate`: Preference updates are reflected within one session and across later sessions.
- `success_criteria`: User style continuity no longer depends on long repeated instructions.
- Why this creates value: It directly improves chat ergonomics.
- Why this creates moat: Persistent preference steering backed by latent anchors is highly reusable.
- Main risk or failure mode: Anchors become simplistic stereotypes.
- Smallest credible first experiment: Save concise response-preference anchors and inject them ahead of generic style instructions.
- Confidence level: 0.83

### Candidate 14: Tool Outcome Latent Compression

- Name: Tool Outcome Latent Compression
- Suggested `phase_id`: `analysis_upgrade`
- Core insight: Compress tool results into typed latent outcome packages so the loop remembers what mattered without re-inserting raw logs.
- External inspiration or analogy: OpenAI prompt caching best practices plus compressed context memory [R9][R24].
- Why it fits Saguaro and Anvil specifically: `UnifiedChatLoop` already has context compression memory and tool result labeling logic.
- Exact places in this codebase where it could wire in: `core/unified_chat_loop.py`, `core/task_memory.py`, `core/memory/latent_memory.py`, `tests/test_memory_conversation_ingest.py`, `tests/test_pipeline_manager.py`.
- Existing primitives it can reuse: context updates, compression memory, task memory manager.
- New primitive, data flow, or subsystem needed: `ToolOutcomeLatent`.
- `repo_scope`: `core/unified_chat_loop.py`, `core/task_memory.py`, `core/memory/latent_memory.py`, `tests/test_memory_conversation_ingest.py`.
- `owning_specialist_type`: `context_economics_architect`
- `allowed_writes`: compression logic, latent write paths, tests.
- `telemetry_contract`: Emit `tool_outcome_compression_ratio`, `tool_outcome_reuse_hits`, `tool_outcome_token_avoidance`.
- `required_evidence`: Repeated tool-heavy workflows consume fewer context tokens without losing answer quality.
- `rollback_criteria`: Important tool detail is lost and later actions fail.
- `promotion_gate`: Compression keeps downstream task success flat while shrinking tool-result footprint.
- `success_criteria`: Tool-heavy sessions stop ballooning the main context.
- Why this creates value: It reduces the most common source of context waste.
- Why this creates moat: Outcome-centric compression is more useful than naive summarization.
- Main risk or failure mode: Overcompression hides edge-case tool evidence.
- Smallest credible first experiment: Compress successful file-read and search results into latent outcome records with file refs preserved.
- Confidence level: 0.88

### Candidate 15: Latent Microscope and Drift Debugger

- Name: Latent Microscope and Drift Debugger
- Suggested `phase_id`: `analysis_upgrade`
- Core insight: Provide operator-facing inspection of what a latent package contains, what it aligns to, and why replay was admitted or blocked.
- External inspiration or analogy: flight recorder tooling, JSTAR replay diagnostics, and F1 telemetry dashboards [R31][R35].
- Why it fits Saguaro and Anvil specifically: `BlackBoxRecorder` already aggregates replay context and `MissionReplayDescriptor` already exists.
- Exact places in this codebase where it could wire in: `core/telemetry/black_box.py`, `core/qsg/latent_bridge.py`, `cli/commands/memory.py`, `tests/test_black_box_recorder.py`, `tests/test_runtime_telemetry.py`.
- Existing primitives it can reuse: event store, replay tapes, mission replay descriptors, telemetry snapshots.
- New primitive, data flow, or subsystem needed: `LatentInspectionReport`.
- `repo_scope`: `core/telemetry/black_box.py`, `core/qsg/latent_bridge.py`, `cli/commands/memory.py`, `tests/test_black_box_recorder.py`.
- `owning_specialist_type`: `observability_contract_engineer`
- `allowed_writes`: telemetry enrichment, CLI inspection, tests.
- `telemetry_contract`: Emit `latent_debug_sessions`, `latent_debug_replay_block_causes`, `latent_inspection_duration_ms`.
- `required_evidence`: Operators can explain replay failures without reading raw tensors.
- `rollback_criteria`: Debug surface leaks sensitive data or becomes too expensive.
- `promotion_gate`: Replay diagnostics become deterministic and operator-usable.
- `success_criteria`: Latent features become debuggable enough to maintain.
- Why this creates value: It lowers operational fear around hidden-state systems.
- Why this creates moat: Debuggability is a real barrier to shipping latent features.
- Main risk or failure mode: Inspection stays superficial and fails to answer operator questions.
- Smallest credible first experiment: Add a CLI subcommand that shows capsule metadata, compatibility mismatches, and supporting memory IDs.
- Confidence level: 0.87

### Candidate 16: Performance Twin as Memory Governor

- Name: Performance Twin as Memory Governor
- Suggested `phase_id`: `deep_test_audit`
- Core insight: Use the existing performance twin to actively tune coconut depth, write rates, frontier width, and replay aggressiveness in-session.
- External inspiration or analogy: NASA health management and F1 race strategy adjustment under telemetry [R33][R35].
- Why it fits Saguaro and Anvil specifically: `NativeQSGEngine.get_runtime_status()` already emits a `performance_twin`.
- Exact places in this codebase where it could wire in: `core/native/native_qsg_engine.py`, `core/unified_chat_loop.py`, `core/qsg/runtime_contracts.py`, `tests/test_qsg_runtime_contracts.py`, `tests/test_runtime_control_policy.py`.
- Existing primitives it can reuse: performance envelope, controller state, runtime control policy.
- New primitive, data flow, or subsystem needed: `MemoryGovernorDecision`.
- `repo_scope`: `core/native/native_qsg_engine.py`, `core/unified_chat_loop.py`, `core/qsg/runtime_contracts.py`, `tests/test_runtime_control_policy.py`.
- `owning_specialist_type`: `telemetry_systems_engineer`
- `allowed_writes`: governor policy, control-loop wiring, tests.
- `telemetry_contract`: Emit `memory_governor_actions`, `memory_governor_reason`, `memory_governor_outcome`.
- `required_evidence`: Governor reacts correctly to queue-bound, drift-bound, and memory-bound regimes.
- `rollback_criteria`: Controller oscillation or unstable session behavior.
- `promotion_gate`: Governor decisions improve either latency or memory economy in benchmarked regimes.
- `success_criteria`: Runtime telemetry meaningfully changes memory behavior rather than only being recorded.
- Why this creates value: It closes the loop between observability and action.
- Why this creates moat: Self-governing memory behavior is difficult to clone cleanly.
- Main risk or failure mode: Policy complexity outruns data quality.
- Smallest credible first experiment: Reduce coconut depth and disable low-value writes under queue-bound conditions.
- Confidence level: 0.8

### Candidate 17: Latent Governance Firewall

- Name: Latent Governance Firewall
- Suggested `phase_id`: `deep_test_audit`
- Core insight: Apply governance and compatibility checks to latent artifacts before they influence prompt assembly or decode-time steering.
- External inspiration or analogy: IV&V gatekeeping and Zero Data Retention constraints around cached KV state [R32][R24].
- Why it fits Saguaro and Anvil specifically: The repo already contains AES gates, compliance context, and runtime invariant checks.
- Exact places in this codebase where it could wire in: `core/qsg/runtime_contracts.py`, `core/qsg/latent_bridge.py`, `core/unified_chat_loop.py`, `core/aes/runtime_gates/telemetry_contract_gate.py`, `tests/test_qsg_runtime_contracts.py`, `tests/test_aes_runtime_gates.py`.
- Existing primitives it can reuse: `evaluate_qsg_runtime_invariants()`, compliance checkpoints, telemetry contract gates.
- New primitive, data flow, or subsystem needed: `LatentGovernanceDecision`.
- `repo_scope`: `core/qsg/runtime_contracts.py`, `core/qsg/latent_bridge.py`, `core/unified_chat_loop.py`, `core/aes/runtime_gates/telemetry_contract_gate.py`, `tests/test_aes_runtime_gates.py`.
- `owning_specialist_type`: `verification_engineer`
- `allowed_writes`: gate logic, evidence fields, tests.
- `telemetry_contract`: Emit `latent_firewall_blocks`, `latent_firewall_warnings`, `latent_firewall_passes`.
- `required_evidence`: Replay and steering are blocked when compatibility, expiry, or provenance requirements fail.
- `rollback_criteria`: Excessive false positives that suppress useful memory behavior.
- `promotion_gate`: Governance policy catches intentionally malformed latent capsules in tests.
- `success_criteria`: Latent memory becomes auditable and safe enough for higher-assurance paths.
- Why this creates value: It protects the feature stack from hidden-state corruption.
- Why this creates moat: Safety at the latent interface is more defensible than safety only at prompts.
- Main risk or failure mode: Firewall becomes a blunt deny-all mechanism.
- Smallest credible first experiment: Require provenance and compatibility match before using latent payloads in Grover or replay.
- Confidence level: 0.84

### Candidate 18: Path-Specific Rehydration

- Name: Path-Specific Rehydration
- Suggested `phase_id`: `development`
- Core insight: Restore only the branch or subspace needed for the current request rather than rehydrating full latent packages.
- External inspiration or analogy: page-level restoration and cache-line fetch rather than whole-process restore [R6][R29].
- Why it fits Saguaro and Anvil specifically: `QSGStatePager` already imports rows and `QSGLatentBridge` already composes replay packages.
- Exact places in this codebase where it could wire in: `core/qsg/state_pager.py`, `core/qsg/continuous_engine.py`, `core/qsg/latent_bridge.py`, `tests/test_qsg_state_pager.py`, `tests/test_qsg_continuous_engine.py`.
- Existing primitives it can reuse: row refs, request-state import/export, latent packet metadata.
- New primitive, data flow, or subsystem needed: segment-indexed row mapping for partial restore.
- `repo_scope`: `core/qsg/state_pager.py`, `core/qsg/continuous_engine.py`, `core/qsg/latent_bridge.py`, `tests/test_qsg_state_pager.py`.
- `owning_specialist_type`: `state_runtime_architect`
- `allowed_writes`: pager import/export logic, restore path, tests.
- `telemetry_contract`: Emit `partial_rehydrate_rows`, `partial_rehydrate_ms`, `partial_rehydrate_success_rate`.
- `required_evidence`: Partial restore consumes less time or memory than full restore while preserving quality.
- `rollback_criteria`: Partial restore causes semantic mismatch or unstable state handles.
- `promotion_gate`: Pager-level tests prove deterministic partial import/export behavior.
- `success_criteria`: The runtime can restore only what the next reasoning phase needs.
- Why this creates value: It keeps replay cheap.
- Why this creates moat: Fine-grained rehydration is a real systems feature, not a marketing feature.
- Main risk or failure mode: Partial latent restore breaks hidden dependencies.
- Smallest credible first experiment: Restore only branch-state segments during coconut re-entry and compare against full replay.
- Confidence level: 0.71

### Candidate 19: Hybrid Multivector plus HD Bundle Retrieval

- Name: Hybrid Multivector plus HD Bundle Retrieval
- Suggested `phase_id`: `feature_map`
- Core insight: Use multivectors and HD bundles as complementary memory surfaces for different retrieval shapes instead of treating dense vectors as the universal format.
- External inspiration or analogy: memory systems with separate associative and exact-match lanes [R12][R13].
- Why it fits Saguaro and Anvil specifically: `MemoryFabricStore` already supports `put_multivector()` and `put_hd_bundle()`.
- Exact places in this codebase where it could wire in: `core/memory/fabric/retrieval_planner.py`, `core/memory/fabric/store.py`, `core/memory/fabric/projectors.py`, `tests/test_memory_multivector.py`, `tests/test_memory_embeddings.py`.
- Existing primitives it can reuse: multivector rows, HD bundle rows, reranker infrastructure.
- New primitive, data flow, or subsystem needed: retrieval planner fusion scoring across dense, multivector, HD, and latent package metadata.
- `repo_scope`: `core/memory/fabric/retrieval_planner.py`, `core/memory/fabric/store.py`, `core/memory/fabric/projectors.py`, `tests/test_memory_multivector.py`.
- `owning_specialist_type`: `retrieval_architect`
- `allowed_writes`: retrieval planner, projection policies, tests.
- `telemetry_contract`: Emit `retrieval_lane_hits`, `retrieval_lane_precision`, `retrieval_fusion_weights`.
- `required_evidence`: Hybrid retrieval beats dense-only recall on memory-heavy tests.
- `rollback_criteria`: Retrieval becomes too expensive or too opaque.
- `promotion_gate`: Fusion scoring outperforms current planner on at least one representative benchmark family.
- `success_criteria`: Different memory types are used for what they are good at.
- Why this creates value: It increases retrieval fidelity without more raw context.
- Why this creates moat: Hybrid retrieval plus replay compatibility is a richer substrate than standard vector search.
- Main risk or failure mode: Signal fusion becomes hard to tune.
- Smallest credible first experiment: Add HD bundle match as a bonus term when reranking dense retrieval results.
- Confidence level: 0.75

### Candidate 20: Utility-Based Forgetting and TTL Optimization

- Name: Utility-Based Forgetting and TTL Optimization
- Suggested `phase_id`: `analysis_upgrade`
- Core insight: Expiry should depend on reuse, utility, novelty decay, and user pinning, not just a static TTL.
- External inspiration or analogy: cache eviction, heavy-hitter retention, and temporal fact invalidation [R7][R21].
- Why it fits Saguaro and Anvil specifically: `LatentPackageRecord` already contains `expires_at`, while read and feedback records already exist in `MemoryFabricStore`.
- Exact places in this codebase where it could wire in: `core/memory/fabric/store.py`, `core/memory/fabric/jobs.py`, `core/qsg/latent_bridge.py`, `tests/test_memory_benchmarks.py`, `tests/test_memory_governance.py`.
- Existing primitives it can reuse: feedback records, read records, expiry timestamps, retention classes.
- New primitive, data flow, or subsystem needed: `MemoryUtilityScore`.
- `repo_scope`: `core/memory/fabric/store.py`, `core/memory/fabric/jobs.py`, `core/qsg/latent_bridge.py`, `tests/test_memory_benchmarks.py`.
- `owning_specialist_type`: `memory_economics_engineer`
- `allowed_writes`: expiry jobs, utility scoring, tests.
- `telemetry_contract`: Emit `memory_evictions`, `memory_utility_mean`, `memory_pinned_fraction`, `memory_zombie_reads`.
- `required_evidence`: Lower storage growth with no loss in important memory recall.
- `rollback_criteria`: Important packages expire too early or user-pinned memory gets evicted.
- `promotion_gate`: Utility-based expiry beats static expiry on replay-hit efficiency.
- `success_criteria`: Memory store growth remains bounded while high-value recall stays high.
- Why this creates value: It makes long-lived deployments sustainable.
- Why this creates moat: Good forgetting is as important as good storage.
- Main risk or failure mode: Utility metrics lag real user needs.
- Smallest credible first experiment: Extend expiry for packages with repeated reads and positive feedback, shorten expiry for cold low-confidence packages.
- Confidence level: 0.89

### Candidate 21: CPU-First Latent Codecs

- Name: CPU-First Latent Codecs
- Suggested `phase_id`: `development`
- Core insight: Compress latent segments for CPU residency using layout-aware codecs rather than raw float32 safetensors everywhere.
- External inspiration or analogy: CacheGen, LMCache, and page compression [R30][R37](https://github.com/UChi-JCL/CacheGen).
- Why it fits Saguaro and Anvil specifically: the repo is explicitly CPU-first and already stores tensor blobs as files.
- Exact places in this codebase where it could wire in: `core/memory/fabric/store.py`, `core/qsg/latent_bridge.py`, `core/qsg/continuous_engine.py`, `tests/test_memory_state_store.py`, `tests/test_memory_benchmarks.py`.
- Existing primitives it can reuse: tensor URI, tensor format field, safetensors IO.
- New primitive, data flow, or subsystem needed: codec registry for `float16`, product-quantized, and delta-coded latent payloads.
- `repo_scope`: `core/memory/fabric/store.py`, `core/qsg/latent_bridge.py`, `core/qsg/continuous_engine.py`, `tests/test_memory_state_store.py`.
- `owning_specialist_type`: `systems_performance_engineer`
- `allowed_writes`: storage formats, codec selection, tests.
- `telemetry_contract`: Emit `latent_codec`, `latent_codec_bytes_saved`, `latent_codec_restore_ms`.
- `required_evidence`: Storage reduction and restore-time measurements with bounded quality loss.
- `rollback_criteria`: Replay mismatch or unacceptable accuracy loss.
- `promotion_gate`: At least one compressed format beats float32 on size and keeps replay behavior within tolerance.
- `success_criteria`: CPU-first deployments can keep larger memory inventories hot.
- Why this creates value: It raises memory density.
- Why this creates moat: Systems-grade codec work is expensive and sticky.
- Main risk or failure mode: Compression artifacts poison replay or decode-time steering.
- Smallest credible first experiment: Add float16 and per-segment delta coding for branch-state segments only.
- Confidence level: 0.74

### Candidate 22: Cross-Session Project Warm Start

- Name: Cross-Session Project Warm Start
- Suggested `phase_id`: `convergence`
- Core insight: Boot new sessions from compact project latent packages and repo delta memory before the first long prompt build.
- External inspiration or analogy: warm boot, project-state blocks, and prompt caching cold-start reduction [R17][R24].
- Why it fits Saguaro and Anvil specifically: `UnifiedChatLoop` and `QSGLatentBridge` already have the ingredients but not a first-session warm-start policy.
- Exact places in this codebase where it could wire in: `core/unified_chat_loop.py`, `core/qsg/latent_bridge.py`, `core/memory/fabric/store.py`, `tests/test_memory_conversation_ingest.py`, `tests/test_qsg_integration.py`.
- Existing primitives it can reuse: latest latent package lookup, mission replay descriptor, repo context fields.
- New primitive, data flow, or subsystem needed: `ProjectWarmStartPlan`.
- `repo_scope`: `core/unified_chat_loop.py`, `core/qsg/latent_bridge.py`, `core/memory/fabric/store.py`, `tests/test_qsg_integration.py`.
- `owning_specialist_type`: `session_experience_architect`
- `allowed_writes`: loop initialization, memory lookup, prompt bootstrapping, tests.
- `telemetry_contract`: Emit `project_warm_start_used`, `project_warm_start_ms_saved`, `project_warm_start_fallback_reason`.
- `required_evidence`: Lower cold-start prompt size and faster early answers on repeated project sessions.
- `rollback_criteria`: Warm starts inject stale or irrelevant state into new sessions.
- `promotion_gate`: Warm-start behavior is correct under both unchanged and changed repo delta conditions.
- `success_criteria`: Returning to a project feels warm immediately instead of after several retrieval passes.
- Why this creates value: It visibly improves user experience.
- Why this creates moat: Warm-start continuity tied to repo delta and latent replay is defensible.
- Main risk or failure mode: Overeager warm starts create false continuity.
- Smallest credible first experiment: On session start, attempt replay of the latest compatible project-level summary package before full evidence gathering.
- Confidence level: 0.86

### Candidate 23: End-User Memory Controls and Audit Surface

- Name: End-User Memory Controls and Audit Surface
- Suggested `phase_id`: `convergence`
- Core insight: Let users see what memory classes exist, what is pinned, what is latent-only, and why a memory was used.
- External inspiration or analogy: Letta ADE, Graphiti local MCP memory, and aviation checklists [R17][R22].
- Why it fits Saguaro and Anvil specifically: the repo already has CLI surfaces, black-box recording, and memory storage; it lacks an operator-grade UX.
- Exact places in this codebase where it could wire in: `cli/commands/memory.py`, `core/telemetry/black_box.py`, `core/qsg/latent_bridge.py`, `tests/test_memory_governance.py`, `tests/test_black_box_recorder.py`.
- Existing primitives it can reuse: read records, feedback records, replay descriptor, event store.
- New primitive, data flow, or subsystem needed: `MemoryAuditEntry` and user-facing memory-class taxonomy.
- `repo_scope`: `cli/commands/memory.py`, `core/telemetry/black_box.py`, `core/qsg/latent_bridge.py`, `tests/test_black_box_recorder.py`.
- `owning_specialist_type`: `memory_product_architect`
- `allowed_writes`: CLI audit commands, telemetry summaries, tests.
- `telemetry_contract`: Emit `memory_audit_views`, `memory_manual_deletes`, `memory_manual_pins`.
- `required_evidence`: Users can explain and change memory behavior without digging through internals.
- `rollback_criteria`: Audit interface exposes confusing or misleading abstractions.
- `promotion_gate`: Memory audits correctly reflect actual memory reads and writes in tests.
- `success_criteria`: Memory features become trustworthy enough for daily use.
- Why this creates value: Trust is a feature.
- Why this creates moat: Transparent memory ops differentiate real systems from demos.
- Main risk or failure mode: UX becomes cluttered.
- Smallest credible first experiment: Add a command that lists recent latent package uses, replay outcomes, and user-pinned memories.
- Confidence level: 0.85

### Candidate 24: Nightly Sleep-Cycle Consolidation

- Name: Nightly Sleep-Cycle Consolidation
- Suggested `phase_id`: `roadmap_draft`
- Core insight: Run deferred consolidation jobs that turn chat traces, tool outcomes, subagent latents, and replay descriptors into stable long-term memory artifacts offline.
- External inspiration or analogy: sleep consolidation, batch graph updates, and background digital twin calibration [R21][R31].
- Why it fits Saguaro and Anvil specifically: `.anvil` already stores artifacts and `core/memory/fabric/jobs.py` exists.
- Exact places in this codebase where it could wire in: `core/memory/fabric/jobs.py`, `core/memory/fabric/community_builder.py`, `core/memory/fabric/temporal_tree.py`, `core/telemetry/black_box.py`, `tests/test_memory_consolidation_jobs.py`, `tests/test_memory_temporal_hierarchy.py`.
- Existing primitives it can reuse: fabric jobs, snapshots, read/feedback records, black-box runs.
- New primitive, data flow, or subsystem needed: `ConsolidationCampaign`.
- `repo_scope`: `core/memory/fabric/jobs.py`, `core/memory/fabric/community_builder.py`, `core/memory/fabric/temporal_tree.py`, `core/telemetry/black_box.py`, `tests/test_memory_consolidation_jobs.py`.
- `owning_specialist_type`: `memory_consolidation_engineer`
- `allowed_writes`: batch jobs, consolidation policies, tests.
- `telemetry_contract`: Emit `memory_consolidation_runs`, `memory_consolidation_outputs`, `memory_consolidation_failures`.
- `required_evidence`: Batch consolidation improves future replay hit rate and reduces duplicated memory objects.
- `rollback_criteria`: Consolidation corrupts provenance or duplicates objects aggressively.
- `promotion_gate`: Offline jobs can be run repeatedly without changing results for unchanged inputs.
- `success_criteria`: The memory system improves between sessions rather than only during sessions.
- Why this creates value: It creates learning without bloating live turns.
- Why this creates moat: Offline memory distillation is a durable system advantage.
- Main risk or failure mode: Consolidation becomes another untrusted summarizer.
- Smallest credible first experiment: Consolidate duplicate subagent-latent signals into stable project-level anchors overnight.
- Confidence level: 0.72

## 6. Critical Pressure Test

### Elegant but likely wrong

- A fully vector-native system with almost no textual summaries is elegant but likely wrong for an engineering product because users and operators still need interpretable checkpoints.
- Non-prefix latent/KV stitching is strategically compelling but may be too model-specific too early for this CPU-first repo.
- A full temporal graph layer could become architecture theater unless it is tied to concrete product surfaces like preference changes and repo delta state.

### Ugly but strategically powerful

- A write controller that says "do not save this latent" more often than it saves is ugly but strategically powerful because memory quality is mostly a curation problem.
- User-facing memory blocks backed by latent packages are less pure than a hidden-state-only design, but they create trust and controllability.
- A CLI-first latent microscope may look unglamorous, but it is the sort of tool that keeps ambitious latent systems maintainable.

### Ideas likely to fail because the repo is missing a key primitive

- Full branch-specific latent replay at native-kernel speed is limited today by the lack of richer segment-level pager semantics.
- High-quality Grover oracle fields need better token-to-embedding and invariant projection plumbing than is currently visible in `core/qsg/grover.py`.
- Nightly consolidation will underdeliver until there is a sharper artifact taxonomy and write policy.

### Repo-specific caution flags

- `GroverAmplifier` is still heuristic, so stacking too many responsibilities on it too early could hide weak signal under clever language.
- `JacobiRefiner` is clearly not yet a systems-grade verifier, so promoted work must include real evaluation gates.
- `LatentMemory` is still structurally shallow; treating it as the finished substrate would constrain the design space prematurely.

## 7. Synthesis

### Strongest ideas overall

- Typed Latent Capsule ABI v3
- Memory Tier Compiler
- Grover Resonance Oracle Field
- COCONUT Branch Reservoir
- Tool Outcome Latent Compression

### Best balance of novelty and plausibility

- Memory Tier Compiler
- Grover Resonance Oracle Field
- Tool Outcome Latent Compression

### Most feasible now

Repo-grounded observation:
Tool Outcome Latent Compression is the most feasible now because `UnifiedChatLoop` already has context-compression machinery and latent payload merge paths.

### Biggest long-term moat bet

Inference:
Typed Latent Capsule ABI v3 plus Multi-Resolution Latent Package Pyramid is the biggest moat bet.
If the system can capture, reason over, replay, and audit structured latent artifacts safely, it stops competing on raw context-window brute force.

### Most cleanly unifies with the current codebase

Repo-grounded observation:
Memory Tier Compiler is the cleanest unifier because it connects:
runtime controller state,
latent bridge,
memory store,
prompt assembly,
and end-user memory behavior
without requiring a brand-new architectural center.

### Which should be prototyped first

Inference:
Prototype order should be:

1. Memory Tier Compiler
2. Tool Outcome Latent Compression
3. Grover Resonance Oracle Field
4. Typed Latent Capsule ABI v3
5. COCONUT Branch Reservoir

## 8. Implementation Program

### Phase 1

- `phase_id`: `research`
- Phase title: Latent Artifact Taxonomy and Capsule ABI
- Objective: Define the canonical latent artifact classes and evolve the capsule ABI so saved state is typed, replayable, and inspectable.
- Dependencies: none
- Repo scope: `core/qsg/runtime_contracts.py`, `core/qsg/continuous_engine.py`, `core/qsg/latent_bridge.py`, `core/memory/latent_memory.py`, `core/memory/fabric/models.py`
- Owning specialist type: `runtime_architect`
- Allowed writes: ABI dataclasses, latent capture/restore paths, package schema, contract tests
- Telemetry contract: `latent_segment_count`, `latent_capsule_bytes`, `latent_capsule_compatibility_score`, `latent_capsule_kind_breakdown`
- Required evidence: round-trip typed capsule tests, mismatch tests, storage schema receipts, replay descriptor coverage
- Rollback criteria: replay compatibility regresses or ABI changes become opaque to operators
- Promotion gate: typed segments survive capture, storage, replay, and mismatch detection across at least branch-state and repo-delta classes
- Success criteria: the system can distinguish and selectively reason about multiple latent segment kinds instead of one merged vector
- Exact wiring points: `core/qsg/runtime_contracts.py`, `core/qsg/continuous_engine.py:525`, `core/qsg/continuous_engine.py:594`, `core/qsg/latent_bridge.py:107`, `core/qsg/latent_bridge.py:203`, `core/memory/latent_memory.py:115`
- Deliverables: ABI v3 dataclasses, typed segment metadata, replay compatibility scoring, updated package builders
- Tests: `tests/test_qsg_runtime_contracts.py`, `tests/test_qsg_continuous_engine.py`, `tests/test_latent_package_capture.py`, new `tests/test_latent_capsule_abi.py`
- Verification commands: `pytest tests/test_qsg_runtime_contracts.py tests/test_qsg_continuous_engine.py tests/test_latent_package_capture.py tests/test_latent_capsule_abi.py`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: typed latent capsules are the default save format for new captures

### Phase 2

- `phase_id`: `feature_map`
- Phase title: Memory Tier Compiler and Save Matrix
- Objective: Compile every memory artifact into an explicit plan describing write admission, storage tier, expiry, visibility, and fallback replay order.
- Dependencies: Phase 1
- Repo scope: `core/qsg/runtime_contracts.py`, `core/qsg/latent_bridge.py`, `core/memory/fabric/store.py`, `core/unified_chat_loop.py`, `core/memory/latent_memory.py`
- Owning specialist type: `memory_compiler_architect`
- Allowed writes: controller policies, loop integration, store routing, planner tests
- Telemetry contract: `memory_plan_kind`, `memory_plan_selected_tier`, `memory_plan_expiry_s`, `memory_plan_visibility`, `latent_write_admit_rate`
- Required evidence: compiled memory matrix for subagent latents, execution capsules, repo delta memory, tool outcomes, and user preference anchors
- Rollback criteria: artifact routing becomes harder to reason about than today or recall degrades on repeated tasks
- Promotion gate: all first-class artifact kinds compile to deterministic plans with passing tests
- Success criteria: memory behavior is unified around an explicit compiler instead of scattered heuristics
- Exact wiring points: `core/qsg/runtime_contracts.py:516`, `core/qsg/latent_bridge.py:203`, `core/memory/fabric/store.py:547`, `core/unified_chat_loop.py:2890`, `core/unified_chat_loop.py:2925`, `core/memory/latent_memory.py:45`
- Deliverables: compiled memory plan API, save matrix, artifact-class policies, utility-based expiry foundation
- Tests: `tests/test_qsg_runtime_contracts.py`, `tests/test_memory_backend_profiles.py`, `tests/test_memory_governance.py`, new `tests/test_memory_tier_compiler.py`
- Verification commands: `pytest tests/test_qsg_runtime_contracts.py tests/test_memory_backend_profiles.py tests/test_memory_governance.py tests/test_memory_tier_compiler.py`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: every promoted latent artifact is routed through a compiled memory plan

### Phase 3

- `phase_id`: `development`
- Phase title: Grover Resonance plus Jacobi Frontier Upgrade
- Objective: Turn Grover into a true latent-and-invariant oracle and turn Jacobi into a frontier verifier rather than a placeholder smoother.
- Dependencies: Phases 1 and 2
- Repo scope: `core/qsg/grover.py`, `core/qsg/jacobi_refiner.py`, `core/qsg/ollama_adapter.py`, `core/qsg/generator.py`, `core/native/native_qsg_engine.py`, `core/unified_chat_loop.py`
- Owning specialist type: `inference_algorithm_architect`
- Allowed writes: Grover oracle logic, Jacobi frontier logic, adapter/runtime wiring, inference tests
- Telemetry contract: `grover_resonance_mean`, `grover_latent_bias_alpha`, `jacobi_frontier_width`, `jacobi_branch_survival_rate`, `jacobi_verify_cost_ms`
- Required evidence: ambiguous-generation tests, structured-prompt acceptance improvements, no-regression checks against current decode behavior
- Rollback criteria: TTFT or output quality regresses beyond agreed thresholds
- Promotion gate: controlled tests show that oracle-guided amplification and frontier verification outperform the current heuristics
- Success criteria: latent state materially shapes decode-time ranking and branch survival
- Exact wiring points: `core/qsg/grover.py:44`, `core/qsg/jacobi_refiner.py:18`, `core/qsg/generator.py:124`, `core/qsg/generator.py:146`, `core/qsg/ollama_adapter.py:613`, `core/unified_chat_loop.py:1726`
- Deliverables: resonance oracle field, frontier verifier, adapter integration, runtime metrics
- Tests: `tests/test_qsg_components.py`, `tests/test_qsg_adapter_unification.py`, `tests/test_qsg_pipeline.py`, `tests/test_native_qsg_engine.py`, new `tests/test_grover_resonance_oracle.py`, new `tests/test_jacobi_frontier_verifier.py`
- Verification commands: `pytest tests/test_qsg_components.py tests/test_qsg_adapter_unification.py tests/test_qsg_pipeline.py tests/test_native_qsg_engine.py tests/test_grover_resonance_oracle.py tests/test_jacobi_frontier_verifier.py`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: decode-time memory usage is measurable and beneficial

### Phase 4

- `phase_id`: `analysis_upgrade`
- Phase title: COCONUT Branch Reservoir and Tool Outcome Compression
- Objective: Persist branch-level continuous thought artifacts and compress tool results into reusable latent outcomes so the main context window stops carrying dead weight.
- Dependencies: Phases 1 and 2
- Repo scope: `core/reasoning/coconut.py`, `core/unified_chat_loop.py`, `core/memory/latent_memory.py`, `core/task_memory.py`, `core/qsg/latent_bridge.py`
- Owning specialist type: `reasoning_systems_architect`
- Allowed writes: coconut session record schema, latent-memory write paths, tool outcome compression, reinjection policy
- Telemetry contract: `coconut_branch_reservoir_size`, `coconut_branch_reuse_hits`, `tool_outcome_compression_ratio`, `tool_outcome_token_avoidance`
- Required evidence: repeated-task improvements with lower context-token pressure and reusable branch artifacts
- Rollback criteria: compression loses critical tool evidence or branch reservoir becomes cold storage
- Promotion gate: repeated repo-analysis and research tasks show better continuity with lower token load
- Success criteria: the system stores and reuses high-value latent work products instead of repeating them
- Exact wiring points: `core/reasoning/coconut.py:119`, `core/reasoning/coconut.py:171`, `core/unified_chat_loop.py:1694`, `core/unified_chat_loop.py:2868`, `core/unified_chat_loop.py:2925`, `core/memory/latent_memory.py:45`
- Deliverables: branch reservoir, tool outcome latent records, reinjection policy upgrade
- Tests: `tests/test_coconut_session_record.py`, `tests/test_coconut_accuracy.py`, `tests/test_adaptive_coconut.py`, `tests/test_subagent_latent_pipeline.py`, `tests/test_memory_conversation_ingest.py`, new `tests/test_tool_outcome_latent_compression.py`
- Verification commands: `pytest tests/test_coconut_session_record.py tests/test_coconut_accuracy.py tests/test_adaptive_coconut.py tests/test_subagent_latent_pipeline.py tests/test_memory_conversation_ingest.py tests/test_tool_outcome_latent_compression.py`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: repeated sessions need fewer explicit tokens to recover prior reasoning context

### Phase 5

- `phase_id`: `deep_test_audit`
- Phase title: Performance Twin, Latent Firewall, and Replay Diagnostics
- Objective: Close the loop between telemetry, governance, and latent replay so advanced memory behavior is controllable and safe.
- Dependencies: Phases 1 through 4
- Repo scope: `core/native/native_qsg_engine.py`, `core/native/runtime_telemetry.py`, `core/telemetry/black_box.py`, `core/qsg/runtime_contracts.py`, `core/qsg/latent_bridge.py`, `core/unified_chat_loop.py`
- Owning specialist type: `telemetry_systems_engineer`
- Allowed writes: governor decisions, telemetry enrichment, replay diagnostics, governance gates, tests
- Telemetry contract: `memory_governor_actions`, `latent_firewall_blocks`, `latent_debug_sessions`, `replay_block_causes`, `performance_twin_regime`
- Required evidence: queue-bound, drift-bound, and compatibility-failure scenarios are observable and handled by policy
- Rollback criteria: controller oscillation, false-positive gating, or unreadable diagnostics
- Promotion gate: synthetic fault injection plus replay mismatch tests pass with deterministic operator-readable reports
- Success criteria: latent memory behavior is monitorable, replayable, and governable like a real runtime subsystem
- Exact wiring points: `core/native/native_qsg_engine.py:3840`, `core/native/native_qsg_engine.py:3858`, `core/telemetry/black_box.py:247`, `core/qsg/latent_bridge.py:264`, `core/qsg/latent_bridge.py:285`, `core/unified_chat_loop.py:943`
- Deliverables: memory governor, latent governance firewall, latent microscope outputs, improved replay tapes
- Tests: `tests/test_runtime_telemetry.py`, `tests/test_qsg_runtime_contracts.py`, `tests/test_black_box_recorder.py`, `tests/test_aes_runtime_gates.py`, new `tests/test_latent_governance_firewall.py`
- Verification commands: `pytest tests/test_runtime_telemetry.py tests/test_qsg_runtime_contracts.py tests/test_black_box_recorder.py tests/test_aes_runtime_gates.py tests/test_latent_governance_firewall.py`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`
- Exit criteria: advanced latent features are operable at production quality

### Phase 6

- `phase_id`: `convergence`
- Phase title: End-User Memory Product Surface
- Objective: Expose the strongest memory features to users through pinned blocks, warm starts, audits, and explicit controls while keeping token usage low.
- Dependencies: Phases 1 through 5
- Repo scope: `cli/commands/memory.py`, `core/unified_chat_loop.py`, `core/prompts/system_prompt_builder.py`, `core/memory/fabric/store.py`, `core/qsg/latent_bridge.py`, `core/telemetry/black_box.py`
- Owning specialist type: `memory_product_architect`
- Allowed writes: CLI/API surfaces, prompt assembly integration, audit views, user-facing tests
- Telemetry contract: `project_warm_start_used`, `user_memory_block_reads`, `user_memory_block_writes`, `memory_audit_views`, `preference_anchor_hits`
- Required evidence: user-visible memory continuity, explicit control paths, and auditability in end-to-end tests
- Rollback criteria: users cannot predict or trust memory behavior
- Promotion gate: users can inspect, pin, update, and delete key memory classes and observe deterministic effects
- Success criteria: latent memory upgrades become a daily-usable feature set rather than a hidden systems exercise
- Exact wiring points: `core/unified_chat_loop.py:1881`, `core/unified_chat_loop.py:2455`, `core/prompts/system_prompt_builder.py:158`, `cli/commands/memory.py`, `core/qsg/latent_bridge.py:203`, `core/telemetry/black_box.py:274`
- Deliverables: pinned latent-backed memory blocks, project warm start, memory audit commands, preference anchors
- Tests: `tests/test_memory_conversation_ingest.py`, `tests/test_memory_governance.py`, `tests/test_qsg_integration.py`, `tests/test_black_box_recorder.py`, new `tests/test_end_user_memory_controls.py`
- Verification commands: `pytest tests/test_memory_conversation_ingest.py tests/test_memory_governance.py tests/test_qsg_integration.py tests/test_black_box_recorder.py tests/test_end_user_memory_controls.py`, `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`, `./venv/bin/saguaro roadmap validate --path QSG_Latent_Memory_Roadmap.md --format json`
- Exit criteria: the memory stack is both technically strong and user-comprehensible

## 9. Implementation Contract

- The system shall implement typed latent capsule capture and replay through `core/qsg/runtime_contracts.py`, `core/qsg/continuous_engine.py`, and `core/qsg/latent_bridge.py`, tested by `tests/test_qsg_runtime_contracts.py`, `tests/test_qsg_continuous_engine.py`, `tests/test_latent_package_capture.py`, and `tests/test_latent_capsule_abi.py`, and verified with `pytest tests/test_qsg_runtime_contracts.py tests/test_qsg_continuous_engine.py tests/test_latent_package_capture.py tests/test_latent_capsule_abi.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall compile memory artifact lifecycle decisions through `core/qsg/runtime_contracts.py`, `core/qsg/latent_bridge.py`, `core/memory/fabric/store.py`, and `core/unified_chat_loop.py`, tested by `tests/test_qsg_runtime_contracts.py`, `tests/test_memory_backend_profiles.py`, `tests/test_memory_governance.py`, and `tests/test_memory_tier_compiler.py`, and verified with `pytest tests/test_qsg_runtime_contracts.py tests/test_memory_backend_profiles.py tests/test_memory_governance.py tests/test_memory_tier_compiler.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall implement a latent-and-invariant Grover oracle through `core/qsg/grover.py`, `core/qsg/ollama_adapter.py`, `core/qsg/generator.py`, and `core/unified_chat_loop.py`, tested by `tests/test_qsg_components.py`, `tests/test_qsg_adapter_unification.py`, `tests/test_grover_resonance_oracle.py`, and `tests/unit/test_coconut_high_fidelity.py`, and verified with `pytest tests/test_qsg_components.py tests/test_qsg_adapter_unification.py tests/test_grover_resonance_oracle.py tests/unit/test_coconut_high_fidelity.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall upgrade Jacobi from probability smoothing to verified frontier control through `core/qsg/jacobi_refiner.py`, `core/qsg/runtime_contracts.py`, and `core/native/native_qsg_engine.py`, tested by `tests/test_qsg_components.py`, `tests/test_native_qsg_engine.py`, and `tests/test_jacobi_frontier_verifier.py`, and verified with `pytest tests/test_qsg_components.py tests/test_native_qsg_engine.py tests/test_jacobi_frontier_verifier.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall persist reusable COCONUT branch artifacts through `core/reasoning/coconut.py`, `core/unified_chat_loop.py`, and `core/memory/latent_memory.py`, tested by `tests/test_coconut_session_record.py`, `tests/test_coconut_accuracy.py`, `tests/test_adaptive_coconut.py`, and `tests/test_branch_reservoir.py`, and verified with `pytest tests/test_coconut_session_record.py tests/test_coconut_accuracy.py tests/test_adaptive_coconut.py tests/test_branch_reservoir.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall compress reusable tool outcomes into latent artifacts through `core/unified_chat_loop.py`, `core/task_memory.py`, and `core/memory/latent_memory.py`, tested by `tests/test_memory_conversation_ingest.py`, `tests/test_pipeline_manager.py`, and `tests/test_tool_outcome_latent_compression.py`, and verified with `pytest tests/test_memory_conversation_ingest.py tests/test_pipeline_manager.py tests/test_tool_outcome_latent_compression.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall govern latent artifact usage through `core/qsg/runtime_contracts.py`, `core/qsg/latent_bridge.py`, `core/unified_chat_loop.py`, and `core/aes/runtime_gates/telemetry_contract_gate.py`, tested by `tests/test_qsg_runtime_contracts.py`, `tests/test_aes_runtime_gates.py`, and `tests/test_latent_governance_firewall.py`, and verified with `pytest tests/test_qsg_runtime_contracts.py tests/test_aes_runtime_gates.py tests/test_latent_governance_firewall.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall expose replay and memory diagnostics through `core/telemetry/black_box.py`, `core/qsg/latent_bridge.py`, and `cli/commands/memory.py`, tested by `tests/test_black_box_recorder.py`, `tests/test_runtime_telemetry.py`, and `tests/test_memory_audit_cli.py`, and verified with `pytest tests/test_black_box_recorder.py tests/test_runtime_telemetry.py tests/test_memory_audit_cli.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall provide end-user memory controls backed by latent packages through `cli/commands/memory.py`, `core/unified_chat_loop.py`, `core/prompts/system_prompt_builder.py`, and `core/memory/fabric/store.py`, tested by `tests/test_memory_governance.py`, `tests/test_memory_conversation_ingest.py`, `tests/test_end_user_memory_controls.py`, and `tests/test_qsg_integration.py`, and verified with `pytest tests/test_memory_governance.py tests/test_memory_conversation_ingest.py tests/test_end_user_memory_controls.py tests/test_qsg_integration.py` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.
- The system shall keep this roadmap validator-friendly through `QSG_Latent_Memory_Roadmap.md`, `saguaro/roadmap`, and the promoted phase deliverables above, tested by `tests/test_qsg_runtime_contracts.py` and `tests/test_memory_governance.py`, and verified with `./venv/bin/saguaro roadmap validate --path QSG_Latent_Memory_Roadmap.md --format json` and `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`.

## Appendix A: Research Sources

- [R1] Hao et al., "Training Large Language Models to Reason in a Continuous Latent Space," arXiv 2412.06769. https://arxiv.org/abs/2412.06769
- [R2] Fu et al., "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding," arXiv 2402.02057. https://arxiv.org/abs/2402.02057
- [R3] Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads," arXiv 2401.10774. https://arxiv.org/abs/2401.10774
- [R4] NVIDIA TensorRT-LLM speculative decoding documentation. https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html
- [R5] Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty," arXiv 2401.15077. https://arxiv.org/abs/2401.15077
- [R6] Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," arXiv 2309.06180. https://arxiv.org/abs/2309.06180
- [R7] H2O official repository and paper summary. https://github.com/FMInference/H2O
- [R8] Li et al., "SnapKV," arXiv 2404.14469. https://arxiv.org/abs/2404.14469
- [R9] Kim et al., "Compressed Context Memory For Online Language Model Interaction," arXiv 2312.03414. https://arxiv.org/abs/2312.03414
- [R10] Context-Memory official repository. https://github.com/snu-mllab/Context-Memory
- [R11] Munkhdalai et al., "Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention," arXiv 2404.07143. https://arxiv.org/abs/2404.07143
- [R12] Bulatov et al., "Recurrent Memory Transformer," arXiv 2207.06881. https://arxiv.org/abs/2207.06881
- [R13] Wu et al., "Memorizing Transformers," arXiv 2203.08913. https://arxiv.org/abs/2203.08913
- [R14] Sun et al., "Learning to (Learn at Test Time): RNNs with Expressive Hidden States," arXiv 2407.04620. https://arxiv.org/abs/2407.04620
- [R15] Behrouz et al., "Titans: Learning to Memorize at Test Time," Google Research. https://research.google/pubs/titans-learning-to-memorize-at-test-time/
- [R16] Behrouz et al., "ATLAS: Learning to Optimally Memorize the Context at Test Time," arXiv 2505.23735. https://arxiv.org/abs/2505.23735
- [R17] Letta memory blocks documentation. https://docs.letta.com/guides/core-concepts/memory/memory-blocks
- [R18] Letta MemGPT architecture documentation. https://docs.letta.com/guides/agents/architectures/memgpt
- [R19] Mem0 direct import documentation. https://docs.mem0.ai/platform/features/direct-import
- [R20] Mem0 multimodal support documentation. https://docs.mem0.ai/open-source/features/multimodal-support
- [R21] Zep agent memory. https://www.getzep.com/product/agent-memory/
- [R22] Graphiti open source. https://www.getzep.com/product/open-source
- [R23] Graphiti overview. https://help.getzep.com/graphiti/graphiti/overview
- [R24] OpenAI prompt caching guide. https://platform.openai.com/docs/guides/prompt-caching
- [R25] Azure OpenAI prompt caching guide. https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/prompt-caching
- [R26] Gemini context caching guide. https://ai.google.dev/gemini-api/docs/caching/
- [R27] Yang et al., "KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse," arXiv 2502.16002. https://arxiv.org/abs/2502.16002
- [R28] Yang et al., "KVShare: Semantic-Aware Key-Value Cache Sharing for Efficient Large Language Model Inference," arXiv 2503.16525. https://arxiv.org/abs/2503.16525
- [R29] Yao et al., "CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion," arXiv 2405.16444. https://arxiv.org/abs/2405.16444
- [R30] LMCache official repository and docs. https://github.com/LMCache/LMCache
- [R31] NASA JSTAR Digital Twins. https://www.nasa.gov/jstar-digital-twins/
- [R32] NASA IV&V services and JSTAR digital twin factory summary. https://www.nasa.gov/ivv-services/
- [R33] NASA Discovery and Systems Health. https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/
- [R34] NASA ISAAC autonomous caretaking project. https://www.nasa.gov/integrated-system-for-autonomous-and-adaptive-caretaking-isaac/
- [R35] Formula 1 powered by AWS telemetry and race strategy. https://aws.amazon.com/sports/f1/
- [R36] Mettu et al., "SRMT: Shared Memory for Multi-agent Lifelong Pathfinding," arXiv 2501.13200. https://arxiv.org/abs/2501.13200
- [R37] CacheGen repository. https://github.com/UChi-JCL/CacheGen
