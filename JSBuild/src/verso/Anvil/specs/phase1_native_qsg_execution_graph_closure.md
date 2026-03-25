# Phase 1 Native QSG Execution Graph Closure Spec

## Scope

This spec closes Phase 1 (Full C Execution Graph) items in `ROADMAP_PERF_OPTIMIZATION.md` for decode-path correctness, coherence safety, and throughput on the granite hybrid path.

## Repair Map

- `PERF-PH1-P0-001` Graph prefill path bypasses C++ graph for multi-token inputs.
  - Severity: P0
  - Root cause: `NativeQSGEngine._get_logits` routes full-graph mode only when `len(token_ids) == 1`.
  - Risk: Graph KV cache diverges from prefill context, invalidating subsequent decode attention state.
  - Fix: Route *all* token spans through `NativeModelGraph.forward_token` in graph mode, sequentially by position.
- `PERF-PH1-P1-001` Missing test that enforces multi-token graph prefill routing.
  - Severity: P1
  - Root cause: Existing tests validate import/symbol/shape only.
  - Fix: Add unit coverage asserting (a) one graph call per prompt token, (b) no Python fallback call, (c) returned logits correspond to last token in span.
- `PERF-PH1-P2-001` Missing explicit phase gate evidence artifact for call-count reduction.
  - Severity: P2
  - Root cause: Roadmap lacks implementation addendum/repair status row with measured deltas.
  - Fix: Add Phase 1 implementation addendum to roadmap with before/after evidence and gate decision.
- `PERF-PH1-P0-003` Hybrid dispatch marked all granite layers as Python-only.
  - Severity: P0
  - Root cause: C++ eligibility required non-MoE FFN, but granite attention layers still carry MoE FFN tensors.
  - Risk: No C++ layer execution in hybrid mode (`0 C++ / 40 Python`), losing intended Phase 1 throughput gains.
  - Fix: Add separate capability maps for `attention-only C++` vs `full-layer C++`; run MoE FFN in Python after C++ attention.
- `PERF-PH1-P0-004` Unsafe full-graph path remained reachable via env toggles.
  - Severity: P0
  - Root cause: `ANVIL_NATIVE_HYBRID_GRAPH=0` could bypass hybrid dispatch and re-enter incoherent full-graph path.
  - Risk: punctuation-only incoherent output while appearing "graph enabled".
  - Fix: Force hybrid mode on `granitehybrid` when hybrid-capable graph is present.
- `PERF-PH1-P1-003` Missing mixed-dispatch coherence regression test.
  - Severity: P1
  - Root cause: No test validated C++ attention + Python FFN composition against Python reference flow.
  - Fix: Add `tests/test_graph_coherence.py` and hybrid dispatch unit tests in `tests/test_native_qsg_engine.py`.
- `PERF-PH1-P1-004` Default thread auto-selection over-subscribed decode on large CPUs.
  - Severity: P1
  - Root cause: heuristic allowed high core counts that degraded latency-bound decode.
  - Fix: cap auto-selected runtime threads to 8 unless manually overridden.

## Design Decisions

1. Keep API contract of `_get_logits(token_ids, start_pos)` unchanged.
2. In full-graph mode, obtain embeddings for all `token_ids` in one lookup and execute `forward_token` per token with monotonically increasing position (`start_pos + i`).
3. If full-graph execution fails for any token, fail over to existing Python forward path to preserve runtime stability.
4. Verification evidence must include:
   - targeted tests for graph prefill routing
   - targeted tests for mixed C++/Python dispatch coherence
   - phase tests from roadmap (`tests/test_model_graph.py`, `tests/test_perf_roadmap_phases.py`)
   - `saguaro verify` output summary

## Out Of Scope

- New kernels (Phase 3+)
- QHPM wiring verification (not required for Phase 1)
- SSM feature-family changes (Phase 6)

## Gate Target (Phase 1)

- Functional: Graph path executes prefill/decode with synchronized KV context and coherent granite hybrid runtime behavior.
- Test: Prefill routing + mixed-dispatch coherence tests + existing Phase 1 tests pass.
- Evidence: Roadmap updated with repair status, throughput/coherence metrics, and gate decision.
