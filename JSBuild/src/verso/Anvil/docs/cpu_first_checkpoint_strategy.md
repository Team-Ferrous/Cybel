# CPU-First Checkpoint Strategy

Date: `2026-03-10`
Roadmap reference: `Latent_Space_QSG_Roadmap.md :: ## 26. Phase 11: Model-Level Changes for Future Checkpoints`

## Purpose

Phase 11 requires a written checkpoint strategy that shapes future model families for CPU serving instead of treating runtime work as the only optimization path. This document is that strategy artifact.

## CPU-first inference properties

Every future checkpoint line should be evaluated against these CPU-first inference properties:

- reduced KV bandwidth through `GQA` or `MQA`
- compatibility with native multi-token proposal flow through `MTP` heads
- trained `early-exit` supervision for predictable self-speculation
- `latent-intercept` hooks that keep latent and text paths on a shared checkpoint family
- explicit latent recurrence support so latent packets can be resumed without shape drift
- evidence projection compatibility so latent checkpoints can be turned back into inspectable text when required
- latent resume robustness goals for suspend and resume under sequence checkpoints
- quantization-friendly activations and norms for low-bit serving
- no Python or NumPy hot path dependency in the serving path
- stable code and reasoning quality under continued pretraining

## Runtime anchors already present in the repository

Phase 11 is a design phase, but it is not speculative in a vacuum. Existing runtime surfaces already constrain the future checkpoint contract:

- `core/model/model_profile.py` already normalizes `n_kv_heads`, derives `gqa`, and exposes the model-side metadata future checkpoints must preserve.
- `core/native/model_graph.cpp` already contains fused `GQA` or `MQA` attention handling and native execution checkpoint capture and restore APIs.
- `core/native/model_graph_wrapper.py` already exports `n_kv_heads` handling and execution checkpoint support into the Python control plane.
- `core/native/native_qsg_engine.py` already infers KV-head layouts from checkpoint tensors, discovers draft-head tensors for `MTP`-style auxiliary heads, and captures exit-hidden state for early-exit self-spec flows.
- `core/native/runtime_telemetry.py` already emits `hot_path_proof`, `native_backend_abi_match`, and `strict_native_qsg`, so future checkpoints must preserve strict native CPU execution.

Those anchor surfaces mean future checkpoints should target concrete native serving paths instead of abstract research-only properties.

## Required checkpoint metadata and export contract

Future checkpoint exporters should stop relying on heuristic tensor-name discovery alone. Every Phase 11-targeted checkpoint family should carry explicit metadata and export rules:

- attention layout metadata:
  - `n_heads`
  - `n_kv_heads`
  - `head_dim`
  - fused-vs-split `qkv_layout`
- `MTP` export metadata:
  - `draft_head_kind`
  - `draft_head_count`
  - `draft head manifest`
  - stable draft-head weight and bias names
- early-exit export metadata:
  - `exit_layer_indices`
  - calibrated exit-confidence head names
  - verifier-compatibility expectations for intermediate hidden states
- latent-intercept export metadata:
  - `latent_packet_schema_version`
  - evidence projection adapter names
  - resume invariants for latent packet restore
- checkpoint integrity and serving metadata:
  - `checkpoint_hash`
  - quantization recipe and calibration provenance
  - architecture family tag showing the checkpoint is intended for CPU-first native serving
  - no Python or NumPy hot path dependency

These metadata requirements are the minimum bridge between the current runtime hooks and future trained checkpoint families.

## Checkpoint family strategy

### Family A: Compatibility Runtime Line

Purpose:
Maintain compatibility with current checkpoints while native runtime phases continue to land.

Requirements:
- no checkpoint format break
- runtime-only wins remain the focus
- use as the regression baseline for all later families

Blocking dependencies:
- none beyond current runtime work

### Family B: Continued-Pretraining MTP Line

Purpose:
Extend the current checkpoint family with `MTP` auxiliary heads and `early-exit` supervision without requiring a full architecture replacement.

Requirements:
- continued pretraining on code- and reasoning-heavy corpora
- auxiliary `MTP` heads sized for native multi-token serving experiments
- early-exit supervision attached to intermediate layers used by self-spec paths
- `latent-intercept` supervision attached to hidden states intended for latent packet capture
- evidence projection heads or adapters that can decode latent packets back into human-auditable summaries
- latent resume robustness evaluation across checkpoint save and restore boundaries
- compatibility plan for later native verification and acceptance telemetry

Blocking dependencies:
- Phase 6 native multi-token proposal flow
- Phase 7 early-exit and self-spec runtime reuse
- Phase 12 quality regression governance

### Family C: Primary CPU-First Checkpoint Line

Purpose:
Establish the first future model line that explicitly targets CPU-first inference properties rather than post-training retrofits.

Explicit CPU-first inference targets:
- `GQA` by default, with `MQA` allowed only if quality remains within Phase 12 gates
- reduced KV head count sized for bandwidth-limited CPU decode
- quantization-friendly activations and norms to support low-bit weights, low-bit KV, and activation-aware calibration
- outlier mitigation to protect 4-bit and lower serving paths
- architecture choices that preserve verifier efficiency for native speculative execution

Training hooks:
- `MTP` auxiliary heads on the base trunk
- `early-exit` supervision on intermediate layers that match planned self-spec exit points
- `latent-intercept` training hooks on the hidden-state surfaces that Phase 11 wants to preserve across suspend and resume
- latent recurrence objectives that keep resumed packet states numerically stable across multiple latent steps
- evidence projection adapters so latent checkpoints can produce reviewable evidence without retraining a separate model family
- quantization-aware continued pretraining once low-bit runtime surfaces are stable

Success criteria:
- this family becomes the default recommendation for the first net-new CPU-first checkpoint effort
- it is judged against throughput, TTFT, memory bandwidth, perplexity, and code-quality gates rather than serving speed alone

### Family D: Blockwise Research Line

Purpose:
Keep a research lane for block diffusion or masked diffusion checkpoints without making it a prerequisite for current CPU-serving wins.

Requirements:
- blockwise state representations
- masked-span or denoise-step supervision
- compatibility with future non-AR runtime infrastructure

Blocking dependencies:
- Phase 8 blockwise runtime substrate
- Phase 12 quality and decision governance

## Decision framework

Use this order of operations for new checkpoint investment:

1. Keep Family A as the baseline and compatibility anchor.
2. Prefer Family C for the first explicit CPU-first checkpoint program.
3. Use Family B when continued pretraining on the current trunk is materially cheaper than an architectural reset.
4. Keep Family D as research-only until blockwise runtime support is real.

## Required training and evaluation gates

- `GQA` or `MQA` changes must be benchmarked against KV bandwidth and decode memory pressure, not only perplexity.
- `MTP` heads must be evaluated for both acceptance-rate utility and downstream code quality.
- `early-exit` supervision must report calibrated exit confidence rather than raw loss deltas alone.
- `latent-intercept` hooks must preserve latent recurrence stability and resume determinism across checkpoint restore.
- evidence projection outputs must stay faithful enough for Phase 12 review and debugging gates.
- quantization-aware efforts must be blocked if they degrade code correctness or coherence beyond Phase 12 limits.

## Current status

- Written strategy exists in-repo.
- Primary CPU-first future line is `Family C`.
- Checkpoint-schema and export requirements are now explicit enough to regression test.
- Runtime and training implementation for these families remains blocked on later roadmap phases.
- No trained future-checkpoint family is operational in this phase; Phase 11 only closes the design contract.
