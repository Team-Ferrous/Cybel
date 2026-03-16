# Phase 2 Shared Prompt Foundation

Purpose: shared governance contract inherited by agent-facing prompts.

## Authority Order

1. Runtime enforcement is authoritative.
2. `standards/AES_RULES.json` defines enforceable rule IDs.
3. `standards/AES_CONDENSED.md` is mandatory baseline policy text.
4. `aes_visuals/v1` and `aes_visuals/v2` are required visual-policy inputs.
5. Prompt prose must not weaken runtime gates.

## AES Visual Pack Inputs

- Canonical reference packs: `aes_visuals/v1/PROMPT_GUIDANCE.md` and `aes_visuals/v2/PROMPT_GUIDANCE.md`.
- Canonical sources: [v1 guidance](../aes_visuals/v1/PROMPT_GUIDANCE.md), [v2 guidance](../aes_visuals/v2/PROMPT_GUIDANCE.md)
- Canonical JSON fallback when guidance markdown is unavailable: `aes_visuals/<version>/directives.json`.
- Injection policy: include concise summaries only; do not dump full visual-pack files into runtime prompts.
- Missing pack behavior: preserve `aes_visuals/v1` + `aes_visuals/v2` contract references and degrade gracefully to AES condensed/domain rules.
- Legacy compatibility is fallback-only: `prompts/aes_visuals/<version>/...` may be read only when canonical repo-root packs are unavailable.
- Source policy reference: [GEMINI prompt governance](./GEMINI.md#aes-visual-pack-inputs)

## AAL Terminology And Severity Matrix

| AAL | Priority | Operational Meaning | Blocking Condition | Required Closure Artifacts |
|---|---|---|---|---|
| `AAL-0` | `P0` | Catastrophic or safety-critical path | Any unresolved violation or missing artifact | traceability record, evidence bundle, independent review signoff, valid waiver when deviating |
| `AAL-1` | `P1` | Critical path with high regression risk | Any unresolved violation or missing artifact | traceability record, evidence bundle, review signoff, valid waiver when deviating |
| `AAL-2` | `P2` | Major path with bounded blast radius | Unresolved blocking rule | evidence-backed tests and verification output |
| `AAL-3` | `P3` | Low-risk hygiene or documentation path | Unresolved policy hygiene gap | review notes and verification summary |

Strictness order is `AAL-0` > `AAL-1` > `AAL-2` > `AAL-3`.

## Mandatory AES Rules

- Preserve `requirement -> design -> code -> test -> verification` traceability.
- Prefer root-cause fixes over masking behavior.
- No bare exception swallowing.
- No silent fallback around required verification.
- No `eval` or `exec` outside explicit sandboxing.
- For `AAL-0` and `AAL-1`, missing required artifacts is blocking.
- Runtime gates override prompt wording when they conflict.

## Mandatory Anti-Patterns

- Marking work complete with failing verification output.
- Reporting performance gains without before/after evidence.
- Returning success while required artifacts are missing.
- Using fallback branches to bypass review or tests.
- Introducing unverifiable behavior on high-assurance paths.

## Runtime 8-Phase Crosswalk

| Phase | Governance Objective | Runtime Surface | Required Output |
|---|---|---|---|
| `0` Preflight And Baseline | classify risk and gather baseline context | request intake and preflight policy setup | selected AAL, initial domain set, baseline metrics scope |
| `1` Runtime Integrity | verify execution path and gate wiring | plan/execution entry checks | gate status, blocking defects list |
| `2` Mathematical Integrity | verify numerics and invariants | implementation and test loop | finite/stable math evidence |
| `3` Learning Dynamics And Control | verify training/control loop behavior | execution + verification loops | control-loop health evidence |
| `4` Architecture And Routing Integrity | verify path routing and adapters | orchestration and routing surfaces | route integrity evidence |
| `5` Performance And HPC Saturation | verify latency/throughput/saturation behavior | benchmark and profiling surfaces | golden signal deltas |
| `6` Config, Feature, And Observability Coverage | verify config/feature/test observability closure | config and verification reports | coverage evidence and unresolved gaps |
| `7` Red Team Final Validation And Sign-Off | final challenge and closure decision | final verification and review stage | PASS or PARTIAL decision with rationale |

## Domain-Conditional Rule Injection Markers

Domain rules are conditionally injected using deterministic markers from `core/aes/domain_detector.py`.

| Domain | Import Markers | Content Markers | Injected Rule File |
|---|---|---|---|
| `ml` | `torch`, `tensorflow`, `keras`, `sklearn`, `jax`, `optax` | `optimizer`, `backward`, `gradient`, `loss_fn`, `train_step` | `standards/domain_rules/ml.md` |
| `quantum` | `qiskit`, `cirq`, `pennylane`, `braket` | `quantum_circuit`, `qubit`, `entangle`, `transpile` | `standards/domain_rules/quantum.md` |
| `physics` | `scipy.integrate`, `sympy.physics`, `fenics`, `fipy` | `conservation`, `hamiltonian`, `lagrangian`, `symplectic` | `standards/domain_rules/physics.md` |
| `hpc` | `immintrin.h`, `omp.h`, `mpi.h`, `cuda_runtime` | `simd`, `avx2`, `vectorize`, `parallel_for`, `thread_pool`, `#pragma omp` | `standards/domain_rules/hpc.md` |

If multiple domains match, inject all matching domain rule files.

## Golden Signals With Domain Adaptations

### Universal Golden Signals

- Latency: `step_time_avg_s`, TTFT, `P50/P95/P99`
- Throughput: `tokens_per_sec_avg`, `throughput_gflops`
- Errors: non-finite counts, assertion failures, verification failures
- Saturation: thread utilization, SIMD coverage, RSS peak

### Domain Adaptations

| Domain | Additional Golden Signal Focus |
|---|---|
| `ml` | gradient finiteness rate, loss stability trend, data-ingest validation failures, reproducibility manifest completeness |
| `quantum` | transpilation success rate, noise-model declaration coverage, shot sufficiency, backend execution failures |
| `physics` | conservation drift bounds, integrator stability violations, unit-consistency errors |
| `hpc` | OpenMP clause coverage, alignment contract coverage, scalar-oracle parity failures, parallel race findings |

## Completion Contract

A task is complete only when all are true:

- code changes are present when required
- relevant tests pass
- required evidence exists for the selected AAL
- roadmap/update artifact is recorded

Otherwise status must remain `PARTIAL`.
