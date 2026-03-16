# Anvil Governance Prompt (Phase 2)

This file defines the Phase 2 prompt governance contract for agents working in this repository.

## Required Policy Inputs

- `standards/AES_CONDENSED.md` (always inject)
- `standards/AES_RULES.json` (rule IDs and enforcement semantics)
- `standards/domain_rules/<domain>.md` (conditionally inject per detected domain)
- `aes_visuals/v1/PROMPT_GUIDANCE.md` (canonical, inject bounded summary only)
- `aes_visuals/v2/PROMPT_GUIDANCE.md` (canonical, inject bounded summary only)
- Visual-pack policy crosswalk: [shared prompt foundation](./shared_prompt_foundation.md#aes-visual-pack-inputs)
- Visual-pack sources: [v1 guidance](../aes_visuals/v1/PROMPT_GUIDANCE.md), [v2 guidance](../aes_visuals/v2/PROMPT_GUIDANCE.md)

Prompt text is guidance. Runtime gates are binding.

## AES Visual Pack Inputs

- `aes_visuals/v1` and `aes_visuals/v2` are mandatory policy references in master, subagent, and system prompt contracts.
- Runtime should resolve repo-root packs first (`aes_visuals/<version>/PROMPT_GUIDANCE.md`, then `aes_visuals/<version>/directives.json`) and keep summaries concise.
- Missing pack files must not crash prompt assembly; fallback to AES condensed/domain policy while preserving `aes_visuals` references.
- Legacy compatibility is allowed only as fallback: `prompts/aes_visuals/<version>/...` is never canonical.

## AAL Contract

Use AES Assurance Assurance Level (AAL) terminology exactly:

| AAL | Severity | Meaning | Minimum Closure Requirement |
|---|---|---|---|
| `AAL-0` | `P0` | catastrophic path | no open violations; independent review; traceability + evidence bundle + signoff + waiver if needed |
| `AAL-1` | `P1` | critical path | no open violations; traceability + evidence bundle + signoff + waiver if needed |
| `AAL-2` | `P2` | major path | no open blocking violations; regression evidence present |
| `AAL-3` | `P3` | low-risk path | hygiene review and verification summary |

Strictness ordering: `AAL-0` > `AAL-1` > `AAL-2` > `AAL-3`.

## Mandatory AES Rules

- Keep traceability chain intact: `requirement -> design -> code -> test -> verification`.
- Apply root-cause remediation, not masking patches.
- Do not swallow exceptions with bare `except`.
- Do not bypass required verification through silent fallbacks.
- Do not use `eval` or `exec` outside explicit sandboxing.
- Treat missing `AAL-0`/`AAL-1` artifacts as blocking.

## Anti-Patterns (Disallowed)

- Declaring completion when verification still reports blocking issues.
- Claiming performance improvement without measured baseline and post-change metrics.
- Emitting success status while evidence bundle or signoff is absent.
- Deferring high-assurance defects to follow-up without waiver and compensating controls.

## Runtime 8-Phase Crosswalk

| Phase | Name | Runtime Intent | Evidence Needed |
|---|---|---|---|
| `0` | Preflight And Baseline | classify AAL and domains, establish baseline | AAL/domain declaration and metric baseline plan |
| `1` | Runtime Integrity | validate execution gates and path correctness | gate check output and blocking defect list |
| `2` | Mathematical Integrity | validate numerical stability and invariants | finite/stable numeric evidence |
| `3` | Learning Dynamics And Control | validate control/training dynamics | loop health evidence and anomalies |
| `4` | Architecture And Routing Integrity | validate orchestration and routing | route and adapter integrity evidence |
| `5` | Performance And HPC Saturation | validate latency, throughput, saturation | golden signal before/after deltas |
| `6` | Config, Feature, And Observability Coverage | validate feature/config/test observability closure | coverage and observability report |
| `7` | Red Team Final Validation And Sign-Off | adversarial review and closure decision | PASS or PARTIAL with explicit rationale |

## Domain-Conditional Rule Injection Markers

Inject domain guidance when markers are present in changed files.

| Domain | Import Markers | Content Markers | Inject |
|---|---|---|---|
| `ml` | `torch`, `tensorflow`, `keras`, `sklearn`, `jax`, `optax` | `optimizer`, `backward`, `gradient`, `loss_fn`, `train_step` | `standards/domain_rules/ml.md` |
| `quantum` | `qiskit`, `cirq`, `pennylane`, `braket` | `quantum_circuit`, `qubit`, `entangle`, `transpile` | `standards/domain_rules/quantum.md` |
| `physics` | `scipy.integrate`, `sympy.physics`, `fenics`, `fipy` | `conservation`, `hamiltonian`, `lagrangian`, `symplectic` | `standards/domain_rules/physics.md` |
| `hpc` | `immintrin.h`, `omp.h`, `mpi.h`, `cuda_runtime` | `simd`, `avx2`, `vectorize`, `parallel_for`, `thread_pool`, `#pragma omp` | `standards/domain_rules/hpc.md` |

If more than one domain matches, inject all matching domain rule files and merged AES rule IDs.

## Golden Signals

Always report the four universal golden signals:

- Latency: `step_time_avg_s`, TTFT, `P50/P95/P99`
- Throughput: `tokens_per_sec_avg`, `throughput_gflops`
- Errors: non-finite counts, assertion failures, verification failures
- Saturation: thread utilization, SIMD coverage, RSS peak

Domain-specific adaptations:

- `ml`: gradient finiteness, loss stability, ingest validation failures, reproducibility manifest status
- `quantum`: transpilation coverage, noise-model declaration, shot sufficiency, backend execution failure rate
- `physics`: conservation drift, integrator stability, unit-consistency violations
- `hpc`: OpenMP clause explicitness, alignment contracts, scalar-oracle parity, race-condition findings

## Completion Rule

Return `PARTIAL` when any required element is missing:

- required code changes
- required tests
- required AAL artifacts
- required roadmap/update evidence

Do not convert `PARTIAL` to complete via prompt narrative alone.
